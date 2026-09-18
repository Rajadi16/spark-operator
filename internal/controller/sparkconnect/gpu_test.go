/*
Copyright 2026 The Kubeflow authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sparkconnect

import (
	"context"
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	"github.com/kubeflow/spark-operator/v2/api/v1alpha1"
	"github.com/kubeflow/spark-operator/v2/pkg/common"
)

var _ = Describe("Spark Connect GPU support", func() {
	var conn *v1alpha1.SparkConnect
	var reconciler *Reconciler

	BeforeEach(func() {
		conn = &v1alpha1.SparkConnect{
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-connect", Namespace: "default", UID: "test-uid"},
			Spec: v1alpha1.SparkConnectSpec{
				SparkVersion: "4.0.0",
				Image:        ptr.To("example.com/spark:gpu"),
			},
		}
		reconciler = &Reconciler{scheme: scheme.Scheme}
		Expect(os.Setenv(common.EnvKubernetesServiceHost, "127.0.0.1")).To(Succeed())
		Expect(os.Setenv(common.EnvKubernetesServicePort, "443")).To(Succeed())
		DeferCleanup(os.Unsetenv, common.EnvKubernetesServiceHost)
		DeferCleanup(os.Unsetenv, common.EnvKubernetesServicePort)
	})

	It("leaves CPU-only sessions without GPU configuration or an executor template", func() {
		args, err := gpuConfOption(conn)
		Expect(err).NotTo(HaveOccurred())
		Expect(args).To(BeEmpty())
		Expect(executorPodTemplate(conn)).To(BeNil())
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: conn.Namespace}}
		Expect(reconciler.mutateServerPod(context.Background(), conn, pod)).To(Succeed())
		Expect(pod.Spec.Containers[0].Resources.Requests).NotTo(HaveKey(corev1.ResourceName("nvidia.com/gpu")))
		Expect(pod.Spec.Containers[0].Resources.Limits).NotTo(HaveKey(corev1.ResourceName("nvidia.com/gpu")))
	})

	It("configures server and executor GPUs independently", func() {
		conn.Spec.Server.GPU = &v1alpha1.GPUSpec{Name: "amd.com/gpu", Quantity: 1}
		conn.Spec.Executor.GPU = &v1alpha1.GPUSpec{Name: "nvidia.com/gpu", Quantity: 2}
		args, err := gpuConfOption(conn)
		Expect(err).NotTo(HaveOccurred())
		Expect(shellParsedSparkConfig(args)).To(Equal(map[string]string{
			"spark.driver.resource.gpu.amount":   "1",
			"spark.driver.resource.gpu.vendor":   "amd.com",
			"spark.executor.resource.gpu.amount": "2",
			"spark.executor.resource.gpu.vendor": "nvidia.com",
		}))
	})

	It("starts an executor-only GPU session with discovery and fractional task resources", func() {
		conn.Spec.Executor.GPU = &v1alpha1.GPUSpec{Name: "nvidia.com/gpu", Quantity: 2}
		conn.Spec.SparkConf = map[string]string{
			"spark.executor.resource.gpu.amount":          "9",
			"spark.executor.resource.gpu.vendor":          "old.example.com",
			"spark.executor.resource.gpu.discoveryScript": "/opt/spark/scripts/discover gpus.sh",
			"spark.task.resource.gpu.amount":              "0.25",
		}
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: conn.Namespace}}
		Expect(reconciler.mutateServerPod(context.Background(), conn, pod)).To(Succeed())
		command := pod.Spec.Containers[0].Args[0]
		Expect(command).To(ContainSubstring("spark.executor.resource.gpu.amount=2"))
		Expect(command).To(ContainSubstring("spark.executor.resource.gpu.vendor=nvidia.com"))
		Expect(command).NotTo(ContainSubstring("spark.driver.resource.gpu.amount"))
		Expect(command).To(ContainSubstring("spark.kubernetes.executor.podTemplateFile="))
		Expect(pod.Spec.Containers[0].Resources.Limits).NotTo(HaveKey(corev1.ResourceName("nvidia.com/gpu")))

		// Typed GPU settings override the same keys in sparkConf.
		args, err := sparkConfOption(conn)
		Expect(err).NotTo(HaveOccurred())
		gpuArgs, err := gpuConfOption(conn)
		Expect(err).NotTo(HaveOccurred())
		config := shellParsedSparkConfig(append(args, gpuArgs...))
		Expect(config).To(HaveKeyWithValue("spark.executor.resource.gpu.amount", "2"))
		Expect(config).To(HaveKeyWithValue("spark.executor.resource.gpu.vendor", "nvidia.com"))
		Expect(config).To(HaveKeyWithValue("spark.executor.resource.gpu.discoveryScript", "/opt/spark/scripts/discover gpus.sh"))
		Expect(config).To(HaveKeyWithValue("spark.task.resource.gpu.amount", "0.25"))

		cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Namespace: conn.Namespace}}
		Expect(reconciler.mutateConfigMap(context.Background(), conn, cm)).To(Succeed())
		var mountedTemplate corev1.PodTemplateSpec
		Expect(yaml.Unmarshal([]byte(cm.Data[ExecutorPodTemplateFileName]), &mountedTemplate)).To(Succeed())
		Expect(mountedTemplate.Spec.Containers).To(HaveLen(1))
		Expect(mountedTemplate.Spec.Containers[0].Name).To(Equal(common.Spark3DefaultExecutorContainerName))
		expectGPUResources(mountedTemplate.Spec.Containers[0], "nvidia.com/gpu", 2)
		Expect(conn.Spec.Executor.Template).To(BeNil())
	})

	DescribeTable("preserves pod templates and sidecars while setting matching GPU requests and limits",
		func(server bool, containerName string) {
			template := &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"custom": "label"}},
				Spec: corev1.PodSpec{Containers: []corev1.Container{
					{Name: "metrics", Image: "example.com/metrics:latest"},
					{Name: containerName, Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1"), "nvidia.com/gpu": resource.MustParse("9")},
						Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("1Gi"), "nvidia.com/gpu": resource.MustParse("9")},
					}},
				}},
			}
			// A custom container name uses the same first-container fallback as image selection.
			index := 1
			if containerName == "custom" {
				template.Spec.Containers[0], template.Spec.Containers[1] = template.Spec.Containers[1], template.Spec.Containers[0]
				index = 0
			}
			before := template.DeepCopy()
			gpu := &v1alpha1.GPUSpec{Name: "nvidia.com/gpu", Quantity: 2}
			var containers []corev1.Container
			if server {
				conn.Spec.Server.GPU, conn.Spec.Server.Template = gpu, template
				pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: conn.Namespace}}
				Expect(reconciler.mutateServerPod(context.Background(), conn, pod)).To(Succeed())
				containers = pod.Spec.Containers
			} else {
				conn.Spec.Executor.GPU, conn.Spec.Executor.Template = gpu, template
				containers = executorPodTemplate(conn).Spec.Containers
			}
			Expect(containers).To(HaveLen(2))
			Expect(containers[1-index]).To(Equal(before.Spec.Containers[1-index]))
			Expect(containers[index].Resources.Requests).To(HaveKeyWithValue(corev1.ResourceCPU, resource.MustParse("1")))
			Expect(containers[index].Resources.Limits).To(HaveKeyWithValue(corev1.ResourceMemory, resource.MustParse("1Gi")))
			expectGPUResources(containers[index], "nvidia.com/gpu", 2)
			Expect(template).To(Equal(before))
		},
		Entry("server named container", true, common.SparkDriverContainerName),
		Entry("server first container", true, "custom"),
		Entry("executor named container", false, common.Spark3DefaultExecutorContainerName),
		Entry("executor first container", false, "custom"),
	)

	It("retains both GPU specifications through the API server and deep copies", func() {
		conn.Name, conn.GenerateName, conn.UID = "", "gpu-connect-", ""
		conn.Spec.Server.GPU = &v1alpha1.GPUSpec{Name: "amd.com/gpu", Quantity: 1}
		conn.Spec.Executor.GPU = &v1alpha1.GPUSpec{Name: "nvidia.com/gpu", Quantity: 2}
		Expect(k8sClient.Create(context.Background(), conn)).To(Succeed())
		DeferCleanup(k8sClient.Delete, context.Background(), conn)
		stored := &v1alpha1.SparkConnect{}
		Expect(k8sClient.Get(context.Background(), client.ObjectKeyFromObject(conn), stored)).To(Succeed())
		Expect(stored.Spec.Server.GPU).To(Equal(conn.Spec.Server.GPU))
		Expect(stored.Spec.Executor.GPU).To(Equal(conn.Spec.Executor.GPU))
		copied := stored.DeepCopy()
		copied.Spec.Server.GPU.Quantity = 3
		copied.Spec.Executor.GPU.Name = "amd.com/gpu"
		Expect(stored.Spec.Server.GPU.Quantity).To(Equal(int64(1)))
		Expect(stored.Spec.Executor.GPU.Name).To(Equal("nvidia.com/gpu"))
	})

	DescribeTable("rejects invalid GPU resources in the CRD and controller",
		func(server bool, name string, quantity int64) {
			conn.Name, conn.GenerateName, conn.UID = "", "invalid-gpu-", ""
			gpu := &v1alpha1.GPUSpec{Name: name, Quantity: quantity}
			if server {
				conn.Spec.Server.GPU = gpu
			} else {
				conn.Spec.Executor.GPU = gpu
			}
			_, err := gpuConfOption(conn)
			Expect(err).To(HaveOccurred())
			err = k8sClient.Create(context.Background(), conn)
			Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected schema rejection, got %v", err)
		},
		Entry("zero server GPUs", true, "nvidia.com/gpu", int64(0)),
		Entry("negative executor GPUs", false, "nvidia.com/gpu", int64(-1)),
		Entry("missing vendor", false, "gpu", int64(1)),
		Entry("empty resource name", true, "", int64(1)),
		Entry("invalid vendor", false, "bad..example/gpu", int64(1)),
		Entry("unsupported resource suffix", true, "nvidia.com/other", int64(1)),
	)
})

func expectGPUResources(container corev1.Container, name string, quantity int64) {
	GinkgoHelper()
	resourceName := corev1.ResourceName(name)
	Expect(container.Resources.Requests).To(HaveKey(resourceName))
	Expect(container.Resources.Limits).To(HaveKey(resourceName))
	Expect(container.Resources.Requests.Name(resourceName, resource.DecimalSI).Value()).To(Equal(quantity))
	Expect(container.Resources.Limits.Name(resourceName, resource.DecimalSI).Value()).To(Equal(quantity))
}
