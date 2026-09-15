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
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"

	"github.com/kubeflow/spark-operator/v2/api/v1alpha1"
	"github.com/kubeflow/spark-operator/v2/pkg/common"
	"github.com/kubeflow/spark-operator/v2/pkg/util"
)

// gpuConfOption configures Spark's resource scheduler in addition to pod resources.
func gpuConfOption(conn *v1alpha1.SparkConnect) ([]string, error) {
	var args []string
	for _, role := range []struct {
		name string
		gpu  *v1alpha1.GPUSpec
	}{
		{name: "driver", gpu: conn.Spec.Server.GPU},
		{name: "executor", gpu: conn.Spec.Executor.GPU},
	} {
		if role.gpu == nil {
			continue
		}
		vendor, name, ok := strings.Cut(role.gpu.Name, "/")
		if !ok || name != "gpu" || len(validation.IsDNS1123Subdomain(vendor)) != 0 {
			return nil, fmt.Errorf("%s GPU resource name must have the form <vendor-domain>/gpu, got %q", role.name, role.gpu.Name)
		}
		if role.gpu.Quantity <= 0 {
			return nil, fmt.Errorf("%s GPU quantity must be positive, got %d", role.name, role.gpu.Quantity)
		}
		args = append(args,
			"--conf", fmt.Sprintf("spark.%s.resource.gpu.amount=%d", role.name, role.gpu.Quantity),
			"--conf", fmt.Sprintf("spark.%s.resource.gpu.vendor=%s", role.name, vendor),
		)
	}
	return args, nil
}

func setGPUResources(container *corev1.Container, gpu *v1alpha1.GPUSpec) {
	if gpu == nil {
		return
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}
	if container.Resources.Limits == nil {
		container.Resources.Limits = corev1.ResourceList{}
	}
	quantity := *resource.NewQuantity(gpu.Quantity, resource.DecimalSI)
	name := corev1.ResourceName(gpu.Name)
	container.Resources.Requests[name] = quantity
	container.Resources.Limits[name] = quantity
}

// executorPodTemplate applies GPU resources to a copy of the user's pod template.
// Both the mounted ConfigMap and the template referenced by Spark use this copy.
func executorPodTemplate(conn *v1alpha1.SparkConnect) *corev1.PodTemplateSpec {
	if conn.Spec.Executor.GPU == nil {
		return conn.Spec.Executor.Template
	}
	template := conn.Spec.Executor.Template.DeepCopy()
	if template == nil {
		template = &corev1.PodTemplateSpec{}
	}
	if len(template.Spec.Containers) == 0 {
		template.Spec.Containers = []corev1.Container{{Name: common.Spark3DefaultExecutorContainerName}}
	}
	container := util.GetContainerByNameOrFirst(template.Spec.Containers, common.Spark3DefaultExecutorContainerName)
	setGPUResources(container, conn.Spec.Executor.GPU)
	return template
}
