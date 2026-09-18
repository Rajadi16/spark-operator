# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for SparkConnect v1alpha1 GPU model generation.

These tests verify that:
- SparkV1alpha1GPUSpec is importable and its fields are correct.
- GPU fields appear in SparkV1alpha1SparkPodSpec, SparkV1alpha1ServerSpec,
  and SparkV1alpha1ExecutorSpec after codegen.
- GPU configuration survives dict/JSON round trips for all three models.
- A complete SparkConnect spec preserves both server and executor GPU settings.
- Omitting GPU preserves CPU-only behavior (no gpu key in serialized output).
- Required imports work without circular-import errors.
"""

import json
import sys
import os

# Allow running directly from this directory or from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from kubeflow_spark_api.models.spark_v1alpha1_gpu_spec import SparkV1alpha1GPUSpec
from kubeflow_spark_api.models.spark_v1alpha1_spark_pod_spec import SparkV1alpha1SparkPodSpec
from kubeflow_spark_api.models.spark_v1alpha1_server_spec import SparkV1alpha1ServerSpec
from kubeflow_spark_api.models.spark_v1alpha1_executor_spec import SparkV1alpha1ExecutorSpec
from kubeflow_spark_api.models.spark_v1alpha1_spark_connect_spec import SparkV1alpha1SparkConnectSpec
from kubeflow_spark_api.models import SparkV1alpha1GPUSpec as GPUSpecFromInit


# ---------------------------------------------------------------------------
# Import / export tests
# ---------------------------------------------------------------------------

class TestGPUSpecImport:
    """SparkV1alpha1GPUSpec is exported from models/__init__.py."""

    def test_importable_from_models_init(self):
        assert GPUSpecFromInit is SparkV1alpha1GPUSpec

    def test_class_has_required_fields(self):
        # The generated model must expose 'name' and 'quantity'.
        spec = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=2)
        assert spec.name == "nvidia.com/gpu"
        assert spec.quantity == 2


# ---------------------------------------------------------------------------
# GPUSpec round-trip tests
# ---------------------------------------------------------------------------

class TestGPUSpecRoundTrip:
    """SparkV1alpha1GPUSpec survives dict and JSON serialization."""

    def test_to_dict(self):
        spec = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=4)
        d = spec.to_dict()
        assert d == {"name": "nvidia.com/gpu", "quantity": 4}

    def test_from_dict(self):
        d = {"name": "amd.com/gpu", "quantity": 1}
        spec = SparkV1alpha1GPUSpec.from_dict(d)
        assert spec.name == "amd.com/gpu"
        assert spec.quantity == 1

    def test_to_json_and_from_json(self):
        spec = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=2)
        json_str = spec.to_json()
        data = json.loads(json_str)
        assert data["name"] == "nvidia.com/gpu"
        assert data["quantity"] == 2
        # Reconstruct from JSON string.
        restored = SparkV1alpha1GPUSpec.from_json(json_str)
        assert restored.name == spec.name
        assert restored.quantity == spec.quantity

    def test_from_dict_none_returns_none(self):
        assert SparkV1alpha1GPUSpec.from_dict(None) is None


# ---------------------------------------------------------------------------
# SparkPodSpec GPU field
# ---------------------------------------------------------------------------

class TestSparkPodSpecGPU:
    """SparkV1alpha1SparkPodSpec exposes an optional gpu field."""

    def test_gpu_field_absent_by_default(self):
        pod_spec = SparkV1alpha1SparkPodSpec()
        assert pod_spec.gpu is None
        # Serialized output must not contain the 'gpu' key.
        d = pod_spec.to_dict()
        assert "gpu" not in d

    def test_gpu_field_set_and_serialized(self):
        gpu = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=2)
        pod_spec = SparkV1alpha1SparkPodSpec(gpu=gpu)
        d = pod_spec.to_dict()
        assert d["gpu"] == {"name": "nvidia.com/gpu", "quantity": 2}

    def test_gpu_round_trip_via_dict(self):
        gpu = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=8)
        original = SparkV1alpha1SparkPodSpec(cores=4, memory="8g", gpu=gpu)
        restored = SparkV1alpha1SparkPodSpec.from_dict(original.to_dict())
        assert restored.gpu is not None
        assert restored.gpu.name == "nvidia.com/gpu"
        assert restored.gpu.quantity == 8
        assert restored.cores == 4
        assert restored.memory == "8g"


# ---------------------------------------------------------------------------
# ServerSpec GPU field
# ---------------------------------------------------------------------------

class TestServerSpecGPU:
    """SparkV1alpha1ServerSpec exposes an optional gpu field."""

    def test_gpu_field_absent_by_default(self):
        server = SparkV1alpha1ServerSpec()
        assert server.gpu is None
        d = server.to_dict()
        assert "gpu" not in d

    def test_gpu_field_set_and_serialized(self):
        gpu = SparkV1alpha1GPUSpec(name="amd.com/gpu", quantity=1)
        server = SparkV1alpha1ServerSpec(gpu=gpu)
        d = server.to_dict()
        assert d["gpu"] == {"name": "amd.com/gpu", "quantity": 1}

    def test_gpu_round_trip_via_dict(self):
        gpu = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=1)
        original = SparkV1alpha1ServerSpec(cores=2, memory="2g", gpu=gpu)
        restored = SparkV1alpha1ServerSpec.from_dict(original.to_dict())
        assert restored.gpu is not None
        assert restored.gpu.name == "nvidia.com/gpu"
        assert restored.gpu.quantity == 1


# ---------------------------------------------------------------------------
# ExecutorSpec GPU field
# ---------------------------------------------------------------------------

class TestExecutorSpecGPU:
    """SparkV1alpha1ExecutorSpec exposes an optional gpu field."""

    def test_gpu_field_absent_by_default(self):
        executor = SparkV1alpha1ExecutorSpec()
        assert executor.gpu is None
        d = executor.to_dict()
        assert "gpu" not in d

    def test_gpu_field_set_and_serialized(self):
        gpu = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=2)
        executor = SparkV1alpha1ExecutorSpec(instances=4, gpu=gpu)
        d = executor.to_dict()
        assert d["gpu"] == {"name": "nvidia.com/gpu", "quantity": 2}
        assert d["instances"] == 4

    def test_gpu_round_trip_via_dict(self):
        gpu = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=4)
        original = SparkV1alpha1ExecutorSpec(cores=4, memory="4g", instances=2, gpu=gpu)
        restored = SparkV1alpha1ExecutorSpec.from_dict(original.to_dict())
        assert restored.gpu is not None
        assert restored.gpu.name == "nvidia.com/gpu"
        assert restored.gpu.quantity == 4
        assert restored.instances == 2


# ---------------------------------------------------------------------------
# Complete SparkConnect spec preserves both server and executor GPU settings
# ---------------------------------------------------------------------------

class TestSparkConnectSpecGPU:
    """Both server and executor GPU settings survive a full spec round trip."""

    def test_server_and_executor_gpu_preserved(self):
        server_gpu = SparkV1alpha1GPUSpec(name="amd.com/gpu", quantity=1)
        executor_gpu = SparkV1alpha1GPUSpec(name="nvidia.com/gpu", quantity=2)
        server = SparkV1alpha1ServerSpec(gpu=server_gpu)
        executor = SparkV1alpha1ExecutorSpec(instances=2, gpu=executor_gpu)
        spec = SparkV1alpha1SparkConnectSpec(
            spark_version="4.0.4",
            server=server,
            executor=executor,
        )
        d = spec.to_dict()
        assert d["server"]["gpu"] == {"name": "amd.com/gpu", "quantity": 1}
        assert d["executor"]["gpu"] == {"name": "nvidia.com/gpu", "quantity": 2}

        # Round-trip through dict.
        restored = SparkV1alpha1SparkConnectSpec.from_dict(d)
        assert restored.server.gpu.name == "amd.com/gpu"
        assert restored.server.gpu.quantity == 1
        assert restored.executor.gpu.name == "nvidia.com/gpu"
        assert restored.executor.gpu.quantity == 2

    def test_cpu_only_spec_has_no_gpu_keys(self):
        """CPU-only configuration must not emit gpu keys."""
        server = SparkV1alpha1ServerSpec(cores=2, memory="2g")
        executor = SparkV1alpha1ExecutorSpec(cores=2, memory="2g", instances=2)
        spec = SparkV1alpha1SparkConnectSpec(
            spark_version="4.0.4",
            server=server,
            executor=executor,
        )
        d = spec.to_dict()
        assert "gpu" not in d.get("server", {})
        assert "gpu" not in d.get("executor", {})
