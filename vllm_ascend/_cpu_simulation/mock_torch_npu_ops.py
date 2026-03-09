#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""
Mock implementation of torch.npu operations for CPU simulation.

This module provides mock implementations for operations in the torch.npu namespace.
"""

import torch

from vllm_ascend._cpu_simulation.time_simulator import maybe_add_simulated_delay


# Create a mock namespace that mimics torch.npu
class MockNpuNamespace:
    """Mock torch.npu namespace."""

    def current_device(self) -> int:
        maybe_add_simulated_delay()
        return 0

    def get_device_name(self, device: int = 0) -> str:
        maybe_add_simulated_delay()
        return "Ascend NPU (Mock)"

    def get_device_properties(self, device: int = 0):
        maybe_add_simulated_delay()
        return torch.cuda.get_device_properties(device) if hasattr(torch, 'cuda') else None

    def set_device(self, device: torch.device):
        maybe_add_simulated_delay()
        pass

    def synchronize(self, device: int = -1):
        maybe_add_simulated_delay()
        pass

    def reset_peak_memory_stats(self, device: int = -1):
        maybe_add_simulated_delay()
        pass

    def max_memory_allocated(self, device: int = -1):
        maybe_add_simulated_delay()
        return 0

    def empty_cache(self):
        maybe_add_simulated_delay()
        pass

    def current_stream(self, device: int = -1):
        maybe_add_simulated_delay()
        return torch.cuda.current_stream(device) if hasattr(torch, 'cuda') else None

    def Stream(self, priority: int = -1):
        maybe_add_simulated_delay()
        return torch.cuda.Stream() if hasattr(torch, 'cuda') else None

    def Event(self, enable_timing: bool = False):
        maybe_add_simulated_delay()
        return torch.cuda.Event(enable_timing) if hasattr(torch, 'cuda') else None

    def graph(self, *args, **kwargs):
        maybe_add_simulated_delay()
        return torch.cuda.graph(*args, **kwargs) if hasattr(torch, 'cuda') else None

    def NPUGraph(self):
        maybe_add_simulated_delay()
        return torch.cuda.Graph() if hasattr(torch, 'cuda') else None

    def ExternalEvent(self):
        maybe_add_simulated_delay()
        return torch.cuda.Event() if hasattr(torch, 'cuda') else None

    def set_compile_mode(self, jit_compile: bool = False):
        maybe_add_simulated_delay()
        pass

    def graph_task_group_begin(self, stream):
        maybe_add_simulated_delay()
        return None

    def graph_task_group_end(self, stream):
        maybe_add_simulated_delay()
        return None

    def graph_task_update_begin(self, stream, handle):
        maybe_add_simulated_delay()
        pass

    def graph_task_update_end(self, stream):
        maybe_add_simulated_delay()
        pass


# Create singleton instance
npu_ops = MockNpuNamespace()
