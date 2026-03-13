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
Mock implementation of torch_npu module for CPU simulation.

This module provides mock implementations that return fake results
without performing actual NPU computations.
"""

import time
from typing import Any, Optional

import torch

from vllm_ascend._cpu_simulation.time_simulator import maybe_add_simulated_delay


class MockNPU:
    """Mock NPU namespace that returns fake results."""

    def current_device(self) -> int:
        """Return mock current device (always 0)."""
        maybe_add_simulated_delay()
        return 0

    def get_device_name(self, device: int = 0) -> str:
        """Return mock device name."""
        maybe_add_simulated_delay()
        return "Ascend NPU (Mock)"

    def get_device_properties(self, device: int = 0) -> Any:
        """Return mock device properties."""
        maybe_add_simulated_delay()

        class MockDeviceProperties:
            def __init__(self):
                self.name = "Ascend NPU (Mock)"
                self.major = 8
                self.minor = 0
                self.total_memory = 32 * 1024 * 1024 * 1024  # 32GB
                self.multi_processor_count = 8
                self.uuid = "mock-npu-uuid-0000"

            def __repr__(self):
                return (
                    f"MockDeviceProperties(name={self.name!r}, "
                    f"major={self.major}, minor={self.minor}, "
                    f"total_memory={self.total_memory}, "
                    f"multi_processor_count={self.multi_processor_count}, "
                    f"uuid={self.uuid!r})"
                )

        return MockDeviceProperties()

    def get_soc_version(self) -> int:
        """Return mock SOC version (A2 = 220)."""
        maybe_add_simulated_delay()
        return 220  # Ascend A2

    def set_device(self, device: torch.device) -> None:
        """Mock set device (no-op)."""
        maybe_add_simulated_delay()
        pass

    def synchronize(self, device: int = -1) -> None:
        """Mock synchronize (no-op)."""
        maybe_add_simulated_delay()
        pass

    def reset_peak_memory_stats(self, device: int = -1) -> None:
        """Mock reset peak memory stats (no-op)."""
        maybe_add_simulated_delay()
        pass

    def max_memory_allocated(self, device: int = -1) -> int:
        """Return mock max memory allocated."""
        maybe_add_simulated_delay()
        return 0

    def empty_cache(self) -> None:
        """Mock empty cache (no-op)."""
        maybe_add_simulated_delay()
        pass

    def current_stream(self, device: int = -1) -> "MockStream":
        """Return a mock stream."""
        maybe_add_simulated_delay()
        return MockStream()

    def Stream(self, priority: int = -1) -> "MockStream":
        """Create a mock stream."""
        return MockStream()

    def Event(self, enable_timing: bool = False) -> "MockEvent":
        """Create a mock event."""
        return MockEvent(enable_timing)

    def graph(self, *args, **kwargs) -> "MockNPUGraph":
        """Create a mock NPU graph context."""
        return MockNPUGraph()

    def NPUGraph(self) -> "MockNPUGraph":
        """Create a mock NPU graph."""
        return MockNPUGraph()

    def ExternalEvent(self) -> "MockEvent":
        """Create a mock external event."""
        return MockEvent()

    def set_compile_mode(self, jit_compile: bool = False) -> None:
        """Mock set compile mode (no-op)."""
        maybe_add_simulated_delay()
        pass

    def graph_task_group_begin(self, stream: "MockStream") -> Any:
        """Mock graph task group begin."""
        maybe_add_simulated_delay()
        return None

    def graph_task_group_end(self, stream: "MockStream") -> Any:
        """Mock graph task group end."""
        maybe_add_simulated_delay()
        return None

    def graph_task_update_begin(self, stream: "MockStream", handle: Any) -> None:
        """Mock graph task update begin."""
        maybe_add_simulated_delay()
        pass

    def graph_task_update_end(self, stream: "MockStream") -> None:
        """Mock graph task update end."""
        maybe_add_simulated_delay()
        pass


class MockStream:
    """Mock NPU stream."""

    def synchronize(self) -> None:
        """Mock stream synchronize (no-op)."""
        maybe_add_simulated_delay()
        pass

    def wait_stream(self, stream: "MockStream") -> None:
        """Mock wait stream (no-op)."""
        maybe_add_simulated_delay()
        pass

    def record_event(self, event: "MockEvent") -> None:
        """Mock record event (no-op)."""
        maybe_add_simulated_delay()
        pass

    def wait_event(self, event: "MockEvent") -> None:
        """Mock wait event (no-op)."""
        maybe_add_simulated_delay()
        pass


class MockEvent:
    """Mock NPU event."""

    def __init__(self, enable_timing: bool = False):
        self._enable_timing = enable_timing
        self._record_time: Optional[float] = None

    def record(self, stream: Optional[MockStream] = None) -> None:
        """Mock record event."""
        maybe_add_simulated_delay()
        if self._enable_timing:
            self._record_time = time.time()

    def synchronize(self) -> None:
        """Mock event synchronize (no-op)."""
        maybe_add_simulated_delay()
        pass

    def elapsed_time(self, end_event: "MockEvent") -> float:
        """Return mock elapsed time."""
        maybe_add_simulated_delay()
        if self._record_time and end_event._record_time:
            return (end_event._record_time - self._record_time) * 1000
        return 0.0


class MockNPUGraph:
    """Mock NPU graph."""

    def __init__(self):
        self._captured = False

    def __enter__(self):
        """Enter graph capture context."""
        self._captured = True
        maybe_add_simulated_delay()
        return self

    def __exit__(self, *args):
        """Exit graph capture context."""
        self._captured = False
        maybe_add_simulated_delay()

    def replay(self) -> None:
        """Mock replay graph."""
        maybe_add_simulated_delay()

    def capture_end(self) -> None:
        """Mock capture end."""
        maybe_add_simulated_delay()


# Create singleton mock npu instance
npu = MockNPU()

# Create mock submodules - return None to avoid actual computation
nn = None
