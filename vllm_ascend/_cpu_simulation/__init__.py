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
CPU Simulation Mode Module.

This module provides mock implementations for NPU operations to enable
running vllm-ascend in a CPU-only environment for development and testing.

Usage:
    export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
    export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5

    # Use bootstrap script to inject mocks before importing vllm
    python -m vllm_ascend.cpu_simulation.bootstrap serve <model> ...
"""

from vllm_ascend import envs


def is_cpu_simulation_enabled() -> bool:
    """Check if CPU simulation mode is enabled."""
    return envs.VLLM_ASCEND_ENABLE_CPU_SIMULATION


def get_simulated_time_ms() -> float:
    """Get the configured simulated execution time in milliseconds."""
    return envs.VLLM_ASCEND_CPU_SIMULATED_TIME_MS


def init_cpu_simulation():
    """
    Initialize CPU simulation mode.

    This function should be called early in the application startup
    to inject mock modules before any NPU-related imports.
    """
    if not is_cpu_simulation_enabled():
        return

    import sys
    from unittest.mock import MagicMock

    # Mock torch_npu module if not already mocked
    if 'torch_npu' not in sys.modules:
        sys.modules['torch_npu'] = MagicMock()

    # Mock torch.npu namespace
    if 'torch.npu' not in sys.modules:
        sys.modules['torch.npu'] = MagicMock()

    # Mock torch_npu._inductor if needed
    if 'torch_npu._inductor' not in sys.modules:
        sys.modules['torch_npu._inductor'] = MagicMock()

    # Mock triton.runtime
    if 'triton.runtime' not in sys.modules:
        triton_runtime = MagicMock()
        triton_runtime.driver.active.utils.get_device_properties.return_value = {
            'num_aic': 8,
            'num_vectorcore': 8,
        }
        sys.modules['triton.runtime'] = triton_runtime


# Import and register mock modules when CPU simulation is enabled
if is_cpu_simulation_enabled():
    init_cpu_simulation()
