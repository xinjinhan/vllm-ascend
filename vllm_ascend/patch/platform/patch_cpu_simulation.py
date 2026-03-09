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
CPU Simulation Patch for vllm-ascend.

This patch applies CPU simulation mode by replacing NPU-specific
operations with CPU-based mock implementations.
"""

import sys
from unittest.mock import MagicMock


def apply_cpu_simulation_patch():
    """
    Apply CPU simulation patch to replace NPU operations with mocks.

    This function should be called early in the application startup,
    preferably during the global patch phase (is_global_patch=True).
    """
    from vllm_ascend import envs

    if not envs.VLLM_ASCEND_ENABLE_CPU_SIMULATION:
        return

    # Mock torch_npu module
    if 'torch_npu' not in sys.modules:
        sys.modules['torch_npu'] = MagicMock()

    # Mock torch.npu namespace
    if 'torch.npu' not in sys.modules:
        sys.modules['torch.npu'] = MagicMock()

    # Mock torch_npu._inductor
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

    # Mock torch_npu.profiler
    if 'torch_npu.profiler' not in sys.modules:
        sys.modules['torch_npu.profiler'] = MagicMock()

    # Mock torch_npu.op_plugin
    if 'torch_npu.op_plugin' not in sys.modules:
        sys.modules['torch_npu.op_plugin'] = MagicMock()

    # Mock torch_npu.npu
    if 'torch_npu.npu' not in sys.modules:
        sys.modules['torch_npu.npu'] = MagicMock()

    # Mock torch_npu.utils
    if 'torch_npu.utils' not in sys.modules:
        sys.modules['torch_npu.utils'] = MagicMock()
