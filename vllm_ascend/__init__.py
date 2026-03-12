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

# CPU Simulation Mode: Inject mock modules BEFORE any other imports
# This must be at the very beginning of this file to prevent torch_npu
# from being loaded from the real NPU driver
import os
if os.environ.get("VLLM_ASCEND_ENABLE_CPU_SIMULATION", "0") == "1":
    import sys
    from unittest.mock import MagicMock

    # Inject mock modules before any NPU-related imports
    if 'torch_npu' not in sys.modules:
        sys.modules['torch_npu'] = MagicMock()
    if 'torch.npu' not in sys.modules:
        sys.modules['torch.npu'] = MagicMock()
    if 'torch_npu._inductor' not in sys.modules:
        sys.modules['torch_npu._inductor'] = MagicMock()
    if 'torch_npu.npu' not in sys.modules:
        sys.modules['torch_npu.npu'] = MagicMock()
    if 'torch_npu.profiler' not in sys.modules:
        sys.modules['torch_npu.profiler'] = MagicMock()
    if 'torch_npu.op_plugin' not in sys.modules:
        sys.modules['torch_npu.op_plugin'] = MagicMock()
    if 'torch_npu.utils' not in sys.modules:
        sys.modules['torch_npu.utils'] = MagicMock()
    # Mock triton.runtime
    if 'triton.runtime' not in sys.modules:
        triton_runtime = MagicMock()
        triton_runtime.driver.active.utils.get_device_properties.return_value = {
            'num_aic': 8,
            'num_vectorcore': 8,
        }
        sys.modules['triton.runtime'] = triton_runtime

    # Mock vllm.platforms if it fails to import current_platform
    # This handles the case where vllm.platforms has import errors
    if 'vllm.platforms' not in sys.modules:
        mock_platforms = MagicMock()
        # Provide mock for Platform and PlatformEnum
        mock_platforms.Platform = MagicMock()
        mock_platforms.PlatformEnum = MagicMock()
        # Provide mock for current_platform if it doesn't exist
        mock_platforms.current_platform = MagicMock()
        mock_platforms.current_platform.get_global_graph_pool = MagicMock(return_value=None)
        mock_platforms.current_platform.get_device_capability = MagicMock(return_value=(8, 0))
        sys.modules['vllm.platforms'] = mock_platforms


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    from vllm_ascend.distributed import register_connector
    register_connector()


def register_model_loader():
    from .model_loader.netloader import register_netloader
    register_netloader()


def register_service_profiling():
    from .profiling_config import generate_service_profiling_config
    generate_service_profiling_config()
