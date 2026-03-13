#!/usr/bin/env python
"""
vllm-ascend CPU Simulation Bootstrap

This script must be used when running vllm serve in CPU simulation mode.
It injects mock modules BEFORE importing vllm to prevent torch_npu from
being loaded.

Usage:
    python -m vllm_ascend.cpu_simulation.bootstrap <vllm command>

Or as a wrapper:
    export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
    python -m vllm_ascend.cpu_simulation.bootstrap serve Qwen/Qwen2-0.5B-Instruct --host 0.0.0.0 --port 8000
"""

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, Mock


def create_mock_module(name):
    """Create a mock module with proper __spec__ attribute."""
    mock = MagicMock(spec=ModuleType)
    mock.__name__ = name
    mock.__spec__ = MagicMock()
    mock.__spec__.name = name
    mock.__spec__.submodule_search_locations = None
    return mock


def inject_cpu_simulation_mocks():
    """Inject mock modules before any other imports."""
    if os.environ.get("VLLM_ASCEND_ENABLE_CPU_SIMULATION", "0") != "1":
        return False

    print("[vllm-ascend] CPU Simulation Mode: Injecting mocks...")

    # Inject mock modules before any NPU-related imports
    mock_modules = [
        'torch_npu',
        'torch.npu',
        'torch_npu._inductor',
        'torch_npu.npu',
        'torch_npu.profiler',
        'torch_npu.op_plugin',
        'torch_npu.utils',
    ]

    for mod_name in mock_modules:
        sys.modules[mod_name] = create_mock_module(mod_name)

    # Import torch to get real dtypes for torch_npu mock
    import torch

    # Set up proper dtype attributes on torch_npu mock
    torch_npu_mock = sys.modules['torch_npu']
    # These are the FP8 dtypes that vllm uses - get them safely
    try:
        torch_npu_mock.float8_e4m3fn = torch.float8_e4m3fn
    except AttributeError:
        torch_npu_mock.float8_e4m3fn = torch.float32
    try:
        torch_npu_mock.float8_e5m2 = torch.float8_e5m2
    except AttributeError:
        torch_npu_mock.float8_e5m2 = torch.float32

    # Mock triton with proper structure
    # Need to mock the full triton module before any submodules
    mock_triton = MagicMock()
    mock_triton.__path__ = ['triton']
    mock_triton.__spec__ = MagicMock()
    mock_triton.__spec__.name = 'triton'
    mock_triton.__spec__.submodule_search_locations = ['triton']

    # Mock triton.runtime with proper nested structure
    triton_runtime = MagicMock()
    triton_runtime.__spec__ = MagicMock()
    triton_runtime.__spec__.name = 'triton.runtime'
    triton_runtime.__spec__.submodule_search_locations = None

    # Create proper nested mock structure for triton.runtime.driver.active.utils
    triton_driver = MagicMock()
    triton_active = MagicMock()
    triton_utils = MagicMock()
    triton_utils.get_device_properties = MagicMock(return_value={
        'num_aic': 8,
        'num_vectorcore': 8,
    })
    triton_active.utils = triton_utils
    triton_driver.active = triton_active
    triton_runtime.driver = triton_driver

    # Add autotune mock - this is critical for vllm
    triton_runtime.autotune = MagicMock()
    triton_runtime.autotune.__spec__ = MagicMock()

    # Add jit mock - needed by vllm
    triton_runtime.jit = MagicMock()
    triton_runtime.jit.__spec__ = MagicMock()

    # Add cache mock - needed by triton
    triton_runtime.cache = MagicMock()

    # Make triton.runtime look like a package (needed for submodules)
    triton_runtime.__path__ = ['triton.runtime']

    mock_triton.runtime = triton_runtime

    # Add more triton submodules that might be imported
    mock_triton.compiler = MagicMock()
    mock_triton.compiler.__spec__ = MagicMock()

    # Add triton.backends mock
    mock_triton.backends = MagicMock()
    mock_triton.backends.__spec__ = MagicMock()
    mock_triton.backends.__path__ = ['triton.backends']

    # Add triton.backends.compiler mock
    mock_triton.backends.compiler = MagicMock()
    mock_triton.backends.compiler.__spec__ = MagicMock()

    # Add triton.runtime.autotuner mock
    mock_triton.runtime.autotuner = MagicMock()
    mock_triton.runtime.autotuner.__spec__ = MagicMock()

    sys.modules['triton'] = mock_triton
    sys.modules['triton.backends'] = mock_triton.backends
    sys.modules['triton.runtime'] = triton_runtime
    sys.modules['triton.runtime.jit'] = triton_runtime.jit
    sys.modules['triton.runtime.cache'] = triton_runtime.cache

    # Mock vllm.platforms with full current_platform support
    mock_platforms = create_mock_module('vllm.platforms')

    # Create a proper mock for CPU platform
    mock_cpu_platform = MagicMock()
    mock_cpu_platform.get_device_capability = MagicMock(return_value=(8, 0))

    # Mock CpuArchEnum
    mock_cpu_arch_enum = MagicMock()
    mock_cpu_arch_enum.ARM = "ARM"
    mock_cpu_arch_enum.X86_64 = "X86_64"

    # Mock Platform and PlatformEnum
    mock_platforms.Platform = MagicMock()
    mock_platforms.PlatformEnum = MagicMock()
    mock_platforms.CpuArchEnum = mock_cpu_arch_enum

    # Mock current_platform with proper dispatch_key and all required methods
    # Use plain Mock with explicit attributes instead of MagicMock
    mock_current_platform = Mock()

    # Set up all required methods
    mock_current_platform.get_global_graph_pool = Mock(return_value=None)
    mock_current_platform.get_device_capability = Mock(return_value=(8, 0))
    mock_current_platform.support_static_graph_mode = Mock(return_value=False)
    mock_current_platform.check_and_update_config = Mock(return_value=None)
    mock_current_platform.is_cuda_alike = Mock(return_value=False)
    mock_current_platform.support_hybrid_kv_cache = Mock(return_value=False)

    # Use string for dispatch_key - vllm code expects string for concatenation
    mock_current_platform.dispatch_key = "CPU"

    mock_platforms.current_platform = mock_current_platform

    mock_platforms.CPUPlatform = mock_cpu_platform
    mock_platforms.CPU = mock_cpu_platform

    sys.modules['vllm.platforms'] = mock_platforms

    # Also mock vllm.platforms.cpu
    sys.modules['vllm.platforms.cpu'] = mock_cpu_platform

    # Also mock vllm.platforms.cuda with proper methods
    mock_cuda_platform = MagicMock()
    mock_cuda_platform.get_device_capability = MagicMock(return_value=(8, 0))
    sys.modules['vllm.platforms.cuda'] = mock_cuda_platform

    # Add torch_npu._C mock for internal functions
    sys.modules['torch_npu._C'] = MagicMock()

    # Mock vllm.utils.torch_utils to skip direct_register_custom_op
    # This avoids dispatch_key issues with torch.library.Library.impl
    mock_torch_utils = MagicMock()

    def mock_direct_register_custom_op(op_name, op_func, **kwargs):
        """Mock that does nothing - skips custom op registration."""
        pass

    mock_torch_utils.direct_register_custom_op = mock_direct_register_custom_op
    sys.modules['vllm.utils.torch_utils'] = mock_torch_utils

    print("[vllm-ascend] CPU Simulation Mode: Mocks injected successfully")
    return True


def main():
    """Main entry point."""
    # Inject mocks FIRST, before any other imports
    inject_cpu_simulation_mocks()

    # Now import and run vllm CLI
    from vllm.entrypoints.cli.main import main as vllm_main

    # Pass through all arguments
    sys.argv[0] = "vllm"  # Fix argv[0] for vllm CLI
    vllm_main()


if __name__ == "__main__":
    main()
