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

    # Patch torch.library.Library.impl to handle non-string dispatch_key
    # This fixes: TypeError: can only concatenate str (not "torch._C.DispatchKey") to str
    # vllm does: key = ns + "/" + name + "/" + dispatch_key
    import torch.library
    _orig_lib_impl = torch.library.Library.impl

    def _fixed_lib_impl(self, op_name, fn, dispatch_key=None, **kwargs):
        # Only convert non-string dispatch_key to string, leave None and strings as-is
        # This avoids conflicts with torch's internal dispatch key handling
        if dispatch_key is not None and not isinstance(dispatch_key, str):
            dispatch_key = str(dispatch_key)
        return _orig_lib_impl(self, op_name, fn, dispatch_key=dispatch_key, **kwargs)

    torch.library.Library.impl = _fixed_lib_impl

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

    # Mock triton.runtime with proper nested structure
    # Use plain MagicMock instead of create_mock_module to ensure attributes work
    triton_runtime = MagicMock()

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

    # Add jit mock - needed by vllm
    triton_runtime.jit = MagicMock()

    sys.modules['triton.runtime'] = triton_runtime
    sys.modules['triton.runtime.jit'] = triton_runtime.jit

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
