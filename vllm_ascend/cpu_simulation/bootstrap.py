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

    # Mock triton.runtime with proper nested structure
    triton_runtime = create_mock_module('triton.runtime')

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

    sys.modules['triton.runtime'] = triton_runtime

    # Mock vllm.platforms with full current_platform support
    mock_platforms = create_mock_module('vllm.platforms')

    # Create a proper mock for CPU platform
    mock_cpu_platform = MagicMock()
    mock_cpu_platform.get_device_capability = MagicMock(return_value=(8, 0))

    # Mock Platform and PlatformEnum
    mock_platforms.Platform = MagicMock()
    mock_platforms.PlatformEnum = MagicMock()

    # Mock current_platform with proper dispatch_key
    mock_current_platform = MagicMock()
    mock_current_platform.get_global_graph_pool = MagicMock(return_value=None)
    mock_current_platform.get_device_capability = MagicMock(return_value=(8, 0))
    # Use a string for dispatch_key to avoid torch.library issues
    mock_current_platform.dispatch_key = "CPU"
    mock_platforms.current_platform = mock_current_platform

    mock_platforms.CPUPlatform = mock_cpu_platform
    mock_platforms.CPU = mock_cpu_platform

    sys.modules['vllm.platforms'] = mock_platforms

    # Also mock vllm.platforms.cpu
    sys.modules['vllm.platforms.cpu'] = mock_cpu_platform

    # Also mock vllm.platforms.cuda
    sys.modules['vllm.platforms.cuda'] = create_mock_module('vllm.platforms.cuda')

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
