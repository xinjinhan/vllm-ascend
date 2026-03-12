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


def inject_cpu_simulation_mocks():
    """Inject mock modules before any other imports."""
    if os.environ.get("VLLM_ASCEND_ENABLE_CPU_SIMULATION", "0") != "1":
        return

    print("[vllm-ascend] CPU Simulation Mode: Injecting mocks...")

    from unittest.mock import MagicMock
    import sys

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
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # Mock triton.runtime
    if 'triton.runtime' not in sys.modules:
        triton_runtime = MagicMock()
        triton_runtime.driver.active.utils.get_device_properties.return_value = {
            'num_aic': 8,
            'num_vectorcore': 8,
        }
        sys.modules['triton.runtime'] = triton_runtime

    # Mock vllm.platforms if needed
    if 'vllm.platforms' not in sys.modules:
        mock_platforms = MagicMock()
        mock_platforms.Platform = MagicMock()
        mock_platforms.PlatformEnum = MagicMock()
        mock_platforms.current_platform = MagicMock()
        mock_platforms.current_platform.get_global_graph_pool = MagicMock(return_value=None)
        mock_platforms.current_platform.get_device_capability = MagicMock(return_value=(8, 0))
        sys.modules['vllm.platforms'] = mock_platforms

    print("[vllm-ascend] CPU Simulation Mode: Mocks injected successfully")


def main():
    """Main entry point."""
    inject_cpu_simulation_mocks()

    # Import and run vllm CLI
    from vllm.entrypoints.cli.main import main as vllm_main

    # Pass through all arguments
    sys.argv[0] = "vllm"  # Fix argv[0] for vllm CLI
    vllm_main()


if __name__ == "__main__":
    main()
