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
Time simulator for CPU simulation mode.

This module provides utilities to simulate NPU execution time
by adding configurable delays to operations.
"""

import time
import threading
from functools import wraps
from typing import Any, Callable, Optional

from vllm_ascend import envs


# Thread-local storage for tracking operation count
_operation_count = threading.local()


def get_simulated_time_ms() -> float:
    """Get the configured simulated execution time in milliseconds."""
    try:
        return envs.VLLM_ASCEND_CPU_SIMULATED_TIME_MS
    except Exception:
        return 0.0


def is_simulation_enabled() -> bool:
    """Check if CPU simulation is enabled."""
    try:
        return envs.VLLM_ASCEND_ENABLE_CPU_SIMULATION
    except Exception:
        return False


def maybe_add_simulated_delay() -> None:
    """
    Add simulated delay if CPU simulation mode is enabled.

    This function is called by mock operations to simulate
    the execution time that would occur on a real NPU.
    """
    if not is_simulation_enabled():
        return

    delay_ms = get_simulated_time_ms()
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


def simulate_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to simulate execution time for a function.

    Usage:
        @simulate_time
        def some_npu_operation():
            # actual operation
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        maybe_add_simulated_delay()
        return func(*args, **kwargs)

    return wrapper


class TimeSimulator:
    """
    Context manager for simulating execution time.

    Usage:
        with TimeSimulator("operation_name"):
            # perform operation
    """

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name

    def __enter__(self):
        maybe_add_simulated_delay()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        maybe_add_simulated_delay()
        return False


def get_operation_count() -> int:
    """Get the current operation count for the current thread."""
    return getattr(_operation_count, 'count', 0)


def increment_operation_count() -> int:
    """Increment and return the operation count for the current thread."""
    if not hasattr(_operation_count, 'count'):
        _operation_count.count = 0
    _operation_count.count += 1
    return _operation_count.count


def reset_operation_count() -> None:
    """Reset the operation count for the current thread."""
    _operation_count.count = 0
