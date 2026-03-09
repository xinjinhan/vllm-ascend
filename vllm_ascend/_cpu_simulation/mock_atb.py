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
Mock implementation of ATB (Ascend Transformer Boost) for CPU simulation.

This module provides mock implementations for ATB operations that return
fake results without performing actual calculations.
"""

from typing import Any, Optional

import torch

from vllm_ascend._cpu_simulation.time_simulator import maybe_add_simulated_delay


class MockATBExtensions:
    """Mock ATB extensions registration."""

    @staticmethod
    def register_extensions() -> None:
        """Mock ATB extensions registration (no-op)."""
        maybe_add_simulated_delay()
        pass


class MockATBOperation:
    """Mock ATB operation."""

    def __init__(self, name: str):
        self.name = name

    def forward(self, *args, **kwargs):
        """Mock forward - returns zeros with same shape as input."""
        maybe_add_simulated_delay()
        if args:
            # Return zeros with same shape and dtype as first tensor argument
            input_tensor = args[0]
            return torch.zeros_like(input_tensor)
        return None


class MockATB:
    """Mock ATB (Ascend Transformer Boost) module."""

    def __init__(self):
        maybe_add_simulated_delay()

    def register_extensions(self) -> None:
        """Register ATB extensions (mock)."""
        maybe_add_simulated_delay()

    def linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mock ATB linear operation.

        Returns a fake output tensor with same shape as input (batch, seq, hidden).
        """
        maybe_add_simulated_delay()
        # Return zeros with same shape as input (batch, seq, weight.shape[0])
        output_shape = (input.shape[0], input.shape[1], weight.shape[0])
        return torch.zeros(output_shape, dtype=input.dtype, device=input.device)

    def layernorm(
        self, input: torch.Tensor, normalized_shape: Any, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mock ATB layer norm operation.

        Returns zeros with same shape as input.
        """
        maybe_add_simulated_delay()
        return torch.zeros_like(input)

    def softmax(self, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Mock ATB softmax operation.

        Returns zeros with same shape as input.
        """
        maybe_add_simulated_delay()
        return torch.zeros_like(input)

    def matmul(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Mock ATB matmul operation.

        Returns zeros with appropriate shape.
        """
        maybe_add_simulated_delay()
        output_shape = input1.shape[:-1] + (input2.shape[-1],)
        return torch.zeros(output_shape, dtype=input1.dtype, device=input1.device)


# Singleton instance
atb = MockATB()
atb_extensions = MockATBExtensions()
