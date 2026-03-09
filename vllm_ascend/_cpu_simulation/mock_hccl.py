#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not be used at all except in compliance with the License.
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
Mock implementation of HCCL (Huawei Collective Communication Library) for CPU simulation.

This module provides mock implementations for HCCL operations to enable
running vllm-ascend in a CPU-only environment.
"""

from typing import Any, Optional

import torch

from vllm_ascend._cpu_simulation.time_simulator import maybe_add_simulated_delay


class MockHCCLComm:
    """Mock HCCL communicator."""

    def __init__(self, comm_id: int = 0):
        self.comm_id = comm_id
        self.rank = 0
        self.world_size = 1

    def all_reduce(self, tensor: torch.Tensor, op: int = 0) -> torch.Tensor:
        """Mock all_reduce - returns the input tensor."""
        maybe_add_simulated_delay()
        return tensor

    def all_reduce_(self, tensor: torch.Tensor, op: int = 0) -> torch.Tensor:
        """Mock in-place all_reduce - returns the input tensor."""
        maybe_add_simulated_delay()
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Mock broadcast - returns the input tensor."""
        maybe_add_simulated_delay()
        return tensor

    def broadcast_(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Mock in-place broadcast - returns the input tensor."""
        maybe_add_simulated_delay()
        return tensor

    def all_to_all(
        self, output: torch.Tensor, input: torch.Tensor, split_dim: int = 0
    ) -> torch.Tensor:
        """Mock all_to_all - returns the input tensor."""
        maybe_add_simulated_delay()
        return input

    def all_to_all_(
        self, output: torch.Tensor, input: torch.Tensor, split_dim: int = 0
    ) -> torch.Tensor:
        """Mock in-place all_to_all - returns the input tensor."""
        maybe_add_simulated_delay()
        return input

    def reduce(self, tensor: torch.Tensor, dst: int = 0, op: int = 0) -> torch.Tensor:
        """Mock reduce - returns the input tensor."""
        maybe_add_simulated_delay()
        return tensor

    def reduce_scatter(
        self, output: torch.Tensor, input: torch.Tensor, op: int = 0
    ) -> torch.Tensor:
        """Mock reduce_scatter - returns the input tensor."""
        maybe_add_simulated_delay()
        return input

    def send(self, tensor: torch.Tensor, dst: int = 0) -> None:
        """Mock send - no-op."""
        maybe_add_simulated_delay()
        pass

    def recv(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Mock recv - returns the input tensor."""
        maybe_add_simulated_delay()
        return tensor

    def barrier(self) -> None:
        """Mock barrier - no-op."""
        maybe_add_simulated_delay()
        pass

    def destroy(self) -> None:
        """Mock destroy - no-op."""
        maybe_add_simulated_delay()
        pass


class MockHCCLLibrary:
    """Mock HCCL library wrapper."""

    def __init__(self, library_path: Optional[str] = None):
        maybe_add_simulated_delay()
        self.library_path = library_path

    def get_comm(self, comm_id: int = 0) -> MockHCCLComm:
        """Get a mock HCCL communicator."""
        maybe_add_simulated_delay()
        return MockHCCLComm(comm_id)

    def get_local_rank_id(self) -> int:
        """Get local rank ID (mock)."""
        maybe_add_simulated_delay()
        return 0


# Type definitions for HCCL
hcclComm_t = MockHCCLComm
hcclUniqueId = Any
hcclDataTypeEnum = int
hcclRedOpTypeEnum = int
buffer_type = int
aclrtStream_t = Any


def hcclGetRankId(comm_id: int = 0) -> int:
    """Get rank ID (mock)."""
    maybe_add_simulated_delay()
    return 0


def hcclGetWorldRankCount(comm_id: int = 0) -> int:
    """Get world rank count (mock)."""
    maybe_add_simulated_delay()
    return 1


# Create singleton instance
hccl_library = MockHCCLLibrary()
