# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.cuda.synchronize()


class _GlooBroadcastGroup:
    """CPU-mediated stand-in for PyNcclCommunicator's broadcast.

    exp2 hybrid colocation puts trainer rank 0 and one vLLM engine on the SAME
    GPU, and an NCCL communicator refuses two members on one CUDA device
    ("invalid usage"). Gloo has no device identity, so the weight sync rides
    host memory instead: src stages the tensor on CPU, gloo broadcasts over
    loopback TCP, receivers copy back to their device. Slower than NCCL —
    acceptable at macro-batch phase cadence, not for per-second syncs.
    """

    def __init__(self, master_address, master_port, rank, world_size):
        import datetime
        import os

        import torch
        import torch.distributed

        self._torch = torch
        store = torch.distributed.TCPStore(
            master_address, master_port, world_size,
            is_master=(rank == 0),
            timeout=datetime.timedelta(minutes=30),
        )
        self.pg = torch.distributed.ProcessGroupGloo(
            store, rank, world_size, datetime.timedelta(minutes=30))
        self.rank = rank
        self.reuse_host_buffer = os.environ.get("MOLT_GLOO_REUSE_BUFFER") == "1"
        self._host_buffer = None

    def broadcast(self, tensor, src=0, stream=None):  # noqa: ARG002 — NCCL-API compatible
        torch = self._torch
        reuse = self.reuse_host_buffer and tensor.is_cuda and tensor.is_contiguous()
        if reuse:
            # Only the sender has useful device data. Reuse a pinned byte buffer;
            # synchronous copies keep Gloo and the next refit from racing CUDA.
            nbytes = tensor.numel() * tensor.element_size()
            if self._host_buffer is None or self._host_buffer.numel() < nbytes:
                self._host_buffer = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
            cpu = self._host_buffer[:nbytes].view(tensor.dtype).view(tensor.shape)
            if self.rank == src:
                cpu.copy_(tensor.detach(), non_blocking=False)
        else:
            cpu = tensor.detach().cpu() if tensor.is_cuda else tensor
        opts = torch.distributed.BroadcastOptions()
        opts.rootRank = src
        opts.rootTensor = 0
        self.pg.broadcast([cpu], opts).wait()
        if tensor.is_cuda and (not reuse or self.rank != src):
            tensor.copy_(cpu, non_blocking=False)
        return tensor


def stateless_init_process_group(master_address, master_port, rank, world_size, device, backend="nccl"):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers. backend="gloo" swaps in the CPU-mediated group above
    for topologies where NCCL cannot form (two members on one GPU).
    """
    if backend == "gloo":
        return _GlooBroadcastGroup(master_address, master_port, rank, world_size)

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    return PyNcclCommunicator(pg, device=device)
