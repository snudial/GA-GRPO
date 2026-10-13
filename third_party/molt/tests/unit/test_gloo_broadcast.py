"""Transfer semantics without reserving GPUs used by an active experiment."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from molt.utils.distributed_util import _GlooBroadcastGroup


class DeviceTensor(torch.Tensor):
    """CPU storage exposing the CUDA branch; actual CUDA is tested by the benchmark."""
    copies = []

    @property
    def is_cuda(self):
        return True

    def cpu(self):
        self.copies.append("device_to_host_legacy")
        return self.as_subclass(torch.Tensor).clone()

    def copy_(self, src, non_blocking=False):
        self.copies.append("host_to_device")
        return super().copy_(src, non_blocking=non_blocking)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("src", [0, 1])
def test_broadcast_repeated_updates_match_exactly(monkeypatch, enabled, dtype, src):
    original_empty = torch.empty
    allocations = []

    def empty(*args, **kwargs):
        allocations.append(kwargs.pop("pin_memory", False))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    mailbox = {}
    groups = []
    for rank in range(2):
        def broadcast(tensors, options, rank=rank):
            assert options.rootRank == src
            if rank == src:
                mailbox['data'] = tensors[0].as_subclass(torch.Tensor).clone()
            else:
                tensors[0].copy_(mailbox['data'])
            return SimpleNamespace(wait=lambda: None)

        group = _GlooBroadcastGroup.__new__(_GlooBroadcastGroup)
        group._torch = torch
        group.rank = rank
        group.reuse_host_buffer = enabled
        group._host_buffer = None
        group.pg = SimpleNamespace(broadcast=broadcast)
        groups.append(group)

    DeviceTensor.copies.clear()
    for count in [32, 16, 64, 8]:
        expected = torch.arange(count, dtype=torch.float32).to(dtype).reshape(2, -1)
        sender = expected.clone().as_subclass(DeviceTensor)
        receiver = torch.full_like(expected, 7).as_subclass(DeviceTensor)
        groups[src].broadcast(sender, src=src)
        groups[1-src].broadcast(receiver, src=src)
        assert torch.equal(sender.as_subclass(torch.Tensor), expected)
        assert torch.equal(receiver.as_subclass(torch.Tensor), expected)
    if enabled:
        assert allocations == [True]*4  # two capacities, two ranks; smaller sends reuse
        assert DeviceTensor.copies == ["host_to_device"]*4
    else:
        assert not allocations
        assert DeviceTensor.copies.count("device_to_host_legacy") == 8
        assert DeviceTensor.copies.count("host_to_device") == 8


def test_cpu_tensor_uses_direct_collective():
    group = _GlooBroadcastGroup.__new__(_GlooBroadcastGroup)
    group._torch = torch
    group.rank = 1
    group.reuse_host_buffer = True
    group._host_buffer = None
    group.pg = SimpleNamespace(broadcast=Mock(return_value=SimpleNamespace(wait=lambda: None)))
    tensor = torch.arange(5)
    assert group.broadcast(tensor) is tensor
    assert group.pg.broadcast.call_args.args[0][0] is tensor
    assert group._host_buffer is None
