"""CPU-offloaded optimizer resume: the Adam state must be created on CPU before DCP
reads the optimizer state dict, or DCP's _init_optim_state creates it on the GPU params
(a 4B resume OOMed there next to the resident vLLM). CPU-only: the "GPU" shards are
plain CPU tensors here; what is checked is the state placement, the no-op of DCP's
own init afterwards, and a DCP save/load round trip through the new path."""
import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "molt"))

from molt.trainer.fsdp.optimizer_offload import CpuOptimizerOffloader  # noqa: E402


def _model_and_optimizer(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
    model[1].bias.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, foreach=False, fused=True)
    return model, opt


def _trainable(opt):
    return [p for g in opt.param_groups for p in g["params"] if p.requires_grad]


def test_init_state_on_cpu_creates_moments_without_touching_params():
    model, opt = _model_and_optimizer()
    before = [p.detach().clone() for p in model.parameters()]
    off = CpuOptimizerOffloader()
    off.init_state_on_cpu(opt, _trainable(opt))
    params = _trainable(opt)
    assert set(map(id, opt.state)) == set(map(id, params))
    for p in params:
        st = opt.state[p]
        assert st["exp_avg"].device.type == "cpu" and st["exp_avg_sq"].device.type == "cpu"
        assert not st["exp_avg"].any() and not st["exp_avg_sq"].any()
        assert p.grad is None
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(g["lr"] == 1e-3 for g in opt.param_groups)
    # DCP's own initializer now has nothing to do (it would run a step on the params).
    from torch.distributed.checkpoint.state_dict import _init_optim_state
    ids = {id(v) for st in opt.state.values() for v in st.values()}
    _init_optim_state(opt)
    assert {id(v) for st in opt.state.values() for v in st.values()} == ids
    # Second call is a no-op; a real step afterwards still works and moves the params.
    off.init_state_on_cpu(opt, params)
    model(torch.randn(3, 8)).sum().backward()
    off.step(opt, [p for p in params if p.grad is not None])
    assert not all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_dcp_round_trip_through_cpu_init():
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

    with tempfile.TemporaryDirectory() as tmp:
        if not dist.is_initialized():
            dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=0, world_size=1)
        try:
            model, opt = _model_and_optimizer()
            off = CpuOptimizerOffloader()
            for _ in range(2):
                model(torch.randn(3, 8)).sum().backward()
                off.step(opt, [p for p in _trainable(opt) if p.grad is not None])
                opt.zero_grad(set_to_none=True)
            dcp.save(get_optimizer_state_dict(model, opt), checkpoint_id=f"{tmp}/optim")
            saved = {k: {n: v.clone() for n, v in st.items()} for k, st in zip(range(99), opt.state.values())}

            model2, opt2 = _model_and_optimizer(seed=1)
            CpuOptimizerOffloader().init_state_on_cpu(opt2, _trainable(opt2))  # what load_ckpt now does first
            sd = get_optimizer_state_dict(model2, opt2)
            dcp.load(sd, checkpoint_id=f"{tmp}/optim")
            # load_ckpt relies on this: dcp.load restores the optimizer's own state tensors
            # in place, so set_optimizer_state_dict (which casts them onto the param device)
            # is not needed for the moments.
            for k, st in zip(range(99), opt2.state.values()):
                for n, v in st.items():
                    assert torch.equal(v, saved[k][n]), n
                    assert v.device.type == "cpu"
            set_optimizer_state_dict(model2, opt2, optim_state_dict=sd)  # the non-offload path still agrees
            for k, st in zip(range(99), opt2.state.values()):
                for n, v in st.items():
                    assert torch.equal(v, saved[k][n]), n
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
