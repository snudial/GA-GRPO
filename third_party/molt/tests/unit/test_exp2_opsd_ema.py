"""exp2 EMA teacher: the update must see the post-step parameters even while
the optimizer kernels are still queued on the GPU."""
import pytest
import torch

from molt.trainer.workers.exp2_opsd import Exp2OpsdMixin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Actor(Exp2OpsdMixin):
    def __init__(self, model):
        self.actor = type("Actor", (), {"model": model})()


def test_ema_step_sees_queued_update(monkeypatch):
    monkeypatch.setenv("EXP2_EMA_BETA", "0.5")
    model = torch.nn.Linear(1024, 1024, device="cuda")
    actor = _Actor(model)
    actor.exp2_setup()
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        torch.cuda._sleep(200_000_000)  # a busy stream, as right after an optimizer step
        for p in model.parameters():
            p.add_(1.0)
    actor._exp2_ema_step()
    for n, _ in model.named_parameters():
        torch.testing.assert_close(actor._exp2_ema[n], before[n] + 0.5)
