"""The fused chunked log-prob/entropy path must match the full fp32 path."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.environ.get("MOLT_DIR", ""))
utils = pytest.importorskip("molt.models.utils")


def reference(logits, labels, temperature):
    full = logits.to(torch.float32)
    log_probs = utils.log_probs_from_logits(full, labels, temperature=temperature)
    src = full / temperature if temperature != 1.0 else full
    entropy = torch.logsumexp(src, dim=-1) - torch.sum(torch.softmax(src, dim=-1) * src, dim=-1)
    return log_probs, entropy


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunked_values_and_gradients_match_full_fp32(temperature):
    torch.manual_seed(0)
    base = (torch.randn(2, 37, 101) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 101, (2, 37))
    weights = torch.randn(2, 37), torch.randn(2, 37)
    results = []
    for fn in (reference, lambda x, y, t: utils.log_probs_and_entropy_from_logits(x, y, t, chunk_size=16)):
        logits = base.clone().requires_grad_(True)
        log_probs, entropy = fn(logits, labels, temperature)
        ((log_probs * weights[0]).sum() + (entropy * weights[1]).sum()).backward()
        results.append((log_probs.detach(), entropy.detach(), logits.grad.float()))
    for got, want in zip(results[1], results[0]):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
