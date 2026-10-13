"""Regressions for hidden observations and discarded small actor batches."""
from types import SimpleNamespace

import pytest
import torch

from appworld_ga.environment import extract_code
from appworld_ga.prompt import initial_message
from molt.trainer.workers.policy_actor import PolicyTrainer


def test_prompt_prints_results_and_only_uses_public_metadata():
    content = initial_message(dict(instruction='Find my playlists', datetime='2023-05-18',
        supervisor=dict(email='public@example.com', secret='NEVER_RENDER'),
        ground_truth='NEVER_RENDER'))['content']
    assert 'NEVER_RENDER' not in content
    assert 'public@example.com' in content and '2023-05-18' in content
    for api in ['show_app_descriptions()', "show_api_descriptions(app_name='spotify')",
                "show_api_doc(app_name='spotify', api_name='login')"]:
        assert f'print(apis.api_docs.{api})' in content
    assert 'Only printed output' in content


def test_first_complete_block_without_rewriting_response():
    raw = '<think>```python\nbad()\n```</think>Text\n```python\nfirst()\n```\n```python\nsecond()\n```'
    assert extract_code(raw) == 'first()'
    assert extract_code('```python\nunfinished()') is None


@pytest.mark.parametrize('samples,expected_steps', [(4, 1), (8, 1), (12, 2), (0, 0)])
def test_real_policy_loop_flushes_and_normalizes_tail(monkeypatch, samples, expected_steps):
    # Exercise the actual PolicyTrainer loop with real autograd/AdamW on CPU.
    # CUDA transfer, distributed reduction and PPO forward are replaced only;
    # accumulation, token denominators and update decisions are production code.
    class Experience:
        def __init__(self, n):
            self.action_mask = torch.ones(1, n)
            self.attention_mask = torch.ones(1, n)
        def to_device(self, device):
            return self

    class Buffer(list):
        sample_batch_size = 1
        cpu_offload = True
        @staticmethod
        def collate_fn(rows):
            return rows[0]

    trainer = object.__new__(PolicyTrainer)
    trainer.args = SimpleNamespace(train=SimpleNamespace(dynamic_batch_enable=False, force_on_policy=False),
                                   fsdp=SimpleNamespace(cp_size=1, tp_size=1, ep_size=1))
    trainer.replay_buffer = Buffer(Experience(i%3+1) for i in range(samples))
    trainer.dataloader_pin_memory = False
    trainer.max_epochs = 1
    trainer._mfu = None
    trainer.strategy = SimpleNamespace(accumulated_gradient=8, dp_size=1, is_rank_0=lambda: False,
                                      global_token_count=lambda n: n, compute_perf_metrics=lambda *args: {})
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=0.)
    trainer.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    denominators, window_tokens, gradients = [], [], []
    def training_step(exp, kl, denominator, microbatches, last):
        n = exp.action_mask.sum()
        (parameter*n/denominator).backward()
        denominators.append(float(denominator))
        window_tokens.append(float(n))
        if last:
            assert sum(window_tokens) == float(denominator)
            gradients.append(parameter.grad.item())
            optimizer.step(); trainer.actor_scheduler.step(); optimizer.zero_grad()
            window_tokens.clear()
        return {}
    trainer.training_step = training_step
    trainer._record_status = lambda *args: None
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    # Production code asks RandomSampler for a nonempty shuffled buffer.
    import molt.trainer.workers.policy_actor as actor_module
    monkeypatch.setattr(actor_module, 'get_model_parallel_size', lambda _: 2)
    result = trainer.policy_train(0.)
    assert result['actor_optimizer_steps_this_cycle'] == expected_steps
    assert result['actor_optimizer_steps_total'] == expected_steps
    assert result['actor_samples_this_cycle'] == samples
    assert len(denominators) == samples
    assert gradients == pytest.approx([1.]*expected_steps)
    if samples:
        assert parameter.item() < 1.
