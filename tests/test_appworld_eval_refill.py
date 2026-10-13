from types import SimpleNamespace

import pytest

from appworld_ga.eval_refill import generate_eval_refill


class Progress:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self): pass
    def set_postfix(self, **kwargs): pass


@pytest.mark.parametrize("count,samples", [(0, 1), (7, 3), (129, 1), (301, 3)])
def test_refill_before_slow_tail_without_missing_or_duplicate_prompts(count, samples):
    dispatched, finished, sizes = [], [], []
    kwargs = dict(n_samples_per_prompt=samples, temperature=0.01, max_new_tokens=4096)
    # Non-unit and partial dataloader batches must not lose any prompts.
    batches = [list(range(i, min(i + 7, count))) for i in range(0, count, 7)]
    generator = SimpleNamespace(
        args=SimpleNamespace(rollout=SimpleNamespace(n_samples_per_prompt=16)),
        eval_dataloader=[(None, b, b, [None]*len(b), [None]*len(b)) for b in batches],
    )
    def dispatch(prompts, labels, **actual):
        assert prompts == labels
        assert {k: actual[k] for k in kwargs} == kwargs
        dispatched.extend(prompts)
        return prompts
    def process(ref, dynamic_filtering, drops, **actual):
        assert dynamic_filtering is False and actual == kwargs
        finished.append(ref)
        # Existing per-response drops remain drops, not replacement prompts.
        if ref == 5:
            drops['existing_drop'] += 1
            return []
        return [(ref, i) for i in range(samples)]
    timed_out = False
    def wait(pending, **actual):
        nonlocal timed_out
        sizes.append(len(pending))
        assert len(pending) <= max(1, 128 // samples)
        if not timed_out:
            timed_out = True
            return [], pending
        # Hold prompt 0 until everything else completes.
        ref = next((r for r in pending if r != 0), 0)
        if ref == 0:
            assert len(dispatched) == count
        return [ref], [r for r in pending if r != ref]
    generator._dispatch_to_agent_runners = dispatch
    generator._filter_group = process
    result = generate_eval_refill(generator, ray=SimpleNamespace(wait=wait),
        logger=SimpleNamespace(info=lambda *args: None), progress=Progress, **kwargs)
    assert dispatched == list(range(count))
    assert sorted(finished) == list(range(count))
    assert result == [(i, s) for i in range(count) if i != 5 for s in range(samples)]
    if count > 128 // samples:
        assert sizes[1:count - 128 // samples + 1] == [128 // samples] * (count - 128 // samples)


def test_dispatch_failure_is_not_silently_skipped():
    def fail(*args, **kwargs):
        raise RuntimeError('dispatch failed')
    generator = SimpleNamespace(
        args=SimpleNamespace(rollout=SimpleNamespace(n_samples_per_prompt=1)),
        eval_dataloader=[(None, ['p'], ['l'], [None], [None])],
        _dispatch_to_agent_runners=fail,
    )
    with pytest.raises(RuntimeError, match='dispatch failed'):
        generate_eval_refill(generator, ray=None, logger=SimpleNamespace(info=lambda *a: None), progress=Progress)
