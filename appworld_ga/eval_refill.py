"""Bounded, continuously refilled evaluation with fixed model weights.

Only scheduling changes: each prompt group is dispatched exactly once, with the
caller's sampling arguments and filtering. Training never calls this helper.
"""
from collections import defaultdict


def generate_eval_refill(generator, *, ray, logger, progress, capacity=128, **generate_kwargs):
    samples = generate_kwargs.get(
        "n_samples_per_prompt", generator.args.rollout.n_samples_per_prompt
    )
    # `capacity` episodes in flight (128 = all physical AppWorld environments of the
    # 2026-09 campaigns); as prompt groups of `samples` episodes each.
    capacity = max(1, capacity // samples)
    logger.info(
        "Eval continuous refill enabled: capacity=%s prompt groups, samples_per_prompt=%s",
        capacity, samples,
    )

    def prompts():
        for _, batch_prompts, labels, images, tools in generator.eval_dataloader:
            if not (len(batch_prompts) == len(labels) == len(images) == len(tools)):
                raise ValueError("Mismatched evaluation dataloader columns")
            yield from zip(batch_prompts, labels, images, tools)

    remaining = iter(prompts())
    pending = {}
    completed = {}
    dispatched = 0
    exhausted = False
    drops = defaultdict(int)

    def refill():
        nonlocal dispatched, exhausted
        while not exhausted and len(pending) < capacity:
            try:
                prompt, label, image, tool = next(remaining)
            except StopIteration:
                exhausted = True
                break
            refs = generator._dispatch_to_agent_runners(
                [prompt], [label], images=[image], tools=[tool], **generate_kwargs
            )
            if len(refs) != 1:
                raise RuntimeError("Expected one evaluation reference per prompt group")
            pending[refs[0]] = dispatched
            dispatched += 1

    with progress(desc="Generate eval (continuous refill)") as bar:
        refill()
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=10.0)
            for ref in ready:
                index = pending.pop(ref)
                completed[index] = generator._filter_group(
                    ref, False, drops, **generate_kwargs
                )
                refill()
                bar.update()
                bar.set_postfix(dispatched=dispatched, pending=len(pending))
    if drops:
        logger.info("Eval rollout drops: %s", dict(drops))
    logger.info("Eval continuous refill completed: prompt_groups=%s", dispatched)
    return [sample for index in range(dispatched) for sample in completed[index]]
