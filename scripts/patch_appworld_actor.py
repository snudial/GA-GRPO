"""Flush partial actor accumulation windows in the isolated AppWorld runtime.

Full windows retain the original loss and AdamW update schedule. Tail windows
use the existing global action-token denominator and step on their last
microbatch. No vendored Molt file is modified.
"""


def patch_actor(runtime):
    path = runtime/'molt/trainer/workers/policy_actor.py'
    text = path.read_text()
    start = text.index('            elif not dynamic:\n                # Only run complete accumulation windows;')
    end = text.index('            # slime global token-mean:', start)
    text = text[:start] + text[end:]
    edits = [
        ('        status_list = []\n        status_mean = {}',
         '        status_list = []\n        status_mean = {}\n        optimizer_steps = 0'),
        ('                    window_end = len(window) == accum_steps',
         '                    window_end = len(window) == accum_steps or step + 1 == max_steps'),
        ('                window = []\n            assert not window',
         '                _ledger_update(self, window)\n'
         '                optimizer_steps += 1\n                window = []\n            assert not window'),
        # Above the @ray.remote decorator, which must stay on the actor class.
        ('@ray.remote(num_gpus=1)\nclass PolicyModelActor(',
         'def _ledger_update(actor, window):\n'
         '    """AppWorld training ledger: the rollouts each AdamW update consumed on this rank."""\n'
         '    root = os.environ.get("APPWORLD_LEDGER_DIR")\n'
         '    if not root:\n'
         '        return\n'
         '    import json as _json\n'
         '    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0\n'
         '    record = {"segment_id": os.environ.get("APPWORLD_SEGMENT_ID"), "time": time.time(), "rank": rank,\n'
         '              "adam_step": int(actor.actor_scheduler.last_epoch),\n'
         '              "rollouts": [r for e in window for r in (getattr(e, "rollout_ids", None) or [])],\n'
         '              "groups": [g for e in window for g in (getattr(e, "group_ids", None) or [])],\n'
         '              "microbatch_action_tokens": [int(e.action_mask.sum()) for e in window]}\n'
         '    os.makedirs(root, exist_ok=True)\n'
         '    with open(os.path.join(root, f"updates.rank{rank}.{os.getpid()}.jsonl"), "a") as stream:\n'
         '        stream.write(_json.dumps(record) + "\\n")\n'
         '\n\n'
         '@ray.remote(num_gpus=1)\nclass PolicyModelActor('),
        ('        status_mean.update(\n            self.strategy.compute_perf_metrics',
         '        if len(dataloader) and not optimizer_steps:\n'
         '            raise RuntimeError("Nonempty actor batch performed no optimizer update")\n'
         '        status_mean["actor_optimizer_steps_this_cycle"] = optimizer_steps\n'
         '        status_mean["actor_optimizer_steps_total"] = self.actor_scheduler.last_epoch\n'
         '        status_mean["actor_samples_this_cycle"] = local_seq_count * self.strategy.dp_size\n'
         '        status_mean.update(\n            self.strategy.compute_perf_metrics'),
    ]
    for before, after in edits:
        if text.count(before) != 1:
            raise RuntimeError(f'Review AppWorld actor patch: {before[:100]}')
        text = text.replace(before, after)
    path.write_text(text)
