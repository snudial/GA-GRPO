"""Freeze BFCL settings and an isolated Molt runtime; prepare official splits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ga_grpo.config import path
from scripts.patch_appworld_actor import patch_actor
DEFAULT_SETUP = path('APPWORLD_SETUPS')/'appworld_bfcl_matched_v2'
DATA = path('APPWORLD_DATA')
# The BFCL 8B append-only run whose molt command every AppWorld arm mirrors.
REFERENCE = ROOT/'config/appworld_reference_manifest.json'


GENERATOR_LEDGER = {
    # module-level writer (AppWorld training ledger, appworld_ga/ledger.py)
    'class SamplesGenerator:\n':
        'def _ledger_write(kind, record):\n'
        '    """AppWorld training ledger: append-only rows, off the training path."""\n'
        '    root = os.environ.get("APPWORLD_LEDGER_DIR")\n'
        '    if not root:\n'
        '        return\n'
        '    import time as _time\n'
        '    os.makedirs(root, exist_ok=True)\n'
        '    record = {"segment_id": os.environ.get("APPWORLD_SEGMENT_ID"), "time": _time.time(), **record}\n'
        '    with open(os.path.join(root, f"{kind}.{os.getpid()}.jsonl"), "a") as stream:\n'
        '        stream.write(json.dumps(record, default=str) + "\\n")\n'
        '\n\n'
        'class SamplesGenerator:\n',
    # cumulative groups consumed when this segment started (the resumed checkpoint's counter)
    '        self._prompt_budget_remaining = maximum - consumed\n':
        '        self._prompt_budget_remaining = maximum - consumed\n'
        '        self._ledger_consumed = consumed\n',
    # creation-time counters ride in each training label (eval labels are left untouched)
    '            self._rr += 1\n            refs.append(':
        '            self._rr += 1\n'
        '            if os.environ.get("APPWORLD_LEDGER_DIR"):\n'
        '                label = self._ledger_dispatch(label)\n'
        '            refs.append(',
    '    def _dispatch_to_agent_runners(\n':
        '    def _ledger_dispatch(self, label):\n'
        '        data = json.loads(label) if isinstance(label, str) else dict(label)\n'
        '        if data.get("eval"):\n'
        '            return label\n'
        '        index = getattr(self, "_ledger_dispatched", 0)\n'
        '        self._ledger_dispatched = index + 1\n'
        '        consumed = getattr(self, "_ledger_consumed", None)\n'
        '        data["ledger_dispatch"] = {"index": index,\n'
        '                                   "groups_dispatched_before": None if consumed is None else consumed + index}\n'
        '        return json.dumps(data) if isinstance(label, str) else data\n'
        '\n'
        '    def _ledger_filter(self, group_samples, unusable, decision, reason, dynamic_filtering):\n'
        '        if not dynamic_filtering or not os.environ.get("APPWORLD_LEDGER_DIR"):\n'
        '            return\n'
        '        # Training only: the eval path calls _filter_group with dynamic_filtering=False, and every\n'
        '        # AppWorld training method filters (reference command: --algo.dynamic_filtering_enable).\n'
        '        first = group_samples[0] if group_samples else None\n'
        '        _ledger_write("filter", {\n'
        '            "group_id": first.group_ids[0] if first is not None and getattr(first, "group_ids", None) else None,\n'
        '            "decision": decision, "reason": reason,\n'
        '            "usable_rollouts": {s.rollout_ids[0]: (s.scores[0].item() if s.scores is not None else None)\n'
        '                                for s in group_samples if getattr(s, "rollout_ids", None)},\n'
        '            "unusable_rollout_reasons": unusable})\n'
        '\n'
        '    def _dispatch_to_agent_runners(\n',
    # the per-group keep/drop decision, as the trainer makes it
    '        group_samples: List[Experience] = []\n        for experience, drop_reason in ray.get(finished_rollout):':
        '        group_samples: List[Experience] = []\n        _unusable = []\n'
        '        for experience, drop_reason in ray.get(finished_rollout):',
    '            elif drop_reason is not None:\n                drop_counts[drop_reason] += 1\n':
        '            elif drop_reason is not None:\n                drop_counts[drop_reason] += 1\n'
        '                _unusable.append(drop_reason)\n',
    '                drop_counts["incomplete_group"] += len(group_samples)\n                return []\n':
        '                drop_counts["incomplete_group"] += len(group_samples)\n'
        '                self._ledger_filter(group_samples, _unusable, "dropped", "incomplete_group", dynamic_filtering)\n'
        '                return []\n',
    '                drop_counts["dynamic_filter"] += len(group_samples)\n':
        '                drop_counts["dynamic_filter"] += len(group_samples)\n'
        '                self._ledger_filter(group_samples, _unusable, "dropped", "uniform_reward", dynamic_filtering)\n',
    '                return []\n        return group_samples\n':
        '                return []\n'
        '        self._ledger_filter(group_samples, _unusable, "accepted" if group_samples else "dropped",\n'
        '                            None if group_samples else "no_usable_rollouts", dynamic_filtering)\n'
        '        return group_samples\n',
    # accepted samples the generator drops at an epoch tail never reach an update
    '                batch_samples = []\n        return batch_samples, rollout_metrics':
        '                _ledger_write("batch", {"event": "dropped", "reason": "epoch_tail_below_dp_size",\n'
        '                                        "rollout_ids": [s.rollout_ids[0] for s in batch_samples\n'
        '                                                        if getattr(s, "rollout_ids", None)]})\n'
        '                batch_samples = []\n        return batch_samples, rollout_metrics',
}


def _patch_generator_ledger(text):
    for before, after in GENERATOR_LEDGER.items():
        if text.count(before) != 1:
            raise RuntimeError(f'Upstream changed; review the generator ledger patch: {before[:80]}')
        text = text.replace(before, after)
    return text


def patch_runtime(runtime):
    replacements = {
        'molt/trainer/rollout/router.py': [
            ('        group_id = uuid4().hex\n        tasks = [',
             '        group_id = uuid4().hex\n'
             '        if hasattr(self._runner, "prepare_group"):\n'
             '            label = self._runner.prepare_group(label, group_id, n_samples)\n        tasks = ['),
            ('        for r in await asyncio.gather(*tasks, return_exceptions=True):  # a failed rollout must not sink the group',
             '        group_results = await asyncio.gather(*tasks, return_exceptions=True)\n'
             '        # Rollout ids first, so the training ledger rows (finish_group) carry the ids the trainer uses.\n'
             '        rollout_ids = [uuid4().hex for _ in group_results]\n'
             '        if hasattr(self._runner, "finish_group"):\n'
             '            self._runner.finish_group(label, group_results, rollout_ids)\n'
             '        for r, rollout_id in zip(group_results, rollout_ids):'),
            ('            rollout_id = uuid4().hex\n            for traj in r if isinstance(r, list) else [r]:',
             '            for traj in r if isinstance(r, list) else [r]:')],
        'molt/trainer/workers/policy_actor.py': [
            # Hand the moments' transient GPU blocks back before the engines wake (train_rl_ray resume_sleep).
            ('            self.checkpoint_states = states\n',
             '            self.checkpoint_states = states\n            torch.cuda.empty_cache()\n')],
        'molt/trainer/rl_trainer.py': [
            ('        maximum_prompts = len(self.prompts_dataloader) * self.args.train.num_episodes',
             '        maximum_prompts = min(len(self.prompts_dataloader) * self.args.train.num_episodes,\n'
             '                              int(os.environ["APPWORLD_MAX_PROMPT_GROUPS"]))'),
            ('                    checkpoint_episode = _checkpoint_episode(\n'
             '                        self.samples_generator.current_epoch() if fill_batch else ep,\n'
             '                        is_exhausted=is_exhausted,\n                    )',
             '                    checkpoint_episode = total_consumed_prompts // dataset_length'),
            ('                        "episode": _checkpoint_episode(\n'
             '                            self.samples_generator.current_epoch() if fill_batch else ep,\n'
             '                            is_exhausted=is_exhausted,\n                        ),',
             '                        "episode": total_consumed_prompts // dataset_length,'),
            ('        experiences = balance_experiences(rollout_samples, self.args)\n',
             '        experiences = balance_experiences(rollout_samples, self.args)\n'
             '        _ledger = os.environ.get("APPWORLD_LEDGER_DIR")\n'
             '        if _ledger:  # AppWorld training ledger (appworld_ga/ledger.py); bookkeeping only\n'
             '            import json as _json\n'
             '            _ids = lambda xs: [r for e in xs for r in (getattr(e, "rollout_ids", None) or [])]\n'
             '            _kept = set(map(id, experiences))\n'
             '            os.makedirs(_ledger, exist_ok=True)\n'
             '            with open(os.path.join(_ledger, f"batch.{os.getpid()}.jsonl"), "a") as _stream:\n'
             '                _stream.write(_json.dumps({"segment_id": os.environ.get("APPWORLD_SEGMENT_ID"),\n'
             '                    "time": time.time(), "event": "cycle", "global_step_before": global_step,\n'
             '                    "trained_rollouts": _ids(experiences),\n'
             '                    "dropped_rollouts": _ids([e for e in rollout_samples if id(e) not in _kept]),\n'
             '                    "dropped_reason": "dp_balance_remainder"}) + "\\n")\n'),
            ('            if budget_spent:\n                break',
             '            if budget_spent or total_consumed_prompts >= maximum_prompts:\n                break'),
            # Contract checkpoint_policy: a rolling save waits until the rollout pool is drained.
            ('        if force or global_step % self.args.ckpt.save_steps == 0:\n            tag = f"global_step{global_step}"',
             '        save_due = force or global_step % self.args.ckpt.save_steps == 0 or getattr(self, "_deferred_save", False)\n'
             '        if save_due:\n'
             '            generator = client_states.get("rollout_generator_state_dict", {})\n'
             '            if generator.get("pending_rollouts", 0) or generator.get("pending_finished_samples", 0):\n'
             '                self._deferred_save = True\n'
             '                logger.info(f"Deferring rolling checkpoint at step {global_step}: rollout pool is not drained")\n'
             '                return\n'
             '            self._deferred_save = False\n'
             '            tag = f"global_step{global_step}"')],
        'molt/cli/train_rl_ray.py': [
            # Resume with Adam on CPU (8B on H100): load_ckpt builds the Adam moments on the GPU before
            # paging them to CPU, which does not fit beside awake engines (OOM at the first 8B resume).
            # The engines sleep while the actor loads. Level 2, as between training cycles: level 1 backs
            # the weights up in pinned host memory that PyTorch's host allocator then keeps (~15 GB per
            # engine), which put the 8B jobs over their RAM at the first step after a resume. A resumed
            # checkpoint always re-broadcasts its weights before the first rollout (RLTrainer.fit).
            ('    # init actor/reference models\n    refs = []\n',
             '    resume_sleep = bool(hybrid_pg is not None and vllm_engines and args.ckpt.load_enable\n'
             '                        and os.path.isfile(os.path.join(args.ckpt.path, "_actor", "latest")))\n'
             '    if resume_sleep:\n'
             '        ray.get([engine.sleep.remote(2) for engine in vllm_engines])\n'
             '    # init actor/reference models\n    refs = []\n'),
            ('    ray.get(refs)\n\n    # train actor model\n',
             '    ray.get(refs)\n'
             '    if resume_sleep:\n'
             '        ray.get([engine.wake_up.remote() for engine in vllm_engines])\n'
             '\n    # train actor model\n'),
            ('    return env_vars',
             '    env_vars.update({k: v for k, v in os.environ.items()\n'
             '                     if k.startswith("APPWORLD_") or k == "EXP2_VERSION_FILE"})\n'
             '    return env_vars')],
    }
    for name, edits in replacements.items():
        path = runtime/name
        text = path.read_text()
        for before, after in edits:
            if text.count(before) != 1:
                raise RuntimeError(f'Upstream changed; review runtime patch: {name}: {before[:80]}')
            text = text.replace(before, after)
        path.write_text(text)
    path = runtime/'molt/trainer/rollout/samples_generator.py'
    text = path.read_text().replace('"BFCL_CURRICULUM"', '"APPWORLD_CURRICULUM"')
    text = _patch_generator_ledger(text)
    # APPWORLD_EVAL_REFILL=N: evaluation keeps N episodes in flight and refills one as each finishes
    # (appworld_ga/eval_refill.py). Whole-split batches put 417 Test-C episodes on one engine: its KV cache
    # stays 96-99% full and re-prefills ~100 prompt tokens per generated token, and episodes waiting for a server
    # lease hit the 1800 s limit (8B on one H100, 2026-10-08). Unset (training, default eval): unchanged.
    before = '        eval_batch_size = self.args.eval.batch_size or self.args.rollout.batch_size\n'
    if text.count(before) != 1:
        raise RuntimeError('Upstream changed; review the eval refill patch')
    text = text.replace(before, before +
        '        refill = int(os.environ.get("APPWORLD_EVAL_REFILL", "0"))\n'
        '        if refill:\n'
        '            from appworld_ga.eval_refill import generate_eval_refill\n'
        '            return generate_eval_refill(self, ray=ray, logger=logger, progress=lambda **kw: tqdm(**kw),\n'
        '                                        capacity=refill, **generate_kwargs)\n')
    text = text.replace('if isinstance(sampler, DistributedSampler):', 'if hasattr(sampler, "set_epoch"):')
    before = '        curriculum_path = os.environ.get("EXP2_CURRICULUM_STATE")'
    insertion = ('        # A drained generation cap may be inside a data epoch. Restore its cursor.\n'
                 '        cap = int(os.environ.get("APPWORLD_MAX_PROMPT_GROUPS", "0"))\n'
                 '        if cap and getattr(self, "_prompt_budget_remaining", None) == 0:\n'
                 '            progress["round_exhausted"] = cap % len(self.prompts_dataloader) == 0\n')
    text = text.replace(before, insertion + before, 1)
    path.write_text(text)
    path = runtime/'tests/unit/test_samples_generator.py'
    path.write_text(path.read_text().replace('"BFCL_CURRICULUM"', '"APPWORLD_CURRICULUM"'))


def prepare(out):
    out = Path(out).resolve()
    if (out/'setup.json').exists():
        config = json.loads((out/'setup.json').read_text())
        for name, expected in config.get('artifact_sha256', {}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Validated AppWorld setup changed: {name}; prepare a new version')
        return config
    out.mkdir(parents=True, exist_ok=True)
    (out/'datasets').mkdir(exist_ok=True)
    (out/'guides').mkdir(exist_ok=True)
    splits = {s: (DATA/'datasets'/f'{s}.txt').read_text().split() for s in
              ['train', 'dev', 'test_normal', 'test_challenge']}
    assert len(set(sum(splits.values(), []))) == sum(map(len, splits.values()))
    guide_root = path('APPWORLD_GUIDES_DIR')
    sources = sorted(guide_root.glob('*.json'))
    if not sources:
        raise RuntimeError(f'No verified training guides in {guide_root}; set APPWORLD_GUIDES_DIR in config/env.sh')
    guides = {}
    for source in sources:
        guide = json.loads(source.read_text())
        task_id = guide['task_id']
        if task_id not in splits['train'] or guide.get('verifier_pass') is not True:
            raise ValueError(f'Guide must be verified and belong to the official training split: {source}')
        turns = [{'code': turn['code']} for turn in guide['turns'] if turn.get('code')]
        if not 1 <= len(turns) <= 30 or task_id in guides or source.stem != task_id:
            raise ValueError(f'Invalid guide identity or length: {source}')
        target = out/'guides'/source.name
        target.write_text(json.dumps({'task_id': task_id, 'turns': turns}, indent=2))
        guides[task_id] = (target, len(turns))
    # Generate task rows locally from the official splits and supplied guides.
    # Token validation later computes the selected model's eligible handoff caps.
    for split, ids in splits.items():
        rows = []
        for task_id in ids:
            label = {'task_id': task_id, 'eval': split != 'train'}
            if split == 'train':
                label['h'] = 0
                if task_id in guides:
                    target, length = guides[task_id]
                    label.update(guide_path=str(target), guide_H=length, h_max=length-1)
            rows.append(json.dumps({'input': task_id, 'label': json.dumps(label)})+'\n')
        (out/'datasets'/f'{split}.jsonl').write_text(''.join(rows))
    for size in ('4b', '8b', '1p7b', '0p6b'):
        shutil.copyfile(out/'datasets/train.jsonl', out/'datasets'/f'train_{size}.jsonl')
    reference = json.loads(REFERENCE.read_text())
    (out/'bfcl_reference.json').write_text(json.dumps(reference, indent=2))
    runtime = out/'runtime'
    runtime.mkdir(exist_ok=True)
    for name in ['molt', 'tests', 'examples/python']:
        shutil.copytree(ROOT/'third_party/molt'/name, runtime/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patch_runtime(runtime)
    # AppWorld runtime edits sealed into every campaign since v2: flush partial actor
    # windows (contract actor_partial_batch), and prune HF-only best saves as best.
    patch_actor(runtime)
    strategy = runtime/'molt/trainer/fsdp/strategy.py'
    before = 'self.checkpoint._prune_checkpoints(ckpt_path, keep_tag, max_num, max_mem, is_best=False)'
    if strategy.read_text().count(before) != 1:
        raise RuntimeError('Upstream changed; review the HF retention patch')
    strategy.write_text(strategy.read_text().replace(
        before, before.replace('is_best=False', 'is_best=keep_tag.startswith("best")')))
    contract = json.loads((ROOT/'contracts/appworld.json').read_text())
    config = dict(schema_version=1, campaign=out.name, contract=contract['name'], reference=str(REFERENCE),
                  data=str(DATA), agent_source=str(ROOT), splits={s: len(ids) for s, ids in splits.items()},
                  guide_tasks=len(list((out/'guides').iterdir())), appworld_package='0.1.3.post1',
                  appworld_data='0.1.0', total_generated_rollouts=contract['total_generated_rollouts'],
                  context_max_length=contract['max_len'], max_turns=contract['max_turns'],
                  prompt_profile=contract['prompt_profile'], curriculum=contract['curriculum'],
                  **contract['setup_fields'],
                  source_sha256={str(p.relative_to(ROOT/'third_party/molt')):
                                 hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/'third_party/molt/molt').rglob('*.py')})
    (out/'setup.json').write_text(json.dumps(config, indent=2))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_SETUP)
    parser.add_argument('--seal', action='store_true', help='Freeze verified source/data identities after CPU tests')
    args = parser.parse_args()
    config = prepare(args.out)
    if args.seal:
        out = args.out.resolve()
        audit = json.loads((out/'guide_validation/report.json').read_text())
        assert audit['passed'] == audit['training_guides'] == config['guide_tasks']
        agent_source = Path(config.get('agent_source', ROOT))
        files = [out/'bfcl_reference.json', out/'guide_validation/report.json',
                 out/'guide_validation/token_lengths.json', agent_source/'appworld_ga/onboarding.txt']
        for folder in ['datasets', 'guides', 'runtime/molt']:
            files.extend(p for p in (out/folder).rglob('*') if p.suffix in ('.py', '.json', '.jsonl'))
        files.extend((agent_source/'appworld_ga').glob('*.py'))
        files.extend(ROOT/'scripts'/name for name in ['prepare_appworld.py', 'run_appworld.py',
                                                     'validate_appworld_guides.py'])
        files.append(ROOT/'scripts/patch_appworld_actor.py')
        config['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (out/'setup.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))
