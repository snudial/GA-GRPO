"""CPU-only replay of training guides; never reads held-out answer keys."""
import argparse
import json
import os
from pathlib import Path
import time
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--token-lengths-only', action='store_true')
    parser.add_argument('--model', help='Measure only this installed AppWorld model')
    args = parser.parse_args()
    setup = args.setup.resolve()
    config = json.loads((setup/'setup.json').read_text())
    context_limit = config.get('context_max_length', 10000)
    root = setup/'guide_validation'
    root.mkdir(exist_ok=True)
    if args.token_lengths_only:
        from transformers import AutoTokenizer
        sys.path.insert(0, str(config.get('agent_source', Path(__file__).resolve().parents[1])))
        from appworld_ga.prompt import initial_messages
        rows = [json.loads(line) for line in (root/'contexts.jsonl').read_text().splitlines()]
        for row in rows:  # the training prompt, rendered as the agent renders it
            prompt = initial_messages(row.pop('task'), config.get('prompt_profile', 'legacy'))
            row.update(initial=len(prompt), messages=prompt + row['messages'])
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ga_grpo.config import path
        models = path('MODELS_DIR')
        report = {}
        sizes = {'Qwen3-4B-Instruct-2507': '4b', 'Qwen3-8B-NonThinking-AppendOnly': '8b',
                 'Qwen3-1.7B-NonThinking-AppendOnly': '1p7b', 'Qwen3-0.6B-NonThinking-AppendOnly': '0p6b'}
        if args.model and args.model not in sizes:
            parser.error(f'Unsupported AppWorld model: {args.model}')
        for name in [args.model] if args.model else sizes:
            # Only selected models need to be installed for a fresh setup.
            if not (models/name/'tokenizer_config.json').exists():
                raise SystemExit(f'{models/name} missing: python3 scripts/fetch_models.py {name} '
                                 '(the setup seal measures both AppWorld models; weights are hardlinked)')
            tokenizer = AutoTokenizer.from_pretrained(models/name, local_files_only=True)
            lengths = [dict(task_id=row['task_id'], h=row['h'], tokens=len(tokenizer.encode(
                tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False))) for row in rows]
            report[name] = dict(max_tokens=max(row['tokens'] for row in lengths),
                                over_context=[row for row in lengths if row['tokens'] >= context_limit], tasks=lengths)
            caps = {}
            prefix_lengths = {}
            for row in rows:
                for h in range(row['h'], -1, -1):
                    rendered = tokenizer.apply_chat_template(row['messages'][:row['initial']+2*h],
                                                             tokenize=False, add_generation_prompt=True)
                    prefix_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
                    # Match BFCL: 4096 is a generation ceiling, not reserved input space.
                    if prefix_tokens < context_limit:
                        caps[row['task_id']] = h
                        prefix_lengths[row['task_id']] = prefix_tokens
                        break
                else:
                    raise ValueError(f"Task prompt alone is too long: {row['task_id']}")
            training = [json.loads(line) for line in (setup/'datasets/train.jsonl').read_text().splitlines()]
            for row in training:
                label = json.loads(row['label'])
                # Contract-authorized caps (prefixes the student cannot act after) bound the measurement.
                cap = caps.get(label['task_id'], 0)
                label['h_max'] = min(cap, config.get('authorized_h_caps', {}).get(label['task_id'], cap))
                row['label'] = json.dumps(label)
            size = sizes[name]
            (setup/'datasets'/f'train_{size}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in training))
            report[name]['handoff_caps'] = caps
            report[name]['handoff_prefix_tokens'] = prefix_lengths
            report[name]['generation_budget_rule'] = f'min(4096, {context_limit} - current_context_tokens)'
            report[name]['capped_tasks'] = sum(caps[row['task_id']] < row['h'] for row in rows)
            if config.get('full_prefix_required') and report[name]['capped_tasks']:
                raise ValueError(f'{name}: full H-1 guide does not fit; do not silently shorten JUMP hints')
            print(name, report[name]['max_tokens'], 'tokens; over context:', len(report[name]['over_context']))
        (root/'token_lengths.json').write_text(json.dumps(report, indent=2))
        return
    data = Path(json.loads((setup/'setup.json').read_text())['data'])
    if not (root/'data').exists():
        (root/'data').symlink_to(data, target_is_directory=True)
    os.environ.update(APPWORLD_ROOT=str(root), CUDA_VISIBLE_DEVICES='')
    os.chdir(root)
    from appworld import AppWorld
    sys.path.insert(0, str(config.get('agent_source', Path(__file__).resolve().parents[1])))
    from appworld_ga.execution_compat import install
    install()  # the training servers' execution, so replayed observations match theirs
    limit = config.get('observation_max_chars', 3000)
    rows = [json.loads(line) for line in (setup/'datasets/train.jsonl').read_text().splitlines()]
    labels = [json.loads(row['label']) for row in rows]
    labels = [label for label in labels if label.get('guide_path')]
    if args.limit:
        labels = labels[:args.limit]
    results = []
    start = time.monotonic()
    with (root/'contexts.jsonl').open('w') as contexts:
        for label in labels:
            turns = json.loads(Path(label['guide_path']).read_text())['turns']
            world = AppWorld(task_id=label['task_id'], experiment_name='train_guide_audit',
                             max_interactions=200, random_seed=42)
            try:
                # Public task fields only; the token pass renders them with the training prompt.
                task = dict(instruction=world.task.instruction, supervisor=dict(world.task.supervisor),
                            datetime=str(world.task.datetime), app_descriptions=[
                                dict(name=n, description=d) for n, d in world.task.app_descriptions.items()])
                messages = []
                failures = 0
                for index, turn in enumerate(turns):
                    if index == len(turns) - 1:
                        contexts.write(json.dumps(dict(task_id=label['task_id'], h=index, task=task,
                                                       messages=messages))+'\n')
                    output = world.execute(turn['code'])
                    failures += int(output.startswith('Execution failed.'))
                    if limit and len(output) > limit:
                        output = output[:limit] + f'\n...[truncated {len(output)-limit} chars]'
                    messages.extend([{'role': 'assistant', 'content': f"```python\n{turn['code']}\n```"},
                                     {'role': 'user', 'content': output}])
                result = dict(task_id=label['task_id'], H=len(turns),
                              success=bool(world.evaluate().to_dict()['success']), execution_errors=failures)
                results.append(result)
                print(json.dumps(result), flush=True)
            finally:
                world.close()
    report = dict(training_guides=len(results), passed=sum(row['success'] for row in results),
                  gpu_used=False, elapsed_seconds=time.monotonic()-start, results=results)
    (root/'report.json').write_text(json.dumps(report, indent=2))
    if report['passed'] != len(results):
        raise SystemExit('Some guides did not reproduce success; resolve before JUMP training')


if __name__ == '__main__':
    main()
