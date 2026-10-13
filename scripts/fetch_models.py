"""Download pinned models (config/models.json) into MODELS_DIR; derive tokenizer views.

    python scripts/fetch_models.py Qwen3-4B-Instruct-2507 [NAME ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huggingface_hub import snapshot_download  # noqa: E402
from ga_grpo.config import path  # noqa: E402
from scripts.prepare_qwen3_8b_nonthinking import prepare  # noqa: E402

MODELS = json.loads((ROOT/'config/models.json').read_text())


def fetch(name):
    spec, target = MODELS[name], path('MODELS_DIR')/name
    if 'derive_from' in spec:
        fetch(spec['derive_from'])
        prepare(path('MODELS_DIR')/spec['derive_from'], target, append_only=spec['append_only'])
        return
    pin = {'repo_id': spec['repo_id'], 'revision': spec['revision']}
    stamp = target/'source_revision.json'
    if stamp.exists() and json.loads(stamp.read_text()) != pin:
        raise RuntimeError(f'{target} holds {stamp.read_text()}, not the pinned {pin}')
    snapshot_download(spec['repo_id'], revision=spec['revision'], local_dir=target)
    stamp.write_text(json.dumps(pin, indent=2) + '\n')


if __name__ == '__main__':
    for name in sys.argv[1:] or ['Qwen3-4B-Instruct-2507']:
        fetch(name)
        print(path('MODELS_DIR')/name)
