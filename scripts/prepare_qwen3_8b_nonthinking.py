"""Create a non-thinking tokenizer view; model tensors are unchanged hardlinks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def prepare(source: Path, target: Path, *, append_only: bool = False):
    revision = json.loads((source / "source_revision.json").read_text())
    original = json.loads((source / "tokenizer_config.json").read_text())
    template = "{%- set enable_thinking = false -%}\n" + original["chat_template"]
    if append_only:
        # Preserve sampled content verbatim; only the terminal ChatML EOS belongs
        # to the envelope. Never strip/rewrite a previous assistant's reasoning.
        start = template.index("        {%- set reasoning_content = '' %}")
        end = template.index("        {%- if message.tool_calls %}", start)
        template = template[:start] + (
            "        {%- if content.endswith('<|im_end|>') %}\n"
            "            {%- set content = content[:-10] %}\n"
            "        {%- endif %}\n"
            "        {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + content }}\n"
        ) + template[end:]
        original["jump_grpo_serialization"] = "qwen3_append_only_v1"
    expected = {**revision, "enable_thinking": False,
                "original_template_sha256": hashlib.sha256(original["chat_template"].encode()).hexdigest(),
                "template_sha256": hashlib.sha256(template.encode()).hexdigest()}
    if append_only:
        expected["serialization"] = "qwen3_append_only_v1"
    if (target / "nonthinking_config.json").exists():
        if json.loads((target / "nonthinking_config.json").read_text()) != expected:
            raise ValueError("Existing non-thinking model has a different source")
        return
    target.mkdir(exist_ok=False)
    for path in source.iterdir():
        if not path.is_file() or path.name in {"tokenizer_config.json", "chat_template.jinja"}:
            continue
        if path.suffix == ".safetensors":
            os.link(path, target / path.name)
        else:
            shutil.copy2(path, target / path.name)
    original["chat_template"] = template
    (target / "tokenizer_config.json").write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n")
    (target / "chat_template.jinja").write_text(template)
    (target / "nonthinking_config.json").write_text(json.dumps(expected, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ga_grpo.config import path
    root = path('MODELS_DIR')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--append-only', action='store_true')
    parser.add_argument('--source', type=Path, default=root / 'Qwen3-8B')
    parser.add_argument('--target', type=Path)
    args = parser.parse_args()
    target = args.target or root / ('Qwen3-8B-NonThinking-AppendOnly' if args.append_only else 'Qwen3-8B-NonThinking')
    prepare(args.source, target, append_only=args.append_only)
    print(target)
