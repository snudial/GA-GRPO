"""Consolidate a Molt/AutoModel DCP safetensors model dir into a bf16 HF checkpoint.

Molt saves each `state/_actor/<step>/model` as four FSDP shards in the DCP
HuggingFace-safetensors format (fp32 master weights, HF key names). The HF
export that training writes to `run/hf` is only produced for the final step,
so evaluating an intermediate step officially needs this offline merge.

Usage:
  python consolidate_dcp_to_hf.py --step-dir RUN/state/_actor/best_global_step50 \
      --reference-hf RUN/hf --out RUN/hf_step50
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", required=True)
    parser.add_argument("--reference-hf", required=True,
                        help="a finished HF export of the same run (config, tokenizer, index layout)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    model_dir = os.path.join(args.step_dir, "model")
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards under {model_dir}")

    ref_index = json.load(open(os.path.join(args.reference_hf, "model.safetensors.index.json")))
    ref_map = ref_index["weight_map"]
    # "model-00002-of-00003.safetensors" -> 2
    fqn_to_index = {k: int(v.split("-")[1]) for k, v in ref_map.items()}

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_fp32_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    consolidate_safetensors_files(model_dir, tmp, fqn_to_index, num_threads=args.threads)

    # Reference shapes for a hard equality check after the merge.
    ref_shapes = {}
    for fname in sorted(set(ref_map.values())):
        with safe_open(os.path.join(args.reference_hf, fname), framework="pt") as f:
            for k in f.keys():
                ref_shapes[k] = tuple(f.get_slice(k).get_shape())

    weight_map, total = {}, 0
    for path in sorted(glob.glob(os.path.join(tmp, "*.safetensors"))):
        tensors = load_file(path)
        out = {}
        for k, v in tensors.items():
            if k not in ref_shapes:
                raise SystemExit(f"unexpected tensor {k}")
            if tuple(v.shape) != ref_shapes[k]:
                raise SystemExit(f"shape mismatch {k}: {tuple(v.shape)} vs {ref_shapes[k]}")
            out[k] = v.to(torch.bfloat16).contiguous()
        name = os.path.basename(path)
        save_file(out, os.path.join(args.out, name), metadata={"format": "pt"})
        for k, v in out.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
        del tensors, out

    missing = sorted(set(ref_map) - set(weight_map))
    if missing:
        raise SystemExit(f"missing tensors after merge: {missing[:5]} (+{len(missing) - 5})")

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    for name in ("config.json", "generation_config.json", "chat_template.jinja",
                 "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(args.reference_hf, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, name))
    shutil.rmtree(tmp)
    print(json.dumps({"out": args.out, "tensors": len(weight_map), "bytes": total,
                      "files": sorted(set(weight_map.values()))}))


if __name__ == "__main__":
    main()
