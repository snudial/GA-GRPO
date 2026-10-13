#!/usr/bin/env python3
"""Write the immutable code/protocol identity for one training run."""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_names() -> list:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    molt = Path(os.environ.get("MOLT_DIR", root / "third_party" / "molt")).resolve()
    from verify_vendor import verify
    lock = verify(molt)
    base = lock['upstream_base_commit']
    expected_commit = lock['source_commit']
    payload = {
        "schema_version": 1,
        "ga_grpo_commit": git(root, "rev-parse", "HEAD"),
        "ga_grpo_dirty": bool(git(root, "status", "--porcelain", "--ignore-submodules=dirty")),
        "molt_base_commit": base,
        "molt_expected_commit": expected_commit,
        "molt_actual_commit": expected_commit,
        "molt_source_verified": True,
        "molt_lock_sha256": sha256(root/'third_party/molt.lock.json'),
        "bfcl_profile": os.environ.get("BFCL_PROFILE"),
        "bfcl_comparability": os.environ.get("BFCL_COMPARABILITY"),
        "bfcl_runtime_commit": os.environ.get("BFCL_RUNTIME_COMMIT"),
        "bfcl_history_mode": os.environ.get("BFCL_HISTORY_MODE"),
        "model": os.environ.get("MODEL"),
        "algo": os.environ.get("ALGO"),
        "curriculum": os.environ.get("BFCL_CURRICULUM"),
        "curriculum_init": os.environ.get("BFCL_CURRICULUM_INIT", "suffix"),
        "handoff_unit": os.environ.get("BFCL_HANDOFF_UNIT", "turn"),
        # window (fixed-step rule) | uniform | bandit (ga_grpo/h_selector.py)
        "h_selector": os.environ.get("BFCL_H_SELECTOR", "window"),
        # run.py method name and the OPSD switches, which change no Molt flag.
        "method": os.environ.get("COMPARISON_METHOD"),
        "opsd": os.environ.get("OPSD", "0"),
        "opsd_lambda": os.environ.get("EXP2_OPSD_LAMBDA"),
        "ema_beta": os.environ.get("EXP2_EMA_BETA"),
        "topology": os.environ.get("TOPOLOGY"),
        "topology_sha256": os.environ.get("TOPOLOGY_SHA256"),
        "gpu_optimizations": {name: os.environ.get(name, "0") for name in
                              ("MOLT_GLOO_REUSE_BUFFER", "MOLT_REUSE_ACTOR_CACHE")},
        "curriculum_window": os.environ.get("EXP2_CURRICULUM_WINDOW"),
        "curriculum_promote": os.environ.get("EXP2_CURRICULUM_PROMOTE", "0.7"),
        "curriculum_demote": os.environ.get("EXP2_CURRICULUM_DEMOTE", "0.1"),
        "agent_sha256": sha256(root / "ga_grpo" / "bfcl_agent.py"),
        "curriculum_sha256": sha256(root / "ga_grpo" / "curriculum.py"),
        "guide_sha256": sha256(root / "ga_grpo" / "bfcl_guide.py"),
        # vLLM memory/scheduler settings (topology profile or explicit); the
        # resulting flags are also in `command`.
        "gpus": gpu_names(),
        "hybrid": os.environ.get("HYBRID"),
        "vllm_gpu_memory_utilization": os.environ.get("VLLM_UTIL"),
        "vllm_max_num_batched_tokens": os.environ.get("VLLM_MAX_BATCHED_TOKENS") or None,
        "vllm_prefix_caching": os.environ.get("VLLM_PREFIX_CACHING", "0"),
        # Adam placement (topology profile by model size, or explicit).
        "fsdp_offload": os.environ.get("FSDP_OFFLOAD"),
        "rollout_runners": os.environ.get("NUM_RUNNERS"),
        # 0 = one sample per training forward; N = packed up to N tokens.
        "train_max_tokens": os.environ.get("TRAIN_MAX_TOKENS"),
        "command": args.command,
    }
    if os.environ.get("TRAIN_DATA"):
        payload["train_data_sha256"] = sha256(Path(os.environ["TRAIN_DATA"]))
    run.mkdir(parents=True, exist_ok=True)
    # Preserve each launch identity. Repeated launches used to overwrite the
    # only manifest, hiding which code produced earlier checkpoints.
    history = run / "manifests"
    history.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if (run / "manifest.json").exists() and not (history / "initial.json").exists():
        (history / "initial.json").write_text((run / "manifest.json").read_text())
    (history / f"{stamp}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp = run / ".manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(run / "manifest.json")


if __name__ == "__main__":
    main()
