"""Run manifest: what code, what model, what data produced a result.

Plan section 15 requires ``manifest.json`` to carry the OPSD, BFCL and AppWorld
commit SHAs plus the model checksum and the resolved config. The absolute model
path is machine-specific, so it is recorded here while result tables keep showing
the canonical name ``Qwen/Qwen3-4B-Instruct-2507`` (plan section 14).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["git_sha", "file_sha256", "model_checksum", "RunManifest"]

CANONICAL_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

_BENCHMARK_ENV = {
    "appworld": "APPWORLD_REPO_PATH",
    "bfcl": "BFCL_REPO_PATH",
}


def git_sha(path: str | os.PathLike[str]) -> str | None:
    """Current commit of the repository containing ``path``, or ``None``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def file_sha256(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def model_checksum(model_path: str | os.PathLike[str]) -> str | None:
    """Cheap, stable identity for a local checkpoint.

    Hashes ``config.json`` plus the *names and sizes* of the weight shards rather
    than their bytes: a multi-GB re-read on every rollout batch is not affordable,
    and this still detects a swapped or truncated checkpoint.
    """
    root = Path(model_path)
    config = root / "config.json"
    if not config.exists():
        return None
    digest = hashlib.sha256()
    digest.update(config.read_bytes())
    for shard in sorted(root.glob("*.safetensors")):
        digest.update(shard.name.encode())
        digest.update(str(shard.stat().st_size).encode())
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        candidate = root / extra
        if candidate.exists():
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    """Versions via package metadata, never by importing.

    Importing torch/vllm here would add tens of seconds to every offline command
    that writes a manifest, for information the distribution metadata already has.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "vllm",
                 "datasets", "numpy", "pandas", "appworld", "bfcl_eval"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


@dataclass
class RunManifest:
    """Everything needed to reproduce or audit one output directory."""

    run: str
    config: dict[str, Any] = field(default_factory=dict)
    model_name: str = CANONICAL_MODEL_NAME
    model_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self, repo_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
        commits: dict[str, str | None] = {"opsd": git_sha(root)}
        for benchmark, env_var in _BENCHMARK_ENV.items():
            path = os.environ.get(env_var)
            commits[benchmark] = git_sha(path) if path else None

        payload: dict[str, Any] = {
            "run": self.run,
            "commits": commits,
            "packages": _package_versions(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "model": {
                "canonical_name": self.model_name,
                "resolved_path": self.model_path,
                "checksum": model_checksum(self.model_path) if self.model_path else None,
            },
            "config": self.config,
        }
        payload.update(self.extra)
        return payload

    def write(self, path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
        payload = self.build(**kwargs)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload
