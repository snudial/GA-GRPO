"""Shared fixtures/builders for the agent-OPSD unit tests.

Everything here is synthetic. No checkpoint, no environment, no GPU
(plan section 16.0).
"""

from __future__ import annotations

import os
import sys

import pytest
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Test the checkout selected by the active protocol profile. The legacy fallback
# keeps direct strict-V3 test invocations working without config/bfcl_protocol.sh.
_bfcl_source = os.environ.get("BFCL_SOURCE_ROOT")
if not _bfcl_source and os.environ.get("BFCL_REPO_PATH"):
    _bfcl_source = str(Path(os.environ["BFCL_REPO_PATH"]) / "berkeley-function-call-leaderboard")
if _bfcl_source and (Path(_bfcl_source) / "bfcl_eval" / "__init__.py").is_file():
    if _bfcl_source in sys.path:
        sys.path.remove(_bfcl_source)
    sys.path.insert(0, _bfcl_source)

# The login node has 96 cores and torch will happily open an OpenMP team on every
# one of them for a two-layer model with a vocabulary of 32. The barrier cost then
# dwarfs the work: one backward through that model measured 2.75s with the default
# thread count and 0.038s pinned to a single thread -- a 70x difference that is
# entirely contention. Every test here is deliberately tiny, so single-threaded is
# both faster and more reproducible. Set before torch is imported anywhere.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:  # torch is optional for the pure-schema tests
    import torch as _torch

    _torch.set_num_threads(1)
except ImportError:  # pragma: no cover - depends on the venv under test
    pass

from bfcl_env.schemas import (  # noqa: E402
    RolloutRecord,
    RubricKind,
    RubricSpec,
    RubricState,
    RubricStateVector,
    StepRecord,
    ToolCall,
)

TOKENS_PER_ACTION = 4


def goal(rubric_id: str, hard: bool = True, **kwargs) -> RubricSpec:
    """A positive goal rubric (AppWorld ``no_op_fail``): starts PENDING."""
    return RubricSpec(
        rubric_id=rubric_id,
        kind=RubricKind.GOAL,
        hard=hard,
        source=kwargs.pop("source", "test"),
        initial_state=RubricState.PENDING,
        **kwargs,
    )


def safety(rubric_id: str, hard: bool = True, **kwargs) -> RubricSpec:
    """A safety rubric (AppWorld ``no_op_pass``): starts SATISFIED."""
    return RubricSpec(
        rubric_id=rubric_id,
        kind=RubricKind.SAFETY,
        hard=hard,
        source=kwargs.pop("source", "test"),
        initial_state=RubricState.SATISFIED,
        **kwargs,
    )


def make_rollout(
    num_steps: int,
    benchmark: str = "synthetic",
    *,
    rollout_id: str = "r0",
    task_id: str = "t0",
    tools: Sequence[str] | None = None,
    turns: Sequence[int] | None = None,
    tokens_per_action: int = TOKENS_PER_ACTION,
    terminated: bool = True,
    truncated: bool = False,
    abort_reason: str | None = None,
) -> RolloutRecord:
    """A rollout whose step ``i`` owns tokens ``[i*k, (i+1)*k)``."""
    steps = []
    for i in range(num_steps):
        tool = tools[i] if tools is not None else f"tool_{i}"
        turn = turns[i] if turns is not None else i
        steps.append(
            StepRecord(
                step=i,
                turn=turn,
                assistant_text=f"<call>{tool}</call>",
                parsed_calls=[ToolCall(tool=tool, arguments={})],
                raw_observation="ok",
                tool_execution_ok=True,
                assistant_token_span=(i * tokens_per_action, (i + 1) * tokens_per_action),
            )
        )
    return RolloutRecord(
        benchmark=benchmark,
        task_id=task_id,
        rollout_id=rollout_id,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        abort_reason=abort_reason,
    )


def make_vectors(
    rollout_id: str,
    specs: Sequence[RubricSpec],
    per_step: Mapping[int, Mapping[str, RubricState]],
    *,
    num_steps: int,
) -> list[RubricStateVector]:
    """Build explicit state vectors, carrying forward the last state per rubric.

    ``per_step`` only needs to list the rubrics that *changed* at a step.
    """
    current = {spec.rubric_id: spec.initial_state for spec in specs}
    vectors = [
        RubricStateVector(rollout_id=rollout_id, step=-1, rubrics=dict(current))
    ]
    for step in range(num_steps):
        for rubric_id, state in (per_step.get(step) or {}).items():
            current[rubric_id] = state
        vectors.append(
            RubricStateVector(rollout_id=rollout_id, step=step, rubrics=dict(current), turn=step)
        )
    return vectors


# --------------------------------------------------------------------------- #
# a real-tokenizer checkpoint, small enough for a login node
# --------------------------------------------------------------------------- #
# Moved here from test_model_loading so the generation backend can use it too:
# a version-dependent tokenizer or template mistake only surfaces against the
# real files, and every GPU failure so far has been of exactly that kind.

def _real_checkpoint() -> str | None:
    import os
    from pathlib import Path

    path = os.environ.get("QWEN3_4B_INSTRUCT_2507_PATH")
    if not path:
        return None
    return path if (Path(path) / "config.json").exists() else None


def _bfcl_available() -> bool:
    try:
        import bfcl_eval  # noqa: F401,PLC0415

        from bfcl_env.adapter import bfcl_data_root  # noqa: PLC0415

        bfcl_data_root()
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.fixture(scope="module")
def tiny_qwen_checkpoint(tmp_path_factory):
    """A tiny model sharing the *real* tokenizer and config class.

    Small enough to run on CPU, but loaded and tokenized exactly the way the GPU
    job does, so a version-dependent API mistake surfaces here.
    """
    source = _real_checkpoint()
    if source is None:
        pytest.skip("the local Qwen3-4B-Instruct-2507 checkpoint is not present")
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")

    config = transformers.AutoConfig.from_pretrained(source)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.max_position_embeddings = 4096

    import torch

    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config)
    path = tmp_path_factory.mktemp("tiny_qwen")
    model.save_pretrained(path)
    transformers.AutoTokenizer.from_pretrained(source).save_pretrained(path)
    return str(path)


def pytest_configure(config):
    """Exercise the same patched AppWorld runtime that preparation installs."""
    import importlib.util
    import shutil
    import tempfile

    if importlib.util.find_spec('molt') is None:
        return
    from scripts.prepare_appworld import patch_runtime
    from scripts.patch_appworld_actor import patch_actor

    runtime = Path(tempfile.mkdtemp(prefix='ga-grpo-test-runtime-'))
    for name in ('molt', 'tests', 'examples/python'):
        shutil.copytree(REPO_ROOT/'third_party/molt'/name, runtime/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patch_runtime(runtime)
    patch_actor(runtime)
    sys.path.insert(0, str(runtime))
    config._ga_runtime = runtime


def pytest_unconfigure(config):
    import shutil

    runtime = getattr(config, '_ga_runtime', None)
    if runtime is not None:
        sys.path.remove(str(runtime))
        shutil.rmtree(runtime)
