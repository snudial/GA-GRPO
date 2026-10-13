"""Fresh-clone planning must not require models, data preparation, or GPUs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args, env=None):
    return subprocess.run([sys.executable, str(ROOT/'run.py'), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_help_available_before_installation():
    result = invoke('--help')
    assert result.returncode == 0
    assert 'ga-grpo-call-bandit' in result.stdout


@pytest.mark.parametrize('environment', ['bfcl', 'appworld'])
def test_planning_does_not_create_outputs_or_require_model(tmp_path, environment):
    overrides = {**os.environ, 'OPSD_ROOT': str(ROOT), 'MODELS_DIR': str(tmp_path/'missing_models'),
                 'RUN_ROOT': str(tmp_path/'runs'), 'CACHE_ROOT': str(tmp_path/'cache')}
    result = invoke('--env', environment, '--model', 'Qwen3-4B-Instruct-2507',
                    '--method', 'ga-grpo-call-bandit', '--seed', '42', '--dry-run', env=overrides)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['method'] == 'ga-grpo-call-bandit'
    assert plan['model'] == str(tmp_path/'missing_models/Qwen3-4B-Instruct-2507')
    assert not list(tmp_path.iterdir())


def test_run_cannot_escape_output_root():
    result = invoke('--env', 'bfcl', '--model', 'Qwen3-4B-Instruct-2507',
                    '--method', 'ga-grpo', '--seed', '42', '--run', '../escape', '--dry-run')
    assert result.returncode == 2
    assert 'single directory name' in result.stderr
