# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from molt.cli.train_rl_ray import _ray_runtime_env_vars
from molt.trainer.rl_trainer import _checkpoint_episode


def test_ray_runtime_env_forwards_wandb_settings(monkeypatch):
    expected = {
        "WANDB_API_KEY": "test-api-key",
        "WANDB_ENTITY": "test-entity",
        "WANDB_MODE": "offline",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    env_vars = _ray_runtime_env_vars()

    assert {name: env_vars[name] for name in expected} == expected


def test_ray_runtime_env_omits_empty_optional_settings(monkeypatch):
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)

    env_vars = _ray_runtime_env_vars()

    assert not {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"} & env_vars.keys()


def test_checkpoint_episode_advances_after_full_episode():
    assert _checkpoint_episode(29, is_exhausted=True) == 30
    assert _checkpoint_episode(29, is_exhausted=False) == 29


def test_ray_runtime_env_forwards_gpu_optimization_rollback(monkeypatch):
    for value in ['0', '1']:
        for name in ['EXP2_HYBRID', 'MOLT_GLOO_REUSE_BUFFER', 'MOLT_REUSE_ACTOR_CACHE']:
            monkeypatch.setenv(name, value)
        monkeypatch.setenv('PYTHONPATH', '/worktree/third_party/molt:/worktree')
        env = _ray_runtime_env_vars()
        assert env['PYTHONPATH'] == '/worktree/third_party/molt:/worktree'
        assert env['MOLT_GLOO_REUSE_BUFFER'] == value
        assert env['MOLT_REUSE_ACTOR_CACHE'] == value
        assert env['EXP2_HYBRID'] == value
