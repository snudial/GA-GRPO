"""Single-rank memory split follows the policy's size (config/topology/single_h100.env)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from ga_grpo.model_profile import param_count, single_rank_defaults

ROOT = Path(__file__).resolve().parents[1]
QWEN3_0_6B = dict(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
                  num_key_value_heads=8, head_dim=128, vocab_size=151936, tie_word_embeddings=True,
                  torch_dtype="bfloat16")
QWEN3_4B = dict(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128, vocab_size=151936, tie_word_embeddings=True,
                torch_dtype="bfloat16")


def model_dir(tmp_path, config, tensors=None):
    d = tmp_path / "model"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    if tensors:   # a safetensors file is an 8-byte header length, the JSON header, then data
        header = json.dumps({name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
                             for name, shape in tensors.items()}).encode()
        (d / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    return d


def test_param_count_from_architecture(tmp_path):
    assert param_count(model_dir(tmp_path / "a", QWEN3_0_6B)) == pytest.approx(0.596e9, rel=0.01)
    assert param_count(model_dir(tmp_path / "b", QWEN3_4B)) == pytest.approx(4.02e9, rel=0.01)


def test_param_count_reads_safetensors_headers_and_skips_a_tied_lm_head(tmp_path):
    tensors = {"model.embed_tokens.weight": [151936, 1024], "lm_head.weight": [151936, 1024],
               "model.layers.0.mlp.up_proj.weight": [3072, 1024]}
    tied = model_dir(tmp_path / "tied", QWEN3_0_6B, tensors)
    assert param_count(tied) == 151936 * 1024 + 3072 * 1024
    untied = model_dir(tmp_path / "untied", dict(QWEN3_0_6B, tie_word_embeddings=False), tensors)
    assert param_count(untied) == 2 * 151936 * 1024 + 3072 * 1024


def test_split_small_model_keeps_adam_on_gpu_and_gives_vllm_most_of_the_card():
    assert single_rank_defaults(596_000_000, 79.6) == {"FSDP_OFFLOAD": "none", "VLLM_UTIL": "0.75",
                                                       "ACTOR_RESIDENT_GIB": "8.9"}


def test_split_1_7b_keeps_adam_on_gpu():
    assert single_rank_defaults(1_720_000_000, 93.6) == {"FSDP_OFFLOAD": "none", "VLLM_UTIL": "0.60",
                                                         "ACTOR_RESIDENT_GIB": "25.6"}


def test_split_4b_matches_the_measured_h100_settings():
    assert single_rank_defaults(4_020_000_000, 79.6) == {"FSDP_OFFLOAD": "optimizer", "VLLM_UTIL": "0.50",
                                                         "ACTOR_RESIDENT_GIB": "30.0"}


def test_split_scales_with_a_94gb_card():
    assert single_rank_defaults(596_000_000, 93.6)["VLLM_UTIL"] == "0.80"
    assert single_rank_defaults(4_020_000_000, 93.6)["VLLM_UTIL"] == "0.55"


def test_split_never_leaves_the_engine_below_the_floor():
    assert single_rank_defaults(8_200_000_000, 79.6)["VLLM_UTIL"] == "0.30"


def source_topology(tmp_path, model, extra=""):
    script = (f'set -euo pipefail; export SINGLE_GPU_GIB=79.6; {extra}{";" if extra else ""} '
              f'export MODEL="{model}"; export TOPOLOGY=single_h100; '
              f'source "{ROOT}/config/topology.sh"; echo "$FSDP_OFFLOAD $VLLM_UTIL $NUM_RUNNERS"')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path,
                          env={**os.environ, "PYTHONPATH": ""})


def test_topology_profile_applies_the_split_per_model(tmp_path):
    small = source_topology(tmp_path, model_dir(tmp_path / "small", QWEN3_0_6B))
    assert small.returncode == 0, small.stderr
    assert small.stdout.split() == ["none", "0.75", "6"]   # RAY_NUM_CPUS 12 -> 6 runners
    big = source_topology(tmp_path, model_dir(tmp_path / "big", QWEN3_4B))
    assert big.stdout.split() == ["optimizer", "0.50", "6"]


def test_topology_profile_assumes_80gb_where_no_gpu_is_visible(tmp_path):
    # A login node: nvidia-smi is missing or fails; the profile must not end a `set -e` caller.
    model = model_dir(tmp_path / "m", QWEN3_0_6B)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed'; exit 9\n")
    (tmp_path / "bin" / "nvidia-smi").chmod(0o755)
    out = source_topology(tmp_path, model, extra=f'unset SINGLE_GPU_GIB; export PATH="{tmp_path}/bin:$PATH"')
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[:2] == ["none", "0.75"]


def test_topology_profile_sizes_runners_from_the_ray_cpu_budget(tmp_path):
    model = model_dir(tmp_path / "m", QWEN3_0_6B)
    assert source_topology(tmp_path, model, extra="export RAY_NUM_CPUS=24").stdout.split()[2] == "16"
    assert source_topology(tmp_path, model, extra="export RAY_NUM_CPUS=4").stdout.split()[2] == "2"
    assert source_topology(tmp_path, model, extra="export RAY_NUM_CPUS=24 NUM_RUNNERS=3").stdout.split()[2] == "3"


def test_topology_profile_keeps_explicit_overrides_and_falls_back_without_a_model(tmp_path):
    pinned = source_topology(tmp_path, model_dir(tmp_path / "small", QWEN3_0_6B), extra="export VLLM_UTIL=0.40")
    assert pinned.stdout.split() == ["none", "0.40", "6"]
    bigger_card = source_topology(tmp_path, model_dir(tmp_path / "small2", QWEN3_0_6B), extra="export SINGLE_GPU_GIB=93.6")
    assert bigger_card.stdout.split() == ["none", "0.80", "6"]
    missing = source_topology(tmp_path, tmp_path / "nowhere")
    assert missing.returncode == 0, missing.stderr
    assert missing.stdout.split() == ["optimizer", "0.50", "6"]
    assert "no model profile" in missing.stderr
