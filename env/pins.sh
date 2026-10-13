# Versions and content hashes reproduced by scripts/bootstrap.sh.

PYTHON_VERSION=3.12.3
UV_VERSION=0.9.18

# torch 2.13.0 + CUDA 13.0 is what vLLM 0.27/0.28 pin; the driver must speak CUDA 13.
TORCH_SPEC="torch==2.13.0+cu130"
TORCH_INDEX=https://download.pytorch.org/whl/cu130
MIN_DRIVER_MAJOR=580
# The TE torch extension is a source build against the installed torch, compiled
# with the CUDA 13 nvcc that ships as a pip wheel (nvidia-cuda-nvcc in env/molt.txt).
TE_TORCH_SPEC="transformer-engine-torch==2.12.0"

# BFCL checkouts (config/bfcl_protocol.sh pins the same commits).
GORILLA_URL=https://github.com/ShishirPatil/gorilla.git
GORILLA_T3RL_COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8   # author_lineage
GORILLA_V3_COMMIT=ea13468e4423454d0c213704fb87cf7cb3990433     # crest_compat (optional)

# Model: the exact Hugging Face revision of every run so far, with the shard hashes.
MODEL_REPO=Qwen/Qwen3-4B-Instruct-2507
MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
MODEL_DIRNAME=Qwen3-4B-Instruct-2507
MODEL_SHA256="\
75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6  model-00001-of-00003.safetensors
0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1  model-00002-of-00003.safetensors
7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d  model-00003-of-00003.safetensors"
