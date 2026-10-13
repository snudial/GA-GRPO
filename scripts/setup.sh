#!/usr/bin/env bash
# One-time checkout setup for GA-GRPO. Does NOT create virtualenvs or download
# models (see README.md); it wires the pieces that live in git.
#   cp config/env.example.sh config/env.sh && edit it
#   bash scripts/setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -f config/env.sh ] || { echo "config/env.sh missing: cp config/env.example.sh config/env.sh and edit it" >&2; exit 1; }
source config/env.sh

echo "== Verify bundled Molt runtime"
python3 "$ROOT/scripts/verify_vendor.py"

echo "== BFCL protocol profile check (BFCL_PROFILE=${BFCL_PROFILE:-author_lineage})"
( source config/bfcl_protocol.sh && echo "profile=$BFCL_PROFILE prefix=$BFCL_ACTUAL_VERSION commit=${BFCL_RUNTIME_COMMIT:0:8} venv=$BFCL_RUNTIME_VENV" )

echo "== task manifests -> Molt JSONL (data/bfcl)"
python3 ga_grpo/prepare_bfcl_data.py
ls -1 data/bfcl

echo "== writable roots"
mkdir -p "${RUN_ROOT:-$ROOT/runs}" "${CACHE_ROOT:-$ROOT/.cache}"
echo "RUN_ROOT=${RUN_ROOT:-$ROOT/runs}  CACHE_ROOT=${CACHE_ROOT:-$ROOT/.cache}"

echo "== protobuf overlay for the official pool (compat/bfcl_protobuf_overlay/README.md)"
OVERLAY="${BFCL_PROTOBUF_OVERLAY:-${CACHE_ROOT:-$ROOT/.cache}/bfcl_protobuf_overlay}"
if [ -n "${MOLT_VENV:-}" ] && [ -d "$MOLT_VENV/lib/python3.12/site-packages/google/protobuf" ]; then
  mkdir -p "$OVERLAY/google"
  ln -sfn "$MOLT_VENV/lib/python3.12/site-packages/google/protobuf" "$OVERLAY/google/protobuf"
  echo "BFCL_PROTOBUF_OVERLAY=$OVERLAY -> $(readlink "$OVERLAY/google/protobuf")"
else
  echo "WARNING: could not build the protobuf overlay (MOLT_VENV google/protobuf not found)"
fi

for v in MOLT_VENV MODELS_DIR; do
  [ -n "${!v:-}" ] && [ -e "${!v}" ] || echo "WARNING: $v is unset or missing (${!v:-})"
done
echo "setup complete"
