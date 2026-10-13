#!/bin/bash
# Hardware topology profile, separate from the method/protocol settings.
#
#   TOPOLOGY=single_h100 bash scripts/train.sh RUN      (or run_seed_rollouts.sh, eval_*.sh)
#
# Sources config/topology/$TOPOLOGY.env. Profiles only assign defaults
# (`: "${VAR:=value}"`), so an explicitly exported variable still wins. Unset
# TOPOLOGY keeps the scripts' built-in multi-GPU defaults (the B200 contract).
# Exports TOPOLOGY and TOPOLOGY_SHA256 for the run manifest.
if [ -n "${TOPOLOGY:-}" ]; then
  _topology_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/topology/${TOPOLOGY}.env"
  [ -f "$_topology_file" ] || { echo "unknown TOPOLOGY=$TOPOLOGY (no $_topology_file)" >&2; return 1 2>/dev/null || exit 1; }
  # shellcheck source=/dev/null
  source "$_topology_file"
  TOPOLOGY_SHA256="$(sha256sum "$_topology_file" | awk '{print $1}')"
  export TOPOLOGY TOPOLOGY_SHA256
  unset _topology_file
fi
