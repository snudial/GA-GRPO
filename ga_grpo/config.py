"""This host's paths, read from config/env.sh — the one file that holds a machine's
layout, so Python and shell launchers can never disagree about where things are.

    from ga_grpo.config import path
    path('MODELS_DIR') / 'Qwen3-4B-Instruct-2507'
"""
import functools
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def env():
    env_sh = ROOT/'config/env.sh'
    if not env_sh.exists():
        env_sh = ROOT/'config/env.example.sh'
    out = subprocess.run(['bash', '-c', 'source "$1" >/dev/null && env -0', '_', str(env_sh)],
                         capture_output=True, check=True).stdout.decode()
    return dict(kv.split('=', 1) for kv in out.split('\0') if '=' in kv)


def path(name):
    return Path(env()[name])


def topology_defaults(name, prefix=''):
    """The variables config/topology/<name>.env assigns (new or changed by sourcing it).

    Profiles only assign defaults (`: "${VAR:=value}"`), so callers apply the
    result with setdefault and an explicitly exported value still wins. An empty
    name (no profile: the built-in B200 defaults) gives {}.
    """
    if not name:
        return {}
    profile = ROOT/'config/topology'/f'{name}.env'
    out = subprocess.run(['env', '-i', 'PATH=/usr/bin:/bin', 'bash', '-c',
                          'env -0; printf "\\0--profile--\\0"; source "$1" >/dev/null && env -0', '_', str(profile)],
                         capture_output=True, check=True).stdout.decode()
    before, after = out.split('\0--profile--\0')
    parse = lambda text: dict(kv.split('=', 1) for kv in text.split('\0') if '=' in kv)
    base = parse(before)
    return {k: v for k, v in parse(after).items() if k.startswith(prefix) and base.get(k) != v}
