"""Checkpoint copying and validation-curve parsing shared by the AppWorld driver."""
import ast
import os
import re
import shutil
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def copy_checkpoint_file(source, destination):
    if Path(source).suffix in {".safetensors", ".distcp"}:
        try:
            os.link(source, destination)
            return destination
        except OSError:
            pass
    return shutil.copy2(source, destination)

def read_curve(run):
    rows, budgets = [], {}
    log = run/'logs/train.log'
    if not log.exists(): return rows
    with log.open(errors='replace') as stream:
        for raw in stream:
            line=ANSI.sub('',raw)
            m=re.search(r'(?:Global step |Generation progress at step )(\d+):\s*(\{.*\})',line)
            if m:
                d=ast.literal_eval(m[2]);budgets[int(m[1])]=int(d.get('rollout/total_prompt_groups_sampled',0)*16)
            m=re.search(r'Eval at step (\d+):\s*(\{.*\})',line)
            if m and int(m[1])>0:
                d=ast.literal_eval(m[2]);step=int(m[1])
                rows.append(dict(optimizer_step=step,generated_rollouts=budgets.get(step),
                    val_avg_at_3=d['eval_default_pass1'], source='in_training',
                    checkpoint=str(run/'state/_hf'/f'best_global_step{step}')))
    return rows
