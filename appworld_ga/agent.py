"""Stateful code actions, strict terminal reward, student-only loss masks."""
import hashlib
import json
import fcntl
import os
import time
import uuid
from pathlib import Path

from openai import AsyncOpenAI
from molt.agents import ChatAgent, ChatAgentRunner, Result
from appworld_ga import curriculum
from appworld_ga import environment as AW
from appworld_ga import ledger
from appworld_ga.prompt import initial_messages
from ga_grpo import banks


class AppWorldAgent(ChatAgent):
    async def run(self, ctx):
        label = json.loads(ctx.label) if isinstance(ctx.label, str) else ctx.label
        # Evaluation starts from h=0 unless the row fixes a handoff (run_appworld.py --handoff-sweep).
        h = label.get('eval_h', 0) if label.get('eval') else label['h']
        turns = json.loads(Path(label['guide_path']).read_text())['turns'] if h else []
        max_turns = int(os.environ.get('APPWORLD_MAX_TURNS', '30'))
        if not 0 <= h < max_turns:
            raise ValueError('Handoff must leave at least one student turn')
        lease = await AW.acquire()
        world = AW.World(lease, label['task_id'])
        info = dict(handoff_h=h, guide_H=label.get('guide_H', 0),
                    turns_used=0, no_code_turns=0, pass_fraction=0., complete_task_called=0, cell_hang=0,
                    opsd_uid_hi=0., opsd_uid_lo=0.)  # links a failed rollout to its opsd_bank record
        boundary = None  # (student context, first student response): the OPSD target span
        completed = False
        truncated = False
        # Training ledger (appworld_ga/ledger.py): bookkeeping only, never read by the episode.
        tracing = ledger.enabled() and not label.get('eval')
        # Token counts come from the trainer's trajectory (curriculum._write_ledger): the chat server
        # reports no prompt tokens.
        cost = dict(started=time.time(), policy_version_start=ledger.policy_version() if tracing else None,
                    generation_seconds=0., environment_seconds=0., reference_seconds=0., execution_errors=0,
                    termination=None)
        steps = []
        try:
            t0 = time.monotonic()
            task = await world.initialize(f'jump_slot{lease.port}')
            messages = initial_messages(task)
            cost['environment_seconds'] += time.monotonic() - t0
            t0 = time.monotonic()
            for turn in turns[:h]:
                output = await world.execute(turn['code'])
                messages.extend([{'role': 'assistant', 'content': f"```python\n{turn['code']}\n```"},
                                 {'role': 'user', 'content': output}])
            cost['reference_seconds'] = time.monotonic() - t0
            cost['environment_seconds'] += cost['reference_seconds']
            # Policy calls retry transport faults (connection reset, 5xx): a generation request touches no
            # AppWorld state and a dropped one returned nothing, so a retry cannot replay uncertain state.
            # At 512 episodes in flight (8B, 2026-10-06) single httpx ReadErrors killed whole runs twice.
            async with AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key, max_retries=3) as client:
                for _ in range(max_turns - h):
                    step = dict(turn=info['turns_used'], history_messages=len(messages),
                                history_sha256=hashlib.sha256(json.dumps(messages).encode()).hexdigest()) if tracing else None
                    t0 = time.monotonic()
                    response = await client.chat.completions.create(
                        model=ctx.model_name, messages=messages,
                        max_tokens=ctx.sampling_params.max_tokens,
                        temperature=ctx.sampling_params.temperature, top_p=ctx.sampling_params.top_p)
                    cost['generation_seconds'] += time.monotonic() - t0
                    choice = response.choices[0]
                    raw = choice.message.content or ''
                    if step is not None:
                        step.update(generation_seconds=time.monotonic() - t0, finish_reason=choice.finish_reason,
                                    completion_tokens=getattr(response.usage, 'completion_tokens', None), response=raw)
                        steps.append(step)
                    if boundary is None:
                        boundary = (list(messages), raw)
                    messages.append({'role': 'assistant', 'content': raw})
                    info['turns_used'] += 1
                    if choice.finish_reason == 'length':
                        truncated = True
                        cost['termination'] = 'generation_length'
                        break
                    code = AW.extract_code(raw)
                    if step is not None:
                        step['code'] = code
                    if code is None:
                        info['no_code_turns'] += 1
                        if info['no_code_turns'] >= 3:
                            cost['termination'] = 'no_code'
                            break
                        messages.append({'role': 'user', 'content':
                                         'Reply with exactly one fenced python code block.'})
                        continue
                    t0 = time.monotonic()
                    try:
                        output = await world.execute(code)
                    except AW.CellHang:
                        # The student's code outlived AppWorld's cell deadline and took the
                        # server down with it: a failed episode, not an infrastructure fault.
                        info['cell_hang'] = 1
                        cost['termination'] = 'cell_hang'
                        cost['environment_seconds'] += time.monotonic() - t0
                        if step is not None:
                            step.update(cell_hang=True, execution_seconds=time.monotonic() - t0)
                        break
                    messages.append({'role': 'user', 'content': output})
                    error = 'Execution failed.' in output
                    cost['execution_errors'] += int(error)
                    completed = await world.completed()
                    cost['environment_seconds'] += time.monotonic() - t0
                    if step is not None:
                        step.update(output=output, execution_error=error, execution_seconds=time.monotonic() - t0,
                                    completed=completed)
                    if completed:
                        cost['termination'] = 'completed'
                        break
            if info['cell_hang']:
                success, fraction, failures = False, 0., []
            else:
                t0 = time.monotonic()
                success, fraction, failures = await world.evaluate(with_failures=True)
                cost['environment_seconds'] += time.monotonic() - t0
            info.update(pass_fraction=fraction, complete_task_called=int(completed),
                        observation_truncations=world.observation_truncations,
                        observation_chars_dropped=world.observation_chars_dropped)
            if (not success and not label.get('eval') and boundary and not info['cell_hang']
                    and os.environ.get('EXP2_BANK_DIR')):
                # Privileged OPSD: the bank the teacher prompt reads gets only the failed
                # requirement sentences; test traces carry ground-truth values and go to
                # the analysis-only raw file.
                uid = banks.new_uid()
                hi, lo = banks.uid_halves(uid)
                info.update(opsd_uid_hi=float(hi), opsd_uid_lo=float(lo))
                meta = dict(uid=uid, env='appworld', task_id=label['task_id'], handoff_h=h,
                            guide_H=info['guide_H'], policy_version=banks.policy_version(), pass_fraction=fraction)
                banks.append('failures_raw.jsonl', {**meta, 'failures': failures})
                banks.append('opsd_bank.jsonl', {**meta, 'certificate': {'failed_requirements': [
                    f['requirement'].strip() for f in failures if isinstance(f, dict) and f.get('requirement')]},
                    'boundary_action': boundary[1], 'context_messages': boundary[0]})
            if tracing:
                # Non-numeric info never reaches the trainer's tensors (samples_generator keeps only
                # scalars); curriculum.finish_group writes these under the trainer's rollout ids.
                info['ledger'] = dict(cost, task_id=label['task_id'], group_id=label.get('group_id'),
                                      reward=float(success), pass_fraction=fraction,
                                      termination=cost['termination'] or 'turn_limit',
                                      student_turns=info['turns_used'], no_code_turns=info['no_code_turns'],
                                      reference_turns=h, guide_H=info['guide_H'], cell_hang=info['cell_hang'],
                                      complete_task_called=int(completed), truncated=int(truncated),
                                      ended=time.time(), policy_version_end=ledger.policy_version())
                info['ledger_turns'] = steps
            if os.environ.get('APPWORLD_TRAJECTORY_DIR'):
                path = Path(os.environ['APPWORLD_TRAJECTORY_DIR'])
                path.mkdir(parents=True, exist_ok=True)
                path = path / f'{uuid.uuid4().hex}.json'
                path.write_text(json.dumps(dict(task_id=label['task_id'], success=success,
                                               messages=messages, **info)))
                info['trajectory_path'] = str(path)
            if label.get('eval') and os.environ.get('APPWORLD_EVAL_RECORDS'):
                with open(os.environ['APPWORLD_EVAL_RECORDS'], 'a') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    stream.write(json.dumps(dict(task_id=label['task_id'], success=success,
                                                 **info)) + '\n')
            return Result(reward=float(success), score=float(success), terminated=completed,
                          truncated=truncated or not completed, info=info)
        finally:
            await world.close()


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(AppWorldAgent)

    prepare_group = staticmethod(curriculum.prepare_group)
    finish_group = staticmethod(curriculum.finish_group)
