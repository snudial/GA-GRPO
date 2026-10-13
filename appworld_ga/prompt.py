"""AppWorld prompt informed by SAGE's public REPL protocol (no skill rewards).

Reference: amazon-science/SAGE, commit 3c9244e82244abb1adc5467ee601a03ba0f433a0,
patches/appworld/experiments/prompts/skill_library_agent.txt.
The text is independently written; no skill library or example solution is injected.
"""
import json
import os
import re
from pathlib import Path

ONBOARDING = Path(__file__).with_name('onboarding.txt').read_text()


def initial_message(task):
    # Explicit allowlist: never render answer keys or arbitrary task metadata.
    profile = {key: task.get('supervisor', {}).get(key, '')
               for key in ('first_name', 'last_name', 'email', 'phone_number')}
    return {'role': 'user', 'content': ONBOARDING.format(
        instruction=task['instruction'], supervisor=json.dumps(profile, ensure_ascii=False),
        datetime=str(task.get('datetime', 'not provided')))}


def initial_messages(task, profile=None):
    """Versioned protocol: legacy remains available for matched comparisons.

    Split the template BEFORE rendering public fields so task text containing
    USER:/ASSISTANT: cannot introduce a chat role. Demonstrations are context,
    never actions to execute or student tokens to train on.
    """
    profile = profile or os.environ.get('APPWORLD_PROMPT_PROFILE', 'legacy')
    if profile == 'legacy':
        return [initial_message(task)]
    if profile != 'rllm_react_corrected':
        raise ValueError(f'Unknown AppWorld prompt profile: {profile}')
    from jinja2 import Environment, StrictUndefined
    template = Path(__file__).with_name('prompts').joinpath('rllm_react_corrected.jinja').read_text()
    parts = re.split(r'(?m)^(USER|ASSISTANT|SYSTEM):\n', template)
    assert not parts[0].strip() and len(parts) % 2 == 1
    profile_fields = {k: task['supervisor'][k] for k in
                      ('first_name', 'last_name', 'email', 'phone_number')}
    public_apps = [{k: entry[k] for k in ('name', 'description')}
                   for entry in task['app_descriptions']]
    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    fields = dict(main_user=profile_fields, app_descriptions=json.dumps(public_apps, indent=1),
                  input_str=task['instruction'])
    return [{'role': parts[i].lower(), 'content': env.from_string(parts[i+1]).render(**fields)}
            for i in range(1, len(parts), 2)]
