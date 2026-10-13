"""run.py registries: every method is a valid train.sh switch set and every
AppWorld method name exists in the AppWorld driver."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def registry(name):
    return {k: v for k, v in json.loads((ROOT/'config'/name).read_text()).items() if not k.startswith('_')}


def test_methods_are_valid_switch_sets():
    for name, spec in registry('methods.json').items():
        bfcl = spec['bfcl']
        assert bfcl['ALGO'] in ('grpo', 'dapo'), name
        assert bfcl['BFCL_CURRICULUM'] in ('0', '1'), name
        # A handoff start or unit only means something with the curriculum on.
        if bfcl['BFCL_CURRICULUM'] == '0':
            assert not {'BFCL_CURRICULUM_INIT', 'BFCL_HANDOFF_UNIT'} & bfcl.keys(), name
        assert bfcl.get('BFCL_CURRICULUM_INIT', 'suffix') in ('suffix', 'autonomous', 'second_last_turn'), name
        if 'EXP2_OPSD_LAMBDA' in bfcl:
            assert bfcl.get('OPSD') == '1', name


def test_appworld_methods_exist_in_the_driver():
    tree = ast.parse((ROOT/'scripts/run_appworld.py').read_text())
    driver = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'METHODS')
    assert {n for n, s in registry('methods.json').items() if 'appworld' in s} <= set(driver)


def test_topologies_define_the_gpu_split_train_sh_reads():
    for path in sorted((ROOT/'config/topology').glob('*.env')):
        text = path.read_text()
        for var in ('HYBRID', 'ACTOR_GPUS', 'VLLM_ENGINES', 'NUM_GPUS', 'EVAL_GPU_COUNT'):
            assert f'${{{var}:=' in text, (path.name, var)


def test_topology_defaults_are_only_what_the_profile_assigns():
    from ga_grpo.config import topology_defaults
    assert topology_defaults('multi_h100', 'APPWORLD_')['APPWORLD_VLLM_UTIL_4B'] == '0.70'
    assert topology_defaults('multi_h100')['VLLM_UTIL'] == '0.70'
    assert 'PATH' not in topology_defaults('multi_h100')
    assert topology_defaults('') == {} and topology_defaults(None) == {}


def test_appworld_knobs_in_profiles_are_read_by_the_driver():
    # A misspelled knob in a profile would be ignored silently.
    from ga_grpo.config import topology_defaults
    driver = (ROOT/'scripts/run_appworld.py').read_text()
    tree = ast.parse(driver)
    models = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODELS')
    for path in sorted((ROOT/'config/topology').glob('*.env')):
        for key in topology_defaults(path.stem, 'APPWORLD_'):
            base = next((key[:-len(m) - 1] for m in models if key.endswith('_' + m.upper())), key)
            # A per-model knob NAME_<MODEL> is read through knob('NAME', model).
            assert f"os.environ.get('{key}'" in driver or (base != key and f"knob('{base}'" in driver), (path.name, key)
    main = driver[driver.index('def main():'):]
    # Applied before the parser, whose --pool-size default reads APPWORLD_POOL_SIZE.
    assert main.index("topology_defaults(os.environ.get('TOPOLOGY'), 'APPWORLD_')") < main.index('ArgumentParser')


def test_model_scoped_appworld_knob_precedence(monkeypatch):
    import scripts.run_appworld as driver
    for name in ('APPWORLD_VLLM_UTIL', 'APPWORLD_VLLM_UTIL_8B', 'APPWORLD_VLLM_UTIL_4B'):
        monkeypatch.delenv(name, raising=False)
    assert driver.knob('APPWORLD_VLLM_UTIL', '8b') is None
    monkeypatch.setenv('APPWORLD_VLLM_UTIL_4B', '0.70')
    assert driver.knob('APPWORLD_VLLM_UTIL', '8b') is None          # another model's default
    monkeypatch.setenv('APPWORLD_VLLM_UTIL_8B', '0.60')
    assert driver.knob('APPWORLD_VLLM_UTIL', '8b') == '0.60'
    monkeypatch.setenv('APPWORLD_VLLM_UTIL', '0.55')                # explicit export, any model
    assert driver.knob('APPWORLD_VLLM_UTIL', '8b') == '0.55' and driver.knob('APPWORLD_VLLM_UTIL', '4b') == '0.55'
