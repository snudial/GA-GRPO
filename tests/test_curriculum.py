import importlib
import pytest
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def _record_once(args):
    state_path, task_id, success, turns = args
    import ga_grpo.curriculum as curriculum

    curriculum.STATE_PATH = state_path
    curriculum.record(task_id, success, turns)


def _curriculum(monkeypatch, tmp_path, *, window=4, promote=0.75, demote=0.25):
    monkeypatch.setenv("EXP2_CURRICULUM_STATE", str(tmp_path / "curriculum.json"))
    monkeypatch.setenv("EXP2_CURRICULUM_WINDOW", str(window))
    monkeypatch.setenv("EXP2_CURRICULUM_PROMOTE", str(promote))
    monkeypatch.setenv("EXP2_CURRICULUM_DEMOTE", str(demote))
    import ga_grpo.curriculum as curriculum

    return importlib.reload(curriculum)


def test_success_promotes_one_turn_and_resets_window(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path)

    assert curriculum.current_h("task-a", h_init=3, H=5) == 3
    for _ in range(3):
        assert curriculum.record("task-a", True, H=5) == 3
    assert curriculum.record("task-a", False, H=5) == 2
    assert curriculum.snapshot()["task-a"] == {"h": 2, "n": 0, "succ": 0}


def test_failure_demotes_one_turn_and_respects_bounds(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, window=2)

    assert curriculum.current_h("task-a", h_init=1, H=3) == 1
    assert curriculum.record("task-a", False, H=3) == 1
    assert curriculum.record("task-a", False, H=3) == 2
    assert curriculum.record("task-a", False, H=3) == 2
    assert curriculum.record("task-a", False, H=3) == 2

    curriculum.current_h("task-b", h_init=0, H=3)
    assert curriculum.record("task-b", True, H=3) == 0
    assert curriculum.record("task-b", True, H=3) == 0


def test_tasks_are_isolated_and_state_survives_reload(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, window=2)

    curriculum.current_h("task-a", h_init=2, H=4)
    curriculum.current_h("task-b", h_init=1, H=4)
    curriculum.record("task-a", True, H=4)
    curriculum.record("task-a", True, H=4)

    curriculum = importlib.reload(curriculum)
    assert curriculum.current_h("task-a", h_init=3, H=4) == 1
    assert curriculum.current_h("task-b", h_init=3, H=4) == 1


def test_concurrent_records_are_not_lost(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, window=64)
    curriculum.current_h("task-a", h_init=3, H=5)
    args = [(curriculum.STATE_PATH, "task-a", True, 5)] * 32

    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        list(pool.map(_record_once, args))

    assert curriculum.snapshot()["task-a"] == {"h": 3, "n": 32, "succ": 32}


def test_need_based_assistance_rises_then_falls_and_survives_resume(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, window=16, promote=0.7, demote=0.1)
    assert curriculum.current_h("hard", curriculum.initial_h(4, "autonomous"), 4) == 0
    for expected in [1, 2]:
        for _ in range(16):
            curriculum.record("hard", False, 4)
        assert curriculum.snapshot()["hard"]["h"] == expected
    curriculum = importlib.reload(curriculum)
    assert curriculum.current_h("hard", curriculum.initial_h(4, "autonomous"), 4) == 2
    for expected in [1, 0, 0]:
        for _ in range(16):
            curriculum.record("hard", True, 4)
        assert curriculum.snapshot()["hard"]["h"] == expected


def test_initialization_preserves_suffix_default_and_single_turn_bounds(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path)
    assert curriculum.initial_h(4) == 3
    assert curriculum.initial_h(1, "suffix") == 0
    assert curriculum.initial_h(1, "autonomous") == 0
    with pytest.raises(ValueError, match="curriculum_init"):
        curriculum.initial_h(4, "typo")


def test_call_positions_flatten_turns_in_assistance_order():
    from ga_grpo.curriculum import call_positions
    positions, starts = call_positions([["a()", "b()"], [], ["c()", "d()", "e()"]])
    assert positions == [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert starts == [0, 2, 3]


def test_call_level_steps_adapt_to_the_window(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, window=4, promote=0.7, demote=0.1)
    starts, H = [0, 2, 3], 6
    curriculum.current_h("t", h_init=5, H=H)
    # All pass: drop to the previous turn boundary, not one call.
    for _ in range(4):
        h = curriculum.record("t", True, H, starts)
    assert h == 3
    # Mostly pass (3/4): one call.
    for s in (True, True, True, False):
        h = curriculum.record("t", s, H, starts)
    assert h == 2
    # All pass from a boundary: the previous boundary.
    for _ in range(4):
        h = curriculum.record("t", True, H, starts)
    assert h == 0
    # All fail: back off exactly one call, so mid-turn positions get tried.
    for _ in range(4):
        h = curriculum.record("t", False, H, starts)
    assert h == 1
    # Mixed middle: stay.
    for s in (True, False, True, False):
        h = curriculum.record("t", s, H, starts)
    assert h == 1
