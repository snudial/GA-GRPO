"""Online handoff selection (ga_grpo/h_selector.py) and its curriculum hook."""
import collections
import importlib
import json
import random

import pytest

from ga_grpo import h_selector


def test_mixed_probability_peaks_at_even_odds():
    assert h_selector.mixed_probability(0.5, 16) > h_selector.mixed_probability(0.9, 16) > h_selector.mixed_probability(1.0, 16) == 0


def test_observe_decays_old_evidence_and_adds_the_window():
    stats = {}
    h_selector.observe(stats, 3, successes=16, failures=0, decay=0.5)
    h_selector.observe(stats, 2, successes=0, failures=16, decay=0.5)
    assert stats["3"] == [8.0, 0.0] and stats["2"] == [0.0, 16.0]
    assert h_selector.posterior(stats, 3)["q"] == pytest.approx(9 / 10)
    assert h_selector.posterior(stats, 5) == dict(n=0.0, q=0.5)   # never visited: the prior


def test_bandit_prefers_the_position_whose_groups_mix_over_observed_extremes():
    stats = {"0": [0.0, 160.0], "1": [80.0, 80.0], "2": [160.0, 0.0]}   # all-fail, mixed, all-pass; 3 unobserved
    picks = collections.Counter(h_selector.select_bandit(stats, [0, 1, 2, 3], random.Random(i),
                                                         group=16, tie=0.1, explore=0.0)[0] for i in range(300))
    assert picks[1] > 200, picks                 # the mixed position dominates
    assert picks[0] <= 10 and picks[2] == 0, picks   # observed all-fail / all-pass positions are (almost) never chosen
    assert picks[3] == 0, picks                  # a later unobserved position never wins a tie against an earlier one
    explored = collections.Counter(h_selector.select_bandit(stats, [0, 1, 2, 3], random.Random(i),
                                                            group=16, tie=0.1, explore=0.1)[0] for i in range(300))
    assert 10 < explored[3] < 60, explored       # ...it is reached through the explore share (least observed)


def test_bandit_unobserved_earlier_position_wins_near_ties_until_it_fails():
    # An untried earlier position draws from the prior and often lands within TIE of a mixed
    # position, so it is tried (the student performs more); one all-fail window then sidelines it.
    stats = {"1": [80.0, 80.0]}
    first = collections.Counter(h_selector.select_bandit(stats, [0, 1], random.Random(i), group=16, tie=0.1, explore=0.0)[0]
                                for i in range(300))
    assert first[0] > 100
    h_selector.observe(stats, 0, successes=0, failures=16, decay=0.8)
    after = collections.Counter(h_selector.select_bandit(stats, [0, 1], random.Random(i), group=16, tie=0.1, explore=0.0)[0]
                                for i in range(300))
    assert after[0] < first[0] / 4, (first, after)


def test_bandit_explore_takes_the_least_observed_position_at_random_among_ties():
    stats = {"3": [8.0, 8.0], "2": [1.0, 1.0]}   # 0 and 1 unobserved: equally least observed
    picks = collections.Counter()
    for i in range(200):
        choice, log = h_selector.select_bandit(stats, [0, 1, 2, 3], random.Random(i), group=16, tie=0.1, explore=1.0)
        assert log["explore"] is True and log["n_before"]["3"] == 16.0
        picks[choice] += 1
    assert set(picks) == {0, 1} and min(picks.values()) > 50, picks


def test_selection_log_reports_scores():
    stats = {"1": [80.0, 80.0]}
    choice, log = h_selector.select_bandit(stats, [1, 2], random.Random(3), group=16, tie=0.1, explore=0.0)
    assert log["explore"] is False and 0 <= log["chosen_score"] <= log["best_score"] <= 1
    assert log["n_before"] == {"1": 160.0, "2": 0.0}


def test_decay_follows_policy_updates_not_groups():
    assert h_selector.decay_factor(0, 0.8, 1.0) == 1.0                      # no update: no fade
    assert h_selector.decay_factor(1, 0.8, 1.0) == pytest.approx(0.8)      # one epoch of updates
    assert h_selector.decay_factor(5, 0.8, 10.0) == pytest.approx(0.8 ** 0.5)


def test_uniform_covers_every_candidate():
    rng = random.Random(1)
    assert {h_selector.select_uniform([0, 1, 2, 3], rng) for _ in range(200)} == {0, 1, 2, 3}


def test_rng_is_reproducible_per_task_and_window(monkeypatch):
    monkeypatch.setenv("TRAIN_SEED", "7")
    a = h_selector.rng_for("task", 3).random()
    assert a == h_selector.rng_for("task", 3).random() != h_selector.rng_for("task", 4).random()


def _curriculum(monkeypatch, tmp_path, selector, window=4):
    monkeypatch.setenv("EXP2_CURRICULUM_STATE", str(tmp_path / "curriculum.json"))
    monkeypatch.setenv("EXP2_CURRICULUM_WINDOW", str(window))
    monkeypatch.setenv("BFCL_H_SELECTOR", selector)
    monkeypatch.setenv("TRAIN_SEED", "42")
    import ga_grpo.curriculum as curriculum
    return importlib.reload(curriculum)


def test_curriculum_window_rule_is_unchanged_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("BFCL_H_SELECTOR", raising=False)
    curriculum = _curriculum(monkeypatch, tmp_path, "window")
    assert curriculum.current_h("t", h_init=3, H=5) == 3
    for _ in range(4):
        h = curriculum.record("t", True, H=5)
    assert h == 2
    event = json.loads((tmp_path / "curriculum.json.events.jsonl").read_text().splitlines()[-1])
    assert event["selector"] == "window" and event["candidates"] is None


def test_curriculum_bandit_records_positions_and_chooses_among_all_candidates(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, "bandit")
    positions, turn_starts = curriculum.call_positions([["a"], ["b", "c"], ["d"]])   # 4 positions
    assert curriculum.current_h("t", turn_starts[2], len(positions)) == 3
    for _ in range(4):
        h = curriculum.record("t", True, len(positions), turn_starts)   # all-pass at 3
    assert 0 <= h < len(positions)
    state = json.loads((tmp_path / "curriculum.json").read_text())["t"]
    assert state["stats"]["3"] == [4.0, 0.0] and state["windows"] == 1 and state["n"] == 0
    event = json.loads((tmp_path / "curriculum.json.events.jsonl").read_text().splitlines()[-1])
    assert event["selector"] == "bandit" and set(event["candidates"]) == {"0", "1", "2", "3"}
    assert event["candidates"]["3"]["q"] == pytest.approx(5 / 6)


def test_curriculum_uniform_moves_without_a_step_rule(monkeypatch, tmp_path):
    curriculum = _curriculum(monkeypatch, tmp_path, "uniform")
    seen = set()
    for window in range(12):
        for _ in range(4):
            h = curriculum.record("t", False, H=6)
        seen.add(h)
    assert len(seen) >= 3 and seen <= set(range(6))


def _groups(monkeypatch, tmp_path, selector="bandit", window=4):
    curriculum = _curriculum(monkeypatch, tmp_path, selector, window)
    monkeypatch.setenv("EXP2_VERSION_FILE", str(tmp_path / "policy_version.txt"))
    (tmp_path / "policy_version.txt").write_text("0")
    return curriculum


def _events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "curriculum.json.events.jsonl").read_text().splitlines()]


def test_group_lease_fixes_h_for_every_rollout_and_opens_a_new_group_after_window(monkeypatch, tmp_path):
    curriculum = _groups(monkeypatch, tmp_path)
    leases = [curriculum.assign("t", 3, 5) for _ in range(4)]
    assert {g for g, _ in leases} == {"t:1"} and {h for _, h in leases} == {3}
    for i, (g, h) in enumerate(leases[:3]):
        curriculum.record("t", True, 5, group_id=g, rollout_id=f"r{i}")
    # a 5th rollout starts before the first group closed: it opens group 2 at the still-current h
    g2, h2 = curriculum.assign("t", 3, 5)
    assert g2 == "t:2" and h2 == 3
    curriculum.record("t", True, 5, group_id="t:1", rollout_id="r3")   # closes group 1 (4/4)
    ev = [e for e in _events(tmp_path) if e["status"] == "complete"]
    assert len(ev) == 1 and ev[0]["group_id"] == "t:1" and ev[0]["success_rate"] == 1.0 and ev[0]["rollouts"] == 4
    state = json.loads((tmp_path / "curriculum.json").read_text())["t"]
    assert state["stats"]["3"] == [4.0, 0.0] and state["open"]["id"] == "t:2" and state["open"]["h"] == 3


def test_duplicate_reports_count_once(monkeypatch, tmp_path):
    curriculum = _groups(monkeypatch, tmp_path)
    g, h = curriculum.assign("t", 2, 4)
    for _ in range(3):
        curriculum.assign("t", 2, 4)
    for _ in range(3):
        curriculum.record("t", True, 4, group_id=g, rollout_id="same")
    assert not [e for e in _events(tmp_path) if e["status"] == "complete"]
    assert sum(e["status"] == "duplicate" for e in _events(tmp_path)) == 2
    for i in range(3):
        curriculum.record("t", False, 4, group_id=g, rollout_id=f"r{i}")
    done = [e for e in _events(tmp_path) if e["status"] == "complete"]
    assert len(done) == 1 and done[0]["success_rate"] == 0.25


def test_dropped_rollout_closes_the_group_as_incomplete_and_excluded(monkeypatch, tmp_path):
    curriculum = _groups(monkeypatch, tmp_path)
    g1 = [curriculum.assign("t", 3, 5) for _ in range(4)][0][0]
    for i in range(3):   # the 4th rollout of group 1 is dropped by the runner and never reports
        curriculum.record("t", True, 5, group_id=g1, rollout_id=f"a{i}")
    for n, tag in ((2, "b"), (3, "c")):   # two later groups complete; group 1 can then not be waited for
        g = [curriculum.assign("t", 3, 5) for _ in range(4)][0][0]
        assert g == f"t:{n}"
        for i in range(4):
            curriculum.record("t", False, 5, group_id=g, rollout_id=f"{tag}{i}")
        inc = [e for e in _events(tmp_path) if e["status"] == "incomplete"]
        assert len(inc) == (0 if n == 2 else 1), (n, inc)
    inc = [e for e in _events(tmp_path) if e["status"] == "incomplete"][0]
    assert inc["group_id"] == "t:1" and inc["rollouts"] == 3 and inc["planned"] == 4 and inc["success_rate"] == 1.0
    done = [e for e in _events(tmp_path) if e["status"] == "complete"]
    assert [e["group_id"] for e in done] == ["t:2", "t:3"]
    stats = json.loads((tmp_path / "curriculum.json").read_text())["t"]["stats"]
    assert sum(a for a, b in stats.values()) == 0.0   # group 1's three successes never entered the records
    assert sum(b for a, b in stats.values()) == pytest.approx(8.0)
    assert json.loads((tmp_path / "curriculum.json").read_text())["t"]["pending"] == []


def test_interleaved_completion_of_two_groups_closes_each_once(monkeypatch, tmp_path):
    curriculum = _groups(monkeypatch, tmp_path)
    g1 = [curriculum.assign("t", 3, 5) for _ in range(4)][0][0]
    g2 = [curriculum.assign("t", 3, 5) for _ in range(4)][0][0]
    order = [(g2, "b0", False), (g1, "a0", True), (g2, "b1", False), (g1, "a1", True),
             (g1, "a2", True), (g2, "b2", False), (g2, "b3", False), (g1, "a3", True)]
    for g, r, ok in order:
        curriculum.record("t", ok, 5, group_id=g, rollout_id=r)
    done = [e for e in _events(tmp_path) if e["status"] == "complete"]
    assert [(e["group_id"], e["success_rate"]) for e in done] == [("t:2", 0.0), ("t:1", 1.0)]
    assert not [e for e in _events(tmp_path) if e["status"] == "incomplete"]
    stats = json.loads((tmp_path / "curriculum.json").read_text())["t"]["stats"]
    assert stats["3"] == [4.0, 4.0]


def test_records_fade_with_policy_updates_only(monkeypatch, tmp_path):
    curriculum = _groups(monkeypatch, tmp_path)
    monkeypatch.setenv("EXP2_BANDIT_DECAY", "0.5")
    monkeypatch.setenv("EXP2_BANDIT_UPDATES_PER_EPOCH", "1")
    def group(ok):
        g = [curriculum.assign("t", 3, 5) for _ in range(4)][0][0]
        for i in range(4):
            curriculum.record("t", ok, 5, group_id=g, rollout_id=f"{g}-{i}")
    successes = lambda: sum(a for a, b in json.loads((tmp_path / "curriculum.json").read_text())["t"]["stats"].values())
    group(True)                                   # version 0: 4 successes recorded (at whatever h was chosen)
    group(True)                                   # still version 0: no fade -> 8
    assert successes() == pytest.approx(8.0)
    (tmp_path / "policy_version.txt").write_text("2")
    group(True)                                   # two updates: 0.5 ** 2 = 0.25 -> 8 * 0.25 + 4
    st = json.loads((tmp_path / "curriculum.json").read_text())["t"]
    assert successes() == pytest.approx(6.0) and st["stats_version"] == 2
    assert [e["decay_factor"] for e in _events(tmp_path) if e["status"] == "complete"] == [1.0, 1.0, 0.25]
