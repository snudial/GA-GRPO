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
