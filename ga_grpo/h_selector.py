"""Online choice of the next handoff position from per-position outcome records.

The window rule in curriculum.py moves h by a fixed step from the last window's
success rate alone. On Qwen3-0.6B BFCL (2026-10-08, 5,000 windows) 77% of the
windows were all-pass or all-fail (no gradient after GRPO's filter) and 57 of
100 tasks bounced between an all-pass h and an all-fail h-1 every window.

This module instead treats the next h as a choice among all candidate
positions of the task, from what each position produced before:

1. Every position keeps decayed success/failure counts (a, b). A window of K
   rollouts at position h adds its k successes and K-k failures to h after all
   of the task's counts were multiplied by DECAY, so old evidence fades and a
   position not visited for a while drifts back to the prior and is re-tried.
2. "bandit": one Thompson draw q_p ~ Beta(a_p+1, b_p+1) per position gives the
   chance that the next K-rollout group at p is mixed, 1 - q^K - (1-q)^K.
   Among positions within TIE of the best draw the earliest (smallest h, the
   student performs the most) wins. With probability EXPLORE the least-observed
   position is taken instead. The step size is whatever the choice implies.
3. "uniform": the next position is drawn uniformly (the control that measures
   the effect of merely visiting many positions).

Success here is the outcome of the whole continuation from that start, not a
verdict on the action at the handoff. Records live in the task's curriculum
entry (curriculum.json, flock-guarded), one Random seeded per (seed, task,
window) so the choice replays identically across processes and resumes.
"""
import os
import random

DEFAULTS = dict(decay=0.8, tie=0.1, explore=0.1, updates_per_epoch=1.0)


def settings():
    """DECAY is per epoch of policy updates: a group closing after dv updates fades the task's
    records by DECAY ** (dv / UPDATES_PER_EPOCH) (0.8 per epoch: half-life ~3.1 epochs); no
    update, no fade. TIE is a difference in the mixed-group score (not in success rate): with
    K=16 and a best score near 1, sampled success rates of about 13-87% all fall within 0.1.
    EXPLORE is the share of choices that take the least-observed position (random among ties)."""
    env = os.environ
    return dict(decay=float(env.get("EXP2_BANDIT_DECAY", DEFAULTS["decay"])),
                tie=float(env.get("EXP2_BANDIT_TIE", DEFAULTS["tie"])),
                explore=float(env.get("EXP2_BANDIT_EXPLORE", DEFAULTS["explore"])),
                updates_per_epoch=float(env.get("EXP2_BANDIT_UPDATES_PER_EPOCH", DEFAULTS["updates_per_epoch"])))


def decay_factor(updates_advanced, decay, updates_per_epoch):
    """Fade applied to a task's records when the policy advanced `updates_advanced` updates."""
    if updates_advanced <= 0:
        return 1.0
    return decay ** (updates_advanced / updates_per_epoch)


def mixed_probability(q, k):
    """Chance that k independent rollouts with success probability q are neither all-pass nor all-fail."""
    return 1.0 - q ** k - (1.0 - q) ** k


def observe(stats, position, successes, failures, decay):
    """Fade every position's counts by `decay` (a factor, see decay_factor), then add this
    group's outcome at `position`."""
    for key, (a, b) in list(stats.items()):
        stats[key] = [a * decay, b * decay]
    a, b = stats.get(str(position), [0.0, 0.0])
    stats[str(position)] = [a + successes, b + failures]


def posterior(stats, position):
    a, b = stats.get(str(position), [0.0, 0.0])
    return dict(n=a + b, q=(a + 1.0) / (a + b + 2.0))


def select_bandit(stats, positions, rng, *, group, tie, explore):
    """Next position from Thompson draws of each position's mixed-group chance.

    Returns (position, log) where log records whether this was an explore pick, the
    chosen and best scores, and every candidate's observation count before the choice."""
    n_before = {str(p): posterior(stats, p)["n"] for p in positions}
    if explore > 0 and rng.random() < explore:
        fewest = min(n_before.values())
        choice = rng.choice([p for p in positions if n_before[str(p)] == fewest])
        return choice, dict(explore=True, chosen_score=None, best_score=None, n_before=n_before)
    draws = {}
    for p in positions:
        a, b = stats.get(str(p), [0.0, 0.0])
        draws[p] = mixed_probability(rng.betavariate(a + 1.0, b + 1.0), group)
    best = max(draws.values())
    choice = min(p for p in positions if draws[p] >= best - tie)
    return choice, dict(explore=False, chosen_score=draws[choice], best_score=best, n_before=n_before)


def select_uniform(positions, rng):
    return rng.choice(list(positions))


def rng_for(task_id, window_index):
    seed = os.environ.get("TRAIN_SEED", "42")
    return random.Random(f"{seed}:{task_id}:{window_index}")


def choose(name, stats, positions, rng, *, group):
    """Dispatch on the selector name; returns (next position, selection log)."""
    if name == "uniform":
        return select_uniform(positions, rng), dict(explore=None, chosen_score=None, best_score=None,
                                                    n_before={str(p): posterior(stats, p)["n"] for p in positions})
    if name == "bandit":
        s = settings()
        return select_bandit(stats, positions, rng, group=group, tie=s["tie"], explore=s["explore"])
    raise ValueError(f"unknown BFCL_H_SELECTOR {name!r} (window, uniform, bandit)")
