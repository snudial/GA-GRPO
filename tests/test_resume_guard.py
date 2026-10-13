import copy

import pytest

from ga_grpo.resume_guard import UnsafeResume, validate_state


def state(consumed=3000, episode=30):
    return dict(episode=episode, global_step=25, total_consumed_prompts=consumed,
                total_trained_groups=782, data_loader_state_dict={},
                rollout_generator_state_dict=dict(pending_rollouts=0,
                    pending_finished_samples=0, round_exhausted=True, curriculum_state={}))


@pytest.mark.parametrize("method", ["grpo", "ga-grpo", "dapo"])
def test_segment_resume_complete_and_extension(method):
    kw = dict(dataset_groups=100, curriculum=method == "ga-grpo")
    assert validate_state(state(), target_groups=5000, **kw) == "resume"
    assert validate_state(state(5000, 50), target_groups=5000, **kw) == "complete"
    assert validate_state(state(5000, 50), target_groups=6000, **kw) == "resume"


@pytest.mark.parametrize("consumed", [5100, 5200, 5300])
def test_overshoot_cannot_be_retried(consumed):
    with pytest.raises(UnsafeResume, match="generated budget mismatch"):
        validate_state(state(consumed, 49), dataset_groups=100, target_groups=5000)


def test_actual_dapo_legacy_episode_29_at_3000_is_rejected():
    with pytest.raises(UnsafeResume, match="episode/cumulative"):
        validate_state(state(3000, 29), dataset_groups=100, target_groups=5000)


@pytest.mark.parametrize("field", ["episode", "global_step", "total_consumed_prompts", "total_trained_groups"])
def test_missing_counter_cannot_silently_resume_at_zero(field):
    s = state()
    del s[field]
    with pytest.raises(UnsafeResume, match="counter"):
        validate_state(s, dataset_groups=100, target_groups=5000)


@pytest.mark.parametrize("pending", ["pending_rollouts", "pending_finished_samples"])
@pytest.mark.parametrize("consumed", [3000, 5000])
def test_pending_work_is_not_silently_lost(pending, consumed):
    s = state(consumed, consumed // 100)
    s["rollout_generator_state_dict"][pending] = 1
    with pytest.raises(UnsafeResume, match="pool"):
        validate_state(s, dataset_groups=100, target_groups=5000)


def test_jump_requires_curriculum_snapshot():
    s = state()
    del s["rollout_generator_state_dict"]["curriculum_state"]
    with pytest.raises(UnsafeResume, match="curriculum"):
        validate_state(s, dataset_groups=100, target_groups=5000, curriculum=True)
    assert validate_state(copy.deepcopy(s), dataset_groups=100, target_groups=5000) == "resume"
