import pytest

from ga_grpo.dapo import overlong_reward


def test_overlong_reward_matches_official_linear_buffer():
    kwargs = dict(max_response_tokens=10000, buffer_tokens=2000, penalty_factor=1.0)
    assert overlong_reward(7999, **kwargs) == 0.0
    assert overlong_reward(8000, **kwargs) == 0.0
    assert overlong_reward(9000, **kwargs) == pytest.approx(-0.5)
    assert overlong_reward(10000, **kwargs) == pytest.approx(-1.0)
    assert overlong_reward(12000, **kwargs) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_response_tokens=0, buffer_tokens=1, penalty_factor=1),
        dict(max_response_tokens=10, buffer_tokens=0, penalty_factor=1),
        dict(max_response_tokens=10, buffer_tokens=11, penalty_factor=1),
        dict(max_response_tokens=10, buffer_tokens=1, penalty_factor=-1),
    ],
)
def test_overlong_reward_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        overlong_reward(1, **kwargs)
