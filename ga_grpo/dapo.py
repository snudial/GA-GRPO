"""DAPO reward utilities shared by BFCL agents and unit tests."""


def overlong_reward(
    response_tokens: int,
    *,
    max_response_tokens: int,
    buffer_tokens: int,
    penalty_factor: float,
) -> float:
    """Official DAPO linear soft penalty near the response-token limit.

    The penalty is zero through ``max_response_tokens - buffer_tokens`` and
    decreases linearly to ``-penalty_factor`` at the limit. As in the official
    implementation, it continues linearly if a caller reports beyond-limit
    tokens. Callers count only
    policy-generated assistant tokens; prompts, guide roll-ins and tool
    observations are deliberately excluded.
    """
    if max_response_tokens <= 0:
        raise ValueError("max_response_tokens must be positive")
    if buffer_tokens <= 0 or buffer_tokens > max_response_tokens:
        raise ValueError("buffer_tokens must be in [1, max_response_tokens]")
    if penalty_factor < 0:
        raise ValueError("penalty_factor must be non-negative")
    expected = max_response_tokens - buffer_tokens
    exceed = max(0, int(response_tokens) - expected)
    return min(-exceed / buffer_tokens * float(penalty_factor), 0.0)
