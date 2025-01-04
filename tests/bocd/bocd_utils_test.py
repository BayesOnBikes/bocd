import numpy as np
import pytest

from bocd.bocd_types import FloatArray, IntArray
from bocd.bocd_utils import (
    changepoint_probability_distribution_from_gap_probability_distribution,
    memoryless_changepoint_probability_distribution,
)


@pytest.fixture(params=[0, 42], ids=lambda t: f"t={t}")
def t(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.mark.parametrize(
    "any_valid_lambda", [0.0, 1.0, 0.5, 0.8], ids=lambda lmd: f"lambda={lmd}"
)
@pytest.mark.parametrize(
    "r_t_1",
    [np.arange(42), np.array([123]), np.array([])],
    ids=["r_t_1 range", "single r_t_1", "empty r_t_1"],
)
def test_memoryless_changepoint_probability_distribution(
    any_valid_lambda: float, t: int, r_t_1: IntArray
) -> None:
    # given
    sut = memoryless_changepoint_probability_distribution(any_valid_lambda)

    # when
    probs = sut(r_t_1, t)

    # then
    assert np.all(probs == any_valid_lambda)


def test_memoryless_changepoint_probability_distribution_raises_on_invalid_lambda() -> (
    None
):
    with pytest.raises(ValueError):
        memoryless_changepoint_probability_distribution(-0.123)


@pytest.mark.parametrize(
    "p_gap",
    [
        np.array([0.1, 0.2, 0.1, 0.3, 0.0, 0.1, 0.2]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0]),
    ],
)
def test_from_gap_probability_distribution(p_gap: FloatArray, t: int) -> None:
    # given
    assert np.sum(p_gap) == pytest.approx(
        1.0
    ), "p_gap must be a probability distribution"
    assert p_gap[-1] > 0.0

    t_max, t_add = len(p_gap), 42
    r_t_1 = np.arange(t_max + t_add + 1)
    sut = changepoint_probability_distribution_from_gap_probability_distribution(p_gap)

    # when
    probs = sut(r_t_1, t)

    # then
    assert np.all(probs <= 1.0)
    assert probs[t_max - 1] == pytest.approx(1.0)
    assert np.all(probs[t_max:] == 0.0)
