import numpy as np

from bocd.bocd import (
    ChangepointProbabilityDistribution,
    _log_probabilities_from_probabilities,
)
from bocd.bocd_types import FloatArray, IntArray


def memoryless_changepoint_probability_distribution(
    lmbda: float,
) -> ChangepointProbabilityDistribution:
    """Returns the 'memoryless' changepoint probability distribution.

    The changepoint probability is defined as the probability that a changepoint occurs
    at time `t`, given a run length `r_{t-1}` at the previous time `t - 1`, i.e. `p(r_{t}=0|r_{t-1})`.

    Note that the so called *growth probabiliy* `p(r_{t}=r_{t-1}+1|r_{t-1})`
    is implicitly defined by specifying the changepoint probability,
    because a run length can either become zero (when a changepoint occurs),
    or it can grow by one step, i.e.: `p(r_{t}=r_{t-1}+1|r_{t-1}) = 1 - p(r_{t}=0|r_{t-1})`.

    In the memoryless case, the transition probabilities are constants:
        - the changepoint transition probability: `lmbda`
        - the growth transition log probability: `1 - lmbda`

    These transition probabilities are used in the paper by Adams and MacKay.
    As mentioned there, the memoryless transition probabilities correspond
    to an exponential geometric distribution for the gap `g` between
    consecutive changepoints:
    `p_{gap}(g) = (1 - lmbda) ** (g - 1) * lmbda` for `g = 1, 2, ...`.

    Args:
        lmbda (float): The constant changepoint probability. Must be in [0, 1].

    Returns:
        ChangepointProbabilityDistribution: The changepoint probability distribution `p(r_{t}=0|r_{t-1})`
        as a function of (a list of) the previous run length `r_{t-1}`, which can take values
        `0, 1, ...`. Note that this distribution is time-independent, which means that
        p(r_{t}=0|r_{t-1}=k) has the same value at each time `t`, for any previous run length `k`.
    """

    if lmbda < 0 or lmbda > 1:
        raise ValueError(f"`lmbda` must be in [0, 1], but got {lmbda}.")

    def changepoint_prob(r_t_1: IntArray, __: int) -> FloatArray:
        return np.full(len(r_t_1), lmbda)

    return changepoint_prob


def changepoint_probability_distribution_from_gap_probability_distribution(
    gap_probabilities: FloatArray,
) -> ChangepointProbabilityDistribution:
    """Computes changepoint probabilities from gap probabilities.

    The changepoint probability is defined as the probability that a changepoint occurs
    at time `t`, given a run length `r_{t-1}` at the previous time `t - 1`, i.e. `p(r_{t}=0|r_{t-1})`.

    Note that the so called *growth probabiliy* `p(r_{t}=r_{t-1}+1|r_{t-1})`
    is implicitly defined by specifying the changepoint probability,
    because a run length can either become zero (when a changepoint occurs),
    or it can grow by one step, i.e.: `p(r_{t}=r_{t-1}+1|r_{t-1}) = 1 - p(r_{t}=0|r_{t-1})`.

    Args:
        gap_probabilities (FloatArray): Array which represents the discrete probability
            distribution of the gap between two changepoints (should add up to 1.0,
            although this is not checked):
            `gap_probabilities = [p_{gap}(g=1), p_{gap}(g=2), ..., p_{gap}(g=len(gap_probabilities))]`.

    Returns:
        ChangepointProbabilityDistribution: The changepoint probability distribution `p(r_{t}=0|r_{t-1})`
        as a function of (a list of) the previous run length `r_{t-1}`, which can take values
        `0, 1, ...`. Note that this distribution is time-independent, which means that
        `p(r_{t}=0|r_{t-1}=k)` has the same value at each time `t`, for any previous run length `k`.
    """
    log_denominator = np.log(np.cumsum(gap_probabilities[::-1])[::-1])
    gap_log_probs = _log_probabilities_from_probabilities(gap_probabilities)
    H = np.exp(
        gap_log_probs - log_denominator
    )  #  H stands for Hazard function (see paper)
    limit = len(H)

    def changepoint_prob(r_t_1: IntArray, _: int) -> FloatArray:
        cp = np.zeros_like(r_t_1, dtype=float)
        mask = r_t_1 < limit
        cp[mask] = H[r_t_1[mask]]
        return cp

    return changepoint_prob
