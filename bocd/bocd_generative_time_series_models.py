"""
Module for generating sample data from piecewise consecutive time series models.
Useful for testing the BOCD algorithm.
"""

from abc import ABC, abstractmethod

import numpy as np

from .bocd import ChangepointProbabilityDistribution
from .bocd_types import FloatArray, IntArray


class GenerativeTimeSeriesModel(ABC):
    """A piecewise consecutive time series model which can generate sample data.

    Useful for generating test data for time series that
    underly a stochastic process whose statistical properties are only constant
    over consecutive time intervals or *segments* of time, thus
    changing their data generative properties according
    to some random process across segments.

    The initial segment starts at time `t = 1`. The time `t` is incremented
    each time a new observation is sampled by the `sample()` method.

    A new segment starts when the `mark_changepoint()` method is called.
    This will instruct the class to reset the internal state of the model
    to some random statistical properties, which will then be
    used for generating samples within this new segment when calling `sample()`.

    The first changepoint is at time `t = 1` after initialization.
    """

    def __init__(self) -> None:
        self._t: int = 1
        self.mark_changepoint()

    @property
    def t0(self) -> int:
        """Time of the start of the current segment."""
        return self._t0

    def mark_changepoint(self) -> None:
        """Marks the current time step as the start of a new segment This point in time is also called a *changepoint*.

        The model will reset its internal state to some random statistical properties
        which will then be used for generating samples for this new segment when calling `sample()`.
        """
        self._t0 = self._t
        self._observations_in_segment = np.array([])
        self._reset_state()

    @abstractmethod
    def _reset_state():
        """Resets the internal state of the model to some random statistical properties.

        This will instruct the class to reset the internal state of the model
        to some random statistical properties, which will then be
        used for generating samples within this new segment when calling `sample()`."""
        ...

    def sample(self) -> tuple[int, float]:
        """Generates a single sample `x_{t}` from the one-step ahead posterior predictive distribution
        `p(x_{t}|x_{t0}, x_{t0+1}, ... x_{t-1})`, which conditions on all observations sampled by
        this method since and including time `t0`, which is the point
        in time were the current segment has started (aka *changepoint*).

        The time `t0` is set by `mark_changepoint()` and is `t0 = 1` after
        initialization, i.e., on its first call this method will generate a sample
        from (the prior) `p(x_{1})`.

        If no `mark_changepoint()` is invoked afterwards, the next
        call will generate a sample from `p(x_{2}|x_{1})`, where `x_{1}` is the
        sample that was generated in the first call, etc.

        If, however, `mark_changepoint()` was called after the first call, the
        second call to `sample()` will generate a sample from (the prior) `p(x_{2})`, as
        `x_{1}` is not associated with the new segment.

        Returns:
            tuple[int, float]: (time `t`, sample `x_{t}`)
        """
        x = self._sample()
        self._observations_in_segment = np.append(self._observations_in_segment, x)
        current_time = self._t
        self._t += 1
        return current_time, x

    @abstractmethod
    def _sample(self) -> float:
        """This method should generate a sample from the one-step ahead posterior predictive distribution
        `p(x_{t}|x_{t0}, x_{t0+1}, ... x_{t-1})`.

        The observations within the current segment are stored in the array
        `self._observations_in_segment` which contains all observations within
        the current segment in the order ascending by time. i.e. `[x_{t0}, x_{t0+1}, ... x_{t-1}]`.

        The current time `t` is given by `self._t`.

        See `sample()` for detailed documentation.
        """


def generate_data(
    model: GenerativeTimeSeriesModel,
    N: int,
    cp_prob: ChangepointProbabilityDistribution,
) -> tuple[IntArray, FloatArray, IntArray]:
    """Generate synthetic time series data.

    Args:
        model (GenerativeTimeSeriesModel): The time series model that is used to sample the data from.
        N (int): The length of the wanted times series.
        cp_prob (ChangepointProbabilityDistribution): The changepoint probability distribution
            `p(r_{t}=0|r_{t-1})`.

    Returns:
        tuple[IntArray, FloatArray, IntArray]: (time, data, changepoint times)
    """
    data = np.zeros(N)
    times = np.zeros(N, dtype=int)
    cps = []
    current_run_length = 0
    for n in range(N):
        if np.random.random() <= cp_prob(np.array([current_run_length]), n + 1):
            model.mark_changepoint()
            current_run_length = 0
            cps.append(model.t0)
        else:
            current_run_length += 1
        times[n], data[n] = model.sample()
    return times, data, np.array(cps)


class GenerativeGaussianUnknownMean(GenerativeTimeSeriesModel):
    """A generative time series model with iid observations and unknown mean and known observation variance.

    The observations `x` are assumed to follow an iid Gaussian likelihood `p(x|mu, varx)`, with known variance `varx`.
    As `varx` is assumed to be known, the only statistical property that changes
    between segments is `meanx`, which is drawn from the (conjugate) prior `N(meanx|mean0, var0)`.
    """

    def __init__(self, mean0: float, var0: float, varx: float) -> None:
        """Ctor

        Args:
            mean0 (float): Mean of the Gaussian prior for the observation mean.
            var0 (float): Variance of the Gaussian prior for the observation mean.
            varx (float): Variance of each observation.
        """
        self._mean0: float = mean0
        self._var0: float = var0
        self._varx: float = varx
        super().__init__()

    def _reset_state(self) -> None:
        # Draw a new observation mean from the prior `N(meanx|mean0, var0)`.
        self._meanx = np.random.normal(self._mean0, self._var0)

    def _sample(self) -> float:
        return np.random.normal(self._meanx, self._varx)
