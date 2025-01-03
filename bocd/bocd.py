"""
Bayesian Online Changepoint Detection (BOCD)

This module implements a general version of the **Bayesian Online Changepoint Detection (BOCD)** algorithm,
which was originally described in the paper by Adams and MacKay (2007).

BOCD is a Bayesian method for detecting *changepoints* in time series data. It does so by modeling the posterior
distribution `p(r_{t}|x_{1:t})` of the so called **run length** `r_{t}` at each point in time `t = 1, ...`, given the
history of observations `(x_{1}, ..., x_{t})` (=: `x_{1:t}` in the paper's notation).

The main class is `BOCD`, which contains the relevant documentation.
"""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, cast

import numpy as np
from scipy.special import logsumexp

from .bocd_types import FloatArray, FloatMatrix, IntArray

"""
Function that represents the changepoint probability distribution `p(r_{t}=0|r_{t-1})`,
i.e., the probability that the run length `r_{t}` at the current time `t` is zero and thus a *changepoint*,
as a function of the previous run length `r_{t-1} = 0, 1, ...`.

Because the distribution can in principle also be a function of time `t = 1, 2, ...`,
it is passed as a second parameter.

Args:
    r_t_minus_1 (IntArray): List of previous run length `r_{t-1}`.
    t (int): Current time `t`.

Returns:
    FloatArray: List of changepoint probabilities `p(r_{t}=0|r_{t-1})`.
"""
ChangepointProbabilityDistribution = Callable[[IntArray, int], FloatArray]


class TimeSeriesModel(ABC):
    """A time series model for making one-step ahead predictions about future observations given
    observations at previous time steps.

    After initialization, the time is `t = 0` and will be incremented with each new observation passed
    to the `new_observation()` method. The first observation will be associated with time `t = 1`,
    the `n`th observation with `t = n`.

    The current time can be obtained with `self.t`.

    One-step ahead posterior predictive (aka forecast) probabilities for a given observation `x_{t+1}`
    can be obtained by calling the `log_pred_probs()` method.
    """

    def __init__(self) -> None:
        self._t: int = 0
        self._n_observations: int = 0

    @property
    def t(self) -> int:
        """Current time step."""
        return self._t

    @abstractmethod
    def log_pred_probs(self, x: float) -> FloatArray:
        """Log-probabilities of `x = x_{t+1}` given previous observations.

        Returns the log-probabilities
        `[p(x_{t+1}), p(x_{t+1}|x_{t}), p(x_{t+1}|x_{t-1}, x_{t}), ..., p(x_{t+1}|x_{1}, ..., x_{t})]`,
        i.e. the one-step ahead posterior predictive (aka forecast) probabilities for
        all `t + 1` past sub-lengths (including a sub-length of 0) of the time series.

        The probabilities can in principle be a function of time `t`, which can be queried by `self.t`.

        At time `t = 0`, i.e., before any observations has been seen,
        this method returns the probability `p(x)`:=`p(x_{1})`, which is the
        *prior probability* of `x` at time `t = 1`. Note that the first element
        of the result array is always a *prior probability* in the sense that
        it represents the probability of `x` at time `t + 1` without taking into
        account any previous observations.

        If the model has been instructed to "forget" old observations by calling `keep_only_latest(n_observations)`,
        any observations before time `t - n_observations + 1` will be discarded and this method will return
        `[p(x_{t+1}), p(x_{t+1}|x_{t}), p(x_{t+1}|x_{t-1}, x_{t}), ..., p(x_{t+1}|x_{t-n_observations+1}, ..., x_{t})]`,
        (an array with `n_observations + 1` elements) immediately afterwards.
        Future observations will again be taken into account. See `keep_only_latest()` for more details.

        Args:
            x (float): The value of the observation at time `t + 1` whose probabilities are to be computed.
        """

    def new_observation(self, x: float) -> None:
        """Updates the model with a new observation.

        Being at time step `t` before calling this method, the observation `x`
        is the observation associated with the next time step `t + 1`,
        so this method will increment the time by one step: `t -> t + 1`.

        The observation `x` will in most cases be used to pre-compute the predictive probabilities
        that will later be queried by `log_pred_probs()` to make predictions about the next time
        given this new (as well as all previous) observations.

        Args:
            x (float): The new observation at new time `t`.
        """
        self._t += 1
        self._n_observations += 1
        self._new_observation(x)

    @abstractmethod
    def _new_observation(self, x: float) -> None:
        """See `new_observation()` for documentation."""
        ...

    def keep_only_latest(self, n_observations: int) -> None:
        """Instructs the model to retain only the latest `n_observations` observations.

        This can become useful if the model keeps track of all observations that were made
        and discarding old observations that are assumed to not significantly contribute
        to the predictions can *save memory and computation time*.

        Observations that were made before time `t - n_observations + 1` will be discarded and
        no longer be considered when computing the predictive probabilities using `log_pred_probs()`.

        After invokation, the predictive probabilities returned by `log_pred_probs()` will return
        ``[p(x_{t+1}), p(x_{t+1}|x_{t}), p(x_{t+1}|x_{t-1}, x_{t}), ..., p(x_{t+1}|x_{t-n_observations+1}, ..., x_{t})]`.

        This method will discard observations only once it is called: Any new observation will make
        the array returned by `log_pred_probs()` grow again.

        E.g., if we are at `t = 10` there are 10 observations (`new_observation()` has been called 10 times),
        and we call `keep_only_latest(2)`, the model will retain only the last two observations: At `t = 9`
        and at `t = 10`. The predictive probabilities returned by `log_pred_probs()` will be
        `[p(x_{11}), p(x_{11}|x_{10}), p(x_{11}|x_{9}, x_{10})]`.
        If a new observation is made at `t = 11`, then `log_pred_probs()` will return
        `[p(x_{12}), p(x_{12}|x_{11}), p(x_{12}|x_{10}, x_{11}), p(x_{12}|x_{9}, x_{10}, x_{11})]`
        afterwards.

        Args:
            n_observations (int): The number of observations to retain. Must be any nonnegative integer.
        """
        if n_observations < 0:
            raise ValueError("`n_observations` must be a nonnegative integer.")
        if n_observations >= self._n_observations:
            return
        self._keep_only_latest(n_observations)
        self._n_observations = n_observations

    @abstractmethod
    def _keep_only_latest(self, n_observations: int) -> None:
        """See `keep_only_latest()` for documentation."""
        ...


class BOCD:
    """Bayesian online changepoint detection

    The aim of BOCD is to model the posterior distribution `p(r_{t}|x_{1:t})`
    of the so called **run length** `r_{t}` at each point in time `t = 1, ...`, given the
    observations `(x_{1}, ..., x{t})` (=: `x_{1:t}` in the paper's notation) up to that point.

    This implementation extends the original BOCD algorithm by supporting the computation
    of the "look ahead" posterior distribution `p(r_{t}|x_{1:t+h})` for `h >= 0`, thus adding
    offline changepoint detection capabilities to BOCD.

    The run length is defined as the number of time steps since the last **changepoint**.
    A changepoint is defined as a point in time where the run length is zero.

    The run length jumping to zero indicates that the time series properties have changed and
    a new *segment* of the time series has started.

    The model assumes independence of observations belonging to different segments.
    Therefore, each segment can be seen as some kind of independent "sub time series",
    where the transition between these sub time series is modeled by a pre-specified
    transition probability `p(r_{t}|r_{t-1})` (see below).

    Note that the time series starts at `t = 1`, which is associated with the first observation, `x_{1}`.
    The point in time `t = 0` is used to apply prior knowledge about the run length.

    `BOCD` needs three major inputs:
        - A *time series model*, which is used to make one-step ahead posterior
          predictions about the current observation `x_{t}` given the observations at previous time steps,
          `p(x_{t}|x_{t-r_{t}}, x_{t-(r_{t}-1)}, ..., x_{t-1})` (=: `p(x_{t}|x_{(t-r_{t}):(t-1)}`), for
          all possible run lengths `r_{t}`. See `TimeSeriesModel`.
        - A *run length prior* `p(r_{0})`, which represents a discrete probability distribution of the run length at time `t = 0`.
        - A *changepoint (transition) probability distribution* `p(r_{t}=0|r_{t-1})`, which represents the probability that
          a changepoint occurs at time `t`, given a run length `r_{t-1}` at the previous time `t - 1`.
          It is defined as a function of `r_{t-1}` (`r_{t-1} = 0, 1, ...`).

          (Note that the so called *growth probabiliy* distribution `p(r_{t}=r_{t-1}+1|r_{t-1})` is
          implicitly defined by specifying the changepoint probability, because a run length can either become zero (when a changepoint occurs),
          or it can grow by one step, i.e.: `p(r_{t}=r_{t-1}+1|r_{t-1}) = 1 - p(r_{t}=0|r_{t-1})`. This is why, although required by the algorithm,
          the additional specification of a seperate growth probabiliy distribution is not needed.)

    After initialization, the time is `t = 0`. The property `run_length_posterior` `p(r_{t}|x_{1:t})` will yield the prior,
    as no data has been observed so far.

    Observations can be provided using `new_observation()` one at a time. This method will optionally
    increment the time step `t` and the run length posterior `p(r_{t}|x_{1:t})` will be updated.

    Calling `step_forward()` will perform the step `p(r_{t}|x_{1:t+h}) -> p(r_{t+1}|x_{1:(t+h)})`, if `h > 0`.

    Combining `new_observation()` and `step_forward()` makes it possible to compute any online and offline
    run length posterior.

    If (part of) a time series `data` is already available, the convenient method `run_length_posteriors(data)` can be used
    to compute the run length posteriors for all observations in `data` at once.

    Note that both methods change state. An instance of `BOCD` cannot be reused!

    Usage:
        ```python

        data = ...    # A time series
        model = ... # A time series model

        # Specify the changepoint probabilities and the prior run length distribution, e.g.:
        cp_prob = memoryless_changepoint_probability_distribution(lmbda = 1 / 100.0)
        p_r0 = np.array([1.0])

        # Initialize BOCD
        bocd = BOCD(model, p_r0, changepoint_prob)

        # Add observations one at a time...
        for x in data:
            bocd.new_observation(x)
            time, r_posterior = bocd.run_length_posterior
            # ... do sth. with `time` and `r_posterior` ...

        # ... or all at once:
        prediction_times, observation_times, run_length_posteriors = bocd.run_length_posteriors(data)

        ```
    """

    def __init__(
        self,
        model: TimeSeriesModel,
        cp_prob: ChangepointProbabilityDistribution,
        p_r0: FloatArray | None = None,
        prob_threshold: float | None = None,
    ) -> None:
        """Ctor

        Args:
            model (TimeSeriesModel): Time series model.
            cp_prob (ChangepointProbabilityDistribution): Probability of a changepoint transition `p(r_{t}=0|r_{t-1})`
            as a function of `r_{t-1}` and the current time `t`.
            p_r0 (FloatArray): Prior run length, i.e., the probability distribution
            `[p(r_{0}=0), p(r_{0}=1), ..., p(r_{0}=T)]`
            of the initial run length at `t = 0` (should add up to 1.0, although this is not checked). Defaults to `None` which means
            that at `t = 0` there is a changepoint, i.e., `p(r_{0}=0) = 1`.
            prob_threshold (float | None, optional): Run lengths in the tail of the calculated run lengths posteriors `p(r_{t}|x_{1:t+h})`
            are discarded at each time step, if their cumulative probability falls below `prob_threshold`.
            This idea is mentioned in section 2.4 of the paper and *can* (but is not guaranteed to)
            reduce computational costs and memory consumption. Defaults to `None`, which means that no thresholding is applied. Must be in [0, 1).
            Note that if there exists an `n` such that `p(r_{t}=0|r_{t-1}=n) = 1`, the run lengths cannot grow beyond `n` and
            the implementation will automatically discard these zero-tail run lengths. I.e., the computational costs can more reliably
            be reduced by choosing a `cp_prob` function which returns `0` for all `t > n`.
        """
        if prob_threshold is not None and (
            prob_threshold < 0.0 or prob_threshold >= 1.0
        ):
            raise ValueError(
                f"`prob_threshold` must be `None` or a float in [0, 1), but {prob_threshold} was provided."
            )

        self._model: TimeSeriesModel = model
        self._cp_prob: ChangepointProbabilityDistribution = cp_prob
        self._prob_threshold: float | None = prob_threshold
        self._initialize(p_r0)

    def _initialize(self, p_r0):
        self._previous_x: float | None = None
        #  The time associated with the latest observation
        # (`t + h` in our mathematical notation).
        self._observation_time: int = 0
        # The time associated with the current prediction
        # (`t` in our mathematical notation).
        self._prediction_time: int = 0
        if p_r0 is None:
            self._run_length_posterior: FloatArray = np.array([1.0])
        else:
            self._run_length_posterior: FloatArray = p_r0.copy()
        self._prediction_time: int = 0
        self._steps: dict[int, BOCD._BOCDTimeStep] = {}
        self._steps[self._observation_time] = BOCD._BOCDTimeStep.time_step_zero(
            _log_probabilities_from_probabilities(p_r0)
        )
        self._empty_array: FloatArray = np.array([])

    def new_observation(
        self, x: float, update_posterior: bool = True, step_forward: bool = True
    ) -> None:
        """Adds an observation to the end of the time series and optionally updates the run length posterior accordingly,
        which can be queried with the property `run_length_posterior`.

        If `step_forward` is `True` (default), the current *prediction time* `t`
        is incremented, `t -> t + 1`, and the run length posterior will be updated
        to that new time step, i.e., `p(r_{t}|x_{1:(t+h)}) -> p(r_{t+1}|x_{1:(t+1+h)})`.

        If `step_forward` is `False`, the current *prediction time* `t` is not
        incremented, i.e., `t -> t`, but the look ahead time step `h` is incremented,
        i.e., `h -> h + 1` and the run length posterior at the current time `t`
        will be updated to account for the new prediction, i.e.,
        `p(r_{t}|x_{1:(t+h)}) -> p(r_{t}|x_{1:(t+h+1)})`.
        If the current time is `t = 0`, though, the run length posterior will
        remain unchanged and will still correspond to the prior.

        Here, `h` is the current number of time steps looked ahead into the future,
        which is zero for "standard" BOCD, i.e., in case this method has never been called
        with `step_forward = False`, or if `step_forward()` has at least been called as many
        times as there are future time steps.

        Args:
            x (float): The observation.
            update_posterior (bool, optional): Whether to update the run length posterior.
            The updated posterior will be available via the `run_length_posterior` property. Defaults to `True`.
            step_forward (bool, optional): Whether to increment the prediction time. Defaults to `True`.
        """

        self._observation_time += 1
        self._create_time_step_for(x)
        if step_forward:
            self.step_forward(update_posterior)
        elif update_posterior:
            self._update_run_length_posterior()

    def step_forward(self, update_posterior: bool = True) -> None:
        """Increments the prediction time, i.e., `t -> t + 1`, and optionally updates the run length posterior
        accordingly, i.e., `p(r_{t}|x_{1:(t+h)}) -> p(r_{t+1}|x_{1:(t+h)})`.

        Thus, the look ahead time step `h` is decremented, i.e., `h -> h - 1`.

        If the look ahead time step `h` is currently zero, this method has no effect.

        Args:
            update_posterior (bool, optional): Whether to update the run length posterior.
            The updated posterior will be available via the `run_length_posterior` property. Defaults to `True`.
        """
        if self._can_step_forward():
            self._steps[self._prediction_time].reset()
            self._steps.pop(self._prediction_time, None)
            self._prediction_time += 1
            if update_posterior:
                self._update_run_length_posterior()

    def _can_step_forward(self) -> bool:
        return self._has_future_observations()

    def _has_future_observations(self) -> bool:
        """Returns `True` only if there are future observations beyond
        the current prediction time (i.e., if `h > 0`), otherwise returns `False`."""
        return self._prediction_time < self._observation_time

    def _create_time_step_for(self, x: float) -> None:
        """This helper function computes the individual components
        associated with the new observation time step `t + h`
        and passes them to a new `_BOCDTimeStep` instance.

        Args:
            x (float): The observation.
        """
        self._update_time_series_model(x)
        log_pred_probs = self._log_pred_probs(x)
        log_cp_probs, log_growth_probs = self._log_transition_probs()
        self._steps[self._observation_time] = self._BOCDTimeStep(
            self._observation_time,
            self._steps[self._observation_time - 1],
            log_cp_probs=log_cp_probs,
            log_growth_probs=log_growth_probs,
            log_pred_probs=log_pred_probs,
        )

    def _update_time_series_model(self, x: float) -> None:
        """Updates the time series model for the current time step."""

        # The time series model needs to lag behind by one time step.
        if self._observation_time > 1:
            assert self._previous_x is not None
            self._model.new_observation(self._previous_x)
        self._previous_x = x

    def _log_pred_probs(self, x: float) -> FloatArray:
        """Computes the log predictive probabilities `p(x_{t}|x_{(t-r_{t}):(t-1)})`
        for observation `x = x_{t}` for `r_{t} = 0, 1, ..., t + T` using the time series model.

        Note that for brevity, we write `t` instead of `t + h` for the observation time here.

        The time series model `self._model` is currently at time `t - 1` and its `log_pred_probs()`
        method returns the log predictive probabilities for the observation `x_{t}` at current time `t`:
        `[p(x_{t}), p(x_{t}|x_{t-1}), p(x_{t}|x_{t-2}, x_{t-1}), ..., p(x_{t}|x_{1}, ..., x_{t-1})]`.

        In BOCD, we need to map these probabilities to the run length hypotheses (in BOCD notation)
        `[p(x_{t}|x_{(t-r_{t}):(t-1))]` for `r_{t} = 0, 1, ..., t`, which means that the last element
        `p(x_{t}|x_{1}, ..., x_{t-1})` repeats once, because `p(x_{t}|x_{0:(t-1)) = p(x_{t}|x_{1:(t-1))`.

        If the run length at time `t=0` (prior) can be at most `T`, then there are *additional*
        `T` repetitions of `p(x_{t}|x_{1:(t-1)})`, i.e. we have
        `p(x_{t}|x_{1:(t-1)}) = `p(x_{t}|x_{0:(t-1)}) = `p(x_{t}|x_{-1:(t-1)}) = ... = `p(x_{t}|x_{-T:(t-1)})`
        so there are `1 + T` repetitions in total.

        However, if small probabilities are discarded at any point in time (see `self._discard_small_probs()`),
        this number of repetitions reduces.

        We can compute the actual number of repetitions by checking whether the current maximum run length
        exceeds the current time step. If the current max. run length equals `t`, we have one repetition.
        Any value above `t` adds to the repetitions. If the max. run length falls below `t`, there are none.
        """
        assert self._observation_time == self._model.t + 1

        log_pred_probs = self._model.log_pred_probs(x)
        n_repetitions = max(0, self._r_t_max - self._observation_time + 1)
        log_pred_probs = np.concatenate(
            [log_pred_probs, np.repeat(log_pred_probs[-1], n_repetitions)]
        )
        return log_pred_probs

    def _log_transition_probs(self) -> tuple[FloatArray, FloatArray]:
        """Calculates the log transition (changepoint and growth) probs."""

        # Note: For brevity, we write `t` instead of `t + h` in the comments below.
        # `T` is the maximum run length of the prior.

        # The transition probs `p(r_{t}=r_{t-1}+1|r_{t-1})` ("growth probs") and
        # `p(r_{t}=0|r_{t-1})` ("changepoint probs") for `r_{t-1} = 0, 1, ..., t - 1 + T`
        # are a function of the previous run length `r_{t-1}`,
        # which at the current observation time `t` can be at most as
        # long as the maximum run length at the previous time `t - 1`.
        r_t_1_max = self._r_t_max - 1
        cp_probs = self._cp_prob(np.arange(r_t_1_max + 1), self._observation_time)
        log_cp_probs = _log_probabilities_from_probabilities(cp_probs)
        log_growth_probs = _log_probabilities_from_probabilities(1.0 - cp_probs)
        return log_cp_probs, log_growth_probs

    @property
    def _r_t_max(self) -> int:
        """The maximum possible run length at the current observation time step `t+h`, i.e.,
        `r_{t+h}` can take on values `0, 1, ..., r_t_max`.

        If the prior was `p(r_{0} = 0) = 1` (meaning a changepoint an `t = 0`), and
        no discarding of small probabilities took place, this property
        would always yield `self._observation_time`, because this is the maximum
        run length at the current time step, when the run length at
        `t = 0` was 0.
        """
        if self._observation_time in self._steps:
            return self._steps[self._observation_time].r_t_max
        else:
            return self._steps[self._observation_time - 1].r_t_max + 1

    def _update_run_length_posterior(self):
        """Calculates the run length posterior `p(r_{t}|x_{1:(t+h)})`."""
        log_joint_probs = self._steps[self._prediction_time].log_joint_probs

        if self._steps[self._prediction_time].is_prior:
            return np.exp(log_joint_probs)

        log_evidence = self._steps[self._observation_time].log_evidence

        # If we look ahead in time `h` additional time steps, i.e. if we know
        # `h` additional observations `x_{(t+1):(t+h)}`, then
        # the run length posterior we seek is actually not `p(r_{t}|x_{1:t})`,
        # but `p(r_{t}|x_{1:(t+h)})`, which we can calculate by multiplying the
        # "standard" posterior `p(r_{t}|x_{1:t})` elementwise by a matrix `M`.
        if self._has_future_observations():
            look_ahead_time_range = range(
                self._prediction_time + 1, self._observation_time + 1
            )
            log_M = reduce(
                _log_matmul_fast,
                [self._steps[t].log_M_components for t in look_ahead_time_range[1:]],
                self._steps[look_ahead_time_range[0]].log_M,
            )
            log_M = logsumexp(log_M, axis=1)
            log_joint_probs = log_joint_probs.copy()
            log_joint_probs += log_M

        self._run_length_posterior = np.exp(log_joint_probs - log_evidence)

        self._discard_small_probs()

    def _discard_small_probs(self):
        """Discards small probabilities in the run length posterior `p(r_{t}|x_{1:(t+h)})`.
        All `h` future steps need to be adapted to account for the discarded probabilities.
        In particular, if the maximum run length at time `q >= t` is set to `r_q_max`, the maximum
        run length at time `q + 1` is `r_q_max + 1`.

        We also discard the current time step `t`, because if the maximum run length at the current time `t`
        would grow again (could it?) when stepping from `p(r_{t}|x_{1:(t+h)})` to `p(r_{t}|x_{1:(t+h+1)})`,
        we had a problem as the future run lengths do no longer have the proper lengths/data
        to account for that increased maximum run length at current time `t`.
        """
        retained_length = self._highest_significant_run_length()
        if retained_length == self._steps[self._prediction_time].r_t_max:
            return
        self._run_length_posterior = self._run_length_posterior[: retained_length + 1]
        self._run_length_posterior /= self._run_length_posterior.sum()
        for i, step in enumerate(self._steps.values()):
            step.retain_run_lengths_up_to(retained_length + i)
        self._model.keep_only_latest(self._r_t_max)

    def _highest_significant_run_length(self) -> int:
        """Determines the run length (which is the same as the index of the run length posterior
        `self._run_length_posterior`) such that the total mass of the posterior run length
        greater than this run length (the tail) is less than `self._prob_threshold)`, if
        `self._prob_threshold` is not `None`. Else, determines the run length after which
        the tail is all zeros.

        Returns:
            int: The run length / index. Equals the last index of `self._run_length_posterior`,
            if no tail falls below the threshold or there is no tail of all zeros
            (i.e., if the whole posterior is 'significant').
        """
        discard_idxs = []
        if self._prob_threshold is not None:
            discard_idxs = np.nonzero(
                np.cumsum(self._run_length_posterior[::-1])[::-1] < self._prob_threshold
            )[0]
        else:
            discard_idxs = (
                np.array([self._run_length_posterior.shape[0] - 1])
                if self._run_length_posterior[-1] == 0.0
                else self._empty_array
            )
        if len(discard_idxs) > 0:
            return discard_idxs[0] - 1  # type: ignore
        else:
            return len(self._run_length_posterior) - 1

    @property
    def run_length_posterior(self) -> tuple[int, int, FloatArray]:
        """The run length posterior `p(r_{t}|x_{1:(t+h))})` at time `t`, given
        historical observations `x_{1:t)}` as well as any `h` known future observations
        `x_{(t+1):(t+h)}`, where `h = 0` for the standard BOCD.
        I.e., returns the run length posterior at time `t` given all available
        observations `x_{1:(t+h)}` for far.

        The run lengths are ordered ascending:
        `p(r_{t}|x_{1:(t+h)})` = `[p(r_{t}=0|x_{1:(t+h)}), p(r_{t}=1|x_{1:(t+h)}), ...]`.

        The posterior is an array of at most length `t + 1 + T`, where `T`
        is the maximum (prior `p(r_{0})`) run length at `t = 0`.

        At time `t = 0`, `p(r_{t}|x_{1:(t+h)})` = `p(r_{0})`, i.e., the prior passed to
        the constructor.

        Returns:
            tuple[int, FloatArray]: `(t, t + h, p(r_{t}|x_{1:(t+h)}))`
        """
        return self._prediction_time, self._observation_time, self._run_length_posterior

    def run_length_posteriors(
        self, data: FloatArray, h: int = 0
    ) -> tuple[IntArray, IntArray, FloatMatrix]:
        """Convenient method that computes the run length posteriors
        `p(r_{t} | x_{1:t+h})` at all times `t` on a batch of observations.

        The `n`th observation (1-based counting) in `data` is associated with time `t = n`.

        Args:
            data (FloatArray): The data with `N` observations to be analyzed.
            h (int): Number of future observations looked ahead at each time step.

        Returns:
            tuple[IntArray, IntArray, FloatMatrix]: A tuple with
            `(prediction_times, observation_times, run_length_posteriors)`, where
            `run_length_posteriors` is a matrix of shape `(N + 1, max_length)` where `max_length`
            is less than or equal to the maximum possible run length at the last time step `t = N`,
            which is `N + T`, where `T` is the maximum possible run length at time `t = 0`, i.e. the
            prior length.
            It is less than this value, if small probabilities have been discarded at any point in time.
            `prediction_times` = `[0, 1, ..., N]` is an array with time steps `run_length_posteriors`'s
            rows are associated with. At time `t = 0` the prior run length `p(r_{0})` can be found.
            `observation_times` = `[h, ..., N]` is an array which indicates the times `t + h` the observations
            x_{1:t+h} that were used to compute the run length posteriors.

            Note that the last `h` posteriors do not look ahead `h` steps into the future, but rather
            `h-1`, `h-2`, ..., `0` steps.
        """
        N = data.shape[0]
        r_posteriors_list = []
        prediction_times = np.zeros(N + 1, dtype=int)
        prediction_times[0], _, r_post = (
            self.run_length_posterior
        )  # this row is the prior (at t=0)
        observation_times = prediction_times.copy()
        r_posteriors_list.append(r_post)
        max_length = r_post.shape[0]

        def save(n: int):
            nonlocal max_length
            prediction_times[n], observation_times[n], r_post = (
                self.run_length_posterior
            )
            r_posteriors_list.append(r_post)
            max_length = max(max_length, r_post.shape[0])

        for n in range(h):
            self.new_observation(data[n], step_forward=False)
        for n, x in enumerate(data[h:], start=1):
            self.new_observation(x)
            save(n)
        for n in range(N - h + 1, N + 1):
            self.step_forward()
            save(n)
        r_posteriors = np.zeros((N + 1, max_length))
        for t, post in enumerate(r_posteriors_list):
            r_posteriors[t, : post.shape[0]] = post
        return prediction_times, observation_times, r_posteriors

    class _BOCDTimeStep:
        """Represents a single time step in the BOCD algorithm.
        Knows all necessary information about the current time step `t` to be able
        to compute
        * the joint probability `p(x_{t}, x_{1:t})`
        * the evidence `p(x_{1:t})`
        * the matrix `M_{t}` which contributes to the product that updates the
          posterior `p(r_{t}|x_{1:t})` to the "look-ahead" posterior `p(r_{t}|x_{1:(t+h)})`,
          if this time step is in `(t, t + h]`
        all of which can be accessed by properties of the class.

        At time `t = 0` the special instance associated with the prior run length
        can be created by the `time_step_zero()` method.
        """

        def __init__(
            self,
            t: int,
            previous_step: "BOCD._BOCDTimeStep",
            log_cp_probs: FloatArray,
            log_growth_probs: FloatArray,
            log_pred_probs: FloatArray,
        ) -> None:
            self._t: int = t  # time (for debugging)
            self._previous_step: "BOCD._BOCDTimeStep | None" = previous_step
            self._log_cp_probs: FloatArray = log_cp_probs  # `log p(r_{t}=0|r_{t-1})`
            self._log_growth_probs: FloatArray = (
                log_growth_probs  # `log p(r_{t}=r_{t-1}+1|r_{t-1})`
            )
            self._log_pred_probs: FloatArray = (
                log_pred_probs  # `log p(x_{t}|x_{(t-r_{t}):(t-1)})`
            )
            self._log_cp_component: FloatArray | None = (
                None  # Components used in algorithms...
            )
            self._log_growth_component: FloatArray | None = (
                None  # ...and cached for performance
            )
            self._log_joint_probs: FloatArray | None = None
            self._log_evidence: float | None = None
            self._log_M: FloatMatrix | None = None
            self._r_t_max: int | None = None
            self.is_prior = False

        @classmethod
        def time_step_zero(cls, log_prior: FloatArray) -> "BOCD._BOCDTimeStep":
            """Creates a special instance which corresponds to the initial step `t = 0`."""
            instance = cls.__new__(cls)
            instance._log_joint_probs = log_prior
            instance._log_evidence = 0.0
            instance.is_prior = True
            instance._t = 0
            return instance

        @property
        def log_joint_probs(self) -> FloatArray:
            """Gets the log joint probabilities `p(r_{t}, x_{1:t})` at time `t`.

            Returns:
                FloatArray: `p(r_{t}, x_{1:t})`
            """
            if self._log_joint_probs is None:
                assert self._previous_step is not None

                # log_cp_component = self._log_pred_probs[0] + self._log_cp_probs
                # log_growth_component = self._log_pred_probs[1:] + self._log_growth_probs
                log_cp_component, log_growth_component = self.log_M_components

                # Calculate growth probabilities (an array of length t + T, if
                # no discarding of small probabilities is applied)
                # p(r_{t}=r_{t-1}+1, x_{1:t}) for r_{t-1} = 0, 1, ..., t + T - 1
                log_growth_probs = (
                    log_growth_component + self._previous_step.log_joint_probs
                )

                # Calculate changepoint probability p(r_{t}=0, x_{1:t}) (a scalar)
                log_cp_prob = logsumexp(
                    log_cp_component + self._previous_step.log_joint_probs
                )

                # Update joint probabilities p(r_{t}, x_{1:t})
                self._log_joint_probs = np.append(log_cp_prob, log_growth_probs)

            return self._log_joint_probs

        @property
        def log_evidence(self) -> float:
            if self._log_evidence is None:
                self._log_evidence = cast(float, logsumexp(self.log_joint_probs))
            return self._log_evidence

        @property
        def log_M(self) -> FloatArray:
            """Log of the factor (a matrix) `M_{t} := p(x_{t}|r_{t-1}, x_{t-r_{t}:(t-1})p(r_{t}|r_{t-1})`
            which contributes to the product `M` that updates the posterior `p(r_{i}|x_{1:i})`
            to the "look-ahead" posterior `p(r_{i}|x_{1:(i+h)})`, where `i` is
            some previous time step: `i < t <= i + h`.

            The full product will elsewhere be computed by the matrix product
            `M = M_{i+1}*M_{i+2}*...*M_{i+h}`
            consisting of the factors of all future (w.r.t. time `i`) time steps
            and summing along the columns of `M`.
            The look ahead posterior `p(r_{i}|x_{1:(i+h)})` can then be computed by
            `p(r_{i}|x_{1:(i+h)}) = p(r_{i}|x_{1:i}) * sum(M, axis=1)`.

            Assuming the prior has a changepoint at time `t = 0` and so the maximum run
            length at time `t` is `t`, the matrix `M_{t}` for, e.g., `t = 3` is the
            elementwise product of the two following matrices:
            ```
            A =
            |p(r_{t}=0|r_{t-1}=0)  p(r_{t}=1|r_{t-1}=0)  0                                         |
            |p(r_{t}=0|r_{t-1}=1)  0                     p(r_{t}=2|r_{t-1}=1)  0                   |
            |p(r_{t}=0|r_{t-1}=2)  0                                           p(r_{t}=3|r_{t-1}=2)|

            B =
            |p(x_{t}=0|r_{t-1}=0)  p(x_{t}=1|r_{t-1}=0, x_{1})  0                                                            |
            |p(x_{t}=0|r_{t-1}=1)  0                            p(x_{t}=2|r_{t-1}=1, x_{1:2})  0                             |
            |p(x_{t}=0|r_{t-1}=2)  0                                                           p(x_{t}=3|r_{t-1}=2, x_{1:2}))|
            ```

            Returns:
                FloatArray: Log of the matrix `M_{t}`.
            """
            if self._log_M is None:
                log_cp_component, log_growth_component = self.log_M_components
                m = len(log_cp_component)
                log_M = np.full((m, m + 1), -np.inf)
                log_M[:, 0] = log_cp_component
                log_M[np.arange(m), np.arange(1, m + 1)] = log_growth_component
                self._log_M = log_M
            return self._log_M

        @property
        def log_M_components(self) -> tuple[FloatArray, FloatArray]:
            """Computes two vectors which are components of the log probability matrix `M_{t}`,
            which is explained in `log_M` and which at the same time are used in the BOCD algorithm
            to update the `log_joint_probs` `p(r_{t}, x_{1:t})`.

            Returns:
                tuple[FloatArray, FloatArray]: The components `v1` and `v2` of `M_{t}`,
                such that `M_{t} = [v1; diag(v2)]`, where `v1` and `v2` are column vectors
                of the same length.
            """
            if self._log_cp_component is None:
                self._log_cp_component = self._log_pred_probs[0] + self._log_cp_probs
                self._log_growth_component = (
                    self._log_pred_probs[1:] + self._log_growth_probs
                )
            return self._log_cp_component, self._log_growth_component  # type: ignore

        def retain_run_lengths_up_to(self, retained_length: int):
            """Retains only the first `retained_length + 1` run lengths and
            also cuts off other arrays such that this instance remains in
            a consistent state."""
            if -1 < retained_length < self.r_t_max:
                # Cut off tail from log joint probs log(p(r_{t}, x_{1:t}))
                assert self._log_joint_probs is not None
                self._log_joint_probs = self._log_joint_probs[: retained_length + 1]

                # Note that from `_log_cp_probs` and `_log_growth_probs`,
                # one more element is cut off as compared to the joint
                # probs above, because their length is always one less the
                # run length (see `self._log_transition_probs()`).
                self._log_cp_probs = self._log_cp_probs[:retained_length]

                # The following if-else block is meant to save computation time
                # (although the effect is potentially negligible in practice).
                if self._log_M is not None:
                    self._log_M = self._log_M[:retained_length, : retained_length + 1]
                else:
                    self._log_growth_probs = self._log_growth_probs[:retained_length]
                    self._log_pred_probs = self._log_pred_probs[: retained_length + 1]
                if self._log_cp_component is not None:
                    self._log_cp_component = self._log_cp_component[:retained_length]
                    self._log_growth_component = self._log_growth_component[  # type: ignore
                        :retained_length
                    ]

                # Force recalculation of log evidence if it is accessed elsewhere
                self._log_evidence = None

        @property
        def r_t_max(self):
            """The maximum possible run length at the current time step `t`.
            I.e., `r_{t}` can take on values `0, 1, ..., r_t_max`.
            """
            if self._log_joint_probs is not None:
                return len(self._log_joint_probs) - 1  # `t = 0`
            else:
                return len(self._log_cp_probs)  # `t > 0`

        def reset(self) -> None:
            """Frees memory."""
            self._previous_step = None


def _log_matmul_fast(
    log_A: FloatMatrix, log_B: tuple[FloatArray, FloatArray]
) -> FloatMatrix:
    """Perform matrix multiplication `A @ B` in the log-space using the log-sum-exp trick where
    matrix `B` has a particular structure and both matrices are of shape `(m, m+1)` and `(m+1, m+2)`,
    respectively.

    The log of the left `(m, m+1)` matrix `A` is given as is and might be dense.

    The right matrix `B` of shape `(m+1, m+2)` is sparse with only its first column and its first
    upper diagonal being occupied. I.e., this matrix can be represented entirely by
    specifying two vectors, `log_a := log_B[0]` and `log_b := log_B[1]`, which are the log
    of the first column and the first upper diagonal of `B`, respectively.
    E.g., if `m = 2`, we have
    ```
    B =
    |a1  b1  0   0 |
    |a2  0   b2  0 |
    |a3  0   0   b3|
    ```
    Args:
        log_A (FloatMatrix): Logarithm of matrix `A` with shape `(m, m+1)`.
        log_B (tuple[FloatArray, FloatArray]): Log of the vector components (as defined above)
        of matrix `B` with shape `(m+1, m+2)`.

    Returns:
        FloatMatrix: Logarithm of the result matrix `C = A @ B` with shape `(m, m+2)`.
    """
    log_M = np.full((log_A.shape[0], log_B[0].shape[0] + 1), -np.inf)
    log_M[:, 0] = cast(FloatArray, logsumexp(log_A + log_B[0][np.newaxis, :], axis=1))
    log_M[:, 1:] = log_A + log_B[1][np.newaxis, :]
    return log_M


def _log_probabilities_from_probabilities(probabilities: FloatArray) -> FloatArray:
    """Helper function to compute `log(probabilities)` in a numerically stable way,
    preventing numpy's `RuntimeWarning: divide by zero encountered` message.

    Parameters:
        probabilities (FloatArray): Array of probabilities to compute the log from.

    Returns:
        FloatArray: Logarithm of probabilities, with `-inf` for zero probabilities.
    """
    log_probs = np.full(probabilities.shape, -np.inf, dtype=probabilities.dtype)
    nonzero_mask = probabilities > 0.0
    log_probs[nonzero_mask] = np.log(probabilities[nonzero_mask])
    return log_probs
