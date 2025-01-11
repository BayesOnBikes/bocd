import numpy as np
import pytest

from bocd.bocd import BOCD, ChangepointProbabilityDistribution, TimeSeriesModel
from bocd.bocd_types import FloatArray, IntArray
from bocd.bocd_utils import (
    changepoint_probability_distribution_from_gap_probability_distribution,
)


class FakeTimeSeriesModel(TimeSeriesModel):
    def __init__(self, priors: dict[int, float] | None = None) -> None:
        """Ctor

        Args:
            priors (dict[int, float] | None, optional): An optional dictionary
            mapping `t` to the prior probability `p(x_{t})`, which is
            the predictive probability given t is a changepoint (= no historical data
            taken into account).
        """
        super().__init__()
        self._priors = priors if priors is not None else {}

    def log_pred_probs(self, _: float) -> FloatArray:
        log_p = np.log(np.random.rand(self._n_observations + 1))
        # Note that the time series model models p(x_{t+1}|...),
        # so, as we are interested in t+1 being our changepoint,
        # we have to add 1 to the time step.
        if self.t + 1 in self._priors:
            log_p[0] = (
                np.log(self._priors[self.t + 1])
                if self._priors[self.t + 1] > 0.0
                else -np.inf
            )
        return log_p

    def _new_observation(self, _: float) -> None: ...

    def _keep_only_latest(self, _: int) -> None: ...


@pytest.fixture
def p_r0() -> FloatArray:
    """A plausible, but arbitrary prior p(r_{0})."""
    return np.array([0.25, 0.25, 0.0, 0.3, 0.2], dtype=np.float64)


@pytest.fixture
def changepoint_prob() -> ChangepointProbabilityDistribution:
    """A plausible, but arbitrary changepoint probability distribution p(r_{t}=0|r_{t-1})."""
    gap_probs = np.array([0.1, 0, 0.3, 0.2, 0.1, 0.3], dtype=np.float64)
    return changepoint_probability_distribution_from_gap_probability_distribution(
        gap_probs
    )


@pytest.fixture(
    params=[np.random.randn(20), np.full(40, 42.0)],
    ids=["random data", "constant data"],
)
def data(request: pytest.FixtureRequest) -> FloatArray:
    return request.param


@pytest.fixture(params=[0, 1, 2], ids=lambda h: f"h={h}")
def h(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[None, 1e-4, 1e-2, 0.1, 0.99], ids=lambda p: f"p_thresh={p}")
def probability_threshold(request: pytest.FixtureRequest) -> float | None:
    """Tail probability to discard"""
    return request.param


@pytest.fixture
def model() -> TimeSeriesModel:
    return FakeTimeSeriesModel()


class TestFakeModel:
    def test_length_of_log_pred_probs(self, model: TimeSeriesModel) -> None:
        # given
        t_max = 42
        for _ in range(t_max):
            model.new_observation(np.random.randn())

        # when
        p = model.log_pred_probs(np.random.randn())

        # then
        assert len(p) == t_max + 1

    def test_length_of_log_pred_probs_after_keep_only_latest(
        self, model: TimeSeriesModel
    ) -> None:
        # given
        t_max = 42
        for _ in range(t_max):
            model.new_observation(np.random.randn())

        # when
        model.keep_only_latest(10)
        p = model.log_pred_probs(np.random.randn())

        # then
        assert len(p) == 10 + 1


class TestNormalCases:
    def test_when_cp_transition_probability_increases_then_the_cp_posterior_increases(
        self,
        data: FloatArray,
        h: int,
        p_r0: FloatArray,
        probability_threshold: float | None,
    ) -> None:
        """
        Plausibility test: Increasing p(r_{t}=0|r_{t-1}) is expected to
        increase the posterior p(r_{t}=0|x_{1:t}) (all other values
        and parameters held constant).
        """

        # given
        t_cp = 5  # this will be the changepoint we are interested in

        def changepoint_prob_factory(
            cp_prob: float,
        ) -> ChangepointProbabilityDistribution:
            def changepoint_prob(r_t_1: IntArray, t: int) -> FloatArray:
                if t == t_cp:
                    return np.full(len(r_t_1), cp_prob)
                else:
                    return np.full(len(r_t_1), 0.123)

            return changepoint_prob

        increasing_cp_probs = np.linspace(0.0, 1.0, 10)

        previous_cp_posterior = -np.inf
        for cp_prob in increasing_cp_probs:
            # Important to reset seed to make sure the model
            # sees the same data on each iteration.
            np.random.seed(43)

            changepoint_prob = changepoint_prob_factory(cp_prob)
            sut = BOCD(
                FakeTimeSeriesModel(),
                changepoint_prob,
                p_r0,
                prob_threshold=probability_threshold,
            )

            # when
            _, _, run_length_posteriors = sut.run_length_posteriors(data, h)
            next_cp_posterior = run_length_posteriors[t_cp, 0]

            # then
            assert previous_cp_posterior <= next_cp_posterior
            assert np.allclose(run_length_posteriors.sum(axis=1), 1.0), (
                "Not a probability distribution"
            )

            previous_cp_posterior = next_cp_posterior

    def test_when_predictive_probability_increases_then_the_cp_posterior_increases(
        self,
        data: FloatArray,
        h: int,
        p_r0: FloatArray,
        changepoint_prob: ChangepointProbabilityDistribution,
        probability_threshold: float | None,
    ) -> None:
        """
        Plausibility test: Increasing p(x_{t}|x_{(t-r_{t}):(t-1)}) is expected to
        increase the posterior p(r_{t}=0|x_{1:t}) (all other values
        and parameters held constant).
        """

        # given
        p_r0 = np.array([1], dtype=np.float64)

        t_cp = 5  # this will be the changepoint we are interested in

        increasing_pred_probs = np.linspace(0.0, 1.0, 10)

        previous_cp_posterior = -np.inf
        for cp_prob in increasing_pred_probs:
            # Important to reset seed to make sure the model
            # sees the same data on each iteration.
            np.random.seed(43)

            sut = BOCD(
                FakeTimeSeriesModel({t_cp: cp_prob}),
                changepoint_prob,
                p_r0,
                prob_threshold=probability_threshold,
            )

            # when
            _, _, run_length_posteriors = sut.run_length_posteriors(data, h)
            next_cp_posterior = run_length_posteriors[t_cp, 0]

            # then
            assert previous_cp_posterior < next_cp_posterior
            assert np.allclose(run_length_posteriors.sum(axis=1), 1.0), (
                "Not a probability distribution"
            )

            previous_cp_posterior = next_cp_posterior


class TestExtremeCases:
    def test_alternating_changepoints(
        self,
        model: TimeSeriesModel,
        data: FloatArray,
        h: int,
        probability_threshold: float | None,
    ) -> None:
        """
        Edge case where the prior and the changepoint probability distributions
        are set up such that there are alternating changepoints at 1, 6, 11, 16, ..., 1 + 5n.
        """

        # given
        p_r0 = np.array([0, 0, 0, 0, 1], dtype=np.float64)
        gap_probs = np.array([0, 0, 0, 0, 1], dtype=np.float64)
        changepoint_prob = (
            changepoint_probability_distribution_from_gap_probability_distribution(
                gap_probs
            )
        )

        sut = BOCD(model, changepoint_prob, p_r0, prob_threshold=probability_threshold)

        # when
        _, _, run_length_posteriors = sut.run_length_posteriors(data, h)

        # then
        cp_mask = 1 + 5 * np.arange(run_length_posteriors.shape[0] // 5)
        no_cp_mask = np.arange(run_length_posteriors.shape[0])
        no_cp_mask[cp_mask] = False
        assert run_length_posteriors[cp_mask, 0] == pytest.approx(1.0)
        assert run_length_posteriors[no_cp_mask, 0] == pytest.approx(0.0)

    def test_no_changepoints(
        self,
        model: TimeSeriesModel,
        data: FloatArray,
        h: int,
        probability_threshold: float | None,
    ) -> None:
        """
        Edge case where the prior and the changepoint probability distributions
        are set up such that there can never be any changepoint.
        """

        # given
        p_r0 = np.array([0, 0, 1], dtype=np.float64)
        gap_probs = np.array([1], dtype=np.float64)
        changepoint_prob = (
            changepoint_probability_distribution_from_gap_probability_distribution(
                gap_probs
            )
        )

        sut = BOCD(model, changepoint_prob, p_r0, prob_threshold=probability_threshold)

        # when
        _, _, run_length_posteriors = sut.run_length_posteriors(data, h)

        # then
        assert run_length_posteriors[:, 0] == pytest.approx(0.0)

    def test_all_changepoints(
        self,
        model: TimeSeriesModel,
        data: FloatArray,
        h: int,
        probability_threshold: float | None,
    ) -> None:
        """
        Edge case where the prior and the changepoint probability distributions
        are set up such that there are only changepoints.
        """

        # given
        p_r0 = np.array([1], dtype=np.float64)
        gap_probs = np.array([1], dtype=np.float64)
        changepoint_prob = (
            changepoint_probability_distribution_from_gap_probability_distribution(
                gap_probs
            )
        )

        sut = BOCD(model, changepoint_prob, p_r0, prob_threshold=probability_threshold)

        # when
        _, _, run_length_posteriors = sut.run_length_posteriors(data, h)

        # then
        assert run_length_posteriors[:, 0] == pytest.approx(1.0)


class TestErrorCases:
    def test_negative_h(
        self,
        model: TimeSeriesModel,
        data: FloatArray,
        p_r0: FloatArray,
        changepoint_prob: ChangepointProbabilityDistribution,
    ) -> None:
        sut = BOCD(model, changepoint_prob, p_r0)
        with pytest.raises(ValueError) as _:
            _, _, _ = sut.run_length_posteriors(data, -1)

    def test_invalid_prob_threshold(
        self,
        model: TimeSeriesModel,
        p_r0: FloatArray,
        changepoint_prob: ChangepointProbabilityDistribution,
    ) -> None:
        negative_prob_threshold = -42.0
        with pytest.raises(ValueError) as _:
            BOCD(model, changepoint_prob, p_r0, prob_threshold=negative_prob_threshold)

        too_large_prob_threshold = 42.0
        with pytest.raises(ValueError) as _:
            BOCD(model, changepoint_prob, p_r0, prob_threshold=too_large_prob_threshold)
