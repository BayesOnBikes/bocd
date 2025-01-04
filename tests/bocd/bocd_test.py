import numpy as np
import pytest

from bocd.bocd import (
    BOCD,
    TimeSeriesModel,
)
from bocd.bocd_types import FloatArray
from bocd.bocd_utils import (
    changepoint_probability_distribution_from_gap_probability_distribution,
)


class FakeTimeSeriesModel(TimeSeriesModel):
    def log_pred_probs(
        self, _: float
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.random.rand(self._n_observations + 1)

    def _new_observation(self, _: float) -> None: ...

    def _keep_only_latest(self, _: int) -> None: ...


@pytest.fixture(params=[np.random.randn(20), np.full(40, 42.0)])
def data(request: pytest.FixtureRequest) -> FloatArray:
    return request.param


@pytest.fixture(params=[0, 1, 2])
def h(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def model() -> TimeSeriesModel:
    return FakeTimeSeriesModel()


class TestExtremeCases:
    def test_alternating_changepoints(
        self, model: TimeSeriesModel, data: FloatArray, h: int
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

        sut = BOCD(model, changepoint_prob, p_r0)

        # when
        _, _, run_length_posteriors = sut.run_length_posteriors(data, h)

        # then
        cp_mask = 1 + 5 * np.arange(run_length_posteriors.shape[0] // 5)
        no_cp_mask = np.arange(run_length_posteriors.shape[0])
        no_cp_mask[cp_mask] = False
        assert run_length_posteriors[cp_mask, 0] == pytest.approx(1.0)
        assert run_length_posteriors[no_cp_mask, 0] == pytest.approx(0.0)

    def test_no_changepoints(
        self, model: TimeSeriesModel, data: FloatArray, h: int
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

        sut = BOCD(model, changepoint_prob, p_r0)

        # when
        _, _, run_length_posteriors = sut.run_length_posteriors(data, h)

        # then
        assert run_length_posteriors[:, 0] == pytest.approx(0.0)

    def test_all_changepoints(
        self, model: TimeSeriesModel, data: FloatArray, h: int
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

        sut = BOCD(model, changepoint_prob, p_r0)

        # when
        _, _, run_length_posteriors = sut.run_length_posteriors(data, h)

        # then
        assert run_length_posteriors[:, 0] == pytest.approx(1.0)
