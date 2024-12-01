"""
Implementations of time series models to be used in the BOCD algorithm.
See `bocd.TimeSeriesModel` for details.
"""

import numpy as np
from scipy.stats import norm

from .bocd import TimeSeriesModel
from .bocd_types import FloatArray


class GaussianUnknownMean(TimeSeriesModel):
    """A time series model with iid observations and unknown mean and known observation variance.

    The observations `x` are assumed to follow an iid Gaussian likelihood `p(x|mu, varx)`, with known variance `varx`.
    The prior for `mu` is a Gaussian prior with `mean0` and `var0`: `p(mu|mean0, var0)`.

    This gives rise to a conjugate prior model whose posterior predictive probabilities
    `p(x_{t+1}|x_{t-r}, x_{t-r+1}, ..., x_{t}) = N(x_{t+1}|m, v)`
    can be computed efficiently.
    """

    def __init__(self, mean0: float, var0: float, varx: float) -> None:
        """Ctor

        Args:
            mean0 (float): Mean of the Gaussian prior for the observation mean.
            var0 (float): Variance of the Gaussian prior for the observation mean.
            varx (float): Variance of each observation.
        """
        super().__init__()
        self._mean0: float = mean0
        self._var0: float = var0
        self._varx: float = varx
        self._initialize()

    def _initialize(self) -> None:
        self._delta_nu: float = 1
        self._nu0: FloatArray = np.array([self._varx / self._var0])
        self._nu: FloatArray = np.array(self._nu0)
        self._chi0: FloatArray = np.array([self._mean0 / self._var0])
        self._chi: FloatArray = np.array(self._chi0)
        self._update_params()

    def _update_params(self) -> None:
        _vars = self._varx / self._nu
        self._means = self._chi * _vars
        self._stds = np.sqrt(_vars + self._varx)

    def log_pred_probs(self, x: float) -> FloatArray:
        return norm(self._means, self._stds).logpdf(x)  # type: ignore

    def _new_observation(self, x: float) -> None:
        self._nu = np.concatenate(
            (
                self._nu0,
                self._nu0 + self._delta_nu,
                self._nu[1:] + self._delta_nu,
            )
        )
        self._chi = np.concatenate(
            (
                self._chi0,
                self._chi0 + x / self._varx,
                self._chi[1:] + x / self._varx,
            )
        )
        self._update_params()

    def _keep_only_latest(self, n_observations: int) -> None:
        self._nu = self._nu[: n_observations + 1]
        self._chi = self._chi[: n_observations + 1]
        self._update_params()
