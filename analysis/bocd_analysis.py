"""
Utility functions for analyzing and plotting results from the Bayesian online changepoint detection algorithm.
"""

from enum import Enum, auto
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from bocd.bocd_types import FloatArray, FloatMatrix, IntArray

# plt.style.use("dark_background")


class PlotType(Enum):
    """Defines how BOCD results are visualised."""

    CHANGEPOINT = auto()
    TOP_RUN_LENGTH = auto()
    EXPECTED_RUN_LENGTH = auto()
    WINDOWED = auto()


def plot_bocd(
    data: FloatArray,
    run_length_posteriors: FloatMatrix | None = None,
    t0: int = 1,
    cps: IntArray | None = None,
    cp_ticks: bool = False,
    type: PlotType = PlotType.CHANGEPOINT,
) -> tuple[ModuleType, list[Axes]] | None:
    """Visualizes the results of a Bayesian online changepoint detection algorithm and/or the data.

    Args:
        data (FloatArray): Time series the BOCD algorithm was applied to.
        run_length_posteriors (FloatMatrix | None, optional): Run length posteriors `p(r_{t}|x_{1:t+h})` in each row, starting
        at `t = t0 - 1` (the point in time associated with the prior). Must have one row more than `data` has elements.
        t0 (int, optional): The point in time associated with the first observation. Defaults to 1.
        Must have as many rows as `t` has elements. If not provided, only the data will be visualized.
        cps (IntArray | None, optional): Changepoints (if known in case the data was simulated).
        Changepoints are visualized as vertical lines. Defaults to None.
        cp_ticks (bool, optional): If `True`, changepoints are visualized with tick labels. Defaults to False.
        type (PlotType, optional): How to visualize the results. Defaults to `PlotType.CHANGEPOINT`.
    """

    interactive_mode = plt.isinteractive()
    plt.ioff()  # Turn off interactive mode

    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(311)
    axs = [ax1]

    t = np.arange(t0 - 1, len(data) + t0)

    x_lim = (t[0], t[-1])

    ax1.plot(t[1:], data, marker="o")
    ax1.margins(0)

    # Run length posteriors
    if run_length_posteriors is not None:
        ax2 = plt.subplot(312, sharex=ax1)
        axs.append(ax2)
        ax2.imshow(
            np.flipud(np.rot90(run_length_posteriors)),
            aspect="auto",
            cmap="gray_r",
            norm=LogNorm(vmin=0.0001, vmax=1),
            origin="lower",
            extent=(
                float(t[0]),
                float(t[-1]),
                float(0),
                float(run_length_posteriors.shape[1]),
            ),
        )
        ax2.set_xlim(x_lim)
        ax2.margins(0)
        ax2.set_ylabel("Run length posteriors")

        # Plot changepoints
        ax3 = plt.subplot(313, sharex=ax1)
        axs.append(ax3)
        if type == PlotType.CHANGEPOINT:
            ax3.plot(t, run_length_posteriors[:, 0] * 100)
            ax3.set_ylabel("Changepoint probability [%]")
            ax3.set_ylim((0, 100))
        elif type == PlotType.TOP_RUN_LENGTH:
            ax3.plot(t, np.argmax(run_length_posteriors, axis=1))
            ax3.set_ylim((0, ax3.get_ylim()[1]))
            ax3.set_ylabel("Most probable run length")
        elif type == PlotType.EXPECTED_RUN_LENGTH:
            ax3.plot(
                t,
                np.sum(
                    run_length_posteriors
                    * np.arange(run_length_posteriors.shape[1])[np.newaxis, :],
                    axis=1,
                ),
            )
            ax3.set_ylim((0, ax3.get_ylim()[1]))
            ax3.set_ylabel("Expected run length")
        elif type == PlotType.WINDOWED:
            cp_prob = run_length_posteriors[:, 0]
            window_size = 5
            local_mean_changepoint_prob = np.convolve(
                cp_prob, np.ones(window_size) / window_size, mode="same"
            )
            ax3.plot(
                t,
                cp_prob / local_mean_changepoint_prob,
            )
            ax3.set_ylim((0, ax3.get_ylim()[1]))
            ax3.set_ylabel("Windowed run length")

    if cps is not None:
        for cp in cps:
            [ax.axvline(cp, c="green", ls="dotted") for ax in axs]
        if cp_ticks:
            ax = axs[-1]
            xticks = sorted(set(ax.get_xticks()).union(cps))
            ax.set_xticks(xticks)
            for label in ax.get_xticklabels():
                tick_val = float(label.get_text())
                if tick_val in cps:
                    label.set_rotation(45)
                    label.set_color("green")
                    label.set_y(-0.05)
        if len(axs) > 1:
            [
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                for ax in axs[:-1]
            ]

    axs[-1].set_xlabel("Time")
    axs[0].set_xlim(x_lim)

    for ax in axs:
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")

    plt.tight_layout()

    if interactive_mode:
        plt.ion()  # Re-enable interactive mode if it was originally on

    return plt, axs
