"""Pairwise correlation of error detections across a 24-qubit chain; bursts correlate neighbours (synthetic data from a toy burst model)."""
import numpy as np

import scqfig as sq
from scqfig.plots.noise import burst_events

TITLE = "Error correlations"
FAMILY = "Noise and errors"


def build():
    rng = np.random.default_rng(53)
    bursts = list(zip(rng.integers(0, 20000, 40), rng.integers(0, 24, 40)))
    t, E = burst_events(nq=24, nt=20000, p_bg=0.02, bursts=bursts, tau=25.0, rng=53)
    fig, ax = sq.figure("column", aspect=0.86)
    sq.plots.error_correlation(ax, E)
    ax.set_xlabel("qubit")
    ax.set_ylabel("qubit")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
