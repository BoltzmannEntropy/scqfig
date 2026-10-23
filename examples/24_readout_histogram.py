"""Single-shot histograms projected on the discrimination axis, with Gaussian fits to each peak and the assignment fidelity; the tail between the peaks is T1 decay during readout (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig.plots.readout import readout_shots

TITLE = "Readout histogram"
FAMILY = "Readout"


def build():
    rng = np.random.default_rng(23)
    centres = [-0.55 + 0.30j, 0.35 + 0.62j]                       # mV
    shots = [readout_shots(m, centres, 0.16, 4000, t_ro=1.5, T1=20.0, p_th=0.01, rng=rng) for m in (0, 1)]
    fig, ax = sq.figure("column", aspect=0.7)
    sq.plots.readout_histogram(ax, shots, log=True)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
