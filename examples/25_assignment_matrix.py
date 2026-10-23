"""Three-state assignment matrix from 5000 simulated shots per prepared state, assigned to the nearest IQ centre; decay during readout fills the lower triangle (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig.plots.readout import readout_shots

TITLE = "Assignment matrix"
FAMILY = "Readout"


def build():
    rng = np.random.default_rng(25)
    centres = np.array([-0.55 + 0.30j, 0.35 + 0.62j, 0.62 - 0.28j])  # mV
    M = np.zeros((3, 3))
    for m in range(3):
        s = readout_shots(m, centres, 0.16, 5000, t_ro=1.5, T1=20.0, p_th=0.01, rng=rng)
        got = np.argmin(np.abs(s[:, None] - centres[None, :]), axis=1)
        M[m] = np.bincount(got, minlength=3)
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.assignment_matrix(ax, M)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
