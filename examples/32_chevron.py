"""Rabi chevron: excited population against drive detuning and pulse duration at a 10 MHz Rabi rate, with the first two population maxima of the model drawn over the left half (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Rabi chevron"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(32)
    det = np.linspace(-30, 30, 121)                      # MHz
    dur = np.linspace(0, 250, 126)                       # ns
    D, T = np.meshgrid(det, dur)
    P = 0.03 + 0.92 * ph.rabi_population(0.010, D * 1e-3, T) + rng.normal(0, 0.03, D.shape)
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.chevron(ax, det, dur, P, rabi=10.0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
