"""Ramsey envelope, Hahn echo and CPMG-16 decays of one qubit on one axis, each with its exponential fit and time constant (synthetic data)."""
import numpy as np

import scqfig as sq

TITLE = "Coherence decays"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(34)
    curves = {}
    for name, T, tmax in ((r"Ramsey $T_2^*$", 18.0, 80), (r"echo $T_{2E}$", 52.0, 200),
                          (r"CPMG-16 $T_{2}$", 118.0, 400)):
        t = np.linspace(0, tmax, 26)
        curves[name] = (t, np.exp(-t / T) + rng.normal(0, 0.015, t.size))
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.coherence_decays(ax, curves)
    ax.set_ylim(-0.05, 1.05)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
