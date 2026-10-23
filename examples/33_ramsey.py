"""Ramsey fringes at 0.8 MHz detuning with a damped-cosine fit and its envelope (synthetic data, T2* = 4.2 us)."""
import numpy as np

import scqfig as sq

TITLE = "Ramsey fringes"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(33)
    t = np.linspace(0, 12, 121)
    p = 0.5 + 0.46 * np.exp(-t / 4.2) * np.cos(2 * np.pi * 0.8 * t) + rng.normal(0, 0.015, t.size)
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.ramsey(ax, t, p)
    ax.set_ylim(0, 1.0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
