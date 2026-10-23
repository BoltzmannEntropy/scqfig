"""Eigenmode frequency of a readout resonator against adaptive mesh pass, with the relative change per pass and a 0.02 % target (synthetic data)."""
import numpy as np

import scqfig as sq

TITLE = "Convergence"
FAMILY = "Design"


def build():
    rng = np.random.default_rng(16)
    n = np.arange(1, 13)
    f = 7.1512 + 0.31 * np.exp(-(n - 1) / 1.45) * (1 + rng.normal(0, 0.06, n.size))
    fig, ax = sq.figure("column", aspect=0.66)
    sq.plots.convergence(ax, n, f, label=r"eigenfrequency $f_r$ (GHz)", tol=2e-4)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
