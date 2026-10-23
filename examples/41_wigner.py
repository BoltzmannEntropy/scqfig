"""Wigner functions of (a) the Fock state |1> and (b) an even cat state with alpha = 2, on one colour scale of +-2/pi (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Wigner functions"
FAMILY = "States"


def build():
    N = 40
    x = np.linspace(-3.5, 3.5, 241)
    fig, axes = sq.figure("column", aspect=0.39, ncols=2)
    sq.plots.wigner(axes[0], ph.fock(1, N), xvec=x, vmax=2 / np.pi, cbar=False)
    sq.plots.wigner(axes[1], ph.cat(2.0, N), xvec=x, vmax=2 / np.pi)
    axes[1].set_ylabel("")
    for ax, k in zip(axes, "ab"):
        sq.panel(ax, k)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
