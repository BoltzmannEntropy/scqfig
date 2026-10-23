"""Bloch sphere: (a) a detuned Rabi drive (Delta = 0.6 Omega) tilts the precession cone off the pole; (b) free precession from |+> spirals to |0> as T2 and T1 shrink the vector (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph
from scqfig.plots.states import bloch_decay

TITLE = "Bloch sphere"
FAMILY = "States"


def build():
    fig, axes = sq.figure("column", aspect=0.56, ncols=2, subplot_kw={"projection": "3d"})
    t = np.linspace(0, 1 / np.hypot(1.0, 0.6), 200)     # one generalised Rabi period
    rabi = ph.rabi_trajectory(1.0, 0.6, t)
    sq.plots.bloch_sphere(axes[0], [rabi[70]], trajectories=[rabi])
    ts = np.linspace(0, 6, 900)                          # µs
    spiral = bloch_decay((1, 0, 0), 1.0, ts, T1=4.0, T2=2.0)
    sq.plots.bloch_sphere(axes[1], [spiral[0], spiral[-1]], trajectories=[spiral])
    for ax, k in zip(axes, "ab"):
        sq.panel(ax, k)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
