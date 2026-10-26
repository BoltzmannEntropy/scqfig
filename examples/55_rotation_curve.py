"""Rotation curve of a model spiral galaxy: the speed that the visible matter alone would give falls off outside the disk, while a dark halo keeps the total flat near 220 km/s (illustrative model, synthetic points)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Rotation curve"
FAMILY = "Dark matter"


def build():
    rng = np.random.default_rng(55)
    r = np.linspace(0.2, 30, 300)                                   # kpc
    vis, halo, _ = ph.rotation_curve(r)
    ro = np.linspace(2, 28, 14)
    vo = ph.rotation_curve(ro)[2] + rng.normal(0, 6, ro.size)
    fig, ax = sq.figure("column", aspect=0.68)
    sq.plots.rotation_curve(ax, r, vis, halo, observed=(ro, vo, np.full(ro.size, 8.0)))
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
