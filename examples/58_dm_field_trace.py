"""Amplitude and phase of a 5 GHz dark-matter wave built from 400 tones with standard-halo velocities: both wander on the 200 µs coherence time, twice the length of the shaded 100 µs qubit shot (simulated, seeded)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Dark-matter field trace"
FAMILY = "Dark matter"


def build():
    t = np.linspace(0, 3000, 1501)                                   # µs
    A = ph.dm_field_trace(t, 5.0, seed=7)
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.field_trace(ax, t, A, window=(1900, 2000), window_label="100 µs shot")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
