"""Line shape of a 5 GHz dark-matter wave for isotropic Maxwell speeds with v0 = 220 km/s: the line is a few kHz wide, one part in a million of its frequency, with a mean offset of 2.0 kHz (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Dark-matter line shape"
FAMILY = "Dark matter"


def build():
    df = np.linspace(0, 8.0, 400)                                    # kHz
    g = ph.dm_lineshape(df, 5e6)                                     # f in kHz
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.lineshape(ax, df, g, mean=float((g * df).sum() / g.sum()))
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
