"""Dispersive shift of a transmon against its frequency for a 7.15 GHz resonator, g = 85 MHz and alpha = -233 MHz, with the straddling regime shaded and the design point marked (computed)."""
import numpy as np

import scqfig as sq

TITLE = "Dispersive shift"
FAMILY = "Design"


def build():
    fq = np.linspace(4.0, 9.0, 4001)
    fig, ax = sq.figure("column", aspect=0.66)
    sq.plots.dispersive_shift(ax, fq, fr=7.15, g=0.085, alpha=-0.233, mark=5.06)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
