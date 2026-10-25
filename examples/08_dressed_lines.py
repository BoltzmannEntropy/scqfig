"""Dressed one-excitation frequencies of a transmon (E_C = 250 MHz) and a 7.5 GHz resonator coupled with g = 100 MHz, swept through the crossing, with the splitting and a design point marked (computed)."""
import numpy as np

import scqfig as sq

TITLE = "Dressed lines"
FAMILY = "Spectrum"


def build():
    fq = np.linspace(5.0, 10.0, 161)
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.dressed_lines(ax, fq, fr=7.5, g=0.1, EC=0.25, mark=6.0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
