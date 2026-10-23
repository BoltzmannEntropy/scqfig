"""Purcell limit on T1 against qubit frequency for a 7.15 GHz readout resonator (kappa = 1.5 MHz, g = 85 MHz), without and with a Q = 25 bandpass filter (computed)."""
import numpy as np

import scqfig as sq

TITLE = "Purcell limit"
FAMILY = "Design"


def build():
    fq = np.linspace(4.0, 7.0, 600)
    fig, ax = sq.figure("column", aspect=0.66)
    sq.plots.purcell_limit(ax, fq, fr=7.15, g=0.085, kappa=0.0015, filter_q=25, mark=5.06, target=500)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
