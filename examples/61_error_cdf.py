"""Cumulative distributions of single-qubit, CZ and readout errors across a 54-qubit processor, medians marked (synthetic data)."""
import numpy as np

import scqfig as sq

TITLE = "Error distributions"
FAMILY = "Processor"


def build():
    rng = np.random.default_rng(61)
    data = {"1Q (RB)": 10 ** rng.normal(-3.0, 0.18, 54),
            "2Q CZ (XEB)": 10 ** rng.normal(-2.2, 0.2, 88),
            "readout": 10 ** rng.normal(-1.8, 0.22, 54)}
    fig, ax = sq.figure("column", aspect=0.68)
    sq.plots.cdf(ax, data)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
