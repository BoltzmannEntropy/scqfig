"""Maxwell capacitance matrix of two transmons, their readout resonators and the feedline, coloured by |C| on a log scale (simulated values)."""
import numpy as np

import scqfig as sq

TITLE = "Capacitance matrix"
FAMILY = "Design"


def build():
    nets = ["Q1", "Q2", "R1", "R2", "feed"]
    ground = np.array([78.4, 80.1, 412.0, 405.0, 690.0])           # fF to ground
    mutual = np.zeros((5, 5))                                      # fF between nets
    for (i, j), c in {(0, 1): 0.48, (0, 2): 4.62, (1, 3): 4.41, (0, 3): 0.061, (1, 2): 0.057, (2, 3): 0.12,
                      (2, 4): 11.3, (3, 4): 10.8, (0, 4): 0.21, (1, 4): 0.19}.items():
        mutual[i, j] = mutual[j, i] = c
    C = np.diag(ground + mutual.sum(1)) - mutual                   # Maxwell form
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.capacitance_matrix(ax, C, nets)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
