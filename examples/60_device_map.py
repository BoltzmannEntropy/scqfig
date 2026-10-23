"""Device map of a 3 x 4 qubit grid with T1 on the qubits and CZ error on the couplers (synthetic data)."""
import numpy as np

import scqfig as sq

TITLE = "Device map"
FAMILY = "Processor"


def build():
    rng = np.random.default_rng(60)
    rows, cols = 3, 4
    pos = [(c, -r) for r in range(rows) for c in range(cols)]
    edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    T1 = rng.lognormal(np.log(72), 0.25, len(pos))
    cz = 100 * rng.lognormal(np.log(6e-3), 0.35, len(edges))
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.device_map(ax, pos, edges, T1, cz, names=[f"Q{k + 1}" for k in range(len(pos))])
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
