"""Real part of the reconstructed Bell-state density matrix as 3D bars, with the ideal state as wireframe outlines (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Density matrix in 3D"
FAMILY = "States"


def build():
    rng = np.random.default_rng(43)
    bell = ph.dm(np.array([1, 0, 0, 1]) / np.sqrt(2))
    rho = 0.92 * bell + 0.08 * np.eye(4) / 4             # depolarising noise
    H = rng.normal(0, 0.012, (4, 4)) + 1j * rng.normal(0, 0.012, (4, 4))
    w, V = np.linalg.eigh(rho + (H + H.conj().T) / 2)   # tomography noise, projected onto a physical state
    rho = (V * np.clip(w, 0, None)) @ V.conj().T
    rho /= np.trace(rho).real
    fig, ax = sq.figure("column", aspect=0.8, subplot_kw={"projection": "3d"})
    sq.plots.density_bars3d(ax, rho, ideal=bell)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
