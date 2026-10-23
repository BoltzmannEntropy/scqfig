"""Two-qubit tomography of the Bell state (|00> + |11>)/sqrt(2): real and imaginary parts of the reconstructed density matrix (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Density matrix"
FAMILY = "States"


def noisy_bell(seed=42):
    """A Bell state with depolarising, dephasing and tomography noise, projected onto a physical state."""
    rng = np.random.default_rng(seed)
    bell = ph.dm(np.array([1, 0, 0, 1]) / np.sqrt(2))
    rho = 0.92 * bell + 0.08 * np.eye(4) / 4
    rho[0, 3] *= np.exp(0.12j)
    rho[3, 0] = rho[0, 3].conj()
    H = rng.normal(0, 0.012, (4, 4)) + 1j * rng.normal(0, 0.012, (4, 4))
    rho = rho + (H + H.conj().T) / 2
    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    rho = (V * w) @ V.conj().T
    return rho / np.trace(rho).real, bell


def build():
    rho, _ = noisy_bell()
    fig, axes = sq.figure("column", aspect=0.52, ncols=2)
    sq.plots.density_matrix(axes, rho)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
