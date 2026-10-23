"""Flux-noise spectral density with a power-law fit, amplitude at 1 Hz and a white floor (synthetic data)."""
import numpy as np

import scqfig as sq

TITLE = "Noise spectral density"
FAMILY = "Noise and errors"


def build():
    rng = np.random.default_rng(50)
    f = np.geomspace(1e-4, 10, 70)
    S = 2.2 ** 2 / f ** 0.88 + 0.6
    S = S * np.exp(rng.normal(0, 0.12, f.size))
    fig, ax = sq.figure("column", aspect=0.72)
    sq.plots.noise_psd(ax, f, S, fit_range=(1e-4, 0.1), white=0.6)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
