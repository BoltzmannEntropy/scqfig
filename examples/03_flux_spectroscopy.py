"""Two-tone spectroscopy of a SQUID transmon against flux, with the exact f01 and f02/2 drawn over the data (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Flux-tuned spectroscopy"
FAMILY = "Spectrum"


def build():
    rng = np.random.default_rng(3)
    flux = np.linspace(-0.6, 0.6, 161)
    freq = np.linspace(3.2, 6.4, 241)
    f = ph.tunable_frequencies(flux, EJmax=18.0, EC=0.22, d=0.35, nlev=3)
    f01, f02h = f[:, 0], f[:, 1] / 2
    F, X = np.meshgrid(freq, flux, indexing="ij")
    line = lambda f0, w, A: A / (1 + ((F - f0[None, :]) / w) ** 2)
    S = line(f01, 0.02, 1.0) + line(f02h, 0.008, 0.45) + rng.normal(0, 0.04, F.shape)
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.flux_spectroscopy(ax, flux, freq, S, model={r"$f_{01}$": f01, r"$f_{02}/2$": f02h})
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
