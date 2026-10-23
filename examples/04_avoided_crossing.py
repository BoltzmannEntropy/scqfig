"""Vacuum-Rabi splitting: resonator transmission as a tunable qubit crosses it, with the bare modes dashed and 2g marked (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Avoided crossing"
FAMILY = "Spectrum"


def build():
    rng = np.random.default_rng(4)
    flux = np.linspace(0.18, 0.32, 141)
    freq = np.linspace(6.85, 7.15, 201)
    fq = ph.tunable_frequencies(flux, EJmax=40.0, EC=0.25, d=0.0, nlev=2)[:, 0]
    fr, g = 7.0, 0.035
    up, lo = ph.dressed_frequencies(fq, fr, g)
    theta = 0.5 * np.arctan2(2 * g, fq - fr)                 # photon weight of each normal mode
    w_lo, w_up = np.cos(theta) ** 2, np.sin(theta) ** 2
    F = freq[:, None]
    dip = lambda f0, w: w / (1 + ((F - f0[None, :]) / 0.004) ** 2)
    S = 1 - 0.85 * (dip(lo, w_up) + dip(up, w_lo)) + rng.normal(0, 0.015, (freq.size, flux.size))
    x0 = flux[np.argmin(np.abs(fq - fr))]
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.avoided_crossing(ax, flux, freq, S, bare=(fq, np.full_like(flux, fr)), g=g, x0=x0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
