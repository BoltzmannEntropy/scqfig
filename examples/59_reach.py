"""Kinetic-mixing reach of a qubit dark-photon search against frequency for one and sixteen sensors at 20 and 50 mK, from an illustrative counting model with thermal dark counts, an excluded region and a synthetic measured limit, with the dark-photon mass on the top axis (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Dark-photon reach"
FAMILY = "Dark matter"


def counting_reach(f, n_q, T, tau=100e-6, t_point=3600.0, p_readout=1e-3, eps0=1e-10, f0=5.0):
    """Smallest coupling whose signal clears max(3, 5 sqrt(dark counts)) in t_point seconds of
    tau-long shots on n_q sensors. The excitation probability per shot grows as eps^2 f, reaching one
    at eps0 for a sensor at f0 GHz; dark counts come from readout error and thermal population."""
    shots = n_q * t_point / tau
    dark = (p_readout + ph.thermal_population(f, T)) * shots
    p_needed = np.maximum(3.0, 5.0 * np.sqrt(dark)) / shots
    return eps0 * np.sqrt(p_needed * f0 / f)


def build():
    t = sq.theme()
    f = np.logspace(0, 1, 200)                                      # GHz
    curves, styles = {}, {}
    for n_q, colour in ((1, t.cycle[0]), (16, t.cycle[1])):
        for T, ls in ((0.02, "-"), (0.05, (0, (4, 2)))):
            name = rf"$n_q$ = {n_q}, {T * 1e3:.0f} mK"
            curves[name] = counting_reach(f, n_q, T)
            styles[name] = dict(color=colour, ls=ls)
    fig, ax = sq.figure("column", aspect=0.85)
    sq.plots.reach(ax, f, curves, styles=styles, excluded="above", limits={"pilot run": (6.2, 4e-12)},
                   loc="lower left")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
