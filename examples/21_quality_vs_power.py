"""Internal quality factor of a resonator against mean photon number, with a fit of the saturable two-level-system model (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Quality factor vs power"
FAMILY = "Readout"


def build():
    rng = np.random.default_rng(21)
    nbar = np.logspace(-1, 6, 22)
    qi = ph.tls_qi(nbar, F_delta=1.3e-6, nc=4.0, beta=0.42, Q_other=2.2e6)
    qi = qi * np.exp(rng.normal(0, 0.04, nbar.size))
    fig, ax = sq.figure("column", aspect=0.7)
    sq.plots.quality_vs_power(ax, nbar, qi, err=0.04 * qi)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
