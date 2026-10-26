"""Dark-matter particles per de Broglie volume against rest energy for 0.45 GeV/cm³ and 220 km/s: below about 40 eV each cell holds many particles and the dark matter acts as one classical wave, which covers the 1 to 10 GHz band of qubit searches (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Occupation number"
FAMILY = "Dark matter"


def build():
    m = np.logspace(-22, 12, 300)                                   # eV
    N = ph.occupation_number(m * 1e6)
    band = tuple(ph.mass_from_frequency([1.0, 10.0]) * 1e-6)        # 1 to 10 GHz in eV
    fig, ax = sq.figure("column", aspect=0.72)
    sq.plots.occupation(ax, m, N, band=band, band_label="1 to 10 GHz")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
