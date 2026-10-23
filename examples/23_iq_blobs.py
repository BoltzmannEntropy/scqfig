"""Single-shot readout in the IQ plane: 4000 shots per prepared state with 1 and 2 sigma contours and the discrimination line; the |1> shots include T1 decay during the 1.5 us integration (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig.plots.readout import readout_shots

TITLE = "IQ blobs"
FAMILY = "Readout"


def build():
    rng = np.random.default_rng(23)
    centres = [-0.55 + 0.30j, 0.35 + 0.62j]                       # mV
    shots = [readout_shots(m, centres, 0.16, 4000, t_ro=1.5, T1=20.0, p_th=0.01, rng=rng) for m in (0, 1)]
    fig, ax = sq.figure("column", aspect=0.85)
    sq.plots.iq_blobs(ax, shots)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
