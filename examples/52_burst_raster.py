"""Correlated error bursts: detections per qubit over time, with the flagged fraction and burst marks above (synthetic data from a toy burst model)."""
import scqfig as sq
from scqfig.plots.noise import burst_events

TITLE = "Correlated error bursts"
FAMILY = "Noise and errors"


def build():
    t, E = burst_events(nq=24, nt=1000, p_bg=0.02, bursts=((220, 6), (690, 17)), tau=25.0, rng=52)
    fig, (top, ax) = sq.figure("page", aspect=0.36, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
    sq.plots.burst_raster(ax, t, E, ax_frac=top)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
