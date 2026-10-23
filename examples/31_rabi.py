"""Power Rabi: excited population against drive amplitude for a 40 ns pulse, with a cosine fit and the pi-pulse amplitude (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Power Rabi"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(31)
    amp = np.linspace(0, 1, 61)
    rabi = 0.030 * amp                                   # Rabi rate (GHz) at full scale 30 MHz
    p = 0.03 + 0.92 * ph.rabi_population(rabi, 0.0005, 40.0) + rng.normal(0, 0.015, amp.size)
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.rabi(ax, amp, p)
    ax.set_ylim(0, 1)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
