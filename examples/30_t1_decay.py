"""Energy relaxation: excited-state population against delay with an exponential fit (synthetic data, T1 = 48 us)."""
import numpy as np

import scqfig as sq

TITLE = "T1 decay"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 240, 41)
    p = 0.93 * np.exp(-t / 48.0) + 0.03 + rng.normal(0, 0.012, t.size)
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.t1_decay(ax, t, p, err=np.full(t.size, 0.012))
    ax.set_ylim(0, 1)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
