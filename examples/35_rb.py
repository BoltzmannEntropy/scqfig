"""Standard and interleaved randomized benchmarking of an X gate, 30 random sequences per length, with the error per Clifford and the interleaved gate error (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Randomized benchmarking"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(35)
    m = np.array([1, 20, 50, 100, 150, 200, 300, 400, 500, 650, 800])
    nseq = 30

    def runs(p):
        mean = ph.rb_decay(m, p, A=0.47, B=0.5)
        spread = 0.004 + 0.03 * (1 - p ** m)             # sequences scatter more as errors pile up
        return np.clip(mean + rng.normal(0, 1, (nseq, m.size)) * spread, 0, 1)

    fig, ax = sq.figure("column", aspect=0.66)
    sq.plots.rb(ax, m, runs(0.9970), interleaved=runs(0.9955), gate=r"X")
    ax.set_ylim(0.45, 1.0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
