"""Dispersive readout: resonator transmission with the qubit in |0> and in |1>, the 2chi shift computed for g = 85 MHz and Delta = -2.09 GHz, and the readout frequency between the two dips (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Dispersive readout"
FAMILY = "Readout"


def build():
    rng = np.random.default_rng(22)
    fr, fq, g, alpha = 7.15, 5.06, 0.085, -0.233
    chi = ph.dispersive_shift(fq, fr, g, alpha)                    # GHz, negative here
    Ql, Qi = fr / (2 * abs(chi)), 3e5                              # kappa = 2|chi|
    Qc = 1 / (1 / Ql - 1 / Qi)
    f = np.linspace(fr - 0.003, fr + 0.003, 121)

    def noise():
        return 0.012 * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))

    s0 = ph.notch_s21(f, fr - chi, Ql, Qc) + noise()                # qubit in |0>
    s1 = ph.notch_s21(f, fr + chi, Ql, Qc) + noise()                # qubit in |1>
    fig, ax = sq.figure("column", aspect=0.72)
    sq.plots.dispersive_readout(ax, f, s0, s1)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
