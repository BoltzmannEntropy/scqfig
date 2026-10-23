"""Notch-resonator transmission as magnitude, phase and circle, with one complex fit and the cable delay, amplitude and phase calibrated out (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "S21 of a resonator"
FAMILY = "Readout"


def build():
    rng = np.random.default_rng(20)
    fr, Qi, Qc, phi = 6.8312, 3.0e4, 1.5e4, 0.12
    Ql = 1 / (1 / Qi + np.cos(phi) / Qc)
    f = np.linspace(fr - 6 * fr / Ql, fr + 6 * fr / Ql, 161)
    s = ph.notch_s21(f, fr, Ql, Qc, phi, a=0.42, alpha=1.1, tau=45.0)          # tau in ns, f in GHz
    s = s + 0.42 * 0.006 * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    r = sq.fit.notch(f, s)
    fig, ax = sq.figure("page", aspect=0.4)
    ax.remove()
    axd = fig.subplot_mosaic([["mag", "circle"], ["phase", "circle"]], width_ratios=[1.55, 1])
    axd["phase"].sharex(axd["mag"])
    sq.plots.s21_magnitude(axd["mag"], f, s, fit=r)
    sq.plots.s21_phase(axd["phase"], f, s, fit=r)
    sq.plots.s21_circle(axd["circle"], f, s, fit=r)
    axd["mag"].set_xlabel("")
    axd["mag"].tick_params(labelbottom=False)
    for k, name in zip("abc", ("mag", "phase", "circle")):
        sq.panel(axd[name], k)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
