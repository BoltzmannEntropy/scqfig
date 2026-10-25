"""Scan-rate gain R = eta^2 dnu/Gamma of a photon counter over a quantum-limited amplifier on the plane of dark-count rate (over the signal linewidth) and efficiency, with R = 1 drawn heavier and three illustrative devices, one an efficiency lower bound (computed map, synthetic points)."""
import numpy as np

import scqfig as sq

TITLE = "Design map"
FAMILY = "Design"


def build():
    x = np.logspace(-6, 2, 240)                                     # dark-count rate / signal linewidth
    eta = np.logspace(-2, 0, 200)                                   # detection efficiency
    R = eta[:, None] ** 2 / x[None, :]
    levels = [1e-2, 1.0, 1e2, 1e4]
    fig, ax = sq.figure("column", aspect=0.8)
    sq.plots.design_map(ax, x, eta, R, levels=levels, emphasis=1.0,
                        fmt=lambda v: rf"$R = 10^{{{np.log10(v):.0f}}}$",
                        points={"counter A": (3e-4, 0.42, "data"), "counter B": (2e-2, 0.75, "data"),
                                "design C": (2e-5, 0.6, "sim")},
                        arrows={"counter A": "up"},
                        xlabel=r"dark-count rate $\Gamma/\Delta\nu$", ylabel=r"efficiency $\eta$")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
