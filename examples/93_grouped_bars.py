"""Logarithmic sensitivities of a transmon's frequency, anharmonicity, coupling and dispersive shift to four design parameters (computed), and surface participation of two designs with the change printed over each pair (synthetic numbers)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Grouped bars"
FAMILY = "Comparison"

NOMINAL = {"C": 90.0, "LJ": 12.0, "Cg": 5.0, "fr": 7.2}       # fF, nH, fF, GHz


def outputs(C, LJ, Cg, fr):
    """f01, alpha, g and chi (GHz) of a transmon read out by a resonator at fr."""
    EJ, EC = 163.47 / LJ, 19.37 / C
    f01, alpha = np.sqrt(8 * EJ * EC) - EC, -EC
    n0 = NOMINAL
    EJ0, EC0 = 163.47 / n0["LJ"], 19.37 / n0["C"]
    g = 0.1 * (Cg / n0["Cg"]) * (n0["C"] / C) * ((EJ / EC) / (EJ0 / EC0)) ** 0.25 * np.sqrt(fr / n0["fr"])
    return f01, alpha, g, ph.dispersive_shift(f01, fr, g, alpha)


def build():
    sens = {}
    for p, label in (("C", r"$C_\Sigma$"), ("LJ", r"$L_J$"), ("Cg", r"$C_g$"), ("fr", r"$f_r$")):
        up, dn = dict(NOMINAL), dict(NOMINAL)
        up[p] *= 1.01
        dn[p] /= 1.01
        sens[label] = (np.log(np.abs(outputs(**up))) - np.log(np.abs(outputs(**dn)))) / (2 * np.log(1.01))
    fig, (a, b) = sq.figure("page", aspect=0.32, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]})
    sq.plots.grouped_bars(a, [r"$f_{01}$", r"$\alpha$", r"$g$", r"$\chi$"], sens)
    a.set_ylabel(r"$\partial\ln X/\partial\ln p$")
    regions = ["metal-air", "metal-substrate", "substrate-air"]
    sq.plots.grouped_bars(b, regions, {"design A": [2.1e-6, 3.2e-5, 4.0e-4], "design B": [9.5e-6, 1.1e-4, 4.4e-4]},
                          log=True, ratios=True, legend="upper left")
    b.set_ylabel(r"surface participation $p_i$")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
