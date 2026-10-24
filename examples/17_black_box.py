"""Black-box quantization of the transmon and readout resonator of example 72: (a) the susceptance the junction sees, whose rising zeros are the modes and whose slopes set their impedances; (b) the first-order Kerr matrix, with exact diagonalization of the same circuit in brackets (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph

TITLE = "Black-box quantization"
FAMILY = "Design"

# example 72's device; the 50 ohm feedline is small beside the 2.8 kohm of C_kappa, so it counts as ground
C_SIGMA, C_G, C_R, C_K, L_J, F_R = 85e-15, 5e-15, 400e-15, 8e-15, 10e-9, 7.0e9
L_R = 1 / ((2 * np.pi * F_R) ** 2 * C_R)


def build():
    C = np.array([[C_SIGMA, -C_G], [-C_G, C_R + C_K + C_G]])         # transmon node, resonator node
    Linv = np.diag([1 / L_J, 1 / L_R])                                 # the junction as its linear inductance
    B = lambda f: ph.port_susceptance(f, C, Linv)
    bb = ph.black_box(B, ph.ej_from_lj(L_J), np.linspace(1, 20, 2001))
    exact = ph.transmon_resonator(C_SIGMA - C_G, C_G, C_R + C_K, L_J, L_R)
    f = np.linspace(4.0, 8.0, 2001)
    fig, (a, b) = sq.figure("page", aspect=0.3, ncols=2, gridspec_kw={"width_ratios": [2.4, 1]})
    sq.plots.admittance(a, f, B(f), modes=bb, labels=["q", "r"])
    sq.plots.kerr_matrix(b, bb["K"], ["q", "r"], exact=exact["K"], cbar=False)
    b.set_xlabel("MHz, exact in brackets")
    sq.panel(a, "a")
    sq.panel(b, "b")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
