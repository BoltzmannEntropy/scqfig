"""Black-box quantization in four steps for the device of example 72: (a) the transmon coupled to its readout resonator; (b) the junction split into its linear inductance and its nonlinear remainder; (c) the linear part as one admittance, probed by a swept current source; (d) that admittance beside the nonlinear element, whose zero-point phase each mode sets (computed)."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph
from scqfig.schematics import symbols as S

TITLE = "Black-box circuit"
FAMILY = "Schematics"

C_SIGMA, C_G, C_R, C_K, L_J, F_R = 85e-15, 5e-15, 400e-15, 8e-15, 10e-9, 7.0e9   # example 72's device
L_R = 1 / ((2 * np.pi * F_R) ** 2 * C_R)
YT, YM, YB = 1.75, 1.1, 0.45                                     # top rail, element centres, ground rail


def branch(ax, x, draw, color=None):
    """A vertical branch from the top rail to the ground rail with an element drawn at YM."""
    S.wire(ax, [x, x], [YT, YM + 0.3], color=color)
    S.wire(ax, [x, x], [YM - 0.3, YB], color=color)
    draw(x)


def capacitor(ax, x, color=None):
    S.wire(ax, [x, x], [YT, YM + 0.035], color=color)
    S.capacitor_v(ax, x, YM, color=color)
    S.wire(ax, [x, x], [YM - 0.035, YB], color=color)


def network(ax, t, fs, split):
    """Resonator, coupling capacitor and transmon; split replaces the junction by L_J and the
    nonlinear element."""
    rc, qc = t.bus, t.qubits[0]
    branch(ax, 0.35, lambda x: S.inductor_v(ax, x, YM + 0.3, YM - 0.3, color=rc), rc)
    capacitor(ax, 0.85, rc)
    for y in (YT, YB):
        S.wire(ax, [0.35, 0.85], [y, y], color=rc)
    S.wire(ax, [0.85, 1.315], [YT, YT])
    S.capacitor_h(ax, 1.35, YT, tile_kind="coupling")
    S.wire(ax, [1.385, 1.85], [YT, YT])
    S.wire(ax, [0.85, 1.85], [YB, YB])
    capacitor(ax, 1.85, qc)
    right = 3.05 if split else 2.5
    for y in (YT, YB):
        S.wire(ax, [1.85, right], [y, y], color=qc)
    if split:
        branch(ax, 2.5, lambda x: S.inductor_v(ax, x, YM + 0.3, YM - 0.3), qc)
        S.wire(ax, [3.05, 3.05], [YT, YB], color=qc)
        S.nonlinear_element(ax, 3.05, YM)
    else:
        S.wire(ax, [2.5, 2.5], [YT, YB], color=qc)
        S.junction(ax, 2.5, YM)
    S.ground(ax, 1.35, YB - 0.08)
    S.wire(ax, [1.35, 1.35], [YB, YB - 0.08])
    lab = dict(fontsize=fs, va="center")
    ax.text(0.2, YM, r"$L_r$", ha="right", color=rc, **lab)
    ax.text(1.0, YM, r"$C_r$", ha="left", color=rc, **lab)
    ax.text(1.35, YT + 0.22, r"$C_g$", ha="center", **lab)
    ax.text(2.0, YM, r"$C_\Sigma$", ha="left", color=qc, **lab)
    ax.text(2.65 if split else 2.68, YM, r"$L_J$", ha="left", **lab)
    if split:
        ax.text(3.05, YM - 0.42, "nl", ha="center", **lab)


def black_box(ax, fs, x=2.6):
    """The linear part as one admittance, its port on the rails at x."""
    S.admittance_box(ax, 0.95, YM, w=1.1, h=0.62, fs=fs + 1)
    for y, side in ((YT, YM + 0.31), (YB, YM - 0.31)):
        S.wire(ax, [0.95, 0.95], [side, y])
        S.wire(ax, [0.95, x], [y, y])


def build():
    t = sq.theme()
    fs = t.size - 1.5
    C = np.array([[C_SIGMA, -C_G], [-C_G, C_R + C_K + C_G]])
    bb = ph.black_box(lambda f: ph.port_susceptance(f, C, np.diag([1 / L_J, 1 / L_R])), ph.ej_from_lj(L_J),
                      np.linspace(1, 20, 2001))
    fig, axes = sq.figure("column", aspect=0.7, nrows=2, ncols=2)
    for ax, k, ylim in zip(axes.flat, "abcd", [(0.2, 2.1)] * 2 + [(0.05, 1.9)] * 2):
        S.blank(ax)
        ax.set_xlim(-0.05, 3.45)
        ax.set_ylim(*ylim)
        sq.panel(ax, k)
    a, b, c, d = axes.flat
    with S.tiles():
        network(a, t, fs, split=False)
        network(b, t, fs, split=True)

    black_box(c, fs)
    S.wire(c, [2.6, 2.6], [YT, YM + 0.13])
    S.wire(c, [2.6, 2.6], [YM - 0.13, YB])
    S.current_source(c, 2.6, YM, up=True)
    S.tunable_mark(c, 2.6, YM, size=0.42)
    c.text(2.85, YM, r"$I(\omega)$", ha="left", va="center", fontsize=fs)
    c.text(1.75, 0.18, r"$Y(\omega) = I/V$", ha="center", va="center", fontsize=fs)

    with S.tiles():
        black_box(d, fs)
        S.wire(d, [2.6, 2.6], [YT, YB])
        S.nonlinear_element(d, 2.6, YM)
    S.flux_arrow(d, (2.82, YM + 0.25), (2.82, YM - 0.25), rad=-0.5, label=r"$\varphi_\mathrm{zpf}$",
                 offset=(0.36, 0.0))
    modes = ", ".join(rf"$\varphi_{n}$ = {p:.2g}" for n, p in zip("qr", bb["phi"]))
    d.text(1.75, 0.18, modes, ha="center", va="center", fontsize=fs)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
