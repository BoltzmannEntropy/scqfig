"""Lumped circuit of a transmon coupled to a quarter-wave readout resonator on a feedline, each Hamiltonian term on the tile of the element that produces it."""
import numpy as np

import scqfig as sq
from scqfig import physics as ph
from scqfig.schematics import symbols as S

TITLE = "Transmon and readout circuit"
FAMILY = "Schematics"

H_PLANCK, E, PHI0 = 6.62607015e-34, 1.602176634e-19, 2.067833848e-15
C_SIGMA, L_J, C_G, C_R, C_K, F_R = 85e-15, 10e-9, 5e-15, 400e-15, 8e-15, 7.0e9


def values():
    """Circuit parameters to energies (GHz): E_C, E_J, the exact f01, L_r for f_r, and g."""
    EC = E ** 2 / (2 * C_SIGMA) / H_PLANCK / 1e9
    EJ = (PHI0 / (2 * np.pi)) ** 2 / L_J / H_PLANCK / 1e9
    f01 = ph.transmon_spectrum(EJ, EC)[1]
    L_R = 1 / ((2 * np.pi * F_R) ** 2 * C_R)
    g = 0.5 * C_G / np.sqrt(C_SIGMA * C_R) * np.sqrt(f01 * F_R / 1e9)
    return EC, EJ, f01, L_R, g


def build():
    t = sq.theme()
    EC, EJ, f01, L_R, g = values()
    fig, ax = sq.figure("column", aspect=0.69, layout=None)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    S.blank(ax)
    fs = t.size - 0.5
    qc, rc = t.qubits[0], t.bus
    yf, yt, yb = 3.3, 2.0, 0.6                                   # feedline, top rail, ground rail

    # feedline from a source to a matched load
    S.wire(ax, [0.4, 6.0], [yf, yf])
    S.wire(ax, [0.4, 0.4], [yf, 2.9])
    S.ac_source(ax, 0.4, 2.75, r=0.15)
    S.wire(ax, [0.4, 0.4], [2.6, 2.35])
    S.ground(ax, 0.4, 2.35)
    S.wire(ax, [6.0, 6.0], [yf, 3.0])
    S.resistor_v(ax, 6.0, 3.0, 2.5)
    S.wire(ax, [6.0, 6.0], [2.5, 2.35])
    S.ground(ax, 6.0, 2.35)
    ax.text(0.62, 2.75, "in", ha="left", va="center", fontsize=fs)
    ax.text(5.8, 2.75, r"$Z_0$ = 50 Ω", ha="right", va="center", fontsize=fs)

    with S.tiles():
        # coupling to the feedline: a port, not a Hamiltonian term, so no tile
        S.node(ax, 1.8, yf)
        S.wire(ax, [1.8, 1.8], [yf, 2.655])
        S.capacitor_v(ax, 1.8, 2.62, tile_kind=False)
        S.wire(ax, [1.8, 1.8], [2.585, yt])
        ax.text(1.97, 2.62, rf"$C_\kappa$ = {C_K * 1e15:.0f} fF", ha="left", va="center", fontsize=fs)

        # resonator: L_r and C_r share one tile, the term h f_r a+a
        S.tile(ax, 1.8, 1.3, 1.5, 1.75, sq.lighten(rc, 0.86))
        S.wire(ax, [1.2, 2.4], [yt, yt], color=rc)
        S.wire(ax, [1.2, 2.4], [yb, yb], color=rc)
        S.wire(ax, [1.2, 1.2], [yt, 1.75], color=rc)
        S.inductor_v(ax, 1.2, 1.75, 0.85, color=rc, tile_kind=False)
        S.wire(ax, [1.2, 1.2], [0.85, yb], color=rc)
        S.wire(ax, [2.4, 2.4], [yt, 1.335], color=rc)
        S.capacitor_v(ax, 2.4, 1.3, color=rc, tile_kind=False)
        S.wire(ax, [2.4, 2.4], [1.265, yb], color=rc)
        S.node(ax, 1.8, yt, color=rc)
        S.node(ax, 2.4, yt, color=rc)
        S.wire(ax, [1.8, 1.8], [yb, 0.45], color=rc)
        S.ground(ax, 1.8, 0.45, color=rc)
        ax.text(0.98, 1.3, r"$L_r$" + "\n" + f"{L_R * 1e9:.2f} nH", ha="right", va="center", fontsize=fs, color=rc)
        ax.text(2.56, 1.3, r"$C_r$" + "\n" + f"{C_R * 1e15:.0f} fF", ha="left", va="center", fontsize=fs, color=rc)

        # coupling capacitor, the term h g n (a + a+)
        S.wire(ax, [2.4, 3.415], [yt, yt])
        S.capacitor_h(ax, 3.45, yt, tile_kind="coupling")
        S.wire(ax, [3.485, 4.2], [yt, yt])
        ax.text(3.45, yt + 0.27, rf"$C_g$ = {C_G * 1e15:.0f} fF", ha="center", va="bottom", fontsize=fs)

        # transmon: junction (nonlinear tile) and shunt capacitor (capacitive tile)
        S.wire(ax, [4.2, 5.6], [yt, yt], color=qc)
        S.wire(ax, [4.2, 5.6], [yb, yb], color=qc)
        S.wire(ax, [4.2, 4.2], [yt, yb], color=qc)
        S.junction(ax, 4.2, 1.3)
        S.wire(ax, [5.6, 5.6], [yt, 1.335], color=qc)
        S.capacitor_v(ax, 5.6, 1.3, color=qc)
        S.wire(ax, [5.6, 5.6], [1.265, yb], color=qc)
        S.node(ax, 4.2, yt, color=qc)
        S.wire(ax, [4.9, 4.9], [yb, 0.45], color=qc)
        S.ground(ax, 4.9, 0.45, color=qc)
        ax.text(4.0, 1.3, r"$L_J$" + "\n" + f"{L_J * 1e9:.0f} nH", ha="right", va="center", fontsize=fs)
        ax.text(5.4, 1.3, r"$C_\Sigma$" + "\n" + f"{C_SIGMA * 1e15:.0f} fF", ha="right", va="center",
                fontsize=fs, color=qc)

    # the Hamiltonian, one tile per term, each tile the colour of the element behind it
    terms = [(r"$4E_C\hat n^2$", rf"$E_C/h$ = {EC * 1e3:.0f} MHz", "capacitive"),
             (r"$-E_J\cos\hat\varphi$", rf"$E_J/h$ = {EJ:.1f} GHz", "nonlinear"),
             (r"$hf_r\hat a^\dagger\hat a$", rf"$f_r$ = {F_R / 1e9:.2f} GHz", sq.lighten(rc, 0.86)),
             (r"$hg\,\hat n(\hat a+\hat a^\dagger)$", rf"$g/h$ = {g * 1e3:.0f} MHz", "coupling")]
    ax.set_xlim(-0.1, 6.5)
    ax.set_ylim(-0.95, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.apply_aspect()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def width(s, size):
        txt = ax.text(0, 0, s, fontsize=size)
        bb = inv.transform(txt.get_window_extent(r))
        txt.remove()
        return bb[1, 0] - bb[0, 0]

    big, small, pad, plus = fs + 0.5, fs - 1.5, 0.14, 0.26
    tiles = [max(width(a, big), width(b, small)) + pad for a, b, _ in terms]
    head = width(r"$\hat H =$", big) + 0.1
    x = 3.2 - (head + sum(tiles) + plus * (len(tiles) - 1)) / 2          # the row centred under the circuit
    y = -0.45
    ax.text(x, y + 0.08, r"$\hat H =$", ha="left", va="center", fontsize=big)
    x += head
    for k, ((term, val, kind), w) in enumerate(zip(terms, tiles)):
        S.tile(ax, x + w / 2, y - 0.03, w, 0.66, kind, z=1)
        ax.text(x + w / 2, y + 0.08, term, ha="center", va="center", fontsize=big, zorder=3)
        ax.text(x + w / 2, y - 0.2, val, ha="center", va="center", fontsize=small, color=t.reference, zorder=3)
        x += w
        if k < len(terms) - 1:
            ax.text(x + plus / 2, y + 0.08, "+", ha="center", va="center", fontsize=big)
            x += plus
    ax.text(4.9, 0.14, rf"$f_{{01}}$ = {f01:.2f} GHz", ha="center", va="center", fontsize=fs - 1, color=t.reference)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
