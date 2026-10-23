"""A 3 x 4 qubit array drawn as spins in a uniform field (the wash), with photons arriving from several directions and three qubits excited."""
import numpy as np
from matplotlib.patches import FancyBboxPatch

import scqfig as sq
from scqfig.schematics import motifs, symbols

TITLE = "Qubit array in a uniform field"
FAMILY = "Schematics"


def build():
    t = sq.theme()
    rng = np.random.default_rng(77)
    rows, cols = 3, 4
    excited = set(rng.choice(rows * cols, 3, replace=False).tolist())
    fig, ax = sq.figure("column", aspect=0.72)
    symbols.blank(ax)
    x0, y0, w, h = -0.6, -0.6, cols - 1 + 1.2, rows - 1 + 1.2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=t.wash, ec="none",
                                zorder=1))
    for k in range(rows * cols):
        x, y = k % cols, rows - 1 - k // cols
        up = k not in excited
        hgt = 0.62 if up else -0.62
        motifs.spin(ax, x, y - hgt / 2, hgt, color=t.qubits[0] if up else t.qubits[1], lw=1.8)
    for p0, p1 in (((-1.1, 3.4), (-0.3, 2.7)), ((-1.85, 1.0), (-0.7, 1.0)), ((1.25, 3.68), (1.55, 2.7)),
                   ((4.55, 3.25), (3.7, 2.35)), ((4.65, -0.55), (3.7, 0.1))):                  # all end at the wash
        motifs.wavy(ax, p0, p1, amp=0.07, waves=4, lw=1.0, head=0.2)
    fs = t.size - 0.5
    ax.text(x0 + w - 0.08, y0 + 0.08, "uniform field", ha="right", va="bottom", fontsize=fs - 1, color=t.reference)
    for k, (lab, up) in enumerate(((r"$|0\rangle$", True), (r"$|1\rangle$", False))):
        yk = -1.05
        xk = 0.2 + 1.5 * k
        hgt = 0.3 if up else -0.3
        motifs.spin(ax, xk, yk - hgt / 2, hgt, color=t.qubits[0] if up else t.qubits[1], lw=1.4)
        ax.text(xk + 0.18, yk, lab, ha="left", va="center", fontsize=fs)
    ax.set_xlim(-2.0, cols + 1.0)
    ax.set_ylim(-1.35, rows + 0.75)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
