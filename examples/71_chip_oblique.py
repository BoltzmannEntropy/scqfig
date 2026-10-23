"""The two-qubit chip of example 70 laid on a slab by one rigid oblique map, with a plan-view zoom on the SQUID of Q1."""
from matplotlib.patches import ConnectionPatch, Rectangle

import scqfig as sq
from scqfig.schematics import chip, motifs, symbols

TITLE = "Chip on a slab"
FAMILY = "Schematics"

W, H = 3600.0, 2400.0
OB = motifs.Oblique(x0=0.0, y0=0.0, shear=0.45, squash=0.5)


def build():
    t = sq.theme()
    ch = t.chip
    fs = t.size - 1
    fig, ax = sq.figure("column", aspect=0.74)
    symbols.blank(ax)
    motifs.ob_slab(ax, OB, W, H, depth=170)
    before = motifs.artists(ax)
    chip.ground_plane(ax, W, H)
    a = chip.coupler_chip(ax, size=(W, H), labels=False, bridges=False)
    for name, (u, v), c in ((r"Q$_1$", (1010, 1110), t.qubits[0]), (r"Q$_2$", (2590, 1110), t.qubits[1]),
                            ("R$_1$", (760, 1420), ch["resonator"]), ("R$_2$", (2860, 1420), ch["resonator"]),
                            ("feedline", (1800, 1960), ch["feedline"])):
        ax.text(u, v, name, ha="center", va="center", fontsize=fs, color=c, zorder=8)
    us, vs = a["Q1"]["squid"]
    win = (us - 95, vs - 62, 190, 124)                     # zoom window in chip units
    ax.add_patch(Rectangle(win[:2], win[2], win[3], fc="none", ec=t.ink, lw=0.6, zorder=9))
    motifs.onto_plane(ax, OB, before)                      # the whole layout through one affine map
    ax.set_xlim(-40, W + OB.shear * H + 40)
    ax.set_ylim(-230, OB.squash * H + 1150)

    ins = ax.inset_axes([0.0, 0.53, 0.42, 0.47])
    ins.set_aspect("equal")
    chip.substrate(ins, W, H)
    chip.ground_plane(ins, W, H)
    chip.coupler_chip(ins, size=(W, H), labels=False, bridges=False)
    ins.set_xlim(win[0], win[0] + win[2])
    ins.set_ylim(win[1], win[1] + win[3])
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor(t.ink)
    ins.text(us, vs, r"$\Phi$", ha="center", va="center", fontsize=fs, color=t.ink, zorder=9)
    ins.annotate("JJ", (us + 14, vs - 2), (us + 50, vs + 24), fontsize=fs, color=t.ink, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=t.ink, lw=0.5, shrinkA=1, shrinkB=2), zorder=9)
    ins.text(us + 26, win[1] + win[3] - 8, r"Q$_1$ arm", ha="left", va="top", fontsize=fs, color=t.qubits[0], zorder=9)
    ins.text(us - 50, win[1] + 10, r"Z$_1$", ha="right", va="bottom", fontsize=fs, color=ch["flux"], zorder=9)
    chip.scale_bar(ins, win[0] + win[2] - 62, win[1] + 14, 50, fs=t.size - 1.5)
    for cx in (0, 1):
        corner = OB(win[0] + cx * win[2], win[1])
        ax.add_artist(ConnectionPatch((cx, 0), corner, coordsA=ins.transAxes, coordsB=ax.transData,
                                      color=t.reference, lw=0.5, zorder=8))
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
