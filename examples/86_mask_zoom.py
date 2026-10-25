"""Xmon cell drawn to scale: a claw couples the quarter-wave resonator, a mask-level SQUID bridges the bottom gap, and two circular magnifiers show the claw and the SQUID, still to scale (designed geometry)."""
import scqfig as sq
from scqfig.schematics import chip, symbols

TITLE = "Mask cell with magnifiers"
FAMILY = "Schematics"

ARM, WIDTH, GAP = 170.0, 24.0, 24.0               # Xmon arm (tip from centre), arm width, gap, in um
W_RES, G_RES = 12.0, 8.0                          # resonator and claw CPW
V_FEED = 736.0                                    # feedline centre
WINDOW = (-420.0, -300.0, 840.0, 1120.0)          # u0, v0, w, h of the drawn part of the chip


def cell(ax):
    """The whole cell in chip units; the magnifiers call it again and show part of it enlarged."""
    u0, v0, w, h = WINDOW
    chip.substrate(ax, w, h, u0=u0, v0=v0)
    chip.ground_plane(ax, w, h, margin=0.0, u0=u0, v0=v0)
    q = chip.xmon(ax, 0.0, 0.0, arm=ARM, width=WIDTH, gap=GAP, squid=False)
    c = chip.claw(ax, *q["top"], arm=WIDTH, clearance=20.0, depth=80.0, w=W_RES, gap=G_RES)
    chip.meander_resonator(ax, 0.0, V_FEED - 36.0, c["attach"][1], couple=300.0, width=240.0, turns=4,
                           w=W_RES, gap=G_RES, side=1)
    chip.cpw(ax, [(u0, V_FEED), (u0 + w, V_FEED)], w=W_RES, gap=G_RES)
    s = chip.squid_detail(ax, 0.0, -ARM - GAP / 2, span=GAP, loop=(16.0, 8.0), lead=1.5, jj=0.5)
    return c, s


def build():
    t = sq.theme()
    fig, ax = sq.figure("column", aspect=0.93)
    symbols.blank(ax)
    c, s = cell(ax)
    claw_at = (c["tip"][0], (c["left"][1] + c["attach"][1]) / 2)
    # each magnifier redraws the cell; its scale bar sits above the disc, magnified by the same factor
    for at, r_at, centre, r, bar, name in ((claw_at, 80.0, (700.0, 470.0), 200.0, 20.0, "claw"),
                                           (s["center"], 22.0, (700.0, -60.0), 170.0, 5.0, "SQUID")):
        z = chip.zoom_circle(ax, at, r_at, centre, r, lambda ax, zoom: cell(ax), label=name)
        L = bar * z["scale"]
        chip.scale_bar(ax, centre[0] - L / 2, centre[1] + r + 30.0, L, label=f"{bar:g} µm")
    chip.scale_bar(ax, -400.0, -350.0, 200.0, below=True)
    kw = dict(fontsize=t.size - 1, zorder=9)
    ax.text(-ARM - 20.0, 40.0, "Xmon", ha="right", va="bottom", color=t.qubits[0], **kw)
    ax.text(-400.0, V_FEED + 26.0, "feedline", ha="left", va="bottom", color=t.chip["feedline"], **kw)
    ax.text(150.0, 420.0, r"$\lambda/4$", ha="left", va="center", color=t.chip["resonator"], **kw)
    ax.set_xlim(-430.0, 910.0)
    ax.set_ylim(-420.0, 830.0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
