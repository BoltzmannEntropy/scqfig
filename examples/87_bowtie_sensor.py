"""Resonator-free bowtie sensor: (a) the two wedge islands to scale beside a feedline, with a charge line below and a schematic magnifier on the junction; (b) its circuit, with the gate set by a dc source through C_g (designed geometry)."""
from matplotlib.patches import Rectangle

import scqfig as sq
from scqfig.schematics import chip, symbols

TITLE = "Bowtie sensor"
FAMILY = "Schematics"

FIN, OPENING, SEP, POCKET = 110.0, 157.5, 12.0, 40.0     # fin radius, opening angle (deg), island gap, pocket, um
FEED_SEP = 60.0                                          # fin tip to feedline edge


def junction_inset(ax, zoom):
    """Inside the magnifier, in disc units: two electrode fingers overlapping at a small junction."""
    t = sq.theme()
    (cx, cy), r = zoom.center, zoom.r
    w = 0.22 * r
    for sgn in (1, -1):
        y0 = cy + sgn * 0.14 * r
        ax.add_patch(Rectangle((cx - w / 2, min(y0, y0 + sgn * r)), w, r, fc=t.qubits[1], ec="none"))
    ax.add_patch(Rectangle((cx - 0.04 * r, cy - 0.16 * r), 0.08 * r, 0.32 * r, fc=t.qubits[1], ec="none"))
    ax.add_patch(Rectangle((cx - 0.08 * r, cy - 0.08 * r), 0.16 * r, 0.16 * r, fc=t.chip["junction"], ec="none"))


def plan(ax):
    t = sq.theme()
    symbols.blank(ax)
    u0, v0, w, h = -300.0, -470.0, 600.0, 760.0
    chip.substrate(ax, w, h, u0=u0, v0=v0)
    chip.ground_plane(ax, w, h, margin=0.0, u0=u0, v0=v0)
    b = chip.bowtie(ax, 0.0, 0.0, fin=FIN, opening=OPENING, sep=SEP, gap=POCKET, angle=90.0, color=t.qubits[1])
    wf, gf = 16.0, 10.0
    vf = b["right"][1] + FEED_SEP + wf / 2                 # the fins point along v at angle = 90
    chip.cpw(ax, [(u0, vf), (u0 + w, vf)], w=wf, gap=gf)
    chip.drive_line(ax, [(0.0, v0), (0.0, b["left"][1] - 210.0)], w=10.0, gap=6.0)
    z = chip.zoom_circle(ax, b["junction"], 22.0, (470.0, -190.0), 130.0, junction_inset, magnify=False,
                         label="junction")
    chip.scale_bar(ax, 180.0, -440.0, 100.0)
    kw = dict(fontsize=t.size - 1, zorder=9)
    ax.text(u0 + 10.0, vf + 22.0, "feedline", ha="left", va="bottom", color=t.chip["feedline"], **kw)
    ax.text(14.0, v0 + 60.0, "charge line", ha="left", va="center", color=t.chip["drive"], **kw)
    # anchors rotate with the bowtie: at angle = 90 its local top is the left side of the pocket
    ax.text(b["pocket_top"][0] - 8.0, -60.0, "island", ha="right", va="center", color=t.qubits[1], **kw)
    ax.set_xlim(u0 - 10.0, 610.0)
    ax.set_ylim(v0 - 10.0, v0 + h + 10.0)
    return z


def circuit(ax):
    t = sq.theme()
    symbols.blank(ax)
    q, bus, gate = t.qubits[1], t.chip["feedline"], t.chip["drive"]
    fs = t.size - 0.5
    yt, yn, x1, x2, xg = 3.0, 1.7, 1.1, 2.7, 3.55
    symbols.wire(ax, [0.2, 3.9], [yt, yt], color=bus, lw=1.4)
    ax.text(0.2, yt + 0.12, "feedline", ha="left", va="bottom", fontsize=fs, color=bus)
    with symbols.tiles():
        for x, name in ((x1, r"$C_{c1}$"), (x2, r"$C_{c2}$")):
            symbols.wire(ax, [x, x], [yt, yn], color=q)
            symbols.capacitor_v(ax, x, 2.35, gap=0.1, w=0.3, color=q, tile_kind="coupling")
            ax.text(x - 0.25, 2.35, name, ha="right", va="center", fontsize=fs, color=q)
        symbols.wire(ax, [x1, x2], [yn, yn], color=q)
        symbols.junction(ax, (x1 + x2) / 2, yn, s=0.09)
        xm = (x1 + x2) / 2
        symbols.wire(ax, [x1, x1, xm - 0.05], [yn, yn - 0.6, yn - 0.6], color=q)
        symbols.wire(ax, [xm + 0.05, x2, x2], [yn - 0.6, yn - 0.6, yn], color=q)
        symbols.capacitor_h(ax, xm, yn - 0.6, gap=0.1, h=0.28, color=q)
        symbols.wire(ax, [x2, xg], [yn, yn], color=q)
        symbols.wire(ax, [xg, xg], [yn, 0.62], color=gate)
        symbols.capacitor_v(ax, xg, 1.2, gap=0.1, w=0.28, color=gate, tile_kind="coupling")
    ax.text(xm, yn + 0.16, r"$E_J$", ha="center", va="bottom", fontsize=fs, color=q)
    ax.text(xm, yn - 0.82, r"$C_\Sigma$", ha="center", va="top", fontsize=fs, color=q)
    ax.text(xg + 0.24, 1.2, r"$C_g$", ha="left", va="center", fontsize=fs, color=gate)
    symbols.dc_source(ax, xg, 0.45, r=0.17, color=gate)
    ax.text(xg + 0.26, 0.45, r"$V_g$", ha="left", va="center", fontsize=fs, color=gate)
    symbols.wire(ax, [xg, xg], [0.28, 0.12], color=gate)
    symbols.ground(ax, xg, 0.12, color=gate)
    ax.set_xlim(0.0, 4.3)
    ax.set_ylim(-0.1, 3.5)


def build():
    fig, (axa, axb) = sq.figure("page", aspect=0.5, ncols=2, width_ratios=[1.2, 1.0])
    plan(axa)
    circuit(axb)
    sq.panel(axa, "a")
    sq.panel(axb, "b")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
