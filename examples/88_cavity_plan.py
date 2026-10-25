"""Two-chamber 3D cavity in plan view, to scale: a storage and a readout chamber joined by a trench that holds a transmon chip with two antenna pads, a coupling pin through the readout wall, and a uniform field washed over both chambers (designed geometry)."""
import scqfig as sq
from scqfig.schematics import chip, motifs, symbols

TITLE = "Cavity plan"
FAMILY = "Schematics"

MM = 1e3                                                     # chip units are micrometres
CHAMBERS = {"storage": (0.0, 0.0, 45 * MM, 40 * MM), "readout": (51 * MM, 0.0, 27 * MM, 40 * MM)}
TRENCH = (41 * MM, 17.5 * MM, 14 * MM, 5 * MM)
CHIP = dict(rect=(41.5 * MM, 19.2 * MM, 13 * MM, 1.6 * MM), pad=(4 * MM, 0.9 * MM), sep=1.6 * MM)


def build():
    t = sq.theme()
    fig, ax = sq.figure("column", aspect=0.72)
    symbols.blank(ax)
    out = chip.cavity_plan(ax, CHAMBERS, trench=TRENCH, chip=CHIP,
                           port=dict(at=(70 * MM, 28 * MM), side="right", label="pin"))
    for k, (u0, v0, w, h) in enumerate(CHAMBERS.values()):
        motifs.wash(ax, u0, v0, w, h, z=1.2, label="uniform field" if k == 0 else None)
    ax.text(out["chip"]["junction"][0], TRENCH[1] - 0.8 * MM, "transmon chip", ha="center", va="top",
            fontsize=t.size - 1, color=t.ink, zorder=9)
    u0, u1, v0, v1 = out["extent"]
    ax.set_xlim(u0 - 1 * MM, u1 + 3 * MM)
    ax.set_ylim(v0 - 1 * MM, v1 + 1 * MM)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
