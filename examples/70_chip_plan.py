"""Plan view of a two-qubit chip: Xmons joined by a tunable coupler, readout resonators on a feedline, drive and flux lines, airbridges."""
import scqfig as sq
from scqfig.schematics import chip, symbols

TITLE = "Chip plan"
FAMILY = "Schematics"

W, H = 3600.0, 2400.0


def build():
    fig, ax = sq.figure("column", aspect=0.74)
    symbols.blank(ax)
    chip.substrate(ax, W, H)
    chip.ground_plane(ax, W, H)
    chip.coupler_chip(ax, size=(W, H))
    chip.scale_bar(ax, W - 500, -110, 500, below=True)
    ax.set_xlim(-20, W + 20)
    ax.set_ylim(-330, H + 20)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
