"""Cross-section of a tantalum CPW on silicon with its metal-air, metal-substrate and substrate-air interfaces marked."""
import scqfig as sq
from scqfig.schematics import chip

TITLE = "Film stack"
FAMILY = "Schematics"


def build():
    fig, ax = sq.figure("column", aspect=0.62)
    chip.cross_section(ax, [("Si (100)", 525e3, "substrate"), ("Ta (α)", 200, "metal")],
                       interfaces={"MA": r"metal–air, Ta$_2$O$_5$", "MS": "metal–substrate", "SA": r"substrate–air, SiO$_x$"})
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
