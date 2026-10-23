"""Wiring of a dilution refrigerator: attenuated and filtered XY, Z and readout-input lines, and a readout output through isolators, a TWPA and a HEMT."""
import scqfig as sq
from scqfig.schematics import wiring

TITLE = "Cryostat wiring"
FAMILY = "Schematics"


def build():
    fig, ax = sq.figure("column", aspect=1.32)
    wiring.cryostat_wiring(ax)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
