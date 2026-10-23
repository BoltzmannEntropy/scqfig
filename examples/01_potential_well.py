"""Transmon potential with the four lowest levels and their computed |psi(phi)|^2 (charge basis, E_J/E_C = 80)."""
import scqfig as sq

TITLE = "Potential well and wavefunctions"
FAMILY = "Spectrum"


def build():
    fig, ax = sq.figure("column", aspect=0.72)
    sq.plots.potential_well(ax, EJ=20.0, EC=0.25, nlev=4)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
