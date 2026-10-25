"""f01 of a SQUID transmon against flux for a symmetric SQUID and a 20 % junction asymmetry, with the sweet spot and the flux that tunes it 1 GHz lower (computed)."""
import scqfig as sq

TITLE = "Flux tuning"
FAMILY = "Spectrum"


def build():
    fig, ax = sq.figure("column", aspect=0.7)
    sq.plots.flux_tuning(ax, EJmax=22.0, EC=0.24, d=(0.0, 0.2), detune=1.0)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
