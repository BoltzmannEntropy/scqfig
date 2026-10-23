"""CZ-gate error budget: contributions, their sum, and the measured error with the unexplained part hatched."""
import scqfig as sq

TITLE = "Error budget"
FAMILY = "Noise and errors"


def build():
    items = {r"$T_1$": 0.21, r"$T_\phi$": 0.12, "leakage": 0.07, "spectator ZZ": 0.05, "control": 0.04}
    fig, ax = sq.figure("column", aspect=0.62)
    sq.plots.error_budget(ax, items, measured=0.56)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
