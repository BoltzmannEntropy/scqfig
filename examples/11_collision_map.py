"""Collision map of two coupled transmons: forbidden bands of the f_q2 against f_q1 plane, with three design points and their 20 MHz fabrication scatter; one point sits on a two-photon band (designed values)."""
import scqfig as sq
from scqfig import physics as ph

TITLE = "Collision map"
FAMILY = "Design"


def build():
    f = ph.transmon_spectrum(16.0, 0.21, nlev=3)
    alpha = f[2] - 2 * f[1]                                      # GHz, about -0.23
    points = {"Q1–Q2": (4.900, 5.062), "Q3–Q2": (4.966, 5.062), "Q4–Q5": (5.100, 5.100 + alpha / 2 + 0.002)}
    fig, ax = sq.figure("column", aspect=0.95)
    sq.plots.collision_map(ax, alpha, points, lim=(4.7, 5.35), spread=0.02)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
