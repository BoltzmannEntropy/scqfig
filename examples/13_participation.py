"""Surface-participation loss budget of a transmon: participation and loss tangent of each region, the quality factor each allows, and the total (simulated participation, assumed loss tangents)."""
import scqfig as sq

TITLE = "Participation budget"
FAMILY = "Design"


def build():
    p = {"MA": 2.0e-5, "MS": 2.6e-4, "SA": 2.1e-4, "substrate": 0.91}
    tand = {"MA": 2.0e-3, "MS": 2.6e-3, "SA": 2.2e-3, "substrate": 5e-8}
    fig, ax = sq.figure("column", aspect=0.66)
    sq.plots.participation(ax, p, tand, freq=4.9)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
