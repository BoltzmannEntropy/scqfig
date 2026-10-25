"""Reports by year in five research lines, one marker per report shaped by its kind; the sparse years before 2018 share a compressed stretch of the axis, and reports of one line and year sit side by side (synthetic record)."""
import numpy as np

import scqfig as sq

TITLE = "Timeline"
FAMILY = "Comparison"


def build():
    rng = np.random.default_rng(99)
    lines = ["Line A", "Line B", "Line C", "Line D", "Line E"]
    kinds = ["search", "device", "design", "theory", "reanalysis"]
    years = np.r_[rng.integers(2007, 2018, 7), rng.integers(2018, 2027, 38)]
    events = [(lines[rng.integers(len(lines))], int(y), kinds[rng.integers(len(kinds))]) for y in np.sort(years)]
    fig, ax = sq.figure("page", aspect=0.36)
    sq.plots.timeline(ax, lines, events, kinds=kinds, break_at=2018)
    ax.set_xlabel("year")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
