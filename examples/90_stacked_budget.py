"""Dark-count budget of four photon counters split by source with each total printed beside its bar, and the share of four value classes in three study groups (synthetic numbers)."""
import scqfig as sq

TITLE = "Stacked budgets"
FAMILY = "Comparison"


def build():
    fig, (a, b) = sq.figure("page", aspect=0.3, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
    counters = ["Counter 1", "Counter 2", "Counter 3", "Counter 4"]
    rates = {"qubit": [62.0, 41.0, 18.0, 26.0], "pump": [0.0, 21.0, 0.0, 9.0], "line photons": [14.0, 12.0, 52.0, 6.0]}
    sq.plots.stacked_bars(a, counters, rates, hatches=["", "////", ""], totals="total")
    a.set_xlabel(r"dark-count rate (s$^{-1}$)")
    groups = ["Group A", "Group B", "Group C"]
    counts = {"measured": [52, 9, 2], "fitted": [18, 4, 0], "predicted": [11, 30, 8], "assumed": [6, 14, 21]}
    sq.plots.stacked_bars(b, groups, counts, normalize=True, totals="values")
    b.set_xlabel("share of values")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
