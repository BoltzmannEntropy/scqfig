"""Detection efficiency of three photon counters as a product of four factors on a log axis with the measured efficiency marked, and parameter values per report stacked by status and grouped by study type (synthetic numbers)."""
import numpy as np

import scqfig as sq

TITLE = "Efficiency chain"
FAMILY = "Comparison"


def build():
    fig, (a, b) = sq.figure("page", aspect=0.34, ncols=2, gridspec_kw={"width_ratios": [1, 1.1]})
    counters = ["Counter A", "Counter B", "Counter C"]
    factors = {"conversion": [0.84, 0.92, 1.0], "readout": [0.95, 0.93, 0.88], "duty cycle": [0.78, 0.86, 0.70],
               "qubit decay": [0.90, 0.80, 0.86]}
    product = np.prod(list(factors.values()), axis=0)
    measured = product * np.array([1.06, 0.93, 0.97])
    sq.plots.stacked_bars(a, counters, factors, log=True, hatches=["", "", "", "////"],
                          markers={"measured": measured}, totals="product")
    a.set_xlim(0.35, 1.04)
    a.set_xticks([0.4, 0.6, 0.8, 1.0], ["0.4", "0.6", "0.8", "1"])
    a.set_xlabel("cumulative efficiency")
    reports = [f"Report {i}" for i in range(1, 10)]
    kinds = ["search"] * 3 + ["design"] * 4 + ["theory"] * 2
    status = {"measured": [38, 22, 9, 4, 6, 2, 0, 0, 0], "predicted": [6, 10, 4, 21, 15, 12, 7, 9, 5],
              "assumed": [3, 2, 5, 8, 4, 9, 3, 6, 4]}
    sq.plots.stacked_bars(b, reports, status, horizontal=False, groups=kinds, totals=True, legend="upper right")
    b.set_ylim(0, 62)
    b.tick_params(axis="x", labelrotation=90)
    b.set_ylabel("values per report")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
