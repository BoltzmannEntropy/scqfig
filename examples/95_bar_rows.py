"""Background probability per measurement in two studies, with a hatched upper limit, open simulated terms and the values printed beside the bars, and measured reduction factors drawn out from 1 with published values as grey ticks (synthetic numbers)."""
import numpy as np

import scqfig as sq
from scqfig.core import sci

TITLE = "Bar rows"
FAMILY = "Comparison"


def build():
    fig, (a, b) = sq.figure("page", aspect=0.36, ncols=2)
    terms = ["detector error", "heating", "qubit dressing", "measured total", "qubit dressing", "other baths",
             "measured total"]
    study = ["Study A"] * 4 + ["Study B"] * 3
    p = [3e-6, 4.1e-4, 1.2e-4, 7.4e-4, 3.0e-4, 8.6e-4, 1.16e-3]
    status = ["upper", "simulated", "simulated", "measured", "simulated", "simulated", "measured"]
    text = [("< " if s == "upper" else "") + sci(v) for v, s in zip(p, status)]
    sq.plots.bar_rows(a, terms, p, status=status, groups=study, log=True, columns={"": text})
    a.set_xlabel("probability per measurement")
    changes = ["lead shield", "copper backing", "separate modules", "underground site", "pump off"]
    factor = [40.0, 6.5, 3.2, 12.0, 5.8]
    published = [30.0, np.nan, 2.5, np.nan, 7.0]
    sq.plots.bar_rows(b, changes, factor, base=1.0, log=True, marks=published,
                      columns={"quantity": ["burst rate", "phonon flux", "correlation", "charge jumps", "parity"]})
    b.set_xlabel("measured reduction factor")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
