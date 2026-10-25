"""Qubit frequency against junction resistance for two transmons on an aluminium junction (gap 180 µeV), with L_J on the top axis, a 4-6 GHz band, an allowed L_J window and the designs with a 3 % junction scatter (computed)."""
import numpy as np

import scqfig as sq

TITLE = "Junction design"
FAMILY = "Design"


def build():
    Rn = np.linspace(4.0, 14.0, 160)                                # kΩ
    fig, ax = sq.figure("column", aspect=0.75)
    out = sq.plots.junction_curve(ax, Rn, {"Q1": 0.20, "Q2": 0.26}, gap=180e-6, points={"Q1": 7.2, "Q2": 10.5},
                                  spread=0.03, window=(7, 15), band=(4, 6))
    q1 = out["labels"]["Q1"]                                        # below its curve, clear of Q2's
    q1.set_position((-5, -4))
    q1.set_ha("right")
    q1.set_va("top")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
