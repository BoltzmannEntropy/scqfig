"""Predicted (open) against measured (filled) efficiency per detector, energy thresholds per sensor as points and ranges with where each energy is counted, and efficiency against dark-count rate with arrows on the two bounds (synthetic numbers)."""
import numpy as np

import scqfig as sq

TITLE = "Dot rows and bounds"
FAMILY = "Comparison"


def build():
    fig, (a, b, c) = sq.figure("page", aspect=0.3, ncols=3, gridspec_kw={"width_ratios": [0.8, 1.25, 1]})
    detectors = ["Detector 1", "Detector 2", "Detector 3"]
    sq.plots.dot_rows(a, detectors, [[0.55, 0.43], [0.61, 0.66], [0.42, 0.40]],
                      status=[["simulated", "measured"]] * 3, connect=True)
    a.set_xlim(0.3, 0.75)
    a.set_xlabel(r"efficiency $\eta$")
    sensors = ["Sensor A", "Sensor B", "Sensor C", "Sensor D", "Sensor E"]
    lo = [np.nan, 3e-3, 4e-2, 0.4, 40.0]
    hi = [np.nan, 3e-2, 0.17, 1.5, 1100.0]
    sq.plots.dot_rows(b, sensors, [1e-3, np.nan, np.nan, np.nan, np.nan], lo=lo, hi=hi, status="simulated", log=True,
                      ref=3.6e-4, columns={"counted in": ["absorber", "substrate", "absorber", "qubit", "substrate"]})
    b.set_xlabel("threshold energy (eV)")
    rate = np.array([0.08, 0.6, 3.0, 40.0])
    eta = np.array([0.43, 0.38, 0.55, 0.62])
    c.plot(rate, eta, **sq.data_kw())
    sq.plots.limit_arrow(c, 0.3, 0.2, "up")
    sq.plots.limit_arrow(c, 300.0, 0.05, "left")
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlim(0.03, 3e3)
    c.set_ylim(0.02, 1.2)
    sq.log_minor(c)
    c.set_xlabel(r"dark-count rate (s$^{-1}$)")
    c.set_ylabel(r"efficiency $\eta$")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
