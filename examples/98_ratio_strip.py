"""Predicted over measured values by parameter type with each point's status and published scatters as grey bars, and forecast (open) against measured lifetime per device joined by a stem, grouped by chip (synthetic numbers)."""
import numpy as np

import scqfig as sq

TITLE = "Ratio strip and stems"
FAMILY = "Comparison"


def build():
    rng = np.random.default_rng(98)
    ratios = {
        "qubit\nfrequency": 1 + rng.normal(0, 0.02, 7),
        "resonator\nfrequency": 1 + rng.normal(0, 0.008, 6),
        "coupling": 1 + rng.normal(0, 0.08, 5),
        "anharmo-\nnicity": 1 + rng.normal(0, 0.04, 5),
        r"$T_1$": np.exp(rng.normal(0.9, 0.6, 5)),
    }
    status = {
        "qubit\nfrequency": ["measured"] * 4 + ["fitted"] * 2 + ["excluded"],
        "coupling": ["measured", "measured", "simulated", "simulated", "excluded"],
        r"$T_1$": ["simulated", "simulated", "measured", "upper", "lower"],
    }
    fig, (a, b) = sq.figure("page", aspect=0.36, ncols=2, gridspec_kw={"width_ratios": [1.5, 1]})
    sq.plots.ratio_strip(a, ratios, status=status, rms={"coupling": 0.06, "anharmo-\nnicity": 0.03},
                         legend={"simulated": "forecast", "excluded": "dependent"})
    devices = [f"D{i}" for i in range(1, 7)]
    forecast = [120.0, 85.0, 60.0, 210.0, 150.0, np.nan]
    measured = [45.0, 52.0, 18.0, 90.0, 160.0, 33.0]
    sq.plots.paired_stems(b, devices, forecast, measured, groups=["Chip 1"] * 3 + ["Chip 2"] * 3, log=True,
                          rotation=0)
    b.set_ylim(8, 600)
    b.set_ylabel(r"$T_1$ (µs)")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
