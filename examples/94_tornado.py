"""One-at-a-time sensitivity of a projected coupling reach to six design inputs, each moved to the ends of its range, with the range and the design step that sets each input (computed from a power-law model)."""
import scqfig as sq

TITLE = "Tornado"
FAMILY = "Comparison"

# input: (exponent of the reach in the input, low / nominal, high / nominal, range text, design step)
INPUTS = {
    "antenna length": (-1.0, 0.5, 1.5, r"150 / 450 µm", "layout"),
    "coherent window": (-0.5, 0.2, 2.0, r"20 / 200 µs", "materials"),
    "dark-count probability": (0.25, 0.03, 10.0, r"$10^{-4}$ / 0.03", "readout"),
    "temperature": (0.6, 0.4, 2.0, "10 / 50 mK", "cryostat"),
    "capacitance": (0.5, 0.9, 1.1, r"$\pm$10 %", "layout"),
    "junction inductance": (0.15, 0.9, 1.1, r"$\pm$10 %", "fabrication"),
}


def build():
    names = list(INPUTS)
    low = [lo ** p for p, lo, _, _, _ in INPUTS.values()]
    high = [hi ** p for p, _, hi, _, _ in INPUTS.values()]
    fig, ax = sq.figure("page", aspect=0.32)
    sq.plots.tornado(ax, names, low, high, columns={"range": [v[3] for v in INPUTS.values()],
                                                     "set by": [v[4] for v in INPUTS.values()]},
                     names=("low end", "high end"), base_label="base reach = 1")
    ax.set_xlabel("factor on the reach (smaller is better)")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
