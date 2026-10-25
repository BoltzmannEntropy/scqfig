"""Transmon frequency, anharmonicity and charge dispersion against E_J/E_C, exact in the charge basis with the large-ratio asymptotes dashed and two designs marked (computed)."""
import scqfig as sq

TITLE = "Scaling with E_J/E_C"
FAMILY = "Spectrum"


def build():
    designs = {"A": 45.0, "B": 90.0}
    fig, axes = sq.figure("page", aspect=0.3, ncols=3)
    for k, (ax, q) in enumerate(zip(axes, ("f01", "alpha", "dispersion"))):
        sq.plots.ratio_scaling(ax, q, points=designs, loc="lower right" if q != "dispersion" else "upper right")
        sq.panel(ax, "abc"[k])
    axes[1].set_ylim(-1.6, 0.2)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
