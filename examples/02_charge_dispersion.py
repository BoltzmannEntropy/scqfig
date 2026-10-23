"""Charge dispersion of the three lowest levels as E_J/E_C grows from the Cooper-pair box to the transmon."""
import scqfig as sq

TITLE = "Charge dispersion"
FAMILY = "Spectrum"


def build():
    fig, axes = sq.figure("page", aspect=0.24, ncols=4, sharey=True)
    for k, (ax, r) in enumerate(zip(axes, (1, 5, 10, 50))):
        sq.plots.charge_dispersion(ax, r, nlev=3)
        sq.panel(ax, "abcd"[k])
        if k:
            ax.set_ylabel("")
    axes[0].set_ylim(-0.12, 3.2)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
