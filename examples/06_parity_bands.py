"""Even and odd charge-parity branches of a charge-sensitive transmon at E_J/E_C = 20: the 0-1 frequency with its splitting 2 chi_0 and the zero-contrast offsets, and the first excited level (computed)."""
import scqfig as sq

TITLE = "Parity bands"
FAMILY = "Spectrum"


def build():
    EJ, EC = 12.0, 0.6                                              # GHz
    fig, axes = sq.figure("page", aspect=0.3, ncols=2)
    sq.plots.parity_bands(axes[0], EJ, EC)
    sq.plots.parity_bands(axes[1], EJ, EC, level=1, zero_contrast=False)
    for k, ax in enumerate(axes):
        sq.panel(ax, "ab"[k])
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
