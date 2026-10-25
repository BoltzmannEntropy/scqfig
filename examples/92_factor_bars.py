"""Signal of two devices against a reference design as signed log factors with the net change, and one device's budget as bars coloured by sign (synthetic numbers)."""
import scqfig as sq

TITLE = "Factor budget"
FAMILY = "Comparison"


def build():
    fig, (a, b) = sq.figure("page", aspect=0.34, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
    labels = ["capacitance", "dipole length", "frequency", "ground-plane factor", "enclosure factor",
              "coherent window"]
    factors = {"Device 1": [0.65, 4.8, 3.8, 4e-4, 2.6e4, 0.02], "Device 2": [0.65, 4.8, 4.0, 4e-4, 230.0, 0.3]}
    sq.plots.factor_bars(a, labels, factors, legend="lower right")
    short = [r"$C$", r"$d^2$", r"$f$", r"$\kappa_1^2$", r"$\kappa_2^2$", r"$\tau^2$"]
    sq.plots.factor_bars(b, short, factors["Device 2"], horizontal=False, values=False)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
