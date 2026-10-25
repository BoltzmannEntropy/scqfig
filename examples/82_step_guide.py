"""A generic validation guide in two columns: the numbered steps run down each column with a mini-plot beside the text and the tools as labelled chips, and two optional steps sit in dashed side boxes joined to their step."""
import numpy as np

import scqfig as sq

TITLE = "Step guide"
FAMILY = "Schematics"


def calibrate(ax):
    lj = np.linspace(6, 16, 100)
    f = lambda L: 5.0 * np.sqrt(10 / L)
    ax.axvspan(9.2, 10.8, color=sq.theme().light, lw=0)
    ax.plot(lj, f(lj), **sq.model_kw())
    ax.plot(10, f(10), **sq.sim_kw(ms=3.5))
    ax.set(xlim=(6, 16), xticks=[8, 12, 16], yticks=[4, 6], xlabel=r"$L_J$ (nH)")


def strip(ax, spread, labels):
    rng = np.random.default_rng(7)
    for i, s in enumerate(spread):
        ax.plot(i + rng.uniform(-0.18, 0.18, 9), 1 + rng.normal(0, s, 9), **sq.data_kw(ms=2.5))
    ax.axhline(1, **sq.ref_kw())
    ax.set(xlim=(-0.5, len(spread) - 0.5), xticks=range(len(spread)), xticklabels=labels, ylim=(0.8, 1.2),
           yticks=[0.9, 1.1])
    ax.tick_params(axis="x", length=0)


def lifetimes(ax):
    t = sq.theme()
    pred, meas = [40, 25, 60], [18, 15, 22]
    for i, (p, m) in enumerate(zip(pred, meas)):
        ax.plot([i, i], [m, p], color=t.reference, lw=0.8)
        ax.plot(i, p, **sq.sim_kw(ms=3.2))
        ax.plot(i, m, **sq.data_kw(ms=3.2))
    ax.set(yscale="log", ylim=(8, 100), xlim=(-0.6, 2.6), xticks=[], ylabel=r"$T_1$ ($\mu$s)")
    ax.minorticks_off()


def misses(ax):
    t = sq.theme()
    ax.bar(range(3), [1.5, 12, 80], 0.55, color=t.light, ec=t.ink, lw=0.5, log=True)
    ax.set(ylim=(0.5, 300), xticks=range(3), xticklabels=["$f$", "$g$", "$T_1$"], ylabel="miss (%)")
    ax.minorticks_off()


def drift(ax):
    d = np.linspace(-5, 5, 50)
    ax.axvspan(-2, 2, color=sq.theme().tint, lw=0)
    ax.plot(d, -25 * d, **sq.model_kw())
    ax.set(xlim=(-5, 5), xticks=[-4, 0, 4], yticks=[-100, 100], xlabel=r"$\delta L_J$ (%)")


def build():
    steps = [
        dict(title="Calibrate the junction", lines=[r"Fit $L_J$ to the measured $f_{01}$", "and keep both values."],
             panel=calibrate, tools=["probe station", "VNA"]),
        dict(title="Compare frequencies", lines=["Qubit and resonator should", "agree within a few per cent."],
             panel=lambda ax: strip(ax, [0.02, 0.01], ["qubit", "res."]), tools=["spectroscopy"]),
        dict(title="Test the couplings", lines=["Fit $g$ from the avoided", r"crossing, not from $\chi$."],
             panel=lambda ax: strip(ax, [0.06], ["$g$"]), tools=["flux sweep", "EM solver"]),
        dict(title="Test the lifetimes", lines=["Set $T_1$ against a dielectric", "plus Purcell budget."],
             panel=lifetimes, tools=["$T_1$ decay"]),
        dict(title="Rank the misses", lines=["Order the quantities by", "how far the model missed."],
             panel=misses, tools=["statistics"]),
        dict(title="Report", lines=["State every convention and", "the inputs you assumed."], panel=None,
             tools=["paper draft"]),
    ]
    side = [dict(after=1, at="left", title="Optional: ageing", lines=["Junctions drift after", "fabrication."],
                 panel=drift, tools=["re-probe"]),
            dict(after=3, at="left", title="Optional: hybrid solve", lines=["Resolve the edge fields", "for surface losses."])]
    res = sq.schematics.recipe_grid(None, steps, ncol=2, order="column", plot="right", box=(2.55, 1.36),
                                    gap=(0.22, 0.3), side=side, side_width=1.32)
    return res["fig"]


if __name__ == "__main__":
    sq.demo_main(build, __file__)
