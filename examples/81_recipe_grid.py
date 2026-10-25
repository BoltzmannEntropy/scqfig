"""A generic transmon design recipe in eight numbered steps: each box holds instruction lines, a mini-plot from toy models and the tools the step uses, arrows run through the grid and two optional steps sit in dashed boxes below."""
import numpy as np

import scqfig as sq

TITLE = "Recipe grid"
FAMILY = "Schematics"


def band(ax):
    t = sq.theme()
    ax.axvspan(4.0, 6.0, color=t.tint, lw=0)
    for f in (4.8, 5.3):
        ax.plot([f, f], [0.1, 0.55], color=t.qubits[0], lw=1.1)
    ax.plot([7.2, 7.2], [0.45, 0.9], color=t.bus, lw=1.1)
    ax.set(xlim=(3.5, 8.0), ylim=(0, 1), yticks=[], xticks=[4, 6, 8], xlabel="GHz")


def junction(ax):
    ej = np.linspace(8, 30, 100)
    ax.plot(ej, np.sqrt(8 * ej * 0.25) - 0.25, **sq.model_kw())
    ax.plot(16, np.sqrt(8 * 16 * 0.25) - 0.25, **sq.sim_kw(ms=3.5))
    ax.set(xlim=(8, 30), xticks=[10, 20, 30], yticks=[4, 6], xlabel=r"$E_J/h$ (GHz)")


def hamiltonian(ax):
    sq.schematics.text_panel(ax, [r"$H/h=f_q\,b^\dagger b+\frac{\alpha}{2}\,b^{\dagger2}b^2$",
                                  r"$+\,f_r\,a^\dagger a+g\,(a^\dagger b+ab^\dagger)$"])


def readout(ax):
    f = np.linspace(-3, 2, 400)
    for f0, c in ((0.0, sq.theme().ink), (-1.6, sq.theme().accent)):
        ax.plot(f, 1 - 0.9 / (1 + ((f - f0) / 0.25) ** 2), color=c, lw=0.9)
    ax.set(xlim=(-3, 2), ylim=(0, 1.08), yticks=[], xticks=[-2, 0], xlabel=r"$f-f_r$ (MHz)")


def losses(ax):
    t = sq.theme()
    ax.bar(range(3), [60, 140, 42], 0.6, color=["white", "white", t.light], ec=t.ink, lw=0.5, log=True)
    ax.set(ylim=(5, 500), xticks=range(3), xticklabels=["TLS", "P", "sum"], ylabel=r"$T_1$ ($\mu$s)")
    ax.minorticks_off()


def spread(ax):
    rn = np.linspace(-4, 4, 200)
    ax.fill_between(rn, np.exp(-rn ** 2 / 2), color=sq.theme().light, lw=0)
    ax.plot(rn, np.exp(-rn ** 2 / 2), **sq.model_kw())
    ax.set(xlim=(-4, 4), ylim=(0, 1.15), yticks=[], xticks=[-3, 0, 3], xlabel=r"$\delta R_n$ (%)")


def validate(ax):
    rng = np.random.default_rng(3)
    for i in range(3):
        ax.plot(i + rng.uniform(-0.2, 0.2, 8), 1 + rng.normal(0, 0.03 * (i + 1), 8), **sq.data_kw(ms=2.5))
    ax.axhline(1, **sq.ref_kw())
    ax.set(xlim=(-0.5, 2.5), xticks=range(3), xticklabels=["$f$", "$g$", r"$\chi$"], ylim=(0.8, 1.2),
           yticks=[0.9, 1.0, 1.1], ylabel="pred./meas.")


def build():
    steps = [
        dict(title="Specify", lines=["Fix the band, the readout", "and the lifetime goals."], panel=band,
             tools=["pen", "paper"]),
        dict(title="Junction", lines=[r"Pick $E_J$ for the target", r"$f_{01}$ at fixed $E_C$."], panel=junction,
             tools=["pen", "code"]),
        dict(title="Capacitances", lines=["Solve the Maxwell matrix;", r"read $E_C$ from its inverse."],
             panel=lambda ax: sq.schematics.text_panel(ax, [r"$E_C=\dfrac{e^2}{2C_\Sigma}$"]), tools=["em"]),
        dict(title="Hamiltonian", lines=["Write every term and keep", "three levels per mode."], panel=hamiltonian,
             tools=["pen"]),
        dict(title="Readout", lines=[r"Size $\kappa$ near $2|\chi|$ and", "check the Purcell limit."], panel=readout,
             tools=["code", "vna"]),
        dict(title="Loss budget", lines=["Add surface losses and", "Purcell decay in series."], panel=losses,
             tools=["em", "stats"]),
        dict(title="Fabricate", lines=["Write the mask; probe the", "junctions at room temperature."],
             panel=spread, tools=["probe"]),
        dict(title="Validate", lines=["Compare prediction and", r"measurement; refit $E_J$."], panel=validate,
             tools=["fridge", "adc"]),
    ]
    side = [dict(after=2, title="Optional: tunable qubit", lines=["Replace the junction by a SQUID", "and budget flux noise."]),
            dict(after=3, title="Optional: package", lines=["Simulate the enclosure; keep", "its modes above the band."])]
    return sq.schematics.recipe_grid(None, steps, ncol=4, side=side)["fig"]


if __name__ == "__main__":
    sq.demo_main(build, __file__)
