"""Transmon ladder (E_J/E_C = 40) to scale, with drive photons for f01, f12 and the two-photon f02/2, and the parity-forbidden single-photon 0-2 transition."""
import scqfig as sq
from scqfig import physics as ph
from scqfig.schematics import motifs, symbols

TITLE = "Level ladder and transitions"
FAMILY = "Schematics"

EJ, EC = 12.0, 0.30                                           # GHz


def build():
    t = sq.theme()
    E = ph.transmon_spectrum(EJ, EC, nlev=3)                   # [0, f01, f02], computed in the charge basis
    f01, f02 = E[1], E[2]
    fig, ax = sq.figure("column", aspect=0.78)
    symbols.blank(ax)
    fs = t.size - 0.5
    x0, x1 = 0.3, 14.4
    c = [2.5, 6.1, 9.7, 13.0]                                  # transition columns
    gap = (c[2] - 1.05, c[2] + 1.05)                           # |1> gives way to the virtual level here
    for n, En in enumerate(E):
        segs = [(x0, gap[0]), (gap[1], x1)] if n == 1 else [(x0, x1)]
        for a, b in segs:
            motifs.ladder(ax, a, b, [En], colors=[t.levels[n]])
        ax.text(x1 + 0.3, En, rf"$|{n}\rangle$", ha="left", va="center", fontsize=fs + 0.5, color=t.levels[n])

    def photon(xc, y, f, label=None):
        motifs.wavy(ax, (xc - 2.3, y), (xc - 0.22, y), amp=0.2, waves=f / 1.25, lw=1.0, head=0.4)
        if label:
            ax.text(xc - 1.3, y + 0.4, label, ha="center", va="bottom", fontsize=fs - 0.5, color=t.signal,
                    linespacing=1.15)

    def ghz(f):
        return f"{f:.2f} GHz"

    motifs.transition(ax, (c[0], 0), (c[0], f01))
    photon(c[0], f01 / 2, f01, r"$f_{01}$" + "\n" + ghz(f01))
    motifs.transition(ax, (c[1], f01), (c[1], f02))
    photon(c[1], (f01 + f02) / 2, f02 - f01, r"$f_{12}$" + "\n" + ghz(f02 - f01))
    ax.plot([c[2] - 0.9, c[2] + 0.9], [f02 / 2] * 2, color=t.reference, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.text(c[2] + 0.95, f02 / 2 - 0.2, "virtual", ha="left", va="top", fontsize=fs - 1.5, color=t.reference)
    motifs.transition(ax, (c[2], 0), (c[2], f02 / 2))
    motifs.transition(ax, (c[2], f02 / 2), (c[2], f02))
    photon(c[2], f02 / 4, f02 / 2, r"$f_{02}/2$" + "\n" + ghz(f02 / 2))
    photon(c[2], 3 * f02 / 4, f02 / 2)
    motifs.transition(ax, (c[3], 0), (c[3], f02), allowed=False)
    photon(c[3], f02 / 4, f02, r"$f_{02}$" + "\n" + ghz(f02))
    ax.text(c[3] + 0.3, f02 * 0.75, "forbidden", rotation=90, ha="left", va="center", fontsize=fs - 1.5, color=t.ink)
    sq.note(ax, rf"$E_J/E_C$ = {EJ / EC:.0f},  $\alpha/h$ = {(f02 - 2 * f01) * 1e3:.0f} MHz".replace("-", "−"), loc="upper left",
            color=t.reference)
    ax.set_xlim(-0.2, x1 + 1.3)
    ax.set_ylim(-0.6, f02 + 1.5)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
