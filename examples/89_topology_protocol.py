"""(a) Coupling graphs of five chip topologies: two qubits without a coupler, a chain, a star, a shared bus and a 2 x 3 lattice. (b) The four-qubit chain's sensing circuit, each wire in its qubit's colour: GHZ encoding, a shaded exposure window where a field acts on every wire at once, decoding and readout."""
import scqfig as sq
from scqfig.schematics import QCircuit, chip, symbols

TITLE = "Topologies and a coloured circuit"
FAMILY = "Schematics"

GRAPHS = [  # name, positions, couplings, keyword options
    ("pair", [(-1.0, 0.0), (1.0, 0.0)], [], dict(dashed=[(0, 1)])),
    ("chain", [(-2.1, 0.0), (-0.7, 0.0), (0.7, 0.0), (2.1, 0.0)], [(0, 1), (1, 2), (2, 3)], {}),
    ("star", [(0.0, 0.0), (-1.2, 0.9), (1.2, 0.9), (-1.2, -0.9), (1.2, -0.9)], [(0, 1), (0, 2), (0, 3), (0, 4)], {}),
    ("bus", [(-1.2, 0.9), (1.2, 0.9), (-1.2, -0.9), (1.2, -0.9)], [], dict(hub=(0.0, 0.0))),
    ("lattice", [(-1.4, 0.8), (0.0, 0.8), (1.4, 0.8), (-1.4, -0.8), (0.0, -0.8), (1.4, -0.8)],
     [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)], {}),
]


def protocol(colors):
    c = QCircuit(4, labels=[r"$|0\rangle$"] * 4, wire_colors=colors)
    c.gate("H", 0)
    for k in range(3):
        c.cnot(k, k + 1)
    c.gate("H", [0, 1, 2, 3])
    c.gate(r"$U_\mathrm{DM}$", [0, 1, 2, 3], kind="plain", fill=sq.theme().wash)
    c.gate("H", [0, 1, 2, 3])
    for k in (2, 1, 0):
        c.cnot(k, k + 1)
    c.gate("H", 0)
    c.measure([0, 1, 2, 3])
    c.window(5, label=r"exposure $t_\mathrm{exp}$")
    return c


def build():
    t = sq.theme()
    colors = list(t.qubits[:4])
    c = protocol(colors)
    cw, ch = c.figsize(scale=0.33)
    fig, (axa, axb) = sq.figure("page", height=ch + 1.25, nrows=2, height_ratios=[1.0, ch / 1.05])
    symbols.blank(axa)
    for k, (name, pos, edges, opts) in enumerate(GRAPHS):
        x0 = 6.0 * k
        P = [(x0 + x, y) for x, y in pos]
        if "hub" in opts:
            opts = dict(opts, hub=(x0 + opts["hub"][0], opts["hub"][1]))
        chip.coupling_graph(axa, P, edges, r=0.3, colors=colors if name == "chain" else None, **opts)
        axa.text(x0, -1.65, name, ha="center", va="top", fontsize=t.size - 0.5, color=t.ink)
    axa.set_xlim(-3.0, 27.0)
    axa.set_ylim(-2.3, 1.4)
    c.draw(axb)
    sq.panel(axa, "a")
    sq.panel(axb, "b")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
