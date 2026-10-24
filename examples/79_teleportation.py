"""(a) Teleportation: a Bell pair, a Bell measurement and classically controlled X and Z corrections. (b) The other gate glyphs: rotation, controlled-U with an open control, CZ, SWAP and a three-qubit box."""
import scqfig as sq
from scqfig.schematics import QCircuit

TITLE = "Teleportation and gate glyphs"
FAMILY = "Schematics"


def teleport():
    c = QCircuit(3, labels=[r"$|\psi\rangle$", r"$|0\rangle$", r"$|0\rangle$"], outputs=[None, None, r"$|\psi\rangle$"])
    c.gate("H", 1)
    c.cnot(1, 2)
    c.group([1, 2], cols=(0, 1), label="Bell pair")
    c.slice()
    c.cnot(0, 1)
    c.gate("H", 0)
    c.measure([0, 1])
    c.gate("X", 2, cond=1)
    c.gate("Z", 2, cond=0)
    return c


def glyphs():
    c = QCircuit(3, labels=[r"$q_0$", r"$q_1$", r"$q_2$"])
    c.gate(r"$R_x(\pi/2)$", 0)
    c.gate("T", 1)
    c.controlled("U", [0], 2, open_controls=[1])
    c.cz(0, 1)
    c.swap(1, 2)
    c.multi("V", [0, 1, 2])
    c.measure(2)
    return c


def build():
    a, b = teleport(), glyphs()
    (wa, ha), (wb, hb) = a.figsize(), b.figsize()
    fig, (ax_a, ax_b) = sq.figure(max(wa, wb), height=ha + hb + 0.25, nrows=2,
                                  gridspec_kw={"height_ratios": [ha, hb]})
    a.draw(ax_a)
    b.draw(ax_b)
    sq.panel(ax_a, "a")
    sq.panel(ax_b, "b")
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
