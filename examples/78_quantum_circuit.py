"""GHZ-state preparation: a Hadamard and two CNOTs entangle three qubits, then all three are measured; dashed slices mark the states Psi_1 to Psi_4."""
import scqfig as sq
from scqfig.schematics import QCircuit

TITLE = "Quantum circuit"
FAMILY = "Schematics"


def build():
    c = QCircuit(3, labels=[r"$|0\rangle$"] * 3)
    c.gate("H", 0)
    c.slice(r"$\Psi_1$")
    c.cnot(0, 1)
    c.slice(r"$\Psi_2$")
    c.cnot(0, 2)
    c.slice(r"$\Psi_3$")
    c.measure([0, 1, 2])
    c.slice(r"$\Psi_4$")
    w, h = c.figsize()
    fig, ax = sq.figure(w, height=h)
    c.draw(ax)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
