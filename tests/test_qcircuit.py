"""Gate-model circuit diagrams: column packing, slices, gate glyphs and figure size."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.schematics import QCircuit  # noqa: E402


def ghz():
    c = QCircuit(3, labels=[r"$|0\rangle$"] * 3)
    c.gate("H", 0)
    c.slice(r"$\Psi_1$")
    c.cnot(0, 1)
    c.slice(r"$\Psi_2$")
    c.cnot(0, 2)
    c.slice(r"$\Psi_3$")
    c.measure([0, 1, 2])
    c.slice(r"$\Psi_4$")
    return c


def test_ghz_columns_and_slices_follow_the_drawing_order():
    c = ghz()
    assert [op["col"] for op in c.ops] == [0, 1, 2, 3, 3, 3]
    assert [s["after"] for s in c.slices] == [0, 1, 2, 3]
    assert c.ncols == 4


def test_independent_gates_share_a_column_and_spans_block_wires():
    c = QCircuit(3)
    c.gate("X", 0)
    c.gate("Y", 2)                 # free wire: same column
    c.cnot(0, 2)                   # spans q1 too
    c.gate("Z", 1)                 # must wait for the CNOT
    assert [op["col"] for op in c.ops] == [0, 0, 1, 2]


def test_explicit_column_and_list_targets():
    c = QCircuit(2)
    c.gate("H", [0, 1])
    c.gate("T", 1, col=4)
    assert [op["col"] for op in c.ops] == [0, 0, 4]
    assert c.ncols == 5


def test_bad_qubit_index_raises():
    c = QCircuit(2)
    with pytest.raises(ValueError):
        c.cnot(0, 2)
    with pytest.raises(ValueError):
        c.cnot(1, 1)


@pytest.mark.parametrize("style", ["field", "folio"])
def test_draw_places_glyphs(style):
    c = ghz()
    with sq.style(style):
        fig, ax = plt.subplots(figsize=c.figsize())
        g = c.draw(ax)
        texts = [t.get_text() for t in ax.texts]
        plt.close(fig)
    assert len(g["controls"]) == 2 and len(g["targets"]) == 2 and len(g["meters"]) == 3
    assert len(g["slices"]) == 4
    assert r"$\Psi_3$" in texts and "H" in " ".join(texts)
    assert g["x"][0] < g["x"][1] < g["x"][2] < g["x"][3]
    assert not ax.axison


def test_wide_gate_widens_its_column_and_the_figure():
    narrow, wide = QCircuit(1), QCircuit(1)
    narrow.gate("H", 0)
    wide.gate(r"$R_x(\pi/2)$", 0)
    assert wide.figsize()[0] > narrow.figsize()[0]


def test_classical_control_and_groups_draw():
    c = QCircuit(3, labels=[r"$|\psi\rangle$", r"$|0\rangle$", r"$|0\rangle$"])
    c.gate("H", 1)
    c.cnot(1, 2)
    c.group([1, 2], cols=(0, 1), label="Bell pair")
    c.cnot(0, 1)
    c.gate("H", 0)
    c.swap(0, 1)
    c.cz(0, 2)
    c.controlled("U", [0, 1], 2)
    c.multi("V", [1, 2])
    c.measure([0, 1])
    c.gate("X", 2, cond=1)
    c.gate("Z", 2, cond=0)
    with sq.style("field"):
        fig, ax = plt.subplots(figsize=c.figsize())
        g = c.draw(ax)
        plt.close(fig)
    assert len(g["groups"]) == 1 and len(g["classical"]) == 2 and len(g["swaps"]) == 1
    # a measured wire turns classical: no gate may act on it as a quantum wire afterwards
    with pytest.raises(ValueError):
        c.gate("H", 0)
