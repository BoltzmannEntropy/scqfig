"""Frequency plan of a three-qubit chain: each transmon's f01 and f12, the readout resonators inside the Purcell-filter passband, and the collision windows each qubit's f01 must avoid (designed values)."""
import scqfig as sq
from scqfig import physics as ph
from scqfig.plots.design import collision_windows

TITLE = "Frequency plan"
FAMILY = "Design"


def build():
    EJ, EC = {"Q1": 15.6, "Q2": 16.6, "Q3": 16.0}, 0.21          # GHz
    qubits = {}
    for name, ej in EJ.items():
        f = ph.transmon_spectrum(ej, EC, nlev=3)
        qubits[name] = (f[1], f[2] - f[1])                       # (f01, f12)
    coupled = {"Q1": ["Q2"], "Q2": ["Q1", "Q3"], "Q3": ["Q2"]}
    collisions = {}
    for name, nbrs in coupled.items():
        wins = []
        for nb in nbrs:
            alpha = 0.5 * sum(qubits[q][1] - qubits[q][0] for q in (name, nb))
            wins += [(qubits[nb][0] + c - w, qubits[nb][0] + c + w) for _, c, w, _ in collision_windows(alpha)]
        collisions[name] = wins
    resonators = {"R1": 7.05, "R2": 7.15, "R3": 7.25}
    fig, ax = sq.figure("page", aspect=0.3)
    sq.plots.frequency_plan(ax, qubits, resonators=resonators, purcell=(6.95, 7.35), collisions=collisions)
    ax.set_xlim(4.5, 7.5)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
