"""Frequency plan of a three-qubit sensor chip on one axis: f01 and f12 lines with their fabrication scatter, one flux scan, the collision windows of the tunable qubit, the readout resonators in a Purcell passband, the 20 mK thermal floor and the rest-energy axis (designed values)."""
import scqfig as sq
from scqfig import physics as ph
from scqfig.plots.design import collision_windows

TITLE = "Frequency lines"
FAMILY = "Design"


def build():
    EJ, EC = {"Q1": 13.2, "Q2": 15.8, "Q3": 18.6}, 0.22              # GHz
    qubits = {}
    for name, ej in EJ.items():
        f = ph.transmon_spectrum(ej, EC, nlev=3)
        qubits[name] = (f[1], f[2] - f[1])                            # (f01, f12)
    f1, f12 = qubits["Q1"]
    windows = [(f1 + c - w, f1 + c + w) for _, c, w, _ in collision_windows(f12 - f1)]   # where Q2 must not sit
    fig, ax = sq.figure("page", aspect=0.24)
    sq.plots.frequency_lines(ax, qubits, resonators={"R1": 6.85, "R2": 6.95, "R3": 7.05}, spread=0.04,
                             scans={"Q2": (qubits["Q2"][0], f1 + 0.3)}, windows=windows,
                             bands={"Purcell filter": (6.75, 7.15)}, floor=ph.thermal_frequency(1e-3, 0.02),
                             guides={"box mode": 7.6}, mass_axis=True, xlim=(2.6, 8.0))
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
