"""Pulse sequence of a conditional-phase measurement: DRAG pulses on two qubits, a CZ flux pulse and a dispersive readout with its integration window."""
import scqfig as sq
from scqfig.schematics import pulses

TITLE = "Pulse sequence"
FAMILY = "Schematics"


def build():
    s = 5.0                                           # Gaussian width (ns); pulses span 4 sigma
    seq = [dict(row=r"XY$_1$", kind="drag", t0=12, sigma=s, amp=0.42, label=r"$\pi/2$"),
           dict(row=r"XY$_2$", kind="drag", t0=12, sigma=s, amp=0.84, label=r"$\pi$"),
           dict(row=r"Z$_1$", kind="square", t0=26, length=40, amp=0.7, rise=3, label="CZ"),
           dict(row=r"XY$_1$", kind="drag", t0=80, sigma=s, amp=0.42, label=r"$\pi/2$"),
           dict(row="RO", kind="readout", t0=96, length=1000, amp=0.6, rise=8, window=(110, 1080), label="readout")]
    fig, ax = sq.figure("column", aspect=0.66)
    pulses.pulse_sequence(ax, [r"XY$_1$", r"XY$_2$", r"Z$_1$", "RO"], seq, sync=[24, 68, 92], cut=(170, 1000),
                          tmax=1110)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
