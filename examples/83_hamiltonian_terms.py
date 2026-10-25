"""A generic transmon-resonator Hamiltonian: (a) every term on the tile of the circuit element that produces it, bracketed by origin, (b) the one-excitation block as a bracketed matrix with its bare and dressed frequencies, (c) the drive term and the inputs that stay assumed (illustrative numbers)."""
import matplotlib.pyplot as plt
import numpy as np

import scqfig as sq

TITLE = "Hamiltonian terms"
FAMILY = "Schematics"

FQ, ALPHA, FR, G = 5.200, -0.210, 7.100, 0.090    # GHz


def build():
    t = sq.theme()
    q = t.qubits[0]
    fig = plt.figure(figsize=(sq.WIDTHS["page"], 3.9))
    axa = fig.add_axes([0.02, 0.56, 0.96, 0.38])
    axb = fig.add_axes([0.02, 0.0, 0.46, 0.48])
    axc = fig.add_axes([0.52, 0.0, 0.46, 0.48])

    sq.schematics.term_tiles(axa, [
        ["$H/h=$",
         dict(text=rf"${FQ:.3f}\,b^\dagger b-\dfrac{{{abs(ALPHA):.3f}}}{{2}}\,b^\dagger b^\dagger b\,b$", kind="nonlinear",
              color=q, brace="transmon: junction and shunt capacitor"),
         "$+$",
         dict(text=rf"${FR:.3f}\,a^\dagger a$", kind="inductive", color="bus", brace="readout resonator")],
        ["$+$",
         dict(text=rf"${G:.3f}\,(a^\dagger b+a\,b^\dagger)$", kind="coupling",
              brace=r"coupling capacitor: $g\propto C_g/\sqrt{C_\Sigma C_r}$"),
         dict(text="rotating-wave form", color="reference", gap=0.2)],
    ])

    block = np.array([[FQ, G], [G, FR]])
    dressed = np.linalg.eigvalsh(block)
    labels = [r"$q$", r"$r$"]
    sq.schematics.matrix_text(axb, block, rows=labels, cols=labels, label_colors=[q, "bus"], fmt="{:.3f}",
                              title="one-excitation block (GHz)",
                              table=(["mode", "bare (GHz)", "dressed (GHz)"],
                                     [[labels[i], f"{block[i, i]:.3f}", f"{dressed[i]:.3f}"] for i in range(2)]))

    sq.schematics.text_panel(axc, [r"$H_d(t)/h=\Omega(t)\cos(2\pi f_d t)\,(b+b^\dagger)$",
                                   r"$\Omega(t)\propto C_d\,V_d(t)\,\langle0|n|1\rangle$",
                                   (r"assumed: $T_1$, $T_2^*$, readout fidelity", "reference")],
                             align="left", valign="top", frame=False, fs=t.size)
    for x, y, letter in ((0.0, 0.95, "a"), (0.0, 0.49, "b"), (0.5, 0.49, "c")):
        sq.fig_letter(fig, x, y, letter)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
