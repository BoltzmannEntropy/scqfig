"""A generic chip design flow on a three-by-three grid: each box lists its quantities with a status tag (specified, assumed, simulated, derived, measured), arrows snake through the steps and a dashed arrow carries the measured frequency back to the junction (illustrative numbers)."""
import scqfig as sq

TITLE = "Design flow"
FAMILY = "Schematics"

TAGS = {"S": dict(color="ink", label="specified"), "A": dict(color="accent", label="assumed"),
        "EM": dict(color="model", label="EM simulation"), "D": dict(color="reference", label="derived"),
        "M": dict(color="ink", fill="light", label="measured")}


def build():
    nodes = {
        "concept": dict(cell=(0, 0), title="Concept", lines=[("one fixed transmon", "S"), ("one readout resonator", "S"),
                                                             ("shared feedline", "S")]),
        "spec": dict(cell=(0, 1), title="(1) Specification", lines=[(r"$f_{01}$ 4.5-5.5 GHz", "S"),
                                                                    (r"$E_J/E_C>50$", "S"), ("lifetime target", "A")]),
        "junction": dict(cell=(0, 2), title="(2) Junction", lines=[(r"$L_J=12.0$ nH", "D"), (r"$E_J/h=13.6$ GHz", "D"),
                                                                   ("critical-current spread", "A")]),
        "caps": dict(cell=(1, 2), title="(3) Electrostatics", lines=[(r"$C_\Sigma=85$ fF", "EM"), (r"$C_g=5.1$ fF", "EM"),
                                                                     (r"$E_C/h=228$ MHz", "D")]),
        "modes": dict(cell=(1, 1), title="(4) Eigenmodes", lines=[(r"$f_r=7.10$ GHz", "EM"), (r"$p_\mathrm{surf}=10^{-4}$", "EM"),
                                                                  ("junction as linear $L_J$", "A")]),
        "quant": dict(cell=(1, 0), title="(5) Quantization", lines=[(r"$f_{01}=4.97$ GHz", "D"), (r"$\alpha=-250$ MHz", "D"),
                                                                    (r"$g=90$ MHz", "D")]),
        "controls": dict(cell=(2, 0), title="(6) Controls", lines=[(r"$\pi$ pulse 40 ns", "D"), ("drive line 0.2 fF", "A"),
                                                                   (r"measured $f_{01}$", "M")]),
        "loss": dict(cell=(2, 1), title="(7) Loss budget", lines=[(r"$T_1^\mathrm{TLS}=60\ \mu$s", "D"),
                                                                  (r"$T_1^\mathrm{Purcell}=140\ \mu$s", "D"),
                                                                  (r"measured $T_1=35\ \mu$s", "M")]),
        "report": dict(cell=(2, 2), title="(8) Report", lines=[("every convention stated", "S"), ("inputs flagged", "S"),
                                                               ("open questions listed", "A")]),
    }
    order = ["concept", "spec", "junction", "caps", "modes", "quant", "controls", "loss", "report"]
    edges = [(a, b) for a, b in zip(order, order[1:]) if (a, b) != ("quant", "controls")]
    edges += [("quant", "controls", dict(offset=-0.75)),
              ("controls", "quant", dict(ls="dashed", offset=0.75, label=r"refit $L_J$",
                                         label_side="left"))]
    fig, ax = sq.figure("page", height=3.95)
    sq.schematics.flow_diagram(ax, nodes, edges, grid=dict(w=2.18, h=1.0, gap=(0.22, 0.3)), tags=TAGS, legend=True)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
