"""Gate-model quantum circuits in the active style, in the manner of quantikz: wires with input
labels, gate boxes on pale tiles, controls and targets, SWAP and CZ, multi-qubit boxes, measurement
meters that turn a wire classical, classically controlled gates, dashed slices with tilted labels,
and dashed groups around a stretch of the circuit.

    c = QCircuit(3, labels=[r"$|0\\rangle$"] * 3)
    c.gate("H", 0); c.slice(r"$\\Psi_1$")
    c.cnot(0, 1); c.cnot(0, 2); c.measure([0, 1, 2])
    w, h = c.figsize()
    fig, ax = sq.figure(w, height=h)
    c.draw(ax)

Operations fill columns from the left. Each takes the first column where every wire it spans is
free, so gates on different wires share a column, and a slice keeps later operations to its right.
Pass col= to place one by hand. Sizes are in wire spacings; figsize() converts them to inches so that
labels keep their size relative to the boxes."""
import re

import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch

from ..core import theme
from .symbols import arrow

__all__ = ["QCircuit"]

KINDS = {"clifford": "nonlinear", "rotation": "inductive", "multi": "capacitive", "plain": None}


def _visible_len(s):
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+", "x", s)
    return max(len(re.sub(r"[{}^_\s]", "", s)), 1)


def _math(name):
    """Short symbolic names (H, X, CZ, U) are set in math italic, like quantikz; anything with
    dollars or spaces is used as given."""
    return name if ("$" in name or " " in name) else f"${name}$"


class QCircuit:
    BOX = 0.62        # minimum gate box, in wire spacings
    GAP = 0.5         # space between neighbouring columns
    SLICE_GAP = 0.3   # extra space where a slice sits

    def __init__(self, n, labels=None, outputs=None):
        self.n = int(n)
        self.labels = list(labels) if labels is not None else [None] * self.n
        self.outputs = list(outputs) if outputs is not None else [None] * self.n
        self.ops, self.slices, self.groups = [], [], []
        self._free = [0] * self.n
        self._floor = 0
        self._measured = {}

    # ---- building ------------------------------------------------------------------------------

    def _check(self, qubits, quantum=True):
        qs = list(qubits)
        for q in qs:
            if not isinstance(q, (int, np.integer)) or not 0 <= q < self.n:
                raise ValueError(f"qubit {q!r} is not on a circuit of {self.n} wires")
            if quantum and q in self._measured:
                raise ValueError(f"qubit {q} was measured; its wire is classical from there on")
        if len(set(qs)) != len(qs):
            raise ValueError(f"an operation names the same qubit twice: {qs}")
        return qs

    def _place(self, rows, col):
        lo, hi = min(rows), max(rows)
        span = range(lo, hi + 1)
        if col is None:
            col = max([self._free[q] for q in span] + [self._floor])
        for q in span:
            self._free[q] = max(self._free[q], col + 1)
        return col

    def _add(self, op, rows, col):
        op["col"] = self._place(rows, col)
        self.ops.append(op)
        return op

    @property
    def ncols(self):
        return max(self._free + [self._floor])

    def gate(self, name, q, *, col=None, kind=None, fill=None, cond=None):
        """A single-qubit gate on q (or on each qubit of a list, in one column). kind picks the tile:
        "clifford" (default), "rotation" (default for names starting with R), "plain" for white;
        fill overrides it. cond=k makes the gate classically controlled by the measured wire k."""
        qs = self._check([q] if np.ndim(q) == 0 else q)
        if cond is not None:
            self._check([cond], quantum=False)
            if cond not in self._measured:
                raise ValueError(f"cond={cond}: wire {cond} has not been measured")
        rows = qs + ([cond] if cond is not None else [])
        c = self._place(rows, col)
        kind = kind or ("rotation" if name.lstrip("$\\").startswith("R") else "clifford")
        for qq in qs:
            self.ops.append(dict(type="gate", q=[qq], name=name, kind=kind, fill=fill, cond=cond, col=c))
        return self

    def multi(self, name, qubits, *, col=None, kind="multi", fill=None):
        """One box across the contiguous wires from min(qubits) to max(qubits)."""
        qs = self._check(qubits)
        rows = list(range(min(qs), max(qs) + 1))
        self._check(rows)
        self._add(dict(type="multi", q=rows, name=name, kind=kind, fill=fill), rows, col)
        return self

    def cnot(self, control, target, *, col=None):
        """Controlled NOT: a control dot and a target circle."""
        self._check([control, target])
        self._add(dict(type="ctrl", controls=[control], open=[], target=target, name=None), [control, target], col)
        return self

    def controlled(self, name, controls, target, *, open_controls=(), col=None, kind=None):
        """Controlled gate: filled dots on controls, open dots on open_controls, a box on target."""
        cs = list(np.atleast_1d(controls))
        oc = list(open_controls)
        self._check(cs + oc + [target])
        kind = kind or ("rotation" if name.lstrip("$\\").startswith("R") else "clifford")
        self._add(dict(type="ctrl", controls=cs, open=oc, target=target, name=name, kind=kind), cs + oc + [target],
                  col)
        return self

    def cz(self, a, b, *, col=None):
        """Controlled Z: two dots joined, symmetric in its qubits."""
        self._check([a, b])
        self._add(dict(type="cz", q=[a, b]), [a, b], col)
        return self

    def swap(self, a, b, *, col=None):
        self._check([a, b])
        self._add(dict(type="swap", q=[a, b]), [a, b], col)
        return self

    def measure(self, q, *, col=None):
        """Meters on q (or each qubit of a list, in one column); each wire turns classical after it."""
        qs = self._check([q] if np.ndim(q) == 0 else q)
        c = self._place(qs, col) if len(qs) == 1 else max([self._free[x] for x in qs] + [self._floor]) if col is None else col
        for qq in qs:
            self._free[qq] = max(self._free[qq], c + 1)
            self.ops.append(dict(type="measure", q=[qq], col=c))
            self._measured[qq] = c
        return self

    def slice(self, label=None):
        """A dashed slice after everything placed so far, with an optional tilted label."""
        after = max(self._free + [self._floor]) - 1
        self.slices.append(dict(after=after, label=label))
        self._floor = after + 1
        return self

    def group(self, qubits, cols, label=None):
        """A dashed rounded box around wires min..max(qubits) over columns cols = (first, last)."""
        qs = self._check(qubits, quantum=False)
        self.groups.append(dict(q=(min(qs), max(qs)), cols=tuple(cols), label=label))
        return self

    # ---- layout --------------------------------------------------------------------------------

    def _op_width(self, op):
        if op["type"] in ("gate", "multi") or (op["type"] == "ctrl" and op["name"]):
            return max(self.BOX, 0.2 * _visible_len(op["name"]) + 0.32)
        if op["type"] == "measure":
            return 0.72
        return 0.42

    def _layout(self):
        n = self.ncols
        w = np.full(n, self.BOX)
        for op in self.ops:
            w[op["col"]] = max(w[op["col"]], self._op_width(op))
        after = {s["after"] for s in self.slices}
        x = np.zeros(n)
        edge = 0.0
        for c in range(n):
            if c - 1 in after:
                edge += self.SLICE_GAP
            x[c] = edge + w[c] / 2
            edge = x[c] + w[c] / 2 + self.GAP
        right = (x[-1] + w[-1] / 2 if n else 0.0) + 0.45 + (self.SLICE_GAP if n - 1 in after else 0)
        left = -0.45
        lab = max([0.17 * _visible_len(s) + 0.2 for s in self.labels if s] + [0.0])
        out = max([0.17 * _visible_len(s) + 0.2 for s in self.outputs if s] + [0.0])
        top = 0.62 + (0.75 if any(s["label"] for s in self.slices) else 0.0) + \
            (0.35 if any(g["label"] and g["q"][0] == 0 for g in self.groups) else 0.0)
        bottom = -(self.n - 1) - 0.62
        return x, w, (left - lab, right + out, bottom, top), left, right

    def figsize(self, scale=0.36):
        """(width, height) in inches at `scale` inches per wire spacing."""
        _, _, (x0, x1, y0, y1), _, _ = self._layout()
        return (x1 - x0) * scale, (y1 - y0) * scale

    def _slice_x(self, x, w, after):
        if after < 0:
            return -0.25
        return x[after] + w[after] / 2 + (self.GAP + self.SLICE_GAP) / 2

    # ---- drawing -------------------------------------------------------------------------------

    def draw(self, ax, *, slice_angle=30, fontsize=None):
        """Draw on ax (made equal-aspect and frameless). Returns the column centres "x", wire heights
        "y" and the drawn glyphs by kind."""
        t = theme()
        ink, lw = t.ink, 0.9
        fs = fontsize or t.size + 1.5
        x, w, (X0, X1, Y0, Y1), left, right = self._layout()
        y = -np.arange(self.n, dtype=float)
        out = {k: [] for k in ("boxes", "controls", "targets", "meters", "slices", "groups", "classical", "swaps")}
        out.update(x=x, y=y)
        round_ = "round,pad=0,rounding_size=0.07" if t.name == "field" else "square,pad=0"

        def box(cx, cy, bw, bh, fc, z=3):
            p = FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh, boxstyle=round_, fc=fc, ec=ink, lw=lw, zorder=z)
            ax.add_patch(p)
            out["boxes"].append(p)
            return p

        def fill_of(op):
            if op.get("fill"):
                return op["fill"]
            tile = KINDS.get(op.get("kind") or "clifford")
            return t.tiles[tile] if tile else "white"

        def vline(cx, ya, yb, z=2):
            ax.plot([cx, cx], [ya, yb], color=ink, lw=lw, ls="-", zorder=z, solid_capstyle="butt")

        def dot(cx, cy, open_=False):
            c = Circle((cx, cy), 0.075, fc="white" if open_ else ink, ec=ink, lw=lw, zorder=4)
            ax.add_patch(c)
            out["controls"].append(c)

        # wires: quantum up to the meter, a double classical line after it
        for q in range(self.n):
            xm = x[self._measured[q]] if q in self._measured else None
            ax.plot([left, right if xm is None else xm], [y[q]] * 2, color=ink, lw=lw, ls="-", zorder=1)
            if xm is not None:
                for s in (-0.035, 0.035):
                    ax.plot([xm, right], [y[q] + s] * 2, color=ink, lw=0.6, ls="-", zorder=1)
            if self.labels[q]:
                ax.text(left - 0.1, y[q], self.labels[q], ha="right", va="center", fontsize=fs, color=ink)
            if self.outputs[q]:
                ax.text(right + 0.1, y[q], self.outputs[q], ha="left", va="center", fontsize=fs, color=ink)

        for op in self.ops:
            cx = x[op["col"]]
            kind = op["type"]
            if kind == "gate":
                q = op["q"][0]
                bw = max(self.BOX, self._op_width(op))
                if op["cond"] is not None:
                    yc = y[op["cond"]]
                    yb = y[q] + (self.BOX / 2 if yc > y[q] else -self.BOX / 2)
                    for s in (-0.035, 0.035):
                        ax.plot([cx + s, cx + s], [yc, yb], color=ink, lw=0.6, ls="-", zorder=2)
                    ax.add_patch(Circle((cx, yc), 0.06, fc=ink, ec=ink, lw=0, zorder=4))
                    out["classical"].append(op)
                box(cx, y[q], bw, self.BOX, fill_of(op))
                ax.text(cx, y[q], _math(op["name"]), ha="center", va="center", fontsize=fs, color=ink, zorder=5)
            elif kind == "multi":
                lo, hi = y[op["q"][0]], y[op["q"][-1]]
                bw = max(self.BOX, self._op_width(op))
                box(cx, (lo + hi) / 2, bw, (lo - hi) + self.BOX, fill_of(op))
                ax.text(cx, (lo + hi) / 2, _math(op["name"]), ha="center", va="center", fontsize=fs, color=ink, zorder=5)
            elif kind == "ctrl":
                rows = op["controls"] + op["open"] + [op["target"]]
                vline(cx, y[min(rows)], y[max(rows)])
                for c in op["controls"]:
                    dot(cx, y[c])
                for c in op["open"]:
                    dot(cx, y[c], open_=True)
                yt = y[op["target"]]
                if op["name"] is None:
                    r = 0.17
                    ax.add_patch(Circle((cx, yt), r, fc="white", ec=ink, lw=lw, zorder=4))
                    ax.plot([cx - r, cx + r], [yt, yt], color=ink, lw=lw, ls="-", zorder=5)
                    ax.plot([cx, cx], [yt - r, yt + r], color=ink, lw=lw, ls="-", zorder=5)
                    out["targets"].append(op)
                else:
                    box(cx, yt, max(self.BOX, self._op_width(op)), self.BOX, fill_of(op))
                    ax.text(cx, yt, _math(op["name"]), ha="center", va="center", fontsize=fs, color=ink, zorder=5)
            elif kind == "cz":
                a, b = op["q"]
                vline(cx, y[a], y[b])
                dot(cx, y[a])
                dot(cx, y[b])
            elif kind == "swap":
                a, b = op["q"]
                vline(cx, y[a], y[b])
                s = 0.11
                for q in (a, b):
                    ax.plot([cx - s, cx + s], [y[q] - s, y[q] + s], color=ink, lw=lw * 1.2, ls="-", zorder=4)
                    ax.plot([cx - s, cx + s], [y[q] + s, y[q] - s], color=ink, lw=lw * 1.2, ls="-", zorder=4)
                out["swaps"].append(op)
            elif kind == "measure":
                q = op["q"][0]
                yq = y[q]
                box(cx, yq, 0.72, 0.54, "white")
                ax.add_patch(Arc((cx, yq - 0.13), 0.46, 0.46, theta1=25, theta2=155, color=ink, lw=lw, zorder=5))
                arrow(ax, (cx, yq - 0.13), (cx + 0.2, yq + 0.17), color=ink, lw=lw, ms=5, zorder=6)
                out["meters"].append(op)

        for g in self.groups:
            (qa, qb), (ca, cb) = g["q"], g["cols"]
            xa, xb = x[ca] - w[ca] / 2 - 0.16, x[cb] + w[cb] / 2 + 0.16
            ya, yb = y[qb] - self.BOX / 2 - 0.16, y[qa] + self.BOX / 2 + 0.16
            p = FancyBboxPatch((xa, ya), xb - xa, yb - ya, boxstyle="round,pad=0,rounding_size=0.1", fc="none",
                               ec=t.reference, lw=0.8, ls=(0, (3, 2)), zorder=2.5)
            ax.add_patch(p)
            if g["label"]:
                ax.text(xa, yb + 0.05, g["label"], ha="left", va="bottom", fontsize=t.size, color=t.reference)
            out["groups"].append(p)

        ytop, ybot = 0.62, -(self.n - 1) - 0.55
        for s in self.slices:
            sx = self._slice_x(x, w, s["after"])
            ln, = ax.plot([sx, sx], [ybot, ytop], color=t.model, lw=lw, ls=(0, (4, 2.5)), zorder=2.6)
            out["slices"].append(ln)
            if s["label"]:
                ax.text(sx, ytop + 0.06, s["label"], ha="left", va="bottom", rotation=slice_angle,
                        rotation_mode="anchor", fontsize=fs, color=ink)

        ax.set_xlim(X0, X1)
        ax.set_ylim(Y0, Y1)
        ax.set_aspect("equal")
        ax.axis("off")
        return out
