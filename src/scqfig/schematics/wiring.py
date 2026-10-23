"""Cryostat wiring diagrams: one column per line crossing the temperature stages, with attenuators,
filters, DC blocks, circulators and isolators, a TWPA, a HEMT, room-temperature amplifiers and the
sample box. Lines take their colour from the chip roles (drive, flux, resonator); components are ink."""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from ..core import theme
from ..themes import lighten
from .symbols import ground, resistor_v

__all__ = ["attenuator", "lowpass", "ir_filter", "dc_block", "circulator", "isolator", "amplifier",
           "sample_box", "cryostat_wiring"]

STAGES = (("300 K", ""), ("50 K", ""), ("4 K", ""), ("still", "0.8 K"), ("cold plate", "0.1 K"), ("MXC", "10 mK"))

LINES = (
    dict(name="XY drive", role="drive",
         parts={"4 K": [("att", 20)], "cold plate": [("att", 10)], "MXC": [("att", 20), ("ir",)]}),
    dict(name="Z flux", role="flux", parts={"4 K": [("att", 10)], "MXC": [("lp",), ("ir",)]}),
    dict(name="readout in", role="resonator",
         parts={"4 K": [("att", 20)], "cold plate": [("att", 10)], "MXC": [("att", 20), ("ir",)]}),
    dict(name="readout out", role="resonator", out=True,
         parts={"300 K": [("amp",), ("dc",)], "4 K": [("hemt",)], "MXC": [("iso",), ("iso",), ("twpa",), ("ir",)]}),
)

_LW = 0.7


def _box(ax, x, y, w, h, z=5):
    t = theme()
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="none", ec=t.ink, lw=_LW, zorder=z))


def _label(ax, x, y, s):
    t = theme()
    ax.text(x, y, s, ha="left", va="center", fontsize=t.size - 2, color=t.reference, zorder=6)


def attenuator(ax, x, y, db=20, *, w=0.5, h=0.2, fs=None):
    """An attenuator: a box with its value in dB."""
    t = theme()
    _box(ax, x, y, w, h)
    ax.text(x, y, f"{db:g} dB", ha="center", va="center", fontsize=fs or t.size - 2, color=t.ink, zorder=6)
    return h / 2


def lowpass(ax, x, y, *, w=0.3, h=0.22):
    """A low-pass filter: a box with a low-frequency wave passed and a high-frequency wave struck out."""
    t = theme()
    _box(ax, x, y, w, h)
    s = np.linspace(-0.32, 0.32, 80) * w
    kw = dict(color=t.ink, lw=0.5, ls="-", zorder=6)
    ax.plot(x + s, y - 0.22 * h + 0.09 * h * np.sin(2 * np.pi * s / (0.64 * w)), **kw)
    ax.plot(x + s, y + 0.22 * h + 0.09 * h * np.sin(2 * np.pi * 3 * s / (0.64 * w)), **kw)
    ax.plot([x - 0.18 * w, x + 0.18 * w], [y + 0.02 * h, y + 0.42 * h], **kw)
    return h / 2


def ir_filter(ax, x, y, *, w=0.3, h=0.22):
    """An infrared (Eccosorb) filter: a box marked IR."""
    t = theme()
    _box(ax, x, y, w, h)
    ax.text(x, y, "IR", ha="center", va="center", fontsize=t.size - 2, color=t.ink, zorder=6)
    return h / 2


def dc_block(ax, x, y, color=None, *, w=0.22, gap=0.06, label="DC"):
    """A DC block: a series capacitor in the line."""
    t = theme()
    for s in (-1, 1):
        ax.plot([x - w / 2, x + w / 2], [y + s * gap / 2] * 2, color=t.ink, lw=1.2, ls="-", zorder=6)
    if label:
        _label(ax, x + w / 2 + 0.06, y, label)
    return gap / 2


def circulator(ax, x, y, *, r=0.13, cw=True):
    """A circulator: a circle with a curved arrow inside (clockwise unless cw=False)."""
    t = theme()
    ax.add_patch(Circle((x, y), r, fc="none", ec=t.ink, lw=_LW, zorder=5))
    a0, a1 = (np.deg2rad(150), np.deg2rad(-110)) if cw else (np.deg2rad(30), np.deg2rad(290))
    s = np.linspace(a0, a1, 40)
    ax.plot(x + 0.55 * r * np.cos(s), y + 0.55 * r * np.sin(s), color=t.ink, lw=0.6, ls="-", zorder=6)
    p1 = (x + 0.55 * r * np.cos(a1), y + 0.55 * r * np.sin(a1))
    d = np.array([np.sin(a1), -np.cos(a1)]) * (1 if cw else -1)
    ax.add_patch(FancyArrowPatch(np.subtract(p1, 0.02 * d), np.add(p1, 0.04 * d), arrowstyle="-|>",
                                 mutation_scale=4.5, color=t.ink, lw=0.6, shrinkA=0, shrinkB=0, zorder=6))
    return r


def isolator(ax, x, y, *, r=0.13):
    """An isolator: a circulator whose third port ends in a matched load to ground."""
    t = theme()
    circulator(ax, x, y, r=r)
    xs = x + r + 0.12
    ax.plot([x + r, xs], [y, y], color=t.ink, lw=_LW, ls="-", zorder=5)
    resistor_v(ax, xs, y, y - 0.17, n=2, r=0.025, lw=0.6)
    ground(ax, xs, y - 0.17, w=0.12)
    return r


def amplifier(ax, x, y, color=None, *, up=True, w=0.32, h=0.26, label=None):
    """An amplifier: a triangle pointing along the signal, filled with a tint of the line colour."""
    t = theme()
    c = color or t.ink
    s = 1 if up else -1
    ax.add_patch(Polygon([(x - w / 2, y - s * h / 2), (x + w / 2, y - s * h / 2), (x, y + s * h / 2)],
                         fc=lighten(c, 0.75), ec=t.ink, lw=_LW, zorder=5))
    if label:
        _label(ax, x + w / 2 + 0.04, y, label)
    return h / 2


def sample_box(ax, x0, x1, y, h=0.42, label="sample"):
    """The sample box: a rectangle from x0 to x1 with its top at y, which every line enters."""
    t = theme()
    ax.add_patch(Rectangle((x0, y - h), x1 - x0, h, fc=t.chip["substrate"], ec=t.ink, lw=_LW, zorder=5))
    ax.text((x0 + x1) / 2, y - h / 2, label, ha="center", va="center", fontsize=t.size - 1, color=t.ink, zorder=6)


def _draw(ax, kind, args, x, y, color, out):
    if kind == "att":
        return attenuator(ax, x, y, *args)
    if kind == "lp":
        return lowpass(ax, x, y)
    if kind == "ir":
        return ir_filter(ax, x, y)
    if kind == "dc":
        return dc_block(ax, x, y, color)
    if kind == "circ":
        return circulator(ax, x, y)
    if kind == "iso":
        return isolator(ax, x, y)
    labels = {"twpa": "TWPA", "hemt": "HEMT", "amp": "RT"}
    return amplifier(ax, x, y, color, up=out, label=args[0] if args else labels[kind])


PARTS = ("att", "lp", "ir", "dc", "circ", "iso", "twpa", "hemt", "amp")


def cryostat_wiring(ax, lines=None, stages=STAGES, *, col=1.0, x0=1.35, pitch=0.42, sample="sample"):
    """Wiring of a dilution refrigerator: one column per line crossing the stage bands (top to bottom,
    each a (name, temperature) pair) down to the sample box at the mixing chamber. Each line is a dict
    with "name", "role" (a theme().chip key giving its colour), "out" (True for an output line, drawn
    flowing up) and "parts": {stage name: [(kind, *args), ...]} listed top to bottom. Kinds: "att" (dB),
    "lp", "ir", "dc", "circ", "iso", "twpa", "hemt", "amp" (optional label). Defaults draw the usual XY,
    Z, readout-in and readout-out lines. Returns the column x, the stage y ranges and every part placed
    as (line, stage, kind, x, y)."""
    t = theme()
    lines = LINES if lines is None else lines
    names = [s for s, _ in stages]
    for ln in lines:
        for st, parts in ln.get("parts", {}).items():
            if st not in names:
                raise ValueError(f"line {ln['name']!r}: unknown stage {st!r}; stages are {names}")
            for p in parts:
                if p[0] not in PARTS:
                    raise ValueError(f"line {ln['name']!r}: unknown part {p[0]!r}; parts are {PARTS}")
    nmax = {s: max([len(ln.get("parts", {}).get(s, [])) for ln in lines] + [0]) for s in names}
    box_h = 0.42
    heights = {s: max(0.62, pitch * (nmax[s] + 0.6)) + (box_h + 0.2 if k == len(names) - 1 else 0)
               for k, s in enumerate(names)}
    xs = {ln["name"]: x0 + k * col for k, ln in enumerate(lines)}
    xr = x0 + (len(lines) - 1) * col + 0.62
    head = 0.62
    y = 0.0
    bands = {}
    for k, s in enumerate(names):
        bands[s] = (y - heights[s], y)
        y -= heights[s]
    ybot = y
    for k, (s, temp) in enumerate(stages):
        lo, hi = bands[s]
        if k % 2 == 0:
            ax.add_patch(Rectangle((0, lo), xr, hi - lo, fc=t.tint, ec="none", zorder=0))
        mid = (lo + hi) / 2 if k < len(stages) - 1 else hi - (hi - lo - box_h - 0.2) / 2
        ax.text(0.06, mid + (0.09 if temp else 0), s, ha="left", va="center", fontsize=t.size - 1, color=t.ink)
        if temp:
            ax.text(0.06, mid - 0.13, temp, ha="left", va="center", fontsize=t.size - 1.5, color=t.reference)
    ax.plot([0, xr], [0, 0], color=t.light, lw=0.6, ls="-", zorder=0)
    ax.plot([0, xr], [ybot, ybot], color=t.light, lw=0.6, ls="-", zorder=0)

    box_top = ybot + 0.12 + box_h
    placed = []
    for ln in lines:
        x = xs[ln["name"]]
        c = ln.get("color") or t.chip[ln.get("role", "feedline")]
        out = bool(ln.get("out"))
        cuts = []
        for s in names:
            parts = ln.get("parts", {}).get(s, [])
            lo, hi = bands[s]
            top = hi
            span = (hi - lo) - (box_h + 0.2 if s == names[-1] else 0)
            first = top - span / 2 + pitch * (len(parts) - 1) / 2
            for i, (kind, *args) in enumerate(parts):
                yp = first - i * pitch
                hh = _draw(ax, kind, args, x, yp, c, out)
                cuts.append((yp - hh, yp + hh))
                placed.append((ln["name"], s, kind, x, yp))
        ytop = head * 0.25
        segs, yc = [], ytop
        for lo, hi in sorted(cuts, key=lambda p: -p[1]):
            segs.append((yc, hi))
            yc = lo
        segs.append((yc, box_top))
        for a, b in segs:
            ax.plot([x, x], [a, b], color=c, lw=1.1, ls="-", solid_capstyle="butt", zorder=3)
        tip, tail = ((ytop + 0.02, ytop - 0.16) if out else (ytop - 0.16, ytop + 0.02))
        ax.add_patch(FancyArrowPatch((x, tail), (x, tip), arrowstyle="-|>", mutation_scale=7, color=c, lw=1.1,
                                     shrinkA=0, shrinkB=0, zorder=4))
        ax.text(x, ytop + 0.1, ln["name"].replace(" ", "\n"), ha="center", va="bottom", fontsize=t.size - 1,
                color=c, linespacing=1.1)
    sample_box(ax, x0 - 0.42, x0 + (len(lines) - 1) * col + 0.42, box_top, box_h, sample)
    ax.set_xlim(0, xr)
    ax.set_ylim(ybot, head * 0.25 + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return {"columns": xs, "stages": bands, "parts": placed}
