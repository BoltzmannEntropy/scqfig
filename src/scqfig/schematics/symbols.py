"""Circuit symbols for lumped-element diagrams, drawn in data coordinates of an equal-aspect axes.

Each subsystem keeps one colour (theme().qubits, .bus), junctions stay ink. Inside `with tiles():`
capacitors, junctions and inductors sit on pale tiles by element kind (capacitive, nonlinear,
inductive; "coupling" for coupling capacitors), so that the Hamiltonian terms they produce can sit on
the same tiles beside the circuit."""
import contextlib

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from ..core import theme

_TILES_ON = [False]


@contextlib.contextmanager
def tiles():
    """Inside this block elements sit on their tiles; pass tile_kind=False to suppress one."""
    _TILES_ON[0] = True
    try:
        yield
    finally:
        _TILES_ON[0] = False


def _kind(tile_kind, default):
    if tile_kind is None:
        return default if _TILES_ON[0] else None
    return tile_kind or None


def _ink(color):
    return theme().ink if color is None else color


def tile(ax, x, y, w, h, kind, z=0.6):
    """Pale rounded tile of a kind (theme().tiles key, or any colour) centred at (x, y)."""
    fc = theme().tiles.get(kind, kind)
    r = 0.18 * min(w, h)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec="none", zorder=z))


def wire(ax, xs, ys, color=None, lw=0.9, **kw):
    kw.setdefault("ls", "-")
    ax.plot(xs, ys, color=_ink(color), lw=lw, solid_capstyle="round", solid_joinstyle="round", **kw)


def node(ax, x, y, color=None, r=0.035, z=6):
    """A filled junction dot where wires meet."""
    ax.add_patch(Circle((x, y), r, fc=_ink(color), ec="none", zorder=z))


def capacitor_v(ax, x, y, gap=0.07, w=0.22, lw=1.0, color=None, tile_kind=None):
    """Capacitor with horizontal plates, for a vertical wire through (x, y)."""
    kind = _kind(tile_kind, "capacitive")
    if kind:
        tile(ax, x, y, w * 1.55, max(w * 1.1, gap * 4.5), kind)
    for s in (-1, 1):
        ax.plot([x - w / 2, x + w / 2], [y + s * gap / 2] * 2, color=_ink(color), lw=lw * 1.4, ls="-", zorder=5)


def capacitor_h(ax, x, y, gap=0.07, h=0.22, lw=1.0, color=None, tile_kind=None):
    """Capacitor with vertical plates, for a horizontal wire through (x, y)."""
    kind = _kind(tile_kind, "capacitive")
    if kind:
        tile(ax, x, y, max(h * 1.1, gap * 4.5), h * 1.55, kind)
    for s in (-1, 1):
        ax.plot([x + s * gap / 2] * 2, [y - h / 2, y + h / 2], color=_ink(color), lw=lw * 1.4, ls="-", zorder=5)


def junction(ax, x, y, s=0.08, lw=1.1, color=None, tile_kind=None):
    """Josephson junction: a cross of half-size s."""
    kind = _kind(tile_kind, "nonlinear")
    if kind:
        tile(ax, x, y, s * 3.6, s * 3.6, kind)
    c = theme().chip["junction"] if color is None else color
    ax.plot([x - s, x + s], [y - s, y + s], color=c, lw=lw, ls="-", zorder=5)
    ax.plot([x - s, x + s], [y + s, y - s], color=c, lw=lw, ls="-", zorder=5)


def squid(ax, x, y, w=0.36, h=0.3, color=None, lw=0.9, tile_kind=None):
    """dc SQUID for a vertical wire: a w by h loop with a junction on each side; the wire attaches at
    (x, y +- h/2)."""
    kind = _kind(tile_kind, "nonlinear")
    if kind:
        tile(ax, x, y, w * 1.7, h * 1.9, kind)
    c = _ink(color)
    ax.plot([x - w / 2, x + w / 2, x + w / 2, x - w / 2, x - w / 2], [y - h / 2, y - h / 2, y + h / 2, y + h / 2, y - h / 2],
            color=c, lw=lw, ls="-", zorder=4)
    for sx in (-1, 1):
        junction(ax, x + sx * w / 2, y, s=0.055, tile_kind=False)


def inductor_v(ax, x, y0, y1, n=4, r=0.08, color=None, lw=1.0, tile_kind=None):
    """Vertical coil from y0 to y1 with n turns of half-width r."""
    kind = _kind(tile_kind, "inductive")
    if kind:
        tile(ax, x, (y0 + y1) / 2, r * 4.2, abs(y1 - y0) + r * 1.6, kind)
    t = np.linspace(0, 1, 240)
    ax.plot(x + r * np.sin(2 * np.pi * n * t), y0 + (y1 - y0) * t, color=_ink(color), lw=lw, ls="-", zorder=5)


def inductor_h(ax, x0, x1, y, n=4, r=0.08, color=None, lw=1.0, tile_kind=None):
    """Horizontal coil from x0 to x1."""
    kind = _kind(tile_kind, "inductive")
    if kind:
        tile(ax, (x0 + x1) / 2, y, abs(x1 - x0) + r * 1.6, r * 4.2, kind)
    t = np.linspace(0, 1, 240)
    ax.plot(x0 + (x1 - x0) * t, y + r * np.sin(2 * np.pi * n * t), color=_ink(color), lw=lw, ls="-", zorder=5)


def resistor_v(ax, x, y0, y1, n=3, r=0.07, color=None, lw=0.9):
    """Zig-zag resistor (a port impedance or loss) from y0 to y1."""
    t = np.linspace(0, 1, 4 * n + 1)
    xs = x + r * np.array([0] + [(-1) ** k for k in range(1, 4 * n)] + [0])
    ax.plot(xs, y0 + (y1 - y0) * t, color=_ink(color), lw=lw, ls="-", zorder=5)


def ground(ax, x, y, w=0.24, color=None):
    for i, f in enumerate((1.0, 0.65, 0.3)):
        ax.plot([x - w * f / 2, x + w * f / 2], [y - i * 0.05] * 2, color=_ink(color), lw=0.9, ls="-")


def ac_source(ax, x, y, r=0.13, color=None):
    ax.add_patch(Circle((x, y), r, fc="white", ec=_ink(color), lw=0.9, zorder=5))
    t = np.linspace(-1, 1, 60)
    ax.plot(x + 0.7 * r * t, y + 0.35 * r * np.sin(np.pi * t), color=_ink(color), lw=0.8, ls="-", zorder=6)


def arrow(ax, p0, p1, color=None, lw=0.7, style="-|>", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=_ink(color), lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def dim(ax, p0, p1, label, offset=(0, 0), color=None, fs=None, **kw):
    """Two-headed dimension arrow with a label at its midpoint (+ offset)."""
    arrow(ax, p0, p1, color=color, style="<|-|>", ms=6, lw=0.6)
    mx, my = (p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1]
    ax.text(mx, my, label, ha="center", va="center", fontsize=fs or theme().size, color=_ink(color), **kw)


def blank(ax, aspect="equal"):
    """Prepare an axes for a schematic: equal aspect, no frame or ticks."""
    ax.set_aspect(aspect)
    ax.axis("off")
    return ax
