"""Schematic motifs of circuit-QED figures: chips on a slanted slab, wavefunctions filled on their
levels inside a potential, transitions, photons as wavy arrows, level ladders, qubits as spins and
regions of uniform field.

Three rules keep them truthful. A chip goes through one rigid affine map (Oblique, onto_plane), so it
keeps its proportions and never bends. A well panel fills densities that a solver computed
(physics.phase_densities), never a sketched bump. A uniform field is a flat wash (wash), never a
radial glow."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

from ..core import theme
from ..themes import lighten
from .symbols import arrow


class Oblique:
    """Oblique map of the chip plane onto the page: chip point (u, v), u along the chip and v across
    it, lands at x = x0 + u + shear v, y = y0 + squash v."""

    def __init__(self, x0=0.0, y0=0.0, shear=0.45, squash=0.5):
        self.x0, self.y0, self.shear, self.squash = x0, y0, shear, squash

    def __call__(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        return self.x0 + u + self.shear * v, self.y0 + self.squash * v

    def transform(self, ax):
        """The same map as a matplotlib transform from chip units to the page."""
        return Affine2D().from_values(1, 0, self.shear, self.squash, self.x0, self.y0) + ax.transData


def artists(ax):
    """Identity set of the artists on ax, to pass later to onto_plane."""
    return {id(a) for a in ax.lines + ax.patches + ax.texts + ax.collections + ax.images}


def onto_plane(ax, ob, before):
    """Lay every artist drawn since `before` (from artists(ax)) onto the slanted plane of `ob`. Draw
    a plan-view chip flat in chip units, then call this: the layout stays rigid and text upright."""
    T = ob.transform(ax)
    for a in ax.lines + ax.patches + ax.texts + ax.collections + ax.images:
        if id(a) not in before:
            a.set_transform(T)


def ob_slab(ax, ob, w, h, depth=0.12, fc=None, z=0.5):
    """Substrate of w by h chip units with a front and a side face of the given depth."""
    t = theme()
    c = [ob(0, 0), ob(w, 0), ob(w, h), ob(0, h)]

    def down(p):
        return (p[0], p[1] - depth)

    ax.add_patch(Polygon([c[0], c[1], down(c[1]), down(c[0])], fc=t.light, ec=t.chip["edge"], lw=0.4, zorder=z))
    ax.add_patch(Polygon([c[1], c[2], down(c[2]), down(c[1])], fc="#c6c6c6", ec=t.chip["edge"], lw=0.4, zorder=z))
    ax.add_patch(Polygon(c, fc=fc or t.chip["substrate"], ec=t.chip["edge"], lw=0.4, zorder=z + 0.01))


def ob_path(ax, ob, us, vs, color=None, lw=2.4, z=3):
    """A metal trace on the chip as a ribbon: the colour with a pale core."""
    color = color or theme().bus
    x, y = ob(us, vs)
    kw = dict(solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(x, y, color=color, lw=lw, ls="-", **kw)
    ax.plot(x, y, color=lighten(color, 0.55), lw=lw * 0.35, ls="-", **kw)


def meander(u0, u1, v0, amp, n, npts=600):
    """A resonator meander from (u0, v0) to (u1, v0) with n rounded bumps of height amp."""
    u = np.linspace(u0, u1, npts)
    return u, v0 + amp * 0.5 * (1 + np.tanh(4 * np.sin(2 * np.pi * n * (u - u0) / (u1 - u0))))


def _cross(ax, x, y, s, color=None, lw=1.1, z=6):
    c = color or theme().chip["junction"]
    ax.plot([x - s, x + s], [y - s, y + s], color=c, lw=lw, ls="-", zorder=z)
    ax.plot([x - s, x + s], [y + s, y - s], color=c, lw=lw, ls="-", zorder=z)


def ob_loop(ax, ob, u, v, s, color=None, lw=1.5, z=4):
    """A qubit loop of side s centred at (u, v), with a junction on its left and right sides."""
    color = color or theme().qubits[0]
    us = u + s / 2 * np.array([-1, 1, 1, -1, -1])
    vs = v + s / 2 * np.array([-1, -1, 1, 1, -1])
    x, y = ob(us, vs)
    ax.fill(x, y, color=lighten(color, 0.85), zorder=z - 0.1, lw=0)
    ax.plot(x, y, color=color, lw=lw, ls="-", solid_joinstyle="round", zorder=z)
    for du in (-s / 2, s / 2):
        _cross(ax, *ob(u + du, v), 0.13 * s, z=z + 1)


def ob_coupler(ax, ob, u, v, r, color=None, lw=0.8, z=4):
    """A flux coupler: a small loop of radius r interrupted by two junctions."""
    t = np.linspace(0, 2 * np.pi, 120)
    x, y = ob(u + r * np.cos(t), v + r * np.sin(t))
    ax.plot(x, y, color=color or theme().ink, lw=lw, ls="-", zorder=z)
    for du in (-r, r):
        _cross(ax, *ob(u + du, v), 0.32 * r, lw=0.8, z=z + 1)


def well(ax, phi, V, energies, densities, scale, colors=None, labels=True, label_x=None, lines="full", z=3):
    """Potential V(phi) in ink, each level as a thin line at its energy and its computed probability
    density filled above that line in the level's colour; `scale` converts density to energy units.
    lines="full" runs each level across the panel, lines="well" only across its allowed region."""
    t = theme()
    colors = colors or t.levels
    fill = 0.25 if t.name == "field" else 0.6
    ax.plot(phi, V, color=t.ink, lw=1.1, ls="-", zorder=z + 2)
    lx = phi[0] if label_x is None else label_x
    for n, (E, d) in enumerate(zip(energies, densities)):
        c = colors[n % len(colors)]
        if lines == "well":
            inside = phi[np.asarray(V) <= E]
            ax.plot([inside.min(), inside.max()], [E, E], color=c, lw=0.9, ls="-", zorder=z)
        else:
            ax.axhline(E, color=c, lw=0.5, alpha=0.8, zorder=z)
        d = np.asarray(d)
        on = d > 1e-2 * d.max()                     # lobes end where the density falls below 1 % of its peak
        top = np.where(on, E + scale * d, np.nan)
        ax.fill_between(phi, E, top, where=on, color=lighten(c, fill), lw=0, zorder=z + 1)
        ax.plot(phi, top, color=c, lw=0.6, ls="-", zorder=z + 1)
        if labels:
            ax.text(lx, E, rf"$|{n}\rangle$", ha="left", va="bottom", fontsize=t.size, color=c)


def transition(ax, p0, p1, allowed=True, label=None, fs=None):
    """A transition between two levels: allowed solid in the accent colour, forbidden dashed ink."""
    t = theme()
    if allowed:
        arrow(ax, p0, p1, color=t.accent, lw=1.2, style="<|-|>", ms=7, zorder=8)
    else:
        arrow(ax, p0, p1, color=t.ink, lw=0.9, style="<|-|>", ms=6, ls=(0, (3, 2)), zorder=8)
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, label, ha="center", va="bottom",
                fontsize=fs or t.size - 0.5, color=t.accent if allowed else t.ink, zorder=9)


def wavy(ax, p0, p1, color=None, amp=0.05, waves=4, lw=1.1, head=0.12, ms=8, z=7):
    """A photon or drive tone from p0 to p1 as a wavy line with an arrowhead; amp and head in data
    units of an equal-aspect axes. The wave dies out before the head."""
    color = color or theme().signal
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    tng, nrm = d / L, np.array([-d[1], d[0]]) / L
    s = np.linspace(0, L - head, 300)
    env = np.clip((L - head - s) / (0.2 * L), 0, 1)
    pts = p0 + np.outer(s, tng) + np.outer(amp * env * np.sin(2 * np.pi * waves * s / (L - head)), nrm)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls="-", solid_capstyle="round", zorder=z)
    arrow(ax, tuple(pts[-1]), tuple(p1), color=color, lw=lw, ms=ms, zorder=z)


def ladder(ax, x0, x1, levels, colors=None, labels=None, lw=2.0, fs=None, z=4):
    """Horizontal level lines from x0 to x1 at the given energies, labelled on the right."""
    t = theme()
    colors = colors or t.levels
    for n, E in enumerate(levels):
        c = colors[n % len(colors)]
        ax.plot([x0, x1], [E, E], color=c, lw=lw, ls="-", solid_capstyle="butt", zorder=z)
        if labels:
            ax.text(x1 + 0.04 * (x1 - x0), E, labels[n], ha="left", va="center", fontsize=fs or t.size, color=c)


_WASH_LOC = {"lower left": (0, 0, "left", "bottom"), "upper left": (0, 1, "left", "top"),
             "lower right": (1, 0, "right", "bottom"), "upper right": (1, 1, "right", "top"),
             "center": (0.5, 0.5, "center", "center")}


def wash(ax, x, y, w, h, *, label=None, loc="lower left", rounded=False, z=0.0, fs=None):
    """A region of uniform field: a flat w by h fill in theme().wash with its lower-left corner at (x, y),
    drawn under everything else (z). The field is the same at every point, so the fill is flat, never a
    radial glow or arrows at positions. label sits inside at loc ("lower left", "upper left", "lower
    right", "upper right" or "center") in the signal colour. rounded=True rounds the corners. Returns
    (patch, label or None)."""
    if not (w > 0 and h > 0):
        raise ValueError(f"a wash needs a positive size, got {w!r} by {h!r}")
    if loc not in _WASH_LOC:
        raise ValueError(f"loc must be one of {sorted(_WASH_LOC)}, got {loc!r}")
    t = theme()
    if rounded:
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={0.08 * min(w, h)}", fc=t.wash,
                           ec="none", lw=0, zorder=z)
    else:
        p = Rectangle((x, y), w, h, fc=t.wash, ec="none", lw=0, zorder=z)
    ax.add_patch(p)
    text = None
    if label:
        fx, fy, ha, va = _WASH_LOC[loc]
        pad = 0.04 * min(w, h)
        tx = x + fx * w + (pad if ha == "left" else -pad if ha == "right" else 0.0)
        ty = y + fy * h + (pad if va == "bottom" else -pad if va == "top" else 0.0)
        text = ax.text(tx, ty, label, ha=ha, va=va, fontsize=fs or t.size, color=t.signal, zorder=z + 0.1)
    return p, text


def spin(ax, x, y, h, color=None, w=None, lw=2.0, z=6):
    """A qubit drawn as a spin: a shaft with a two-shade cone head, up for |0> (h > 0) and down for
    |1> (h < 0), as on the Bloch sphere."""
    color = color or theme().qubits[0]
    w = 0.32 * abs(h) if w is None else w
    tip, base = y + h, y + 0.62 * h
    ax.plot([x, x], [y, base], color=color, lw=lw, ls="-", solid_capstyle="round", zorder=z)
    ax.add_patch(Polygon([(x - w / 2, base), (x, tip), (x, base - 0.06 * h)], fc=lighten(color, 0.35), ec="none",
                         zorder=z + 1))
    ax.add_patch(Polygon([(x, base - 0.06 * h), (x, tip), (x + w / 2, base)], fc=color, ec="none", zorder=z + 1))
