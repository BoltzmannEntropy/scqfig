"""Plan-view chip drawing in chip units (micrometres), laid obliquely with motifs.onto_plane.

Metal is drawn as polygons in chip units, so one affine map carries the whole layout rigidly onto a
slanted plane. A ground plane (theme().chip["ground"]) covers the chip; every trace and qubit sits in
an opening of bare substrate (the CPW gaps), drawn below all metal. Colours come from theme().chip and
theme().qubits. Junctions and SQUIDs are drawn as symbols (crosses on a loop) at their true position;
everything else is to scale, so scale_bar tells the truth."""
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from ..core import theme
from ..themes import lighten

__all__ = ["rounded", "cpw", "substrate", "ground_plane", "launcher", "meander_resonator", "xmon", "double_pad",
           "tunable_coupler", "flux_line", "drive_line", "airbridge", "airbridges", "scale_bar", "coupler_chip",
           "cross_section"]

Z_GAP, Z_METAL, Z_JJ, Z_BRIDGE, Z_TEXT = 2.0, 3.0, 5.0, 6.0, 8.0


# ---- geometry --------------------------------------------------------------------------------

def _dedupe(P):
    P = np.asarray(P, float)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-12]
    return P[keep]


def rounded(pts, r, n=16):
    """The polyline through pts with every interior corner replaced by a circular arc of radius r (a
    fillet), shrunk where a segment is too short to hold it. Returns the (u, v) arrays."""
    P = _dedupe(pts)
    if len(P) < 3 or r <= 0:
        return P[:, 0].copy(), P[:, 1].copy()
    out = [P[0]]
    for i in range(1, len(P) - 1):
        A, B, C = P[i - 1], P[i], P[i + 1]
        a, c = A - B, C - B
        la, lc = np.hypot(*a), np.hypot(*c)
        a, c = a / la, c / lc
        theta = np.arccos(np.clip(a @ c, -1.0, 1.0))         # interior angle at B
        if theta > np.pi - 1e-9:
            out.append(B)
            continue
        d = min(r / np.tan(theta / 2), la / 2, lc / 2)         # tangent length from B
        rr = d * np.tan(theta / 2)
        bis = (a + c) / np.hypot(*(a + c))
        ctr = B + bis * rr / np.sin(theta / 2)
        t1, t2 = B + a * d - ctr, B + c * d - ctr
        a1, a2 = np.arctan2(t1[1], t1[0]), np.arctan2(t2[1], t2[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        s = a1 + da * np.linspace(0, 1, n)
        out.extend(ctr + rr * np.column_stack([np.cos(s), np.sin(s)]))
    out.append(P[-1])
    Q = _dedupe(np.array(out))
    return Q[:, 0], Q[:, 1]


def _ribbon(P, half, ext0=0.0, ext1=0.0):
    """Closed outline of a band of half-width `half` around the polyline P, with mitred vertices, and
    the ends pushed out along the path by ext0 and ext1 (the gap around an open end)."""
    P = _dedupe(P)
    seg = np.diff(P, axis=0)
    tng = seg / np.hypot(*seg.T)[:, None]
    nrm = np.column_stack([-tng[:, 1], tng[:, 0]])
    nv = np.empty_like(P)
    nv[0], nv[-1] = nrm[0], nrm[-1]
    if len(P) > 2:
        m = nrm[:-1] + nrm[1:]
        m /= np.hypot(*m.T)[:, None]
        nv[1:-1] = m / np.clip(np.sum(m * nrm[1:], axis=1), 0.2, None)[:, None]
    P = P.copy()
    P[0] -= tng[0] * ext0
    P[-1] += tng[-1] * ext1
    return np.vstack([P + half * nv, (P - half * nv)[::-1]])


def _place(xy, u, v, angle=0.0):
    """Local outline points rotated by angle (degrees, counter-clockwise) and moved to (u, v)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return np.column_stack([u + c * xy[:, 0] - s * xy[:, 1], v + s * xy[:, 0] + c * xy[:, 1]])


def _poly(ax, xy, fc, z, ec="none", lw=0.0):
    return ax.add_patch(Polygon(xy, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))


def _gap(ax, xy):
    return _poly(ax, xy, theme().chip["substrate"], Z_GAP)


def _cross_xy(a, ext):
    """Plus-shaped outline of half-width a with arm lengths ext = (right, top, left, bottom)."""
    r, t, lft, b = ext
    return [(a, a), (r, a), (r, -a), (a, -a), (a, -b), (-a, -b), (-a, -a), (-lft, -a), (-lft, a), (-a, a), (-a, t),
            (a, t)]


def _squid(ax, u, v, angle, y_top, y_bot, s, lw=0.6):
    """A SQUID symbol on the local x = 0 axis between y_top and y_bot: two leads, a loop of width s
    and a junction cross on each side of the loop. Returns the loop centre in chip units."""
    c = theme().chip["junction"]
    yc, h = (y_top + y_bot) / 2, 0.42 * abs(y_top - y_bot)
    kw = dict(color=c, lw=lw, ls="-", solid_capstyle="butt", solid_joinstyle="miter", zorder=Z_JJ)
    for xy in ([(0, y_top), (0, yc + h / 2)], [(0, yc - h / 2), (0, y_bot)],
               [(-s / 2, yc - h / 2), (s / 2, yc - h / 2), (s / 2, yc + h / 2), (-s / 2, yc + h / 2), (-s / 2, yc - h / 2)]):
        p = _place(xy, u, v, angle)
        ax.plot(p[:, 0], p[:, 1], **kw)
    k = 0.2 * s
    for sx in (-s / 2, s / 2):
        for d in (1, -1):
            p = _place([(sx - k, yc - d * k), (sx + k, yc + d * k)], u, v, angle)
            ax.plot(p[:, 0], p[:, 1], **dict(kw, lw=lw * 1.5))
    return tuple(_place([(0, yc)], u, v, angle)[0])


def _anchors(u, v, angle, **local):
    return {k: tuple(_place([p], u, v, angle)[0]) for k, p in local.items()}


# ---- chip, ground and lines ------------------------------------------------------------------

def substrate(ax, w, h, *, u0=0.0, v0=0.0, z=0.8):
    """The bare chip: a w by h rectangle of substrate with a thin edge."""
    t = theme()
    return ax.add_patch(Rectangle((u0, v0), w, h, fc=t.chip["substrate"], ec=t.chip["edge"], lw=0.6, zorder=z))


def ground_plane(ax, w, h, *, margin=40.0, u0=0.0, v0=0.0, z=1.0):
    """The ground plane covering the chip up to `margin` from its edge."""
    return ax.add_patch(Rectangle((u0 + margin, v0 + margin), w - 2 * margin, h - 2 * margin,
                                  fc=theme().chip["ground"], ec="none", zorder=z))


def cpw(ax, pts, *, w=20.0, gap=12.0, r=None, color=None, open_start=False, open_end=False):
    """A coplanar waveguide along the polyline pts: a centre conductor of width w in `color` (the
    feedline role by default) inside two gaps of bare substrate. r rounds the corners; an open end gets
    a gap around its tip, a closed end meets the ground plane. Returns the centreline (N x 2)."""
    color = color or theme().chip["feedline"]
    P = np.column_stack(rounded(pts, r)) if r else _dedupe(pts)
    _gap(ax, _ribbon(P, w / 2 + gap, gap * open_start, gap * open_end))
    _poly(ax, _ribbon(P, w / 2), color, Z_METAL)
    return P


def flux_line(ax, pts, **kw):
    """A flux-bias line (flux role): a CPW whose end shorts to the ground plane beside a SQUID."""
    kw.setdefault("color", theme().chip["flux"])
    return cpw(ax, pts, **kw)


def drive_line(ax, pts, **kw):
    """A microwave drive line (drive role): a CPW ending open, weakly coupled to its qubit."""
    kw.setdefault("color", theme().chip["drive"])
    kw.setdefault("open_end", True)
    return cpw(ax, pts, **kw)


_SIDES = {"left": 0.0, "bottom": 90.0, "right": 180.0, "top": 270.0}


def launcher(ax, u, v, *, side="left", pad=200.0, taper=160.0, w=20.0, gap=12.0, pad_gap=40.0, color=None):
    """A wire-bond pad centred at (u, v) on the given chip edge, tapering to a CPW of width w that heads
    into the chip. Returns the point where the CPW starts."""
    if side not in _SIDES:
        raise ValueError(f"side must be one of {sorted(_SIDES)}")
    color = color or theme().chip["pad"]
    a = _SIDES[side]
    p, L, e = pad / 2, pad / 2 + taper, 0.6 * w
    metal = [(-p, -p), (p, -p), (L, -w / 2), (L + e, -w / 2), (L + e, w / 2), (L, w / 2), (p, p), (-p, p)]
    g, gp = w / 2 + gap, p + pad_gap
    hole = [(-gp, -gp), (p, -gp), (L, -g), (L + e, -g), (L + e, g), (L, g), (p, gp), (-gp, gp)]
    _gap(ax, _place(hole, u, v, a))
    _poly(ax, _place(metal, u, v, a), color, Z_METAL)
    return tuple(_place([(L, 0.0)], u, v, a)[0])


def meander_resonator(ax, u, v0, v1, *, couple=300.0, width=360.0, turns=6, lead=60.0, tail=60.0, side=-1,
                      w=20.0, gap=12.0, r=None, color=None):
    """A quarter-wave resonator from a coupling elbow at height v0 (running `couple` along a feedline,
    to the left for side=-1) down a meander of `turns` runs of the given width, centred on u, to an open
    end at (u, v1) beside its qubit. Corners are rounded. Returns the centreline (N x 2)."""
    span = abs(v1 - v0)
    pitch = (span - lead - tail) / turns
    if pitch <= w + 2 * gap:
        raise ValueError(f"{turns} turns do not fit in a span of {span:g}; reduce turns or lead")
    sg = np.sign(v1 - v0)
    pts = [(u + side * couple, v0), (u, v0), (u, v0 + sg * lead)]
    x, y = u + width / 2, v0 + sg * lead
    pts.append((x, y))
    for k in range(turns):
        y += sg * pitch
        pts.append((x, y))
        x = 2 * u - x
        if k < turns - 1:
            pts.append((x, y))
    pts += [(u, y), (u, v1)]
    rr = min(r or pitch / 2, pitch / 2, width / 4)
    return cpw(ax, pts, w=w, gap=gap, r=rr, color=color or theme().chip["resonator"], open_end=True)


# ---- qubits and couplers ---------------------------------------------------------------------

def xmon(ax, u, v, *, arm=200.0, width=40.0, gap=30.0, color=None, angle=0.0, squid=True, squid_len=None):
    """An Xmon: a cross of four arms (tip `arm` from the centre, `width` wide) in a gap of bare
    substrate, with a SQUID from the end of the bottom arm to ground (rotate with angle). Returns its
    anchors in chip units: centre, the four arm tips, the SQUID and the ground edge below it."""
    color = color or theme().qubits[0]
    a = width / 2
    sl = (2.0 * width if squid_len is None else squid_len) if squid else 0.0
    _gap(ax, _place(_cross_xy(a + gap, (arm + gap,) * 3 + (arm + max(gap, sl),)), u, v, angle))
    _poly(ax, _place(_cross_xy(a, (arm,) * 4), u, v, angle), color, Z_METAL)
    out = _anchors(u, v, angle, center=(0, 0), right=(arm, 0), top=(0, arm), left=(-arm, 0), bottom=(0, -arm),
                   ground=(0, -arm - max(gap, sl)))
    if squid:
        out["squid"] = _squid(ax, u, v, angle, -arm, -arm - sl, 0.8 * width)
    return out


def double_pad(ax, u, v, *, pad=(120.0, 300.0), sep=60.0, gap=60.0, color=None, angle=0.0):
    """A floating double-pad transmon: two pads (w, h) a distance sep apart, joined by a junction on
    thin leads, in a pocket of bare substrate `gap` wide. Returns anchors: centre (= junction), the outer
    pad edges (left, right) and the pocket edges (top, bottom)."""
    t = theme()
    color = color or t.qubits[1]
    pw, ph = pad
    x1 = sep / 2 + pw
    _gap(ax, _place([(-x1 - gap, -ph / 2 - gap), (x1 + gap, -ph / 2 - gap), (x1 + gap, ph / 2 + gap),
                     (-x1 - gap, ph / 2 + gap)], u, v, angle))
    for s in (-1, 1):
        xs = sorted((s * sep / 2, s * x1))
        _poly(ax, _place([(xs[0], -ph / 2), (xs[1], -ph / 2), (xs[1], ph / 2), (xs[0], ph / 2)], u, v, angle),
              color, Z_METAL)
    lead = _place([(-sep / 2, 0), (sep / 2, 0)], u, v, angle)
    ax.plot(lead[:, 0], lead[:, 1], color=t.chip["junction"], lw=0.6, ls="-", zorder=Z_JJ)
    k = 0.12 * sep
    for d in (1, -1):
        p = _place([(-k, -d * k), (k, d * k)], u, v, angle)
        ax.plot(p[:, 0], p[:, 1], color=t.chip["junction"], lw=0.9, ls="-", zorder=Z_JJ)
    return _anchors(u, v, angle, center=(0, 0), junction=(0, 0), left=(-x1, 0), right=(x1, 0),
                    top=(0, ph / 2 + gap), bottom=(0, -ph / 2 - gap))


def tunable_coupler(ax, u, v, *, length=560.0, width=30.0, stem=140.0, gap=24.0, color=None, angle=0.0,
                    squid_len=None):
    """A grounded tunable coupler: a bar of the given length (its ends face two qubits) on a stem that
    ends in a SQUID to ground, all in a gap of bare substrate. Returns anchors: centre, the bar ends
    (left, right), the SQUID and the ground edge."""
    color = color or theme().chip["coupler"]
    L, a = length / 2, width / 2
    sl = 2.0 * width if squid_len is None else squid_len
    metal = [(-L, -a), (-a, -a), (-a, -stem), (a, -stem), (a, -a), (L, -a), (L, a), (-L, a)]
    g = a + gap
    hole = [(-L - gap, -g), (-g, -g), (-g, -stem - sl), (g, -stem - sl), (g, -g), (L + gap, -g), (L + gap, g),
            (-L - gap, g)]
    _gap(ax, _place(hole, u, v, angle))
    _poly(ax, _place(metal, u, v, angle), color, Z_METAL)
    out = _anchors(u, v, angle, center=(0, 0), left=(-L, 0), right=(L, 0), ground=(0, -stem - sl))
    out["squid"] = _squid(ax, u, v, angle, -stem, -stem - sl, 0.9 * width)
    return out


# ---- airbridges and scale bar ----------------------------------------------------------------

def airbridge(ax, u, v, *, angle=0.0, span=60.0, width=14.0, foot=12.0):
    """An airbridge across a line running at `angle` (degrees) through (u, v): a strap of the given span
    with a landing foot on the ground at each end."""
    t = theme()
    s, f = span / 2, foot
    strap = _place([(-width / 2, -s), (width / 2, -s), (width / 2, s), (-width / 2, s)], u, v, angle)
    _poly(ax, strap, t.tint, Z_BRIDGE, ec=t.reference, lw=0.3)
    for sy in (-1, 1):
        y0, y1 = sorted((sy * s, sy * (s + f)))
        _poly(ax, _place([(-0.8 * width, y0), (0.8 * width, y0), (0.8 * width, y1), (-0.8 * width, y1)], u, v, angle),
              t.chip["edge"], Z_BRIDGE, ec=t.reference, lw=0.3)


def airbridges(ax, path, spacing, *, w=20.0, gap=12.0, **kw):
    """Airbridges every `spacing` along a CPW centreline (from cpw), the first half a spacing in; they
    tie the two ground planes together. Returns the number placed."""
    P = _dedupe(path)
    d = np.hypot(*np.diff(P, axis=0).T)
    s = np.r_[0, np.cumsum(d)]
    kw.setdefault("span", w + 2 * gap + 16)
    n = 0
    for at in np.arange(spacing / 2, s[-1], spacing):
        i = min(np.searchsorted(s, at, side="right") - 1, len(d) - 1)
        f = (at - s[i]) / d[i]
        q = P[i] + f * (P[i + 1] - P[i])
        ang = np.degrees(np.arctan2(*(P[i + 1] - P[i])[::-1]))
        airbridge(ax, q[0], q[1], angle=ang, **kw)
        n += 1
    return n


def scale_bar(ax, u, v, length, label=None, *, color=None, lw=1.2, fs=None, below=False):
    """A scale bar of `length` chip units from (u, v) with end ticks and its label (default "N µm")
    above it (below=True puts it underneath). Returns the bar line."""
    t = theme()
    c = color or t.ink
    k = 0.06 * length
    (line,) = ax.plot([u, u + length], [v, v], color=c, lw=lw, ls="-", solid_capstyle="butt", zorder=Z_TEXT)
    for x in (u, u + length):
        ax.plot([x, x], [v - k / 2, v + k / 2], color=c, lw=lw * 0.7, ls="-", zorder=Z_TEXT)
    ax.text(u + length / 2, v + (-k if below else k), label or f"{length:g} µm", ha="center",
            va="top" if below else "bottom", fontsize=fs or t.size - 1, color=c, zorder=Z_TEXT)
    return line


# ---- a complete two-qubit chip ---------------------------------------------------------------

def coupler_chip(ax, *, labels=True, bridges=True, size=(3600.0, 2400.0)):
    """Two Xmons joined by a grounded tunable coupler on a 3.6 x 2.4 mm chip: a feedline across the top
    with a quarter-wave readout resonator per qubit, and drive (XY), flux (Z) and coupler-flux lines
    from bond pads along the bottom edge. Draw substrate() and ground_plane() first in plan view, or lay
    the result on a slab with motifs.onto_plane. Returns the anchors of every element."""
    t = theme()
    ch = t.chip
    W, H = size
    fs = t.size - 1
    pad_c = 40 + 40 + 100                                  # ground margin + pad gap + half pad
    yf = H - 560
    q = {"Q1": (1250.0, 900.0), "Q2": (2350.0, 900.0)}
    A = {}
    for k, (name, (u, v)) in enumerate(q.items()):
        A[name] = xmon(ax, u, v, color=t.qubits[k])
    A["C"] = tunable_coupler(ax, 1800.0, 900.0, length=640)

    # feedline with launchers at both side edges
    a0 = launcher(ax, pad_c, yf, side="left", color=ch["feedline"])
    a1 = launcher(ax, W - pad_c, yf, side="right", color=ch["feedline"])
    feed = cpw(ax, [a0, a1], color=ch["feedline"])
    A["feedline"] = feed

    # readout resonators, open end 24 um above each qubit's top arm
    for k, (name, width, turns) in enumerate((("R1", 360.0, 6), ("R2", 420.0, 6))):
        qn = f"Q{k + 1}"
        u, top = A[qn]["top"]
        A[name] = meander_resonator(ax, u, yf - 54, top + 24, width=width, turns=turns,
                                    side=-1 if k == 0 else 1)

    # bottom launchers and their lines
    xs = {"XY1": 450.0, "Z1": 1180.0, "Zc": 1800.0, "Z2": 2420.0, "XY2": 3150.0}
    ends = {name: launcher(ax, x, pad_c, side="bottom",
                           color=ch["drive"] if name.startswith("XY") else ch["flux"]) for name, x in xs.items()}
    q1, q2 = A["Q1"], A["Q2"]
    A["XY1"] = drive_line(ax, [ends["XY1"], (450, q1["left"][1]), (q1["left"][0] - 60, q1["left"][1])], r=120)
    A["XY2"] = drive_line(ax, [ends["XY2"], (3150, q2["right"][1]), (q2["right"][0] + 60, q2["right"][1])], r=120)
    A["Z1"] = flux_line(ax, [ends["Z1"], (1180, q1["ground"][1] - 40), (q1["squid"][0] - 34, q1["ground"][1] - 40),
                             (q1["squid"][0] - 34, q1["ground"][1])], r=20)
    A["Z2"] = flux_line(ax, [ends["Z2"], (2420, q2["ground"][1] - 40), (q2["squid"][0] + 34, q2["ground"][1] - 40),
                             (q2["squid"][0] + 34, q2["ground"][1])], r=20)
    c = A["C"]
    A["Zc"] = flux_line(ax, [ends["Zc"], (1800, c["ground"][1] - 40), (c["squid"][0] - 30, c["ground"][1] - 40),
                             (c["squid"][0] - 30, c["ground"][1])], r=20)
    A["launchers"] = dict(ends, **{"in": a0, "out": a1})

    if bridges:
        airbridges(ax, feed, 400)
        for name in ("XY1", "XY2"):
            airbridges(ax, A[name], 500)

    if labels:
        kw = dict(fontsize=fs, ha="center", va="center", zorder=Z_TEXT)
        for k, name in enumerate(("Q1", "Q2")):
            u, v = A[name]["center"]
            ax.text(u - 120 if k == 0 else u + 120, v + 120, name, color=t.qubits[k], **kw)
        ax.text(1800, 900 + 70, "C", color=ch["coupler"], **kw)
        for name, (u, v) in ends.items():
            nm = {"XY1": r"XY$_1$", "XY2": r"XY$_2$", "Z1": r"Z$_1$", "Z2": r"Z$_2$", "Zc": r"Z$_c$"}[name]
            ax.text(u + 45, v + 30, nm, color=ch["drive"] if name.startswith("XY") else ch["flux"],
                    **dict(kw, ha="left", va="bottom"))
        ax.text(a0[0] + 40, yf + 60, "in", color=ch["feedline"], **dict(kw, ha="left", va="bottom"))
        ax.text(a1[0] - 40, yf + 60, "out", color=ch["feedline"], **dict(kw, ha="right", va="bottom"))
        for k, name in enumerate(("R1", "R2")):
            P = A[name]
            u = P[:, 0].min() - 40 if k == 0 else P[:, 0].max() + 40
            ax.text(u, P[:, 1].mean(), name, color=ch["resonator"], **dict(kw, ha="right" if k == 0 else "left"))
    return A


# ---- film-stack cross-section ----------------------------------------------------------------

_IFACE = {"MA": "metal–air", "MS": "metal–substrate", "SA": "substrate–air"}


def cross_section(ax, layers, *, width=10.0, gaps=((3.0, 4.0), (6.0, 7.0)), interfaces=None, labels=True):
    """Cross-section of a coplanar waveguide through its film stack. layers lists (name, thickness in
    nm, kind) from the bottom: one "substrate", optional uniform "film" layers, and one patterned
    "metal" split into ground, centre conductor and ground by `gaps` (x ranges in drawing units of a
    `width`-wide section). The substrate is drawn with a break and films to a compressed scale, so the
    labels carry the true thicknesses. The three lossy interfaces are traced and labelled: MA on the
    top and walls of the metal, MS under it, SA on the bare substrate in the gaps; `interfaces` maps
    MA, MS and SA to label text. Returns the interface artists and the heights used."""
    t = theme()
    kinds = [k for _, _, k in layers]
    if kinds.count("substrate") != 1 or kinds.count("metal") != 1:
        raise ValueError("layers need exactly one 'substrate' and one 'metal'")
    text = dict(_IFACE, **(interfaces or {}))
    colors = {"MA": t.qubits[0], "MS": t.qubits[1], "SA": t.qubits[2]}
    films = [lay for lay in layers if lay[2] != "substrate"]
    tmax = max(th for _, th, _ in films)
    hs = 2.0
    heights = {name: 0.25 + 0.55 * np.sqrt(th / tmax) for name, th, k in films}

    # substrate with a break at the bottom
    xb = np.linspace(0, width, 200)
    yb = 0.12 * np.sin(2 * np.pi * xb / (width / 3))
    sub = next(lay for lay in layers if lay[2] == "substrate")
    ax.fill_between(xb, yb, hs, color=t.chip["substrate"], lw=0, zorder=1)
    ax.plot(xb, yb, color=t.chip["edge"], lw=0.6, ls="-", zorder=2)
    for x in (0, width):
        ax.plot([x, x], [np.interp(x, xb, yb), hs], color=t.chip["edge"], lw=0.6, ls="-", zorder=2)

    # films from the bottom up; the metal is patterned
    metal_x = []
    edges = [0.0] + [e for g in gaps for e in g] + [width]
    for i in range(0, len(edges), 2):
        metal_x.append((edges[i], edges[i + 1]))
    y = hs
    metal_y = None
    lab = []
    for name, th, kind in layers:
        if kind == "substrate":
            continue
        h = heights[name]
        spans = metal_x if kind == "metal" else [(0.0, width)]
        fc = t.chip["edge"] if kind == "metal" else lighten(t.chip["coupler"], 0.75)
        for x0, x1 in spans:
            ax.add_patch(Rectangle((x0, y), x1 - x0, h, fc=fc, ec=t.ink, lw=0.5, zorder=3))
        if kind == "metal":
            metal_y = (y, y + h)
        lab.append((name, th, y + h / 2))
        y += h
    y0, y1 = metal_y

    # the three interfaces
    lw = 2.0
    kw = dict(lw=lw, ls="-", solid_capstyle="butt", zorder=6)
    art = {"MA": [], "MS": [], "SA": []}
    for x0, x1 in metal_x:
        xs, ys = [x0, x1], [y1, y1]
        if x0 > 0:
            xs, ys = [x0] + xs, [y0] + ys
        if x1 < width:
            xs, ys = xs + [x1], ys + [y0]
        art["MA"] += ax.plot(xs, ys, color=colors["MA"], solid_joinstyle="miter", **kw)
        art["MS"] += ax.plot([x0, x1], [y0, y0], color=colors["MS"], **kw)
    below = y0 if kinds.index("metal") == min(i for i, k in enumerate(kinds) if k != "substrate") else hs
    for g0, g1 in gaps:
        art["SA"] += ax.plot([g0, g1], [below, below], color=colors["SA"], **kw)

    top = y + 0.75
    if labels:
        fs = t.size - 0.5
        gx, mx = sum(gaps[0]) / 2, sum(metal_x[-1]) / 2
        # SA over the first gap (label from the left edge), MA on the last ground plane (label ending at the
        # right edge), MS under that ground plane (label in the substrate); leaders run straight down
        for key, px, py, ty, tx, ha in (("SA", gx, below, top, 0.0, "left"), ("MA", mx, y1, top, width, "right"),
                                        ("MS", mx, y0, hs - 0.75, width - 0.1, "right")):
            ax.plot([px, px], [py, ty], color=t.ink, lw=0.5, ls="-", zorder=7)
            ax.plot([px], [py], marker="o", ms=2.2, color=t.ink, ls="none", zorder=7)
            up = ty > py
            ax.text(tx, ty + (0.06 if up else -0.06), f"{key}  {text[key]}", ha=ha, va="bottom" if up else "top",
                    fontsize=fs, color=colors[key], zorder=8)
        for name, th, yc in lab:
            ax.text(width + 0.25, yc, f"{name}  {_nm(th)}", ha="left", va="center", fontsize=fs, color=t.ink)
        ax.text(0.25, 0.75, f"{sub[0]}  {_nm(sub[1])}", ha="left", va="center", fontsize=fs, color=t.ink, zorder=4)
        ax.text(0.05, y1 + 0.45, "vacuum", ha="left", va="center", fontsize=fs, color=t.reference)
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.3, top + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    return {"interfaces": art, "surface": hs, "metal_top": y1, "metal_bottom": y0}


def _nm(th):
    return f"{th / 1e3:g} µm" if th >= 1e3 else f"{th:g} nm"
