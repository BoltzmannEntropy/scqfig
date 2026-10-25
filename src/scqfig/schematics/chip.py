"""Plan-view chip drawing in chip units (micrometres), laid obliquely with motifs.onto_plane.

Metal is drawn as polygons in chip units, so one affine map carries the whole layout rigidly onto a
slanted plane. A ground plane (theme().chip["ground"]) covers the chip; every trace and qubit sits in
an opening of bare substrate (the CPW gaps), drawn below all metal. Colours come from theme().chip and
theme().qubits. Junctions and SQUIDs are drawn as symbols (crosses on a loop) at their true position;
everything else is to scale, so scale_bar tells the truth. squid_detail draws a SQUID at mask level
instead, and zoom_circle magnifies a small region (a claw, a SQUID) into a disc beside the chip, still
to scale. cavity_plan draws a machined 3D cavity in the same units; coupling_graph is a topology glyph
in any units."""
from collections import namedtuple

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D

from ..core import theme
from ..themes import lighten
from .motifs import artists as _artists
from .symbols import arrow

__all__ = ["rounded", "cpw", "substrate", "ground_plane", "launcher", "meander_resonator", "xmon", "double_pad",
           "tunable_coupler", "flux_line", "drive_line", "airbridge", "airbridges", "scale_bar", "coupler_chip",
           "cross_section", "zoom_circle", "claw", "squid_detail", "bowtie", "cavity_plan", "coupling_graph"]

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


_JUNCTIONS = ("cross", "squid", None)


def double_pad(ax, u, v, *, pad=(120.0, 300.0), sep=60.0, gap=60.0, color=None, angle=0.0, pocket=True,
               junction="cross"):
    """A floating double-pad transmon: two pads (w, h) a distance sep apart, joined by a junction on
    thin leads, in a pocket of bare substrate `gap` wide. junction picks the element between the pads:
    "cross" (one junction), "squid" (a loop with a junction on each side, the leads entering the loop
    along the pad axis) or None (pads only). pocket=False leaves out the pocket, for a schematic drawn
    on bare substrate or over a wash. Returns anchors: centre (= junction), the outer pad edges (left,
    right) and the pocket edges (top, bottom)."""
    if junction not in _JUNCTIONS:
        raise ValueError(f"junction must be one of {_JUNCTIONS}, got {junction!r}")
    t = theme()
    color = color or t.qubits[1]
    pw, ph = pad
    x1 = sep / 2 + pw
    if pocket:
        _gap(ax, _place([(-x1 - gap, -ph / 2 - gap), (x1 + gap, -ph / 2 - gap), (x1 + gap, ph / 2 + gap),
                         (-x1 - gap, ph / 2 + gap)], u, v, angle))
    for s in (-1, 1):
        xs = sorted((s * sep / 2, s * x1))
        _poly(ax, _place([(xs[0], -ph / 2), (xs[1], -ph / 2), (xs[1], ph / 2), (xs[0], ph / 2)], u, v, angle),
              color, Z_METAL)
    if junction == "squid":                    # _squid runs along its local y axis: turn it onto the pad axis
        _squid(ax, u, v, angle - 90.0, sep / 2, -sep / 2, 0.42 * sep)
    elif junction == "cross":
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


def _positive(**vals):
    for name, val in vals.items():
        if not val > 0:
            raise ValueError(f"{name} must be positive, got {val!r}")


def _non_negative(**vals):
    for name, val in vals.items():
        if not val >= 0:
            raise ValueError(f"{name} must not be negative, got {val!r}")


def claw(ax, u, v, *, arm, clearance, depth, w=20.0, gap=12.0, angle=0.0, color=None):
    """A claw coupler: a U of coplanar waveguide (centre conductor w wide in gaps of `gap`) wrapped around
    the tip (u, v) of a qubit arm `arm` wide, its metal `clearance` from the arm on all three sides and its
    fingers running back along the arm for `depth` (centreline length from the base). angle rotates it:
    0 wraps an arm that points up, as xmon()["top"]. Both finger ends are open. Returns anchors: tip,
    attach (the middle of the base, where a resonator leaves away from the qubit), the finger ends (left,
    right) and the centreline (N x 2)."""
    _positive(arm=arm, depth=depth, w=w)
    _non_negative(clearance=clearance, gap=gap)
    yb, xf = clearance + w / 2, arm / 2 + clearance + w / 2
    local = [(-xf, yb - depth), (-xf, yb), (xf, yb), (xf, yb - depth)]
    P = cpw(ax, _place(local, u, v, angle), w=w, gap=gap, color=color or theme().chip["resonator"],
            open_start=True, open_end=True)
    out = _anchors(u, v, angle, tip=(0, 0), attach=(0, yb), left=(-xf, yb - depth), right=(xf, yb - depth))
    out["centreline"] = P
    return out


def squid_detail(ax, u, v, *, span, loop=(30.0, 12.0), lead=2.0, jj=0.2, overlap=None, angle=0.0, color=None):
    """A SQUID at mask level, centred at (u, v): a rectangular loop of outer size loop = (w, h) made of
    strips `lead` wide, two leads of the same width along the local y axis that bridge a gap `span` wide
    between two metal edges (reaching `overlap` into each, one lead width by default), and a junction of
    side jj at the middle of each side of the loop, all to scale. Use it in place of the SQUID symbol of
    xmon(squid=False) or under a zoom_circle. Leads are drawn in the reference grey, junctions in the
    junction colour; angle rotates it. Returns anchors: center, top and bottom (the gap edges) and the two
    junctions (left_junction, right_junction)."""
    lx, ly = loop
    _positive(lead=lead, jj=jj)
    if not (lx > 2 * lead and ly > 2 * lead):
        raise ValueError(f"loop {loop} must exceed twice the lead width {lead} in both directions")
    if not span >= ly:
        raise ValueError(f"span {span!r} is shorter than the loop height {ly}")
    overlap = lead if overlap is None else overlap
    _non_negative(overlap=overlap)
    t = theme()
    c, h, a = color or t.reference, span / 2, lead / 2
    strips = [(-a, ly / 2, a, h + overlap), (-a, -h - overlap, a, -ly / 2),                 # leads
              (-lx / 2, ly / 2 - lead, lx / 2, ly / 2), (-lx / 2, -ly / 2, lx / 2, -ly / 2 + lead),
              (-lx / 2, -ly / 2, -lx / 2 + lead, ly / 2), (lx / 2 - lead, -ly / 2, lx / 2, ly / 2)]
    for x0, y0, x1, y1 in strips:
        _poly(ax, _place([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], u, v, angle), c, Z_JJ - 0.5)
    xj, s = lx / 2 - a, jj / 2
    for sx in (-xj, xj):
        _poly(ax, _place([(sx - s, -s), (sx + s, -s), (sx + s, s), (sx - s, s)], u, v, angle), t.chip["junction"],
              Z_JJ)
    return _anchors(u, v, angle, center=(0, 0), top=(0, h), bottom=(0, -h), left_junction=(-xj, 0),
                    right_junction=(xj, 0))


def bowtie(ax, u, v, *, fin=110.0, opening=157.5, sep=12.0, gap=40.0, color=None, angle=0.0):
    """A bowtie qubit: two wedge islands, fins of radius `fin` and full opening angle `opening` (degrees),
    whose apexes face each other across sep and meet a junction on a thin lead, in a rectangular pocket of
    bare substrate `gap` beyond the fins. angle rotates the bowtie axis (0 is horizontal, 90 vertical).
    Returns anchors: center (= junction), the fin tips on the axis (left, right), the top and bottom of
    the fins and the pocket edges (pocket_top, pocket_bottom)."""
    _positive(fin=fin, sep=sep)
    _non_negative(gap=gap)
    if not 0 < opening < 180:
        raise ValueError(f"opening must lie strictly between 0 and 180 degrees, got {opening!r}")
    t = theme()
    color = color or t.qubits[0]
    half = np.radians(opening / 2)
    a = np.linspace(-half, half, 48)
    right = np.vstack([(sep / 2, 0.0), np.column_stack([sep / 2 + fin * np.cos(a), fin * np.sin(a)])])
    xe, ye = sep / 2 + fin, fin * np.sin(half)
    _gap(ax, _place([(-xe - gap, -ye - gap), (xe + gap, -ye - gap), (xe + gap, ye + gap), (-xe - gap, ye + gap)],
                    u, v, angle))
    for sx in (-1, 1):
        _poly(ax, _place(right * (sx, 1), u, v, angle), color, Z_METAL)
    lead = _place([(-sep / 2, 0), (sep / 2, 0)], u, v, angle)
    ax.plot(lead[:, 0], lead[:, 1], color=t.chip["junction"], lw=0.6, ls="-", zorder=Z_JJ)
    k = 0.06 * fin
    for d in (1, -1):
        p = _place([(-k, -d * k), (k, d * k)], u, v, angle)
        ax.plot(p[:, 0], p[:, 1], color=t.chip["junction"], lw=0.9, ls="-", zorder=Z_JJ)
    return _anchors(u, v, angle, center=(0, 0), junction=(0, 0), left=(-xe, 0), right=(xe, 0), top=(0, ye),
                    bottom=(0, -ye), pocket_top=(0, ye + gap), pocket_bottom=(0, -ye - gap))


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


# ---- circular magnifier ----------------------------------------------------------------------

Zoom = namedtuple("Zoom", "target r_target center r scale")


def _tangents(c1, r1, c2, r2):
    """End points of the two external common tangents of the circles (c1, r1) and (c2, r2)."""
    d = np.subtract(c2, c1)
    th, beta = np.arctan2(d[1], d[0]), np.arccos((r1 - r2) / np.hypot(*d))
    out = []
    for sgn in (1, -1):
        n = np.array([np.cos(th + sgn * beta), np.sin(th + sgn * beta)])
        out.append((np.add(c1, r1 * n), np.add(c2, r2 * n)))
    return out


def zoom_circle(ax, target, r_target, center, r, draw, *, magnify=True, label=None, label_side="below", z=20.0,
                lw=0.8):
    """A circular magnifier: a ring of radius r_target around `target`, a disc of radius r at `center`
    and the two external tangents that join them. draw(ax, zoom) fills the disc; zoom holds target,
    r_target, center, r and scale = r / r_target. With magnify=True the callback draws in the chip's own
    units around the target (call xmon, claw or squid_detail again) and every artist it adds is mapped
    onto the disc, scaled by zoom.scale, so the inset stays to scale. With magnify=False it draws directly
    in the disc (zoom.center, zoom.r), for symbols. Either way its artists are clipped to the disc and
    lifted above it. The target ring must lie wholly outside the disc. label goes below or above the
    disc (label_side). Returns the ring, tangents, disc, outline, content, label and scale."""
    _positive(r_target=r_target, r=r)
    if label_side not in ("below", "above"):
        raise ValueError(f"label_side must be 'below' or 'above', got {label_side!r}")
    if not callable(draw):
        raise TypeError("draw must be a callable draw(ax, zoom)")
    target, center = np.asarray(target, float).reshape(2), np.asarray(center, float).reshape(2)
    if np.hypot(*(center - target)) <= r + r_target:
        raise ValueError("the target ring overlaps the magnifier; move center further from the target")
    t = theme()
    ring = ax.add_patch(Circle(tuple(target), r_target, fc="none", ec=t.ink, lw=0.6, zorder=z))
    tangents = [ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=t.ink, lw=0.45, ls="-", zorder=z - 0.5)[0]
                for p0, p1 in _tangents(target, r_target, center, r)]
    disc = ax.add_patch(Circle(tuple(center), r, fc="white", ec="none", lw=0, zorder=z))
    zoom = Zoom(tuple(target), r_target, tuple(center), r, r / r_target)
    before = _artists(ax)
    draw(ax, zoom)
    content = [a for a in ax.lines + ax.patches + ax.texts + ax.collections + ax.images if id(a) not in before]
    T = Affine2D().translate(-target[0], -target[1]).scale(zoom.scale).translate(*center) + ax.transData
    clip = Circle(tuple(center), r, transform=ax.transData)
    for a in content:
        if magnify:
            a.set_transform(T)
        a.set_clip_path(clip)
        a.set_clip_on(True)
        a.set_zorder(a.get_zorder() + z + 1)
    top = max([a.get_zorder() for a in content], default=z + 1) + 0.1
    outline = ax.add_patch(Circle(tuple(center), r, fc="none", ec=t.ink, lw=lw, zorder=top))
    text = None
    if label:
        below = label_side == "below"
        text = ax.text(center[0], center[1] + (-1 if below else 1) * 1.1 * r, label, ha="center",
                       va="top" if below else "bottom", fontsize=t.size - 1, color=t.ink, zorder=top)
    return dict(target=ring, tangents=tangents, disc=disc, outline=outline, content=content, label=text,
                scale=zoom.scale)


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


def _um(x):
    return f"{x / 1e3:.4g} mm" if x >= 1e3 else f"{x:.4g} µm"


# ---- machined 3D cavity in plan view ---------------------------------------------------------

def _rect(r, what):
    if np.size(r) != 4:
        raise ValueError(f"{what} must be a (u0, v0, w, h) rectangle, got {r!r}")
    u0, v0, w, h = (float(q) for q in np.ravel(r))
    if not (w > 0 and h > 0):
        raise ValueError(f"{what} needs a positive width and height, got {r!r}")
    return u0, v0, w, h


def _options(opts, what, allowed, required=()):
    opts = dict(opts)
    unknown = set(opts) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {what} option(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in opts]
    if missing:
        raise ValueError(f"{what} needs {missing}")
    return opts


def _rows(spans):
    """Row index for each (lo, hi) span so that spans in one row do not overlap."""
    rows, ends = [], []
    for lo, hi in spans:
        k = next((i for i, e in enumerate(ends) if all(hi <= a or lo >= b for a, b in e)), len(ends))
        if k == len(ends):
            ends.append([])
        ends[k].append((lo, hi))
        rows.append(k)
    return rows


_PORT_SIDES = {"top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0)}


def cavity_plan(ax, chambers, *, block=None, wall=None, trench=None, chip=None, port=None, dims=True, labels=True):
    """A machined 3D cavity in plan view, to scale in chip units: a metal block (ground role) with vacuum
    chambers cut into it, an optional trench, a chip in the trench with two antenna pads and a junction,
    and a coupling pin through one wall.

    chambers maps names to (u0, v0, w, h) rectangles, or is a list of them. block is the outer rectangle,
    by default the chambers and trench grown by `wall` (15 % of the smallest chamber side). trench is a
    rectangle drawn as vacuum joined to the chambers. chip is a dict: rect (required), pad = (length,
    width) of each antenna pad, sep between their inner ends, at = the junction (the chip centre); the
    pads run along the chip's long side and must fit on it. port is a dict: at (the pin tip inside a
    chamber, required), side ("top", "bottom", "left" or "right", the wall it passes), stub (length
    outside the block) and label. dims=True adds width and height dimensions outside the block, labels
    names the chambers. Unknown options raise ValueError. Returns the patches, the chip and port anchors
    and the extent (u0, u1, v0, v1) of everything drawn."""
    t = theme()
    named = dict(chambers) if isinstance(chambers, dict) else dict(enumerate(chambers))
    if not named:
        raise ValueError("cavity_plan needs at least one chamber")
    rects = {k: _rect(r, f"chamber {k!r}") for k, r in named.items()}
    holes = list(rects.values()) + ([_rect(trench, "trench")] if trench is not None else [])
    lo = np.min([(u0, v0) for u0, v0, _, _ in holes], axis=0)
    hi = np.max([(u0 + w, v0 + h) for u0, v0, w, h in holes], axis=0)
    if block is None:
        m = 0.15 * min(min(w, h) for _, _, w, h in rects.values()) if wall is None else wall
        _non_negative(wall=m)
        block = (lo[0] - m, lo[1] - m, hi[0] - lo[0] + 2 * m, hi[1] - lo[1] + 2 * m)
    bu, bv, bw, bh = _rect(block, "block")
    out = {"block": ax.add_patch(Rectangle((bu, bv), bw, bh, fc=t.chip["ground"], ec=t.chip["edge"], lw=0.7,
                                           zorder=0.5))}
    # openings: outlines first, then fills over them, so the trench mouth leaves no line across it
    for u0, v0, w, h in holes:
        ax.add_patch(Rectangle((u0, v0), w, h, fc="white", ec=t.chip["edge"], lw=1.4, zorder=1.0))
    out["chambers"] = {k: ax.add_patch(Rectangle((u0, v0), w, h, fc="white", ec="none", lw=0, zorder=1.1))
                       for k, (u0, v0, w, h) in rects.items()}
    if trench is not None:
        u0, v0, w, h = holes[-1]
        out["trench"] = ax.add_patch(Rectangle((u0, v0), w, h, fc="white", ec="none", lw=0, zorder=1.1))
    ext = [bu, bu + bw, bv, bv + bh]
    fs = t.size - 1

    if chip is not None:
        c = _options(chip, "chip", ("rect", "pad", "sep", "at"), ("rect",))
        u0, v0, w, h = _rect(c["rect"], "chip rect")
        along = np.array((1.0, 0.0) if w >= h else (0.0, 1.0))
        L, S = max(w, h), min(w, h)
        pl, pw = c.get("pad", (0.22 * L, 0.55 * S))
        sep = c.get("sep", 0.45 * L)
        at = np.asarray(c.get("at", (u0 + w / 2, v0 + h / 2)), float)
        _positive(pad_length=pl, pad_width=pw, sep=sep)
        reach = at @ along - (u0, v0) @ along
        if sep / 2 + pl > min(reach, L - reach) + 1e-9 or pw > S:
            raise ValueError("the antenna pads do not fit on the chip; shorten pad or sep")
        ax.add_patch(Rectangle((u0, v0), w, h, fc=t.chip["substrate"], ec=t.chip["edge"], lw=0.6, zorder=2))
        across = along[::-1]
        qc = t.qubits[0]
        for s in (-1, 1):
            p0 = at + s * along * sep / 2
            p1 = at + s * along * (sep / 2 + pl)
            corners = [p0 - across * pw / 2, p1 - across * pw / 2, p1 + across * pw / 2, p0 + across * pw / 2]
            _poly(ax, corners, qc, Z_METAL)
        lead = np.array([at - along * sep / 2, at + along * sep / 2])
        ax.plot(lead[:, 0], lead[:, 1], color=qc, lw=0.9, ls="-", solid_capstyle="butt", zorder=Z_METAL)
        k = 0.35 * pw
        for d in (1, -1):
            p = np.array([at - along * k - across * d * k, at + along * k + across * d * k])
            ax.plot(p[:, 0], p[:, 1], color=t.chip["junction"], lw=0.9, ls="-", zorder=Z_JJ)
        out["chip"] = dict(junction=tuple(at), pads=(tuple(at - along * (sep / 2 + pl / 2)),
                                                     tuple(at + along * (sep / 2 + pl / 2))))

    if port is not None:
        p = _options(port, "port", ("at", "side", "stub", "label"), ("at",))
        side = p.get("side", "top")
        if side not in _PORT_SIDES:
            raise ValueError(f"port side must be one of {sorted(_PORT_SIDES)}, got {side!r}")
        d = np.array(_PORT_SIDES[side], float)
        stub = p.get("stub", 0.06 * max(bw, bh))
        _positive(stub=stub)
        tip = np.asarray(p["at"], float)
        edge = {"top": bv + bh, "bottom": bv, "left": bu, "right": bu + bw}[side]
        wall_pt = tip.copy()
        wall_pt[0 if d[0] else 1] = edge
        end = wall_pt + d * stub
        ax.plot([tip[0], end[0]], [tip[1], end[1]], color=t.chip["feedline"], lw=1.6, ls="-", solid_capstyle="butt",
                zorder=Z_METAL)
        nrm = d[::-1] * 0.25 * stub
        _poly(ax, [wall_pt - nrm + d * 0.35 * stub, end - nrm, end + nrm, wall_pt + nrm + d * 0.35 * stub],
              t.chip["pad"], Z_METAL - 0.1)
        if p.get("label"):
            ax.text(*(end + d * 0.15 * stub), p["label"], ha={1: "left", -1: "right"}.get(int(d[0]), "center"),
                    va={1: "bottom", -1: "top"}.get(int(d[1]), "center"), fontsize=fs, color=t.ink, zorder=Z_TEXT)
        out["port"] = dict(tip=tuple(tip), wall=tuple(wall_pt), end=tuple(end))
        ext = [min(ext[0], end[0]), max(ext[1], end[0]), min(ext[2], end[1]), max(ext[3], end[1])]

    if labels and isinstance(chambers, dict):
        for name, (u0, v0, w, h) in rects.items():
            ax.text(u0 + w / 2, v0 + h - 0.06 * min(w, h), name, ha="center", va="top", fontsize=t.size - 0.5,
                    color=t.ink, zorder=Z_TEXT)

    if dims:                                   # one dimension per distinct span, stacked where spans overlap
        step = 0.07 * max(bw, bh)
        widths = list(dict.fromkeys((u0, u0 + w) for u0, _, w, _ in rects.values()))
        heights = list(dict.fromkeys((v0, v0 + h) for _, v0, _, h in rects.values()))
        for (a, b), k in zip(widths, _rows(widths)):
            y = bv - (k + 0.6) * step
            arrow(ax, (a, y), (b, y), color=t.ink, style="<|-|>", ms=6, lw=0.6, zorder=Z_TEXT)
            ax.text((a + b) / 2, y - 0.12 * step, _um(b - a), ha="center", va="top", fontsize=fs, color=t.ink,
                    zorder=Z_TEXT)
            ext[2] = min(ext[2], y - 0.7 * step)
        for (a, b), k in zip(heights, _rows(heights)):
            x = bu - (k + 0.6) * step
            arrow(ax, (x, a), (x, b), color=t.ink, style="<|-|>", ms=6, lw=0.6, zorder=Z_TEXT)
            ax.text(x - 0.12 * step, (a + b) / 2, _um(b - a), ha="right", va="center", rotation=90, fontsize=fs,
                    color=t.ink, zorder=Z_TEXT)
            ext[0] = min(ext[0], x - 0.7 * step)
    out["extent"] = tuple(ext)
    return out


# ---- coupling-graph glyph --------------------------------------------------------------------

def coupling_graph(ax, pos, edges=(), *, dashed=(), hub=None, r=0.4, colors=None, hub_color=None):
    """A chip's topology as a small graph, in any units: a disc of radius r per qubit at pos (N x 2),
    a solid coupler-coloured line per coupling in edges, a dashed reference line per pair in dashed (no
    coupler, or a direct coupling only) and, if hub = (x, y) is given, a bus box there with a dashed bus
    link to every qubit, in hub_color (the bus role by default; pass the coupler role where the chip
    draws its bus resonator as a coupling). Qubits take colors (one per qubit) or the first qubit colour.
    Returns the nodes, edges, dashed lines, hub and links."""
    t = theme()
    P = np.asarray(pos, float)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) == 0:
        raise ValueError(f"pos must be a non-empty N x 2 array, got shape {P.shape}")
    _positive(r=r)
    n = len(P)
    if colors is not None and len(colors) != n:
        raise ValueError(f"colors needs one colour per qubit ({n}), got {len(colors)}")

    def pairs(es, what):
        out = []
        for e in es:
            i, j = e
            if not all(isinstance(q, (int, np.integer)) and 0 <= q < n for q in (i, j)) or i == j:
                raise ValueError(f"{what} {e!r} must join two different qubits of {n}")
            out.append((int(i), int(j)))
        return out

    solid, broken = pairs(edges, "edge"), pairs(dashed, "dashed pair")
    out = dict(edges=[], dashed=[], links=[], hub=None, nodes=[])
    for i, j in solid:
        out["edges"] += ax.plot(*zip(P[i], P[j]), color=t.chip["coupler"], lw=1.4, ls="-", solid_capstyle="butt",
                                zorder=3)
    for i, j in broken:
        out["dashed"] += ax.plot(*zip(P[i], P[j]), color=t.reference, lw=0.9, ls=(0, (2, 1.5)), zorder=3)
    if hub is not None:
        hx, hy = np.asarray(hub, float).reshape(2)
        hc = hub_color or t.bus
        for q in P:
            out["links"] += ax.plot([q[0], hx], [q[1], hy], color=hc, lw=1.0, ls=(0, (2, 1.2)), zorder=3)
        out["hub"] = ax.add_patch(Rectangle((hx - r, hy - 0.75 * r), 2 * r, 1.5 * r, fc=hc, ec="none", zorder=4))
    for k, q in enumerate(P):
        fc = colors[k] if colors is not None else t.qubits[0]
        out["nodes"].append(ax.add_patch(Circle(tuple(q), r, fc=fc, ec=t.ink, lw=0.4, zorder=5)))
    return out
