"""Flow diagrams: titled boxes of short lines joined by solid, dashed or dotted arrows, placed freely
or on a grid.

    nodes = {"spec": dict(cell=(0, 0), title="Specify", lines=[("band 4-6 GHz", "S")]),
             "chip": dict(cell=(0, 1), title="Layout", lines=["mask"])}
    edges = [("spec", "chip"), ("chip", "spec", dict(ls="dashed", offset=-0.2, label="refit"))]
    flow_diagram(ax, nodes, edges, grid=dict(w=2.0, h=1.0, gap=(0.3, 0.3)), tags={"S": "ink"})

Coordinates are data units of an equal-aspect axes. Inches read best: text keeps its printed size while
boxes scale with the axes, so a diagram drawn in inches on an axes of about that size keeps its text
inside its boxes. Text that would still overflow a box is set smaller."""
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import offset_copy

from ..core import theme
from .layout import _color, _measure, _renderer, _settle
from .symbols import arrow

__all__ = ["flow_diagram"]

_NODE_KEYS = {"xy", "cell", "span", "w", "h", "title", "lines", "text", "align", "color", "fill", "ls", "shape",
              "frame", "draw", "inset", "style", "fs", "pitch", "pad"}
_PITCH = {"left": 1.75, "center": 1.45}     # line pitch in units of the font size, by alignment
_EDGE_KEYS = {"src", "dst", "ls", "color", "lw", "label", "label_at", "label_side", "ports", "at", "offset", "route",
              "via", "path", "head"}
_FAN_KEYS = {"src", "port", "items", "at", "span", "pitch", "trunk", "length", "header", "color"}
_LS = {"solid": "-", "dashed": (0, (3, 2)), "dotted": (0, (1, 1.6))}
_SIDES = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, 1.0), "bottom": (0.0, -1.0)}
_LABEL_OFF = {"right": (4, 0, "left", "center"), "left": (-4, 0, "right", "center"),
              "above": (0, 3, "center", "bottom"), "below": (0, -3, "center", "top")}
_ROUTES = (None, "straight", "hv", "vh")
_PAD = 5.0          # text inset inside a box, points


def _ls(v):
    return _LS.get(v, v) if isinstance(v, str) else v


def _off(ax, dx, dy):
    """Data coordinates shifted by (dx, dy) points."""
    return offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")


# ---- normalising the input ------------------------------------------------------------------

def _norm_nodes(nodes):
    if isinstance(nodes, dict):
        items = [(k, dict(v)) for k, v in nodes.items()]
    else:
        items = []
        for v in nodes:
            v = dict(v)
            if "id" not in v:
                raise ValueError("a node given in a list needs an 'id'")
            items.append((v.pop("id"), v))
    ids = [k for k, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate node ids in {ids}")
    for k, v in items:
        bad = set(v) - _NODE_KEYS
        if bad:
            raise ValueError(f"node {k!r}: unknown keys {sorted(bad)}")
        if v.get("text") is not None and v.get("lines"):
            raise ValueError(f"node {k!r}: give text or lines, not both")
        if v.get("align") not in (None, "left", "center"):
            raise ValueError(f"node {k!r}: align must be 'left' or 'center'")
        if v.get("shape", "round") not in ("round", "square"):
            raise ValueError(f"node {k!r}: shape must be 'round' or 'square'")
        if v.get("draw") is not None and not callable(v["draw"]):
            raise ValueError(f"node {k!r}: draw must be a callable draw(ax, (x, y, w, h))")
        if "pitch" in v and not (isinstance(v["pitch"], (int, float)) and v["pitch"] > 0):
            raise ValueError(f"node {k!r}: pitch must be a positive number (line pitch / font size)")
        if "pad" in v and not (isinstance(v["pad"], (int, float)) and v["pad"] >= 0):
            raise ValueError(f"node {k!r}: pad must be a number of points, zero or more")
    return dict(items)


def _norm_tags(tags):
    t = theme()
    out = {}
    for k, v in (tags or {}).items():
        d = dict(v) if isinstance(v, dict) else dict(color=v)
        bad = set(d) - {"color", "fill", "label"}
        if bad:
            raise ValueError(f"tag {k!r}: unknown keys {sorted(bad)}")
        c, fill = _color(d.get("color"), "ink"), d.get("fill")
        out[k] = dict(color=c, fill="white" if fill is None else _color(fill), text=c if fill is None else t.ink,
                      label=d.get("label"))
    return out


def _node_lines(k, v, tags):
    if v.get("text") is not None:
        return [(s, None) for s in str(v["text"]).split("\n")]
    out = []
    for ln in v.get("lines") or ():
        if isinstance(ln, str):
            out += [(s, None) for s in ln.split("\n")]
        elif isinstance(ln, (tuple, list)) and len(ln) == 2:
            if ln[1] is not None and ln[1] not in tags:
                raise ValueError(f"node {k!r}: unknown tag {ln[1]!r}; define it in tags=")
            out.append((str(ln[0]), ln[1]))
        else:
            raise ValueError(f"node {k!r}: line {ln!r} is neither a string nor a (text, tag) pair")
    return out


def _boxes(nodes, grid, size):
    """Each node's (x, y, w, h); grid cells count rows from the top, so row 0 is the top row."""
    boxes, G = {}, None
    cells = {k: v for k, v in nodes.items() if "cell" in v}
    if cells:
        if grid is None:
            raise ValueError("nodes placed by cell need grid=dict(w=..., h=..., gap=(gx, gy))")
        g = dict(grid)
        gx, gy = (float(v) for v in g.get("gap", (0.3, 0.3)))
        spans = {}
        for k, v in cells.items():
            r, c = v["cell"]
            nr, nc = v.get("span", (1, 1))
            if not all(isinstance(i, (int, np.integer)) for i in (r, c, nr, nc)) or min(r, c) < 0 or min(nr, nc) < 1:
                raise ValueError(f"node {k!r}: cell and span must be integers, cell >= 0 and span >= 1")
            spans[k] = (r, c, nr, nc)
        nrows = max(r + nr for r, c, nr, nc in spans.values())
        ncols = max(c + nc for r, c, nr, nc in spans.values())
        widths = [float(w) for w in g["widths"]] if "widths" in g else [float(g.get("w", size[0]))] * ncols
        heights = [float(h) for h in g["heights"]] if "heights" in g else [float(g.get("h", size[1]))] * nrows
        if len(widths) < ncols or len(heights) < nrows:
            raise ValueError("grid widths and heights must cover every column and row in use")
        xs = [sum(widths[:c]) + c * gx for c in range(ncols)]
        top = sum(heights[:nrows]) + (nrows - 1) * gy
        labels = g.get("col_labels")
        if labels is not None and len(labels) > ncols:
            raise ValueError(f"{len(labels)} column labels for {ncols} columns")
        G = dict(xs=xs, widths=widths, top=top, labels=labels)
        for k, (r, c, nr, nc) in spans.items():
            w = sum(widths[c:c + nc]) + (nc - 1) * gx
            h = sum(heights[r:r + nr]) + (nr - 1) * gy
            x, y = xs[c], top - sum(heights[:r]) - r * gy - h
            v = nodes[k]
            if "w" in v or "h" in v:            # a smaller box centred in its cell
                w2, h2 = float(v.get("w", w)), float(v.get("h", h))
                x, y, w, h = x + (w - w2) / 2, y + (h - h2) / 2, w2, h2
            boxes[k] = (x, y, w, h)
    for k, v in nodes.items():
        if "cell" in v:
            continue
        if "xy" not in v:
            raise ValueError(f"node {k!r}: give xy=(x, y) for its lower-left corner or cell=(row, col)")
        x, y = (float(c) for c in v["xy"])
        boxes[k] = (x, y, float(v.get("w", size[0])), float(v.get("h", size[1])))
    for k, (x, y, w, h) in boxes.items():
        if w <= 0 or h <= 0:
            raise ValueError(f"node {k!r}: width and height must be positive")
    return boxes, G


def _end(v, boxes):
    """(node id, None) or (None, point) for one end of an edge."""
    try:
        if v in boxes:
            return v, None
    except TypeError:
        pass
    if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(c, (int, float, np.number)) for c in v):
        return None, (float(v[0]), float(v[1]))
    raise ValueError(f"edge end {v!r} is neither a node id nor an (x, y) point")


def _norm_edges(edges, boxes):
    out = []
    for e in edges or ():
        if isinstance(e, dict):
            d = dict(e)
        elif isinstance(e, (tuple, list)) and len(e) in (2, 3):
            d = dict(e[2]) if len(e) == 3 else {}
            d.update(src=e[0], dst=e[1])
        else:
            raise ValueError(f"edge {e!r}: give (src, dst), (src, dst, options) or a dict")
        bad = set(d) - _EDGE_KEYS
        if bad:
            raise ValueError(f"edge {d.get('src')!r} -> {d.get('dst')!r}: unknown keys {sorted(bad)}")
        if "path" not in d:
            if "src" not in d or "dst" not in d:
                raise ValueError(f"edge {d!r} needs src and dst")
            _end(d["src"], boxes)
            _end(d["dst"], boxes)
            if isinstance(d["src"], str) and d["src"] == d["dst"] and not d.get("via"):
                raise ValueError(f"edge from {d['src']!r} to itself: give via=[...] corner points or a path")
        if d.get("label_side") not in (None, *_LABEL_OFF):
            raise ValueError(f"label_side must be one of {sorted(_LABEL_OFF)}")
        if d.get("route") not in _ROUTES:
            raise ValueError(f"route must be one of {_ROUTES[1:]}")
        ports = d.get("ports")
        if ports is not None and (len(ports) != 2 or any(p is not None and p not in _SIDES for p in ports)):
            raise ValueError(f"ports must be two of {sorted(_SIDES)} (or None)")
        out.append(d)
    return out


def _norm_fans(fans, boxes):
    out = []
    for f in fans or ():
        d = dict(f)
        bad = set(d) - _FAN_KEYS
        if bad:
            raise ValueError(f"fan: unknown keys {sorted(bad)}")
        if d.get("src") not in boxes:
            raise ValueError(f"fan source {d.get('src')!r} is not a node")
        d["port"] = d.get("port", "bottom")
        if d["port"] not in ("bottom", "right", "left"):
            raise ValueError("a fan leaves its node at the 'bottom', 'right' or 'left'")
        d["items"] = [str(s) for s in d.get("items", ())]
        if d.get("at") is not None and len(d["at"]) != len(d["items"]):
            raise ValueError(f"{len(d['at'])} positions for {len(d['items'])} fan items")
        out.append(d)
    return out


# ---- edge geometry --------------------------------------------------------------------------

def _side_pt(b, side, at=None):
    x, y, w, h = b
    if side in ("left", "right"):
        return (x if side == "left" else x + w, y + h / 2 if at is None else float(at))
    return (x + w / 2 if at is None else float(at), y + h if side == "top" else y)


def _shift(p, side, gap):
    dx, dy = _SIDES[side]
    return p[0] + gap * dx, p[1] + gap * dy


def _centre(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def _facing(b, target, vertical):
    cx, cy = _centre(b)
    if vertical:
        return "top" if target[1] > cy else "bottom"
    return "right" if target[0] > cx else "left"


def _dominant(b, target):
    cx, cy = _centre(b)
    return _facing(b, target, abs(target[1] - cy) / b[3] > abs(target[0] - cx) / b[2])


def _orth(A, B):
    """Facing sides and the middle of the shared span when two boxes sit side by side or one above
    the other; None otherwise."""
    ax_, ay, aw, ah = A
    bx, by, bw, bh = B
    oy = min(ay + ah, by + bh) - max(ay, by)
    ox = min(ax_ + aw, bx + bw) - max(ax_, bx)
    if oy > 0 >= ox:
        mid = (max(ay, by) + min(ay + ah, by + bh)) / 2
        return ("right", "left", mid) if bx >= ax_ + aw else ("left", "right", mid)
    if ox > 0 >= oy:
        mid = (max(ax_, bx) + min(ax_ + aw, bx + bw)) / 2
        return ("bottom", "top", mid) if by + bh <= ay else ("top", "bottom", mid)
    return None


def _edge_path(e, boxes, gap):
    if "path" in e:
        pts = [tuple(float(c) for c in p) for p in e["path"]]
        if len(pts) < 2:
            raise ValueError("an edge path needs at least two points")
        return pts
    ka, pa = _end(e["src"], boxes)
    kb, pb = _end(e["dst"], boxes)
    A = boxes[ka] if pa is None else None
    B = boxes[kb] if pb is None else None
    via = [tuple(float(c) for c in p) for p in e.get("via", ())]
    sa, sb = e.get("ports") or (None, None)
    at = e.get("at")
    at_a, at_b = at if isinstance(at, (tuple, list)) else (at, at)
    route = e.get("route")
    off = float(e.get("offset", 0.0))
    if A is not None and B is not None and not via and sa is None and sb is None and route is None:
        auto = _orth(A, B)
        if auto is not None:
            sa, sb, mid = auto
            ca = mid + off if at_a is None else at_a
            cb = ca if at_b is None else at_b
            return [_shift(_side_pt(A, sa, ca), sa, gap), _shift(_side_pt(B, sb, cb), sb, gap)]
        route = "vh"
    route = route or "straight"
    first = via[0] if via else (pb if B is None else _centre(B))
    last = via[-1] if via else (pa if A is None else _centre(A))
    if A is not None:
        if sa is None:
            sa = _dominant(A, first) if route == "straight" else _facing(A, first, route == "vh")
        start = _shift(_side_pt(A, sa, at_a), sa, gap)
    else:
        start = pa
    if B is not None:
        if sb is None:
            sb = _dominant(B, last) if route == "straight" else _facing(B, last, route == "hv")
        end = _shift(_side_pt(B, sb, at_b), sb, gap)
    else:
        end = pb
    if via or route == "straight":
        pts = [start, *via, end]
    elif route == "hv":
        pts = [start, (end[0], start[1]), end]
    else:
        pts = [start, (start[0], end[1]), end]
    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    if len(clean) < 2:
        raise ValueError(f"edge {e['src']!r} -> {e['dst']!r} has no length")
    return clean


def _draw_path(ax, pts, color, lw, ls, head):
    arts = []
    if len(pts) > 2:
        arts.append(ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw, ls=ls,
                            solid_capstyle="butt", solid_joinstyle="miter", zorder=2)[0])
    if head:
        arts.append(arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=8, ls=ls, zorder=2))
    else:
        arts.append(ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=color, lw=lw, ls=ls,
                            zorder=2)[0])
    return arts


def _path_label(ax, pts, text, frac, side, fs, color):
    P = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    target = float(np.clip(frac, 0.0, 1.0)) * seg.sum()
    k = min(int(np.searchsorted(np.cumsum(seg), target)), len(seg) - 1)
    u = (target - seg[:k].sum()) / seg[k] if seg[k] > 0 else 0.0
    p, d = P[k] + u * (P[k + 1] - P[k]), P[k + 1] - P[k]
    side = side or ("right" if abs(d[1]) >= abs(d[0]) else "above")
    dx, dy, ha, va = _LABEL_OFF[side]
    return ax.text(p[0], p[1], text, transform=_off(ax, dx, dy), ha=ha, va=va, fontsize=fs, color=color, zorder=4)


# ---- node text ------------------------------------------------------------------------------

def _badge(ax, x, y, dx, dy, name, spec, f, ha="right"):
    return ax.text(x, y, name, transform=_off(ax, dx, dy), ha=ha, va="center", fontsize=f, weight="bold",
                   color=spec["text"], zorder=5,
                   bbox=dict(boxstyle="round,pad=0.12", fc=spec["fill"], ec=spec["color"], lw=0.7))


def _node_text(ax, v, region, lines, f, tags, pad=_PAD):
    t = theme()
    x, y, w, h = region
    title = v.get("title")
    align = v.get("align") or ("left" if title and lines else "center")
    style = v.get("style", "normal")
    texts, badges = [], []
    kw = dict(color=t.ink, style=style, zorder=4)
    if align == "left":
        pitch, y0 = v.get("pitch", _PITCH["left"]) * f, -pad
        if title:
            texts.append(ax.text(x, y + h, title, transform=_off(ax, pad, -pad), ha="left", va="top",
                                 fontsize=f + 0.9, weight="bold", **kw))
            y0 = -(pad + 1.3 * (f + 0.9) + 2)
        for k, (s, tag) in enumerate(lines):
            yk = y0 - (k + 0.5) * pitch
            texts.append(ax.text(x, y + h, s, transform=_off(ax, pad, yk), ha="left", va="center", fontsize=f, **kw))
            if tag is not None:
                badges.append(_badge(ax, x + w, y + h, -pad - 1, yk, tag, tags[tag], f))
    else:
        pitch = v.get("pitch", _PITCH["center"]) * f
        block =([(title, True, None)] if title else []) + [(s, False, tag) for s, tag in lines]
        n = len(block)
        for k, (s, bold, tag) in enumerate(block):
            yk = ((n - 1) / 2 - k) * pitch
            texts.append(ax.text(x + w / 2, y + h / 2, s, transform=_off(ax, 0, yk), ha="center", va="center",
                                 fontsize=f + (0.9 if bold else 0.0), weight="bold" if bold else "normal", **kw))
            if tag is not None:
                badges.append(_badge(ax, x + w, y + h / 2, -pad - 1, yk, tag, tags[tag], f))
    return texts, badges, align


def _text_scale(ax, region, texts, badges, align, f, pad=_PAD):
    """Factor (at most 1) by which the node's font must shrink to stay inside its box."""
    fig = ax.figure
    r, k = _renderer(fig), fig.dpi / 72
    x, y, w, h = region
    (X0, Y0), (X1, Y1) = ax.transData.transform([(x, y), (x + w, y + h)])
    ext = [a.get_window_extent(r) for a in texts if a.get_text()]
    if not ext:
        return 1.0
    badge = max([b.get_window_extent(r).width for b in badges], default=0.0)
    extra = badge + (2 * 0.12 * f + 4) * k if badges else 0.0
    need_w = max(e.width for e in ext) + extra + 2 * pad * k
    if align == "left":
        need_h = Y1 - min(e.y0 for e in ext) + pad * k
    else:
        need_h = max(e.y1 for e in ext) - min(e.y0 for e in ext) + 2 * pad * k
    return min(1.0, (X1 - X0) / need_w, (Y1 - Y0) / need_h)


# ---- fans, legend and fitting ---------------------------------------------------------------

def _draw_fan(ax, d, box, fs, gap):
    """A trunk out of a node with one short arrow per item, each labelled at its tip. Returns the
    artists, the labels and the points that bound the fan."""
    x, y, w, h = box
    n = len(d["items"])
    if n == 0:
        return [], [], []
    col = _color(d.get("color"), "ink")
    pitch, length = float(d.get("pitch", 0.45)), float(d.get("length", 0.4))
    sgn = -1.0 if d["port"] == "left" else 1.0
    cy = y + h / 2
    if d["port"] == "bottom":
        xt = float(d.get("trunk", x + w / 2))
        default = [y - pitch * (k + 1) for k in range(n)]
    else:
        xt = float(d.get("trunk", x + w + 0.35 if sgn > 0 else x - 0.35))
        default = [cy + pitch * ((n - 1) / 2 - k) for k in range(n)]
    if d.get("at") is not None:
        ys = [float(v) for v in d["at"]]
    elif d.get("span") is not None:
        ys = list(np.linspace(float(d["span"][0]), float(d["span"][1]), n))
    else:
        ys = default
    kw = dict(color=col, lw=0.8, ls="-", zorder=2)
    arts = []
    if d["port"] == "bottom":
        arts += ax.plot([xt, xt], [y - gap, min(ys)], **kw)
        pts = [(xt, y)]
    else:
        edge = x + w + gap if sgn > 0 else x - gap
        arts += ax.plot([edge, xt], [cy, cy], **kw)
        arts += ax.plot([xt, xt], [min(ys + [cy]), max(ys + [cy])], **kw)
        pts = [(edge, cy)]
    labels = []
    ha = "left" if sgn > 0 else "right"
    for yk, s in zip(ys, d["items"]):
        tip = (xt + sgn * length, yk)
        arts.append(arrow(ax, (xt, yk), tip, color=col, lw=0.8, ms=6, zorder=2))
        labels.append(ax.text(tip[0], yk, s, transform=_off(ax, sgn * 4, 0), ha=ha, va="center", fontsize=fs,
                              color=theme().ink, zorder=4))
        pts += [(xt, yk), tip]
    if d.get("header"):
        labels.append(ax.text(xt + sgn * length, max(ys), d["header"], transform=_off(ax, sgn * 4, 1.1 * fs),
                              ha=ha, va="bottom", fontsize=fs, style="italic", color=theme().ink, zorder=4))
    return arts, labels, pts


def _fit_limits(ax, geo, texts, margin):
    P = np.asarray(geo, dtype=float)
    (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    ax.set_aspect("equal", adjustable="box")
    for _ in range(4):
        ax.set_xlim(x0 - margin, x1 + margin)
        ax.set_ylim(y0 - margin, y1 + margin)
        _settle(ax)
        inv, r = ax.transData.inverted(), _renderer(ax.figure)
        grown = False
        for tx in texts:
            bb = tx.get_window_extent(r)
            (a, b), (c, d) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            if a < x0 - 1e-9 or b < y0 - 1e-9 or c > x1 + 1e-9 or d > y1 + 1e-9:
                x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
                grown = True
        if not grown:
            break
    return x0, x1, y0, y1


def flow_diagram(ax, nodes, edges=(), *, grid=None, size=(2.0, 1.0), tags=None, legend=False, fans=(), fs=None,
                 fit=True, margin=0.08, min_fs=None, pad=None):
    """Boxes with a bold title and short lines (each line may carry a status tag), joined by arrows.

    nodes: a dict {id: spec} or a list of specs with an "id". A spec places the box by xy=(x, y) (lower-left
    corner, with w and h, default `size`) or by cell=(row, col) on the grid (row 0 at the top; span=(rows,
    cols); w and h shrink the box inside its cell). Text is title plus lines (strings or (text, tag)
    pairs), or text, a centred string with newlines; align is "left" (default with a title and lines) or
    "center". Box options: color (edge), fill, ls ("solid", "dashed", "dotted"), shape ("round",
    "square"), frame=False for a bare label, style (font style, e.g. "italic"), fs, pitch (line pitch in
    units of the font size, 1.75 for left-aligned and 1.45 for centred text by default), and draw, a callable
    draw(ax, (x, y, w, h)) that draws an icon in the box (in its left `inset` width when inset is set,
    with the text centred in the rest), and pad, the text inset from the box edge in points.
    grid: dict(w, h, gap=(gx, gy)), or widths and heights lists for uneven columns and rows, and
    col_labels, italic headings above the columns.
    edges: (src, dst), (src, dst, options) or dicts; an end is a node id or an (x, y) point. Boxes side by
    side or one above the other are joined straight across the middle of their shared span (shifted by
    offset, or placed at at=coordinate); other pairs get an elbow (route "vh", or "hv", or "straight").
    ports=(side, side) picks the sides ("left", "right", "top", "bottom"), at=(a, b) the coordinate along
    each, via a list of corner points, path the whole polyline. ls, color, lw, head=False, and label
    with label_at (fraction along the path) and label_side ("left", "right", "above", "below").
    tags: {name: colour or dict(color, fill, label)}; a tag is a bold badge at the right end of its line,
    and legend=True lays the labelled tags in a row under the diagram.
    fans: dicts with src, port ("bottom", "right", "left"), items, and at (positions along the trunk),
    span=(first, last) or pitch; trunk and length set the trunk position and the arrow length, header an
    italic heading over the items.
    fit=True sets equal-aspect limits around everything drawn and shrinks box text that would overflow;
    fit=False leaves limits, aspect and text sizes to the caller. pad sets the text inset of every box
    in points (5 by default; a node's own pad wins), for narrow boxes whose text must keep its size.

    Returns a dict: boxes {id: (x, y, w, h)}, frames, texts and badges by node, edges (artists), labels,
    fans, legend and extent."""
    t = theme()
    N = _norm_nodes(nodes)
    T = _norm_tags(tags)
    lines = {k: _node_lines(k, v, T) for k, v in N.items()}
    boxes, G = _boxes(N, grid, size)
    E = _norm_edges(edges, boxes)
    F = _norm_fans(fans, boxes)
    fs = fs or t.size - 0.5
    min_fs = min_fs or t.size - 2.0
    if pad is not None and not (isinstance(pad, (int, float)) and pad >= 0):
        raise ValueError("pad must be a number of points, zero or more")
    pad = _PAD if pad is None else float(pad)
    gap = 0.03
    ax.axis("off")
    out = dict(boxes=boxes, frames={}, texts={}, badges={}, edges=[], labels=[], fans=[], legend=[], extent=None)
    geo, outside, nodetext = [], [], {}
    for k, v in N.items():
        x, y, w, h = boxes[k]
        geo += [(x, y), (x + w, y + h)]
        if v.get("frame", True):
            style = "round,pad=0,rounding_size=0.06" if v.get("shape", "round") == "round" else "square,pad=0"
            p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=_color(v.get("fill", "white")),
                               ec=_color(v.get("color"), "ink"), lw=0.8, ls=_ls(v.get("ls", "solid")), zorder=3)
            ax.add_patch(p)
            out["frames"][k] = p
        inset = float(v.get("inset", 0.0))
        if v.get("draw") is not None:
            v["draw"](ax, (x, y, inset if inset > 0 else w, h))
        region = (x + inset, y, w - inset, h)
        f = v.get("fs", fs)
        p = float(v.get("pad", pad))
        texts, badges, align = _node_text(ax, v, region, lines[k], f, T, p)
        out["texts"][k], out["badges"][k] = texts, badges
        nodetext[k] = (region, align, f, p)
        if not v.get("frame", True):
            outside += texts
    for e in E:
        pts = _edge_path(e, boxes, gap)
        col = _color(e.get("color"), "ink")
        out["edges"] += _draw_path(ax, pts, col, float(e.get("lw", 0.9)), _ls(e.get("ls", "solid")),
                                   e.get("head", True))
        geo += pts
        if e.get("label"):
            lab = _path_label(ax, pts, e["label"], e.get("label_at", 0.5), e.get("label_side"), fs, col)
            out["labels"].append(lab)
            outside.append(lab)
    for d in F:
        arts, labels, pts = _draw_fan(ax, d, boxes[d["src"]], fs, gap)
        out["fans"] += arts
        out["labels"] += labels
        outside += labels
        geo += pts
    if G is not None and G["labels"]:
        for c, s in enumerate(G["labels"]):
            if s:
                lab = ax.text(G["xs"][c] + G["widths"][c] / 2, G["top"], s, transform=_off(ax, 0, 4), ha="center",
                              va="bottom", fontsize=fs, style="italic", color=t.ink)
                out["labels"].append(lab)
                outside.append(lab)
    keyed = [(k, s) for k, s in T.items() if s["label"]]
    if legend and keyed and geo:
        P = np.asarray(geo, dtype=float)
        (x0, y0), (x1, _) = P.min(axis=0), P.max(axis=0)
        yl, slot = y0 - 0.3, (x1 - x0) / len(keyed)
        for i, (k, s) in enumerate(keyed):
            xl = x0 + i * slot + 0.05
            b = _badge(ax, xl, yl, 2, 0, k, s, fs, ha="left")
            wpt = _measure(ax.figure, k, fs, weight="bold")[0] * 72 + 2 * 0.12 * fs + 2
            lab = ax.text(xl, yl, s["label"], transform=_off(ax, 2 + wpt + 5, 0), ha="left", va="center", fontsize=fs,
                          color=t.ink)
            out["legend"] += [b, lab]
            outside += [b, lab]
            geo.append((xl, yl))
    if fit and geo:
        out["extent"] = _fit_limits(ax, geo, outside, margin)
        _settle(ax)
        for k, v in N.items():
            if not v.get("frame", True):
                continue
            region, align, f, p = nodetext[k]
            s = _text_scale(ax, region, out["texts"][k], out["badges"][k], align, f, p)
            if s < 0.995:
                for a in out["texts"][k] + out["badges"][k]:
                    a.remove()
                texts, badges, _ = _node_text(ax, v, region, lines[k], max(min_fs, f * s * 0.97), T, p)
                out["texts"][k], out["badges"][k] = texts, badges
    return out
