"""Figure frames built from text: numbered recipe grids with mini-plots, framed equation panels,
bracketed matrices with coloured labels and Hamiltonian terms laid on their element tiles.

    res = recipe_grid(None, [dict(title="Specify", lines=["Fix the band."], panel=draw_band,
                                  tools=["pen", "paper"]), ...], ncol=4)
    sq.save(res["fig"], "figures/recipe")

recipe_grid lays out a whole figure in inches and sets the figure size to fit it. text_panel,
matrix_text and term_tiles take over one axes: they turn its ticks off and measure their text with the
figure's renderer, so text that would not fit is rewrapped or set smaller. Colours are theme roles
("ink", "bus", "reference", "accent", a tile kind such as "nonlinear") or plain colours, resolved
when the frame is drawn; font sizes are offsets from theme().size."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from .. import core
from ..core import theme
from ..themes import lighten
from .symbols import arrow, tile

__all__ = ["recipe_grid", "text_panel", "matrix_text", "term_tiles"]

ROLES = ("ink", "data", "model", "sim", "reference", "assumed", "accent", "signal", "bus", "drive", "wash",
         "light", "tint", "overlay")


# ---- shared text helpers --------------------------------------------------------------------

def _color(c, default=None):
    """A theme role ("ink", "bus", ...), a tile kind ("nonlinear", ...) or a plain colour."""
    t = theme()
    if c is None:
        return t.ink if default is None else _color(default)
    if isinstance(c, str):
        if c in ROLES:
            return getattr(t, c)
        if c in t.tiles:
            return t.tiles[c]
    return c


def _renderer(fig):
    get = getattr(fig.canvas, "get_renderer", None)
    return get() if get is not None else fig._get_renderer()


def _measure(fig, s, fs, **kw):
    """(width, ascent, descent) of the one-line string s at fs points, in inches. A mathtext error
    raises ValueError here, before anything is drawn."""
    if not s:
        return 0.0, 0.0, 0.0
    t = fig.text(0, 0, s, fontsize=fs, va="baseline", **kw)
    try:
        bb = t.get_window_extent(_renderer(fig))
    finally:
        t.remove()
    return bb.width / fig.dpi, max(bb.y1, 0.0) / fig.dpi, max(-bb.y0, 0.0) / fig.dpi


def _settle(ax):
    """Let the figure's layout engine place ax, then return its box in pixels."""
    fig = ax.figure
    if fig.get_layout_engine() is not None:
        fig.draw_without_rendering()
    ax.apply_aspect()
    return ax.get_window_extent(_renderer(fig))


def _words(s):
    """Split at spaces outside $...$, so a mathtext span never breaks across lines."""
    out, cur, math = [], "", False
    for ch in s:
        if ch == "$":
            math = not math
        if ch == " " and not math:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [w for w in out if w]


def _wrap(fig, text, width, fs, first=None, **kw):
    """Greedy wrap of text at width inches (the first line at `first`). A word wider than the line
    stays on a line of its own."""
    first = width if first is None else first
    out, cur = [], ""
    for word in _words(text):
        trial = f"{cur} {word}".strip()
        if cur and _measure(fig, trial, fs, **kw)[0] > (first if not out else width):
            out.append(cur)
            cur = word
        else:
            cur = trial
    if cur or not out:
        out.append(cur)
    return out


def _fit_lines(fig, lines, width, fs, **kw):
    """Keep the author's line breaks when every line fits width inches; otherwise rewrap the whole
    paragraph at that width."""
    if all(_measure(fig, ln, fs, **kw)[0] <= width for ln in lines):
        return list(lines)
    return _wrap(fig, " ".join(ln for ln in lines if ln), width, fs, **kw)


_HANG = 0.1         # hanging indent of a wrapped item under wrap="lines", inches


def _box_lines(fig, lines, width, fs, wrap):
    """The lines of a box as (text, x offset) pairs. wrap="paragraph" keeps the author's breaks when
    every line fits and otherwise rewraps the whole paragraph; wrap="lines" keeps each line as an item
    of its own and wraps only a line that runs past the box, its continuation set in by _HANG;
    wrap="shrink" never wraps (recipe_grid has already set the size at which the lines fit)."""
    if wrap == "paragraph":
        return [(s, 0.0) for s in _fit_lines(fig, lines, width, fs)]
    if wrap == "shrink":
        return [(s, 0.0) for s in lines]
    out = []
    for ln in lines:
        parts = _wrap(fig, ln, width - _HANG, fs, first=width)
        out += [(p, 0.0 if k == 0 else _HANG) for k, p in enumerate(parts)]
    return out


def _as_lines(lines):
    if lines is None:
        return []
    if isinstance(lines, str):
        return lines.split("\n") if lines else []
    out = []
    for ln in lines:
        out += str(ln).split("\n")
    return out


def _inch_axes(ax):
    """Take over ax: no frame or ticks, data units in inches from its lower-left corner."""
    fig = ax.figure
    bb = _settle(ax)
    aw, ah = bb.width / fig.dpi, bb.height / fig.dpi
    ax.set_xlim(0, aw)
    ax.set_ylim(0, ah)
    ax.set_aspect("auto")
    ax.axis("off")
    return aw, ah


# ---- recipe grid ----------------------------------------------------------------------------

TOOL_ICONS = ("pen", "paper", "code", "calc", "pc", "em", "probe", "vna", "adc", "fridge", "stats")
_TOOL_LABELS = {"pen": "pen", "paper": "paper", "code": "code", "calc": "calc", "pc": "computer", "em": "EM",
                "probe": "probe", "vna": "VNA", "adc": "ADC", "fridge": "fridge", "stats": "stats"}
_ORDERS = ("row", "snake", "column")
_PLOTS = ("below", "right")
_WRAPS = ("paragraph", "lines", "shrink")
_SIDES = ("below", "left", "right", "between")
_STEP_KEYS = {"title", "lines", "panel", "tools", "number", "panel_left"}
_SIDE_KEYS = {"after", "at", "title", "lines", "panel", "tools", "width", "x", "cols", "panel_left"}
# vertical metrics of a box in inches; numbered boxes keep room for the circled number
_TITLE_Y = {True: 0.15, False: 0.13}
_LINES_TOP = {True: 0.31, False: 0.27}
_TOOLS_Y = 0.14
_PANEL_BOTTOM = {"below": 0.56, "right": 0.50}
_PAD = 0.08


def _icon(ax, name, x, y):
    """Tool pictogram in inches on the background axes, starting at x and centred on y; returns its
    width."""
    t = theme()
    ink, s, lw = t.ink, 0.075, 0.7
    k = dict(fc="white", ec=ink, lw=lw, zorder=5)
    line = dict(color=ink, zorder=6)
    if name == "pen":
        ax.add_patch(Polygon([[x, y - s], [x + 0.03, y - 0.6 * s], [x + 0.13, y + 0.6 * s], [x + 0.10, y + s]],
                             closed=True, **k))
        return 0.13
    if name == "paper":
        ax.add_patch(Polygon([[x, y - s], [x + 0.11, y - s], [x + 0.11, y + 0.5 * s], [x + 0.07, y + s], [x, y + s]],
                             closed=True, **k))
        for i in range(3):
            ax.plot([x + 0.02, x + 0.09], [y + 0.03 - 0.035 * i] * 2, lw=0.4, **line)
        return 0.11
    if name == "code":
        ax.text(x, y, r"$\{\,\}$", fontsize=t.size + 0.5, ha="left", va="center", color=ink, zorder=5)
        return 0.13
    if name == "calc":
        ax.add_patch(FancyBboxPatch((x, y - 0.8 * s), 1.6 * s, 1.6 * s, boxstyle="round,pad=0,rounding_size=0.02",
                                    **k))
        ax.text(x + 0.8 * s, y, r"$\Sigma$", fontsize=t.size - 1.5, ha="center", va="center", color=ink, zorder=6)
        return 1.6 * s
    if name == "pc":
        ax.add_patch(Rectangle((x, y - 0.25 * s), 2 * s, 1.25 * s, **k))
        ax.add_patch(Rectangle((x + 0.2 * s, y - 0.05 * s), 1.6 * s, 0.85 * s, fc=t.light, ec="none", zorder=6))
        ax.plot([x + s, x + s], [y - 0.25 * s, y - 0.7 * s], lw=lw, **line)
        ax.plot([x + 0.45 * s, x + 1.55 * s], [y - 0.7 * s] * 2, lw=lw, **line)
        return 2 * s
    if name == "em":
        ax.add_patch(Rectangle((x, y - s), 2 * s, 2 * s, **k))
        for f in (1 / 3, 2 / 3):
            ax.plot([x + 2 * s * f] * 2, [y - s, y + s], lw=0.4, **line)
            ax.plot([x, x + 2 * s], [y - s + 2 * s * f] * 2, lw=0.4, **line)
        ax.plot([x, x + 2 * s], [y - s, y + s], lw=0.4, **line)
        return 2 * s
    if name == "probe":
        ax.add_patch(Polygon([(x + 0.4 * s, y + s), (x + 0.8 * s, y + s), (x + 0.6 * s, y - 0.35 * s)], closed=True,
                             fc=t.reference, ec=ink, lw=0.5, zorder=6))
        ax.add_patch(Rectangle((x, y - s), 1.2 * s, 0.3 * s, fc=t.light, ec=ink, lw=0.5, zorder=6))
        return 1.2 * s
    if name in ("vna", "adc"):
        w = 2.4 * s
        ax.add_patch(Rectangle((x, y - s), w, 2 * s, **k))
        u = np.linspace(0, 1, 40)
        xx = x + 0.02 + u * (w - 0.04)
        if name == "vna":
            ax.plot(xx, y + 0.035 - 0.08 / (1 + ((u - 0.5) / 0.08) ** 2), color=t.model, lw=0.6, zorder=6)
        else:
            ax.plot(xx, y + 0.035 * np.sin(4 * np.pi * u), color=t.accent, lw=0.6, zorder=6)
        return w
    if name == "fridge":
        for i in range(3):
            w = 2 * s * (1 - 0.18 * i)
            ax.plot([x + s - w / 2, x + s + w / 2], [y + s - 0.065 * i] * 2, lw=1.0, **line)
        ax.plot([x + s, x + s], [y + s, y - s], lw=0.5, **line)
        return 2 * s
    if name == "stats":
        for i, hh in enumerate((0.4, 0.9, 1.0, 0.55, 0.25)):
            ax.add_patch(Rectangle((x + 0.032 * i, y - s), 0.028, 2 * s * hh, fc=t.light, ec=ink, lw=0.4, zorder=5))
        return 0.16
    raise ValueError(f"unknown tool icon {name!r}; choose one of {TOOL_ICONS}")


def _tool_items(items):
    """Tools as (icon or None, label): an icon name, an (icon, label) pair, or any other string,
    which becomes a labelled chip."""
    out = []
    for it in items or ():
        if isinstance(it, str):
            out.append((it, _TOOL_LABELS[it]) if it in TOOL_ICONS else (None, it))
        elif isinstance(it, (tuple, list)) and len(it) == 2:
            if it[0] not in TOOL_ICONS:
                raise ValueError(f"unknown tool icon {it[0]!r}; choose one of {TOOL_ICONS}")
            out.append((it[0], str(it[1])))
        else:
            raise ValueError(f"tool {it!r}: give an icon name, an (icon, label) pair or a label")
    return out


def _tools_row(ctx, tools, x, y, right):
    """Draw the "You use:" row; return True when it fits before `right`."""
    t, fig, ax, fs = theme(), ctx["fig"], ctx["bg"], ctx["body"]
    if not tools:
        return True
    ux = x
    if ctx["tools_label"]:
        ax.text(x, y, ctx["tools_label"], fontsize=fs, ha="left", va="center", style="italic", color=t.ink, zorder=5)
        ux += _measure(fig, ctx["tools_label"], fs, style="italic")[0] + 0.08
    ch = 1.75 * fs / 72
    for name, label in tools:
        tw = _measure(fig, label, fs)[0]
        if name:
            ux += _icon(ax, name, ux, y) + 0.03
            if label:
                ax.text(ux, y, label, fontsize=fs, ha="left", va="center", color=t.ink, zorder=5)
            ux += tw + 0.09
        else:
            cw = tw + 0.12
            ax.add_patch(FancyBboxPatch((ux, y - ch / 2), cw, ch, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=t.tint, ec=t.reference, lw=0.5, zorder=4))
            ax.text(ux + cw / 2, y, label, fontsize=fs, ha="center", va="center", color=t.ink, zorder=5)
            ux += cw + 0.06
    return ux - 0.06 <= right


def _style_panel(pax, fs):
    for sp in pax.spines.values():
        sp.set_linewidth(0.6)
    pax.tick_params(labelsize=fs, length=2.2, pad=1.5, width=0.5)
    pax.tick_params(which="minor", length=1.2, width=0.4)
    for a in (pax.xaxis, pax.yaxis):
        a.label.set_size(fs)
        a.labelpad = 1.0


def _n_fit(top, bottom, pitch, line_h):
    return max(0, int(np.floor((top - bottom - line_h) / pitch + 1e-9)) + 1)


def _text_room(spec, w, mode, side, plot_w):
    """Width in inches left for the instruction lines of a box w wide, as _draw_box lays it out."""
    panel = spec.get("panel", None if side else True)
    if side or mode == "below" or panel is None or panel is False:
        return w - 2 * _PAD
    return w - 0.08 - plot_w - spec.get("panel_left", 0.34) - _PAD


def _shrink_body(fig, boxes, fs, floor):
    """The largest size up to fs (not below floor) at which every line of every box fits its room on
    one line; boxes holds (lines, room in inches) pairs. Text widths do not scale exactly with size,
    so the smaller size is measured again."""
    f = fs
    for _ in range(4):
        s = 1.0
        for lines, room in boxes:
            widest = max((_measure(fig, ln, f)[0] for ln in lines), default=0.0)
            if widest > room:
                s = min(s, max(room, 0.0) / widest)
        if s >= 1.0 or f <= floor:
            break
        f = max(floor, f * s * 0.995)
    return f


def _draw_box(ctx, spec, x, y, w, h, number, mode, side, name):
    """Frame, circled number, title, wrapped lines, mini axes and tools row of one box. Returns the
    mini axes, or None."""
    t, fig, ax = theme(), ctx["fig"], ctx["bg"]
    fs, body, pitch, line_h = ctx["fs"], ctx["body"], ctx["pitch"], ctx["line_h"]
    numbered = number is not None
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09",
                                fc=lighten(t.tint, 0.4) if side else "white", ec=t.reference if side else t.ink,
                                lw=0.8 if side else 0.9, ls=(0, (3, 2)) if side else "-", zorder=2))
    top = y + h
    tx = x + _PAD
    if numbered:
        ax.add_patch(Circle((x + 0.15, top - 0.15), 0.098, fc=t.ink, ec="none", zorder=4))
        ax.text(x + 0.15, top - 0.152, str(number), color="white", fontsize=t.size + 0.2, ha="center", va="center",
                weight="bold", zorder=5)
        tx = x + 0.29
    title = spec.get("title") or ""
    if title:
        tfs, room = t.size + 0.5, x + w - 0.06 - tx
        tw = _measure(fig, title, tfs, weight="bold")[0]
        if tw > room:
            tfs = max(fs, tfs * room / tw)
            if _measure(fig, title, tfs, weight="bold")[0] > room:
                ctx["overfull"].append(f"{name}: title wider than the box")
        ax.text(tx, top - _TITLE_Y[numbered], title, fontsize=tfs, weight="bold", ha="left", va="center",
                color=t.ink, zorder=5)
    panel = spec.get("panel", None if side else True)
    tools = spec["_tools"]
    lines_top = top - _LINES_TOP[numbered]
    bottom = y + (_TOOLS_Y + 0.11 if tools else 0.05)
    text_w = _text_room(spec, w, mode, side, ctx["plot_w"])
    raw = _as_lines(spec.get("lines"))
    geom = None
    if panel is not None and panel is not False:
        if side:      # a side box sizes its panel from its own text
            lines = _box_lines(fig, raw, text_w, body, ctx["wrap"])
            used = _LINES_TOP[numbered] + max(len(lines) - 1, 0) * pitch + line_h + 0.06
            left = spec.get("panel_left", 0.42)
            ph = h - used - _PANEL_BOTTOM["right"]
            if ph < 0.15:
                raise ValueError(f"{name}: box too short for a panel; shorten its text or raise the box height")
            geom = (x + left, y + _PANEL_BOTTOM["right"], w - left - 0.09, ph)
        elif mode == "below":
            left = spec.get("panel_left", 0.42)
            geom = (x + left, y + _PANEL_BOTTOM["below"], w - left - 0.09, ctx["plot_h"])
        else:
            pw = ctx["plot_w"]
            geom = (x + w - 0.08 - pw, y + _PANEL_BOTTOM["right"], pw, ctx["plot_h"])
        if mode == "below" or side:
            bottom = geom[1] + geom[3] + 0.05
    lines = _box_lines(fig, raw, text_w, body, ctx["wrap"])
    if len(lines) > _n_fit(lines_top, bottom, pitch, line_h):
        ctx["overfull"].append(f"{name}: {len(lines)} lines do not fit")
    if ctx["wrap"] == "shrink" and any(_measure(fig, ln, body)[0] > text_w + 1e-3 for ln, _ in lines):
        ctx["overfull"].append(f"{name}: a line is wider than the box at the smallest size")
    for k, (ln, dx) in enumerate(lines):
        ax.text(x + _PAD + dx, lines_top - k * pitch, ln, fontsize=body, ha="left", va="top", color=t.ink,
                zorder=5)
    pax = None
    if geom is not None:
        px, py, pw, ph = geom
        W, H = ctx["W"], ctx["H"]
        pax = fig.add_axes([px / W, py / H, pw / W, ph / H])
        _style_panel(pax, fs)
        if callable(panel):
            panel(pax)
    if not _tools_row(ctx, tools, x + _PAD, y + _TOOLS_Y, x + w - 0.05):
        ctx["overfull"].append(f"{name}: tools run past the box")
    return pax


def _cells(n, ncol, order):
    if order == "column":
        nrow = -(-n // ncol)
        ncols = -(-n // nrow)
    else:
        ncols = min(ncol, n)
        nrow = -(-n // ncols)
    cells = []
    for i in range(n):
        if order == "column":
            c, r = divmod(i, nrow)
        else:
            r, c = divmod(i, ncols)
            if order == "snake" and r % 2:
                c = ncols - 1 - c
        cells.append((r, c))
    return cells, nrow, ncols


def _blocked(cells, nrow, arrows, below_cols):
    """Per row gap: columns crossed by a vertical connector, and column boundaries crossed by one."""
    cols = {g: set() for g in range(nrow - 1)}
    bounds = {g: set() for g in range(nrow - 1)}
    if arrows:
        for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
            if c0 == c1 and r1 > r0:
                for g in range(r0, r1):
                    cols[g].add(c0)
            elif r1 > r0:                 # elbow: the final descent crosses the lower part of the gap
                cols[r1 - 1].add(c1)
            elif r1 < r0:                 # column wrap: up through the column gap
                for g in range(r1, r0):
                    bounds[g].add(min(c0, c1))
    for c in below_cols:                  # dashed connectors from the lowest box of a column
        r_low = max(r for r, cc in cells if cc == c)
        for g in range(r_low, nrow - 1):
            cols[g].add(c)
    return cols, bounds


def _strip_cols(c, ncols, cols, bounds):
    runs, cur = [], []
    for k in range(ncols):
        if k in cols:
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur and (k - 1) in bounds:
            runs.append(cur)
            cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    if not runs:
        raise ValueError("no free columns for a 'between' side box; pass cols=(first, last)")
    run = next((r for r in runs if c in r), max(runs, key=len))
    return run[0], run[-1]


def recipe_grid(fig, steps, *, ncol=4, order="row", box=(1.58, 2.32), gap=(0.22, 0.36), plot="below",
                plot_size=None, text_lines=3, side=(), side_width=None, arrows=True, arrow_lw=1.1, arrow_y=0.55,
                arrow_x=0.5, tools_label="You use:", margin=0.04, wrap="paragraph"):
    """A step-by-step figure: numbered rounded boxes, each with a bold title, instruction lines, a framed
    mini-axes and a row of the tools the step uses, joined by arrows; optional steps sit in dashed grey
    boxes. Everything is laid out in inches and the figure is resized to fit (fig=None makes one).

    steps: dicts with title, lines (a list or a string with newlines; lines that run past the box are
    rewrapped, mathtext spans kept whole), panel (missing or True: an empty mini-axes is made and
    returned; a callable is called with it; None or False: no axes), tools (TOOL_ICONS names, (icon,
    label) pairs, or plain strings drawn as chips), number (default i + 1, None for none) and
    panel_left (inches kept left of the axes for tick labels).
    order: "row" fills rows and wraps through the row gap, "snake" reverses every second row, "column"
    fills columns and climbs to the next column through the column gap. plot puts the mini-axes below
    the text or to its "right"; plot_size=(w, h) in inches overrides its size.
    side: optional-step boxes, dicts with after (1-based step), at, title, lines and tools. at="below"
    puts them in a strip under the grid joined to the step's column; "left" or "right" in an extra column
    beside the step's row (these may hold a panel); "between" in a dashed strip in the row gap next to
    the step's row, spanning the columns no connector crosses (or cols=(first, last)).
    wrap="paragraph" treats a box's lines as one paragraph: when one runs past the box, all are rewrapped
    together. wrap="lines" treats each line as an instruction of its own: only a line that runs past the
    box wraps, and its continuation is set in by a hanging indent (strips "between" rows stay prose).
    wrap="shrink" never wraps: the lines and tools of every box are set at one size, the largest (up to
    the default) at which each line fits its box, but not below theme().size - 2; a line still too wide
    is reported in overfull.

    Returns a dict: fig, panels (one Axes or None per step), side_panels, frames and side_frames
    ((x, y, w, h) in inches), background (the inch axes), overfull (boxes whose text did not fit) and
    body (the size of the boxes' lines in points)."""
    t = theme()
    steps = [dict(s) for s in steps]
    if not steps:
        raise ValueError("recipe_grid needs at least one step")
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, not {order!r}")
    if plot not in _PLOTS:
        raise ValueError(f"plot must be one of {_PLOTS}, not {plot!r}")
    if wrap not in _WRAPS:
        raise ValueError(f"wrap must be one of {_WRAPS}, not {wrap!r}")
    if int(ncol) != ncol or ncol < 1:
        raise ValueError(f"ncol must be a positive integer, not {ncol!r}")
    bw, bh = (float(v) for v in box)
    gx, gy = (float(v) for v in gap)
    if bw <= 0 or bh <= 0 or gx < 0 or gy < 0:
        raise ValueError("box sizes must be positive and gaps non-negative")
    for i, s in enumerate(steps):
        bad = set(s) - _STEP_KEYS
        if bad:
            raise ValueError(f"step {i + 1}: unknown keys {sorted(bad)}")
        s["_tools"] = _tool_items(s.get("tools"))
    n = len(steps)
    sides = []
    for s in side or ():
        d = dict(s)
        bad = set(d) - _SIDE_KEYS
        if bad:
            raise ValueError(f"side box: unknown keys {sorted(bad)}")
        after = d.get("after")
        if not isinstance(after, (int, np.integer)) or not 1 <= after <= n:
            raise ValueError(f"side box after={after!r}: give a step number from 1 to {n}")
        d["at"] = d.get("at", "below")
        if d["at"] not in _SIDES:
            raise ValueError(f"side box at={d['at']!r}: choose one of {_SIDES}")
        if d["at"] in ("below", "between") and d.get("panel") not in (None, False):
            raise ValueError("only 'left' and 'right' side boxes may hold a panel")
        d["_tools"] = _tool_items(d.get("tools"))
        sides.append(d)

    if fig is None:
        if not core._applied[0]:
            core.use(t.name)
        fig = plt.figure()
    elif fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    fs = t.size - 0.5
    pitch, line_h = 1.5 * fs / 72, 1.2 * fs / 72
    cells, nrow, ncols = _cells(n, int(ncol), order)
    sw = float(side_width) if side_width else 0.55 * bw
    plot_w = (plot_size[0] if plot_size else 0.38 * bw) if plot == "right" else None

    # the size of the boxes' lines and tools; under wrap="shrink" the largest at which every line of every
    # box fits on one line, one size for all boxes
    body = fs
    if wrap == "shrink":
        rooms = [(_as_lines(s.get("lines")), _text_room(s, bw, plot, False, plot_w)) for s in steps]
        rooms += [(_as_lines(d.get("lines")),
                   _text_room(d, sw if d["at"] in ("left", "right") else float(d.get("width", bw)), "below", True,
                              plot_w))
                  for d in sides if d["at"] != "between"]
        body = _shrink_body(fig, rooms, fs, t.size - 2.0)
    bpitch, bline_h = 1.5 * body / 72, 1.2 * body / 72

    # beside-the-grid columns: one box per row and side
    for at in ("left", "right"):
        rows = [cells[d["after"] - 1][0] for d in sides if d["at"] == at]
        if len(set(rows)) != len(rows):
            raise ValueError(f"two '{at}' side boxes share a row")
    left_w = sw + gx if any(d["at"] == "left" for d in sides) else 0.0
    right_w = sw + gx if any(d["at"] == "right" for d in sides) else 0.0

    # strips in a row gap: pick the columns, wrap the text, size the gap
    below_cols = [cells[d["after"] - 1][1] for d in sides if d["at"] == "below"]
    bcols, bbounds = _blocked(cells, nrow, arrows, below_cols)
    strip_h = {}
    for d in sides:
        if d["at"] != "between":
            continue
        if nrow < 2:
            raise ValueError("a 'between' side box needs at least two rows of steps")
        r, c = cells[d["after"] - 1]
        g = r if r < nrow - 1 else r - 1
        if g in strip_h:
            raise ValueError("two 'between' side boxes share a row gap")
        c0, c1 = d.get("cols") or _strip_cols(c, ncols, bcols[g], bbounds[g])
        if not 0 <= c0 <= c1 < ncols:
            raise ValueError(f"cols={d.get('cols')!r} is not a column range of the grid")
        width = (c1 - c0 + 1) * bw + (c1 - c0) * gx
        title = d.get("title") or ""
        tw = _measure(fig, title, fs, weight="bold")[0] + 0.05 if title else 0.0
        d["_wrapped"] = _wrap(fig, " ".join(_as_lines(d.get("lines"))), width - 2 * _PAD, fs, first=width - 2 * _PAD - tw)
        d["_geom"] = (g, c0, width, tw)
        strip_h[g] = 0.08 + (len(d["_wrapped"]) - 1) * pitch + line_h

    # strip of side boxes under the grid
    below = [d for d in sides if d["at"] == "below"]
    side_h, spans = 0.0, []
    for d in below:
        wd = float(d.get("width", bw))
        c = cells[d["after"] - 1][1]
        x = float(d.get("x", margin + left_w + c * (bw + gx) + (bw - wd) / 2))
        if any(x < b and a < x + wd for a, b in spans):
            raise ValueError(f"side boxes below the grid overlap (after={d['after']}); attach them to steps in "
                             "different columns or give x and width")
        spans.append((x, x + wd))
        d["_x"] = x
        d["_lines"] = _box_lines(fig, _as_lines(d.get("lines")), wd - 2 * _PAD, body, wrap)
        need = _LINES_TOP[False] + max(len(d["_lines"]) - 1, 0) * bpitch + bline_h + 0.07
        side_h = max(side_h, need + (_TOOLS_Y + 0.11 if d["_tools"] else 0.0))
    below_h = side_h + 0.8 * gy if below else 0.0

    gaps = [gy + strip_h.get(g, 0.0) for g in range(nrow - 1)]
    W = 2 * margin + left_w + ncols * bw + (ncols - 1) * gx + right_w
    H = 2 * margin + nrow * bh + sum(gaps) + below_h
    xs = [margin + left_w + c * (bw + gx) for c in range(ncols)]
    ys, top = [], H - margin
    for r in range(nrow):
        ys.append(top - bh)
        top = ys[-1] - (gaps[r] if r < nrow - 1 else 0.0)

    if plot == "below":
        used = _LINES_TOP[True] + (text_lines - 1) * bpitch + bline_h + 0.06
        plot_h = plot_size[1] if plot_size else bh - used - _PANEL_BOTTOM["below"]
    else:
        plot_h = plot_size[1] if plot_size else bh - 0.30 - _PANEL_BOTTOM["right"]
    if any(s.get("panel", True) not in (None, False) for s in steps) and plot_h < 0.2:
        raise ValueError(f"box height {bh} leaves {plot_h:.2f} in for the panels; raise the box height or lower "
                         "text_lines")

    fig.set_size_inches(W, H)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    ctx = dict(fig=fig, bg=bg, W=W, H=H, fs=fs, body=body, pitch=bpitch, line_h=bline_h, plot_w=plot_w,
               plot_h=plot_h, tools_label=tools_label, overfull=[], wrap=wrap)

    panels, frames = [], []
    for i, (st, (r, c)) in enumerate(zip(steps, cells)):
        x, y = xs[c], ys[r]
        frames.append((x, y, bw, bh))
        panels.append(_draw_box(ctx, st, x, y, bw, bh, st.get("number", i + 1), plot, False, f"step {i + 1}"))

    ink, ms = t.ink, max(6.0, 10.0 * arrow_lw / 1.1)
    head = "-|>,head_length=0.5,head_width=0.28"
    if arrows:
        for i in range(n - 1):
            (r0, c0), (r1, c1) = cells[i], cells[i + 1]
            x0, y0, x1, y1 = xs[c0], ys[r0], xs[c1], ys[r1]
            if r0 == r1:
                ym = y0 + arrow_y * bh
                pts = [(x0 + bw + 0.02, ym), (x1 - 0.015, ym)] if c1 > c0 else [(x0 - 0.02, ym), (x1 + bw + 0.015, ym)]
            elif c0 == c1:
                xa = x0 + arrow_x * bw
                pts = [(xa, y0 - 0.01), (xa, y1 + bh + 0.01)] if r1 > r0 else [(xa, y0 + bh + 0.01), (xa, y1 - 0.01)]
            elif r1 > r0:
                yg = y0 - (gy / 4 if r0 in strip_h else gaps[r0] / 2)
                xa, xb = x0 + arrow_x * bw, x1 + arrow_x * bw
                pts = [(xa, y0 - 0.01), (xa, yg), (xb, yg), (xb, y1 + bh + 0.01)]
            else:
                ya, yb = y0 + arrow_y * bh, y1 + arrow_y * bh
                xm = x0 + bw + gx / 2 if c1 > c0 else x0 - gx / 2
                ea, eb = (x0 + bw + 0.01, x1 - 0.01) if c1 > c0 else (x0 - 0.01, x1 + bw + 0.01)
                pts = [(ea, ya), (xm, ya), (xm, yb), (eb, yb)]
            if len(pts) > 2:
                bg.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=ink, lw=arrow_lw,
                        solid_capstyle="projecting", solid_joinstyle="miter", zorder=6)
            arrow(bg, pts[-2], pts[-1], color=ink, lw=arrow_lw, style=head, ms=ms, zorder=6)

    side_panels, side_frames = [], []
    dashed = dict(color=t.reference, lw=0.9, style="-|>,head_length=0.5,head_width=0.3", ms=8, ls=(0, (3, 2)),
                  zorder=1)
    for d in sides:
        r, c = cells[d["after"] - 1]
        label = f"side box after step {d['after']}"
        if d["at"] == "below":
            wd, x = float(d.get("width", bw)), d["_x"]
            side_panels.append(_draw_box(ctx, d, x, margin, wd, side_h, None, "below", True, label))
            side_frames.append((x, margin, wd, side_h))
            r_low = max(rr for rr, cc in cells if cc == c)
            sx = x + wd / 2
            arrow(bg, (sx, ys[r_low] - 0.01), (sx, margin + side_h + 0.01), **dashed)
        elif d["at"] in ("left", "right"):
            x = margin if d["at"] == "left" else W - margin - sw
            side_panels.append(_draw_box(ctx, d, x, ys[r], sw, bh, None, "below", True, label))
            side_frames.append((x, ys[r], sw, bh))
            ym = ys[r] + bh / 2
            if d["at"] == "left":
                arrow(bg, (xs[0] - 0.01, ym), (x + sw + 0.01, ym), **dashed)
            else:
                arrow(bg, (xs[-1] + bw + 0.01, ym), (x - 0.01, ym), **dashed)
        else:
            g, c0, width, tw = d["_geom"]
            sh = strip_h[g]
            x, y = xs[c0], ys[g + 1] + bh + gy / 2
            bg.add_patch(FancyBboxPatch((x, y), width, sh, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=lighten(t.tint, 0.4), ec=t.reference, lw=0.8, ls=(0, (3, 2)), zorder=2))
            ty = y + sh - 0.04
            if d.get("title"):
                bg.text(x + _PAD, ty, d["title"], fontsize=fs, weight="bold", ha="left", va="top", color=t.ink,
                        zorder=5)
            for k, ln in enumerate(d["_wrapped"]):
                bg.text(x + _PAD + (tw if k == 0 else 0.0), ty - k * pitch, ln, fontsize=fs, ha="left", va="top",
                        color=t.ink, zorder=5)
            side_panels.append(None)
            side_frames.append((x, y, width, sh))
    return dict(fig=fig, panels=panels, side_panels=side_panels, frames=frames, side_frames=side_frames,
                background=bg, overfull=ctx["overfull"], size=(W, H), body=body)


# ---- equations, matrices and Hamiltonian terms ------------------------------------------------

def text_panel(ax, lines, *, fs=None, align="center", valign="spread", frame=True, colors=None, min_fs=None):
    """Display equations or short statements, one per line. lines holds strings or (string, colour)
    pairs; colors gives one colour per line instead. valign="spread" spreads the lines evenly down ax
    (an equation box); "top" and "center" set them at a fixed line pitch from the top or around the
    middle (a list). All lines shrink together until the widest fits (down to min_fs); a line still too
    wide is rewrapped at word breaks outside mathtext. frame=True keeps a thin full frame, frame=False
    draws text only. Returns the Text artists."""
    t = theme()
    if align not in ("center", "left", "right"):
        raise ValueError(f"align must be 'center', 'left' or 'right', not {align!r}")
    if valign not in ("spread", "top", "center"):
        raise ValueError(f"valign must be 'spread', 'top' or 'center', not {valign!r}")
    items = []
    for ln in lines or ():
        if isinstance(ln, str):
            items += [(s, None) for s in ln.split("\n")]
        elif isinstance(ln, (tuple, list)) and len(ln) == 2:
            items += [(s, ln[1]) for s in str(ln[0]).split("\n")]
        else:
            raise ValueError(f"line {ln!r}: give a string or a (string, colour) pair")
    if colors is not None:
        if len(colors) != len(items):
            raise ValueError(f"{len(colors)} colours for {len(items)} lines")
        items = [(s, c) for (s, _), c in zip(items, colors)]
    fs = fs or t.size + 0.5
    min_fs = min_fs or t.size - 1.5
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(which="both", length=0)
    if frame:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_color(t.ink)
    else:
        ax.axis("off")
    if not items:
        return []
    fig = ax.figure
    bb = _settle(ax)
    room_w = (0.92 if frame else 1.0) * bb.width / fig.dpi
    room_h = 0.92 * bb.height / fig.dpi

    def size(s, f):
        w, a, d = _measure(fig, s, f)
        return w, a + d

    def tall(rows, f):
        if valign != "spread":
            return len(rows) * 1.9 * f / 72
        return sum(max(size(s, f)[1], 1.2 * f / 72) for s, _ in rows)

    widest = max(size(s, fs)[0] for s, _ in items) or 1.0
    f = max(min_fs, fs * min(1.0, room_w / widest, room_h / tall(items, fs)))
    out = []
    for s, c in items:
        out += [(p, c) for p in (_wrap(fig, s, room_w, f) if size(s, f)[0] > room_w else [s])]
    f = max(min_fs, f * min(1.0, room_h / tall(out, f)))
    x, ha = {"center": (0.5, "center"), "left": (0.04 if frame else 0.0, "left"),
             "right": (0.96 if frame else 1.0, "right")}[align]
    m = len(out)
    if valign == "spread":
        ys = [1 - (k + 0.5) / m for k in range(m)]
    else:
        p = 1.9 * f / 72 / (bb.height / fig.dpi)
        y0 = (0.96 if frame else 1.0) - p / 2 if valign == "top" else 0.5 + (m - 1) * p / 2
        ys = [y0 - k * p for k in range(m)]
    return [ax.text(x, yk, s, fontsize=f, ha=ha, va="center", color=_color(c), transform=ax.transAxes)
            for yk, (s, c) in zip(ys, out)]


def _cell(v, fmt):
    if isinstance(v, str):
        return v
    return fmt(v) if callable(fmt) else fmt.format(v)


def matrix_text(ax, M, rows=None, cols=None, *, fmt="{:g}", brackets=True, label_colors=None, title=None,
                table=None, notes=None, fs=None, x=0.0, y=1.0, min_fs=None, beside=None):
    """A matrix typeset as text in square brackets, with row labels on the left and column labels on top
    in label_colors (one theme role or colour per row; the columns take them too when they match in
    number). Entries may be numbers (set with fmt, a format string or a callable) or strings (mathtext
    allowed). title is one line or a list of lines set above the matrix; beside, one line or a list,
    sits right of the closing bracket, centred on the rows (the definition of a symbol in the matrix,
    say). Optional pieces stack below: table=(header, rows), a plain table whose first cell in each row
    takes the row colour, and notes, a list of lines. The block starts at (x, y) in axes fraction (its
    top-left corner) and shrinks to fit the axes. Returns a dict of the drawn artists (title: the first
    title line, titles: all of them) and the font size used."""
    t = theme()
    A = np.asarray(M, dtype=object)
    if A.ndim == 1:
        A = A[None, :]
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"matrix_text needs a non-empty 2-D matrix, not shape {np.shape(M)}")
    n, m = A.shape
    if rows is not None and len(rows) != n:
        raise ValueError(f"{len(rows)} row labels for {n} rows")
    if cols is not None and len(cols) != m:
        raise ValueError(f"{len(cols)} column labels for {m} columns")
    if label_colors is not None and len(label_colors) != n:
        raise ValueError(f"{len(label_colors)} label colours for {n} rows")
    head, body = (list(table[0]), [list(r) for r in table[1]]) if table is not None else (None, [])
    if head is not None and any(len(r) != len(head) for r in body):
        raise ValueError("every table row needs as many cells as the header")
    cells = [[_cell(v, fmt) for v in row] for row in A]
    body = [[_cell(v, fmt) for v in r] for r in body]
    notes = _as_lines(notes)
    titles = _as_lines(title)
    aside = _as_lines(beside)
    fig = ax.figure
    aw, ah = _inch_axes(ax)
    x0, ytop = x * aw, y * ah
    fs = fs or t.size - 0.5
    min_fs = min_fs or t.size - 2.0
    rc = [_color(c) for c in label_colors] if label_colors is not None else [t.ink] * n
    cc = rc if (cols is not None and len(rc) == m) else [t.ink] * m

    def plan(f):
        wid = lambda s, **kw: _measure(fig, s, f, **kw)[0]
        pitch, pad = 1.75 * f / 72, 0.12
        colw = [max(wid(cells[i][j]) for i in range(n)) + pad for j in range(m)]
        if cols is not None:
            colw = [max(w, wid(cols[j]) + pad) for j, w in enumerate(colw)]
        labw = max(wid(r) for r in rows) + 0.07 if rows is not None else 0.0
        P = dict(f=f, pitch=pitch, title=[])
        yc, right = ytop, x0
        for k, s_ in enumerate(titles):          # the last title line keeps a little more room below it
            P["title"].append(yc)
            yc -= (1.25 if k == len(titles) - 1 else 1.0) * pitch
            right = max(right, x0 + wid(s_))
        if cols is not None:
            P["cols"] = yc - pitch / 2
            yc -= pitch
        xl = x0 + labw
        inner = xl + 0.06
        P["xl"], P["xc"] = xl, [inner + sum(colw[:j]) + colw[j] / 2 for j in range(m)]
        P["xr"] = inner + sum(colw) + 0.06
        P["rows"] = [yc - pitch * (i + 0.5) for i in range(n)]
        P["mtop"], P["mbot"] = yc, yc - pitch * n
        yc -= pitch * (n + 0.4)
        right = max(right, P["xr"])
        if aside:
            P["ax"] = P["xr"] + 0.1
            mid = (P["mtop"] + P["mbot"]) / 2
            P["ay"] = [mid + pitch * ((len(aside) - 1) / 2 - k) for k in range(len(aside))]
            right = max(right, P["ax"] + max(wid(s_) for s_ in aside))
        if head is not None:
            tw = [max([wid(head[j])] + [wid(r[j]) for r in body]) + 0.16 for j in range(len(head))]
            P["tx"] = [x0 + sum(tw[:j]) + (0 if j == 0 else tw[j] / 2) for j in range(len(head))]
            P["thead"], P["trule"] = yc - pitch / 2, yc - pitch - 0.01
            P["trows"] = [yc - pitch * (k + 1.5) - 0.03 for k in range(len(body))]
            P["tright"] = x0 + sum(tw)
            yc -= pitch * (len(body) + 1) + 0.06 + 0.2 * pitch
            right = max(right, P["tright"])
        P["notes"] = [yc - pitch * (k + 0.5) for k in range(len(notes))]
        yc -= pitch * len(notes)
        right = max([right] + [x0 + wid(s) for s in notes])
        return P, right - x0, ytop - yc

    P, need_w, need_h = plan(fs)
    for _ in range(3):           # text widths do not scale exactly with size, so check the smaller set again
        s = min(1.0, (aw - x0) / need_w if need_w > 0 else 1.0, ytop / need_h if need_h > 0 else 1.0)
        if s >= 1.0 or P["f"] <= min_fs:
            break
        P, need_w, need_h = plan(max(min_fs, P["f"] * s * 0.98))
    f, ink = P["f"], t.ink
    out = dict(cells=[], row_labels=[], col_labels=[], brackets=[], table=[], notes=[], title=None, titles=[],
               beside=[], fs=f)
    out["titles"] = [ax.text(x0, yt, s_, fontsize=f, ha="left", va="top", color=ink)
                     for yt, s_ in zip(P["title"], titles)]
    out["title"] = out["titles"][0] if out["titles"] else None
    out["beside"] = [ax.text(P["ax"], ya, s_, fontsize=f, ha="left", va="center", color=ink)
                     for ya, s_ in zip(P.get("ay", []), aside)]
    if cols is not None:
        out["col_labels"] = [ax.text(P["xc"][j], P["cols"], cols[j], fontsize=f, ha="center", va="center", color=cc[j])
                             for j in range(m)]
    for i in range(n):
        out["cells"].append([ax.text(P["xc"][j], P["rows"][i], cells[i][j], fontsize=f, ha="center", va="center",
                                     color=ink) for j in range(m)])
        if rows is not None:
            out["row_labels"].append(ax.text(P["xl"] - 0.03, P["rows"][i], rows[i], fontsize=f, ha="right",
                                             va="center", color=rc[i]))
    if brackets:
        yt, yb, arm = P["mtop"] - 0.05 * P["pitch"], P["mbot"] + 0.05 * P["pitch"], 0.06
        for xx, sgn in ((P["xl"], 1), (P["xr"], -1)):
            ln, = ax.plot([xx + sgn * arm, xx, xx, xx + sgn * arm], [yt, yt, yb, yb], color=ink, lw=0.8,
                          solid_joinstyle="miter")
            out["brackets"].append(ln)
    if head is not None:
        for j, h in enumerate(head):
            out["table"].append(ax.text(P["tx"][j], P["thead"], h, fontsize=f, ha="left" if j == 0 else "center",
                                        va="center", color=ink))
        out["table"].append(ax.plot([x0, P["tright"]], [P["trule"]] * 2, color=ink, lw=0.6)[0])
        for k, r in enumerate(body):
            for j, v in enumerate(r):
                col = rc[k] if (j == 0 and label_colors is not None and len(body) == n) else ink
                out["table"].append(ax.text(P["tx"][j], P["trows"][k], v, fontsize=f, ha="left" if j == 0 else "center",
                                            va="center", color=col))
    for k, s_ in enumerate(notes):
        out["notes"].append(ax.text(x0, P["notes"][k], s_, fontsize=f, ha="left", va="center", color=ink))
    return out


_TERM_KEYS = {"text", "kind", "color", "brace", "gap"}
_TP_X, _TP_Y, _SP = 0.035, 0.035, 0.05             # tile padding and spacing between items, inches
_BR_GAP, _BR_ARM, _BR_STEM, _LINE_GAP = 0.05, 0.05, 0.05, 0.10


def _terms(lines):
    t = theme()
    out = []
    for li, items in enumerate(lines):
        if isinstance(items, (str, dict)):
            raise ValueError(f"line {li + 1}: give a list of items, not a single {type(items).__name__}")
        row = []
        for it in items:
            if isinstance(it, str):
                it = dict(text=it)
            elif not isinstance(it, dict):
                raise ValueError(f"term {it!r}: give a string or a dict")
            bad = set(it) - _TERM_KEYS
            if bad:
                raise ValueError(f"term {it.get('text')!r}: unknown keys {sorted(bad)}")
            if "text" not in it:
                raise ValueError(f"term {it!r} has no text")
            kind = it.get("kind")
            if kind is not None and kind not in t.tiles and not is_color_like(kind):
                raise ValueError(f"tile kind {kind!r}: choose one of {sorted(t.tiles)} or give a colour")
            row.append(dict(it))
        out.append(row)
    return out


def term_tiles(ax, lines, *, fs=None, x=0.0, y=1.0, indent=0.3, braces=True, min_fs=None):
    """A Hamiltonian typeset term by term, each term on the pale tile of the circuit element it comes
    from (theme().tiles: "nonlinear", "capacitive", "inductive", "coupling"), with brackets underneath
    that name where the terms come from.

    lines: one list per line. An item is a string (glue such as "$H/h=$" or "$+$", set in ink without a
    tile) or a dict with text, kind (tile kind, any colour, or None for no tile), color (theme role or
    colour of the text), brace (label set under a bracket) and gap (extra space before the item, in
    inches). Consecutive items with the same brace label share one bracket; glue between them joins it.
    The block starts at (x, y) in axes fraction and every line after the first is indented by `indent`
    inches. The block shrinks to fit the axes; a line still too wide at min_fs breaks between terms
    instead, a "+" before the break moving down with its term. Returns a dict of texts, tiles, braces,
    brace labels and the font size used."""
    t = theme()
    rows = _terms(lines)
    fig = ax.figure
    aw, ah = _inch_axes(ax)
    x0, ytop = x * aw, y * ah
    fs = fs or t.size + 0.6
    min_fs = min_fs or t.size - 1.5

    def split(f):
        out = []
        for items in rows:
            spans = [_measure(fig, it["text"], f)[0] + (2 * _TP_X if it.get("kind") is not None else 0.0)
                     + float(it.get("gap", 0.0)) for it in items]
            cur = []
            for it, s in zip(items, spans):
                room = aw - x0 - (indent if out else 0.0)
                if cur and sum(c[1] for c in cur) + s + _SP * len(cur) > room:
                    carry = [cur.pop()] if len(cur) > 1 and cur[-1][0].get("kind") is None else []
                    out.append([c[0] for c in cur])
                    cur = carry
                cur.append((it, s))
            out.append([c[0] for c in cur])
        return out

    def plan(f, wrap=False):
        planned, yc, right = [], ytop, x0
        lf = f - 0.6
        for li, items in enumerate(split(f) if wrap else rows):
            xc = x0 + (indent if li else 0.0)
            segs, a_t, d_t, asc, desc = [], 0.0, 0.0, 0.0, 0.0
            for it in items:
                w, a, d = _measure(fig, it["text"], f)
                tiled = it.get("kind") is not None
                if segs:
                    xc += _SP
                xc += float(it.get("gap", 0.0))
                tx = xc + (_TP_X if tiled else 0.0)
                segs.append(dict(it, x=tx, w=w, tiled=tiled))
                xc = tx + w + (_TP_X if tiled else 0.0)
                asc, desc = max(asc, a), max(desc, d)
                if tiled:
                    a_t, d_t = max(a_t, a), max(d_t, d)
            if any(s["tiled"] for s in segs):
                asc, desc = max(asc, a_t + _TP_Y), max(desc, d_t + _TP_Y)
            base = yc - asc
            groups, cur = [], None
            for s in segs if braces else ():
                lab = s.get("brace")
                if lab is None:
                    continue
                if cur is not None and cur["label"] == lab:
                    cur["last"] = s
                else:
                    cur = dict(label=lab, first=s, last=s)
                    groups.append(cur)
            low = base - desc
            if groups:
                yb = low - _BR_GAP
                lab_h = max(sum(_measure(fig, g["label"], lf)[1:]) for g in groups)
                levels = []                    # labels that would collide drop to a lower row
                for g in groups:
                    w = _measure(fig, g["label"], lf)[0]
                    g["x0"] = g["first"]["x"] - (_TP_X if g["first"]["tiled"] else 0.0)
                    g["x1"] = g["last"]["x"] + g["last"]["w"] + (_TP_X if g["last"]["tiled"] else 0.0)
                    g["xm"], g["yb"], g["lw"] = (g["x0"] + g["x1"]) / 2, yb, w
                    span = (g["xm"] - w / 2 - 0.06, g["xm"] + w / 2 + 0.06)
                    # first row where the label is free and its stem crosses no label above it
                    lev = next((i for i, used in enumerate(levels)
                                if all(span[1] <= a or span[0] >= b for a, b in used)
                                and not any(a < g["xm"] < b for row in levels[:i] for a, b in row)), len(levels))
                    if lev == len(levels):
                        levels.append([])
                    levels[lev].append(span)
                    g["stem"] = _BR_STEM + lev * (lab_h + 0.03)
                    right = max(right, g["xm"] + w / 2)
                low = yb - _BR_STEM - 0.02 - len(levels) * lab_h - (len(levels) - 1) * 0.03
            planned.append(dict(segs=segs, base=base, a_t=a_t, d_t=d_t, groups=groups))
            right = max(right, xc)
            yc = low - _LINE_GAP
        left = min([x0] + [g["xm"] - g["lw"] / 2 for p in planned for g in p["groups"]])
        return planned, f, right - min(left, x0), ytop - (yc + _LINE_GAP), min(left, x0)

    def scale(need_w, need_h):
        return min(1.0, aw / need_w if need_w > 0 else 1.0, ytop / need_h if need_h > 0 else 1.0)

    planned, f, need_w, need_h, left = plan(fs)
    s = scale(need_w, need_h)
    if s < 1.0 and fs * s * 0.98 >= min_fs:
        planned, f, need_w, need_h, left = plan(fs * s * 0.98)
    elif s < 1.0:                    # too wide even at min_fs: break lines between terms
        planned, f, need_w, need_h, left = plan(fs, wrap=True)
        s = scale(need_w, need_h)
        if s < 1.0:
            planned, f, need_w, need_h, left = plan(max(min_fs, fs * s * 0.98), wrap=True)
    shift = max(0.0, -left)          # a centred label must not leave the axes on the left
    out = dict(texts=[], tiles=[], braces=[], labels=[], fs=f)
    for p in planned:
        base = p["base"]
        for sg in p["segs"]:
            out["texts"].append(ax.text(sg["x"] + shift, base, sg["text"], fontsize=f, ha="left", va="baseline",
                                        color=_color(sg.get("color")), zorder=3))
            if sg["tiled"]:
                h = p["a_t"] + p["d_t"] + 2 * _TP_Y
                tile(ax, sg["x"] + shift + sg["w"] / 2, base + (p["a_t"] - p["d_t"]) / 2, sg["w"] + 2 * _TP_X, h,
                     sg["kind"], z=1)
                out["tiles"].append(ax.patches[-1])
        for g in p["groups"]:
            x0_, x1_, xm, yb = g["x0"] + shift, g["x1"] + shift, g["xm"] + shift, g["yb"]
            out["braces"].append(ax.plot([x0_, x0_, x1_, x1_], [yb + _BR_ARM, yb, yb, yb + _BR_ARM], color=t.ink,
                                         lw=0.7, solid_joinstyle="miter")[0])
            out["braces"].append(ax.plot([xm, xm], [yb, yb - g["stem"]], color=t.ink, lw=0.7)[0])
            out["labels"].append(ax.text(xm, yb - g["stem"] - 0.02, g["label"], fontsize=f - 0.6, ha="center",
                                         va="top", color=t.ink))
    return out
