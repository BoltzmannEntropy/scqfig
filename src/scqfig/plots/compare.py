"""Comparison and budget charts: stacked, grouped and signed-factor bars, sensitivity tornados,
labelled bar and dot rows, predicted-over-measured strips, paired before-and-after stems and event
timelines. Rows are categories (studies, devices, loss channels), so every function labels them.

Shared conventions. Row charts list their rows from the top. A status names how a value was obtained
and picks its marker by the grammar of core: "measured" filled (open in folio), "simulated" open,
"assumed" open grey, "fitted" small grey, "excluded" a grey cross for a value drawn but not counted,
"upper" or "lower" an open marker with an arrow toward the side where the true value lies, and
"inferred" a light-filled marker for a value derived indirectly from measurements.
columns adds text columns right of the frame: {header: one string per row}, or a list of
(header, strings) pairs; "" leaves a column without a header. groups gives one label per row, and
each run of equal consecutive labels forms a group."""
from collections.abc import Mapping

import matplotlib as mpl
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator
from matplotlib.transforms import Bbox

from ..core import MINUS, assumed_kw, data_kw, log_minor, note, ref_kw, sim_kw, theme
from ..themes import lighten

__all__ = ["stacked_bars", "factor_bars", "grouped_bars", "tornado", "bar_rows", "dot_rows", "study_strips",
           "ratio_strip", "paired_stems", "timeline", "limit_arrow"]

STATUSES = ("measured", "simulated", "assumed", "fitted", "upper", "lower", "excluded", "inferred")
_WORDS = {"measured": "measured", "simulated": "simulated", "assumed": "assumed", "fitted": "fitted",
          "upper": "upper bound", "lower": "lower bound", "excluded": "not counted", "inferred": "inferred"}
_HATCH = ("", "////", "....", "xxxx", "\\\\\\\\", "++")      # folio tells segments apart in greyscale
_DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


# ---- shared helpers ---------------------------------------------------------------------------

def _need(n, what="rows", **seqs):
    """Raise ValueError unless every sequence given has n entries."""
    for name, s in seqs.items():
        if s is not None and not isinstance(s, str) and len(s) != n:
            raise ValueError(f"{name} has {len(s)} entries for {n} {what}")


def _check_status(s):
    if s not in STATUSES:
        raise ValueError(f"unknown status {s!r}; choose one of {', '.join(STATUSES)}")
    return s


def _row_status(status, n):
    """One status per row from None (all measured), one string, or a sequence."""
    if status is None or isinstance(status, str):
        return [_check_status(status or "measured")] * n
    _need(n, status=status)
    return [_check_status(s) for s in status]


def _num(fmt, v):
    """v formatted with fmt, with a typographic minus outside mathtext."""
    s = fmt.format(v)
    return s if "$" in s else s.replace("-", MINUS)


def _fill(k, hatches=None):
    """Bar keywords of segment or series k: a tint of the k-th series colour with an ink edge. Folio
    hatches every segment after the first; hatches (one per segment) replaces that choice."""
    t = theme()
    h = hatches[k] if hatches is not None else (_HATCH[k % len(_HATCH)] if t.name == "folio" else "")
    return dict(color=lighten(t.cycle[k % len(t.cycle)], 0.45), edgecolor=t.ink, hatch=h or None, lw=0.5)


def _role_color(c):
    """A theme role that names one colour ("light", "sim", "accent", ...) or a plain colour."""
    v = getattr(theme(), c, None) if isinstance(c, str) else None
    if isinstance(v, str) and mpl.colors.is_color_like(v):
        return v
    if not mpl.colors.is_color_like(c):
        raise ValueError(f"colour {c!r}: give a theme role such as 'light' or 'sim', or a colour")
    return c


def _patch(kw, label):
    return Patch(facecolor=kw["color"], edgecolor=kw["edgecolor"], hatch=kw.get("hatch"), lw=kw.get("lw", 0.5),
                 label=label)


def _hide_empty(bars, lengths):
    """Hide bars of zero or undefined length, whose edges would otherwise draw a stray line."""
    for r, v in zip(bars, lengths):
        if not np.isfinite(v) or v == 0:
            r.set_visible(False)


def _marker(status, color=None, ms=None):
    """Marker keywords of a point of the given status, by the marker grammar of core."""
    t = theme()
    _check_status(status)
    if status == "measured":
        return data_kw(color, ms=ms)
    if status == "inferred":
        return data_kw(color, ms=ms, mfc=t.light)
    if status in ("simulated", "upper", "lower"):
        return sim_kw(color, ms=ms)
    if status == "assumed":
        return assumed_kw(ms=ms)
    size = ms or mpl.rcParams["lines.markersize"]
    if status == "fitted":
        return data_kw(t.reference, ms=0.6 * size, mfc=t.reference, zorder=3)
    return dict(marker="x", ls="none", color=t.reference, mec=t.reference, ms=size, mew=0.8, zorder=3)


def _status_handles(present, names=None, color=None, horizontal=False):
    """Legend handles for the statuses present, in the order of STATUSES; names renames them. A bound
    shows as an arrow toward the side where the true value lies, as limit_arrow draws it."""
    arrows = {"upper": r"$\leftarrow$" if horizontal else r"$\downarrow$",
              "lower": r"$\rightarrow$" if horizontal else r"$\uparrow$"}
    out = []
    for s in STATUSES:
        if s in present:
            kw = _marker(s, color)
            if s in arrows:
                kw.update(marker=arrows[s], mfc=kw["mec"], mew=0.4, ms=7)
            out.append(Line2D([], [], label=(names or {}).get(s, _WORDS[s]), **kw))
    return out


def _shape_handles(present, names=None, horizontal=False):
    """Legend handles for the (status, shape) pairs present, in the order of STATUSES, the status's own
    marker first and then the other shapes in order of first appearance. names maps a status or a
    (status, shape) pair to its label; a label of None leaves the entry out, and entries that share a
    label show once. A pair without a shape shows as _status_handles draws it; a bound drawn with a
    shape of its own shows that shape."""
    names = names or {}
    out, seen = [], set()
    for s in STATUSES:
        for st, shape in sorted(present, key=lambda p: p[1] is not None):     # stable: keeps first appearance
            if st != s:
                continue
            label = names[(st, shape)] if (st, shape) in names else names.get(st, _WORDS[st])
            if label is None or label in seen:
                continue
            seen.add(label)
            if shape is None:
                out += _status_handles({st}, {st: label}, horizontal=horizontal)
            else:
                out.append(Line2D([], [], label=label, **{**_marker(st), "marker": shape}))
    return out


def _legend(ax, handles, where):
    """Legend of handles: above the frame, at most three to a line, for where=True; inside at loc
    `where` for a string; a mapping passes its keywords to ax.legend (loc, bbox_to_anchor, ncol,
    frameon, ...), for a key placed beside or below the frame."""
    if not where or not handles:
        return None
    if isinstance(where, Mapping):
        return ax.legend(handles=handles, **{"loc": "best", **where})
    if where is True:
        lines = -(-len(handles) // 3)
        return ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=-(-len(handles) // lines),
                         frameon=False, handlelength=1.3, columnspacing=1.2, borderaxespad=0.2)
    return ax.legend(handles=handles, loc=where)


def _layout(n, groups, gap, lead):
    """Row positions 0, 1, 2, ... with `gap` extra before each new group, shifted by `lead` (a header
    row above the first group), and the groups as (label, first position, last position)."""
    if groups is None:
        return np.arange(n, dtype=float), []
    _need(n, groups=groups)
    pos, spans, p = np.empty(n), [], float(lead)
    for i, g in enumerate(groups):
        if i == 0 or g != groups[i - 1]:
            p += gap if i else 0.0
            spans.append([g, p, p])
        pos[i] = spans[-1][2] = p
        p += 1.0
    return pos, [tuple(s) for s in spans]


def _category_axis(ax, pos, labels, horizontal, lead=0.0):
    """Row labels as tick labels without tick marks, and limits half a row beyond the ends; horizontal
    rows run from the top."""
    pos = np.asarray(pos, float)
    lo = pos.min() - lead if pos.size else 0.0
    hi = pos.max() if pos.size else 0.0
    if horizontal:
        ax.set_yticks(pos, labels)
        ax.set_ylim(hi + 0.6, lo - 0.6)
        ax.tick_params(axis="y", which="both", length=0)
    else:
        ax.set_xticks(pos, labels)
        ax.set_xlim(lo - 0.6, hi + 0.6)
        ax.tick_params(axis="x", which="both", length=0)


def _headers(ax, spans, horizontal):
    """Group labels: an italic header in the row above each group of horizontal rows, or a label above
    the frame over each group of vertical bars with thin rules between the groups."""
    t = theme()
    out = []
    pad = mpl.rcParams["ytick.major.pad"]
    for g, a, b in spans:
        if horizontal:
            out.append(ax.annotate(str(g), xy=(0, a - 1), xycoords=("axes fraction", "data"), xytext=(-pad, 0),
                                   textcoords="offset points", ha="right", va="center", fontsize=t.size,
                                   style="italic", color=t.ink, annotation_clip=False))
        else:
            out.append(ax.annotate(str(g), xy=((a + b) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 3),
                                   textcoords="offset points", ha="center", va="bottom", fontsize=t.size - 0.5,
                                   color=t.ink, annotation_clip=False))
    if not horizontal:
        for (_, _, b), (_, a, _) in zip(spans[:-1], spans[1:]):
            ax.axvline((a + b) / 2, color=t.light, lw=0.6, zorder=0)
    return out


def _pairs(columns):
    if not columns:
        return []
    return list(columns.items()) if isinstance(columns, Mapping) else [tuple(c) for c in columns]


def _right_of(texts, ax):
    """For annotate: the window box around texts (the axes box when there are none)."""
    def box(renderer):
        boxes = [a.get_window_extent(renderer) for a in texts]
        return Bbox.union(boxes) if boxes else ax.bbox
    return box


def _columns(ax, pos, columns):
    """Text columns right of the frame, one string per row at the row positions. Each column starts a
    small gap right of the widest entry of the column before. Returns one list of Text per column,
    with the header last."""
    pairs = _pairs(columns)
    for header, strings in pairs:
        _need(len(pos), **{f"column {header!r}": strings})
    t = theme()
    out, prev = [], None
    for header, strings in pairs:
        xc, gap = ("axes fraction", 4) if prev is None else (_right_of(prev, ax), 7)
        col = [ax.annotate(str(s), xy=(1, p), xycoords=(xc, "data"), xytext=(gap, 0), textcoords="offset points",
                           ha="left", va="center", fontsize=t.size - 0.5, color=t.ink, annotation_clip=False)
               for s, p in zip(strings, pos)]
        if header:
            col.append(ax.annotate(str(header), xy=(1, 1), xycoords=(xc, "axes fraction"), xytext=(gap, 2),
                                   textcoords="offset points", ha="left", va="bottom", fontsize=t.size - 0.5,
                                   color=t.ink, annotation_clip=False))
        out.append(col)
        prev = col
    return out


def _end_label(ax, v, p, s, horizontal, base=0.0, fs=None):
    """Text s just beyond the end of a bar that runs from base to v at row position p."""
    t = theme()
    out = 1 if v >= base else -1
    kw = dict(textcoords="offset points", fontsize=fs or t.size - 1, color=t.ink, annotation_clip=False)
    if horizontal:
        return ax.annotate(s, xy=(v, p), xytext=(3 * out, 0), ha="left" if out > 0 else "right", va="center", **kw)
    return ax.annotate(s, xy=(p, v), xytext=(0, 2 * out), ha="center", va="bottom" if out > 0 else "top", **kw)


def _log_ticks(ax, which, nmax=5):
    """Labels at 1, 2 and 5 times each decade (or 1 and 3, or 1, whichever gives 2 to nmax labels) on a
    log axis that spans less than two decades, where the default locator labels one tick or none."""
    lo, hi = sorted(ax.get_xlim() if which == "x" else ax.get_ylim())
    if lo <= 0 or np.log10(hi / lo) >= 2:
        return
    e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    for mants in ((1, 2, 5), (1, 3), (1,)):
        ticks = [m * 10.0 ** e for e in range(e0, e1 + 1) for m in mants if lo <= m * 10.0 ** e <= hi]
        if 2 <= len(ticks) <= nmax:
            (ax.set_xticks if which == "x" else ax.set_yticks)(ticks, [f"{v:g}" for v in ticks])
            return


def _span(v, log):
    """Axis limits around the values v with a margin: 0.3 decades on a log axis, 12 % of the range else."""
    v = np.asarray(v, float)
    if not v.size:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo, hi = v.min(), v.max()
    if log:
        return 10 ** (np.log10(lo) - 0.3), 10 ** (np.log10(hi) + 0.3)
    pad = 0.12 * (hi - lo) or 0.5 * abs(hi) or 1.0
    return lo - pad, hi + pad


# ---- bars -------------------------------------------------------------------------------------

def stacked_bars(ax, rows, segments, *, horizontal=True, normalize=False, log=False, hatches=None, totals=False,
                 fmt="{:.3g}", markers=None, groups=None, columns=None, width=0.62, legend=True, colors=None):
    """Stacked bars, one per row, of segments {name: one value per row} in segment order. Positive values
    stack from the base in the positive direction and negative values in the negative direction.
    normalize=True divides each row by its total, so the bars show shares (a row that sums to zero
    stays empty). log=True reads the values as factors that multiply: the value axis is logarithmic,
    each bar starts at 1, and each factor carries it to the running product, factors above 1 to the
    right and below 1 to the left. NaN values, and non-positive factors under log, are left out.
    Segments take tints of the series colours with ink edges; folio adds hatches, and hatches (one per
    segment, "" for none) replaces them to single out a class; colors (one theme role such as "light"
    or "accent", or a colour, per segment) fills the segments as given instead of the tints, for
    classes a caption names by colour. totals=True (or a header string)
    prints each row's total, the product under log, with fmt in a column right of the frame, or above
    each bar when vertical. markers {label: one value per row, NaN for none} overlays a marker per row,
    a measured total say, filled in field and open in folio. groups adds a header above each group of
    rows; columns adds text columns after the totals (horizontal bars only). legend=True names segments
    and markers above the frame; a location string puts the legend inside.
    Returns {"bars": [BarContainer per segment], "totals": array, "pos": row positions, "texts": total
    labels or text columns, "markers": [Line2D], "headers": [Text], "legend": Legend or None}."""
    t = theme()
    rows = list(rows)
    n, names = len(rows), list(segments)
    _need(n, **{f"segment {k!r}": segments[k] for k in names})
    _need(len(names), "segments", hatches=hatches, colors=colors)
    _need(n, **{f"marker {k!r}": v for k, v in (markers or {}).items()})
    fills = [_fill(k, hatches) for k in range(len(names))]
    for k, c in enumerate(colors or ()):
        fills[k]["color"] = _role_color(c)
    pairs = _pairs(columns)
    for header, strings in pairs:
        _need(n, **{f"column {header!r}": strings})
    if normalize and log:
        raise ValueError("normalize and log exclude each other")
    if pairs and not horizontal:
        raise ValueError("text columns need horizontal bars")
    V = np.array([np.asarray(segments[k], float).ravel() for k in names]).reshape(len(names), n)
    ok = np.isfinite(V) & ((V > 0) if log else True)
    V = np.where(ok, V, 1.0 if log else 0.0)
    total = V.prod(axis=0) if log else V.sum(axis=0)
    L = np.log10(V) if log else V
    if normalize:
        L = np.divide(L, total, out=np.zeros_like(L), where=total != 0)
    lead = 1.0 if (groups is not None and horizontal) else 0.0
    pos, spans = _layout(n, groups, 1.0 if horizontal else 0.8, lead)
    up, down = np.zeros(n), np.zeros(n)
    bars = []
    for k, name in enumerate(names):
        v = L[k]
        start = np.where(v >= 0, up, down)
        end = start + v
        a, b = (10.0 ** start, 10.0 ** end) if log else (start, end)
        kw = fills[k]
        bar = (ax.barh(pos, b - a, left=a, height=width, label=name, **kw) if horizontal
               else ax.bar(pos, b - a, bottom=a, width=width, label=name, **kw))
        _hide_empty(bar, v)
        bars.append(bar)
        up, down = np.where(v >= 0, end, up), np.where(v < 0, end, down)
    labels = [_num(fmt, x) for x in total]
    texts = []
    if horizontal:
        head = totals if isinstance(totals, str) else ""
        texts = _columns(ax, pos, ([(head, labels)] if totals else []) + pairs)
    elif totals:
        top = 10.0 ** up if log else up
        texts = [ax.annotate(s, xy=(p, y), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                             fontsize=t.size - 1, color=t.ink, annotation_clip=False)
                 for p, y, s in zip(pos, top, labels)]
    mk = []
    for j, (label, vals) in enumerate((markers or {}).items()):
        vals = np.asarray(vals, float).ravel()
        xy = (vals, pos) if horizontal else (pos, vals)
        mk.append(ax.plot(*xy, label=label, **data_kw(marker=("D", "v", "s", "^")[j % 4], ms=4.5, zorder=5))[0])
    _category_axis(ax, pos, rows, horizontal, lead)
    heads = _headers(ax, spans, horizontal)
    line = ax.axvline if horizontal else ax.axhline
    if log:
        (ax.set_xscale if horizontal else ax.set_yscale)("log")
        log_minor(ax, "x" if horizontal else "y")
        line(1.0, color=t.ink, lw=0.6, zorder=3)
        _log_ticks(ax, "x" if horizontal else "y")
    elif (L < 0).any():
        line(0.0, color=t.ink, lw=0.6, zorder=3)
    if normalize and not (L < 0).any():
        (ax.set_xlim if horizontal else ax.set_ylim)(0, 1.0 if horizontal or not totals else 1.12)
    leg = _legend(ax, [_patch(fills[k], name) for k, name in enumerate(names)] + mk, legend)
    return {"bars": bars, "totals": total, "pos": pos, "texts": texts, "markers": mk, "headers": heads, "legend": leg}


def factor_bars(ax, labels, factors, *, log10=True, net=True, net_label="net", horizontal=True, values=True,
                digits=1, hatches=None, width=0.7, legend=True, colors=None):
    """Signed factor bars: one row per input with a bar from 0 of length log10(factor) on a linear axis,
    so a factor above 1 points right (or up) and one below 1 left (or down), each labelled 10^x at its
    end with `digits` decimals. factors is one value per row, or {series name: one value per row} for
    series side by side within each row (a second device, say). A single series is coloured by sign,
    gains in the sim colour and losses in the accent colour; several series take the segment fills of
    stacked_bars, with hatches (one per series) as there. colors (one theme role such as "light", or a
    colour, per series) fills every bar of a series as given, with an ink edge, in place of either
    choice. log10=False draws the values as given, for factors already in logarithms or for
    additive contributions. net=True adds a row below a dotted rule with each series' sum, the log of
    the total factor. NaN, and non-positive factors under log10, are left out of bars and sums.
    Returns {"bars": [BarContainer per series], "net": {series: sum} (key None for a single series),
    "texts": value labels, "pos": row positions, "legend": Legend or None}."""
    t = theme()
    labels = list(labels)
    n = len(labels)
    series = dict(factors) if isinstance(factors, Mapping) else {None: factors}
    _need(n, **{f"series {k!r}": v for k, v in series.items()})
    _need(len(series), "series", hatches=hatches, colors=colors)
    m = len(series)
    fills = [_fill(k, hatches) for k in range(m)]
    for k, c in enumerate(colors or ()):
        fills[k]["color"] = _role_color(c)
    rows = labels + ([net_label] if net else [])
    pos = np.arange(len(rows), dtype=float)
    if net:
        pos[-1] += 0.3
    h = width / max(m, 1)
    bars, texts, nets, shown = [], [], {}, [np.zeros(1)]
    for k, (name, f) in enumerate(series.items()):
        f = np.asarray(f, float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(f) if log10 else f.copy()
        v[~np.isfinite(v)] = np.nan
        nets[name] = float(np.nansum(v))
        vals = np.r_[v, nets[name]] if net else v
        draw = np.nan_to_num(vals)
        y = pos + (k - (m - 1) / 2) * h
        if m == 1 and colors is None:
            cs = [t.sim if x >= 0 else t.accent for x in draw]
            kw = dict(color=[lighten(c, 0.35) for c in cs] or None, edgecolor=cs or None, lw=0.6)
        else:
            kw = fills[k]
        bar = ax.barh(y, draw, height=h, **kw) if horizontal else ax.bar(y, draw, width=h, **kw)
        _hide_empty(bar, vals)
        bars.append(bar)
        shown.append(draw)
        for yi, x in zip(y, vals):
            if values and np.isfinite(x):
                s = rf"$10^{{{x:.{digits}f}}}$" if log10 else _num(f"{{:.{digits + 1}g}}", x)
                texts.append(_end_label(ax, x, yi, s, horizontal))
    (ax.axvline if horizontal else ax.axhline)(0.0, color=t.ink, lw=0.6, zorder=3)
    if net and n:
        (ax.axhline if horizontal else ax.axvline)((pos[-2] + pos[-1]) / 2, **ref_kw(ls=":"))
    _category_axis(ax, pos, rows, horizontal)
    allv = np.concatenate(shown)
    pad = 0.25 * (allv.max() - allv.min()) or 1.0
    (ax.set_xlim if horizontal else ax.set_ylim)(allv.min() - pad, allv.max() + pad)
    if log10:
        (ax.set_xlabel if horizontal else ax.set_ylabel)(r"$\log_{10}$ factor")
    leg = _legend(ax, [_patch(fills[k], str(nm)) for k, nm in enumerate(series)], legend) if m > 1 else None
    return {"bars": bars, "net": nets, "texts": texts, "pos": pos, "legend": leg}


def grouped_bars(ax, groups, series, *, log=False, horizontal=False, values=False, fmt="{:.3g}", ratios=False,
                 hatches=None, width=0.8, legend=True, colors=None, pack=False, ratio_fmt=None, label_fs=None):
    """Bars side by side: one cluster per group and one bar per series {name: one value per group}, in
    the segment fills of stacked_bars; colors (one theme role such as "light" or "sim", or a colour, per
    series) fills them as given instead, for series a caption names by colour. log=True puts the values
    on a log axis, where zero, negative and
    NaN values draw no bar. values=True prints each value at its bar end with fmt. ratios=True prints,
    beyond the last series of each group, its ratio to the first series ("×12"), for the change between
    two variants; it replaces that series' value labels. A zero line marks values of both signs.
    horizontal=True runs the groups from the top. legend names the series when there are several.
    pack=True closes the gaps a missing value (NaN, or non-positive under log) leaves: each group centres
    the bars it has, so a group with a value of one series only (a single reference bar) gets one centred
    bar, and ratios then compare the first and last bars a group has. ratio_fmt formats the ratio labels:
    a format string for the number after "×", or a function of the ratio that returns the whole label.
    label_fs sets the size of the value and ratio labels in points (default theme().size - 1.5).
    Returns {"bars": [BarContainer per series], "texts": printed labels, "pos": group positions,
    "legend": Legend or None}."""
    t = theme()
    groups = list(groups)
    n, names = len(groups), list(series)
    m = len(names)
    _need(n, "groups", **{f"series {k!r}": series[k] for k in names})
    _need(m, "series", hatches=hatches, colors=colors)
    V = np.array([np.asarray(series[k], float).ravel() for k in names]).reshape(m, n)
    ok = np.isfinite(V) & ((V > 0) if log else True)
    x = np.arange(n, dtype=float)
    w = width / max(m, 1)
    lfs = label_fs or t.size - 1.5
    # bar centres: every series in its own slot, or under pack the bars a group has centred in it
    off = np.array([x + (k - (m - 1) / 2) * w for k in range(m)]).reshape(m, n)
    if pack:
        for i in range(n):
            have = np.flatnonzero(ok[:, i])
            off[have, i] = x[i] + (np.arange(have.size) - (have.size - 1) / 2) * w

    def ratio_label(r):
        if callable(ratio_fmt):
            return str(ratio_fmt(r))
        return "×" + _num(ratio_fmt or ("{:.0f}" if abs(r) >= 10 else "{:.2g}"), r)

    bars, texts = [], []
    fills = [_fill(k, hatches) for k in range(m)]
    for k, c in enumerate(colors or ()):
        fills[k]["color"] = _role_color(c)
    for k, name in enumerate(names):
        v = np.where(ok[k], V[k], 0.0)
        kw = fills[k]
        bar = (ax.barh(off[k], v, height=w, label=name, **kw) if horizontal
               else ax.bar(off[k], v, width=w, label=name, **kw))
        _hide_empty(bar, np.where(ok[k], V[k], np.nan))
        bars.append(bar)
        if values and not (ratios and k == m - 1 and m > 1):
            texts += [_end_label(ax, val, p, _num(fmt, val), horizontal, fs=lfs)
                      for p, val, good in zip(off[k], V[k], ok[k]) if good]
    if ratios and m > 1:
        for i in range(n):
            have = np.flatnonzero(ok[:, i]) if pack else np.array([0, m - 1])
            if have.size < 2 or not (ok[have[0], i] and ok[have[-1], i]):
                continue
            a, b = V[have[0], i], V[have[-1], i]
            if a != 0:
                texts.append(_end_label(ax, b, off[have[-1], i], ratio_label(b / a), horizontal, fs=lfs))
    _category_axis(ax, x, groups, horizontal)
    if log:
        (ax.set_xscale if horizontal else ax.set_yscale)("log")
        good = V[ok]
        if good.size:
            lim = 10.0 ** np.floor(np.log10(good.min())), 10.0 ** np.ceil(np.log10(good.max()) + 0.15)
            (ax.set_xlim if horizontal else ax.set_ylim)(*lim)
        log_minor(ax, "x" if horizontal else "y")
    elif (V[ok] < 0).any():
        (ax.axvline if horizontal else ax.axhline)(0.0, color=t.ink, lw=0.6, zorder=3)
    leg = _legend(ax, [_patch(fills[k], nm) for k, nm in enumerate(names)], legend) if m > 1 else None
    return {"bars": bars, "texts": texts, "pos": x, "legend": leg}


def tornado(ax, labels, low, high, *, base=1.0, log=True, sort=True, names=("low input", "high input"),
            values=True, fmt="{:.3g}", columns=None, base_label=None, legend=True):
    """One-at-a-time sensitivity. For each input (row), two bars run from the base outcome to the
    outcome with that input at its low value (a tint of the model colour) and at its high value (the
    full model colour); when both fall on one side of the base, the shorter bar sits in front. log=True
    (the default) draws a log axis, for outcomes that scale as factors; the base line marks base.
    sort=True puts the widest swing (on the log scale under log) at the top. values=True prints each
    outcome at its bar end, outward from the base, with fmt; an outcome at the base gets no label.
    columns {header: one string per row, in input order} adds text columns right of the frame (the
    input range, the design step), reordered with the rows. base_label prints in the lower right
    corner. names labels the two bars in the legend.
    Returns {"order": input indices from the top, "bars": (low, high) BarContainers, "texts": value
    labels, "columns": [Text lists], "legend": Legend or None}."""
    t = theme()
    tints = (lighten(t.model, 0.55), t.model)
    labels = list(labels)
    n = len(labels)
    lo, hi = np.asarray(low, float).ravel(), np.asarray(high, float).ravel()
    _need(n, low=lo, high=hi)
    pairs = _pairs(columns)
    for header, strings in pairs:
        _need(n, **{f"column {header!r}": strings})

    def f(v):
        v = np.asarray(v, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(v > 0, v, np.nan)) if log else np.where(np.isfinite(v), v, np.nan)

    fb = float(f(base))
    extent = np.fmax(np.fmax(f(lo), f(hi)), fb) - np.fmin(np.fmin(f(lo), f(hi)), fb)
    order = np.argsort(-extent, kind="stable") if sort else np.arange(n)
    pos = np.arange(n, dtype=float)
    bars, texts = [], []
    for vals, c, label in zip((lo[order], hi[order]), tints, names):
        length = np.where(np.isfinite(f(vals)), vals - base, np.nan)
        bar = ax.barh(pos, np.nan_to_num(length), left=base, height=0.56, color=c, edgecolor="none", label=label)
        _hide_empty(bar, length)
        bars.append(bar)
        if values:
            texts += [_end_label(ax, v, p, _num(fmt, v), True, base=base) for p, v in zip(pos, vals)
                      if np.isfinite(f(v)) and not np.isclose(f(v), fb, atol=5e-3)]
    d = np.nan_to_num(np.abs(f(lo[order]) - fb)), np.nan_to_num(np.abs(f(hi[order]) - fb))
    for i in range(n):                                   # the shorter bar of a row in front
        bars[0][i].set_zorder(1.2 if d[0][i] < d[1][i] else 1.0)
        bars[1][i].set_zorder(1.0 if d[0][i] < d[1][i] else 1.2)
    ax.axvline(base, color=t.ink, lw=0.7, zorder=3)
    _category_axis(ax, pos, [labels[i] for i in order], True)
    shown = np.r_[f(lo), f(hi), fb]
    shown = shown[np.isfinite(shown)]
    pad = 0.25 * (shown.max() - shown.min()) or 0.3
    a, b = shown.min() - pad, shown.max() + pad
    if log:
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** a, 10.0 ** b)
        log_minor(ax, "x")
        _log_ticks(ax, "x")
    else:
        ax.set_xlim(a, b)
    cols = _columns(ax, pos, [(h, [s[i] for i in order]) for h, s in pairs])
    if base_label:
        note(ax, base_label, loc="lower right")
    leg = _legend(ax, [Patch(facecolor=c, edgecolor="none", label=nm) for c, nm in zip(tints, names)], legend)
    return {"order": order, "bars": tuple(bars), "texts": texts, "columns": cols, "legend": leg}


def _bar_status(status, c):
    """(face, edge, hatch, line style) of a bar of the given status in colour c."""
    t = theme()
    return {"measured": (lighten(c, 0.55), c, None, "-"), "simulated": ("white", c, None, "-"),
            "assumed": ("white", t.assumed, None, "-"), "fitted": (t.light, t.reference, None, "-"),
            "upper": ("white", c, "////", "-"), "lower": ("white", c, "////", "-"),
            "excluded": ("white", t.reference, None, "--"), "inferred": (t.light, c, None, "-")}[status]


def bar_rows(ax, labels, values, *, status=None, groups=None, base=0.0, log=False, ref=None, marks=None,
             columns=None, color=None, width=0.55, end_labels=False, fmt="{:.3g}", fill=None):
    """Labelled horizontal bars, one row per label from the top, each from base to its value. status (one
    per row, or one for all) sets the fill: "measured" a tint of color (the model colour by default),
    "simulated" open, "assumed" open grey, "fitted" grey, "upper" or "lower" open and hatched (a limit),
    "excluded" open with a dashed grey edge, "inferred" light grey with an edge in color. fill (a theme
    role such as "light", or a colour) replaces the tint of "measured" bars, for a fill a caption names or
    a light fill on which the grey marks stay readable. log=True draws a log axis whose limits span whole decades
    around the values; zero, negative and NaN values draw no bar there, and bars start at the left edge
    when base is not positive. base=1 draws factors out from 1. groups adds an italic header above each
    group of rows. ref draws a guide line at that value; marks (one per row, NaN for none) draws a grey
    tick on each row at a reference value, a published scatter say. columns adds text columns right of
    the frame; end_labels=True prints each value just beyond its bar end with fmt instead, for counts read
    off the bars. Returns {"bars": BarContainer, "pos": row positions, "headers": [Text], "marks":
    Line2D or None, "columns": [Text lists], "texts": the printed values}."""
    t = theme()
    c = color or t.model
    labels = list(labels)
    n = len(labels)
    v = np.asarray(values, float).ravel()
    _need(n, values=v, marks=marks)
    st = _row_status(status, n)
    lead = 1.0 if groups is not None else 0.0
    pos, spans = _layout(n, groups, 1.0, lead)
    good = np.isfinite(v) & ((v > 0) if log else True)
    start = base
    if log:
        cand = [v[good], [base] if base > 0 else [], [ref] if ref is not None and ref > 0 else []]
        if marks is not None:
            mv = np.asarray(marks, float)
            cand.append(mv[np.isfinite(mv) & (mv > 0)])
        cand = np.concatenate([np.asarray(x, float) for x in cand])
        lim = ((10.0 ** np.floor(np.log10(cand.min())), 10.0 ** np.ceil(np.log10(cand.max()) + 0.05)) if cand.size
               else (1.0, 10.0))
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        start = base if base > 0 else lim[0]
    length = np.where(good, v - start, np.nan)
    fc, ec, hatch, ls = zip(*[_bar_status(s, c) for s in st]) if n else ((), (), (), ())
    if fill is not None:
        face = _role_color(fill)
        fc = tuple(face if s == "measured" else f for s, f in zip(st, fc))
    bars = ax.barh(pos, np.nan_to_num(length), left=start, height=width, color=list(fc) or None,
                   edgecolor=list(ec) or None, lw=0.6)
    for r, h, s in zip(bars, hatch, ls):
        r.set_hatch(h)
        r.set_linestyle(s)
    _hide_empty(bars, length)
    if ref is not None:
        ax.axvline(ref, **ref_kw())
    mk = None
    if marks is not None:
        mk = ax.plot(np.asarray(marks, float), pos, ls="none", marker="|", ms=8, mew=1.4, color=t.reference,
                     zorder=4)[0]
    _category_axis(ax, pos, labels, True, lead)
    heads = _headers(ax, spans, True)
    if log:
        log_minor(ax, "x")
        _log_ticks(ax, "x")
    cols = _columns(ax, pos, columns)
    texts = [_end_label(ax, x, p, _num(fmt, x), True, base=start) for x, p, ok in zip(v, pos, good) if ok] if end_labels else []
    return {"bars": bars, "pos": pos, "headers": heads, "marks": mk, "columns": cols, "texts": texts}


# ---- points -----------------------------------------------------------------------------------

def dot_rows(ax, labels, values, *, lo=None, hi=None, status=None, groups=None, connect=False, ref=None, band=None,
             log=False, columns=None, color=None, markers=None):
    """Cleveland dot rows: one row per label from the top with a marker at each value. A row's value is
    one number or a list (two estimators, a prediction and a measurement), NaN for none. lo and hi (one
    per row, NaN for none) draw a thick range bar where hi > lo, for a value known only as a range.
    status (one per row, a list per row to match its values, or one for all) picks each marker: see the
    module docstring; "upper" and "lower" add an arrow (limit_arrow). connect=True joins the points of
    each row with a thin grey line. ref draws a guide line at that value (1 for a ratio), band (lo, hi)
    a tint behind the rows. log=True puts the values on a log axis and skips non-positive ones. groups
    and columns work as in bar_rows; color replaces the marker colour of every status but "assumed",
    "fitted" and "excluded". markers (one shape for all, one per row, or a list per row to match its
    values, None for the status default) sets the marker shapes, for kinds of comparison a caption
    names by shape (squares, diamonds, circles); the status still decides fill and colour.
    Returns {"pos": row positions, "points": [Line2D], "ranges": [Line2D],
    "arrows": [Annotation], "headers": [Text], "columns": [Text lists]}."""
    t = theme()
    labels = list(labels)
    n = len(labels)
    _need(n, values=values, lo=lo, hi=hi)
    rowst = [status] * n if status is None or isinstance(status, str) else list(status)
    _need(n, status=rowst)
    rowmk = [markers] * n if markers is None or isinstance(markers, str) else list(markers)
    _need(n, markers=rowmk)
    lead = 1.0 if groups is not None else 0.0
    pos, spans = _layout(n, groups, 1.0, lead)
    if band is not None:
        ax.axvspan(band[0], band[1], color=t.tint, lw=0, zorder=0)
    if ref is not None:
        ax.axvline(ref, **ref_kw(zorder=1))
    L = np.full(n, np.nan) if lo is None else np.asarray(lo, float)
    H = np.full(n, np.nan) if hi is None else np.asarray(hi, float)
    ranges, points, arrows = [], [], []
    for i, y in enumerate(pos):
        if np.isfinite(L[i]) and np.isfinite(H[i]) and H[i] > L[i] and (L[i] > 0 or not log):
            ranges.append(ax.plot([L[i], H[i]], [y, y], color=color or t.ink, lw=2.2, solid_capstyle="butt",
                                  zorder=3)[0])
        xs = np.atleast_1d(np.asarray(values[i], float))
        ss = rowst[i]
        ss = [ss or "measured"] * xs.size if ss is None or isinstance(ss, str) else list(ss)
        if len(ss) != xs.size:
            raise ValueError(f"row {labels[i]!r} has {xs.size} values and {len(ss)} statuses")
        mm = rowmk[i]
        mm = [mm] * xs.size if mm is None or isinstance(mm, str) else list(mm)
        if len(mm) != xs.size:
            raise ValueError(f"row {labels[i]!r} has {xs.size} values and {len(mm)} markers")
        keep = np.isfinite(xs) & ((xs > 0) if log else True)
        if connect and keep.sum() > 1:
            ax.plot([xs[keep].min(), xs[keep].max()], [y, y], color=t.reference, lw=0.8, zorder=2)
        for x, s, k, shape in zip(xs, ss, keep, mm):
            _check_status(s)
            if not k:
                continue
            if s in ("upper", "lower"):
                m, a = limit_arrow(ax, x, y, "left" if s == "upper" else "right", color=color, marker=shape or "o")
                points.append(m)
                arrows.append(a)
            else:
                kw = _marker(s, color)
                if shape:
                    kw["marker"] = shape
                points.append(ax.plot([x], [y], **kw)[0])
    _category_axis(ax, pos, labels, True, lead)
    heads = _headers(ax, spans, True)
    if log:
        ax.set_xscale("log")
        log_minor(ax, "x")
        _log_ticks(ax, "x")
    cols = _columns(ax, pos, columns)
    return {"pos": pos, "points": points, "ranges": ranges, "arrows": arrows, "headers": heads, "columns": cols}


def _entries(cell):
    """(value, status) pairs of a study_strips cell: a number, a (number, status) pair or a list of them."""
    if isinstance(cell, tuple) and len(cell) == 2 and isinstance(cell[1], str):
        return [(float(cell[0]), _check_status(cell[1]))]
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [e for c in cell for e in _entries(c)]
    return [(float(cell), "measured")]


def study_strips(axes, rows, columns, *, lims=None, log=True, missing="n.r.", color=None):
    """Cross-study dot strips: the rows (studies, devices) run down every axes in axes, one axes per
    quantity, each with a value axis of its own. columns maps each quantity's axis label to {row: entry}
    in axes order, where an entry is a number, a (number, status) pair or a list of them; the status
    picks the marker as in dot_rows, "measured" by default. A row absent from a quantity prints missing
    at the centre of that strip. NaN, and non-positive values on a log strip, are skipped. lims {label:
    (lo, hi)} fixes a strip's range; otherwise it spans its values with a margin. color replaces the
    marker colour of every status but "assumed", "fitted" and "excluded", as in dot_rows. Row labels go
    on the first axes only, and the strips work with or without a shared y axis.
    Returns {"pos": {row: position}, "points": {label: [Line2D]}, "missing": [Text]}."""
    t = theme()
    axes, names, rows = list(np.ravel(axes)), list(columns), list(rows)
    if len(axes) != len(names):
        raise ValueError(f"{len(names)} quantity columns need as many axes, got {len(axes)}")
    pos = {r: float(i) for i, r in enumerate(rows)}
    p = np.arange(len(rows), dtype=float)
    points, gaps = {}, []
    for j, (ax, name) in enumerate(zip(axes, names)):
        cells = columns[name]
        extra = [r for r in cells if r not in pos]
        if extra:
            raise ValueError(f"column {name!r} has rows {extra} that are not in rows")
        for y in p:
            ax.axhline(y, color=t.light, lw=0.5, zorder=0)
        pts, drawn = [], []
        for r in rows:
            for x, s in (_entries(cells[r]) if r in cells else []):
                if np.isfinite(x) and (x > 0 or not log):
                    pts.append(ax.plot([x], [pos[r]], **_marker(s, color))[0])
                    drawn.append(x)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(numticks=3))
            log_minor(ax, "x")
        else:
            ax.xaxis.set_major_locator(MaxNLocator(3))
        a, b = (lims or {}).get(name) or _span(drawn, log)
        ax.set_xlim(a, b)
        if log:
            _log_ticks(ax, "x", nmax=3)
        mid = np.sqrt(a * b) if log else 0.5 * (a + b)
        gaps += [ax.text(mid, pos[r], missing, ha="center", va="center", fontsize=t.size - 1, color=t.reference,
                         zorder=3, bbox=dict(fc="white", ec="none", pad=0.8)) for r in rows if r not in cells]
        ax.set_xlabel(name)
        if j == 0:
            _category_axis(ax, p, rows, True)
        else:
            ax.set_yticks(p)
            ax.set_ylim(p.max() + 0.6 if p.size else 0.6, -0.6)
            ax.tick_params(axis="y", which="both", length=0, labelleft=False, labelright=False)
        points[name] = pts
    return {"pos": pos, "points": points, "missing": gaps}


def ratio_strip(ax, groups, *, status=None, ref=1.0, band=None, rms=None, log=True, jitter=0.22, seed=0,
                horizontal=False, label="predicted / measured", legend=True, markers=None, ms=None, legend_at=None):
    """Ratios such as predicted over measured as a strip (beeswarm) per category: groups {category:
    ratios}. Each point sits at its category, spread sideways by a uniform jitter of half width `jitter`
    from a generator seeded with seed, so a rebuild puts every point in the same spot. status {category:
    one status per ratio} picks markers as in dot_rows; "excluded" points (drawn but not counted) line
    up just left of their strip, and "upper" and "lower" bounds carry an arrow. A line marks ref, perfect
    agreement; band (a fraction, read as ref(1 +- band), or a (lo, hi) pair) shades the agreement window;
    rms {category: fraction} draws a narrow grey bar of ref(1 +- rms) beside that category, for a
    published scatter. log=True puts the ratios on a log axis and skips non-positive ones. horizontal=True
    lays the categories out as rows from the top with the ratio along x. label names the ratio axis.
    legend=True names the statuses present above the frame; a dict renames them. markers {category: one
    shape per ratio, None for the status default} sets the marker shapes, for kinds a caption names by
    shape (squares for design targets, triangles for bounds); the status still decides fill and colour.
    With markers the legend names each (status, shape) pair present, and the dict may key a label by the
    status or by a (status, shape) pair; a label of None leaves an entry out, and entries that share a
    label show once. ms sets the marker size, for a mini-plot. legend_at places the legend: None above
    the frame, a location string inside it, or a dict of ax.legend keywords.
    Returns {"pos": {category: point coordinates across the strip}, "points": {category: [Line2D]},
    "legend": Legend or None}."""
    t = theme()
    cats = list(groups)
    status, rms, markers = status or {}, rms or {}, markers or {}
    extra = [c for c in list(status) + list(rms) + list(markers) if c not in groups]
    if extra:
        raise ValueError(f"categories {extra} are not in groups")
    rng = np.random.default_rng(seed)
    if band is not None:
        b0, b1 = (ref * (1 - band), ref * (1 + band)) if np.ndim(band) == 0 else band
        (ax.axvspan if horizontal else ax.axhspan)(b0, b1, color=t.tint, lw=0, zorder=0)
    (ax.axvline if horizontal else ax.axhline)(ref, color=t.ink, lw=0.6, zorder=2)
    down, up = ("left", "right") if horizontal else ("down", "up")
    pos, points, present = {}, {}, []
    for i, c in enumerate(cats):
        r = np.asarray(groups[c], float).ravel()
        ss = list(status.get(c, ["measured"] * r.size))
        if len(ss) != r.size:
            raise ValueError(f"category {c!r} has {r.size} ratios and {len(ss)} statuses")
        mm = list(markers.get(c, [None] * r.size))
        if len(mm) != r.size:
            raise ValueError(f"category {c!r} has {r.size} ratios and {len(mm)} markers")
        side = np.array([_check_status(s) == "excluded" for s in ss], bool)
        q = i + rng.uniform(-jitter, jitter, r.size)
        q[side] = i - jitter - 0.12 + rng.uniform(-0.03, 0.03, side.sum())
        pts = []
        for qi, ri, s, shape in zip(q, r, ss, mm):
            if not np.isfinite(ri) or (log and ri <= 0):
                continue
            if (s, shape) not in present:
                present.append((s, shape))
            x, y = (ri, qi) if horizontal else (qi, ri)
            if s in ("upper", "lower"):
                pts.append(limit_arrow(ax, x, y, down if s == "upper" else up, marker=shape or "o", ms=ms)[0])
            else:
                kw = _marker(s, ms=ms)
                if shape:
                    kw["marker"] = shape
                pts.append(ax.plot([x], [y], **kw)[0])
        pos[c], points[c] = q, pts
        if c in rms:
            f, xc = rms[c], i + jitter + 0.12
            box = ((ref * (1 - f), xc - 0.04), 2 * f * ref, 0.08) if horizontal else ((xc - 0.04, ref * (1 - f)), 0.08,
                                                                                      2 * f * ref)
            ax.add_patch(Rectangle(*box, facecolor=t.light, edgecolor="none", zorder=1))
    _category_axis(ax, np.arange(len(cats), dtype=float), cats, horizontal)
    which = "x" if horizontal else "y"
    if log:
        (ax.set_xscale if horizontal else ax.set_yscale)("log")
        log_minor(ax, which)
        _log_ticks(ax, which)
    if label:
        (ax.set_xlabel if horizontal else ax.set_ylabel)(label)
    names = legend if isinstance(legend, Mapping) else None
    handles = (_shape_handles(present, names, horizontal) if markers
               else _status_handles({s for s, _ in present}, names, horizontal=horizontal))
    where = True if names is not None else legend
    if legend_at is not None and where:
        where = legend_at
    leg = _legend(ax, handles, where)
    return {"pos": pos, "points": points, "legend": leg}


def paired_stems(ax, items, a, b, *, groups=None, log=False, labels=("forecast", "measured"), markers=("D", "o"),
                 rotation=90, legend=True, colors=None, ms=None):
    """Two values per item joined by a grey stem, with the items along x: a as an open marker (proposed,
    forecast, simulated) and b as a measured one (built, measured), filled in field and open in folio;
    markers (for a, for b) gives the two shapes, which keep them apart in folio. colors (for a, for b)
    replaces the sim and data colours, and ms (one size, or one for a and one for b) the marker sizes, a
    larger open marker ringing the measured one where the two values agree. NaN leaves a marker
    out, and its stem. log=True puts the
    values on a log axis and skips non-positive ones; log="symlog" keeps zero on a linear stretch below
    1, for counts that can be 0. groups spaces the groups apart, labels each above the frame and
    separates them with thin rules. Item names run along x at rotation degrees; pass empty strings to
    hide them. legend names the two markers with labels, above the frame, or inside it when there are
    groups. Returns {"x": item positions, "open": Line2D, "filled": Line2D, "stems": LineCollection,
    "headers": [Text], "legend": Legend or None}."""
    t = theme()
    items = list(items)
    n = len(items)
    A, B = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    _need(n, a=A, b=B)
    x, spans = _layout(n, groups, 0.8, 0.0)

    def ok(v):
        return np.isfinite(v) & ((v > 0) if log is True else True)

    both = ok(A) & ok(B)
    stems = LineCollection([[(xi, ai), (xi, bi)] for xi, ai, bi in zip(x[both], A[both], B[both])],
                           colors=t.reference, linewidths=0.8, zorder=2)
    ax.add_collection(stems)
    ca, cb = colors if colors is not None else (None, None)
    sa, sb = ms if np.ndim(ms) == 1 else (ms, ms)
    op, = ax.plot(x[ok(A)], A[ok(A)], label=labels[0], **sim_kw(ca, marker=markers[0], ms=sa))
    fl, = ax.plot(x[ok(B)], B[ok(B)], label=labels[1], **data_kw(cb, marker=markers[1], ms=sb))
    _category_axis(ax, x, items, False)
    ax.tick_params(axis="x", labelrotation=rotation)
    heads = _headers(ax, spans, False)
    if log == "symlog":
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", MINUS)))
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.25), hi)
    elif log:
        ax.set_yscale("log")
        log_minor(ax, "y")
    leg = _legend(ax, [op, fl], "best" if (spans and legend is True) else legend)
    return {"x": x, "open": op, "filled": fl, "stems": stems, "headers": heads, "legend": leg}


def timeline(ax, rows, events, *, kinds=None, break_at=None, compress=0.36, ncol=2, dx=None, dy=0.36, bands=True,
             ticks=None, legend=True, styles=None):
    """Events in time along category rows, one marker per event. events lists (row, time) or (row, time,
    kind) tuples; rows gives the row labels from the top. Each kind takes a series colour and marker,
    filled for as many kinds as the colour cycle holds and open for the next as many; kinds sets their
    order (first appearance otherwise). styles {kind: marker keywords} (marker, ms, color, mec, mfc, mew,
    zorder) replaces that choice for a kind, for a marker key that a second panel or the caption shares.
    Events that share a row and a time sit side by side, ncol to a line, dx apart
    in time (2.5 % of the time span by default) and dy rows apart, so none hides another. break_at
    compresses the times before it by the factor compress, for a sparse early record, stacks the shared
    cells there in one column and marks the break on the time axis; ticks lists the times to label
    (default: the first time and evenly spaced times from the break). bands=True tints every other row.
    legend names the kinds above the frame. Returns {"xmap": function from time to x coordinate,
    "points": {kind: Line2D}, "legend": Legend or None}."""
    t = theme()
    rows, ev = list(rows), [tuple(e) for e in events]
    for e in ev:
        if len(e) not in (2, 3):
            raise ValueError(f"event {e!r} is not (row, time) or (row, time, kind)")
        if e[0] not in rows:
            raise ValueError(f"event row {e[0]!r} is not one of rows")
    kind_of = [e[2] if len(e) == 3 else None for e in ev]
    order = list(dict.fromkeys(kind_of)) if kinds is None else list(kinds)
    unknown = sorted({str(k) for k in kind_of if k not in order})
    if unknown:
        raise ValueError(f"event kinds {unknown} are not in kinds")
    styles = dict(styles or {})
    extra = sorted(str(k) for k in styles if k not in order)
    if extra:
        raise ValueError(f"styles name kinds {extra} that are not in kinds")
    times = np.array([float(e[1]) for e in ev])
    brk = break_at if (break_at is not None and times.size and times.min() < break_at) else None
    t0, gap = (times.min() if times.size else 0.0), 0.55

    def xmap(tt):
        tt = np.asarray(tt, float)
        if brk is None:
            return tt
        return np.where(tt < brk, (tt - t0) * compress, (brk - t0) * compress + gap + (tt - brk))

    X = xmap(times)
    sx = dx if dx is not None else 0.025 * max(float(np.ptp(X)) if X.size else 0.0, 1.0)
    cells = {}
    for i, e in enumerate(ev):
        cells.setdefault((e[0], times[i]), []).append(i)
    xs, ys = np.empty(len(ev)), np.empty(len(ev))
    for (r, tt), idx in cells.items():
        idx.sort(key=lambda i: (order.index(kind_of[i]), i))
        nc = 1 if (brk is not None and tt < brk) else max(1, min(len(idx), ncol))
        nr = -(-len(idx) // nc)
        sy = dy if nr <= 2 else min(dy, 0.8 / (nr - 1))
        for j, i in enumerate(idx):
            xs[i] = xmap(tt) + (j % nc - (nc - 1) / 2) * sx
            ys[i] = rows.index(r) + (j // nc - (nr - 1) / 2) * sy
    pts, nk = {}, len(t.cycle)
    for j, kind in enumerate(order):
        sel = [i for i, k in enumerate(kind_of) if k == kind]
        c = t.cycle[j % nk]
        kw = dict(marker=t.markers[j % len(t.markers)], ms=4.5, color=c, mec=c, mfc=c if (j // nk) % 2 == 0 else "white",
                  mew=0.9, zorder=4)
        kw.update(styles.get(kind, {}))
        pts[kind] = ax.plot(xs[sel], ys[sel], ls="none", label=None if kind is None else str(kind), **kw)[0]
    if bands:
        for i in range(0, len(rows), 2):
            ax.axhspan(i - 0.5, i + 0.5, color=t.tint, lw=0, zorder=0)
    _category_axis(ax, np.arange(len(rows), dtype=float), rows, True)
    if brk is not None:
        xb, tr = (brk - t0) * compress + gap / 2, ax.get_xaxis_transform()
        for d in (-0.07, 0.07):
            ax.plot([xb + d - 0.05, xb + d + 0.05], [-0.025, 0.025], transform=tr, color=t.ink, lw=t.axes_lw,
                    clip_on=False, zorder=6)
        if ticks is None:
            tmax, early = max(times.max(), brk), max(1.0, np.ceil((brk - t0) / 3))
            ticks = (list(np.arange(t0, brk - 0.5 * early, early))
                     + list(np.arange(brk, tmax + 1e-9, max(1.0, np.ceil((tmax - brk) / 5)))))
    if ticks is not None:
        ax.set_xticks(xmap(ticks), [f"{v:g}" for v in ticks])
    if X.size:
        ax.set_xlim(xs.min() - 1.5 * sx, xs.max() + 1.5 * sx)
    leg = _legend(ax, [pts[k] for k in order if k is not None], legend)
    return {"xmap": xmap, "points": pts, "legend": leg}


def limit_arrow(ax, x, y, direction, *, length=9.0, status="simulated", color=None, marker="o", ms=None):
    """A bound: a marker at (x, y) with an arrow `length` points long toward the side where the true value
    lies, direction "up", "down", "left" or "right". The marker follows the grammar of status, open
    ("simulated") by default; the arrow takes the marker's colour. The length is set in points, so the
    arrow looks the same on linear and log axes. Returns (marker Line2D, arrow Annotation)."""
    if direction not in _DIRS:
        raise ValueError(f"unknown direction {direction!r}; choose one of {', '.join(_DIRS)}")
    kw = {**_marker(status, color, ms), "marker": marker}
    m, = ax.plot([x], [y], **kw)
    c = kw.get("mec", kw["color"])
    r = 0.5 * m.get_markersize() + 1.0
    ux, uy = _DIRS[direction]
    a = ax.annotate("", xy=(x, y), xytext=(ux * (length + r), uy * (length + r)), textcoords="offset points",
                    arrowprops=dict(arrowstyle="<|-", mutation_scale=6, color=c, lw=0.7, shrinkA=0, shrinkB=r))
    return m, a
