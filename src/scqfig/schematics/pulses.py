"""Pulse-sequence timing diagrams. Envelopes are computed from their formulas: truncated Gaussian and
DRAG pulses on the XY lines, Gaussian-filtered square pulses on the flux and readout lines."""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.special import erf

from ..core import ref_kw, theme
from ..themes import lighten

__all__ = ["gaussian_pulse", "drag_pulse", "square_pulse", "pulse_sequence"]

_RISE = 2 * 1.2815515655446004                    # 10-90 % width of a Gaussian-filtered step, in sigma


def gaussian_pulse(t, t0, sigma, amp=1.0, n_sigma=2.0):
    """Gaussian envelope centred at t0, cut at t0 +- n_sigma sigma and lifted so that it starts and ends
    at zero: amp (g - g_c)/(1 - g_c), g = exp(-(t - t0)^2 / 2 sigma^2), g_c its value at the cut."""
    t = np.asarray(t, float)
    gc = np.exp(-n_sigma ** 2 / 2)
    g = np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    return np.where(np.abs(t - t0) <= n_sigma * sigma, amp * (g - gc) / (1 - gc), 0.0)


def drag_pulse(t, t0, sigma, amp=1.0, beta=1.0, n_sigma=2.0):
    """DRAG pulse (Motzoi et al. 2009): the in-phase Gaussian I(t) and the quadrature Q(t) = -beta dI/dt,
    which cancels leakage to |2> and the phase error for beta near -1/alpha (beta in time units).
    Returns (I, Q)."""
    t = np.asarray(t, float)
    gc = np.exp(-n_sigma ** 2 / 2)
    g = np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    inside = np.abs(t - t0) <= n_sigma * sigma
    dI = np.where(inside, -amp * (t - t0) / sigma ** 2 * g / (1 - gc), 0.0)
    return gaussian_pulse(t, t0, sigma, amp, n_sigma), -beta * dI


def square_pulse(t, t0, length, amp=1.0, rise=2.0):
    """Square pulse from t0 to t0 + length whose edges are a step filtered by a Gaussian, so that
    `rise` is the 10-90 % rise time: amp/2 [erf((t - t0)/(sqrt2 s)) - erf((t - t0 - length)/(sqrt2 s))]."""
    t = np.asarray(t, float)
    s = rise / _RISE
    return amp / 2 * (erf((t - t0) / (np.sqrt(2) * s)) - erf((t - t0 - length) / (np.sqrt(2) * s)))


def _span(p):
    k = p["kind"]
    if k in ("gauss", "drag"):
        w = p.get("n_sigma", 2.0) * p["sigma"]
        return p["t0"] - w, p["t0"] + w
    return p["t0"] - 2 * p.get("rise", 2.0), p["t0"] + p["length"] + 2 * p.get("rise", 2.0)


def _envelope(p, t):
    k = p["kind"]
    if k == "gauss":
        return gaussian_pulse(t, p["t0"], p["sigma"], p.get("amp", 0.8), p.get("n_sigma", 2.0)), None
    if k == "drag":
        return drag_pulse(t, p["t0"], p["sigma"], p.get("amp", 0.8), p.get("beta", 0.5 * p["sigma"]),
                          p.get("n_sigma", 2.0))
    return square_pulse(t, p["t0"], p["length"], p.get("amp", 0.5), p.get("rise", 2.0)), None


_KINDS = ("gauss", "drag", "square", "readout")


def pulse_sequence(ax, rows, pulses, *, sync=(), cut=None, tmax=None, unit="ns", row_h=1.0, height=0.62,
                   window_label="integration", fs=None):
    """Timing diagram with one baseline per row (top to bottom). Each pulse is a dict with "row",
    "kind" and its parameters: "gauss" and "drag" (centre t0, sigma, amp, beta; drive colour, the DRAG
    quadrature dashed), "square" (start t0, length, amp, rise; flux colour, amp may be negative) and
    "readout" (t0, length, amp, rise and an integration "window" (start, end), shaded; readout colour).
    "label" prints above the pulse. Amplitudes are fractions of `height` of a row. sync times get
    dashed lines through every row; cut = (t1, t2) removes that stretch of a long pulse and marks the
    break on every row and on the time axis. Returns the row heights and the time-to-page map."""
    t = theme()
    fs = fs or t.size - 0.5
    for p in pulses:
        if p.get("row") not in rows:
            raise ValueError(f"pulse row {p.get('row')!r} is not one of {list(rows)}")
        if p.get("kind") not in _KINDS:
            raise ValueError(f"pulse kind {p.get('kind')!r} is not one of {_KINDS}")
    ys = {name: -k * row_h for k, name in enumerate(rows)}
    t1 = tmax if tmax is not None else max(_span(p)[1] for p in pulses) * 1.03
    t0 = 0.0
    if cut is not None:
        c0, c1 = cut
        gapw = 0.035 * (t1 - (c1 - c0))
    else:
        c0 = c1 = np.inf
        gapw = 0.0

    def tmap(x):
        x = np.asarray(x, float)
        out = np.where(x <= c0, x, np.where(x >= c1, x - (c1 - c0) + gapw, np.nan))
        return out.item() if out.ndim == 0 else out

    colors = {"gauss": t.chip["drive"], "drag": t.chip["drive"], "square": t.chip["flux"],
              "readout": t.chip["resonator"]}
    X0, X1 = tmap(t0), tmap(t1)
    for name, y in ys.items():
        if cut is None:
            ax.plot([X0, X1], [y, y], color=t.ink, lw=0.6, ls="-", zorder=2)
        else:
            ax.plot([X0, tmap(c0)], [y, y], color=t.ink, lw=0.6, ls="-", zorder=2)
            ax.plot([tmap(c1), X1], [y, y], color=t.ink, lw=0.6, ls="-", zorder=2)
    tt = np.linspace(t0, t1, 6000)
    keep = ~np.isnan(tmap(tt))
    X = tmap(tt)
    H = height * row_h
    for p in pulses:
        y, c = ys[p["row"]], p.get("color") or colors[p["kind"]]
        a, b = _span(p)
        on = keep & (tt >= a) & (tt <= b)
        env, q = _envelope(p, tt)
        if p["kind"] == "readout" and "window" in p:
            w0, w1 = p["window"]
            for s0, s1 in ([(w0, min(w1, c0)), (max(w0, c1), w1)] if cut else [(w0, w1)]):
                if s1 > s0:
                    ax.add_patch(Rectangle((tmap(s0), y - 0.12 * row_h), tmap(s1) - tmap(s0),
                                           H * abs(p.get("amp", 0.5)) + 0.24 * row_h, fc=lighten(c, 0.85),
                                           ec="none", zorder=1))
            ax.text(tmap(w1) - 0.01 * (X1 - X0), y - 0.06 * row_h, window_label, ha="right", va="top",
                    fontsize=fs - 1, color=t.reference, zorder=6)
        ax.fill_between(X, y, y + H * env, where=on, color=lighten(c, 0.72), lw=0, zorder=3)
        ax.plot(np.where(on, X, np.nan), y + H * env, color=c, lw=t.lw * 0.9, ls="-", zorder=4)
        if q is not None:
            ax.plot(np.where(on, X, np.nan), y + H * q, color=c, lw=t.lw * 0.8, ls=(0, (2.4, 1.4)), zorder=4)
        if p.get("label"):
            peak = np.nanmax(np.where(on, env, np.nan)) if np.any(on) else 0
            top = y + H * max(peak, 0) + 0.06 * row_h
            run = np.flatnonzero(on)
            run = run[: np.argmax(np.diff(run) > 1) + 1] if np.any(np.diff(run) > 1) else run   # before any cut
            mid = tmap(p["t0"]) if p["kind"] in ("gauss", "drag") else X[run].mean()
            ax.text(mid, top, p["label"], ha="center", va="bottom", fontsize=fs, color=c, zorder=6)
    ytop, ybot = ys[rows[0]] + 0.95 * row_h, ys[rows[-1]] - 0.3 * row_h
    for s in sync:
        ax.plot([tmap(s)] * 2, [ybot, ytop], **ref_kw(zorder=0))
    if cut is not None:
        _break_marks(ax, tmap(c0), tmap(c1), list(ys.values()), row_h)
        ticks, labels = [], []
        for lo, hi in ((t0, c0), (c1, t1)):
            for v in MaxNLocator(3 if hi - lo > 0.3 * (t1 - t0) else 2).tick_values(lo, hi):
                if lo <= v <= hi:
                    ticks.append(tmap(v))
                    labels.append(f"{v:g}")
        ax.set_xticks(ticks, labels)
        ax.spines["bottom"].set_visible(False)                  # the time axis breaks with the rows
        for a, b in ((X0, tmap(c0)), (tmap(c1), X1)):
            ax.plot([a, b], [ybot, ybot], color=t.ink, lw=t.axes_lw, ls="-", solid_capstyle="butt", clip_on=False)
        _break_marks(ax, tmap(c0), tmap(c1), [ybot], row_h)
    ax.set_xlim(X0, X1)
    ax.set_ylim(ybot, ytop)
    ax.set_yticks(list(ys.values()), list(rows))
    ax.tick_params(axis="y", length=0, left=False, right=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", top=False)
    ax.set_xlabel(f"time ({unit})")
    return {"rows": ys, "map": tmap}


def _break_marks(ax, x0, x1, ys, row_h):
    t = theme()
    xm, d = 0.5 * (x0 + x1), 0.18 * row_h
    for y in ys:
        for dx in (-0.3, 0.3):
            ax.plot([xm + dx * (x1 - x0) - 0.25 * (x1 - x0), xm + dx * (x1 - x0) + 0.25 * (x1 - x0)],
                    [y - d / 2, y + d / 2], color=t.ink, lw=0.6, ls="-", zorder=5)
