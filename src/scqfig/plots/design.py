"""Design-stage plots: frequency plans and collision maps, capacitance matrices, loss participation,
dispersive shift and Purcell limits, simulation convergence."""
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, SymmetricalLogLocator

from .. import physics as ph
from ..core import MINUS as _MINUS
from ..core import model_kw, note, ref_kw, sci, series, sim_kw, theme
from ..themes import lighten
from ._base import annotated_matrix, cell_text

__all__ = ["frequency_plan", "collision_map", "capacitance_matrix", "participation", "dispersive_shift",
           "purcell_limit", "convergence"]

def collision_windows(alpha, delta=0.017, delta_12=0.030, delta_2ph=0.004):
    """Forbidden detunings Delta = f_q2 - f_q1 (GHz) of two coupled transmons that share the
    anharmonicity alpha (GHz, negative). Three kinds: the two 01 transitions degenerate
    (|Delta| < delta), one qubit's 01 transition on the other's 12 transition (|Delta -+ alpha| <
    delta_12), and the two-photon f02/2 of one qubit on the other's f01 (|Delta -+ alpha/2| <
    delta_2ph). The default half widths are typical design values; set your own for your gate.
    Returns a list of (kind, centre, half width, label) with kind "01-01", "01-12" or "2-photon"."""
    a = float(alpha)
    return [("01-01", 0.0, delta, r"$\Delta = 0$"),
            ("01-12", a, delta_12, r"$\Delta = \alpha$"),
            ("01-12", -a, delta_12, r"$\Delta = -\alpha$"),
            ("2-photon", a / 2, delta_2ph, r"$\Delta = \alpha/2$"),
            ("2-photon", -a / 2, delta_2ph, r"$\Delta = -\alpha/2$")]


def frequency_plan(ax, qubits, *, resonators=None, purcell=None, collisions=None, unit="GHz", legend=True):
    """A frequency plan with one row per element on a shared frequency axis. qubits {name: (f01, f12)}
    draws each qubit's f01 (circle) and f12 (triangle) in its qubit colour, joined by a line one
    anharmonicity long. resonators {name: fr} draws the readout resonators (diamonds) in the bus
    colour. purcell (f_lo, f_hi) shades the Purcell-filter passband across every row and gives the
    filter a row of its own. collisions {row name: [(lo, hi), ...]} fills the windows that row's
    element must avoid; collision_windows gives them for a coupled pair. Returns {name: row}."""
    t = theme()
    rows = list(qubits) + list(resonators or {}) + (["Purcell filter"] if purcell else [])
    y = {n: i for i, n in enumerate(rows)}
    h = 0.56
    for i in range(len(rows)):
        ax.axhline(i, color=t.light, ls="-", lw=0.4, zorder=0.5)
    window = dict(fc=t.light, ec=t.reference, lw=0.4, zorder=1)
    band = lighten(t.bus, 0.88)
    if purcell:
        lo, hi = purcell
        ax.axvspan(lo, hi, fc=band, ec="none", zorder=0)
        ax.add_patch(Rectangle((lo, y["Purcell filter"] - h / 2), hi - lo, h, fc=lighten(t.bus, 0.6), ec=t.bus,
                               lw=0.6, zorder=2))
    for name, wins in (collisions or {}).items():
        for lo, hi in wins:
            ax.add_patch(Rectangle((lo, y[name] - h / 2), hi - lo, h, **window))
    for k, (name, (f01, f12)) in enumerate(qubits.items()):
        c = t.qubits[k % len(t.qubits)]
        ax.plot([f12, f01], [y[name]] * 2, color=c, ls="-", lw=0.8, zorder=3)
        ax.plot(f01, y[name], **sim_kw(color=c, marker="o", ms=5))
        ax.plot(f12, y[name], **sim_kw(color=c, marker="v", ms=4.5))
    for name, fr in (resonators or {}).items():
        ax.plot(fr, y[name], **sim_kw(color=t.bus, marker="D", ms=4))
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlabel(f"frequency ({unit})")
    if legend:
        handles = [Line2D([], [], label=r"$f_{01}$", **sim_kw(color=t.ink, marker="o", ms=5)),
                   Line2D([], [], label=r"$f_{12}$", **sim_kw(color=t.ink, marker="v", ms=4.5))]
        if resonators:
            handles.append(Line2D([], [], label="readout resonator", **sim_kw(color=t.bus, marker="D", ms=4)))
        if purcell:
            handles.append(Patch(fc=band, ec="none", label="Purcell passband"))
        if collisions:
            handles.append(Patch(label="collision window", **window))
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                  frameon=False, handlelength=1.3, columnspacing=1.4, borderaxespad=0.2)
    return y


def collision_map(ax, alpha, points=None, *, lim=None, windows=None, spread=None, unit="GHz", labels=True):
    """The f_q2 against f_q1 plane of two coupled transmons with the forbidden bands of
    collision_windows(alpha) shaded (windows= replaces them) and each band labelled near its upper
    end. Design points {label: (f_q1, f_q2)} are marked and labelled along their gap, below and to the
    left; spread (GHz) draws a circle of that radius around each, for the fabrication scatter. A design
    point inside a band becomes an ink cross. lim (lo, hi) sets both axes, which keep equal scales so
    the bands run at 45 degrees. Returns {label: [kinds of band it sits in]}."""
    t = theme()
    windows = collision_windows(alpha) if windows is None else windows
    pts = dict(points or {})
    if lim is None:
        f = np.array(list(pts.values()), float).ravel() if pts else np.array([4.5, 5.5])
        pad = 1.6 * abs(alpha)
        lim = (f.min() - pad, f.max() + pad)
    lo, hi = lim
    hue = {"01-01": t.accent, "01-12": t.signal, "2-photon": t.bus}
    x = np.array([lo - 1.0, hi + 1.0])
    for kind, c, w, label in windows:
        ax.fill_between(x, x + c - w, x + c + w, fc=lighten(hue[kind], 0.6), ec=hue[kind], lw=0.4, zorder=1)
        if labels:
            s = 0.9                                            # fraction along the visible segment
            x0 = (lo + s * (hi - lo - abs(c))) if c >= 0 else (lo - c + s * (hi - lo + c))
            ax.annotate(label, (x0 - w / 2, x0 + c + w / 2), xytext=(-1.0, 1.0), textcoords="offset points",
                        rotation=45, rotation_mode="anchor", ha="right", va="bottom", fontsize=t.size - 1.5,
                        color=t.ink, zorder=5)
    hits = {}
    for name, (f1, f2) in pts.items():
        hits[name] = [kind for kind, c, w, _ in windows if abs(f2 - f1 - c) < w]
        col = t.ink if hits[name] else t.sim
        ax.plot(f1, f2, **sim_kw(color=col, marker="X" if hits[name] else "o", ms=4.5, zorder=6))
        if spread:
            ax.add_patch(Circle((f1, f2), spread, fill=False, ec=col, ls=(0, (2, 1.5)), lw=0.6, zorder=5))
        k = (spread or 0.0) / np.sqrt(2)                           # leave the scatter circle along the gap
        ax.annotate(name, (f1 - k, f2 - k), xytext=(-2.5, -2.5), textcoords="offset points", rotation=45,
                    rotation_mode="anchor", ha="right", va="center", fontsize=t.size - 1, color=t.ink, zorder=6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(rf"$f_{{q1}}$ ({unit})")
    ax.set_ylabel(rf"$f_{{q2}}$ ({unit})")
    return hits


def capacitance_matrix(ax, C, labels, *, unit="fF", digits=3, cbar=True, fontsize=None):
    """The Maxwell capacitance matrix C (in unit) as annotated cells: total capacitance of each net on
    the diagonal, the negative mutual capacitances off it. Cells are coloured by |C| on a log scale,
    so couplings across several decades stay visible, and print the signed value; zero cells stay
    blank. Returns (image, colour bar or None)."""
    C = np.asarray(C, float)
    A = np.abs(C)
    norm = LogNorm(vmin=A[A > 0].min(), vmax=A.max())
    signed = [[cell_text(f"{{:.{digits}g}}", c) if c else "" for c in row] for row in C]   # the signed Maxwell entry
    return annotated_matrix(ax, A, labels, norm=norm, text=signed, cbar=cbar, label=rf"$|C_{{ij}}|$ ({unit})",
                            fontsize=fontsize, skip=lambda i, j, v: v == 0)


def participation(ax, p, tand, *, freq=None):
    """Surface-participation loss budget. p and tand map each region (metal-air MA, metal-substrate
    MS, substrate-air SA, substrate) to its participation ratio p_i and loss tangent tan(delta_i).
    Each region gets a bar of Q_i = 1 / (p_i tan(delta_i)) on a log axis, with p_i and tan(delta_i)
    printed above it and its share of the total loss at its end. The last bar is the total
    1/Q = sum_i p_i tan(delta_i); with freq (GHz) it also gives T1 = Q / (2 pi f).
    Returns {region: Q_i, ..., "total": Q}."""
    t = theme()
    names = list(p)
    Q = {n: 1.0 / (p[n] * tand[n]) for n in names}
    Qt = 1.0 / sum(1.0 / q for q in Q.values())
    rows = names + ["total"]
    vals = [Q[n] for n in names] + [Qt]
    x0 = 10.0 ** np.floor(np.log10(min(vals)) - 0.3)
    x1 = 10.0 ** np.ceil(np.log10(max(vals)) + 0.25)
    fc = [lighten(t.sim, 0.6)] * len(names) + [lighten(t.accent, 0.35)]
    ec = [t.sim] * len(names) + [t.accent]
    yb = np.arange(len(rows)) + 0.14
    ax.barh(yb, np.array(vals) - x0, left=x0, height=0.42, color=fc, edgecolor=ec, lw=0.6, zorder=2)
    ax.axvline(Qt, **ref_kw(zorder=1))
    fs = t.size - 1.5
    over = dict(ha="left", va="bottom", fontsize=fs, color=t.ink, zorder=3, bbox=dict(fc="white", ec="none", pad=0.4))
    for i, n in enumerate(names):
        ax.text(x0 * 1.12, i - 0.12, rf"$p$ = {sci(p[n])},  $\tan\delta$ = {sci(tand[n])}", **over)
        ax.text(Q[n] * 1.12, yb[i], f"{100 * Qt / Q[n]:.0f} %", ha="left", va="center", fontsize=fs, color=t.ink)
    ax.text(Qt * 1.12, yb[-1], rf"$Q$ = {sci(Qt)}", ha="left", va="center", fontsize=fs, color=t.ink)
    if freq:
        T1 = Qt / (2 * np.pi * freq * 1e3)                           # GHz -> microseconds
        ax.text(x0 * 1.12, len(names) - 0.12, rf"$T_1$ = {T1:.3g} µs at {freq:g} GHz", **over)
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_yticks(yb, rows)
    ax.set_ylim(len(rows) - 0.45, -0.6)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlabel(r"quality factor $Q_i = 1/(p_i \tan\delta_i)$")
    return {**Q, "total": Qt}


def dispersive_shift(ax, fq, fr, g, alpha, *, mark=None, scale="symlog", ylim=None, linthresh=None,
                     loc="upper left"):
    """Dispersive shift chi = g^2 alpha / (Delta (Delta + alpha)), Delta = fq - fr, against qubit
    frequency fq (GHz), from physics.dispersive_shift and drawn in MHz. The straddling regime
    0 < Delta < |alpha| is shaded and the poles at Delta = 0 and Delta = -alpha are marked; the curve
    is broken at each pole. The default symlog axis shows the small negative chi of a detuned qubit
    and the large positive chi inside the straddling regime together; it is linear within +-linthresh
    (default: the decade below |chi| at the end of the range farthest from fr). scale="linear" with
    ylim zooms on one branch. mark=fq0 marks a design point and prints its chi at loc.
    Returns chi (MHz) on fq."""
    t = theme()
    fq = np.asarray(fq, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = 1e3 * ph.dispersive_shift(fq, fr, g, alpha)
    far = abs(1e3 * ph.dispersive_shift(fq[np.argmax(np.abs(fq - fr))], fr, g, alpha))
    if ylim is None:
        top = 10.0 ** np.ceil(np.log10(8e3 * g ** 2 / abs(alpha))) if scale == "symlog" else 10 * far
        ylim = (-top, top)
    if scale == "symlog":
        lt = linthresh or 10.0 ** np.floor(np.log10(far))
        ax.set_yscale("symlog", linthresh=lt, linscale=0.5)
        dec = 10.0 ** np.arange(np.log10(lt) + 1, np.log10(max(abs(ylim[0]), abs(ylim[1]))) + 0.5)
        ax.yaxis.set_major_locator(FixedLocator(np.r_[-dec[::-1], 0.0, dec]))
        ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=lt, subs=np.arange(2, 10)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", _MINUS)))
    big = 3 * max(abs(ylim[0]), abs(ylim[1]))
    curve = np.where(np.abs(chi) > big, np.nan, chi)
    lo, hi = sorted((fr, fr - alpha))
    edge = ax.get_xaxis_transform()                                  # x in data, y in axes
    ax.axvspan(lo, hi, fc=lighten(t.accent, 0.72), ec="none", zorder=0)
    ax.text((lo + hi) / 2, 0.5, "straddling", transform=edge, rotation=90, ha="center", va="center",
            fontsize=t.size - 1.5, color=t.ink, bbox=dict(fc=lighten(t.accent, 0.72), ec="none", pad=0.5))
    ax.axhline(0, color=t.light, ls="-", lw=0.6, zorder=1)
    for f0, s, ha in ((fr, r"$\Delta = 0$ ", "right"), (fr - alpha, r" $\Delta = -\alpha$", "left")):
        ax.axvline(f0, **ref_kw())
        ax.text(f0, 0.97, s, transform=edge, ha=ha, va="top", fontsize=t.size - 1, color=t.ink)
    ax.plot(fq, curve, **model_kw())
    if mark is not None:
        c0 = 1e3 * ph.dispersive_shift(mark, fr, g, alpha)
        ax.plot(mark, c0, **sim_kw(color=t.accent, ms=4.5, zorder=5))
        note(ax, rf"$\chi/2\pi$ = {c0:.3g} MHz" "\n" rf"at $f_q$ = {mark:g} GHz".replace("-", _MINUS), loc=loc)
    ax.set_xlim(fq.min(), fq.max())
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"qubit frequency $f_q$ (GHz)")
    ax.set_ylabel(r"dispersive shift $\chi/2\pi$ (MHz)")
    return chi


def _duration(us):
    """A time in µs printed with two significant figures, in ms from 1000 µs."""
    if us >= 1e3:
        return f"{us / 1e3:.2g} ms" if us < 1e5 else f"{us / 1e3:.0f} ms"
    return f"{us:.2g} µs" if us < 100 else f"{us:.0f} µs"


def purcell_limit(ax, fq, fr, g, kappa, *, filter_q=None, mark=None, target=None, loc="lower left"):
    """Purcell-limited T1 against qubit frequency fq (GHz) on a log axis, from physics.purcell_t1, for
    a qubit coupled with strength g to a readout resonator at fr of linewidth kappa (all GHz).
    filter_q adds the curve with a bandpass Purcell filter of that quality factor. mark=fq0 marks the
    design point on each curve with its limit; target (µs) draws a guide at that T1. The resonator
    frequency is marked when it lies in range. Returns {"bare": T1, "filter": T1 or None} in µs."""
    t = theme()
    fq = np.asarray(fq, float)
    out = {"bare": ph.purcell_t1(fq, fr, g, kappa), "filter": None}
    curves = [("bare", "no filter", None)]
    if filter_q:
        out["filter"] = ph.purcell_t1(fq, fr, g, kappa, filter_q)
        curves.append(("filter", rf"Purcell filter, $Q_F$ = {filter_q:g}", filter_q))
    for i, (key, label, q) in enumerate(curves):
        ax.plot(fq, out[key], label=label, **{**series(i), "marker": None, "lw": t.lw})
        if mark is not None:
            T0 = float(ph.purcell_t1(mark, fr, g, kappa, q))
            ax.plot(mark, T0, **sim_kw(color=series(i)["color"], ms=4.5, zorder=5))
            ax.annotate(_duration(T0), (mark, T0), xytext=(5, 2), textcoords="offset points", ha="left",
                        va="bottom", fontsize=t.size - 1, color=t.ink)
    if target:
        ax.axhline(target, **ref_kw())
        ax.text(fq.min(), target, f" target {_duration(target)}", ha="left", va="bottom", fontsize=t.size - 1.5,
                color=t.reference)
    if fq.min() < fr < fq.max():
        ax.axvline(fr, **ref_kw())
    ax.set_yscale("log")
    ax.set_xlim(fq.min(), fq.max())
    ax.set_xlabel(r"qubit frequency $f_q$ (GHz)")
    ax.set_ylabel(r"Purcell limit $T_1$ (µs)")
    ax.legend(loc=loc)
    return out


def convergence(ax, passes, values, *, label="simulated value", tol=None, xlabel="adaptive pass",
                loc="upper right"):
    """A simulated quantity against adaptive pass (or mesh cutoff) on ax, and the relative change
    between successive passes |x_n - x_(n-1)| / |x_n| on a log axis to the right. tol draws the
    convergence target there and rings the first pass that meets it. The legend at loc names the two
    series. Returns {"rel": relative changes, "ax2": the right axes, "converged": that pass or None}."""
    t = theme()
    n, v = np.asarray(passes), np.asarray(values, float)
    rel = np.abs(np.diff(v)) / np.abs(v[1:])
    h1, = ax.plot(n, v, label="value", **sim_kw(ls="-", lw=0.8))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    h2, = ax2.plot(n[1:], rel, label="relative change",
                   **sim_kw(color=t.accent, marker="s", ms=3.5, ls=(0, (3, 1.5)), lw=0.8))
    ax2.set_yscale("log")
    done = None
    if tol:
        ax2.axhline(tol, **ref_kw())
        ax2.text(n[0], tol, " target", ha="left", va="bottom", fontsize=t.size - 1.5, color=t.reference)
        ok = np.nonzero(rel <= tol)[0]
        if ok.size:
            done = n[1:][ok[0]]
            ax2.plot(done, rel[ok[0]], **sim_kw(color=t.accent, marker="o", ms=8, mew=0.8, mfc="none", zorder=5))
            ax2.annotate(f"converged, pass {done}", (done, rel[ok[0]]), xytext=(6, 5), textcoords="offset points",
                         ha="left", va="bottom", fontsize=t.size - 1, color=t.ink)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label, color=t.sim)
    ax2.set_ylabel(r"relative change $|\delta x/x|$", color=t.accent)
    ax2.legend(handles=[h1, h2], loc=loc)                          # on ax2, so its lines stay beneath
    return {"rel": rel, "ax2": ax2, "converged": done}
