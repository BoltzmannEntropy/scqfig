"""Time-domain characterisation: decays, oscillations and benchmarking, each with its fit drawn and
the fitted value annotated. Also holds the TLS-limited T1 model behind the TLS map."""
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import blended_transform_factory

from .. import fit as F
from ..core import data_kw, model_kw, note, pm, ref_kw, series, theme
from ..themes import lighten
from ._base import heatmap

__all__ = ["t1_decay", "rabi", "chevron", "ramsey", "coherence_decays", "rb", "coherence_timeseries", "tls_map"]

# frequency unit that matches a time unit, for detunings read off a fit in cycles per time unit
_RATE = {"ns": "GHz", "µs": "MHz", "us": "MHz", "ms": "kHz", "s": "Hz"}


def _points(ax, x, y, err=None, **kw):
    if err is None:
        return ax.plot(x, y, **data_kw(**kw))
    return ax.errorbar(x, y, yerr=err, elinewidth=0.6, **data_kw(**kw))


# ---- models -----------------------------------------------------------------------------------

def pi_amplitude(r):
    """Drive amplitude of the first maximum at x > 0 of a cosine Fit (fit.cosine): the pi pulse."""
    A, f, phi = r["A"], r["f"], r["phi"]
    if A < 0:
        phi += np.pi
    if f < 0:
        f, phi = -f, -phi
    x = np.mod(-phi, 2 * np.pi) / (2 * np.pi * f)
    return x if x > 1e-12 else 1.0 / f


def interleaved_error(p, p_int, d=2, p_err=None, p_int_err=None):
    """Gate error from interleaved RB, r = (1 - p_int/p)(d - 1)/d, with its one-sigma error from the
    independent errors of the two depolarising parameters. Returns (r, error or None)."""
    r = (1 - p_int / p) * (d - 1) / d
    if p_err is None or p_int_err is None:
        return r, None
    e = (d - 1) / d * np.hypot(p_int_err / p, p_int * p_err / p ** 2)
    return r, e


def tls_t1(freq, tls_freq, g, gamma, t1_bg):
    """T1 in µs of a qubit at freq (GHz) coupled to two-level defects at tls_freq (GHz). Each defect
    adds the rate 2 g^2 gamma / (gamma^2 + Delta^2) in angular units, Delta = 2 pi (freq - f_TLS), on
    top of the background 1/t1_bg (µs). g/2pi is the coupling and gamma/2pi the defect's decoherence
    rate, both in MHz. tls_freq, g and gamma broadcast along a last axis that runs over defects, and
    freq broadcasts against the axes before it, so a frequency grid of shape (nf, 1) and defect
    tracks of shape (nt, K) give T1 of shape (nf, nt)."""
    w = 2 * np.pi * 1e3 * (np.asarray(freq, float)[..., None] - np.asarray(tls_freq, float))   # rad/µs
    G, Gm = 2 * np.pi * np.asarray(g, float), 2 * np.pi * np.asarray(gamma, float)
    return 1.0 / (1.0 / t1_bg + np.sum(2 * G ** 2 * Gm / (Gm ** 2 + w ** 2), axis=-1))


# ---- plots ------------------------------------------------------------------------------------

def t1_decay(ax, t, p, *, err=None, fit=True, unit="µs", ylabel=r"excited population $P_1$", loc="upper right"):
    """Energy relaxation: measured P1 against delay with an exponential fit and T1 annotated.
    Returns the Fit (or None)."""
    _points(ax, t, p, err)
    r = None
    if fit:
        r = F.exp_decay(t, p, sigma=err)
        tt = np.linspace(0, np.max(t), 400)
        ax.plot(tt, r(tt), **model_kw())
        note(ax, rf"$T_1$ = {pm(r['T'], r.errors['T'], unit)}", loc=loc)
    ax.set_xlabel(f"delay ({unit})")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    return r


def rabi(ax, amp, p, *, err=None, fit=True, xlabel="drive amplitude (arb. units)",
         ylabel=r"excited population $P_1$", symbol=r"A_\pi", digits=3):
    """Rabi oscillation against drive amplitude at fixed pulse length, with a cosine fit. The
    pi-pulse amplitude (the first maximum of the fit) gets a guide line and its value, printed on the
    line where the oscillation leaves room. Returns the Fit with derived["amp_pi"] (or None)."""
    _points(ax, amp, p, err)
    r = None
    if fit:
        r = F.cosine(amp, p, sigma=err)
        a_pi = pi_amplitude(r)
        r.derived["amp_pi"] = a_pi
        x = np.linspace(np.min(amp), np.max(amp), 600)
        ax.plot(x, r(x), **model_kw())
        ax.axvline(a_pi, **ref_kw())
        ax.text(a_pi, 0.5, rf"${symbol}$ = {a_pi:.{digits}g}", transform=blended_transform_factory(ax.transData,
                ax.transAxes), ha="center", va="center", fontsize=theme().size - 0.5, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.2))
    ax.set_xlim(np.min(amp), np.max(amp))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return r


def chevron(ax, detuning, duration, P, *, rabi=None, orders=2, model_side="left", scale=1e3, cbar=True,
            label=r"excited population $P_1$", xlabel=r"detuning $\Delta/2\pi$ (MHz)",
            ylabel="pulse duration (ns)"):
    """Rabi chevron: excited population (shape len(duration) x len(detuning)) against drive detuning
    and pulse duration as a heatmap. Given the Rabi rate (in the detuning's unit), the population
    maxima t_k = (k + 1/2) / sqrt(rabi^2 + Delta^2) for k < orders are drawn over the left half
    (model_side="left"; "right" or "both"); scale converts one over the detuning unit into the
    duration unit (1e3 for MHz and ns). Returns (mesh, colorbar or None)."""
    t = theme()
    mesh, cb = heatmap(ax, detuning, duration, P, label=label, cbar=cbar, vmin=0, vmax=1)
    d = np.asarray(detuning, float)
    if rabi is not None:
        mid = 0.5 * (d.min() + d.max())
        keep = {"left": d <= mid, "right": d >= mid, "both": np.ones(d.size, bool)}[model_side]
        dd = np.linspace(d[keep].min(), d[keep].max(), 400)
        for k in range(orders):
            ax.plot(dd, (k + 0.5) / np.hypot(rabi, dd) * scale, color=t.overlay, lw=0.9,
                    ls=(0, (4, 2)) if k == 0 else (0, (1, 1.5)))
    ax.set_xlim(d.min(), d.max())
    ax.set_ylim(np.min(duration), np.max(duration))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mesh, cb


def ramsey(ax, t, p, *, err=None, fit=True, envelope=True, unit="µs", ylabel=r"excited population $P_1$",
           loc="upper right"):
    """Ramsey fringes with a damped-cosine fit, its decay envelope dashed, and T2* and the detuning
    annotated (the detuning in the frequency unit that matches the time unit). Returns the Fit (or
    None)."""
    _points(ax, t, p, err, ms=3)
    r = None
    if fit:
        r = F.damped_cosine(t, p, sigma=err)
        tt = np.linspace(0, np.max(t), 1200)
        if envelope:
            env = r["A"] * np.exp(-tt / r["T"])
            ax.plot(tt, r["B"] + env, **ref_kw())
            ax.plot(tt, r["B"] - env, **ref_kw())
        ax.plot(tt, r(tt), **model_kw())
        rate = _RATE.get(unit, f"1/{unit}")
        note(ax, rf"$T_2^*$ = {pm(r['T'], r.errors['T'], unit)}" "\n"
                 rf"$\Delta/2\pi$ = {pm(abs(r['f']), r.errors['f'], rate)}", loc=loc)
    ax.set_xlabel(f"free evolution ({unit})")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    return r


def coherence_decays(ax, curves, *, fit=True, unit="µs", ylabel="coherence (normalised)", loc="upper right",
                     ms=3):
    """Several decays on one axis, such as the Ramsey envelope, a Hahn echo and CPMG-N. The i-th
    series takes the colour, marker and line style of series(i) and gets an exponential fit, whose
    T goes into its legend entry. curves maps a name ending in the time constant's symbol, such as
    r"Hahn echo $T_{2E}$", to (t, y) or (t, y, err). Returns {name: Fit}."""
    out, handles, names = {}, [], []
    for i, (name, c) in enumerate(curves.items()):
        x, y = np.asarray(c[0], float), np.asarray(c[1], float)
        e = c[2] if len(c) > 2 else None
        s = series(i)
        pts = _points(ax, x, y, e, color=s["color"], marker=s["marker"], ms=ms)
        h = pts[0]
        label = name
        if fit:
            r = F.exp_decay(x, y, sigma=e)
            xx = np.linspace(0, x.max(), 400)
            (line,) = ax.plot(xx, r(xx), **model_kw(color=s["color"], ls=s["ls"]))
            h = (line, h)
            label = f"{name} = {pm(r['T'], r.errors['T'], unit)}"
            out[name] = r
        handles.append(h)
        names.append(label)
    ax.legend(handles, names, loc=loc, handler_map={tuple: HandlerTuple(ndivide=1)}, handlelength=2.4)
    ax.set_xlabel(f"delay ({unit})")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    return out


def rb(ax, m, seqs, *, interleaved=None, d=2, labels=("standard", "interleaved"), gate=r"\mathrm{gate}",
       faint=True, loc="lower left", legend_loc="upper right", ylabel=r"survival probability $P_0$"):
    """Randomized benchmarking. seqs (and interleaved) hold one row of survival probabilities per
    random sequence, one column per length m. Each sequence is drawn faintly, the mean per length in
    the marker of series(i) with an RB fit A p^m + B (fit.rb, weighted by the standard error).
    Annotates the error per Clifford and, with interleaved data, the gate error
    r = (1 - p_int/p)(d - 1)/d. Returns {"standard": Fit, "interleaved": Fit, "gate_error": (r, err)}."""
    m = np.asarray(m, float)
    out, handles = {}, []
    sets = [("standard", seqs)] + ([("interleaved", interleaved)] if interleaved is not None else [])
    for i, (key, S) in enumerate(sets):
        S = np.atleast_2d(np.asarray(S, float))
        s = series(i)
        if faint:
            ax.plot(m, S.T, ls="none", marker=".", ms=1.6, color=lighten(s["color"], 0.6), zorder=1)
        mean = S.mean(axis=0)
        sem = S.std(axis=0, ddof=1) / np.sqrt(S.shape[0]) if S.shape[0] > 1 else None
        r = F.rb(m, mean, sigma=sem, d=d)
        mm = np.linspace(0, m.max(), 400)
        (line,) = ax.plot(mm, r(mm), **model_kw(color=s["color"], ls=s["ls"]))
        (pts,) = ax.plot(m, mean, **data_kw(color=s["color"], marker=s["marker"], ms=3.5))
        handles.append((line, pts))
        out[key] = r
    std = out["standard"]
    lines = [rf"$r_\mathrm{{Clif}}$ = {pm(100 * std['epc'], 100 * std['epc_err'], '%')}"]
    if interleaved is not None:
        ri = out["interleaved"]
        g, ge = interleaved_error(std["p"], ri["p"], d, std.errors["p"], ri.errors["p"])
        out["gate_error"] = (g, ge)
        lines.append(rf"$r_{{{gate}}}$ = {pm(100 * g, 100 * ge, '%')}")
    note(ax, "\n".join(lines), loc=loc)
    ax.legend(handles, labels[:len(handles)], loc=legend_loc, handler_map={tuple: HandlerTuple(ndivide=1)},
              handlelength=2.4)
    ax.set_xlim(0, m.max() * 1.02)
    ax.set_xlabel(r"sequence length $m$ (Cliffords)")
    ax.set_ylabel(ylabel)
    return out


def coherence_timeseries(ax, ax_hist, time, T1, *, T2=None, unit="µs", bins=24,
                         labels=(r"$T_1$", r"$T_{2E}$"), xlabel="time (h)", ms=2.2):
    """T1 (and optionally T2) against time on ax, with their histograms on ax_hist, a narrow axes to
    the right that shares the y axis (make the pair with sharey=True). The legend sits above the
    frame, clear of the data, and gives each mean and standard deviation. Returns
    {"T1": (mean, sd), "T2": (mean, sd)}."""
    data = [("T1", labels[0], np.asarray(T1, float))]
    if T2 is not None:
        data.append(("T2", labels[1], np.asarray(T2, float)))
    allv = np.concatenate([v for _, _, v in data])
    edges = np.histogram_bin_edges(allv[np.isfinite(allv)], bins)
    out, handles, names = {}, [], []
    for i, (key, name, v) in enumerate(data):
        s = series(i)
        (h,) = ax.plot(time, v, **data_kw(color=s["color"], marker=s["marker"], ms=ms, ls=s["ls"], lw=0.45,
                                          mew=0.6))
        ok = v[np.isfinite(v)]
        ax_hist.hist(ok, bins=edges, orientation="horizontal", histtype="stepfilled",
                     fc=lighten(s["color"], 0.72), ec=s["color"], ls=s["ls"], lw=0.7, alpha=0.75 if i else 1.0)
        mean, sd = ok.mean(), ok.std(ddof=1)
        out[key] = (mean, sd)
        handles.append(h)
        names.append(f"{name} = {pm(mean, sd, unit)}")
    ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(handles), frameon=False,
              borderaxespad=0, columnspacing=1.2, handletextpad=0.4, handlelength=1.6, title="mean ± s.d.",
              alignment="left")
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"coherence time ({unit})")
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_xlim(left=0)
    ax_hist.set_xlabel("count")
    return out


def tls_map(ax, time, freq, T1, *, cmap=None, vmin=None, vmax=None, cbar=True, label=r"$T_1$ (µs)",
            xlabel="time (h)", ylabel="qubit frequency (GHz)"):
    """T1 spectroscopy over time: T1 (shape len(freq) x len(time)) against time and qubit frequency.
    Two-level defects show as lines of short T1 that wander in frequency (spectral diffusion). The
    map runs reversed so short T1 is dark, unless cmap is given. Returns (mesh, colorbar or None)."""
    t = theme()
    mesh, cb = heatmap(ax, time, freq, T1, label=label, cbar=cbar, cmap=cmap or t.cmap + "_r", vmin=vmin,
                       vmax=vmax)
    ax.set_xlim(np.min(time), np.max(time))
    ax.set_ylim(np.min(freq), np.max(freq))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mesh, cb
