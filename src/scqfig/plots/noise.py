"""Noise and errors: spectral densities, error budgets, correlated-error bursts and correlations."""
import numpy as np

from .. import fit as F
from ..core import data_kw, log_minor, model_kw, note, pm, ref_kw, theme
from ..themes import lighten
from ._base import annotated_matrix, colorbar

__all__ = ["noise_psd", "error_budget", "burst_raster", "error_correlation"]

_HATCH = ("", "////", "....", "xxxx", "\\\\\\\\", "++")      # folio tells budget items apart in greyscale


# ---- a toy model of correlated error bursts ---------------------------------------------------

def burst_events(nq=24, nt=1000, p_bg=0.02, bursts=((220, 6), (690, 17)), tau=25.0, p_hit=0.9, reach=6.0, rng=0):
    """Error detections of nq qubits in a chain over nt time steps: a background probability p_bg per
    qubit and step, plus bursts (step, centre qubit) that raise it to p_hit at the centre, falling off
    as exp(-distance/reach) along the chain and recovering as exp(-(t - t0)/tau). This is the shape
    of radiation-induced bursts (McEwen et al. 2022), for demos and tests. Returns (steps, E) with E
    boolean of shape (nq, nt)."""
    g = np.random.default_rng(rng)
    t = np.arange(nt)
    q = np.arange(nq)[:, None]
    p = np.full((nq, nt), float(p_bg))
    for t0, qc in bursts:
        decay = np.exp(-np.clip(t - t0, 0, None) / tau) * (t >= t0)
        p = p + (p_hit - p_bg) * np.exp(-np.abs(q - qc) / reach) * decay[None, :]
    return t, g.random((nq, nt)) < np.clip(p, 0, 1)


# ---- plots ------------------------------------------------------------------------------------

def noise_psd(ax, f, S, *, fit=True, fit_range=None, white=None, amp_unit=r"$\mu\Phi_0/\sqrt{\mathrm{Hz}}$",
              xlabel="frequency (Hz)", ylabel=r"$S_\Phi$ ($\mu\Phi_0^2$/Hz)", loc="upper right"):
    """Noise spectral density on log-log axes with a power-law fit S = A / f^alpha over fit_range (all
    points by default) and an optional white floor drawn dashed. The note gives the amplitude at
    1 Hz, sqrt(A), in amp_unit and the exponent. Returns the Fit, with derived["amplitude"] = sqrt(A)."""
    f, S = np.asarray(f, float), np.asarray(S, float)
    ax.plot(f, S, **data_kw(ms=2.6))
    r = None
    if fit:
        lo, hi = fit_range or (f.min(), f.max())
        sel = (f >= lo) & (f <= hi)
        r = F.power_law(f[sel], S[sel])
        amp, amp_err = np.sqrt(r["A"]), 0.5 * r.errors["A"] / np.sqrt(r["A"])
        r.derived["amplitude"] = amp
        ff = np.geomspace(lo, hi, 200)
        ax.plot(ff, r(ff), **model_kw())
        note(ax, rf"$\sqrt{{A}}$ = {pm(amp, amp_err)} {amp_unit}" "\n" rf"$\alpha$ = {pm(r['alpha'], r.errors['alpha'])}",
             loc=loc)
    if white is not None:
        ax.axhline(white, **ref_kw())
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_minor(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return r


def error_budget(ax, items, *, measured=None, unit="%", xlabel=None, digits=2):
    """Error budget as horizontal bars: one bar per contribution (largest first), then the budget as
    one stacked bar of the same segments, then the measured error with the unexplained part hatched.
    items maps a label to an error in `unit`. Folio adds hatches so segments read in greyscale.
    Returns {"budget": sum, "unexplained": measured - sum or None, "bars": [...]}."""
    t = theme()
    names = sorted(items, key=items.get, reverse=True)
    vals = np.array([items[n] for n in names], float)
    total = vals.sum()
    rows = names + ["budget"] + (["measured"] if measured is not None else [])
    y = -np.arange(len(rows), dtype=float)
    y[len(names):] -= 0.4                                     # a small gap before the totals
    fs = t.size - 0.5
    bars = []
    for k, (n, v) in enumerate(zip(names, vals)):
        c = t.cycle[k % len(t.cycle)]
        hatch = _HATCH[k % len(_HATCH)] if t.name == "folio" else ""
        b = ax.barh(y[k], v, height=0.62, color=lighten(c, 0.55), edgecolor=c, lw=0.7, hatch=hatch)
        bars.append(b)
        ax.text(v + 0.012 * total, y[k], f"{v:.{digits}g}", va="center", ha="left", fontsize=fs)
        ax.barh(y[len(names)], v, left=vals[:k].sum(), height=0.62, color=lighten(c, 0.55), edgecolor=c, lw=0.7,
                hatch=hatch)
    yb = y[len(names)]
    ax.text(total + 0.012 * total, yb, f"{total:.{digits}g}", va="center", ha="left", fontsize=fs, weight="bold")
    gap = None
    if measured is not None:
        ym = y[-1]
        gap = measured - total
        ax.barh(ym, min(total, measured), height=0.62, color=t.light, edgecolor=t.ink, lw=0.7)
        if gap > 0:
            ax.barh(ym, gap, left=total, height=0.62, color="white", edgecolor=t.ink, lw=0.7, hatch="////")
        ax.text(measured + 0.012 * total, ym, f"{measured:.{digits}g}", va="center", ha="left", fontsize=fs,
                weight="bold")
        if gap > 0:
            ax.text(measured + 0.012 * total, ym - 0.42, f"{gap:.{digits}g} unexplained", va="top", ha="left",
                    fontsize=fs - 0.5, color=t.reference)
    ax.set_yticks(y, rows)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.3 * max(total, measured or 0))
    ax.set_ylim(y[-1] - (0.95 if gap and gap > 0 else 0.6), 0.6)
    ax.set_xlabel(xlabel or f"error ({unit})")
    return {"budget": total, "unexplained": gap, "bars": bars}


def burst_raster(ax, t, E, *, ax_frac=None, threshold=None, min_gap=None, unit="ms", ylabel="qubit", mark=True):
    """Correlated errors over time: each detection of qubit q at time t as a small ink square, and,
    on ax_frac when given, the fraction of qubits flagged per step with a threshold line (default: mean
    plus six standard deviations of the quietest 95 % of steps). A crossing marks a burst, and later
    crossings within min_gap (default 5 % of the time span) count as the same burst. Each burst gets
    an orange band on the raster and an orange star on the fraction trace. Returns the burst times."""
    tc = theme()
    t, E = np.asarray(t, float), np.asarray(E, bool)
    q, k = np.nonzero(E)
    ax.plot(t[k], q, ls="none", marker="s", ms=1.1, mew=0, color=tc.data, rasterized=True, zorder=3)
    frac = E.mean(axis=0)
    quiet = frac[frac <= np.quantile(frac, 0.95)]
    thr = threshold if threshold is not None else quiet.mean() + 6 * quiet.std()
    hot = frac > thr
    gap = 0.05 * np.ptp(t) if min_gap is None else min_gap
    starts = []
    for s in t[np.flatnonzero(hot & ~np.r_[False, hot[:-1]])]:
        if not starts or s - starts[-1] > gap:
            starts.append(s)
    starts = np.array(starts)
    if mark:
        for s in starts:
            ax.axvspan(s - 0.004 * np.ptp(t), s + 0.03 * np.ptp(t), color=lighten(tc.accent, 0.6), lw=0, zorder=1)
    ax.set_ylim(-0.8, E.shape[0] - 0.2)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f"time ({unit})")
    if ax_frac is not None:
        ax_frac.plot(t, frac, color=tc.ink, lw=0.6, ls="-")
        ax_frac.axhline(thr, **ref_kw())
        if mark:
            for s in starts:
                i = np.searchsorted(t, s)
                peak = frac[i:i + 5].max()
                ax_frac.plot([s], [peak], ls="none", marker="*", ms=8, mfc=tc.accent, mec=tc.ink, mew=0.5, zorder=5)
        ax_frac.set_ylim(0, 1.08)
        ax_frac.set_ylabel("flagged")
        ax_frac.tick_params(labelbottom=False)
    return starts


def error_correlation(ax, E, labels=None, *, annotate=None, cbar=True, label="correlation"):
    """Pairwise Pearson correlation of error detections E (qubits x time) as a matrix, the diagonal
    left blank. Cells carry their value when annotate is True (default for up to 8 qubits).
    Returns (image, colour bar or None)."""
    E = np.asarray(E, float)
    C = np.corrcoef(E)
    np.fill_diagonal(C, np.nan)
    n = C.shape[0]
    labels = labels or [f"Q{i + 1}" if n <= 12 else (str(i + 1) if i % 4 == 0 else "") for i in range(n)]
    show = n <= 8 if annotate is None else annotate
    vmax = np.nanmax(C)
    im, cb = annotated_matrix(ax, np.ma.masked_invalid(C), labels, fmt="{:.2f}", vmin=0, vmax=vmax, cbar=False,
                              text=None if show else [[""] * n for _ in range(n)])
    if cbar:
        cb = colorbar(ax, im, label)
    ax.tick_params(labelsize=theme().size - 1)
    return im, cb if cbar else None

