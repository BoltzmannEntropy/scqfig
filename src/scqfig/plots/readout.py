"""Resonators and readout: S21 magnitude, phase and circle, quality factor against power, dispersive
response, single-shot IQ blobs, histograms and assignment matrices."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import FixedLocator

from .. import fit as F
from .. import physics as ph
from ..core import data_kw, log_minor, model_kw, note, ref_kw, sci, sim_kw, theme
from ..schematics.symbols import dim
from ..themes import lighten
from ._base import annotated_matrix

__all__ = ["s21_magnitude", "s21_phase", "s21_circle", "quality_vs_power", "dispersive_readout", "iq_blobs",
           "readout_histogram", "assignment_matrix"]


# ---- models and fits behind the readout plots ---------------------------------------------------

def readout_shots(level, centres, sigma, n, *, t_ro=1.0, T1=30.0, p_th=0.0, rng=None):
    """Simulated integrated single-shot IQ values (complex, in the unit of centres) of a transmon
    prepared in `level`. centres[m] is the noiseless signal of level m, and Gaussian noise of standard
    deviation sigma is added to each quadrature. During the integration time t_ro (µs) level m decays
    to m - 1 at rate m/T1 (T1 in µs), so a shot that decays lands between the centres, weighted by the
    time spent in each level. With probability p_th a preparation of |0> or |1> ends in the other
    state (thermal population or a failed pi pulse). Returns n complex values."""
    rng = np.random.default_rng() if rng is None else rng
    c = np.asarray(centres, complex)
    lvl = np.full(n, int(level))
    if p_th and level in (0, 1):
        lvl[rng.random(n) < p_th] = 1 - level
    t = np.zeros(n)
    acc = np.zeros(n, complex)
    for _ in range(int(lvl.max())):
        up = lvl > 0
        dt = np.full(n, np.inf)
        dt[up] = rng.exponential(T1 / lvl[up])
        rem = t_ro - t
        step = np.minimum(dt, rem)
        acc += c[lvl] * step
        t += step
        lvl = np.where(up & (dt < rem), lvl - 1, lvl)
    acc += c[lvl] * (t_ro - t)
    return acc / t_ro + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def iq_discriminator(shots):
    """Gaussian centre and covariance of each prepared state's single-shot IQ cloud, and the linear
    discriminant between the first two. shots holds one complex array (I + iQ) per prepared state.
    Each centre and covariance come from the cloud's core, the points within three standard deviations
    (iterated, then corrected for the clipping), so shots that decayed or were mis-prepared do not drag
    them. A shot s is assigned to the second state when Re(conj(w) s) > threshold, w being the unit
    normal of the boundary. Returns {"centres", "covs", "w", "threshold"}."""
    r2 = 9.0
    e = np.exp(-r2 / 2)
    corr = 2.0 / (2.0 - r2 * e / (1 - e))                         # chi^2(2) truncated at r2
    centres, covs = [], []
    for s in shots:
        s = np.asarray(s, complex)
        X = np.c_[s.real, s.imag]
        mu, S = np.median(X, axis=0), np.cov(X.T)
        for _ in range(30):
            D = X - mu
            keep = np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D) < r2
            mu, S = X[keep].mean(axis=0), np.cov(X[keep].T) * corr
        centres.append(mu[0] + 1j * mu[1])
        covs.append(S)
    d = {"centres": np.array(centres), "covs": covs, "w": None, "threshold": None}
    if len(shots) >= 2:
        dmu = np.array([(centres[1] - centres[0]).real, (centres[1] - centres[0]).imag])
        wv = np.linalg.solve(0.5 * (covs[0] + covs[1]), dmu)
        wv /= np.linalg.norm(wv)
        d["w"] = wv[0] + 1j * wv[1]
        d["threshold"] = float(np.real(np.conj(d["w"]) * (centres[0] + centres[1]) / 2))
    return d


# ---- resonator scattering ---------------------------------------------------------------------

def _resonator(f, s21, fit, calibrate):
    """(data, fine grid, model on it, Fit or None) of a notch resonator. With calibrate the fitted
    environment a e^{i alpha} e^{-2 pi i f tau} is divided out of data and model alike."""
    f, s = np.asarray(f, float), np.asarray(s21, complex)
    r = F.notch(f, s) if fit is True else (fit or None)
    if r is None:
        return s, None, None, None
    ff = np.linspace(f.min(), f.max(), 1500)
    if not calibrate:
        return s, ff, r(ff), r
    s = s / (r["a"] * np.exp(1j * r["alpha"]) * np.exp(-2j * np.pi * f * r["tau"]))
    return s, ff, ph.notch_s21(ff, r["fr"], r["Ql"], r["Qc"], r["phi"]), r


def _centre(f, s, r):
    return r["fr"] if r is not None else np.asarray(f, float)[np.argmin(np.abs(s))]


def _qnote(ax, r, loc):
    if r is None or loc is None:
        return None
    return note(ax, "\n".join([rf"$f_r$ = {r['fr']:.5f} GHz", rf"$Q_l$ = {sci(r['Ql'], r.errors['Ql'])}",
                               rf"$Q_c$ = {sci(r['Qc'], r.errors['Qc'])}", rf"$Q_i$ = {sci(r['Qi'], digits=3)}"]),
                loc=loc)


def s21_magnitude(ax, f, s21, *, fit=True, calibrate=True, db=False, loc="lower right"):
    """|S21| of a notch-coupled resonator against detuning f - f_r in MHz (f in GHz): measured points
    and the fit.notch model. fit=True fits, a Fit reuses one (so the magnitude, phase and circle panels
    share it), False draws the data alone. calibrate divides out the fitted cable delay, amplitude and
    phase, so the response sits at 1 off resonance. Prints f_r, Q_l, Q_c and Q_i at loc (None for no
    note). Returns the Fit."""
    s, ff, m, r = _resonator(f, s21, fit, calibrate)
    f0 = _centre(f, s, r)
    mag = (lambda z: 20 * np.log10(np.abs(z))) if db else np.abs
    x = (np.asarray(f, float) - f0) * 1e3
    ax.plot(x, mag(s), **data_kw(ms=2.5))
    if r is not None:
        ax.plot((ff - f0) * 1e3, mag(m), **model_kw(zorder=5))
    _qnote(ax, r, loc)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(r"detuning $f - f_r$ (MHz)")
    ax.set_ylabel(r"$|S_{21}|$ (dB)" if db else r"$|S_{21}|$")
    return r


def s21_phase(ax, f, s21, *, fit=True, calibrate=True, loc=None):
    """Unwrapped phase of S21 against detuning f - f_r in MHz, with the fit.notch model; fit,
    calibrate and loc as in s21_magnitude. Returns the Fit."""
    s, ff, m, r = _resonator(f, s21, fit, calibrate)
    f0 = _centre(f, s, r)
    x = (np.asarray(f, float) - f0) * 1e3
    ax.plot(x, np.unwrap(np.angle(s)), **data_kw(ms=2.5))
    if r is not None:
        ax.plot((ff - f0) * 1e3, np.unwrap(np.angle(m)), **model_kw(zorder=5))
    _qnote(ax, r, loc)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(r"detuning $f - f_r$ (MHz)")
    ax.set_ylabel(r"arg $S_{21}$ (rad)")
    return r


def s21_circle(ax, f, s21, *, fit=True, calibrate=True, loc=None):
    """S21 in the complex plane with the fit.notch circle. The off-resonant point and the resonance are
    joined by the dashed diameter, whose length is Q_l/Q_c; the resonance is marked. fit, calibrate and
    loc as in s21_magnitude. Returns the Fit."""
    t = theme()
    s, ff, m, r = _resonator(f, s21, fit, calibrate)
    ax.plot(s.real, s.imag, **data_kw(ms=2.5))
    if r is not None:
        ax.plot(m.real, m.imag, **model_kw(zorder=5))
        far = 1.0 if calibrate else r["a"] * np.exp(1j * r["alpha"]) * np.exp(-2j * np.pi * r["fr"] * r["tau"])
        res = far * ph.notch_s21(r["fr"], r["fr"], r["Ql"], r["Qc"], r["phi"])
        ax.plot([np.real(far), np.real(res)], [np.imag(far), np.imag(res)], **ref_kw())
        ax.plot(np.real(res), np.imag(res), **sim_kw(color=t.accent, ms=4.5, zorder=6))
    _qnote(ax, r, loc)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"Re $S_{21}$")
    ax.set_ylabel(r"Im $S_{21}$")
    return r


def quality_vs_power(ax, nbar, qi, *, err=None, fit=True, loc="lower right"):
    """Internal quality factor against mean photon number on log-log axes, with a fit of the
    saturable-TLS model physics.tls_qi (fit.tls; a Fit may be passed instead, False for none) and its
    parameters printed at loc. Returns the Fit."""
    if err is None:
        ax.plot(nbar, qi, **data_kw())
    else:
        ax.errorbar(nbar, qi, yerr=err, elinewidth=0.6, **data_kw())
    r = F.tls(nbar, qi, err) if fit is True else (fit or None)
    if r is not None:
        nn = np.logspace(np.log10(np.min(nbar)) - 0.3, np.log10(np.max(nbar)) + 0.3, 400)
        ax.plot(nn, r(nn), **model_kw())
        if loc:
            e = r.errors
            note(ax, "\n".join([rf"$F\delta_\mathrm{{TLS}}$ = {sci(r['F_delta'], e['F_delta'])}",
                                rf"$n_c$ = {sci(r['nc'], e['nc'])}", rf"$\beta$ = {sci(r['beta'], e['beta'])}",
                                rf"$Q_\mathrm{{other}}$ = {sci(r['Q_other'], e['Q_other'])}"]), loc=loc)
        ax.set_xlim(nn[0], nn[-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_minor(ax, "x")
    ax.set_xlabel(r"mean photon number $\langle n \rangle$")
    ax.set_ylabel(r"internal quality factor $Q_i$")
    return r


# ---- qubit-state readout ----------------------------------------------------------------------

def dispersive_readout(ax, f, s0, s1, *, f_ro=None, fit=True, labels=(r"$|0\rangle$", r"$|1\rangle$"),
                       loc="lower left"):
    """Resonator transmission |S21| against frequency with the qubit in |0> and in |1> (f in GHz; s0,
    s1 complex or magnitudes), drawn as detuning from the readout frequency f_ro in MHz. A Lorentzian
    fit to each |S21|^2 (fit=True) locates the two resonances, whose separation 2 chi is marked with an
    arrow; f_ro (default midway) gets a guide line. The legend sits at loc. Returns {"f0", "f1",
    "chi", "f_ro"} in GHz."""
    t = theme()
    f = np.asarray(f, float)
    mags = [np.abs(np.asarray(s)) for s in (s0, s1)]
    fits = [F.lorentzian(f, a ** 2) for a in mags] if fit else [None, None]
    res = [r["f0"] if r is not None else f[np.argmin(a)] for r, a in zip(fits, mags)]
    f_ro = 0.5 * (res[0] + res[1]) if f_ro is None else f_ro
    x = (f - f_ro) * 1e3
    ff = np.linspace(f.min(), f.max(), 1500)
    for m, (a, r) in enumerate(zip(mags, fits)):
        c = t.levels[m % len(t.levels)]
        ax.plot(x, a, **data_kw(color=c, marker=t.markers[m], ms=2.5))
        if r is not None:
            ax.plot((ff - f_ro) * 1e3, np.sqrt(np.clip(r(ff), 0, None)), label=labels[m],
                    **model_kw(color=c, ls=t.dashes[m]))
        else:
            ax.plot([], [], label=labels[m], **data_kw(color=c, marker=t.markers[m]))
    lo = min(float(np.min(a)) for a in mags)
    top = max(float(np.max(a)) for a in mags)
    y = lo - 0.07 * (top - lo)
    dim(ax, ((res[0] - f_ro) * 1e3, y), ((res[1] - f_ro) * 1e3, y), "")
    chi = 0.5 * (res[1] - res[0])
    ax.text(0, y - 0.04 * (top - lo), rf"$2|\chi|/2\pi$ = {2e3 * abs(chi):.2f} MHz", ha="center", va="top",
            fontsize=t.size - 1, color=t.ink, zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.axvline(0, **ref_kw())
    ax.text(0, 0.97, r" $f_\mathrm{ro}$", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=t.size - 0.5, color=t.ink)
    ax.legend(loc=loc)
    note(ax, rf"$f_\mathrm{{ro}}$ = {f_ro:.4f} GHz", loc="lower right")
    ax.set_xlim(x.min(), x.max())
    ymax = top + 0.16 * (top - lo)
    ax.set_ylim(y - 0.2 * (top - lo), ymax)
    ticks = ax.yaxis.get_major_locator().tick_values(0, ymax)          # the arrow sits below the dips;
    ax.yaxis.set_major_locator(FixedLocator([v for v in ticks if 0 <= v <= ymax]))   # no ticks below zero
    ax.set_xlabel(r"detuning $f - f_\mathrm{ro}$ (MHz)")
    ax.set_ylabel(r"$|S_{21}|$")
    return {"f0": res[0], "f1": res[1], "chi": chi, "f_ro": f_ro}


def _state_labels(labels, n):
    return list(labels) if labels is not None else [rf"$|{k}\rangle$" for k in range(n)]


def iq_blobs(ax, shots, *, labels=None, sigmas=(1, 2), line=True, unit="mV", ms=1.2, alpha=0.3,
             loc="lower left"):
    """Single-shot IQ clouds, one per prepared state (shots: complex arrays I + iQ), in the level
    colours, with the 1 sigma and 2 sigma contours of each cloud's Gaussian core and the discrimination
    line between the first two states (iq_discriminator). Each cloud is labelled on its far side and a
    legend at loc (None for none) keys the contours and the line. The points are rasterized, so a
    vector PDF stays small. Returns the iq_discriminator dict."""
    t = theme()
    d = iq_discriminator(shots)
    labels = _state_labels(labels, len(shots))
    styles = list(zip(sigmas, ("-", (0, (3, 1.5)), ":")))
    mid = np.mean(d["centres"])
    for m, s in enumerate(shots):
        c = t.levels[m % len(t.levels)]
        s = np.asarray(s, complex)
        ax.plot(s.real, s.imag, ls="none", marker="o", ms=ms, mew=0, color=c, alpha=alpha, rasterized=True, zorder=2)
        mu, S = d["centres"][m], d["covs"][m]
        ev, V = np.linalg.eigh(S)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        for k, ls in styles:
            ax.add_patch(Ellipse((mu.real, mu.imag), 2 * k * np.sqrt(ev[1]), 2 * k * np.sqrt(ev[0]), angle=ang,
                                 fill=False, ec=c, ls=ls, lw=0.9, zorder=4))
        u = (mu - mid) / abs(mu - mid) if abs(mu - mid) > 0 else 1j
        p = mu + u * (max(sigmas) + 0.9) * np.sqrt(ev[1])
        ax.text(p.real, p.imag, labels[m], ha="center", va="center", fontsize=t.size, color=c, zorder=5)
    keys = [Line2D([], [], color=t.ink, ls=ls, lw=0.9, label=rf"${k:g}\sigma$") for k, ls in styles]
    if line and d["w"] is not None:
        p0 = d["w"] * d["threshold"]
        p1 = p0 + 1j * d["w"]
        kw = ref_kw(color=t.ink, ls=(0, (6, 2, 1, 2)), lw=0.7)
        ax.axline((p0.real, p0.imag), (p1.real, p1.imag), **kw)
        keys.append(Line2D([], [], label="threshold", **kw))
    if loc:
        ax.legend(handles=keys, loc=loc, handlelength=2.2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"I ({unit})")
    ax.set_ylabel(f"Q ({unit})")
    return d


def _core(x, k=3.0):
    """Centre and width of the main peak of samples x, by iterated clipping at k standard deviations."""
    mu, s = np.median(x), np.std(x)
    for _ in range(30):
        sel = np.abs(x - mu) < k * s
        mu, s = x[sel].mean(), x[sel].std()
    return mu, s


def readout_histogram(ax, shots, *, disc=None, bins=70, log=False, fit=True, labels=None, unit="mV",
                      loc="upper right"):
    """Histograms of single shots of the first two prepared states projected on the discrimination
    axis of iq_discriminator (or disc), with the threshold at zero. A Gaussian fit to the core of each
    histogram is drawn over its full range, so decay tails stand out (log=True shows them best). The
    assignment fidelity F = 1 - (P(1|0) + P(0|1))/2, counted from the shots, and both errors are printed
    at loc. Returns {"F", "P10", "P01", "fits"}."""
    t = theme()
    d = disc or iq_discriminator(shots)
    labels = _state_labels(labels, 2)
    x = [np.real(np.conj(d["w"]) * np.asarray(s, complex)) - d["threshold"] for s in shots[:2]]
    edges = np.linspace(min(v.min() for v in x), max(v.max() for v in x), bins + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    xx = np.linspace(edges[0], edges[-1], 600)
    fits = []
    for m, v in enumerate(x):
        c = t.levels[m % len(t.levels)]
        counts, _ = np.histogram(v, edges)
        ax.stairs(counts, edges, fill=True, fc=lighten(c, 0.72), ec="none", zorder=1)
        ax.stairs(counts, edges, color=c, lw=0.7, label=labels[m], zorder=2)
        r = None
        if fit:
            mu, s = _core(v)
            sel = (np.abs(mids - mu) < 2.5 * s) & (counts > 0)
            r = F.gaussian(mids[sel], counts[sel], sigma=np.sqrt(counts[sel]))
            ax.plot(xx, r(xx), **model_kw(color=c, ls=t.dashes[m], lw=0.9, zorder=3))
        fits.append(r)
    p10, p01 = float(np.mean(x[0] > 0)), float(np.mean(x[1] <= 0))
    Fa = 1 - (p10 + p01) / 2
    ax.axvline(0, **ref_kw())
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.7, None)
    ax.set_xlim(edges[0], edges[-1])
    if loc:
        note(ax, rf"$F$ = {100 * Fa:.1f} %" "\n" rf"$P(1|0)$ = {100 * p10:.1f} %" "\n" rf"$P(0|1)$ = {100 * p01:.1f} %",
             loc=loc)
    ax.legend(loc="upper left")
    ax.set_xlabel(f"signal along discrimination axis ({unit})")
    ax.set_ylabel("counts")
    return {"F": Fa, "P10": p10, "P01": p01, "fits": fits}


def assignment_matrix(ax, M, labels=None, *, fmt="{:.3f}", cbar=False):
    """Assignment matrix P(measured | prepared) as annotated cells, rows prepared and columns measured;
    counts are normalised row by row. The mean assignment fidelity, the average of the diagonal, is
    printed above the matrix on the right. Returns (image, colour bar or None)."""
    t = theme()
    M = np.asarray(M, float)
    M = M / M.sum(axis=1, keepdims=True)
    labels = _state_labels(labels, M.shape[0])
    im, cb = annotated_matrix(ax, M, labels, fmt=fmt, vmin=0, vmax=1, cbar=cbar, label="probability")
    ax.set_xlabel("measured")
    ax.set_ylabel("prepared")
    ax.set_title(rf"$\bar F$ = {100 * np.trace(M) / M.shape[0]:.1f} %", loc="right", fontsize=t.size)
    return im, cb
