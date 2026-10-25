"""Spectrum of a transmon: the potential with computed levels and wavefunctions, charge dispersion
and its parity branches, scaling with E_J/E_C, flux tuning and flux-tuned spectroscopy, the
vacuum-Rabi avoided crossing as a map and as dressed lines."""
import numpy as np

from .. import physics as ph
from ..core import MINUS, model_kw, note, ref_kw, series, sim_kw, theme
from ..schematics.motifs import well
from ..schematics.symbols import dim
from ._base import heatmap
from .design import _log_ticks, _points

__all__ = ["potential_well", "charge_dispersion", "parity_bands", "ratio_scaling", "flux_tuning",
           "flux_spectroscopy", "avoided_crossing", "dressed_lines"]


def potential_well(ax, EJ, EC, nlev=4, ng=0.0, harmonic=True, ncut=30, lobe=0.7, label_levels=True):
    """Cosine potential U = EJ (1 - cos phi) with the lowest nlev levels of the charge-basis solution,
    each with its |psi(phi)|^2 filled on it, and the harmonic approximation dashed. Energies in GHz
    from the bottom of the well. Returns the level energies."""
    t = theme()
    E, V = ph.transmon_levels(EJ, EC, ng, ncut, nlev)
    E = E + EJ
    phi = np.linspace(-np.pi, np.pi, 1201)
    U = EJ * (1 - np.cos(phi))
    dens = ph.phase_densities(V, phi)
    if harmonic:
        ax.plot(phi, EJ * phi ** 2 / 2, **ref_kw())
    well(ax, phi, U, E, dens, scale=lobe * (E[1] - E[0]) / dens.max(), labels=False, lines="well")
    ends = [phi[d > 1e-2 * d.max()].max() for d in dens]          # right end of each drawn lobe
    if label_levels:
        for m, Em in enumerate(E):
            ax.text(ends[m] + 0.08, Em, rf"$|{m}\rangle$", ha="left", va="center", fontsize=t.size,
                    color=t.levels[m % len(t.levels)])
    half = min(np.pi, max(ends) + 0.55)
    ax.set_xlim(-half, half)
    ax.set_ylim(0, E[-1] + 1.15 * (E[1] - E[0]))
    ticks = [k * np.pi / 4 for k in range(-4, 5) if abs(k * np.pi / 4) <= half + 1e-9]
    names = {-4: r"$-\pi$", -3: r"$-3\pi/4$", -2: r"$-\pi/2$", -1: r"$-\pi/4$", 0: "0", 1: r"$\pi/4$",
             2: r"$\pi/2$", 3: r"$3\pi/4$", 4: r"$\pi$"}
    ax.set_xticks(ticks, [names[round(x / (np.pi / 4))] for x in ticks])
    ax.set_xlabel(r"phase $\varphi$")
    ax.set_ylabel(r"energy $E/h$ (GHz)")
    return E


def charge_dispersion(ax, ratio, nlev=3, ng=None, label=True):
    """Levels against offset charge for E_J/E_C = ratio, in units of the 0-1 transition energy at the
    charge degeneracy point (the presentation of Koch et al. 2007, Fig. 2)."""
    t = theme()
    ng = np.linspace(-2, 2, 401) if ng is None else ng
    E = ph.charge_dispersion(ratio, 1.0, ng, nlev)
    E01 = np.diff(ph.charge_dispersion(ratio, 1.0, [0.5], 2)[0])[0]
    E = (E - E[:, 0].min()) / E01
    for m in range(nlev):
        ax.plot(ng, E[:, m], color=t.levels[m % len(t.levels)], ls="-", lw=t.lw)
    ax.set_xlim(ng[0], ng[-1])
    ax.set_xlabel(r"offset charge $n_g$")
    ax.set_ylabel(r"$E_m/E_{01}$")
    if label:
        ax.set_title(rf"$E_J/E_C = {ratio:g}$", loc="center", fontsize=t.size)
    return E


_UNITS = {"GHz": 1.0, "MHz": 1e3, "kHz": 1e6}


def parity_bands(ax, EJ, EC, *, ng=None, level=None, unit="MHz", zero_contrast=True, mark=True, ncut=20):
    """Even and odd charge-parity branches against offset charge n_g (in units of 2e), from
    physics.parity_branches. By default each branch's 0-1 frequency is drawn relative to the midpoint
    of the two branches, so the even-odd splitting that a parity measurement reads is centred on zero;
    level=m draws the energy of level m instead, relative to its own midpoint. Even parity is solid and
    odd parity dashed. zero_contrast marks the offset charges where the branches cross and a parity
    measurement sees nothing; mark draws the full splitting at n_g = 0 (2 chi_0 = |eps_1 - eps_0| for
    the transition, |eps_m| for a level) and prints it. unit is "GHz", "MHz" or "kHz". Returns
    {"ng", "even", "odd" (centred, in unit), "centre" (GHz), "splitting" (in unit)}."""
    t = theme()
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {sorted(_UNITS)}, not {unit!r}")
    if level is not None and (int(level) != level or level < 0):
        raise ValueError(f"level must be a non-negative integer, not {level!r}")
    s = _UNITS[unit]
    ng = np.linspace(-0.5, 0.5, 241) if ng is None else np.asarray(ng, float)
    nlev = 2 if level is None else int(level) + 1
    pick = (lambda E: E[:, 1] - E[:, 0]) if level is None else (lambda E: E[:, int(level)])
    ev, od = ph.parity_branches(EJ, EC, ng, nlev, ncut)
    e0, o0 = (pick(B)[0] for B in ph.parity_branches(EJ, EC, [0.0], nlev, ncut))   # at n_g = 0 and 1/2
    centre = 0.5 * (e0 + o0)
    even, odd = (pick(ev) - centre) * s, (pick(od) - centre) * s
    if level is None:
        split = abs(ph.level_dispersion(EJ, EC, 1, ncut) - ph.level_dispersion(EJ, EC, 0, ncut)) * s
        name = r"2\chi_0"
    else:
        split = abs(ph.level_dispersion(EJ, EC, int(level), ncut)) * s
        name = rf"|\epsilon_{int(level)}|"
    ax.plot(ng, even, color=t.cycle[0], ls="-", lw=t.lw, label="even", zorder=3)
    ax.plot(ng, odd, color=t.cycle[1], ls=(0, (4, 2)), lw=t.lw, label="odd", zorder=3)
    if zero_contrast:
        k = np.arange(np.ceil(ng.min() * 2 - 0.5), np.floor(ng.max() * 2 - 0.5) + 1)
        for j, x0 in enumerate(0.25 + k / 2):
            ax.axvline(x0, **ref_kw(label="zero contrast" if j == 0 else None))
    if mark and ng.min() <= 0 <= ng.max():
        dim(ax, (0.0, (o0 - centre) * s), (0.0, (e0 - centre) * s), "", color=t.ink)
        note(ax, rf"${name}/h$ = {split:.3g} {unit}", loc="upper right")
    A = max(np.abs(even).max(), np.abs(odd).max())
    ax.set_ylim(-1.35 * A, 2.5 * A)                                    # headroom for the legend and the note
    ax.set_xlim(ng.min(), ng.max())
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel(r"offset charge $n_g$ ($2e$)")
    if level is None:
        ax.set_ylabel(rf"$f_{{01}} - \bar f_{{01}}$ ({unit})")
    else:
        ax.set_ylabel(rf"$(E_{int(level)} - \bar E_{int(level)})/h$ ({unit})")
    return {"ng": ng, "even": even, "odd": odd, "centre": centre, "splitting": split}


_RATIO = {   # y label, asymptote label, log axis
    "f01": (r"$f_{01}/E_C$", r"$\sqrt{8E_J/E_C} - 1$", False),
    "alpha": (r"$\alpha/E_C$", r"$-1$", False),
    "dispersion": (r"$|\epsilon_1 - \epsilon_0|/E_C$", "asymptotic", True),
    "parity": (r"$2\chi_0/f_{01}$", "asymptotic", True),
}


def ratio_scaling(ax, quantity, ratios=None, *, asymptote=True, points=None, bands=None, legend=True, loc="best",
                  ncut=15):
    """One transmon property against E_J/E_C, computed exactly in the charge basis (solid), with its
    large-ratio asymptote dashed. quantity "f01" draws f01/E_C with sqrt(8 E_J/E_C) - 1, "alpha" the
    anharmonicity alpha/E_C with -1 (physics.transmon_asymptotic), "dispersion" the peak-to-peak charge
    dispersion of the 0-1 transition |eps_1 - eps_0|/E_C, and "parity" the even-odd splitting over the
    mean frequency, 2 chi_0/f01 = |eps_1 - eps_0| / f01(n_g = 1/4). The last two use a log axis and
    the asymptote of physics.koch_dispersion (over the exact f01 for "parity"), and drop exact values
    below 100 times the rounding error of the diagonalisation, where they are noise. asymptote=False
    omits the dashed curve; a number starts it at that ratio (default 5 for the dispersions, the whole
    range otherwise). points {label: ratio} marks designs on the exact curve with open markers, and
    {label: (ratio, value)}, (ratio, value, kind) or (ratio, value, kind, style) places given values
    with the marker grammar (design._points: style sets the marker and moves the label). bands
    {label: (lo, hi)} shades ranges of the ratio, each labelled at the top. Returns {"ratio", "exact",
    "asymptote" (NaN where not drawn), "points": {label: (ratio, value)}}."""
    t = theme()
    if quantity not in _RATIO:
        raise ValueError(f"quantity must be one of {sorted(_RATIO)}, not {quantity!r}")
    ylabel, alabel, log = _RATIO[quantity]
    if ratios is None:
        ratios = np.linspace(5, 100, 400) if quantity == "parity" else np.linspace(1, 200, 400)
    r = np.asarray(ratios, float)
    if r.ndim != 1 or r.size < 2 or r.min() <= 0:
        raise ValueError("ratios must be a 1-D array of at least two positive values")

    def exact(rr):
        rr = np.atleast_1d(np.asarray(rr, float))
        if not log:
            s = np.array([ph.transmon_spectrum(x, 1.0, ncut=ncut, nlev=3) for x in rr])
            return s[:, 1] if quantity == "f01" else s[:, 2] - 2 * s[:, 1]
        eps = np.abs(ph.level_dispersion(rr, 1.0, 1, ncut) - ph.level_dispersion(rr, 1.0, 0, ncut))
        eps = np.where(eps < 100 * np.finfo(float).eps * (4 * ncut ** 2 + rr), np.nan, eps)
        if quantity == "parity":
            eps = eps / np.array([ph.transmon_spectrum(x, 1.0, 0.25, ncut, 2)[1] for x in rr])
        return eps

    y = exact(r)
    if quantity in ("f01", "alpha"):
        ya = ph.transmon_asymptotic(r, 1.0)[0 if quantity == "f01" else 1]
    else:
        ya = np.abs(ph.koch_dispersion(r, 1.0, 1) - ph.koch_dispersion(r, 1.0, 0))
        if quantity == "parity":
            ya = ya / np.array([ph.transmon_spectrum(x, 1.0, 0.25, ncut, 2)[1] for x in r])
    start = (5.0 if log else r.min()) if asymptote is True else (asymptote if asymptote else np.inf)
    ya = np.where(r >= start, ya, np.nan)
    for label, (lo, hi) in (bands or {}).items():
        ax.axvspan(lo, hi, fc=t.tint, ec="none", zorder=0)
        ax.text((lo + hi) / 2, 0.97, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=t.size - 1, color=t.ink, zorder=6)
    ax.plot(r, y, label="exact", **model_kw())
    if np.isfinite(ya).any():
        ax.plot(r, ya, label=alabel, **ref_kw(lw=0.9))
    marked = {}
    for label, p in (points or {}).items():
        marked[label] = (float(p), float(exact(p)[0])) if np.ndim(p) == 0 else tuple(p)
    _points(ax, marked)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(r.min(), r.max())
    if log:
        _log_ticks(ax, "y")
    if legend:
        ax.legend(loc=loc)
    ax.set_xlabel(r"$E_J/E_C$")
    ax.set_ylabel(ylabel)
    return {"ratio": r, "exact": y, "asymptote": ya, "points": {k: v[:2] for k, v in marked.items()}}


_STYLES = ("-", (0, (4, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5)), (0, (2, 1)))


def flux_tuning(ax, EJmax, EC, *, flux=None, d=(0.0,), detune=None, sweet_spot=True, legend=True, ncut=20,
                color=None, sweet_label="sweet spot"):
    """f01 of a SQUID transmon against flux (in Phi_0), exact, from physics.tunable_frequencies: one curve
    per junction asymmetry d = (EJ1 - EJ2)/(EJ1 + EJ2), a number or a sequence, each named in the
    legend. The curves take the series colours, or with color one colour (a qubit's, say) and line
    styles solid, dashed, dotted in turn. sweet_spot marks the flux-insensitive maximum at zero flux,
    labelled sweet_label (None for no label); detune (GHz, one value or several) marks the flux at which
    f01 sits that far below the maximum on the first curve, at positive flux, from
    physics.flux_at_frequency, and prints it. Returns {"flux", "f01": {d: GHz}, "marks": {detune: flux}}."""
    t = theme()
    flux = np.linspace(-0.5, 0.5, 301) if flux is None else np.asarray(flux, float)
    ds = [float(v) for v in np.atleast_1d(d)]
    if not ds:
        raise ValueError("d needs at least one asymmetry")
    out = {"flux": flux, "f01": {}, "marks": {}}
    for i, dd in enumerate(ds):
        out["f01"][dd] = f = ph.tunable_frequencies(flux, EJmax, EC, dd, nlev=2, ncut=ncut)[:, 0]
        st = series(i) if color is None else dict(color=color, ls=_STYLES[i % len(_STYLES)])
        ax.plot(flux, f, color=st["color"], ls=st["ls"], lw=t.lw, label=rf"$d$ = {dd:g}", zorder=3)
    f0 = float(ph.tunable_frequencies(0.0, EJmax, EC, ds[0], nlev=2, ncut=ncut)[0, 0])
    if sweet_spot:
        ax.plot(0.0, f0, **sim_kw(color=color, ms=4.5, zorder=5))
        if sweet_label:
            ax.annotate(sweet_label, (0.0, f0), xytext=(4, 3), textcoords="offset points", ha="left", va="bottom",
                        fontsize=t.size - 1, color=t.ink)
    for dt in [] if detune is None else [float(v) for v in np.atleast_1d(detune)]:
        x = ph.flux_at_frequency(f0 - dt, EJmax, EC, ds[0], ncut)
        out["marks"][dt] = x
        ax.plot(x, f0 - dt, **sim_kw(color=t.accent, marker="s", ms=4, zorder=5))
        ax.annotate(rf"{MINUS}{dt:g} GHz at $\Phi = {x:.3f}\,\Phi_0$", (x, f0 - dt), xytext=(-5, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=t.size - 1, color=t.ink)
    ax.set_xlim(flux.min(), flux.max())
    ax.set_ylim(0, 1.12 * max(f.max() for f in out["f01"].values()))
    if legend:
        ax.legend(loc="lower center")
    ax.set_xlabel(r"flux $\Phi/\Phi_0$")
    ax.set_ylabel(r"$f_{01}$ (GHz)")
    return out


def flux_spectroscopy(ax, flux, freq, signal, *, model=None, model_side="left", label="signal (arb. units)",
                      cbar=True, xlabel=r"flux $\Phi/\Phi_0$", ylabel="frequency (GHz)"):
    """Two-tone spectroscopy against flux as a heatmap (signal shape len(freq) x len(flux)), with
    model transitions {label: f(flux)} drawn over it. The model covers only the left half of the
    flux range (model_side="left"; "right", or "both"), so the data stay visible beside it."""
    t = theme()
    mesh, cb = heatmap(ax, flux, freq, signal, label=label, cbar=cbar)
    flux = np.asarray(flux)
    mid = 0.5 * (flux.min() + flux.max())
    keep = {"left": flux <= mid, "right": flux >= mid, "both": np.ones(flux.size, bool)}[model_side]
    for k, (name, f) in enumerate((model or {}).items()):
        ax.plot(flux[keep], np.asarray(f)[keep], color=t.overlay, lw=0.9, ls=(0, (4, 2)) if k == 0 else (0, (1, 1.5)),
                label=name)
    if model:
        ax.legend(loc="lower center", frameon=False, handlelength=2.2)
    ax.set_xlim(np.min(flux), np.max(flux))
    ax.set_ylim(np.min(freq), np.max(freq))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mesh, cb


def avoided_crossing(ax, x, freq, signal, *, bare=None, g=None, x0=None, label=r"$|S_{21}|$", cmap=None,
                     cbar=True, xlabel=r"flux $\Phi/\Phi_0$", ylabel="frequency (GHz)"):
    """Resonator transmission across a qubit-resonator crossing (signal shape len(freq) x len(x)).
    bare = (f_qubit(x), f_resonator(x)) draws the uncoupled modes dashed; g and the crossing point x0
    mark the vacuum-Rabi splitting 2g. Transmission dips are the signal, so the map runs reversed
    (dips dark) unless cmap is given."""
    t = theme()
    mesh, cb = heatmap(ax, x, freq, signal, label=label, cbar=cbar, cmap=cmap or t.cmap + "_r")
    if bare is not None:
        for f in bare:
            ax.plot(x, f, color=t.overlay, lw=0.7, ls=(0, (4, 2)))
    if g is not None and x0 is not None and bare is not None:
        fc = np.interp(x0, x, bare[1])
        dim(ax, (x0, fc - g), (x0, fc + g), "", color=t.ink)
        ax.text(x0 + 0.015 * np.ptp(x), fc, rf"$2g/2\pi = {2 * g * 1e3:.0f}$ MHz", ha="left", va="center",
                fontsize=t.size - 0.5, color=t.ink, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(freq), np.max(freq))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mesh, cb


def dressed_lines(ax, fq, fr, g, *, EC=None, mark=None, points=None, split=True, nlev=6, nfock=8, branches=None,
                  color=None, bare=True, fr_label=r"bare $f_r$", split_label="{:.3g} MHz"):
    """The two one-excitation frequencies of a qubit coupled to a resonator against the bare qubit
    frequency fq (GHz), with the bare qubit (diagonal) and bare resonator (horizontal) dashed. Without
    EC the qubit is a two-level system (physics.dressed_frequencies); with EC it is a transmon of that
    charging energy, solved with its higher levels and counter-rotating terms
    (physics.transmon_dressed). branches (lower, upper) on fq draws a spectrum computed elsewhere
    instead, one whose coupling changes along the sweep, say. split draws the smallest splitting, 2g
    on resonance, as a dimension arrow and prints it in MHz with split_label, a format string. mark=fq0
    marks the design point on both branches, a circle on the qubit-like branch and a square on the
    resonator-like one. points {label: (fq, f), (fq, f, kind) or (fq, f, kind, style)} places other
    values with the marker grammar. color draws the branches in that colour; bare=False leaves out the
    bare lines and fr_label (None for none), so a second call can add the branches of another
    coupling to the same axes. Returns {"lower", "upper" (GHz on fq), "split" (GHz), "at" (the fq of
    the smallest splitting)}."""
    t = theme()
    fq = np.asarray(fq, float)
    if branches is not None:
        given = [np.asarray(b, float) for b in branches]
        if len(given) != 2 or any(b.shape != fq.shape for b in given):
            raise ValueError("branches needs (lower, upper), each with one value per fq")

    def solve(x):
        if branches is not None:
            return tuple(np.atleast_1d(np.interp(x, fq, b)) for b in given)
        if EC is None:
            up, lo = ph.dressed_frequencies(np.atleast_1d(x), fr, g)
            return lo, up
        f = ph.transmon_dressed(x, fr, g, EC, nlev, nfock)["f"]
        return f[:, 0], f[:, 1]

    lower, upper = solve(fq)
    if bare:
        ax.plot(fq, fq, **ref_kw())
        ax.axhline(fr, **ref_kw())
        if fr_label:
            ax.text(fq.max() - 0.015 * np.ptp(fq), fr, fr_label, ha="right", va="bottom", fontsize=t.size - 1,
                    color=t.reference)
    ax.plot(fq, lower, **model_kw(color=color))
    ax.plot(fq, upper, **model_kw(color=color))
    k = int(np.argmin(upper - lower))
    gap = float(upper[k] - lower[k])
    if split:
        dim(ax, (fq[k], lower[k]), (fq[k], upper[k]), "", color=t.ink)
        ax.annotate(split_label.format(gap * 1e3), (fq[k], lower[k]), xytext=(5, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=t.size - 1, color=t.ink)      # below the lower branch
    if mark is not None:
        lo0, up0 = (float(v[0]) for v in solve([mark]))
        qubit_low = abs(lo0 - mark) <= abs(up0 - mark)
        for f, m in ((lo0, "o" if qubit_low else "s"), (up0, "s" if qubit_low else "o")):
            ax.plot(mark, f, **sim_kw(color=t.accent, marker=m, ms=4.5, zorder=5))
        ax.annotate("design", (mark, lo0 if qubit_low else up0), xytext=(5, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=t.size - 1, color=t.ink)
    _points(ax, points)
    pad = 0.04 * (upper.max() - lower.min())
    ax.set_xlim(fq.min(), fq.max())
    ax.set_ylim(lower.min() - pad, upper.max() + pad)
    ax.set_xlabel(r"bare qubit frequency $f_q$ (GHz)")
    ax.set_ylabel("dressed frequency (GHz)")
    return {"lower": lower, "upper": upper, "split": gap, "at": float(fq[k])}
