"""Dark-matter plots: the evidence (a galaxy rotation curve), the wave (occupation number, line shape
and a field trace), and the coupling reach against frequency with measured limits and a dark-photon
mass axis."""
import numpy as np

from .. import physics as ph
from ..core import data_kw, log_minor, model_kw, ref_kw, series, theme
from ..themes import lighten
from .design import _plain_log

__all__ = ["reach", "rotation_curve", "occupation", "lineshape", "field_trace"]


def rotation_curve(ax, r, visible, halo, *, observed=None, labels=("visible matter", "dark halo", "total"),
                   legend=True, loc="lower right", xlabel="radius (kpc)", ylabel="orbital speed (km/s)"):
    """Rotation curve of a galaxy against radius r: the circular speed that the visible matter alone
    would give (grey dashes), the halo's share (signal colour) and their sum in quadrature (ink).
    physics.rotation_curve gives a model. observed (r, v) or (r, v, err) adds measured points. Returns
    {"visible", "halo", "total": Line2D}."""
    t = theme()
    r, visible, halo = (np.asarray(a, float) for a in (r, visible, halo))
    if not (r.shape == visible.shape == halo.shape) or r.ndim != 1 or r.size < 2:
        raise ValueError("r, visible and halo must be 1-D arrays of one length, at least two points")
    out = {"visible": ax.plot(r, visible, label=labels[0], color=t.reference, ls=(0, (4, 2)), lw=t.lw,
                              zorder=3)[0],
           "halo": ax.plot(r, halo, label=labels[1], color=t.signal, ls=t.dashes[2], lw=t.lw, zorder=3)[0],
           "total": ax.plot(r, np.hypot(visible, halo), label=labels[2], **model_kw(t.ink))[0]}
    if observed is not None:
        ro, vo, *err = observed
        ax.errorbar(ro, vo, yerr=err[0] if err else None, **data_kw(ms=3), elinewidth=0.6, capsize=0)
    ax.set_xlim(0, r.max())
    ax.set_ylim(0, 1.15 * max(np.nanmax(np.hypot(visible, halo)), np.max(observed[1]) if observed else 0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=loc)
    return out


def occupation(ax, m, N, *, threshold=1.0, band=None, band_label="qubit band", labels=("wave-like",
               "particle-like"), xlabel=r"rest energy $mc^2$ (eV)",
               ylabel=r"particles per de Broglie volume"):
    """Occupation number N against rest energy m (eV) on log-log axes, in the signal colour, with the
    wash over the masses where N exceeds threshold (the wave-like side) and a grey guide at the
    threshold. physics.occupation_number gives N (take m in µeV there). band (m_lo, m_hi) in eV shades
    a mass range in the model colour, such as the band a qubit search covers. Returns {"line": Line2D,
    "crossing": the mass where N equals threshold, interpolated in log}."""
    t = theme()
    m, N = np.asarray(m, float), np.asarray(N, float)
    if m.shape != N.shape or m.ndim != 1 or m.size < 2 or m.min() <= 0 or N.min() <= 0:
        raise ValueError("m and N must be 1-D arrays of one length with positive values")
    order = np.argsort(m)
    lm, lN = np.log10(m[order]), np.log10(N[order])
    lt = np.log10(threshold)
    crossing = 10 ** np.interp(-lt, -lN, lm) if lN.min() < lt < lN.max() else None
    line, = ax.plot(m, N, color=t.signal, ls="-", lw=t.lw, zorder=4)
    ax.axhline(threshold, **ref_kw())
    side = ax.get_xaxis_transform()                      # x in data, y in axes fraction
    if crossing is not None:
        ax.axvspan(m.min(), crossing, fc=t.wash, ec="none", zorder=0)
        ax.axvline(crossing, **ref_kw())
        # The curve falls to the right, so the wave-like label sits low on the left and the particle-like
        # label high on the right, both clear of the line.
        ax.text(np.sqrt(m.min() * crossing), 0.06, labels[0], transform=side, ha="center", va="bottom",
                fontsize=t.size - 0.5, color=t.ink, zorder=6)
        ax.text(np.sqrt(crossing * m.max()), 0.94, labels[1], transform=side, ha="center", va="top",
                fontsize=t.size - 0.5, color=t.ink, zorder=6)
    if band is not None:
        ax.axvspan(*band, fc=lighten(t.model, 0.45), ec="none", zorder=1)
        ax.text(band[1], 0.06, " " + band_label, transform=side, ha="left", va="bottom",
                fontsize=t.size - 0.5, color=t.ink, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(m.min(), m.max())
    ax.set_ylim(10 ** np.floor(lN.min()), 10 ** np.ceil(lN.max()))
    for a, (lo, hi) in ((ax.xaxis, (lm.min(), lm.max())), (ax.yaxis, (lN.min(), lN.max()))):
        _decade_ticks(a, lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return {"line": line, "crossing": crossing}


def _decade_ticks(axis, lo, hi, most=8):
    """Major ticks at round powers of ten (every 1, 2, 5, 10, 20, 25 or 50 decades) on a log axis that
    spans lo to hi decades, with no more than most ticks and no minor ticks when the stride exceeds one."""
    from matplotlib.ticker import FixedLocator, NullLocator
    span = hi - lo
    step = next((s for s in (1, 2, 5, 10, 20, 25, 50, 100) if span / s <= most), 100)
    axis.set_major_locator(FixedLocator(10.0 ** np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)))
    if step > 1:
        axis.set_minor_locator(NullLocator())


def lineshape(ax, df, g, *, mean=None, mean_label="mean", xlabel="offset above $mc^2/h$ (kHz)",
              ylabel="spectral density (1/kHz)"):
    """Line shape g of a dark-matter wave against the offset df above its rest frequency, in the signal
    colour over its wash, from physics.dm_lineshape. mean marks the mean offset with a grey guide and
    a label. Returns {"line": Line2D, "fill": PolyCollection}."""
    t = theme()
    df, g = np.asarray(df, float), np.asarray(g, float)
    if df.shape != g.shape or df.ndim != 1 or df.size < 2:
        raise ValueError("df and g must be 1-D arrays of one length, at least two points")
    fill = ax.fill_between(df, 0, g, fc=t.wash, ec="none", zorder=1)
    line, = ax.plot(df, g, color=t.signal, ls="-", lw=t.lw, zorder=3)
    if mean is not None:
        ax.axvline(mean, **ref_kw())
        ax.annotate(mean_label, (mean, 0.92 * g.max()), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=t.size - 0.5, color=t.ink)
    ax.set_xlim(df.min(), df.max())
    ax.set_ylim(0, 1.12 * g.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return {"line": line, "fill": fill}


def field_trace(ax, t_, A, *, window=None, window_label="one shot", phase=True, xlabel="time (µs)",
                ylabel="amplitude / rms", phase_label=r"phase ($\pi$)"):
    """Slowly varying amplitude |A| of a dark-matter wave against time t_, in the signal colour and
    normalised to its rms, from physics.dm_field_trace. phase adds the phase of A in units of pi as grey
    dashes on a right axis. window (t0, t1) shades a time span, such as one qubit shot, over which the
    wave hardly changes. Returns {"amplitude": Line2D, "phase": Line2D or None, "twin": Axes or None}."""
    t = theme()
    t_, A = np.asarray(t_, float), np.asarray(A, complex)
    if t_.shape != A.shape or t_.ndim != 1 or t_.size < 2:
        raise ValueError("t and A must be 1-D arrays of one length, at least two points")
    amp = np.abs(A) / np.sqrt(np.mean(np.abs(A) ** 2))
    if window is not None:
        ax.axvspan(*window, fc=t.tint, ec="none", zorder=0)
        ax.annotate(window_label, (window[1], 0.97), xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="top", fontsize=t.size - 0.5, color=t.ink)
    line, = ax.plot(t_, amp, color=t.signal, ls="-", lw=t.lw, zorder=3)
    ax.set_xlim(t_.min(), t_.max())
    ax.set_ylim(0, 1.15 * amp.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    out = {"amplitude": line, "phase": None, "twin": None}
    if phase:
        tw = ax.twinx()
        ph_ = np.angle(A) / np.pi
        ph_[1:][np.abs(np.diff(ph_)) > 1] = np.nan       # a jump from +pi to -pi is one angle, so break it
        out["phase"], = tw.plot(t_, ph_, color=t.reference, ls=(0, (4, 2)), lw=0.8, zorder=2)
        tw.set_ylim(-1.15, 1.15)
        tw.set_yticks([-1, 0, 1])
        tw.set_ylabel(phase_label, color=t.reference)
        tw.tick_params(axis="y", colors=t.reference)
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
        out["twin"] = tw
    return out


def reach(ax, f, eps, *, label=None, styles=None, limits=None, excluded=None, mass_axis=True, legend=True,
          loc="best", xlabel="frequency (GHz)", ylabel=r"kinetic mixing $\epsilon$",
          mass_label=r"dark-photon mass $m_{A'}c^2$ (µeV)"):
    """Projected coupling reach against frequency f (GHz) on log-log axes. eps is one computed curve
    (named by label) or {label: curve}, each on f; the curves take the series styles, and styles
    {label: line keywords} overrides them (one colour per register size and one dash per temperature,
    say). limits {label: (f, eps)} marks measured limits as stars in the accent colour, labelled above.
    excluded="above" (or "below") shades the side of the best curve, the lowest (or highest) at each
    frequency, that a null result would exclude. mass_axis puts the rest energy m c^2 = h f in µeV on
    the top axis (physics.mass_from_frequency). Axes spanning fewer than three decades are labelled at
    1, 2 and 5 times each power of ten. Returns {"lines": {label: Line2D}, "envelope": the best reach on
    f, "top": the top axis or None}."""
    t = theme()
    if excluded not in (None, "above", "below"):
        raise ValueError(f"excluded must be None, 'above' or 'below', not {excluded!r}")
    f = np.asarray(f, float)
    if f.ndim != 1 or f.size < 2 or f.min() <= 0:
        raise ValueError("f must be a 1-D array of at least two positive frequencies")
    curves = dict(eps) if isinstance(eps, dict) else {label: eps}
    if not curves:
        raise ValueError("reach needs at least one curve")
    E = []
    lines = {}
    for i, (name, e) in enumerate(curves.items()):
        e = np.asarray(e, float)
        if e.shape != f.shape:
            raise ValueError(f"curve {name!r} has {e.size} values for {f.size} frequencies")
        E.append(e)
        kw = {**series(i), "marker": None, "lw": t.lw, "zorder": 3, **(styles or {}).get(name, {})}
        lines[name], = ax.plot(f, e, label=name, **kw)
    E = np.array(E)
    with np.errstate(invalid="ignore"):
        env = np.nanmax(E, axis=0) if excluded == "below" else np.nanmin(E, axis=0)
    marks = list((limits or {}).values())
    vals = np.concatenate([E[np.isfinite(E) & (E > 0)], [m[1] for m in marks]])
    ylim = (10.0 ** np.floor(np.log10(vals.min())), 10.0 ** np.ceil(np.log10(vals.max())))
    if excluded:
        edge = ylim[1] * 10 if excluded == "above" else ylim[0] / 10
        ax.fill_between(f, env, edge, fc=lighten(t.reference, 0.78), ec="none", zorder=0, label="excluded")
    for name, (fl, el) in (limits or {}).items():
        ax.plot(fl, el, **data_kw(color=t.accent, marker="*", ms=8, zorder=5))
        ax.annotate(name, (fl, el), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=t.size - 1, color=t.ink, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(*ylim)
    log_minor(ax)
    _plain_log(ax.xaxis)
    top = None
    if mass_axis:
        top = ax.secondary_xaxis("top", functions=(ph.mass_from_frequency, ph.frequency_from_mass))
        top.set_xlabel(mass_label)
        _plain_log(top.xaxis, ph.mass_from_frequency([f.min(), f.max()]))
        ax.tick_params(axis="x", which="both", top=False)
    if legend and (len(curves) > 1 or label is not None or excluded):
        ax.legend(loc=loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return {"lines": lines, "envelope": env, "top": top}
