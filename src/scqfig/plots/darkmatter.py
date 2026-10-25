"""Dark-matter sensitivity plots: coupling reach against frequency, with measured limits and a
dark-photon mass axis."""
import numpy as np

from .. import physics as ph
from ..core import data_kw, log_minor, series, theme
from ..themes import lighten
from .design import _plain_log

__all__ = ["reach"]


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
