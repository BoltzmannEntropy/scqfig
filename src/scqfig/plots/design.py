"""Design-stage plots: frequency plans on rows or on one axis, collision maps, junction design curves,
capacitance matrices, loss participation, dispersive shift and Purcell limits, black-box
quantization, simulation convergence and figure-of-merit design maps."""
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, SymmetricalLogLocator

from .. import physics as ph
from ..core import MINUS as _MINUS
from ..core import assumed_kw, data_kw, log_minor, model_kw, note, ref_kw, sci, series, sim_kw, theme
from ..themes import lighten
from ._base import annotated_matrix, cell_text

__all__ = ["frequency_plan", "frequency_lines", "collision_map", "junction_curve", "capacitance_matrix",
           "participation", "dispersive_shift", "purcell_limit", "admittance", "kerr_matrix", "convergence",
           "design_map"]

_KINDS = {"data": data_kw, "sim": sim_kw, "assumed": assumed_kw}
_MODEL_DASHES = ((0, (4, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5)), (0, (2, 1)))


_LABEL_KEYS = ("label", "offset", "ha", "va")


def _points(ax, points, *, kind="sim", color=None, ms=4.5, offset=(4, 3), zorder=5):
    """Mark points {label: (x, y)}, {label: (x, y, kind)} or {label: (x, y, kind, style)} with the
    marker grammar: kind "data" for measured values (filled in field, open in folio), "sim" for designed
    or simulated values and "assumed" for values taken from elsewhere. Each point takes the next marker
    of the style and its label sits beside it, offset in points. style, a dict, sets this point's
    marker keywords (marker, color, ms; assumed points stay grey) and its label: "label" prints other
    text than the key (None prints none, so the key only names the point), "offset" moves it (in
    points), "ha" and "va" align it. Returns {label: Line2D}."""
    t = theme()
    out = {}
    for k, (label, p) in enumerate((points or {}).items()):
        if len(p) not in (2, 3, 4):
            raise ValueError(f"point {label!r} needs (x, y), (x, y, kind) or (x, y, kind, style), got {p!r}")
        kd = p[2] if len(p) >= 3 else kind
        if kd not in _KINDS:
            raise ValueError(f"point kind {kd!r} is not one of {sorted(_KINDS)}")
        style = dict(p[3]) if len(p) == 4 else {}
        text = dict(label=label, offset=offset, ha="left", va="bottom")
        text.update({key: style.pop(key) for key in _LABEL_KEYS if key in style})
        kw = dict(marker=t.markers[k % len(t.markers)], ms=ms, zorder=zorder)
        if kd != "assumed":
            kw["color"] = color
        else:
            style.pop("color", None)                                 # assumed values stay grey
        kw.update(style)
        x, y = float(p[0]), float(p[1])
        out[label], = ax.plot(x, y, **_KINDS[kd](**kw))
        if text["label"]:
            ax.annotate(text["label"], (x, y), xytext=text["offset"], textcoords="offset points", ha=text["ha"],
                        va=text["va"], fontsize=t.size - 1, color=t.ink, zorder=zorder + 1)
    return out


def _errorbar(ax, x, y, color, **kw):
    """Error bars alone, in one colour. The caps get that colour as their line colour too, which
    matplotlib would otherwise leave at the cycle's first colour, unseen but resolved against
    whatever style is active when the figure is checked."""
    eb = ax.errorbar(x, y, fmt="none", ecolor=color, elinewidth=0.6, **kw)
    for cap in eb.lines[1]:
        cap.set_color(color)
    return eb


def _log_ticks(ax, axis="y"):
    """Minor ticks on a log axis: at 2..9 over a few decades, once per decade beyond six, where the
    2..9 ticks would merge into a comb."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = sorted(ax.get_ylim() if axis == "y" else ax.get_xlim())
    if lo > 0 and np.log10(hi / lo) > 6:
        a.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=40))
        a.set_minor_formatter(NullFormatter())
    else:
        log_minor(ax, axis)


def _plain_log(axis, lim=None):
    """Label a log axis that spans fewer than three decades at 1, 2 and 5 times each power of ten,
    printed as plain numbers. lim gives the span for an axis whose limits are set at draw time (a
    secondary axis)."""
    lo, hi = sorted(axis.get_view_interval() if lim is None else lim)
    if lo > 0 and np.log10(hi / lo) < 3:
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=30))
        axis.set_minor_formatter(NullFormatter())

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


def frequency_lines(ax, qubits, *, resonators=None, spread=None, scans=None, windows=None, bands=None,
                    guides=None, floor=None, floor_label="thermal floor", mass_axis=False, xlim=None, unit="GHz",
                    color=None):
    """A frequency plan on one shared axis, the qubits in the upper half and the resonators in the lower
    half. qubits {name: (f01, f12)} draws each f01 as a tall line in its qubit colour (or in color, one
    colour for every qubit, the chip's pad colour say), labelled above, and its f12 as a short dashed
    line beside it. resonators {name: fr} draws short lines in the bus
    colour, their labels staggered where they crowd. spread, one value or {name: value}, adds the
    fabrication scatter of each f01 as an error bar. scans {name: (f_from, f_to)} draws a flux-tuning
    range as an arrow in the flux-line colour. windows [(lo, hi), ...] greys the collision windows in
    the qubit half; bands {label: (lo, hi)} shades a passband (a Purcell filter, say) in the resonator
    half; guides {label: f} draws dashed bounds (a box mode, say) there. floor hatches everything below
    that frequency, labelled floor_label; a floor left of the axis is not drawn. mass_axis adds the
    boson rest energy h f in µeV on the top axis (unit must be GHz). xlim defaults to the drawn
    frequencies with a margin. Returns {"qubits": {name: Line2D}, "resonators": {name: Line2D},
    "top": the top axis or None}."""
    t = theme()
    if mass_axis and unit != "GHz":
        raise ValueError("mass_axis converts GHz to µeV, so unit must be 'GHz'")
    res = dict(resonators or {})
    freqs = ([f for pair in qubits.values() for f in pair] + list(res.values())
             + [f for pair in (scans or {}).values() for f in pair] + [f for b in (bands or {}).values() for f in b]
             + [f for w in (windows or []) for f in w] + list((guides or {}).values()))
    if not freqs:
        raise ValueError("frequency_lines needs at least one qubit or resonator")
    if xlim is None:
        lo, hi = min(freqs), max(freqs)
        pad = 0.08 * (hi - lo) if hi > lo else 0.5
        xlim = (lo - pad, hi + pad)
    x0, x1 = xlim
    span = x1 - x0
    fs = t.size - 0.5
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    if floor is not None and floor > x0:
        ax.add_patch(Rectangle((x0, 0), floor - x0, 1, fc=t.tint, ec=t.light, lw=0, hatch="////", zorder=0))
        ax.text(x0 + 0.01 * span, 0.30, floor_label, ha="left", va="center", fontsize=fs, color=t.ink, zorder=6)
    for lo, hi in windows or []:
        ax.add_patch(Rectangle((lo, 0.46), hi - lo, 0.40, fc=t.light, ec="none", zorder=1))
    for label, (lo, hi) in (bands or {}).items():
        ax.add_patch(Rectangle((lo, 0.04), hi - lo, 0.36, fc=lighten(t.bus, 0.85), ec="none", zorder=1))
        inside = lo - x0 < 0.15 * span
        ax.text(lo + (0.01 if inside else -0.01) * span, 0.16, label, ha="left" if inside else "right", va="center",
                fontsize=fs, color=t.ink, zorder=6)
    for label, f in (guides or {}).items():
        ax.axvline(f, ymin=0.05, ymax=0.45, **ref_kw(lw=0.9))
        right = x1 - f < 0.15 * span                                 # near the right edge: label on the left
        ax.text(f + (-0.008 if right else 0.008) * span, 0.12, label, ha="right" if right else "left", va="bottom",
                fontsize=fs, color=t.ink, zorder=6)
    for k, (name, (fa, fb)) in enumerate((scans or {}).items()):
        y = 0.62 if k % 2 == 0 else 0.66
        ax.add_patch(FancyArrowPatch((fa, y), (fb, y), arrowstyle="-|>", mutation_scale=6, color=t.chip["flux"],
                                     lw=1.0, shrinkA=0, shrinkB=0, zorder=4))
    out = {"qubits": {}, "resonators": {}, "top": None}
    for k, (name, (f01, f12)) in enumerate(qubits.items()):
        c = color or t.qubits[k % len(t.qubits)]
        out["qubits"][name], = ax.plot([f01, f01], [0.50, 0.86], color=c, lw=1.8, solid_capstyle="butt", zorder=5)
        ax.plot([f12, f12], [0.50, 0.70], color=c, lw=1.0, ls=(0, (2, 1.2)), zorder=5)
        ax.text(f01, 0.88, name, ha="center", va="bottom", fontsize=fs, color=t.ink, zorder=6)
        s = spread.get(name) if isinstance(spread, dict) else spread
        if s:
            _errorbar(ax, [f01], [0.53], t.ink, xerr=[[s], [s]], capsize=1.8, zorder=6)
    prev, low = None, False
    for name, fr in sorted(res.items(), key=lambda kv: kv[1]):
        low = (not low) if prev is not None and fr - prev < 0.035 * span else False
        top = 0.28 if low else 0.36
        out["resonators"][name], = ax.plot([fr, fr], [0.08, top], color=t.bus, lw=1.6, solid_capstyle="butt",
                                           zorder=5)
        ax.text(fr, top + 0.02, name, ha="center", va="bottom", fontsize=fs, color=t.ink, zorder=6)
        prev = fr
    if mass_axis:
        out["top"] = ax.secondary_xaxis("top", functions=(ph.mass_from_frequency, ph.frequency_from_mass))
        out["top"].set_xlabel(r"rest energy $mc^2$ (µeV)")
        ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlabel(f"frequency ({unit})")
    return out


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


def junction_curve(ax, x, EC, *, kind="Rn", gap=None, points=None, spread=None, cap_spread=None, window=None,
                   band=None, top_axis=True, ncut=30, legend=True):
    """Transmon frequency f01 against the junction's normal-state resistance (kind="Rn", x in kΩ) or
    its Josephson inductance (kind="LJ", x in nH), on the exact charge-basis spectrum: E_J from L_J
    (physics.ej_from_lj) and, for R_n, L_J from the Ambegaokar-Baratoff relation with the gap in eV
    (physics.lj_from_rn). EC (GHz) is one charging energy or {name: EC}, one curve each in its qubit
    colour with a legend (legend=False leaves it out, for curves the point labels already name).
    points {name: x0} marks each design on its curve with an open marker and its value; with several
    curves the names must match. spread, a relative junction scatter such as 0.03,
    adds error bars of +-spread x0 and the f01 range that scatter causes. cap_spread, a relative
    capacitance error, draws each curve again dotted for C (1 +- cap_spread). window (lo, hi) marks an
    allowed L_J range in nH with dashed guides, and band (lo, hi) shades a specified f01 range in GHz.
    top_axis puts the other junction quantity on the top axis, which needs the gap. Returns
    {"f01": {name: GHz on x}, "points": {name: (x0, f01, (f_lo, f_hi) or None)}, "labels": {name: the
    point's label, to move where curves crowd it}}."""
    t = theme()
    if kind not in ("Rn", "LJ"):
        raise ValueError(f"kind must be 'Rn' or 'LJ', not {kind!r}")
    if kind == "Rn" and gap is None:
        raise ValueError("kind='Rn' needs the superconducting gap in eV")
    x = np.asarray(x, float)
    to_lj = (lambda v: ph.lj_from_rn(np.asarray(v, float) * 1e3, gap)) if kind == "Rn" else \
        (lambda v: np.asarray(v, float) * 1e-9)

    def f01(v, ec):
        return np.array([ph.transmon_spectrum(ej, ec, ncut=ncut, nlev=2)[1]
                         for ej in np.atleast_1d(ph.ej_from_lj(to_lj(v)))])

    ecs = dict(EC) if isinstance(EC, dict) else {None: EC}
    colour = {name: t.qubits[k % len(t.qubits)] for k, name in enumerate(ecs)}
    out = {"f01": {}, "points": {}, "labels": {}}
    if band:
        ax.axhspan(band[0], band[1], fc=t.tint, ec="none", zorder=0)
        ax.text(0.97, band[1], f"{band[0]:g}–{band[1]:g} GHz band", transform=ax.get_yaxis_transform(), ha="right",
                va="top", fontsize=t.size - 1, color=t.ink, zorder=6)
    for name, ec in ecs.items():
        out["f01"][name] = f = f01(x, ec)
        ax.plot(x, f, label=name, **model_kw(color=colour[name]))
        if cap_spread:
            for s in (1 + cap_spread, 1 - cap_spread):
                ax.plot(x, f01(x, ec / s), **model_kw(color=colour[name], ls=(0, (1, 1.5)), lw=0.7))
    unit = "kΩ" if kind == "Rn" else "nH"
    for name, x0 in (points or {}).items():
        if len(ecs) > 1 and name not in ecs:
            raise ValueError(f"point {name!r} names no curve; the curves are {list(ecs)}")
        key = name if name in ecs else next(iter(ecs))
        f0 = float(f01(x0, ecs[key])[0])
        rng = None
        if spread:
            rng = tuple(sorted(f01([x0 * (1 - spread), x0 * (1 + spread)], ecs[key])))
            _errorbar(ax, [x0], [f0], colour[key], xerr=[[spread * x0], [spread * x0]],
                      yerr=[[f0 - rng[0]], [rng[1] - f0]], capsize=1.5, zorder=4)
        ax.plot(x0, f0, **sim_kw(color=colour[key], ms=4.5, zorder=5))
        out["labels"][name] = ax.annotate(f"{name}\n{x0:.3g} {unit}", (x0, f0), xytext=(5, 3),
                                          textcoords="offset points", ha="left", va="bottom", fontsize=t.size - 1,
                                          color=t.ink, linespacing=1.0, zorder=6)
        out["points"][name] = (x0, f0, rng)
    if window:
        edge = ax.get_xaxis_transform()
        for L, ha in zip(sorted(window), ("left", "right")):        # each label inside the window
            xv = ph.rn_from_lj(L * 1e-9, gap) / 1e3 if kind == "Rn" else L
            ax.axvline(xv, **ref_kw())
            ax.text(xv, 0.97, rf" $L_J$ = {L:g} nH ", transform=edge, ha=ha, va="top", fontsize=t.size - 1,
                    color=t.reference)
    if top_axis and gap is not None:
        if kind == "Rn":
            fw, bw = (lambda r: ph.lj_from_rn(r * 1e3, gap) * 1e9), (lambda L: ph.rn_from_lj(L * 1e-9, gap) / 1e3)
            top_label = r"$L_J$ (nH)"
        else:
            fw, bw = (lambda L: ph.rn_from_lj(L * 1e-9, gap) / 1e3), (lambda r: ph.lj_from_rn(r * 1e3, gap) * 1e9)
            top_label = r"$R_n$ (kΩ)"
        top = ax.secondary_xaxis("top", functions=(fw, bw))
        top.set_xlabel(top_label)
        ax.tick_params(axis="x", which="both", top=False)
    if len(ecs) > 1 and legend:
        ax.legend(loc="lower left")
    lo = min(f.min() for f in out["f01"].values())
    hi = max(f.max() for f in out["f01"].values())
    if band:
        lo, hi = min(lo, band[0]), max(hi, band[1])
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.16 * (hi - lo))    # headroom for the window labels
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(r"$R_n$ (kΩ)" if kind == "Rn" else r"$L_J$ (nH)")
    ax.set_ylabel(r"$f_{01}$ (GHz)")
    return out


def capacitance_matrix(ax, C, labels, *, unit="fF", digits=3, cbar=True, fontsize=None):
    """The Maxwell capacitance matrix C (in unit) as annotated cells: total capacitance of each net on
    the diagonal, the negative mutual capacitances off it. Cells are coloured by |C| on a log scale,
    so couplings across several decades stay visible, and print the signed value; zero cells stay
    blank. Returns (image, colour bar or None)."""
    C = np.asarray(C, float)
    signed = [[cell_text(f"{{:.{digits}g}}", c) if c else "" for c in row] for row in C]   # the signed Maxwell entry
    return _log_cells(ax, C, labels, signed, rf"$|C_{{ij}}|$ ({unit})", cbar, fontsize)


def _log_cells(ax, M, labels, text, label, cbar, fontsize):
    """M as annotated cells coloured by |M| on a log scale and printed with the strings in text; zero
    cells stay blank."""
    A = np.abs(M)
    norm = LogNorm(vmin=A[A > 0].min(), vmax=A.max())
    return annotated_matrix(ax, A, labels, norm=norm, text=text, cbar=cbar, label=label, fontsize=fontsize,
                            skip=lambda i, j, v: v == 0)


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
                     loc="upper left", models=None, points=None, label=None, detuning=False, legend_loc="best",
                     formula=True):
    """Dispersive shift chi = g^2 alpha / (Delta (Delta + alpha)), Delta = fq - fr, against qubit
    frequency fq (GHz), from physics.dispersive_shift and drawn in MHz. The straddling regime
    0 < Delta < |alpha| is shaded and the poles at Delta = 0 and Delta = -alpha are marked; the curve
    is broken at each pole. The default symlog axis shows the small negative chi of a detuned qubit
    and the large positive chi inside the straddling regime together; it is linear within +-linthresh
    (default: the decade below |chi| at the end of the range farthest from fr). scale="linear" with
    ylim zooms on one branch; scale="log" draws |chi| on a log axis, for comparing models over several
    decades. mark=fq0 marks a design point and prints its chi at loc.

    models {label: chi} overlays other calculations of chi in GHz (physics.transmon_dressed gives the
    exact one): each value is an array on fq, a callable chi(fq), a pair (fq', chi) for a curve along
    another sweep, or a triple (fq', chi, style) whose dict of line keywords (color, ls, lw) replaces
    the default look. They take the series colours with dashes, and a legend at legend_loc (None for
    none) names every curve, with label naming the formula curve. formula=False leaves the formula
    curve out, for models computed elsewhere along a sweep in which g and alpha change. points
    {label: (fq, chi), (fq, chi, kind) or (fq, chi, kind, style)} marks published or simulated values
    (chi in GHz) with the marker grammar. The straddling regime and the poles are labelled only where
    they fall inside the range of fq. detuning=True puts Delta = fq - fr on the x axis. Returns chi
    (MHz) on fq."""
    t = theme()
    if scale not in ("symlog", "linear", "log"):
        raise ValueError(f"scale must be 'symlog', 'linear' or 'log', not {scale!r}")
    fq = np.asarray(fq, float)
    X = (lambda f: np.asarray(f, float) - fr) if detuning else (lambda f: np.asarray(f, float))
    shown = np.abs if scale == "log" else (lambda v: v)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = 1e3 * ph.dispersive_shift(fq, fr, g, alpha)
    far = abs(1e3 * ph.dispersive_shift(fq[np.argmax(np.abs(fq - fr))], fr, g, alpha))
    if ylim is None:
        top = 10.0 ** np.ceil(np.log10(8e3 * g ** 2 / abs(alpha))) if scale != "linear" else 10 * far
        if scale == "log":
            low = np.abs(chi[np.isfinite(chi) & (chi != 0)])
            ylim = (10.0 ** np.floor(np.log10(low.min())), top)
        else:
            ylim = (-top, top)
    if scale == "symlog":
        lt = linthresh or 10.0 ** np.floor(np.log10(far))
        ax.set_yscale("symlog", linthresh=lt, linscale=0.5)
        dec = 10.0 ** np.arange(np.log10(lt) + 1, np.log10(max(abs(ylim[0]), abs(ylim[1]))) + 0.5)
        ax.yaxis.set_major_locator(FixedLocator(np.r_[-dec[::-1], 0.0, dec]))
        ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10, linthresh=lt, subs=np.arange(2, 10)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", _MINUS)))
    elif scale == "log":
        ax.set_yscale("log")
    big = 3 * max(abs(ylim[0]), abs(ylim[1]))
    curve = np.where(np.abs(chi) > big, np.nan, shown(chi))
    lo, hi = sorted((float(X(fr)), float(X(fr - alpha))))
    left, right = float(X(fq.min())), float(X(fq.max()))
    edge = ax.get_xaxis_transform()                                  # x in data, y in axes
    ax.axvspan(lo, hi, fc=lighten(t.accent, 0.72), ec="none", zorder=0)
    # a regime or pole off the range of fq keeps its (unseen) shading and guide but no label, which
    # would otherwise print outside the frame
    if lo < right and hi > left:
        ax.text((max(lo, left) + min(hi, right)) / 2, 0.5, "straddling", transform=edge, rotation=90, ha="center",
                va="center", fontsize=t.size - 1.5, color=t.ink, bbox=dict(fc=lighten(t.accent, 0.72), ec="none",
                                                                            pad=0.5))
    if scale != "log":
        ax.axhline(0, color=t.light, ls="-", lw=0.6, zorder=1)
    for f0, s, ha in ((fr, r"$\Delta = 0$ ", "right"), (fr - alpha, r" $\Delta = -\alpha$", "left")):
        ax.axvline(X(f0), **ref_kw())
        if left <= float(X(f0)) <= right:
            ax.text(X(f0), 0.97, s, transform=edge, ha=ha, va="top", fontsize=t.size - 1, color=t.ink)
    handles = []
    if formula:
        handles.append(ax.plot(X(fq), curve, label=label or r"$g^2\alpha/\Delta(\Delta+\alpha)$", **model_kw())[0])
    for i, (name, m) in enumerate((models or {}).items()):
        style = {}
        if callable(m):
            xm, ym = fq, m(fq)
        elif isinstance(m, tuple) and len(m) in (2, 3):
            xm, ym = m[:2]
            style = dict(m[2]) if len(m) == 3 else {}
        else:
            xm, ym = fq, m
        xm, ym = np.asarray(xm, float), 1e3 * np.asarray(ym, float)
        if xm.shape != ym.shape:
            raise ValueError(f"model {name!r}: {ym.size} values of chi for {xm.size} frequencies")
        ym = np.where(np.abs(ym) > big, np.nan, shown(ym))
        kw = dict(color=series(i + 1)["color"], ls=_MODEL_DASHES[i % len(_MODEL_DASHES)], lw=t.lw, zorder=3)
        kw.update(style)
        handles.append(ax.plot(X(xm), ym, label=name, **kw)[0])
    if points:
        _points(ax, {k: (X(p[0]), shown(1e3 * p[1]), *p[2:]) for k, p in points.items()})
    if mark is not None:
        c0 = 1e3 * ph.dispersive_shift(mark, fr, g, alpha)
        ax.plot(X(mark), shown(c0), **sim_kw(color=t.accent, ms=4.5, zorder=5))
        note(ax, rf"$\chi/2\pi$ = {c0:.3g} MHz" "\n" rf"at $f_q$ = {mark:g} GHz".replace("-", _MINUS), loc=loc)
    if (models or points) and legend_loc and handles:
        ax.legend(handles=handles, loc=legend_loc)
    ax.set_xlim(X(fq.min()), X(fq.max()))
    ax.set_ylim(*ylim)
    if scale == "log":
        _log_ticks(ax, "y")
    ax.set_xlabel(r"detuning $\Delta = f_q - f_r$ (GHz)" if detuning else r"qubit frequency $f_q$ (GHz)")
    ax.set_ylabel(r"$|\chi|/2\pi$ (MHz)" if scale == "log" else r"dispersive shift $\chi/2\pi$ (MHz)")
    return chi


def _duration(us):
    """A time in µs printed with two significant figures, in ms from 1000 µs."""
    if us >= 1e3:
        return f"{us / 1e3:.2g} ms" if us < 1e5 else f"{us / 1e3:.0f} ms"
    return f"{us:.2g} µs" if us < 100 else f"{us:.0f} µs"


def purcell_limit(ax, fq, fr, g, kappa, *, filter_q=None, mark=None, target=None, loc="lower left",
                  detuning=False, mark_label=None):
    """Purcell-limited T1 against qubit frequency fq (GHz) on a log axis, from physics.purcell_t1, for
    a qubit coupled with strength g to a readout resonator at fr of linewidth kappa (all GHz).
    filter_q adds the curve with a bandpass Purcell filter of that quality factor. mark=fq0 marks the
    design point on each curve with its limit, printed as a duration or with mark_label, a format
    string that takes T1 in µs; target (µs) draws a guide at that T1. The resonator frequency is
    marked when it lies in range. detuning=True puts Delta = fq - fr on the x axis. The legend sits at
    loc (None for none). Returns {"bare": T1, "filter": T1 or None} in µs."""
    t = theme()
    fq = np.asarray(fq, float)
    X = (lambda f: np.asarray(f, float) - fr) if detuning else (lambda f: np.asarray(f, float))
    out = {"bare": ph.purcell_t1(fq, fr, g, kappa), "filter": None}
    curves = [("bare", "no filter", None)]
    if filter_q:
        out["filter"] = ph.purcell_t1(fq, fr, g, kappa, filter_q)
        curves.append(("filter", rf"Purcell filter, $Q_F$ = {filter_q:g}", filter_q))
    for i, (key, label, q) in enumerate(curves):
        ax.plot(X(fq), out[key], label=label, **{**series(i), "marker": None, "lw": t.lw})
        if mark is not None:
            T0 = float(ph.purcell_t1(mark, fr, g, kappa, q))
            ax.plot(X(mark), T0, **sim_kw(color=series(i)["color"], ms=4.5, zorder=5))
            ax.annotate(_duration(T0) if mark_label is None else mark_label.format(T0), (X(mark), T0), xytext=(5, 2),
                        textcoords="offset points", ha="left", va="bottom", fontsize=t.size - 1, color=t.ink)
    if target:
        ax.axhline(target, **ref_kw())
        ax.text(X(fq.min()), target, f" target {_duration(target)}", ha="left", va="bottom", fontsize=t.size - 1.5,
                color=t.reference)
    if fq.min() < fr < fq.max():
        ax.axvline(X(fr), **ref_kw())
    ax.set_yscale("log")
    ax.set_xlim(X(fq.min()), X(fq.max()))
    ax.set_xlabel(r"detuning $\Delta = f_q - f_r$ (GHz)" if detuning else r"qubit frequency $f_q$ (GHz)")
    ax.set_ylabel(r"Purcell limit $T_1$ (µs)")
    if loc:
        ax.legend(loc=loc)
    return out


def admittance(ax, f, B, *, modes=None, labels=None, unit="mS", ylim=None):
    """Susceptance Im Y that a junction sees, against frequency f (GHz), with B in siemens drawn in unit
    ("mS" or "S"). This is the curve black-box quantization reads. The curve breaks at each pole, where
    B falls through infinity, and a guide line marks the pole. modes, the dict from physics.black_box,
    rings each mode inside the range at its zero of B in the accent colour and draws the tangent there,
    whose slope sets the mode's impedance Z. Each mode is labelled with Z and phi_zpf; labels names the
    modes ("q", "r"). ylim defaults to the range that holds most of the curve. Returns
    {"curve": Line2D, "poles": GHz, "marks": one Line2D per ringed mode}."""
    t = theme()
    f = np.asarray(f, float)
    s = {"mS": 1e3, "S": 1.0}[unit]
    y = s * np.asarray(B, float)
    fall = np.nonzero((y[:-1] > 0) & (y[1:] < 0))[0]
    with np.errstate(divide="ignore"):
        u = 1 / y                                                   # 1/B passes linearly through each pole
    poles = f[fall] - u[fall] * (f[fall + 1] - f[fall]) / (u[fall + 1] - u[fall])
    if ylim is None:
        top = 1.3 * np.percentile(np.abs(y), 90)
        ylim = (-top, top)
    y = np.where(np.abs(y) > 3 * max(np.abs(ylim)), np.nan, y)
    y[fall] = np.nan
    ax.axhline(0, color=t.light, ls="-", lw=0.6, zorder=1)
    for p in poles:
        ax.axvline(p, **ref_kw())
    curve, = ax.plot(f, y, **model_kw())
    marks = []
    if modes is not None:
        span = ylim[1] - ylim[0]
        for m, (fm, Z, phi) in enumerate(zip(modes["f"], modes["Z"], modes["phi"])):
            if not f.min() <= fm <= f.max():
                continue
            slope = s * 2 / (fm * Z)                                # dB/df at the mode
            d = min(0.04 * (f.max() - f.min()), 0.18 * span / slope)
            ax.plot([fm - d, fm + d], [-slope * d, slope * d], color=t.accent, ls="-", lw=1.4, zorder=4)
            mark, = ax.plot(fm, 0.0, **sim_kw(color=t.accent, ms=6, zorder=5))
            marks.append(mark)
            name = labels[m] if labels else str(m)
            ax.annotate(rf"$Z_{{{name}}}$ = {Z:.3g} Ω" "\n" rf"$\varphi_{{{name}}}$ = {phi:.3g}", (fm, 0.0),
                        xytext=(6, -6), textcoords="offset points", ha="left", va="top", fontsize=t.size - 1,
                        color=t.ink, zorder=6)
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", _MINUS)))
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(rf"susceptance Im $Y$ ({unit})")
    return {"curve": curve, "poles": poles, "marks": marks}


def kerr_matrix(ax, K, labels, *, exact=None, digits=3, cbar=True, fontsize=None):
    """Kerr matrix of coupled modes (GHz, as physics.black_box returns it), printed in MHz: each mode's
    anharmonicity on the diagonal and the cross-Kerr shifts off it. Cells are coloured by |K| on a log
    scale, which spans the decades between a transmon and a resonator. exact, a matrix of the same
    shape (from physics.transmon_resonator, say), prints its value in brackets under each entry.
    Returns (image, colour bar or None)."""
    fmt = lambda v: cell_text(f"{{:.{digits}g}}", v)
    K = 1e3 * np.asarray(K, float)
    text = [[fmt(v) for v in row] for row in K]
    if exact is not None:
        X = 1e3 * np.asarray(exact, float)
        text = [[f"{a}\n({fmt(x)})" for a, x in zip(row, xrow)] for row, xrow in zip(text, X)]
    return _log_cells(ax, K, labels, text, r"$|K_{mn}|/2\pi$ (MHz)", cbar, fontsize)


def convergence(ax, passes, values, *, label=None, tol=None, xlabel="adaptive pass", loc="upper right",
                error=False, reference=None, floor=None, floor_label="floor"):
    """A simulated quantity against adaptive pass (or mesh cutoff) on ax, and the relative change
    between successive passes |x_n - x_(n-1)| / |x_n| on a log axis to the right. tol draws the
    convergence target there and rings the first pass that meets it. The legend at loc names the two
    series.

    error=True draws instead the absolute error |x_n - x_ref| on one log axis, with x_ref the
    reference value (default: the last pass, whose zero error is left out, as is any exact zero).
    tol then rings the first pass whose error is at most tol, and floor draws a guide at that error,
    a numerical floor or a required accuracy, labelled floor_label. Over more than six decades the
    minor ticks fall once per decade. label names the y axis in either mode.

    Returns {"rel": relative changes or None, "err": errors or None, "ax2": the right axes or None,
    "converged": that pass or None}."""
    t = theme()
    n, v = np.asarray(passes), np.asarray(values, float)
    if n.shape != v.shape:
        raise ValueError(f"{v.size} values for {n.size} passes")
    if error:
        return _error_convergence(ax, n, v, label, tol, xlabel, reference, floor, floor_label)
    label = "simulated value" if label is None else label
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
    return {"rel": rel, "err": None, "ax2": ax2, "converged": done}


def _error_convergence(ax, n, v, label, tol, xlabel, reference, floor, floor_label):
    """The error mode of convergence: |x_n - x_ref| against pass on one log axis."""
    t = theme()
    last = reference is None
    ref = v[-1] if last else float(reference)
    err = np.abs(v - ref)
    shown = np.where(err > 0, err, np.nan)
    if last:
        shown[-1] = np.nan
    if not np.isfinite(shown).any():
        raise ValueError("every error is zero, so there is nothing to draw on a log axis")
    ax.plot(n, shown, **sim_kw(ls="-", lw=0.8))
    ax.set_yscale("log")
    done = None
    if floor:
        ax.axhline(floor, **ref_kw())
        ax.text(n[-1], floor, f"{floor_label} ", ha="right", va="bottom", fontsize=t.size - 1.5, color=t.reference)
    if tol:
        ax.axhline(tol, **ref_kw())
        ax.text(n[0], tol, " target", ha="left", va="bottom", fontsize=t.size - 1.5, color=t.reference)
        ok = np.nonzero((err <= tol) & np.isfinite(shown))[0]
        if ok.size:
            done = n[ok[0]]
            ax.plot(done, err[ok[0]], **sim_kw(color=t.accent, marker="o", ms=8, mew=0.8, mfc="none", zorder=5))
            ax.annotate(f"converged, pass {done}", (done, err[ok[0]]), xytext=(6, 5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=t.size - 1, color=t.ink)
    _log_ticks(ax, "y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"error $|x_n - x_\mathrm{ref}|$" if label is None else label)
    return {"rel": None, "err": err, "ax2": None, "converged": done}


_ARROWS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def design_map(ax, x, y, Z, *, levels, fill=True, fmt=None, emphasis=None, points=None, arrows=None,
               xscale="log", yscale="log", xlabel=None, ylabel=None):
    """Contours of a computed figure of merit Z (shape len(y) x len(x)) on the x-y design plane: ink
    lines at levels, each labelled inline by fmt (a format string or callable; default the value),
    over pale tints of the style's sequential map that fill the bands between them (fill=False leaves
    the plane white). emphasis names one level drawn heavier, a break-even line, say. points
    {label: (x, y) or (x, y, kind)} places devices or designs with the marker grammar (kind "data" for
    measured, "sim" for designed, "assumed" for values from elsewhere), labelled beside each marker.
    arrows {label: direction or list of directions} adds an arrow from that point ("up", "down",
    "left", "right") for a value that is only a bound. Both axes are logarithmic unless xscale or
    yscale say otherwise. Returns {"lines": ContourSet, "fill": ContourSet or None,
    "points": {label: Line2D}}."""
    t = theme()
    x, y, Z = np.asarray(x, float), np.asarray(y, float), np.asarray(Z, float)
    if Z.shape != (y.size, x.size):
        raise ValueError(f"Z has shape {Z.shape}; it must be (len(y), len(x)) = {(y.size, x.size)}")
    levels = np.sort(np.atleast_1d(np.asarray(levels, float)))
    if fill and levels.size < 2:
        raise ValueError("filled bands need at least two levels; pass fill=False for a single contour")
    for name in arrows or {}:
        if name not in (points or {}):
            raise ValueError(f"arrow {name!r} names no point")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    filled = None
    if fill:
        cmap = mpl.colormaps[t.cmap]
        colors = [to_hex(cmap(s)) for s in np.linspace(0.0, 0.42, levels.size + 1)]
        filled = ax.contourf(x, y, Z, levels=levels, colors=colors, extend="both", zorder=0)
    widths = [1.1 if emphasis is not None and np.isclose(v, emphasis) else 0.6 for v in levels]
    lines = ax.contour(x, y, Z, levels=levels, colors=[t.ink], linewidths=widths, zorder=2)
    if fmt is None:
        fmt = lambda v: f"{v:g}".replace("-", _MINUS)
    elif isinstance(fmt, str):
        fmt = (lambda s: lambda v: s.format(v).replace("-", _MINUS))(fmt)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    spots = [max(segs, key=len)[len(max(segs, key=len)) // 2] for segs in lines.allsegs if segs]
    if spots:                                                       # each label mid-line, clear of the frame
        ax.clabel(lines, fmt=fmt, fontsize=t.size - 1.5, inline=True, manual=spots)
    handles = _points(ax, points, zorder=5)
    for name, dirs in (arrows or {}).items():
        for d in [dirs] if isinstance(dirs, str) else dirs:
            if d not in _ARROWS:
                raise ValueError(f"arrow direction {d!r} is not one of {sorted(_ARROWS)}")
            dx, dy = _ARROWS[d]
            ax.annotate("", (points[name][0], points[name][1]), xytext=(15 * dx, 15 * dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="<|-", color=t.ink, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=4),
                        zorder=5)
    for scale, axis in ((xscale, "x"), (yscale, "y")):
        if scale == "log":
            log_minor(ax, axis)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return {"lines": lines, "fill": filled, "points": handles}
