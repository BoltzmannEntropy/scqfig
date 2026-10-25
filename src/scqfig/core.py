"""Style switching, figure sizes, saving, panel letters and the marker grammar every plot shares.

Figures are drawn at print size, so the font sizes in themes.py are the printed sizes: a column
figure from figure("column") goes into the paper at width=\\columnwidth without scaling."""
import argparse
import contextlib
import dataclasses
import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from .themes import FIELD, THEMES

# Printed widths in inches. "column" and "page" suit revtex, IOP and most two-column journals.
WIDTHS = {
    "column": 3.4,
    "page": 7.0,
    "aps-column": 3.375,
    "aps-page": 7.0,
    "nature-single": 3.504,
    "nature-double": 7.205,
    "ieee-column": 3.5,
    "ieee-page": 7.16,
    "thesis": 6.0,
    "slide": 10.0,
}

_current = [FIELD]
_applied = [False]


def theme():
    """The active Theme; every plot and schematic takes its colours from here at call time."""
    return _current[0]


def _register_cmaps():
    for t in THEMES.values():
        for name, stops in ((t.cmap, t.cmap_stops), (t.cmap_div, t.cmap_div_stops)):
            if name not in mpl.colormaps:
                cm = LinearSegmentedColormap.from_list(name, list(stops), N=256)
                mpl.colormaps.register(cm)
                mpl.colormaps.register(cm.reversed(), name=name + "_r")


def _cache_dir():
    root = os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache"
    return pathlib.Path(root) / "scqfig" / "fonts"


def _register_helvetica():
    """macOS ships Helvetica as a .ttc collection, of which matplotlib reads only the first face, so
    bold and oblique labels would fall back to the regular face. Extract the regular, bold and oblique
    faces once into a user cache, renamed HelveticaX. The extracted files stay on this machine."""
    ttc = pathlib.Path("/System/Library/Fonts/Helvetica.ttc")
    if not ttc.exists():
        return
    cache = _cache_dir()
    faces = {0: "Regular", 1: "Bold", 2: "Italic"}
    try:
        if not all((cache / f"HelveticaX-{s}.ttf").exists() for s in faces.values()):
            from fontTools.ttLib import TTCollection
            cache.mkdir(parents=True, exist_ok=True)
            col = TTCollection(str(ttc))
            for i, sub in faces.items():
                f = col.fonts[i]
                for rec in f["name"].names:
                    if rec.nameID in (1, 16):
                        rec.string = "HelveticaX"
                    elif rec.nameID in (2, 17):
                        rec.string = sub
                    elif rec.nameID == 4:
                        rec.string = f"HelveticaX {sub}"
                    elif rec.nameID == 6:
                        rec.string = f"HelveticaX-{sub}"
                f["OS/2"].usWeightClass = 700 if sub == "Bold" else 400
                f.save(cache / f"HelveticaX-{sub}.ttf")
        have = {f.fname for f in font_manager.fontManager.ttflist}
        for p in sorted(cache.glob("HelveticaX-*.ttf")):
            if str(p) not in have:
                font_manager.fontManager.addfont(str(p))
    except Exception:
        pass


def available_fonts(t=None):
    """The theme's fonts that are installed, in preference order."""
    t = t or theme()
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in t.fonts if n in have] or ["DejaVu Sans" if t.family == "sans-serif" else "DejaVu Serif"]


def rc_params(t, usetex=False):
    """The matplotlib settings of theme t."""
    if t.family == "sans-serif":
        _register_helvetica()
    fonts = available_fonts(t)
    rc = {
        "font.family": t.family,
        f"font.{t.family}": fonts,
        "font.size": t.size,
        "axes.labelsize": t.size + 0.5,
        "axes.titlesize": t.size + 0.5,
        "xtick.labelsize": t.size,
        "ytick.labelsize": t.size,
        "legend.fontsize": t.size - 0.5,
        "legend.title_fontsize": t.size - 0.5,
        "figure.titlesize": t.size + 1,
        "axes.linewidth": t.axes_lw,
        "axes.edgecolor": t.ink,
        "axes.labelcolor": t.ink,
        "axes.titlecolor": t.ink,
        "text.color": t.ink,
        "xtick.color": t.ink,
        "ytick.color": t.ink,
        "axes.prop_cycle": cycler(color=list(t.cycle)),     # dashes and markers via series(i)
        "image.cmap": t.cmap,
        "xtick.direction": t.ticks,
        "ytick.direction": t.ticks,
        "xtick.top": t.box,
        "ytick.right": t.box,
        "axes.spines.top": t.box,
        "axes.spines.right": t.box,
        "xtick.major.width": t.axes_lw * 0.85,
        "ytick.major.width": t.axes_lw * 0.85,
        "xtick.minor.width": t.axes_lw * 0.65,
        "ytick.minor.width": t.axes_lw * 0.65,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "axes.labelpad": 2.5,
        "axes.titlepad": 4.0,
        "axes.grid": False,
        "grid.color": t.light,
        "grid.linewidth": 0.4,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "legend.frameon": t.box,
        "legend.edgecolor": t.ink,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.4,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.5,
        "lines.linewidth": t.lw,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.8,
        "errorbar.capsize": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.usetex": usetex,
    }
    if t.family == "serif":
        rc["mathtext.fontset"] = "stix"
    elif fonts[0] == "DejaVu Sans":
        rc["mathtext.fontset"] = "dejavusans"
    else:
        rc.update({"mathtext.fontset": "custom", "mathtext.rm": fonts[0], "mathtext.it": f"{fonts[0]}:italic",
                   "mathtext.bf": f"{fonts[0]}:bold", "mathtext.sf": fonts[0], "mathtext.fallback": "stixsans"})
    if usetex:
        rc["text.latex.preamble"] = t.usetex_preamble
    return rc


def use(name="field", usetex=False, **overrides):
    """Switch every later figure to style `name` ("field" or "folio"). usetex=True sets all text with
    LaTeX in the matching face (helvet + sansmath, or stix2), at the cost of slower rendering.
    Keyword overrides replace Theme fields for this session, for example use("field", size=8.5) for a
    document whose figures are set half a point larger. Returns the active Theme."""
    if name not in THEMES:
        raise ValueError(f"unknown style {name!r}; choose one of {sorted(THEMES)}")
    _register_cmaps()
    t = dataclasses.replace(THEMES[name], **overrides) if overrides else THEMES[name]
    mpl.rcParams.update(rc_params(t, usetex))
    _current[0] = t
    _applied[0] = True
    return t


@contextlib.contextmanager
def style(name, usetex=False, **overrides):
    """Draw inside the block in style `name`, then restore the previous style and settings."""
    prev = _current[0]
    with mpl.rc_context():
        t = use(name, usetex, **overrides)
        try:
            yield t
        finally:
            _current[0] = prev


def figure(width="column", aspect=0.75, nrows=1, ncols=1, height=None, layout="constrained", **kw):
    """A figure at print width (a WIDTHS key or inches) with nrows x ncols axes. Height is
    width * aspect unless given in inches. Returns (fig, ax) or (fig, axes array). Applies the
    default style (field) if use() has not been called."""
    if not _applied[0]:
        use(_current[0].name)
    w = WIDTHS[width] if isinstance(width, str) else float(width)
    h = float(height) if height is not None else w * aspect
    return plt.subplots(nrows, ncols, figsize=(w, h), layout=layout, **kw)


def save(fig, path, formats=("pdf", "png"), dpi=300, close=True):
    """Write fig to path.<ext> for each format (vector PDF for the paper, PNG for previews)."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.with_suffix("") if p.suffix.lstrip(".") in formats else p
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=dpi if ext == "png" else None)
    if close:
        plt.close(fig)
    return stem


def panel(ax, letter, inside=False, x=None, y=None, fontsize=None):
    """Panel letter, "(a)" by default. Outside, it sits as a left title above the frame, where
    constrained layout keeps it clear of neighbours; inside=True puts it in the upper-left corner."""
    t = theme()
    s = t.letter.format(letter)
    fs = fontsize or t.size + 1
    if inside:
        return ax.text(0.03 if x is None else x, 0.96 if y is None else y, s, transform=ax.transAxes,
                       ha="left", va="top", fontsize=fs, weight=t.letter_weight, zorder=20)
    if getattr(ax, "name", "") == "3d":
        return ax.text2D(0.0 if x is None else x, 1.0 if y is None else y, s, transform=ax.transAxes,
                         ha="left", va="bottom", fontsize=fs, weight=t.letter_weight)
    return ax.set_title(s, loc="left", fontsize=fs, fontweight=t.letter_weight,
                        **({"x": x} if x is not None else {}))


def title(ax, letter, text=None, finding=None):
    """Panel letter at the upper-left corner of the frame, flush right against it. With both a title
    and a one-line finding, the letter and the title sit on one line above the frame and the finding
    in grey on the line below; a bare title is ignored, since the caption names the panel."""
    t = theme()
    s = t.letter.format(letter)
    if text and finding:
        a = ax.text(0.0, 1.10, f"{s}  {text}", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=t.size + 0.5, weight="bold")
        b = ax.text(0.0, 1.02, finding, transform=ax.transAxes, ha="left", va="bottom", fontsize=t.size - 0.7,
                    color=t.reference)
        return a, b
    return ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=t.size + 0.5,
                   weight=t.letter_weight)


def fig_letter(fig, x, y, letter):
    """Panel letter at figure coordinates (x, y), for panels that are not a single axes."""
    t = theme()
    return fig.text(x, y, t.letter.format(letter), fontsize=t.size + 0.5, va="bottom", weight=t.letter_weight)


# ---- the marker grammar ---------------------------------------------------------------------

def data_kw(color=None, marker="o", ms=None, **kw):
    """Measured points: filled in field, open in folio."""
    t = theme()
    c = color or t.data
    d = dict(marker=marker, ls="none", mec=c, color=c, mfc=c if t.data_filled else "white", zorder=4)
    if ms:
        d["ms"] = ms
    d.update(kw)
    return d


def sim_kw(color=None, marker="o", ms=None, **kw):
    """Simulated, designed or projected values: open markers."""
    c = color or theme().sim
    d = dict(marker=marker, ls="none", mec=c, color=c, mfc="white", zorder=4)
    if ms:
        d["ms"] = ms
    d.update(kw)
    return d


def assumed_kw(marker="o", **kw):
    """Values taken from another source: open grey markers."""
    return sim_kw(theme().assumed, marker, **kw)


def model_kw(color=None, **kw):
    """Fit and theory curves."""
    d = dict(color=color or theme().model, ls="-", lw=theme().lw, zorder=3)
    d.update(kw)
    return d


def ref_kw(**kw):
    """Guide lines: thin grey dashes."""
    d = dict(color=theme().reference, ls=(0, (4, 2)), lw=0.7, zorder=2)
    d.update(kw)
    return d


def series(i):
    """Colour, line style and marker of the i-th series: ax.plot(x, y, **sq.series(i)). In folio the
    line style changes with the series, so curves stay apart in greyscale print."""
    t = theme()
    k = i % len(t.cycle)
    return dict(color=t.cycle[k], ls=t.dashes[k], marker=t.markers[k])


def log_minor(ax, axis="both"):
    """Minor ticks at 2..9 on every log axis."""
    from matplotlib.ticker import LogLocator, NullFormatter
    for name in ("x", "y") if axis == "both" else (axis,):
        a = ax.xaxis if name == "x" else ax.yaxis
        if (ax.get_xscale() if name == "x" else ax.get_yscale()) == "log":
            a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=30))
            a.set_minor_formatter(NullFormatter())


def note(ax, text, loc="upper right", fontsize=None, color=None, **kw):
    """A short text block (a fitted value, a parameter) in a corner of the axes."""
    pos = {"upper right": (0.97, 0.95, "right", "top"), "upper left": (0.03, 0.95, "left", "top"),
           "lower right": (0.97, 0.05, "right", "bottom"), "lower left": (0.03, 0.05, "left", "bottom"),
           "center right": (0.97, 0.5, "right", "center"), "upper center": (0.5, 0.95, "center", "top")}
    x, y, ha, va = pos[loc]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize or theme().size - 0.5,
                   color=color or theme().ink, linespacing=1.25, zorder=20, **kw)


def pm(value, err=None, unit="", digits=None):
    """Format a value with its uncertainty to the error's significant figure: pm(45.23, 0.81, "µs")
    gives "45.2 ± 0.8 µs"; a leading 1 in the error keeps two figures."""
    u = f" {unit}" if unit else ""
    if err is None or not np.isfinite(err) or err <= 0:
        d = 3 if digits is None else digits
        return f"{value:.{d}g}{u}"
    e = float(err)
    exp = int(np.floor(np.log10(e)))
    sig = 2 if f"{e:e}"[0] == "1" else 1
    dec = max(0, -(exp - sig + 1))
    return f"{value:.{dec}f} ± {e:.{dec}f}{u}"


MINUS = "\u2212"    # typographic minus for text that is not set by mathtext


def sci(v, err=None, digits=2):
    """Mathtext for a value: plain between 0.1 and 999 (0.92, 120), otherwise 2.4 x 10^5. With an
    error it reads (2.41 +- 0.03) x 10^5, rounded to the error as pm rounds."""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return f"${v:g}$"
    k = int(np.floor(np.log10(abs(v))))
    plain = -1 <= k <= 2
    if err is not None and np.isfinite(err) and err > 0:
        if plain:
            return "$" + pm(v, err).replace("±", r"\pm") + "$"
        return "$(" + pm(v / 10.0 ** k, err / 10.0 ** k).replace("±", r"\pm") + rf")\times10^{{{k}}}$"
    if plain:
        return f"${v:.{max(0, digits - 1 - k)}f}$"
    m = round(v / 10.0 ** k, digits - 1)
    if abs(m) >= 10:
        m, k = m / 10, k + 1
    return rf"${m:.{digits - 1}f}\times10^{{{k}}}$"


def demo_main(build, file):
    """Command line of each example: python examples/t1_decay.py --style folio [--out path]."""
    ap = argparse.ArgumentParser(description=(build.__doc__ or "").strip())
    ap.add_argument("--style", default="field", choices=sorted(THEMES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--usetex", action="store_true")
    a = ap.parse_args()
    use(a.style, a.usetex)
    fig = build()
    out = a.out or f"{pathlib.Path(file).stem}_{a.style}"
    stem = save(fig, out)
    print(f"wrote {stem}.pdf and {stem}.png")


_register_cmaps()
