"""Check a finished figure against the colour rule: every colour must be one of the four hues
(themes.HUES), a tint of one, or a grey, and every colormap one of the styles' maps. Contributors run
this on their examples; tests/test_examples.py runs it on all of them in both styles."""
import colorsys

import matplotlib as mpl
import numpy as np
from matplotlib.collections import Collection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from .themes import HUES, PALETTE, THEMES

_HUE_RGB = [np.array(to_rgb(PALETTE[h])) for h in HUES]
CMAPS = {n for t in THEMES.values() for base in (t.cmap, t.cmap_div) for n in (base, base + "_r")}
HUE_BANDS = ((200.0, 335.0), (35.0, 48.0))    # blue through purple to magenta; orange and its tints


def cmap_ok(name, n=256, sat=0.12):
    """True if every colour of the colormap is a grey (saturation below sat, or nearly black) or has
    a hue in HUE_BANDS. Interpolating orange straight to magenta passes through red and fails."""
    for r, g, b, _ in mpl.colormaps[name](np.linspace(0, 1, n)):
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s >= sat and v >= 0.12 and not any(lo <= 360 * h <= hi for lo, hi in HUE_BANDS):
            return False
    return True


def allowed(color, tol=0.03):
    """True if the colour, blended over white by its alpha, is a grey or a tint of an allowed hue."""
    r, g, b, a = to_rgba(color)
    c = np.array([r, g, b]) * a + (1 - a)
    if np.ptp(c) <= tol:
        return True
    for h in _HUE_RGB:
        f = np.mean((c - h) / (1 - h))
        if -tol <= f <= 1 + tol and np.max(np.abs(h + (1 - h) * f - c)) <= tol:
            return True
    return False


def _colours(a):
    if isinstance(a, Line2D):
        for c in (a.get_color(), a.get_markerfacecolor(), a.get_markeredgecolor()):
            yield c
    elif isinstance(a, Patch):
        yield a.get_facecolor()
        yield a.get_edgecolor()
    elif isinstance(a, Text):
        if a.get_text().strip():
            yield a.get_color()
    elif isinstance(a, Collection) and a.get_array() is None:
        yield from a.get_facecolors()
        yield from a.get_edgecolors()


def violations(fig):
    """List of (artist, colour or colormap) pairs that break the rule."""
    bad = []
    for a in fig.findobj():
        if not a.get_visible():
            continue
        if getattr(a, "levels", None) is not None and getattr(a, "colors", None) is not None:
            cols = [a.colors] if isinstance(a.colors, str) else a.colors     # contours drawn in fixed colours
            bad += [(a, c) for c in cols if not allowed(c)]
            continue
        if isinstance(a, (AxesImage, Collection)) and a.get_array() is not None and np.ndim(a.get_array()) <= 2:
            name = a.get_cmap().name
            if name not in CMAPS and not cmap_ok(name):
                bad.append((a, f"cmap {name}"))
            continue
        for c in _colours(a):
            if isinstance(c, str) and c.lower() == "none":
                continue
            if np.ndim(c) == 1 and len(c) == 4 and c[3] == 0:
                continue
            if not allowed(c):
                bad.append((a, c))
    return bad
