"""Building blocks shared by the plot families: colour bars, heatmaps, annotated matrices."""
import numpy as np

from ..core import MINUS, theme


def colorbar(ax, mappable, label=None, **kw):
    """A slim colour bar beside ax (an Axes, or a list of Axes that share it), framed and ticked in
    the active style."""
    t = theme()
    axes = list(np.ravel(ax)) if isinstance(ax, (list, tuple, np.ndarray)) else ax
    fig = (axes[0] if isinstance(axes, list) else axes).figure
    cb = fig.colorbar(mappable, ax=axes, pad=0.02, fraction=0.06, aspect=22, **kw)
    cb.outline.set_linewidth(t.axes_lw)
    cb.ax.tick_params(direction=t.ticks, length=2.2, width=t.axes_lw * 0.8, labelsize=t.size - 0.5)
    if label:
        cb.set_label(label, fontsize=t.size)
    return cb


def heatmap(ax, x, y, Z, *, label=None, cmap=None, diverging=False, vmin=None, vmax=None, cbar=True):
    """Z (shape len(y) x len(x)) on the x-y grid in the style's sequential (or diverging) map; the
    mesh is rasterized so a vector PDF stays small. Returns (mesh, colorbar or None)."""
    t = theme()
    Z = np.asarray(Z, float)
    if diverging and vmin is None and vmax is None:
        m = np.nanmax(np.abs(Z))
        vmin, vmax = -m, m
    mesh = ax.pcolormesh(x, y, Z, cmap=cmap or (t.cmap_div if diverging else t.cmap), vmin=vmin, vmax=vmax,
                         shading="auto", rasterized=True)
    return mesh, (colorbar(ax, mesh, label) if cbar else None)


def cell_text(fmt, v):
    """A cell value formatted by fmt (format string or callable), with a typographic minus and no
    negative zero ("-0.00" prints as "0.00")."""
    s = fmt(v) if callable(fmt) else fmt.format(v)
    if s.startswith("-") and not any(ch in "123456789" for ch in s.lower().split("e")[0]):
        s = s[1:]
    return s.replace("-", MINUS)


def annotated_matrix(ax, M, row_labels, col_labels=None, *, fmt="{:.2f}", text=None, cmap=None, vmin=None,
                     vmax=None, norm=None, diverging=False, cbar=False, label=None, fontsize=None, skip=None):
    """A matrix as coloured cells with each value printed in its cell (capacitance, assignment,
    correlation matrices). Text turns white on dark cells. text (same shape as M) replaces the printed
    strings, for cells coloured by one quantity and labelled with another; an empty string or
    skip(i, j, v) -> True leaves a cell blank."""
    t = theme()
    M = np.asarray(M, float)
    if diverging and vmin is None and vmax is None and norm is None:
        m = np.nanmax(np.abs(M))
        vmin, vmax = -m, m
    cm = cmap or (t.cmap_div if diverging else t.cmap)
    im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, norm=norm, aspect="equal", interpolation="nearest")
    nrm = im.norm
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v) or (skip and skip(i, j, v)):
                continue
            s_txt = cell_text(fmt, v) if text is None else str(text[i][j])
            if not s_txt:
                continue
            s = float(nrm(v))
            dark = abs(2 * s - 1) > 0.55 if diverging else s > 0.55
            ax.text(j, i, s_txt, ha="center", va="center", fontsize=fontsize or t.size - 1,
                    color="white" if dark else t.ink)
    col_labels = row_labels if col_labels is None else col_labels
    ax.set_xticks(range(M.shape[1]), col_labels)
    ax.set_yticks(range(M.shape[0]), row_labels)
    ax.tick_params(length=0, top=False, right=False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(t.axes_lw)
    return im, (colorbar(ax, im, label) if cbar else None)
