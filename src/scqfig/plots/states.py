"""Quantum states: Bloch sphere, Wigner functions and density matrices. Also holds the free-decay
Bloch model behind the spiral on the Bloch sphere."""
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .. import physics as ph
from ..core import series, theme
from ._base import annotated_matrix, heatmap

__all__ = ["bloch_sphere", "wigner", "density_matrix", "density_bars3d"]


# ---- models -----------------------------------------------------------------------------------

def bloch_decay(v0, detuning, t, T1, T2):
    """Free evolution of a Bloch vector in the frame of the drive: precession about +z at `detuning`
    (cycles per time unit) while the transverse part decays as exp(-t/T2) and z relaxes to +1 (|0>)
    as exp(-t/T1). Physical decay needs T2 <= 2 T1. Returns shape (len(t), 3)."""
    t = np.asarray(t, float)
    x0, y0, z0 = np.asarray(v0, float)
    xy = ph.rotate((x0, y0, 0.0), (0.0, 0.0, 1.0), 2 * np.pi * detuning * t) * np.exp(-t / T2)[:, None]
    xy[:, 2] = 1 + (z0 - 1) * np.exp(-t / T1)
    return xy


def basis_labels(d):
    """Computational-basis kets for a d-dimensional space: |00>, |01>, ... when d is a power of two,
    else |0>, |1>, ..."""
    n = int(round(np.log2(d)))
    if d > 2 and 2 ** n == d:
        return [rf"$|{k:0{n}b}\rangle$" for k in range(d)]
    return [rf"$|{k}\rangle$" for k in range(d)]


def _bloch(s):
    """Bloch vector of a 3-vector, a qubit state vector or a 2 x 2 density matrix."""
    a = np.asarray(s)
    if a.shape == (3,):
        return a.real.astype(float)
    if a.shape == (2,):
        return ph.bloch_vector(ph.dm(a))
    if a.shape == (2, 2):
        return ph.bloch_vector(a)
    raise ValueError(f"cannot read a Bloch vector from shape {a.shape}")


# ---- plots ------------------------------------------------------------------------------------

def bloch_sphere(ax, vectors=(), *, trajectories=(), labels=True, elev=18, azim=-58, zoom=1.3, lw=None):
    """Bloch sphere on a 3d axes (subplot_kw={"projection": "3d"}). A light wireframe sphere, dashed
    on its far side, with the equator, two meridians and the x, y, z axes; |0> at the top, |1> at the
    bottom and |+>, |-> on the x axis. vectors (Bloch vectors, qubit state vectors or density
    matrices) are drawn as arrows and trajectories (arrays of shape (n, 3)) as lines, the i-th of
    each in the colour and line style of series(i). Returns {"arrows": [...], "lines": [...]}."""
    t = theme()
    lw = lw or t.lw
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    e, a = np.radians(elev), np.radians(azim)
    eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u = np.linspace(0, 2 * np.pi, 361)

    def ring(p, q, c=(0.0, 0.0, 0.0)):
        return np.outer(np.cos(u), p) + np.outer(np.sin(u), q) + np.asarray(c)

    faint, main = [], []
    for lat in np.radians([-50, 50]):
        r = np.cos(lat)
        faint.append(ring((r, 0, 0), (0, r, 0), (0, 0, np.sin(lat))))
    for lon in np.radians([45, 135]):
        faint.append(ring((np.cos(lon), np.sin(lon), 0), (0, 0, 1)))
    main += [ring((1, 0, 0), (0, 1, 0)), ring((1, 0, 0), (0, 0, 1)), ring((0, 1, 0), (0, 0, 1))]
    for pts, col, w in [(p, t.light, 0.45) for p in faint] + [(p, t.reference, 0.5) for p in main]:
        front = (pts @ eye >= 0)[:, None]
        ax.plot(*np.where(front, pts, np.nan).T, color=col, ls="-", lw=w)
        ax.plot(*np.where(front, np.nan, pts).T, color=col, ls=(0, (1.5, 2)), lw=w * 0.9)
    p = np.cross(eye, (0, 0, 1.0))
    p /= np.linalg.norm(p)
    ax.plot(*ring(p, np.cross(eye, p)).T, color=t.reference, ls="-", lw=0.6)           # silhouette
    for k in range(3):
        seg = np.zeros((2, 3))
        seg[:, k] = (-1, 1)
        ax.plot(*seg.T, color=t.reference, ls="-", lw=0.45)
    if labels:
        fs = t.size
        kw = dict(fontsize=fs, color=t.ink)
        ax.text(0, 0, 1.13, r"$|0\rangle$", ha="center", va="bottom", **kw)
        ax.text(0, 0, -1.13, r"$|1\rangle$", ha="center", va="top", **kw)
        ax.text(1.2, 0, 0, r"$|{+}\rangle$", ha="left" if eye[1] < 0 else "right", va="center", **kw)
        ax.text(-1.2, 0, 0, r"$|{-}\rangle$", ha="right" if eye[1] < 0 else "left", va="center", **kw)
        ax.text(0, 1.15, 0, r"$y$", ha="left", va="center", fontsize=fs - 0.5, color=t.reference)
    arrows, lines = [], []
    for i, tr in enumerate(trajectories):
        s = series(i)
        tr = np.asarray(tr, float)
        lines += ax.plot(*tr.T, color=s["color"], ls=s["ls"], lw=lw * 0.85)
    for i, v in enumerate(vectors):
        v = _bloch(v)
        arrows.append(ax.quiver(0, 0, 0, *v, color=series(i)["color"], lw=lw * 1.2, arrow_length_ratio=0.16))
    lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off()
    return {"arrows": arrows, "lines": lines}


def wigner(ax, state=None, *, W=None, xvec=None, yvec=None, vmax=None, cbar=True, contour=True,
           label=r"$W(\alpha)$"):
    """Wigner function W(alpha) as a heatmap in the diverging map, with the W = 0 contour, equal
    aspect and a colour bar. Give a state vector or density matrix in the Fock basis, computed with
    physics.wigner on xvec (by default a grid that covers the state), or a precomputed W with its
    xvec (and yvec). vmax fixes the symmetric colour scale; 2/pi, the largest |W| any state reaches,
    puts several panels on one scale. Returns (mesh, colorbar or None)."""
    t = theme()
    if W is None:
        if state is None:
            raise ValueError("give a state or W")
        rho = np.asarray(state, complex)
        rho = ph.dm(rho) if rho.ndim == 1 else rho
        if xvec is None:
            nbar = float(np.real(np.trace(rho @ np.diag(np.arange(rho.shape[0])))))
            lim = np.ceil(2 * (np.sqrt(nbar) + 2.2)) / 2
            xvec = np.linspace(-lim, lim, 241)
        W = ph.wigner(rho, xvec, yvec)
    elif xvec is None:
        raise ValueError("a precomputed W needs its xvec")
    yvec = xvec if yvec is None else yvec
    W = np.asarray(W, float)
    m = vmax if vmax is not None else np.nanmax(np.abs(W))
    mesh, cb = heatmap(ax, xvec, yvec, W, label=label, diverging=True, vmin=-m, vmax=m, cbar=cbar)
    if contour:
        import contourpy
        tol = 1e-3 * np.nanmax(np.abs(W))
        Wc = np.where(np.abs(W) < tol, tol, W)            # skip contours in the vanishing tails
        for seg in contourpy.contour_generator(np.asarray(xvec, float), np.asarray(yvec, float), Wc).lines(0.0):
            ax.plot(seg[:, 0], seg[:, 1], color=t.ink, ls="-", lw=0.45)
    ax.set_aspect("equal")
    ax.set_xlim(np.min(xvec), np.max(xvec))
    ax.set_ylim(np.min(yvec), np.max(yvec))
    ax.set_xlabel(r"$\mathrm{Re}\,\alpha$")
    ax.set_ylabel(r"$\mathrm{Im}\,\alpha$")
    return mesh, cb


def density_matrix(axes, rho, *, labels=None, fmt="{:.2f}", fontsize=None,
                   titles=(r"$\mathrm{Re}\,\rho$", r"$\mathrm{Im}\,\rho$")):
    """Real and imaginary parts of a density matrix on a pair of axes, as cells in the diverging map
    with each value printed, both on one colour scale of +-max|rho_ij| so the parts compare. labels
    default to computational-basis kets. Returns (image of Re, image of Im)."""
    rho = np.asarray(rho, complex)
    labels = basis_labels(rho.shape[0]) if labels is None else labels
    m = np.max(np.abs(np.r_[rho.real.ravel(), rho.imag.ravel()]))
    dec = len(fmt.format(0.0).split(".")[-1]) if "." in fmt.format(0.0) else 0
    ims = []
    for ax, part, title in zip(axes, (rho.real, rho.imag), titles):
        M = np.round(part, dec) + 0.0                    # no "-0.00" cells
        im, _ = annotated_matrix(ax, M, labels, fmt=fmt, diverging=True, vmin=-m, vmax=m,
                                 fontsize=fontsize)
        ax.set_title(title, loc="center", fontsize=theme().size + 0.5)
        ims.append(im)
    return tuple(ims)


def _outline(x0, y0, w, h, b, eye):
    """Visible edges of a wire box over the footprint (x0, y0, w) from z = 0 to h, standing on a solid
    bar of height b: the end square at h, the three near vertical edges, and the parts of the far
    vertical edge that the bar does not hide."""
    xs, ys = (x0, x0 + w), (y0, y0 + w)
    far = (xs[0] if eye[0] > 0 else xs[1], ys[0] if eye[1] > 0 else ys[1])
    sq = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
    segs = [np.array([(*sq[k], h), (*sq[(k + 1) % 4], h)]) for k in range(4)]
    segs += [np.array([(*c, 0.0), (*c, h)]) for c in sq if c != far]
    lo, hi = sorted((0.0, h))
    blo, bhi = sorted((0.0, b))
    for z0, z1 in ((lo, min(hi, blo)), (max(lo, bhi), hi)):
        if z1 > z0:
            segs.append(np.array([(*far, z0), (*far, z1)]))
    return segs


def density_bars3d(ax, rho, *, ideal=None, labels=None, part="real", width=0.62, elev=30, azim=-55,
                   zlim=None, title=None):
    """Re(rho) (or Im with part="imag") as 3d bars on a 3d axes, with the ideal matrix as wire box
    outlines. A bar takes its colour from the diverging map and its faces are tints of the same hue,
    so the shading obeys the colour rule. Bars and outlines are stacked from the back, so a bar hides
    what lies behind it. The plotted part is the title, which keeps clear of the layout better than a
    3d axis label. Returns {"bars": [Poly3DCollection], "ideal": [Line3DCollection]}."""
    t = theme()
    rho = np.asarray(rho, complex)
    d = rho.shape[0]
    take = (lambda M: np.asarray(M, complex).real) if part == "real" else (lambda M: np.asarray(M, complex).imag)
    Z = take(rho)
    Zi = None if ideal is None else take(ideal)
    labels = basis_labels(d) if labels is None else labels
    cmap = mpl.colormaps[t.cmap_div]
    ref = np.abs(Z).max() if Zi is None else max(np.abs(Z).max(), np.abs(Zi).max())
    norm = mpl.colors.Normalize(-ref, ref)
    e, a = np.radians(elev), np.radians(azim)
    eye = (np.cos(e) * np.cos(a), np.cos(e) * np.sin(a))
    cells = sorted(((i, j) for i in range(d) for j in range(d)), key=lambda c: c[1] * eye[0] + c[0] * eye[1])
    ax.computed_zorder = False
    bars, outlines = [], []
    for rank, (i, j) in enumerate(cells):                # back to front; column j along x, row i along y
        v = Z[i, j]
        c = np.array(cmap(norm(v)))
        h = np.array(cmap(1.0 if v >= 0 else 0.0))
        top = c + (1 - c) * 0.3
        side = h + (c - h) * 0.55
        # bar3d face order: -z, +z, -y, +y, -x, +x
        faces = [top, top, c, c, c, side] if eye[0] > 0 else [top, top, c, c, side, c]
        bars.append(ax.bar3d(j - width / 2, i - width / 2, 0.0, width, width, v, color=faces, shade=False,
                             edgecolor=t.reference, linewidth=0.25, zorder=2 * rank + 1))
        if Zi is not None and abs(Zi[i, j]) > 1e-9:
            segs = _outline(j - width / 2, i - width / 2, width, Zi[i, j], v, eye)
            outlines.append(ax.add_collection3d(Line3DCollection(segs, colors=t.ink, linewidths=0.6,
                                                                 linestyles="-", zorder=2 * rank + 2)))
    lo = min(0.0, Z.min(), 0.0 if Zi is None else Zi.min())
    hi = max(Z.max(), 0.0 if Zi is None else Zi.max())
    z0, z1 = zlim or (lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_zlim(z0, z1)
    ax.set_xlim(-0.6, d - 0.4)
    ax.set_ylim(-0.6, d - 0.4)
    ax.view_init(elev=elev, azim=azim)
    # basis labels as 3d text: fixed 3d tick labels hold stale positions before the first draw and
    # collapse constrained layout
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(range(d))
        axis.set_major_formatter(mpl.ticker.NullFormatter())
    fs = t.size - 0.5
    yfront = -0.6 if eye[1] < 0 else d - 0.4
    xside = d - 0.4 if eye[0] > 0 else -0.6
    for k, s in enumerate(labels):
        ax.text(k, yfront + np.sign(yfront - (d - 1) / 2) * 0.55, z0, s, ha="center", va="top", fontsize=fs)
        ax.text(xside + np.sign(xside - (d - 1) / 2) * 0.45, k, z0, s, ha="left", va="top", fontsize=fs)
    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.tick_params(labelsize=t.size - 1, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_linewidth(t.axes_lw)
    ax.set_title(title or (r"$\mathrm{Re}\,\rho$" if part == "real" else r"$\mathrm{Im}\,\rho$"), loc="center",
                 fontsize=t.size + 0.5)
    ax.set_box_aspect((1, 1, 0.72), zoom=1.0)       # larger zooms push the 3d labels off the saved page
    return {"bars": bars, "ideal": outlines}
