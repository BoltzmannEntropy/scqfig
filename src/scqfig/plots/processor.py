"""Processor-level plots: device maps with per-qubit and per-pair metrics, cumulative distributions."""
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from ..core import log_minor, ref_kw, sci, series, theme
from ._base import colorbar

__all__ = ["device_map", "cdf"]


def device_map(ax, pos, edges, node_values, edge_values, *, node_label=r"$T_1$ (µs)", edge_label="CZ error (%)",
               node_fmt="{:.0f}", names=None, node_cmap=None, edge_cmap=None, node_lim=None, edge_lim=None, r=0.3,
               edge_lw=5.0, cbar=True):
    """Qubit connectivity graph: qubit k is a circle at pos[k] filled by node_values[k] in the
    sequential map with the value printed inside, and the coupler between qubits i and j (edges) a bar
    coloured by edge_values. Both metrics get a colour bar. Returns a dict with the node and edge
    collections, the value texts and the two colour bars."""
    t = theme()
    pos = np.asarray(pos, float)
    nv, ev = np.asarray(node_values, float), np.asarray(edge_values, float)
    if len(nv) != len(pos):
        raise ValueError(f"{len(nv)} node values for {len(pos)} qubits")
    if len(ev) != len(edges):
        raise ValueError(f"{len(ev)} edge values for {len(edges)} edges")
    nn = Normalize(*(node_lim or (nv.min(), nv.max())))
    en = Normalize(*(edge_lim or (ev.min(), ev.max())))
    segs = [pos[[i, j]] for i, j in edges]
    ax.add_collection(LineCollection(segs, colors=t.ink, linewidths=edge_lw + 1.2, capstyle="butt", zorder=1))
    lc = LineCollection(segs, cmap=edge_cmap or t.cmap, norm=en, linewidths=edge_lw, capstyle="butt", zorder=2)
    lc.set_array(ev)
    ax.add_collection(lc)
    pc = PatchCollection([Circle(p, r) for p in pos], cmap=node_cmap or t.cmap, norm=nn, edgecolors=t.ink,
                         linewidths=0.7, zorder=3)
    pc.set_array(nv)
    ax.add_collection(pc)
    texts = []
    for p, v in zip(pos, nv):
        dark = nn(v) > 0.55
        texts.append(ax.text(*p, node_fmt.format(v), ha="center", va="center", fontsize=t.size - 1,
                             color="white" if dark else t.ink, zorder=4))
    for k, name in enumerate(names or []):
        ax.text(pos[k, 0] - 0.78 * r, pos[k, 1] + 0.78 * r, name, ha="right", va="bottom", fontsize=t.size - 1.5,
                color=t.reference, zorder=4)
    pad = r + 0.25
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad + (0.15 if names else 0))
    ax.set_aspect("equal")
    ax.axis("off")
    ecb = colorbar(ax, lc, edge_label, shrink=0.92) if cbar else None     # constrained layout puts the
    ncb = colorbar(ax, pc, node_label, shrink=0.92) if cbar else None     # last bar nearest the map
    return {"nodes": pc, "edges": lc, "values": texts, "node_cbar": ncb, "edge_cbar": ecb}


def cdf(ax, data, *, xlabel="error per operation", ylabel="cumulative fraction", median=True, loc="best"):
    """Empirical cumulative distributions of several error types ({label: values}) as steps on a log
    x axis. Each median is marked on the half-way line and its value printed beside the marker, on
    whichever side keeps clear of the other curves. Returns {label: median}."""
    t = theme()
    out, steps = {}, []
    for i, (name, vals) in enumerate(data.items()):
        x = np.sort(np.asarray(vals, float))
        if np.any(x <= 0):
            raise ValueError(f"{name}: values must be positive on a log axis")
        y = np.arange(1, x.size + 1) / x.size
        s = series(i)
        out[name] = float(np.median(x))
        steps += ax.step(np.r_[x[0], x], np.r_[0, y], where="post", color=s["color"], ls=s["ls"], lw=t.lw, label=name,
                         zorder=3)
    ax.set_xscale("log")
    log_minor(ax, "x")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if median:
        ax.axhline(0.5, **ref_kw())
        for i, m in enumerate(out.values()):
            s = series(i)
            ax.plot([m, m], [0, 0.5], color=s["color"], ls=(0, (1, 1.5)), lw=0.8, zorder=2)
            ax.plot([m], [0.5], marker=s["marker"], ms=4.5, mec=s["color"], mfc=s["color"] if t.data_filled else "white",
                    ls="none", zorder=5)
        _median_labels(ax, list(out.values()), steps)
    ax.legend(loc=loc, frameon=False, handlelength=2.2)
    return out


def _median_labels(ax, medians, steps):
    """Print each median beside its marker: above-left or below-right of it, the quadrants its own
    curve leaves empty, choosing the first that no other curve crosses."""
    t = theme()
    r = ax.figure.canvas.get_renderer()
    paths = [ln.get_transform().transform_path(ln.get_path()) for ln in steps]
    spots = ((1 / 1.07, 0.535, "right", "bottom"), (1.07, 0.465, "left", "top"))
    for i, m in enumerate(medians):
        for k, (fx, y, ha, va) in enumerate(spots):
            txt = ax.text(m * fx, y, sci(m), ha=ha, va=va, fontsize=t.size - 1.5, color=series(i)["color"], zorder=6)
            bb = txt.get_window_extent(r).expanded(1.08, 1.25)
            inside = ax.get_window_extent(r).contains(bb.x0, bb.y0) and ax.get_window_extent(r).contains(bb.x1, bb.y1)
            if inside and not any(p.intersects_bbox(bb, filled=False) for j, p in enumerate(paths) if j != i):
                break
            if k < len(spots) - 1:
                txt.remove()
