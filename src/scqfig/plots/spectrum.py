"""Spectrum of a transmon: the potential with computed levels and wavefunctions, charge dispersion,
flux-tuned spectroscopy and the vacuum-Rabi avoided crossing."""
import numpy as np

from .. import physics as ph
from ..core import ref_kw, theme
from ..schematics.motifs import well
from ..schematics.symbols import dim
from ._base import heatmap

__all__ = ["potential_well", "charge_dispersion", "flux_spectroscopy", "avoided_crossing"]


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
