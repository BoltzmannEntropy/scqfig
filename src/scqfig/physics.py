"""Small exact models behind the plots, so that every curve, level and wavefunction a figure shows is
computed, never sketched: the transmon in the charge basis with its charge-parity branches and
large-E_J/E_C asymptotes, SQUID tuning, junction inductance from normal-state resistance,
Jaynes-Cummings, dispersive formulas and the exact dressed transmon-resonator spectrum, the Purcell
limit, notch-resonator scattering, TLS-limited quality factors, normal modes and black-box
quantization of lumped circuits, Fock-space states with their Wigner functions, Bloch-vector
dynamics, thermal occupation and the mass-frequency conversion of dark-matter searches.

Units: energies and frequencies in GHz (E/h), times in the reciprocal unit (ns), except purcell_t1,
which returns microseconds. Flux is in units of the flux quantum. Circuit elements (capacitance,
inductance, resistance, susceptance) are in SI units, superconducting gaps in eV, temperatures in K
and boson rest energies in µeV."""
from math import factorial

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import eval_genlaguerre

_H, _E, _KB = 6.62607015e-34, 1.602176634e-19, 1.380649e-23
_HBAR, _PHI0 = _H / (2 * np.pi), _H / (2 * _E)
_RQ = _H / (4 * _E ** 2)                                 # superconducting resistance quantum h/4e^2


# ---- transmon -----------------------------------------------------------------------------

def transmon_levels(EJ, EC, ng=0.0, ncut=30, nlev=5):
    """Lowest nlev eigenenergies and charge-basis eigenvectors of H = 4 EC (n - ng)^2 - EJ cos(phi)."""
    n = np.arange(-ncut, ncut + 1)
    H = np.diag(4.0 * EC * (n - ng) ** 2) - 0.5 * EJ * (np.eye(n.size, k=1) + np.eye(n.size, k=-1))
    E, V = np.linalg.eigh(H)
    return E[:nlev], V[:, :nlev]


def transmon_spectrum(EJ, EC, ng=0.0, ncut=30, nlev=3):
    """Level energies above the ground state: [0, f01, f02, ...]."""
    E = transmon_levels(EJ, EC, ng, ncut, nlev)[0]
    return E - E[0]


def phase_densities(V, phi):
    """|psi_m(phi)|^2 of charge-basis eigenvectors V (columns), shape (nlev, len(phi)), each normalised
    to 1 over a period."""
    ncut = (V.shape[0] - 1) // 2
    n = np.arange(-ncut, ncut + 1)
    psi = np.exp(1j * np.outer(phi, n)) @ V / np.sqrt(2 * np.pi)
    return (np.abs(psi) ** 2).T


def charge_dispersion(EJ, EC, ng, nlev=3, ncut=20):
    """Level energies against offset charge, shape (len(ng), nlev)."""
    return np.array([transmon_levels(EJ, EC, g, ncut, nlev)[0] for g in np.atleast_1d(ng)])


def parity_branches(EJ, EC, ng, nlev=3, ncut=20):
    """Levels of the even and odd charge-parity states against offset charge. A quasiparticle that
    tunnels across the junction shifts the offset charge by half a Cooper pair, so the odd branch is
    the even one at n_g + 1/2. Returns (even, odd), each of shape (len(ng), nlev)."""
    ng = np.atleast_1d(np.asarray(ng, float))
    return charge_dispersion(EJ, EC, ng, nlev, ncut), charge_dispersion(EJ, EC, ng + 0.5, nlev, ncut)


def level_dispersion(EJ, EC, m=0, ncut=30):
    """Exact charge dispersion of level m, eps_m = E_m(n_g = 1/2) - E_m(n_g = 0) (GHz): positive for
    even m and negative for odd m. The 0-1 transition swings peak to peak by |eps_1 - eps_0|, which a
    charge-parity measurement calls 2 chi_0."""
    def one(ej):
        return transmon_levels(ej, EC, 0.5, ncut, m + 1)[0][m] - transmon_levels(ej, EC, 0.0, ncut, m + 1)[0][m]
    EJ = np.asarray(EJ, float)
    out = np.array([one(ej) for ej in EJ.ravel()]).reshape(EJ.shape)
    return out if out.ndim else float(out)


def koch_dispersion(EJ, EC, m=0):
    """Asymptotic charge dispersion of level m for E_J >> E_C (Koch et al. 2007, Eq. 2.5):
    eps_m = (-1)^m EC 2^(4m+5)/m! sqrt(2/pi) (EJ/2EC)^(m/2+3/4) exp(-sqrt(8 EJ/EC)), in the unit of EC."""
    r = np.asarray(EJ, float) / EC
    return ((-1) ** m * EC * 2.0 ** (4 * m + 5) / factorial(m) * np.sqrt(2 / np.pi) * (r / 2) ** (m / 2 + 0.75)
            * np.exp(-np.sqrt(8 * r)))


def transmon_asymptotic(EJ, EC):
    """Leading large-E_J/E_C forms of the transmon (Koch et al. 2007): f01 = sqrt(8 EJ EC) - EC and
    alpha = -EC. Returns (f01, alpha) in the unit of EC."""
    EJ = np.asarray(EJ, float)
    return np.sqrt(8 * EJ * EC) - EC, 0.0 * EJ - EC


def ej_for_f01(f01, EC, ncut=30):
    """Josephson energy (GHz) that gives a transmon of charging energy EC the 0-1 frequency f01, by
    root finding on the exact charge-basis spectrum at n_g = 0. f01 rises from 4 EC at E_J = 0, so a
    target at or below 4 EC has no solution and raises ValueError."""
    f01 = float(f01)
    if f01 <= 4 * EC:
        raise ValueError(f"f01 = {f01:g} is not above 4 EC = {4 * EC:g}, the lowest f01 any E_J gives")
    gap = lambda ej: transmon_spectrum(ej, EC, ncut=ncut, nlev=2)[1] - f01
    hi = max(f01 ** 2 / (8 * EC), EC)
    while gap(hi) < 0:
        hi *= 2
    return brentq(gap, 0.0, hi, xtol=1e-12, rtol=1e-13)


def squid_ej(flux, EJmax, d=0.0):
    """Josephson energy of a SQUID with junction asymmetry d = (EJ1 - EJ2)/(EJ1 + EJ2)."""
    x = np.pi * np.asarray(flux, float)
    return EJmax * np.sqrt(np.cos(x) ** 2 + d ** 2 * np.sin(x) ** 2)


def tunable_frequencies(flux, EJmax, EC, d=0.0, nlev=3, ncut=20):
    """Transitions f01, f02, ... of a SQUID transmon against flux, exact; shape (len(flux), nlev-1)."""
    return np.array([transmon_spectrum(EJ, EC, ncut=ncut, nlev=nlev)[1:]
                     for EJ in np.atleast_1d(squid_ej(flux, EJmax, d))])


def flux_at_frequency(f01, EJmax, EC, d=0.0, ncut=20):
    """Flux (in Phi_0, between 0 and 1/2) at which a SQUID transmon's f01 equals the target, found on
    the exact spectrum. The tuning curve is even in flux, so -flux works as well. A target above the
    sweet-spot f01 or below the f01 at half a flux quantum raises ValueError."""
    f = lambda x: tunable_frequencies(x, EJmax, EC, d, nlev=2, ncut=ncut)[0, 0] - f01
    top, bottom = f(0.0), f(0.5)
    if top < 0 or bottom > 0:
        raise ValueError(f"f01 = {f01:g} lies outside the tuning range {f01 + bottom:.4g} to {f01 + top:.4g}")
    if bottom == 0:
        return 0.5
    return brentq(f, 0.0, 0.5, xtol=1e-13)


# ---- junctions ---------------------------------------------------------------------------------

def lj_from_rn(Rn, gap):
    """Josephson inductance (H) of a junction of normal-state resistance Rn (ohm), from the
    Ambegaokar-Baratoff relation at zero temperature, Ic Rn = pi Delta / 2e, with the superconducting
    gap Delta in eV: L_J = Phi0 / (2 pi Ic) = hbar Rn / (pi e Delta)."""
    return _HBAR * np.asarray(Rn, float) / (np.pi * _E * gap)


def rn_from_lj(LJ, gap):
    """Normal-state resistance (ohm) of a junction of Josephson inductance LJ (H); the inverse of
    lj_from_rn for the gap Delta in eV."""
    return np.asarray(LJ, float) * np.pi * _E * gap / _HBAR


# ---- coupling to a resonator ------------------------------------------------------------------

def dressed_frequencies(fq, fr, g):
    """Upper and lower normal modes of a qubit (two-level) coupled to a resonator with strength g."""
    fq, fr = np.asarray(fq, float), np.asarray(fr, float)
    mid, half = (fq + fr) / 2, np.sqrt(g ** 2 + (fq - fr) ** 2 / 4)
    return mid + half, mid - half


def dispersive_shift(fq, fr, g, alpha):
    """Transmon dispersive shift chi = g^2 alpha / (Delta (Delta + alpha)), Delta = fq - fr; the
    resonator frequencies with the qubit in |0> and |1> differ by 2 chi (Koch et al. 2007)."""
    D = np.asarray(fq, float) - fr
    return g ** 2 * alpha / (D * (D + alpha))


def transmon_dressed(fq, fr, g, EC, nlev=6, nfock=8, ncut=30, rwa=False):
    """Exact dressed spectrum of a transmon of charging energy EC coupled to a resonator at fr, against
    the bare qubit frequency fq (all GHz). For each fq the transmon's E_J comes from ej_for_f01, its
    levels and charge matrix elements from the charge basis, and the coupling is
    g (n_ij / n_01) (a + a^dagger), so g is the 0-1 coupling. rwa=True keeps only the
    excitation-conserving terms between neighbouring levels, which turns the result into the exact
    solution of the rotating-wave model with the transmon's own n_12/n_01.

    Returns {"f": shape (len(fq), 2), the two dressed one-excitation frequencies in ascending order,
    "chi": the dispersive shift (E11 - E10 - E01 + E00)/2 in the convention of dispersive_shift, from
    eigenstates labelled by their largest bare component and so meaningful only away from resonance,
    "alpha": the bare anharmonicity, "lam": n_12/n_01}."""
    if nlev < 3 or nfock < 3:
        raise ValueError("transmon_dressed needs nlev >= 3 and nfock >= 3 to resolve chi")
    fq = np.atleast_1d(np.asarray(fq, float))
    a = np.diag(np.sqrt(np.arange(1, nfock)), 1)
    n = np.arange(-ncut, ncut + 1)
    i10, i01 = nfock, 1                                             # bare |1,0> and |0,1> in the product basis
    f, chi, alpha, lam = np.empty((fq.size, 2)), np.empty(fq.size), np.empty(fq.size), np.empty(fq.size)
    for k, f0 in enumerate(fq):
        E, V = transmon_levels(ej_for_f01(f0, EC, ncut), EC, ncut=ncut, nlev=nlev)
        E = E - E[0]
        c = V.T @ (n[:, None] * V)
        c = c / c[0, 1]
        if rwa:
            low = np.diag(np.diag(c, 1), 1)                          # |i><i+1|, lowers the transmon
            coupling = np.kron(low, a.T) + np.kron(low.T, a)
        else:
            coupling = np.kron(c, a + a.T)
        H = np.kron(np.diag(E), np.eye(nfock)) + np.kron(np.eye(nlev), fr * np.diag(np.arange(nfock))) + g * coupling
        w, U = np.linalg.eigh(H)
        level = lambda i, j: w[np.argmax(np.abs(U[i * nfock + j]))]
        one = np.sort(np.argsort(U[i10] ** 2 + U[i01] ** 2)[-2:])  # the two states richest in one excitation
        E00 = level(0, 0)
        f[k] = np.sort(w[one]) - E00
        chi[k] = (level(1, 1) - level(1, 0) - level(0, 1) + E00) / 2
        alpha[k] = E[2] - 2 * E[1]
        lam[k] = abs(c[1, 2])
    return {"f": f, "chi": chi, "alpha": alpha, "lam": lam}


def purcell_t1(fq, fr, g, kappa, filter_q=None):
    """Purcell-limited T1 in microseconds for resonator linewidth kappa (GHz). With a bandpass Purcell
    filter of quality factor filter_q the rate drops by (fq/fr)(fr / (2 filter_q Delta)) (Sete et al.
    2015)."""
    fq = np.asarray(fq, float)
    D = fq - fr
    gamma = 2 * np.pi * kappa * (g / D) ** 2
    if filter_q is not None:
        gamma = gamma * (fq / fr) * (fr / (2 * filter_q * np.abs(D)))
    return 1e-3 / gamma


# ---- resonators -------------------------------------------------------------------------------

def notch_s21(f, fr, Ql, Qc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """Transmission past a notch-coupled resonator with environment terms (Probst et al. 2015):
    a e^{i alpha} e^{-2 pi i f tau} [1 - (Ql/Qc) e^{i phi} / (1 + 2 i Ql (f/fr - 1))]."""
    f = np.asarray(f, float)
    env = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau)
    return env * (1 - (Ql / Qc) * np.exp(1j * phi) / (1 + 2j * Ql * (f / fr - 1)))


def tls_qi(nbar, F_delta, nc, beta=0.5, Q_other=np.inf):
    """Internal quality factor limited by saturable two-level systems:
    1/Qi = F delta / (1 + nbar/nc)^beta + 1/Q_other."""
    return 1.0 / (F_delta / (1 + np.asarray(nbar, float) / nc) ** beta + 1.0 / Q_other)


# ---- thermal occupation and dark-matter conversions -------------------------------------------

def thermal_population(f, T):
    """Bose-Einstein occupation 1 / (exp(h f / k T) - 1) of a mode at frequency f (GHz) and
    temperature T (K); zero at T = 0."""
    f, T = np.asarray(f, float), np.asarray(T, float)
    with np.errstate(divide="ignore", over="ignore"):
        x = _H * f * 1e9 / (_KB * T)
        return 1.0 / np.expm1(x)


def thermal_frequency(nbar, T):
    """Frequency (GHz) at which a mode at temperature T (K) holds nbar thermal photons; the inverse of
    thermal_population."""
    return _KB * np.asarray(T, float) * np.log1p(1.0 / np.asarray(nbar, float)) / _H / 1e9


def mass_from_frequency(f):
    """Rest energy m c^2 (µeV) of a boson whose Compton frequency is f (GHz): m c^2 = h f."""
    return np.asarray(f, float) * _H * 1e9 / _E * 1e6


def frequency_from_mass(m):
    """Compton frequency (GHz) of a boson of rest energy m (µeV); the inverse of mass_from_frequency."""
    return np.asarray(m, float) * 1e-6 * _E / _H / 1e9


# ---- circuits: normal modes and black-box quantization ----------------------------------------

def ej_from_lj(LJ):
    """Josephson energy E_J/h in GHz of a junction of inductance LJ (H): (Phi0/2 pi)^2 / LJ."""
    return (_PHI0 / (2 * np.pi)) ** 2 / (np.asarray(LJ, float) * _H) / 1e9


def normal_modes(C, Linv):
    """Normal modes of a lossless lumped circuit from its node capacitance matrix C (F, Maxwell form)
    and inverse-inductance matrix Linv (1/H), from Linv v = w^2 C v. Returns the frequencies in GHz,
    ascending, and the mode vectors as the columns of V, normalised so that V^T C V = 1."""
    w2, V = eigh(np.asarray(Linv, float), np.asarray(C, float))
    return np.sqrt(np.clip(w2, 0.0, None)) / (2e9 * np.pi), V


def port_susceptance(f, C, Linv, port=0):
    """Susceptance B = Im Y (S) of a lossless lumped circuit seen between node `port` and ground,
    against frequency f (GHz): Y = 1 / Z_pp with Z = (i w C + Linv / (i w))^-1. The circuit's modes are
    the zeros of B, which rises through each, and its poles are the series resonances seen from the
    port."""
    w = 2e9 * np.pi * np.atleast_1d(np.asarray(f, float))[:, None, None]
    K = w * np.asarray(C, float) - np.asarray(Linv, float) / w          # Y = i K
    e = np.zeros((len(w), K.shape[1], 1))
    e[:, port] = 1.0
    B = 1.0 / np.linalg.solve(K, e)[:, port, 0]
    return B if np.ndim(f) else B[0]


def black_box(B, EJ, f):
    """Black-box quantization (Nigg et al. 2012) of a junction shunted by a lossless linear circuit,
    to first order in the quartic term of the junction's cosine.

    B(f) is the susceptance (S) that the junction sees at frequency f (GHz), with the junction's own
    linear inductance included: port_susceptance gives it for a lumped circuit, and an interpolant of
    simulated data works too. The modes are the zeros of B on the search grid f where B rises. Each
    mode has an effective impedance Z = 2 / (f dB/df) (ohm) and a zero-point phase across the junction
    phi = sqrt(pi Z / R_Q), with R_Q = h / 4e^2. With EJ (GHz) these give the Kerr matrix K (GHz): the
    anharmonicity alpha_m = -EJ phi_m^4 / 2 on the diagonal and the cross-Kerr shift
    chi_mn = -EJ phi_m^2 phi_n^2 off it, the full shift of mode m per photon in mode n (2 chi in the
    convention of dispersive_shift). First order evaluates everything at the linear modes, so it
    overstates chi for a strongly anharmonic transmon; transmon_resonator gives the exact values.
    Returns {"f": GHz, "Z": ohm, "phi", "K"}, the modes in ascending frequency."""
    f = np.asarray(f, float)
    g = lambda x: float(B(x))
    b = np.asarray(B(f), float)
    rise = np.nonzero((b[:-1] < 0) & (b[1:] >= 0))[0]
    if not rise.size:
        raise ValueError("B has no rising zero on the search grid, so the circuit has no mode there")
    fm = np.array([brentq(g, f[k], f[k + 1], xtol=1e-13) for k in rise])
    h = 1e-7 * fm
    slope = np.array([(g(x + d) - g(x - d)) / (2 * d) for x, d in zip(fm, h)])   # S per GHz
    Z = 2.0 / (fm * slope)
    phi = np.sqrt(np.pi * Z / _RQ)
    K = -EJ * np.outer(phi ** 2, phi ** 2)
    K[np.diag_indices_from(K)] /= 2
    return {"f": fm, "Z": Z, "phi": phi, "K": K}


def transmon_resonator(Cq, Cg, Cr, LJ, Lr, ncut=25, nlev=12, nfock=12):
    """Exact spectrum of a transmon (Cq and a junction of inductance LJ to ground) coupled through Cg to
    an LC resonator (Cr, Lr), all in SI units. The transmon is solved in the charge basis (nlev levels
    kept) and the resonator in its Fock basis (nfock levels); they couple through the inverse
    capacitance matrix and are diagonalised together, and each eigenstate is labelled by its largest
    bare component. Returns {"f": [fq, fr], "K": [[alpha, chi], [chi, K_r]], "g"}, all in GHz: the
    dressed 0-1 frequencies, the transmon anharmonicity, the cross-Kerr shift
    chi = E11 - E10 - E01 + E00 (2 chi in the convention of dispersive_shift), the resonator self-Kerr
    and the charge coupling g = |<0|n|1>| times the coupling energy."""
    Ci = np.linalg.inv([[Cq + Cg, -Cg], [-Cg, Cr + Cg]])
    E, V = transmon_levels(ej_from_lj(LJ), _E ** 2 * Ci[0, 0] / 2 / _H / 1e9, ncut=ncut, nlev=nlev)
    n = np.arange(-ncut, ncut + 1)
    nop = V.T @ (n[:, None] * V)                                        # <i|n|j> between transmon levels
    fr = np.sqrt(Ci[1, 1] / Lr) / (2e9 * np.pi)                          # resonator loaded by 1/Ci[1, 1]
    gq = Ci[0, 1] * 2 * _E * np.sqrt(_HBAR / (2 * np.sqrt(Lr * Ci[1, 1]))) / _H / 1e9
    a = np.diag(np.sqrt(np.arange(1, nfock)), 1)
    H = (np.kron(np.diag(E - E[0]), np.eye(nfock)) + np.kron(np.eye(nlev), fr * np.diag(np.arange(nfock)))
         + gq * np.kron(nop, a + a.T))
    w, U = np.linalg.eigh(H)
    level = lambda i, j: w[np.argmax(np.abs(U[i * nfock + j]))]
    E00, E10, E01 = level(0, 0), level(1, 0), level(0, 1)
    chi = level(1, 1) - E10 - E01 + E00
    K = np.array([[level(2, 0) - 2 * E10 + E00, chi], [chi, level(0, 2) - 2 * E01 + E00]])
    return {"f": np.array([E10 - E00, E01 - E00]), "K": K, "g": abs(gq * nop[0, 1])}


# ---- Fock space, Wigner functions, Bloch vectors ---------------------------------------------

def fock(n, N):
    v = np.zeros(N, complex)
    v[n] = 1.0
    return v


def coherent(alpha, N):
    """Coherent state |alpha> truncated to N levels (renormalised)."""
    c = np.zeros(N, complex)
    c[0] = 1.0
    for k in range(1, N):
        c[k] = c[k - 1] * alpha / np.sqrt(k)
    return c / np.linalg.norm(c)


def cat(alpha, N, parity=+1):
    """Even (parity = +1) or odd (-1) cat state |alpha> + parity |-alpha>, normalised."""
    v = coherent(alpha, N) + parity * coherent(-alpha, N)
    return v / np.linalg.norm(v)


def dm(psi):
    psi = np.asarray(psi, complex)
    return np.outer(psi, psi.conj())


def wigner(rho, xvec, yvec=None):
    """Wigner function W(alpha) on the grid alpha = x + i y, from a state vector or density matrix in
    the Fock basis; normalised so that the integral of W over d^2 alpha is 1 (vacuum peak 2/pi)."""
    rho = np.asarray(rho, complex)
    if rho.ndim == 1:
        rho = dm(rho)
    yvec = xvec if yvec is None else yvec
    X, Y = np.meshgrid(xvec, yvec)
    A = X + 1j * Y
    B = 4 * np.abs(A) ** 2
    W = np.zeros_like(X)
    N = rho.shape[0]
    for m in range(N):
        if abs(rho[m, m]) > 0:
            W += np.real(rho[m, m] * (-1) ** m * eval_genlaguerre(m, 0, B))
        for n in range(m + 1, N):
            if abs(rho[m, n]) > 0:
                W += 2 * np.real(rho[m, n] * (-1) ** m * (2 * A) ** (n - m) * np.sqrt(factorial(m) / factorial(n))
                                 * eval_genlaguerre(m, n - m, B))
    return W * np.exp(-B / 2) * 2 / np.pi


def bloch_vector(rho):
    """(x, y, z) of a qubit density matrix, with |0> at +z."""
    rho = np.asarray(rho, complex)
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


def rotate(v, axis, angle):
    """Rotate vector v about a unit axis by angle(s) (Rodrigues); returns shape (len(angle), 3)."""
    v, k = np.asarray(v, float), np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.atleast_1d(angle)[:, None]
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * np.dot(k, v) * (1 - np.cos(a))


def rabi_trajectory(rabi, detuning, t, v0=(0.0, 0.0, 1.0)):
    """Bloch vector under a drive of Rabi rate `rabi` detuned by `detuning` (both in cycles per time
    unit), starting from v0: precession about (rabi, 0, detuning) at the generalised Rabi rate."""
    W = np.hypot(rabi, detuning)
    return rotate(v0, (rabi, 0.0, detuning), 2 * np.pi * W * np.asarray(t, float))


def rabi_population(rabi, detuning, t):
    """Excited population after driving |0> for time t at Rabi rate `rabi` and detuning (cycles per
    time unit): Omega^2/W^2 sin^2(pi W t) with W = sqrt(Omega^2 + Delta^2) (the Rabi chevron)."""
    W2 = np.asarray(rabi, float) ** 2 + np.asarray(detuning, float) ** 2
    return rabi ** 2 / W2 * np.sin(np.pi * np.sqrt(W2) * np.asarray(t, float)) ** 2


def rb_decay(m, p, A=0.5, B=0.5):
    """Randomized-benchmarking survival A p^m + B."""
    return A * np.asarray(p, float) ** np.asarray(m, float) + B


def one_over_f(f, A, alpha=1.0):
    """Noise spectral density A / f^alpha (A is the density at 1 in the unit of f)."""
    return A / np.asarray(f, float) ** alpha
