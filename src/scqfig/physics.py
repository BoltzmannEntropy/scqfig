"""Small exact models behind the plots, so that every curve, level and wavefunction a figure shows is
computed, never sketched: the transmon in the charge basis, SQUID tuning, Jaynes-Cummings and
dispersive formulas, the Purcell limit, notch-resonator scattering, TLS-limited quality factors,
Fock-space states with their Wigner functions, and Bloch-vector dynamics.

Units: energies and frequencies in GHz (E/h), times in the reciprocal unit (ns), except purcell_t1,
which returns microseconds. Flux is in units of the flux quantum."""
from math import factorial

import numpy as np
from scipy.special import eval_genlaguerre


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


def squid_ej(flux, EJmax, d=0.0):
    """Josephson energy of a SQUID with junction asymmetry d = (EJ1 - EJ2)/(EJ1 + EJ2)."""
    x = np.pi * np.asarray(flux, float)
    return EJmax * np.sqrt(np.cos(x) ** 2 + d ** 2 * np.sin(x) ** 2)


def tunable_frequencies(flux, EJmax, EC, d=0.0, nlev=3, ncut=20):
    """Transitions f01, f02, ... of a SQUID transmon against flux, exact; shape (len(flux), nlev-1)."""
    return np.array([transmon_spectrum(EJ, EC, ncut=ncut, nlev=nlev)[1:]
                     for EJ in np.atleast_1d(squid_ej(flux, EJmax, d))])


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
