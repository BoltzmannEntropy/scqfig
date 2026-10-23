"""Least-squares fits that the time-domain, spectroscopy and readout plots draw and annotate. Each
returns a Fit: callable on x, indexable by parameter name, with one-sigma errors in .errors."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, least_squares

from . import physics as ph


@dataclass
class Fit:
    model: callable
    params: dict
    errors: dict
    derived: dict = field(default_factory=dict)
    cov: object = None           # parameter covariance, in the order of params, when the fit gives one

    def __call__(self, x):
        return self.model(np.asarray(x, float), *[self.params[k] for k in self.params])

    def __getitem__(self, k):
        return self.params[k] if k in self.params else self.derived[k]


def _curve(model, x, y, p0, names, sigma=None, bounds=(-np.inf, np.inf)):
    popt, pcov = curve_fit(model, np.asarray(x, float), np.asarray(y, float), p0=p0, sigma=sigma,
                           bounds=bounds, maxfev=40000)
    err = np.sqrt(np.clip(np.diag(pcov), 0, None))
    return Fit(model, dict(zip(names, popt)), dict(zip(names, err)), cov=pcov)


def exp_model(t, A, T, B):
    return A * np.exp(-t / T) + B


def exp_decay(t, y, sigma=None):
    """A e^{-t/T} + B (T1, echo, any single exponential)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    B, A = y[-1], y[0] - y[-1]
    T = t[np.argmin(np.abs(y - (B + A / np.e)))] or np.ptp(t) / 3
    return _curve(exp_model, t, y, [A, T, B], ["A", "T", "B"], sigma)


def _fft_freq(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    dt = np.mean(np.diff(t))
    spec = np.abs(np.fft.rfft(y - y.mean(), n=8 * t.size))
    fr = np.fft.rfftfreq(8 * t.size, dt)
    return fr[1:][np.argmax(spec[1:])]


def damped_cosine_model(t, A, f, T, phi, B):
    return A * np.exp(-t / T) * np.cos(2 * np.pi * f * t + phi) + B


def damped_cosine(t, y, sigma=None):
    """A e^{-t/T} cos(2 pi f t + phi) + B (Ramsey fringes, damped Rabi oscillations)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    f = _fft_freq(t, y)
    A, B = (y.max() - y.min()) / 2, y.mean()
    best = None
    for phi in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        try:
            r = _curve(damped_cosine_model, t, y, [A, f, np.ptp(t) / 2, phi, B], ["A", "f", "T", "phi", "B"], sigma)
        except RuntimeError:
            continue
        res = np.sum((r(t) - y) ** 2)
        if best is None or res < best[0]:
            best = (res, r)
    if best[1]["A"] < 0:                               # report a positive amplitude
        best[1].params["A"] *= -1
        best[1].params["phi"] += np.pi
    return best[1]


def cosine_model(x, A, f, phi, B):
    return A * np.cos(2 * np.pi * f * x + phi) + B


def cosine(x, y, sigma=None):
    """A cos(2 pi f x + phi) + B (Rabi oscillation against amplitude)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = _fft_freq(x, y)
    best = None
    for phi in (0.0, np.pi):
        r = _curve(cosine_model, x, y, [(y.max() - y.min()) / 2, f, phi, y.mean()], ["A", "f", "phi", "B"], sigma)
        res = np.sum((r(x) - y) ** 2)
        if best is None or res < best[0]:
            best = (res, r)
    return best[1]


def lorentzian_model(f, A, f0, fwhm, B):
    return A * (fwhm / 2) ** 2 / ((f - f0) ** 2 + (fwhm / 2) ** 2) + B


def lorentzian(f, y, sigma=None):
    """Peak or dip A (w/2)^2 / ((f - f0)^2 + (w/2)^2) + B."""
    f, y = np.asarray(f, float), np.asarray(y, float)
    B = np.median(y)
    k = np.argmax(np.abs(y - B))
    A = y[k] - B
    half = np.abs(y - B) > np.abs(A) / 2
    w = max(np.ptp(f[half]), 2 * np.mean(np.diff(f)))
    return _curve(lorentzian_model, f, y, [A, f[k], w, B], ["A", "f0", "fwhm", "B"], sigma)


def notch(f, s21):
    """Complex fit of notch-resonator transmission (physics.notch_s21) with cable delay, amplitude and
    phase offsets. Derived Qi = 1 / (1/Ql - cos(phi)/Qc)."""
    f, s = np.asarray(f, float), np.asarray(s21, complex)
    edge = np.r_[: max(3, f.size // 20), -max(3, f.size // 20):]
    a = np.mean(np.abs(s[edge]))
    ph_un = np.unwrap(np.angle(s))
    tau = -np.polyfit(f[edge], ph_un[edge], 1)[0] / (2 * np.pi)
    s0 = s * np.exp(2j * np.pi * f * tau)
    k = np.argmin(np.abs(s0))
    fr = f[k]
    depth = 1 - np.abs(s0[k]) / a
    below = np.abs(s0) ** 2 < (a ** 2 + np.abs(s0[k]) ** 2) / 2
    Ql = fr / max(np.ptp(f[below]), 2 * np.mean(np.diff(f)))
    Qc = Ql / max(depth, 1e-3)
    alpha = np.angle(np.mean(s0[edge]))

    def resid(p):
        m = ph.notch_s21(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        return np.r_[(m - s).real, (m - s).imag]

    p0 = [fr, Ql, Qc, 0.0, a, alpha, tau]
    sc = [fr / Ql, Ql, Qc, 1.0, a, 1.0, 1.0 / np.ptp(f)]
    r = least_squares(resid, p0, x_scale=sc, method="lm", max_nfev=20000)
    J = r.jac
    dof = max(1, 2 * f.size - len(p0))
    try:
        cov = np.linalg.inv(J.T @ J) * np.sum(r.fun ** 2) / dof
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan)
    names = ["fr", "Ql", "Qc", "phi", "a", "alpha", "tau"]
    p = dict(zip(names, r.x))
    Qi = 1 / (1 / p["Ql"] - np.cos(p["phi"]) / p["Qc"])
    return Fit(lambda x, *q: ph.notch_s21(x, *q), p, dict(zip(names, err)), {"Qi": Qi})


def rb(m, y, sigma=None, d=2):
    """Randomized-benchmarking decay A p^m + B; derived error per Clifford epc = (1 - p)(d - 1)/d."""
    m, y = np.asarray(m, float), np.asarray(y, float)
    r = _curve(lambda x, A, p, B: ph.rb_decay(x, p, A, B), m, y, [y[0] - y[-1], 0.99, y[-1]], ["A", "p", "B"],
               sigma, bounds=([-1, 0, -1], [2, 1, 2]))
    r.derived["epc"] = (1 - r["p"]) * (d - 1) / d
    r.derived["epc_err"] = r.errors["p"] * (d - 1) / d
    return r


def tls(nbar, qi, sigma=None):
    """Fit internal quality factor against mean photon number to physics.tls_qi,
    1/Qi = F delta / (1 + nbar/nc)^beta + 1/Q_other, by least squares on log Qi (relative errors);
    sigma are one-sigma errors of qi. Returns a Fit with F_delta, nc, beta and Q_other, and the derived
    low-power limit Qi_low = 1 / (F delta + 1/Q_other)."""
    n, q = np.asarray(nbar, float), np.asarray(qi, float)
    w = np.ones_like(q) if sigma is None else q / np.asarray(sigma, float)
    Qo = 1.2 * q.max()
    tls = np.clip(1 / q - 1 / Qo, 1e-30, None)                   # TLS part of the loss
    Fd = tls.max()
    half = n[np.argmax(tls < Fd / 2)] if np.any(tls < Fd / 2) else np.sqrt(n.min() * n.max())
    x0 = [np.log(Fd), np.log(half / 3), 0.5, np.log(Qo)]          # half the TLS loss at nbar = 3 nc for beta = 1/2

    def resid(x):
        return w * (np.log(q) - np.log(ph.tls_qi(n, np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]))))

    r = least_squares(resid, x0, bounds=([-np.inf, -np.inf, 0.05, -np.inf], [np.inf, np.inf, 2.0, np.inf]))
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun ** 2) / max(1, n.size - 4)
        e = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        e = np.full(4, np.nan)
    p = {"F_delta": np.exp(r.x[0]), "nc": np.exp(r.x[1]), "beta": r.x[2], "Q_other": np.exp(r.x[3])}
    err = {"F_delta": p["F_delta"] * e[0], "nc": p["nc"] * e[1], "beta": e[2], "Q_other": p["Q_other"] * e[3]}
    return Fit(ph.tls_qi, p, err, {"Qi_low": 1 / (p["F_delta"] + 1 / p["Q_other"])})


def gaussian_model(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian(x, y, sigma=None):
    """Least-squares Gaussian A exp(-(x - mu)^2 / (2 sigma^2)) through histogram counts y at bin
    centres x; sigma are the errors of y. Returns a Fit with A, mu and sigma (positive)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = np.argmax(y)
    s0 = np.sqrt(np.sum(y * (x - x[k]) ** 2) / np.sum(y))
    popt, pcov = curve_fit(gaussian_model, x, y, p0=[y[k], x[k], s0], sigma=sigma, maxfev=20000)
    popt[2] = abs(popt[2])
    names = ["A", "mu", "sigma"]
    return Fit(gaussian_model, dict(zip(names, popt)), dict(zip(names, np.sqrt(np.clip(np.diag(pcov), 0, None)))))


def power_law(f, S):
    """S = A / f^alpha by a straight line in log-log; A is the density at f = 1."""
    lf, lS = np.log(np.asarray(f, float)), np.log(np.asarray(S, float))
    (slope, icpt), cov = np.polyfit(lf, lS, 1, cov=True)
    A, alpha = np.exp(icpt), -slope
    return Fit(lambda x, A, alpha: ph.one_over_f(x, A, alpha), {"A": A, "alpha": alpha},
               {"A": A * np.sqrt(cov[1, 1]), "alpha": np.sqrt(cov[0, 0])})
