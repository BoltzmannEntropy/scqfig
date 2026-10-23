import numpy as np
import pytest

from scqfig import fit, physics as ph

rng = np.random.default_rng(7)


def test_exp_decay_recovers_t1():
    t = np.linspace(0, 200, 60)
    y = 0.9 * np.exp(-t / 45.0) + 0.05 + rng.normal(0, 0.01, t.size)
    r = fit.exp_decay(t, y)
    assert r["T"] == pytest.approx(45.0, rel=0.05)
    assert r.errors["T"] > 0
    assert r(np.array([0.0]))[0] == pytest.approx(0.95, abs=0.03)


def test_damped_cosine_recovers_detuning_and_t2():
    t = np.linspace(0, 20, 150)
    y = 0.5 * np.exp(-t / 8.0) * np.cos(2 * np.pi * 0.6 * t) + 0.5 + rng.normal(0, 0.01, t.size)
    r = fit.damped_cosine(t, y)
    assert r["f"] == pytest.approx(0.6, rel=0.02)
    assert r["T"] == pytest.approx(8.0, rel=0.1)


def test_cosine_recovers_rabi_rate():
    a = np.linspace(0, 1, 80)
    y = 0.48 * (1 - np.cos(2 * np.pi * 1.3 * a)) + 0.02 + rng.normal(0, 0.01, a.size)
    r = fit.cosine(a, y)
    assert abs(r["f"]) == pytest.approx(1.3, rel=0.02)


def test_lorentzian_recovers_centre_and_width():
    f = np.linspace(4.9, 5.1, 200)
    y = 1.0 * (0.01 / 2) ** 2 / ((f - 5.02) ** 2 + (0.01 / 2) ** 2) + 0.1 + rng.normal(0, 0.01, f.size)
    r = fit.lorentzian(f, y)
    assert r["f0"] == pytest.approx(5.02, abs=5e-4)
    assert abs(r["fwhm"]) == pytest.approx(0.01, rel=0.05)


def test_notch_fit_recovers_quality_factors():
    fr, Ql, Qc = 7.1234, 8e3, 1.2e4
    f = np.linspace(fr - 6 * fr / Ql, fr + 6 * fr / Ql, 401)
    s = ph.notch_s21(f, fr, Ql, Qc, phi=0.15, a=0.8, alpha=0.4, tau=2.0)
    s = s + (rng.normal(0, 0.004, f.size) + 1j * rng.normal(0, 0.004, f.size))
    r = fit.notch(f, s)
    assert r["fr"] == pytest.approx(fr, abs=fr / Ql / 50)
    assert r["Ql"] == pytest.approx(Ql, rel=0.05)
    assert r["Qc"] == pytest.approx(Qc, rel=0.05)
    Qi = 1 / (1 / Ql - np.cos(0.15) / Qc)
    assert r["Qi"] == pytest.approx(Qi, rel=0.1)


def test_rb_fit_gives_error_per_clifford():
    m = np.array([1, 5, 10, 20, 50, 100, 200, 400])
    y = 0.48 * 0.995 ** m + 0.5 + rng.normal(0, 0.003, m.size)
    r = fit.rb(m, y)
    assert r["p"] == pytest.approx(0.995, abs=0.001)
    assert r["epc"] == pytest.approx((1 - 0.995) / 2, rel=0.2)


def test_power_law_recovers_exponent():
    f = np.logspace(-3, 1, 50)
    S = 2e-12 / f ** 0.9 * np.exp(rng.normal(0, 0.05, f.size))
    r = fit.power_law(f, S)
    assert r["alpha"] == pytest.approx(0.9, abs=0.05)
    assert r["A"] == pytest.approx(2e-12, rel=0.15)


def test_pm_formatting():
    from scqfig.core import pm
    assert pm(45.23, 0.81, "µs") == "45.2 ± 0.8 µs"
    assert pm(45.23, 0.14) == "45.23 ± 0.14"
    assert pm(1234.0, 56.0) == "1234 ± 56"
    assert pm(0.99512, None) == "0.995"
