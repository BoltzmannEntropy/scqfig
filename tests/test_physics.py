import numpy as np
import pytest

from scqfig import physics as ph


def test_transmon_matches_koch_asymptotics_deep_in_transmon_regime():
    EJ, EC = 20.0, 0.25                                     # E_J/E_C = 80
    f = ph.transmon_spectrum(EJ, EC, nlev=3)
    f01, alpha = f[1], f[2] - 2 * f[1]
    s = np.sqrt(8 * EJ * EC)
    assert f[0] == 0.0
    # next-order Koch: f01 = s - EC - EC^2/s, alpha = -EC - 9 EC^2 / (4 s)
    assert f01 == pytest.approx(s - EC - EC ** 2 / s, rel=1e-3)
    assert alpha == pytest.approx(-EC - 9 * EC ** 2 / (4 * s), rel=2e-2)


def test_charge_dispersion_is_periodic_and_flattens_with_ratio():
    ng = np.linspace(-1, 1, 81)
    narrow = ph.charge_dispersion(1.0, 1.0, ng, nlev=2)
    flat = ph.charge_dispersion(50.0, 1.0, ng, nlev=2)
    assert narrow.shape == (81, 2)
    np.testing.assert_allclose(narrow[0], narrow[-1], atol=1e-9)      # period 1 in n_g
    spread = lambda E: np.ptp(E[:, 1] - E[:, 0])
    assert spread(flat) < 1e-3 * spread(narrow)


def test_phase_densities_are_normalised():
    E, V = ph.transmon_levels(20.0, 0.25, nlev=3)
    phi = np.linspace(-np.pi, np.pi, 2001)
    d = ph.phase_densities(V, phi)
    assert d.shape == (3, phi.size)
    np.testing.assert_allclose(np.trapz(d, phi, axis=1), 1.0, atol=1e-6)


def test_squid_ej_symmetric_and_asymmetric():
    assert ph.squid_ej(0.0, 30.0) == pytest.approx(30.0)
    assert ph.squid_ej(0.5, 30.0) == pytest.approx(0.0, abs=1e-12)
    assert ph.squid_ej(0.5, 30.0, d=0.2) == pytest.approx(6.0)


def test_tunable_frequencies_peak_at_sweet_spot():
    flux = np.linspace(-0.5, 0.5, 101)
    f = ph.tunable_frequencies(flux, 30.0, 0.25, d=0.3, nlev=3)
    assert f.shape == (101, 2)
    assert np.argmax(f[:, 0]) == 50
    np.testing.assert_allclose(f[:, 0], f[::-1, 0], rtol=1e-9)


def test_dressed_frequencies_split_by_2g_on_resonance():
    up, down = ph.dressed_frequencies(6.0, 6.0, 0.05)
    assert up - down == pytest.approx(0.1)
    up, down = ph.dressed_frequencies(5.0, 7.0, 0.05)
    assert down == pytest.approx(5.0 - 0.05 ** 2 / 2, rel=1e-4)


def test_dispersive_shift_sign_changes_in_straddling_regime():
    g, alpha = 0.1, -0.25
    fr = 7.0
    assert ph.dispersive_shift(fr - 1.5, fr, g, alpha) < 0       # qubit far below the resonator
    assert ph.dispersive_shift(fr + 0.1, fr, g, alpha) > 0       # 0 < Delta < |alpha|: f12 < fr < f01
    # large detuning limit: chi -> g^2 alpha / Delta^2
    D = -3.0
    assert ph.dispersive_shift(fr + D, fr, g, alpha) == pytest.approx(g ** 2 * alpha / (D * (D + alpha)))


def test_purcell_filter_extends_t1():
    bare = ph.purcell_t1(5.0, 7.0, 0.1, 0.005)
    # gamma = 2 pi kappa (g/Delta)^2 in 1/ns -> T1 in us
    assert bare == pytest.approx(1e-3 / (2 * np.pi * 0.005 * (0.1 / 2.0) ** 2))
    assert ph.purcell_t1(5.0, 7.0, 0.1, 0.005, filter_q=30) > 5 * bare


def test_notch_s21_dip_depth():
    fr, Ql, Qc = 7.0, 1e4, 2e4
    s = ph.notch_s21(np.array([fr]), fr, Ql, Qc)
    assert abs(s[0]) == pytest.approx(1 - Ql / Qc)
    far = ph.notch_s21(np.array([fr * 1.2]), fr, Ql, Qc)
    assert abs(far[0]) == pytest.approx(1.0, abs=1e-3)


def test_tls_qi_rises_with_power():
    n = np.logspace(-1, 6, 30)
    q = ph.tls_qi(n, F_delta=2e-6, nc=10.0, Q_other=2e6)
    assert np.all(np.diff(q) > 0)
    assert q[0] < 1 / 2e-6 * 1.1
    assert q[-1] < 2e6


def test_wigner_vacuum_fock_and_coherent():
    x = np.linspace(-3, 3, 241)
    W0 = ph.wigner(ph.fock(0, 10), x)
    assert W0.max() == pytest.approx(2 / np.pi, rel=1e-3)
    dA = (x[1] - x[0]) ** 2
    assert W0.sum() * dA == pytest.approx(1.0, abs=1e-3)
    W1 = ph.wigner(ph.fock(1, 10), x)
    assert W1[120, 120] == pytest.approx(-2 / np.pi, rel=1e-3)
    beta = 1.0 + 0.5j
    Wc = ph.wigner(ph.coherent(beta, 20), x)
    iy, ix = np.unravel_index(np.argmax(Wc), Wc.shape)
    assert x[ix] == pytest.approx(beta.real, abs=0.03)
    assert x[iy] == pytest.approx(beta.imag, abs=0.03)


def test_cat_state_has_negative_fringes_and_unit_norm():
    psi = ph.cat(2.0, 25, parity=+1)
    assert np.vdot(psi, psi).real == pytest.approx(1.0)
    x = np.linspace(-4, 4, 161)
    W = ph.wigner(psi, x)
    assert W.min() < -0.2
    assert W.sum() * (x[1] - x[0]) ** 2 == pytest.approx(1.0, abs=2e-3)


def test_bloch_vector_and_rabi_trajectory():
    up = ph.dm(np.array([1, 0]))
    np.testing.assert_allclose(ph.bloch_vector(up), [0, 0, 1])
    plus = ph.dm(np.array([1, 1]) / np.sqrt(2))
    np.testing.assert_allclose(ph.bloch_vector(plus), [1, 0, 0], atol=1e-12)
    t = np.linspace(0, 1, 5)
    v = ph.rabi_trajectory(rabi=0.5, detuning=0.0, t=t)          # half a Rabi period at t = 1
    np.testing.assert_allclose(v[0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(v[-1], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)


def test_rb_decay_model():
    m = np.array([0, 10, 100])
    y = ph.rb_decay(m, p=0.99, A=0.5, B=0.5)
    np.testing.assert_allclose(y, 0.5 * 0.99 ** m + 0.5)


def test_rabi_population_resonant_and_detuned():
    assert ph.rabi_population(1.0, 0.0, 0.5) == pytest.approx(1.0)
    assert ph.rabi_population(1.0, 1.0, 1 / (2 * np.sqrt(2))) == pytest.approx(0.5)
