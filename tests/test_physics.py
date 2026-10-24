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


# ---- circuits: normal modes and black-box quantization ---------------------------------------

E, HBAR = 1.602176634e-19, 6.62607015e-34 / (2 * np.pi)


def resonator(f, Z):
    """L and C of an LC resonator of frequency f (Hz) and impedance Z (ohm)."""
    w = 2 * np.pi * f
    return Z / w, 1 / (Z * w)


def two_node(Cq, Cg, Cr, LJ, Lr):
    """Capacitance and inverse-inductance matrices of a transmon (node 0, junction as L_J) coupled
    through Cg to an LC resonator (node 1)."""
    return np.array([[Cq + Cg, -Cg], [-Cg, Cr + Cg]]), np.diag([1 / LJ, 1 / Lr])


def charging_energy(C):
    return E ** 2 / (2 * C) / (2 * np.pi * HBAR) / 1e9


def test_ej_from_lj_follows_the_163_ghz_nh_rule():
    assert ph.ej_from_lj(10e-9) == pytest.approx(16.35, rel=1e-3)


def test_normal_modes_of_two_lc_resonators_coupled_by_a_capacitor():
    L, C, Cc = 2e-9, 300e-15, 10e-15
    Cm = np.array([[C + Cc, -Cc], [-Cc, C + Cc]])
    f, V = ph.normal_modes(Cm, np.diag([1 / L, 1 / L]))
    # in phase the coupling capacitor carries no current; out of phase it loads each side with 2 Cc
    slow, fast = 1 / (2 * np.pi * np.sqrt(L * (C + 2 * Cc))), 1 / (2 * np.pi * np.sqrt(L * C))
    np.testing.assert_allclose(f, np.array([slow, fast]) / 1e9, rtol=1e-10)
    np.testing.assert_allclose(V.T @ Cm @ V, np.eye(2), atol=1e-10)


def test_port_susceptance_of_a_parallel_lc():
    L, C = 10e-9, 100e-15
    f = np.array([3.0, 5.0, 8.0])
    w = 2 * np.pi * f * 1e9
    B = ph.port_susceptance(f, np.array([[C]]), np.array([[1 / L]]))
    np.testing.assert_allclose(B, w * C - 1 / (w * L), rtol=1e-12)


def test_black_box_finds_the_normal_modes_and_their_zero_point_flux():
    Lr, Cr = resonator(7.2e9, 50.0)
    C, Linv = two_node(65e-15, 5e-15, Cr, 11e-9, Lr)
    bb = ph.black_box(lambda f: ph.port_susceptance(f, C, Linv), ph.ej_from_lj(11e-9), np.linspace(1, 20, 2001))
    f, V = ph.normal_modes(C, Linv)
    np.testing.assert_allclose(bb["f"], f, rtol=1e-9)
    # an independent route: phi_zpf^2 = (2e/hbar)^2 hbar/(2 w) v_port^2 for C-normalised mode vectors
    w = 2 * np.pi * f * 1e9
    np.testing.assert_allclose(bb["phi"] ** 2, 2 * E ** 2 * V[0] ** 2 / (HBAR * w), rtol=1e-6)
    np.testing.assert_allclose(bb["Z"], bb["phi"] ** 2 * 6.62607015e-34 / (4 * np.pi * E ** 2), rtol=1e-12)
    K = bb["K"]
    assert np.all(K < 0) and K[0, 1] == K[1, 0]
    assert K[0, 1] ** 2 == pytest.approx(4 * K[0, 0] * K[1, 1])     # first order: chi^2 = 4 alpha_q alpha_r


def test_black_box_gives_minus_ec_for_a_single_junction_mode():
    LJ, C = 12e-9, 80e-15
    B = lambda f: ph.port_susceptance(f, np.array([[C]]), np.array([[1 / LJ]]))
    bb = ph.black_box(B, ph.ej_from_lj(LJ), np.linspace(1, 15, 141))
    EC = charging_energy(C)
    assert bb["K"][0, 0] == pytest.approx(-EC, rel=1e-6)
    assert bb["phi"][0] ** 2 == pytest.approx(np.sqrt(2 * EC / ph.ej_from_lj(LJ)), rel=1e-6)


def test_black_box_needs_a_mode_in_the_search_range():
    B = lambda f: ph.port_susceptance(f, np.array([[80e-15]]), np.array([[1 / 12e-9]]))   # one mode at 5.1 GHz
    with pytest.raises(ValueError):
        ph.black_box(B, 13.6, np.linspace(8, 12, 41))


def test_transmon_resonator_without_coupling_returns_the_bare_parts():
    Lr, Cr = resonator(7.0e9, 50.0)
    x = ph.transmon_resonator(80e-15, 1e-21, Cr, 10e-9, Lr)
    f = ph.transmon_spectrum(ph.ej_from_lj(10e-9), charging_energy(80e-15))
    np.testing.assert_allclose(x["f"], [f[1], 7.0], rtol=1e-6)
    assert x["K"][0, 0] == pytest.approx(f[2] - 2 * f[1], rel=1e-6)
    assert abs(x["K"][0, 1]) < 1e-9 and abs(x["K"][1, 1]) < 1e-9 and x["g"] < 1e-6


def test_black_box_agrees_with_exact_diagonalization_for_weak_anharmonicity():
    Lr, Cr = resonator(9e9, 50.0)
    args = (400e-15, 10e-15, Cr, 2e-9, Lr)
    bb = ph.black_box(lambda f: ph.port_susceptance(f, *two_node(*args)), ph.ej_from_lj(2e-9),
                      np.linspace(1, 20, 2001))
    x = ph.transmon_resonator(*args)
    assert bb["K"][0, 0] == pytest.approx(x["K"][0, 0], rel=0.05)
    assert bb["K"][0, 1] == pytest.approx(x["K"][0, 1], rel=0.05)


def test_first_order_black_box_overstates_chi_of_a_strongly_anharmonic_transmon():
    Lr, Cr = resonator(7.2e9, 50.0)
    args = (65e-15, 5e-15, Cr, 11e-9, Lr)
    bb = ph.black_box(lambda f: ph.port_susceptance(f, *two_node(*args)), ph.ej_from_lj(11e-9),
                      np.linspace(1, 20, 2001))
    x = ph.transmon_resonator(*args)
    assert x["K"][0, 1] < 0 and abs(bb["K"][0, 1]) > 1.3 * abs(x["K"][0, 1])
