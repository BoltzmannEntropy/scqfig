"""Models, fits and return values of the design and readout plot families."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig import physics as ph  # noqa: E402
from scqfig.core import sci  # noqa: E402
from scqfig.plots.design import collision_windows  # noqa: E402
from scqfig.fit import gaussian as fit_gaussian, tls as fit_tls  # noqa: E402
from scqfig.plots.readout import iq_discriminator, readout_shots  # noqa: E402

rng = np.random.default_rng(11)


# ---- design ---------------------------------------------------------------------------------

def test_sci_plain_scientific_and_with_error():
    assert sci(0.92) == "$0.92$"
    assert sci(120.0) == "$120$"
    assert sci(2.43e5) == r"$2.4\times10^{5}$"
    assert sci(9.96e5) == r"$1.0\times10^{6}$"              # rounding carries into the exponent
    assert sci(-3.1e-6) == r"$-3.1\times10^{-6}$"
    assert sci(1.0034e4, 52.0) == r"$(1.003 \pm 0.005)\times10^{4}$"
    assert sci(5.3, 1.7) == r"$5.3 \pm 1.7$"
    assert sci(0.0) == "$0$"


def test_collision_windows_are_symmetric_in_detuning():
    w = collision_windows(-0.22, delta=0.017, delta_12=0.03, delta_2ph=0.004)
    centres = sorted(c for _, c, _, _ in w)
    np.testing.assert_allclose(centres, [-0.22, -0.11, 0.0, 0.11, 0.22])
    assert {k for k, *_ in w} == {"01-01", "01-12", "2-photon"}
    assert all(h > 0 for _, _, h, _ in w)


def test_collision_map_flags_points_inside_bands():
    alpha = -0.22
    with sq.style("field"):
        fig, ax = sq.figure()
        hits = sq.plots.collision_map(ax, alpha, {"ok": (4.90, 5.06), "two-photon": (5.0, 5.0 - alpha / 2),
                                                  "degenerate": (5.0, 5.005)})
        plt.close(fig)
    assert hits == {"ok": [], "two-photon": ["2-photon"], "degenerate": ["01-01"]}


def test_participation_total_is_the_parallel_sum():
    p = {"MA": 2e-5, "MS": 2.6e-4, "SA": 2.1e-4, "substrate": 0.91}
    tand = {"MA": 2e-3, "MS": 2.6e-3, "SA": 2.2e-3, "substrate": 5e-8}
    with sq.style("folio"):
        fig, ax = sq.figure()
        Q = sq.plots.participation(ax, p, tand)
        plt.close(fig)
    assert Q["MS"] == pytest.approx(1 / (2.6e-4 * 2.6e-3))
    assert 1 / Q["total"] == pytest.approx(sum(p[k] * tand[k] for k in p))


def test_capacitance_matrix_prints_signed_entries():
    C = np.array([[80.0, -0.5, 0.0], [-0.5, 85.0, -4.6], [0.0, -4.6, 420.0]])
    with sq.style("field"):
        fig, ax = sq.figure()
        sq.plots.capacitance_matrix(ax, C, ["Q1", "Q2", "R"], cbar=False)
        texts = [t.get_text() for t in ax.texts]
        plt.close(fig)
    assert texts == ["80", "−0.5", "−0.5", "85", "−4.6", "−4.6", "420"]   # zero cells blank


def test_dispersive_shift_and_purcell_curves_follow_physics():
    fq = np.linspace(4.0, 6.5, 50)
    with sq.style("field"):
        fig, ax = sq.figure()
        chi = sq.plots.dispersive_shift(ax, fq, 7.0, 0.08, -0.22)
        T1 = sq.plots.purcell_limit(ax, fq, 7.0, 0.08, 0.001, filter_q=30)
        plt.close(fig)
    np.testing.assert_allclose(chi, 1e3 * ph.dispersive_shift(fq, 7.0, 0.08, -0.22))
    np.testing.assert_allclose(T1["bare"], ph.purcell_t1(fq, 7.0, 0.08, 0.001))
    assert np.all(T1["filter"] > T1["bare"])


def test_convergence_finds_first_pass_below_tolerance():
    v = np.array([7.4, 7.2, 7.16, 7.152, 7.1512, 7.1511])
    with sq.style("folio"):
        fig, ax = sq.figure()
        out = sq.plots.convergence(ax, np.arange(1, 7), v, tol=2e-4)
        plt.close(fig)
    np.testing.assert_allclose(out["rel"], np.abs(np.diff(v)) / v[1:])
    assert out["converged"] == 5


# ---- readout --------------------------------------------------------------------------------

def test_readout_shots_without_decay_are_gaussian_about_the_centre():
    s = readout_shots(0, [1 + 1j, -1 - 1j], 0.2, 20000, rng=np.random.default_rng(1))
    assert np.mean(s) == pytest.approx(1 + 1j, abs=0.01)
    assert np.std(s.real) == pytest.approx(0.2, rel=0.03)
    assert np.std(s.imag) == pytest.approx(0.2, rel=0.03)


def test_readout_shots_decay_fraction_matches_t1():
    n, t_ro, T1 = 40000, 2.0, 10.0
    s = readout_shots(1, [0.0, 1.0], 1e-6, n, t_ro=t_ro, T1=T1, rng=np.random.default_rng(2))
    decayed = np.mean(s.real < 1 - 1e-4)
    assert decayed == pytest.approx(1 - np.exp(-t_ro / T1), abs=0.006)
    assert s.real.min() >= -1e-4                                   # a decayed shot lies between the centres


def test_readout_shots_second_level_cascades_and_thermal_flip():
    s = readout_shots(2, [0.0, 1.0, 2.0], 1e-6, 30000, t_ro=1.0, T1=5.0, rng=np.random.default_rng(3))
    assert np.mean(s.real > 2 - 1e-4) == pytest.approx(np.exp(-2 * 1.0 / 5.0), abs=0.01)   # rate 2/T1 from |2>
    s0 = readout_shots(0, [0.0, 1.0], 1e-6, 30000, t_ro=1e-6, p_th=0.05, rng=np.random.default_rng(4))
    assert np.mean(s0.real > 0.5) == pytest.approx(0.05, abs=0.006)


def test_fit_tls_recovers_parameters():
    nbar = np.logspace(-1, 6, 30)
    q = ph.tls_qi(nbar, 1.3e-6, 4.0, 0.42, 2.2e6) * np.exp(rng.normal(0, 0.01, nbar.size))
    r = fit_tls(nbar, q)
    assert r["F_delta"] == pytest.approx(1.3e-6, rel=0.05)
    assert r["beta"] == pytest.approx(0.42, rel=0.1)
    assert r["Q_other"] == pytest.approx(2.2e6, rel=0.05)
    assert r["nc"] == pytest.approx(4.0, rel=0.4)
    assert r["Qi_low"] == pytest.approx(1 / (1.3e-6 + 1 / 2.2e6), rel=0.05)
    assert all(np.isfinite(v) and v > 0 for v in r.errors.values())


def test_fit_gaussian_recovers_peak():
    x = np.linspace(-3, 3, 61)
    y = 250 * np.exp(-0.5 * ((x - 0.4) / 0.7) ** 2) + rng.normal(0, 2, x.size)
    r = fit_gaussian(x, y)
    assert r["mu"] == pytest.approx(0.4, abs=0.02)
    assert r["sigma"] == pytest.approx(0.7, rel=0.03)
    assert r["A"] == pytest.approx(250, rel=0.03)


def test_iq_discriminator_ignores_decay_tail():
    g = np.random.default_rng(5)
    c = [-1.0 + 0.0j, 1.0 + 0.0j]
    shots = [readout_shots(m, c, 0.25, 8000, t_ro=1.0, T1=8.0, rng=g) for m in (0, 1)]
    d = iq_discriminator(shots)
    np.testing.assert_allclose(d["centres"], c, atol=0.02)
    for S in d["covs"]:                                            # the core, not the tail, sets the width
        np.testing.assert_allclose(np.diag(S), [0.25 ** 2] * 2, rtol=0.1)
    assert np.var(shots[1].real) > 1.3 * 0.25 ** 2                # the naive variance is inflated
    assert d["w"].real == pytest.approx(1.0, abs=0.02)
    assert d["threshold"] == pytest.approx(0.0, abs=0.03)


def test_histogram_fidelity_matches_threshold_counts():
    g = np.random.default_rng(6)
    c = [-0.5 + 0.3j, 0.4 + 0.6j]
    shots = [readout_shots(m, c, 0.16, 4000, t_ro=1.5, T1=20.0, p_th=0.01, rng=g) for m in (0, 1)]
    with sq.style("field"):
        fig, ax = sq.figure()
        out = sq.plots.readout_histogram(ax, shots)
        plt.close(fig)
    d = iq_discriminator(shots)
    proj = [np.real(np.conj(d["w"]) * s) - d["threshold"] for s in shots]
    assert out["P10"] == pytest.approx(np.mean(proj[0] > 0))
    assert out["F"] == pytest.approx(1 - (np.mean(proj[0] > 0) + np.mean(proj[1] <= 0)) / 2)
    assert 0.9 < out["F"] < 1.0


def test_notch_panels_share_one_fit():
    fr, Ql, Qc = 6.83, 1.0e4, 1.5e4
    f = np.linspace(fr - 6 * fr / Ql, fr + 6 * fr / Ql, 201)
    s = ph.notch_s21(f, fr, Ql, Qc, 0.1, a=0.5, alpha=0.7, tau=40.0)
    with sq.style("folio"):
        fig, axes = sq.figure("page", ncols=3)
        r = sq.plots.s21_magnitude(axes[0], f, s)
        assert sq.plots.s21_phase(axes[1], f, s, fit=r) is r
        assert sq.plots.s21_circle(axes[2], f, s, fit=r) is r
        line = axes[0].lines[0].get_ydata()                       # calibrated data sit at 1 off resonance
        plt.close(fig)
    assert r["Ql"] == pytest.approx(Ql, rel=1e-3)
    assert line[0] == pytest.approx(1.0, abs=0.02)


def test_dispersive_readout_measures_two_chi():
    f = np.linspace(7.147, 7.153, 241)
    chi = -0.35e-3
    s0, s1 = (np.abs(ph.notch_s21(f, 7.15 - chi * k, 1.0e4, 1.03e4)) for k in (1, -1))
    with sq.style("field"):
        fig, ax = sq.figure()
        out = sq.plots.dispersive_readout(ax, f, s0, s1)
        plt.close(fig)
    assert out["chi"] == pytest.approx(chi, rel=0.02)
    assert out["f_ro"] == pytest.approx(7.15, abs=1e-6)


def test_assignment_matrix_normalises_rows():
    with sq.style("field"):
        fig, ax = sq.figure()
        im, _ = sq.plots.assignment_matrix(ax, [[90, 10], [20, 80]])
        title = ax.get_title(loc="right")
        plt.close(fig)
    np.testing.assert_allclose(im.get_array(), [[0.9, 0.1], [0.2, 0.8]])
    assert "85.0" in title
