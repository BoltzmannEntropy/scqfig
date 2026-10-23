"""Models and derived quantities behind the time-domain and state plots."""
import numpy as np
import pytest

from scqfig import fit, physics as ph
from scqfig.plots.states import _outline, basis_labels, bloch_decay
from scqfig.plots.timedomain import interleaved_error, pi_amplitude, tls_t1

rng = np.random.default_rng(11)


# ---- pi amplitude -----------------------------------------------------------------------------

@pytest.mark.parametrize("a_pi", [0.2, 0.417, 0.8])
def test_pi_amplitude_is_first_maximum_of_cosine_fit(a_pi):
    a = np.linspace(0, 2.2 * a_pi, 90)
    y = 0.03 + 0.92 * np.sin(np.pi * a / (2 * a_pi)) ** 2 + rng.normal(0, 0.01, a.size)
    assert pi_amplitude(fit.cosine(a, y)) == pytest.approx(a_pi, rel=0.01)


def test_pi_amplitude_handles_negative_amplitude_and_frequency():
    f = fit.Fit(fit.cosine_model, {"A": -0.5, "f": -1.25, "phi": 0.0, "B": 0.5}, {})
    # -0.5 cos(-2 pi 1.25 x) + 0.5 peaks first at x = 1 / (2 * 1.25)
    assert pi_amplitude(f) == pytest.approx(0.4)


def test_pi_amplitude_skips_a_maximum_at_zero():
    f = fit.Fit(fit.cosine_model, {"A": 0.5, "f": 2.0, "phi": 0.0, "B": 0.5}, {})
    assert pi_amplitude(f) == pytest.approx(0.5)


# ---- interleaved RB -----------------------------------------------------------------------------

def test_interleaved_error_formula_and_propagation():
    r, e = interleaved_error(0.997, 0.9955, d=2, p_err=1e-4, p_int_err=2e-4)
    assert r == pytest.approx((1 - 0.9955 / 0.997) / 2)
    assert e == pytest.approx(0.5 * np.hypot(2e-4 / 0.997, 0.9955 * 1e-4 / 0.997 ** 2))


def test_interleaved_error_two_qubit_dimension_and_no_errors():
    r, e = interleaved_error(0.98, 0.97, d=4)
    assert r == pytest.approx((1 - 0.97 / 0.98) * 3 / 4)
    assert e is None


def test_interleaved_error_is_zero_for_a_perfect_gate():
    assert interleaved_error(0.99, 0.99)[0] == 0.0


# ---- TLS-limited T1 -----------------------------------------------------------------------------

def test_tls_t1_on_resonance_matches_the_peak_rate():
    g, gamma, bg = 0.1, 1.0, 100.0                       # MHz, MHz, µs
    peak = 2 * (2 * np.pi * g) ** 2 / (2 * np.pi * gamma)
    assert tls_t1(5.2, [5.2], g, gamma, bg) == pytest.approx(1 / (1 / bg + peak))


def test_tls_t1_far_detuned_returns_background_and_half_width_is_gamma():
    g, gamma, bg = 0.1, 1.0, 100.0
    assert tls_t1(5.0, [5.2], g, gamma, bg) == pytest.approx(bg, rel=1e-3)
    on = 1 / tls_t1(5.2, [5.2], g, gamma, bg) - 1 / bg
    half = 1 / tls_t1(5.2 + gamma * 1e-3, [5.2], g, gamma, bg) - 1 / bg
    assert half == pytest.approx(on / 2)


def test_tls_t1_broadcasts_a_frequency_grid_against_defect_tracks():
    freq = np.linspace(5.1, 5.3, 7)
    tracks = np.array([[5.15, 5.25], [5.16, 5.24], [5.17, 5.23]])        # (time, defect)
    T1 = tls_t1(freq[:, None], tracks, g=[0.1, 0.05], gamma=[1.0, 2.0], t1_bg=80.0)
    assert T1.shape == (7, 3)
    assert np.all((T1 > 0) & (T1 <= 80.0))
    single = tls_t1(freq[3], tracks[1], g=[0.1, 0.05], gamma=[1.0, 2.0], t1_bg=80.0)
    assert T1[3, 1] == pytest.approx(single)


# ---- Bloch decay --------------------------------------------------------------------------------

def test_bloch_decay_starts_at_v0_and_relaxes_to_ground():
    v = bloch_decay((1, 0, 0), 1.0, [0.0, 1e3], T1=4.0, T2=2.0)
    assert v[0] == pytest.approx([1, 0, 0])
    assert v[1] == pytest.approx([0, 0, 1], abs=1e-9)


def test_bloch_decay_shrinks_transverse_part_with_t2_and_z_with_t1():
    t = np.linspace(0, 6, 50)
    v = bloch_decay((0, -1, 0), 0.7, t, T1=4.0, T2=2.0)
    assert np.hypot(v[:, 0], v[:, 1]) == pytest.approx(np.exp(-t / 2.0))
    assert v[:, 2] == pytest.approx(1 - np.exp(-t / 4.0))
    assert np.all(np.linalg.norm(v, axis=1) <= 1 + 1e-12)


def test_bloch_decay_without_decay_matches_free_precession():
    t = np.linspace(0, 1, 30)
    v = bloch_decay((1, 0, 0), 0.5, t, T1=np.inf, T2=np.inf)
    assert v == pytest.approx(ph.rotate((1, 0, 0), (0, 0, 1), 2 * np.pi * 0.5 * t))


# ---- labels and outlines ------------------------------------------------------------------------

def test_basis_labels():
    assert basis_labels(4) == [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
    assert basis_labels(2) == [r"$|0\rangle$", r"$|1\rangle$"]
    assert basis_labels(3)[-1] == r"$|2\rangle$"
    assert len(basis_labels(8)) == 8 and basis_labels(8)[5] == r"$|101\rangle$"


def test_outline_hides_the_far_edge_inside_the_bar():
    eye = (0.6, -0.7)                                    # viewer at +x, -y: far corner is (x0, y0 + w)
    segs = _outline(0.0, 0.0, 1.0, h=0.5, b=0.45, eye=eye)
    far = [s for s in segs if np.allclose(s[:, :2], [[0.0, 1.0]] * 2)]
    assert len(segs) == 4 + 3 + 1
    assert far[0][:, 2] == pytest.approx([0.45, 0.5])
    assert len(_outline(0.0, 0.0, 1.0, h=0.5, b=0.6, eye=eye)) == 7      # bar taller: far edge hidden
    neg = _outline(0.0, 0.0, 1.0, h=-0.5, b=0.2, eye=eye)                 # bar points the other way
    assert [s[:, 2].tolist() for s in neg if np.allclose(s[:, :2], [[0.0, 1.0]] * 2)] == [[-0.5, 0.0]]
