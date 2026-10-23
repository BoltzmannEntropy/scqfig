"""Noise and error family: the toy burst model and the values the plots return."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.plots import noise  # noqa: E402


def test_burst_events_background_and_bursts():
    t, E = noise.burst_events(nq=20, nt=4000, p_bg=0.01, bursts=((1000, 8), (3000, 2)), tau=30, rng=1)
    assert E.shape == (20, 4000) and E.dtype == bool and t.size == 4000
    quiet = E[:, 2000:2900].mean()
    assert quiet == pytest.approx(0.01, abs=0.004)
    assert E[:, 1000:1010].mean() > 0.3                       # a burst flags most qubits at once
    near, far = E[8, 1000:1030].mean(), E[19, 1000:1030].mean()
    assert near > far                                         # strength falls off with distance from the hit


def test_noise_psd_fit_and_annotation():
    f = np.logspace(-3, 1, 60)
    S = 4.0 / f ** 0.9
    with sq.style("field"):
        fig, ax = sq.figure("column")
        r = noise.noise_psd(ax, f, S)
        plt.close(fig)
    assert r["alpha"] == pytest.approx(0.9, abs=1e-6)
    assert r.derived["amplitude"] == pytest.approx(2.0, rel=1e-6)


def test_error_budget_returns_sum_and_gap():
    with sq.style("folio"):
        fig, ax = sq.figure("column")
        out = noise.error_budget(ax, {"T1": 0.3, "T2": 0.2, "leakage": 0.1}, measured=0.75)
        plt.close(fig)
    assert out["budget"] == pytest.approx(0.6)
    assert out["unexplained"] == pytest.approx(0.15)


def test_error_correlation_masks_diagonal():
    rng = np.random.default_rng(0)
    E = rng.random((5, 400)) < 0.1
    with sq.style("field"):
        fig, ax = sq.figure("column")
        im, _ = noise.error_correlation(ax, E)
        plt.close(fig)
    C = im.get_array()
    assert np.all(np.ma.getmaskarray(C).diagonal()) or np.all(np.isnan(np.asarray(C.filled(np.nan)).diagonal()))
