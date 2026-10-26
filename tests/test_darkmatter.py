"""The dark-matter wave models (field amplitude, occupation number, coherence time, line shape, field
trace, rotation curve) and the plots that draw them."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig import physics as ph  # noqa: E402
from scqfig.check import violations  # noqa: E402


def draw(fn, *args, style="field", **kw):
    with sq.style(style):
        fig, ax = sq.figure()
        out = fn(ax, *args, **kw)
        fig.canvas.draw()
    return out, fig


# ---- physics ------------------------------------------------------------------------------------

def test_dm_field_is_eps_times_sqrt_2rho_over_eps0():
    # 0.45 GeV/cm^3 = 7.21e-5 J/m^3, so the free-space field is 4.04e3 V/m per unit mixing.
    assert ph.dm_field(1.0) == pytest.approx(4035.5, rel=1e-4)
    assert ph.dm_field(1e-11) == pytest.approx(4.0355e-8, rel=1e-4)
    assert ph.dm_field(1.0, rho=0.3) == pytest.approx(ph.dm_field(1.0) * np.sqrt(0.3 / 0.45), rel=1e-12)


def test_occupation_number_scales_as_m_to_minus_four():
    # 20 µeV at v = 1e-3 c: n = 2.25e13 per cm^3, de Broglie length 62 m, about 5.4e24 per cell.
    assert ph.occupation_number(20.0, v=299.792458) == pytest.approx(5.36e24, rel=5e-3)
    np.testing.assert_allclose(ph.occupation_number(40.0) / ph.occupation_number(20.0), 1 / 16, rtol=1e-12)
    m1 = 20.0 * ph.occupation_number(20.0) ** 0.25 * 1e-6           # eV where N = 1
    assert 20 < m1 < 60


def test_coherence_time_conventions_differ_by_two_pi():
    assert ph.coherence_time(5.0) == pytest.approx(200.0)
    assert ph.coherence_time(6.0, angular=True) == pytest.approx(26.53, rel=1e-3)
    assert ph.coherence_time(5.0) / ph.coherence_time(5.0, angular=True) == pytest.approx(2 * np.pi)


def test_lineshape_is_normalised_with_mean_one_and_a_half_df0():
    df = np.linspace(0, 20e-6, 200001)                              # GHz offsets at 5 GHz
    g = ph.dm_lineshape(df, 5.0)
    d0 = 5.0 * (220e3) ** 2 / (2 * 299792458.0 ** 2)
    assert (g * np.gradient(df)).sum() == pytest.approx(1.0, rel=1e-4)
    assert (g * df * np.gradient(df)).sum() == pytest.approx(1.5 * d0, rel=1e-3)
    assert d0 * 1e9 == pytest.approx(1346, rel=1e-3)               # Hz
    assert ph.dm_lineshape(-1e-6, 5.0) == 0.0


def test_field_trace_is_seeded_and_slow_on_one_shot():
    t = np.linspace(0, 3000, 1501)
    A, B = ph.dm_field_trace(t, 5.0, seed=3), ph.dm_field_trace(t, 5.0, seed=3)
    np.testing.assert_array_equal(A, B)
    assert not np.allclose(A, ph.dm_field_trace(t, 5.0, seed=4))
    # Over 10 µs (well inside the ~200 µs coherence time) the envelope moves by a few per cent at most.
    assert np.max(np.abs(A[1:6] - A[0])) < 0.1 * np.sqrt(np.mean(np.abs(A) ** 2))


def test_rotation_curve_is_flat_with_a_halo_and_falls_without():
    r = np.array([8.0, 15.0, 30.0])
    vis, halo, tot = ph.rotation_curve(r)
    np.testing.assert_allclose(tot, np.hypot(vis, halo))
    assert np.ptp(tot) < 0.05 * tot.mean() and 200 < tot[0] < 240
    assert vis[2] < 0.6 * vis[0]


# ---- plots --------------------------------------------------------------------------------------

def test_rotation_curve_plot_draws_three_lines_and_points():
    r = np.linspace(0.2, 30, 60)
    vis, halo, tot = ph.rotation_curve(r)
    out, fig = draw(sq.plots.rotation_curve, r, vis, halo, observed=(r[::10], tot[::10], np.full(6, 5.0)))
    np.testing.assert_allclose(out["total"].get_ydata(), tot)
    assert out["halo"].get_color() == sq.PALETTE["magenta"]
    assert fig.axes[0].get_legend() is not None
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.rotation_curve, r, vis[:5], halo)
    plt.close("all")


def test_occupation_finds_the_crossing_and_shades_the_band():
    m = np.logspace(-22, 12, 200)
    N = ph.occupation_number(m * 1e6)
    out, fig = draw(sq.plots.occupation, m, N, band=(4e-6, 4e-5))
    ax = fig.axes[0]
    assert ax.get_xscale() == ax.get_yscale() == "log"
    m1 = (ph.occupation_number(1.0) ** 0.25) * 1e-6                 # where N = 1, in eV
    assert out["crossing"] == pytest.approx(m1, rel=1e-2)
    assert {"wave-like", "particle-like"} <= {t.get_text() for t in ax.texts}
    plt.close(fig)
    out, fig = draw(sq.plots.occupation, m[:20], N[:20])             # never crosses one
    assert out["crossing"] is None
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.occupation, m, -N)
    plt.close("all")


def test_lineshape_and_field_trace_plots():
    df = np.linspace(0, 8, 100)
    out, fig = draw(sq.plots.lineshape, df, ph.dm_lineshape(df, 5e6), mean=2.0)
    assert out["line"].get_xdata()[-1] == 8
    plt.close(fig)
    t = np.linspace(0, 1000, 201)
    out, fig = draw(sq.plots.field_trace, t, ph.dm_field_trace(t, 5.0), window=(100, 200))
    amp = out["amplitude"].get_ydata()
    assert np.mean(amp ** 2) == pytest.approx(1.0) and out["twin"] is not None
    plt.close(fig)
    out, fig = draw(sq.plots.field_trace, t, ph.dm_field_trace(t, 5.0), phase=False)
    assert out["phase"] is None and len(fig.axes) == 1
    plt.close("all")


CASES = [
    (sq.plots.rotation_curve, (np.linspace(0.2, 30, 40),) + ph.rotation_curve(np.linspace(0.2, 30, 40))[:2], {}),
    (sq.plots.occupation, (np.logspace(-22, 12, 80), ph.occupation_number(np.logspace(-16, 18, 80))),
     dict(band=(4e-6, 4e-5))),
    (sq.plots.lineshape, (np.linspace(0, 8, 80), ph.dm_lineshape(np.linspace(0, 8, 80), 5e6)), dict(mean=2.0)),
    (sq.plots.field_trace, (np.linspace(0, 1000, 101), ph.dm_field_trace(np.linspace(0, 1000, 101), 5.0)),
     dict(window=(100, 200))),
]


@pytest.mark.parametrize("style", ["field", "folio"])
@pytest.mark.parametrize("fn,args,kw", CASES, ids=[c[0].__name__ for c in CASES])
def test_dark_matter_plots_keep_the_colour_rule(fn, args, kw, style):
    _, fig = draw(fn, *args, style=style, **kw)
    bad = violations(fig)
    plt.close(fig)
    assert not bad, [(type(a).__name__, c) for a, c in bad[:8]]
