"""Physics and drawing of the spectrum, design and dark-matter plot types: transmon scaling with
E_J/E_C, parity branches, flux tuning, dressed lines, junction curves, frequency lines, design maps,
the dispersive-shift and convergence extensions, and the coupling reach."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig import physics as ph  # noqa: E402
from scqfig.check import violations  # noqa: E402

HBAR, E = 1.054571817e-34, 1.602176634e-19
PHI0 = 2 * np.pi * HBAR / (2 * E)


def draw(fn, *args, style="field", **kw):
    """Run a plot function on a fresh axes; return (its result, the figure)."""
    with sq.style(style):
        fig, ax = sq.figure()
        out = fn(ax, *args, **kw)
        fig.canvas.draw()
    return out, fig


# ---- physics ------------------------------------------------------------------------------------

def test_ej_for_f01_inverts_the_exact_spectrum():
    for f01, EC in ((5.0, 0.25), (3.2, 0.3), (8.0, 0.18)):
        EJ = ph.ej_for_f01(f01, EC)
        assert ph.transmon_spectrum(EJ, EC, nlev=2)[1] == pytest.approx(f01, abs=1e-9)


def test_ej_for_f01_rejects_targets_at_or_below_4ec():
    with pytest.raises(ValueError):
        ph.ej_for_f01(1.0, 0.25)                                    # 4 EC = 1.0 needs E_J = 0
    with pytest.raises(ValueError):
        ph.ej_for_f01(0.5, 0.25)


def test_transmon_asymptotic_forms():
    f01, alpha = ph.transmon_asymptotic(np.array([50.0, 200.0]), 1.0)
    np.testing.assert_allclose(f01, [19.0, 39.0])
    np.testing.assert_allclose(alpha, [-1.0, -1.0])
    f, a = ph.transmon_asymptotic(50.0, 0.2)
    assert f == pytest.approx(np.sqrt(8 * 50 * 0.2) - 0.2) and a == pytest.approx(-0.2)


def test_level_dispersion_signs_and_approach_to_koch():
    assert ph.level_dispersion(20.0, 1.0, 0) > 0 > ph.level_dispersion(20.0, 1.0, 1)
    r = np.array([30.0, 60.0, 120.0])
    ex = np.abs(ph.level_dispersion(r, 1.0, 1) - ph.level_dispersion(r, 1.0, 0))
    ko = np.abs(ph.koch_dispersion(r, 1.0, 1) - ph.koch_dispersion(r, 1.0, 0))
    ratio = ex / ko
    assert np.all((ratio > 0.75) & (ratio < 1.0))                 # leading order, approached from below
    assert np.all(np.diff(ratio) > 0)
    assert np.isscalar(ph.level_dispersion(20.0, 1.0, 0))


def test_koch_dispersion_scales_with_ec():
    assert ph.koch_dispersion(10.0, 0.5, 1) == pytest.approx(0.5 * ph.koch_dispersion(20.0, 1.0, 1))


def test_parity_branches_odd_is_even_shifted_by_half_a_cooper_pair():
    ng = np.linspace(-0.5, 0.5, 21)
    even, odd = ph.parity_branches(12.0, 0.6, ng, nlev=3)
    assert even.shape == odd.shape == (21, 3)
    np.testing.assert_allclose(odd, ph.charge_dispersion(12.0, 0.6, ng + 0.5, 3), atol=1e-12)
    f_e, f_o = even[:, 1] - even[:, 0], odd[:, 1] - odd[:, 0]
    i = np.argmin(np.abs(ng - 0.25))
    assert f_e[i] == pytest.approx(f_o[i], abs=1e-12)               # zero contrast at n_g = 1/4
    assert np.ptp(f_e) == pytest.approx(abs(ph.level_dispersion(12.0, 0.6, 1, 20)
                                            - ph.level_dispersion(12.0, 0.6, 0, 20)), rel=1e-6)


def test_flux_at_frequency_inverts_the_tuning_curve():
    for d in (0.0, 0.3):
        x = ph.flux_at_frequency(4.5, 25.0, 0.25, d)
        assert 0 < x < 0.5
        assert ph.tunable_frequencies(x, 25.0, 0.25, d, nlev=2)[0, 0] == pytest.approx(4.5, abs=1e-9)
    assert ph.flux_at_frequency(ph.tunable_frequencies(0.0, 25.0, 0.25, nlev=2)[0, 0], 25.0, 0.25) == \
        pytest.approx(0.0, abs=1e-9)


def test_flux_at_frequency_rejects_targets_outside_the_range():
    with pytest.raises(ValueError):
        ph.flux_at_frequency(9.0, 25.0, 0.25)                       # above the sweet spot
    with pytest.raises(ValueError):
        ph.flux_at_frequency(2.0, 25.0, 0.25, d=0.3)                # below the asymmetric minimum


def test_junction_inductance_from_resistance():
    Rn, gap = 1.0e4, 180e-6
    Ic = np.pi * gap * E / (2 * E * Rn)                             # Ambegaokar-Baratoff
    assert ph.lj_from_rn(Rn, gap) == pytest.approx(PHI0 / (2 * np.pi * Ic), rel=1e-9)
    assert ph.lj_from_rn(Rn, gap) == pytest.approx(11.64e-9, rel=1e-3)
    np.testing.assert_allclose(ph.rn_from_lj(ph.lj_from_rn([5e3, 2e4], gap), gap), [5e3, 2e4])


def test_transmon_dressed_splits_by_2g_on_resonance():
    r = ph.transmon_dressed(np.linspace(6.9, 7.1, 41), 7.0, 0.05, 0.25)
    gap = (r["f"][:, 1] - r["f"][:, 0]).min()
    assert gap == pytest.approx(0.1, rel=0.01)
    assert np.all(np.diff(r["f"], axis=1) > 0)


def test_transmon_dressed_rwa_chi_matches_perturbation_with_exact_matrix_elements():
    fq, fr, g, EC = 5.0, 7.0, 0.05, 0.25
    r = ph.transmon_dressed(fq, fr, g, EC, rwa=True)
    D, a, lam = fq - fr, r["alpha"][0], r["lam"][0]
    chi = g ** 2 / D - lam ** 2 * g ** 2 / (2 * (D + a))           # Koch Eqs. 3.9-3.10, exact lambda
    assert r["chi"][0] == pytest.approx(chi, rel=0.02)
    assert 1.3 < lam < np.sqrt(2)                                   # tends to sqrt 2 from below
    ladder = ph.dispersive_shift(fq, fr, g, a)                      # lambda = sqrt 2 differs
    assert abs(r["chi"][0] - ladder) > 0.1 * abs(ladder)


def test_transmon_dressed_far_detuned_branches_sit_near_the_bare_modes():
    r = ph.transmon_dressed(4.0, 8.0, 0.05, 0.25)
    np.testing.assert_allclose(r["f"][0], [4.0, 8.0], atol=2e-3)


def test_transmon_dressed_needs_three_levels():
    with pytest.raises(ValueError):
        ph.transmon_dressed(5.0, 7.0, 0.05, 0.25, nlev=2)


def test_thermal_population_and_its_inverse():
    assert ph.thermal_population(5.0, 0.0) == 0.0
    n = ph.thermal_population(5.0, 0.05)
    assert n == pytest.approx(1 / np.expm1(6.62607015e-34 * 5e9 / (1.380649e-23 * 0.05)))
    assert ph.thermal_frequency(n, 0.05) == pytest.approx(5.0)
    assert ph.thermal_population(1.0, 1.0) == pytest.approx(1.380649e-23 / 6.62607015e-34 / 1e9, rel=0.03)  # classical


def test_mass_frequency_conversion():
    assert ph.mass_from_frequency(1.0) == pytest.approx(4.135667696, rel=1e-9)
    np.testing.assert_allclose(ph.frequency_from_mass(ph.mass_from_frequency([1.0, 7.3])), [1.0, 7.3])


# ---- spectrum plots -----------------------------------------------------------------------------

def test_ratio_scaling_f01_approaches_the_asymptote():
    out, fig = draw(sq.plots.ratio_scaling, "f01")
    plt.close(fig)
    r, y, ya = out["ratio"], out["exact"], out["asymptote"]
    assert np.allclose(ya, np.sqrt(8 * r) - 1)
    s = np.sqrt(8 * r[-1])
    assert y[-1] == pytest.approx(s - 1 - 1 / s, abs=5e-3)          # next order: f01/EC = s - 1 - 1/s
    rel = np.abs(y - ya) / ya
    assert rel[-1] < 1e-3 and rel[-1] < rel[len(r) // 2]


def test_ratio_scaling_alpha_tends_to_minus_one():
    out, fig = draw(sq.plots.ratio_scaling, "alpha", np.linspace(10, 300, 60), style="folio")
    plt.close(fig)
    y = out["exact"]
    assert np.all(np.diff(y) > 0) and -1.07 < y[-1] < -1.0
    np.testing.assert_allclose(out["asymptote"], -1.0)


def test_ratio_scaling_dispersion_masks_values_below_the_rounding_floor():
    out, fig = draw(sq.plots.ratio_scaling, "dispersion", np.linspace(5, 300, 100))
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)
    y = out["exact"]
    assert np.isfinite(y[:20]).all() and np.isnan(y[-1])            # 300 is far below the floor
    assert np.nanmin(y) > 1e-12
    assert np.isnan(out["asymptote"][out["ratio"] < 5]).all()


def test_ratio_scaling_parity_divides_by_the_mean_frequency_and_marks_points():
    pts = {"A": 20.0, "meas": (30.0, 1e-3, "data")}
    out, fig = draw(sq.plots.ratio_scaling, "parity", np.linspace(8, 60, 53), points=pts)
    plt.close(fig)
    r0, v0 = out["points"]["A"]
    eps = abs(ph.level_dispersion(20.0, 1.0, 1, 15) - ph.level_dispersion(20.0, 1.0, 0, 15))
    assert v0 == pytest.approx(eps / ph.transmon_spectrum(20.0, 1.0, 0.25, 15, 2)[1], rel=1e-9)
    assert out["points"]["meas"] == (30.0, 1e-3)


def test_ratio_scaling_asymptote_options_and_bad_input():
    out, fig = draw(sq.plots.ratio_scaling, "f01", asymptote=False)
    plt.close(fig)
    assert np.isnan(out["asymptote"]).all()
    out, fig = draw(sq.plots.ratio_scaling, "f01", asymptote=50.0)
    plt.close(fig)
    assert np.isnan(out["asymptote"][out["ratio"] < 50]).all() and np.isfinite(out["asymptote"][-1])
    with pytest.raises(ValueError):
        draw(sq.plots.ratio_scaling, "T1")
    with pytest.raises(ValueError):
        draw(sq.plots.ratio_scaling, "f01", [10.0])
    with pytest.raises(ValueError):
        draw(sq.plots.ratio_scaling, "f01", [0.0, 10.0])
    plt.close("all")


def test_parity_bands_splitting_and_centring():
    EJ, EC = 12.8, 0.626
    out, fig = draw(sq.plots.parity_bands, EJ, EC)
    plt.close(fig)
    split = 1e3 * abs(ph.level_dispersion(EJ, EC, 1, 20) - ph.level_dispersion(EJ, EC, 0, 20))
    assert out["splitting"] == pytest.approx(split)
    i0 = np.argmin(np.abs(out["ng"]))
    assert out["even"][i0] - out["odd"][i0] == pytest.approx(split, rel=1e-6)
    assert out["even"][i0] == pytest.approx(-out["odd"][i0], rel=1e-9)   # centred on the midpoint
    k = np.argmin(np.abs(out["ng"] - 0.25))
    assert abs(out["even"][k] - out["odd"][k]) < 1e-6


def test_parity_bands_units_levels_and_bad_input():
    a, fig = draw(sq.plots.parity_bands, 12.0, 0.6)
    plt.close(fig)
    b, fig = draw(sq.plots.parity_bands, 12.0, 0.6, unit="kHz", mark=False, style="folio")
    plt.close(fig)
    np.testing.assert_allclose(b["even"], 1e3 * a["even"])
    lv, fig = draw(sq.plots.parity_bands, 12.0, 0.6, level=2, ng=np.linspace(-1, 1, 81))
    plt.close(fig)
    assert lv["splitting"] == pytest.approx(1e3 * abs(ph.level_dispersion(12.0, 0.6, 2, 20)))
    with pytest.raises(ValueError):
        draw(sq.plots.parity_bands, 12.0, 0.6, unit="Hz")
    with pytest.raises(ValueError):
        draw(sq.plots.parity_bands, 12.0, 0.6, level=-1)
    plt.close("all")


def test_parity_bands_colours_legend_and_note_options():
    from matplotlib.colors import to_hex
    blue, orange = sq.PALETTE["blue"], sq.PALETTE["orange"]
    out, fig = draw(sq.plots.parity_bands, 16.3, 0.78, level=1, colors=(blue, orange), legend=False, note=False)
    ax = fig.axes[0]
    even, odd = ax.lines[0], ax.lines[1]
    assert (to_hex(even.get_color()), to_hex(odd.get_color())) == (blue, orange)
    assert ax.get_legend() is None
    assert not [t for t in ax.texts if "epsilon" in t.get_text()]      # the arrow stays, its note does not
    assert any(p.get_arrowstyle() for p in ax.patches)
    plt.close(fig)
    _, fig = draw(sq.plots.parity_bands, 16.3, 0.78, level=1)       # defaults unchanged: legend and note
    ax = fig.axes[0]
    assert ax.get_legend() is not None and [t for t in ax.texts if "epsilon" in t.get_text()]
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.parity_bands, 16.3, 0.78, colors=(blue,))
    plt.close("all")


def test_design_map_fill_takes_another_colormap():
    from matplotlib.colors import to_hex
    x, y = np.logspace(-3, 1, 30), np.logspace(-2, 0, 20)
    Z = y[:, None] ** 2 / x[None, :]
    out, fig = draw(sq.plots.design_map, x, y, Z, levels=[1.0, 2.0, 3.0], cmap="Greys")
    fills = [to_hex(c) for c in out["fill"].get_facecolor()]
    assert len(fills) == 4 and fills[0] == "#ffffff"
    assert all(h[1:3] == h[3:5] == h[5:7] for h in fills)           # every band a grey
    assert not violations(fig)
    plt.close(fig)
    default, fig = draw(sq.plots.design_map, x, y, Z, levels=[1.0, 2.0, 3.0])
    assert [to_hex(c) for c in default["fill"].get_facecolor()] != fills    # default: the style's map
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, y, Z, levels=[1.0, 2.0], cmap="no_such_map")
    plt.close("all")


def test_design_map_places_a_contour_label_where_asked():
    x, y = np.logspace(-3, 1, 60), np.logspace(-2, 0, 40)
    Z = y[:, None] ** 2 / x[None, :]
    out, fig = draw(sq.plots.design_map, x, y, Z, levels=[1.0, 10.0], fill=False, label_xy={1.0: (1e-2, 0.1)})
    ax = fig.axes[0]
    labels = {t.get_text(): t.get_position() for t in ax.texts}
    lx, ly = labels["1"]
    assert lx == pytest.approx(1e-2, rel=0.15) and ly == pytest.approx(0.1, rel=0.15)   # on R = 1 nearest the spot
    assert "10" in labels                                                               # the other keeps its default
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, y, Z, levels=[1.0, 10.0], label_xy={5.0: (1e-2, 0.1)})
    plt.close("all")


def test_ratio_scaling_colour_of_the_exact_curve():
    from matplotlib.colors import to_hex
    ink = sq.PALETTE["ink"]
    _, fig = draw(sq.plots.ratio_scaling, "parity", np.linspace(8, 60, 27), color=ink, legend=False)
    assert to_hex(fig.axes[0].lines[0].get_color()) == ink
    plt.close(fig)
    _, fig = draw(sq.plots.ratio_scaling, "parity", np.linspace(8, 60, 27))     # default: the model colour
    assert to_hex(fig.axes[0].lines[0].get_color()) == to_hex(sq.FIELD.model)
    plt.close(fig)


def test_flux_tuning_curves_and_marks():
    out, fig = draw(sq.plots.flux_tuning, 22.0, 0.24, d=(0.0, 0.2), detune=(0.5, 1.0))
    plt.close(fig)
    np.testing.assert_allclose(out["f01"][0.0], ph.tunable_frequencies(out["flux"], 22.0, 0.24, nlev=2)[:, 0])
    assert out["f01"][0.2].min() > out["f01"][0.0].min()            # asymmetry keeps a floor
    f0 = ph.tunable_frequencies(0.0, 22.0, 0.24, nlev=2)[0, 0]
    for dt, x in out["marks"].items():
        assert ph.tunable_frequencies(x, 22.0, 0.24, nlev=2)[0, 0] == pytest.approx(f0 - dt, abs=1e-9)
    assert out["marks"][1.0] > out["marks"][0.5]
    with pytest.raises(ValueError):
        draw(sq.plots.flux_tuning, 22.0, 0.24, d=0.2, detune=6.0)   # below the asymmetric floor
    plt.close("all")


def test_dressed_lines_two_level_matches_jaynes_cummings():
    fq = np.linspace(6.0, 8.0, 201)                                 # contains fr = 7.0
    out, fig = draw(sq.plots.dressed_lines, fq, 7.0, 0.06, mark=6.5, points={"seen": (7.5, 7.56, "data")})
    plt.close(fig)
    up, lo = ph.dressed_frequencies(fq, 7.0, 0.06)
    np.testing.assert_allclose(out["upper"], up)
    np.testing.assert_allclose(out["lower"], lo)
    assert out["split"] == pytest.approx(0.12) and out["at"] == pytest.approx(7.0)


def test_dressed_lines_transmon_splitting_is_close_to_2g():
    out, fig = draw(sq.plots.dressed_lines, np.linspace(6.5, 7.5, 101), 7.0, 0.08, EC=0.25, style="folio")
    plt.close(fig)
    assert out["split"] == pytest.approx(0.16, rel=0.02)
    assert np.all(out["upper"] > out["lower"])


# ---- design plots -------------------------------------------------------------------------------

def test_junction_curve_follows_the_exact_spectrum():
    Rn = np.linspace(5.0, 12.0, 15)
    out, fig = draw(sq.plots.junction_curve, Rn, {"A": 0.2, "B": 0.25}, gap=180e-6, points={"A": 7.0},
                    spread=0.05, window=(8, 14), band=(4, 6))
    assert len(fig.axes[0].child_axes) == 1                         # the L_J axis on top
    plt.close(fig)
    want = [ph.transmon_spectrum(ph.ej_from_lj(ph.lj_from_rn(r * 1e3, 180e-6)), 0.2, nlev=2)[1] for r in Rn]
    np.testing.assert_allclose(out["f01"]["A"], want)
    x0, f0, (lo, hi) = out["points"]["A"]
    assert lo < f0 < hi and np.all(np.diff(out["f01"]["A"]) < 0)    # f01 falls as R_n rises


def test_junction_curve_lj_axis_agrees_with_the_resistance_axis():
    LJ = np.linspace(6.0, 14.0, 9)
    a, fig = draw(sq.plots.junction_curve, LJ, 0.22, kind="LJ", style="folio")
    assert not fig.axes[0].child_axes                               # no gap, no top axis
    plt.close(fig)
    b, fig = draw(sq.plots.junction_curve, ph.rn_from_lj(LJ * 1e-9, 180e-6) / 1e3, 0.22, gap=180e-6)
    plt.close(fig)
    np.testing.assert_allclose(a["f01"][None], b["f01"][None], rtol=1e-9)


def test_junction_curve_rejects_bad_input():
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, [5.0, 6.0], 0.2)              # R_n without a gap
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, [5.0, 6.0], 0.2, kind="Ic", gap=1.8e-4)
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, [5.0, 6.0], {"A": 0.2, "B": 0.3}, gap=1.8e-4, points={"C": 5.5})
    plt.close("all")


def test_junction_curve_draws_given_curves_and_points_and_shades_an_x_band():
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    LJ = np.linspace(7.0, 15.0, 9)
    f = 5.0 * np.sqrt(10.0 / LJ)                                    # a curve the caller computed
    out, fig = draw(sq.plots.junction_curve, LJ, None, kind="LJ", f01=f, points={"design": (10.0, 5.01)},
                    xband=(9.2, 10.8), spread=0.08)
    ax = fig.axes[0]
    np.testing.assert_allclose(out["f01"][None], f)                 # drawn as given, not solved
    x0, f0, (lo, hi) = out["points"]["design"]
    assert (x0, f0) == (10.0, 5.01)                                 # the caller's frequency, not interpolated
    assert lo == pytest.approx(np.interp(10.8, LJ, f)) and hi == pytest.approx(np.interp(9.2, LJ, f))
    band = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_facecolor() == to_rgba(sq.theme().light)]
    assert len(band) == 1
    assert not violations(fig)
    plt.close(fig)
    out, fig = draw(sq.plots.junction_curve, LJ, None, kind="LJ", f01={"A": f, "B": f - 0.3}, points={"B": 12.0},
                    legend=False)
    assert out["points"]["B"][1] == pytest.approx(np.interp(12.0, LJ, f - 0.3))   # interpolated on its curve
    plt.close(fig)


def test_junction_curve_given_curves_reject_bad_input():
    LJ = np.linspace(7.0, 15.0, 9)
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, LJ, None, kind="LJ")                          # neither EC nor f01
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, LJ, None, kind="LJ", f01=LJ[:4])              # wrong length
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, LJ, {"A": 0.2}, kind="LJ", f01={"B": LJ})     # names differ
    with pytest.raises(ValueError):
        draw(sq.plots.junction_curve, LJ, None, kind="LJ", f01=LJ, cap_spread=0.05)
    plt.close("all")


def test_frequency_lines_places_every_element():
    qubits = {"Q1": (4.8, 4.58), "Q2": (5.3, 5.08)}
    res = {"R1": 6.90, "R2": 6.93, "R3": 7.30}
    out, fig = draw(sq.plots.frequency_lines, qubits, resonators=res, spread={"Q1": 0.03}, scans={"Q2": (5.3, 5.0)},
                    windows=[(4.9, 4.95)], bands={"filter": (6.8, 7.4)}, guides={"box": 7.8}, floor=4.2,
                    mass_axis=True, xlim=(4.0, 8.0))
    plt.close(fig)
    assert out["qubits"]["Q2"].get_xdata()[0] == 5.3
    tops = {k: line.get_ydata()[1] for k, line in out["resonators"].items()}
    assert tops["R2"] < tops["R1"] == tops["R3"]                    # the crowded one is staggered
    fwd, _ = out["top"]._functions
    assert fwd(5.0) == pytest.approx(ph.mass_from_frequency(5.0))


def test_frequency_lines_edge_cases():
    out, fig = draw(sq.plots.frequency_lines, {"Q": (5.0, 4.8)}, floor=1.0)
    ax = fig.axes[0]
    assert out["top"] is None and not out["resonators"]
    assert not any(p.get_hatch() for p in ax.patches)               # a floor left of the axis is skipped
    assert ax.get_xlim()[0] < 4.8 and ax.get_xlim()[1] > 5.0
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.frequency_lines, {})
    with pytest.raises(ValueError):
        draw(sq.plots.frequency_lines, {"Q": (5.0, 4.8)}, mass_axis=True, unit="MHz")
    plt.close("all")


def _gain_map():
    x, eta = np.logspace(-4, 1, 60), np.logspace(-2, 0, 50)
    return x, eta, eta[:, None] ** 2 / x[None, :]


def test_design_map_contours_points_and_arrows():
    x, eta, R = _gain_map()
    out, fig = draw(sq.plots.design_map, x, eta, R, levels=[1.0, 10.0, 100.0], emphasis=1.0,
                    points={"A": (1e-3, 0.3, "data"), "B": (1e-2, 0.6)}, arrows={"A": ["up", "left"]})
    ax = fig.axes[0]
    np.testing.assert_allclose(out["lines"].levels, [1.0, 10.0, 100.0])
    assert out["fill"] is not None and ax.get_xscale() == ax.get_yscale() == "log"
    a, b = out["points"]["A"], out["points"]["B"]
    assert a.get_markerfacecolor() == a.get_markeredgecolor()       # measured: filled in field
    assert b.get_markerfacecolor() in ("white", "w", (1.0, 1.0, 1.0, 1.0))   # designed: open
    assert sum(1 for t in ax.texts if getattr(t, "arrow_patch", None) is not None) == 2
    plt.close(fig)


def test_design_map_rejects_bad_input():
    x, eta, R = _gain_map()
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, eta, R.T, levels=[1.0, 10.0])
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, eta, R, levels=[1.0])
    out, fig = draw(sq.plots.design_map, x, eta, R, levels=[1.0], fill=False, xscale="linear")
    assert out["fill"] is None
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, eta, R, levels=[1.0, 10.0], points={"A": (1e-3, 0.3)}, arrows={"A": "sideways"})
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, eta, R, levels=[1.0, 10.0], arrows={"Z": "up"})
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, eta, R, levels=[1.0, 10.0], points={"A": (1e-3,)})
    with pytest.raises(ValueError):
        draw(sq.plots.design_map, x, eta, R, levels=[1.0, 10.0], points={"A": (1e-3, 0.3, "guess")})
    plt.close("all")


def test_dispersive_shift_default_call_is_unchanged():
    fq = np.linspace(4.0, 6.5, 50)
    chi, fig = draw(sq.plots.dispersive_shift, fq, 7.0, 0.08, -0.22)
    ax = fig.axes[0]
    assert ax.get_legend() is None and ax.get_yscale() == "symlog"
    assert len(ax.lines) == 4                                       # zero line, two pole guides, the curve
    plt.close(fig)
    np.testing.assert_allclose(chi, 1e3 * ph.dispersive_shift(fq, 7.0, 0.08, -0.22))


def test_dispersive_shift_overlays_models_and_points():
    fq = np.linspace(4.5, 6.4, 40)
    fr, g, alpha = 7.0, 0.08, -0.22
    exact = ph.transmon_dressed(fq, fr, g, 0.25)["chi"]
    sweep = (np.array([5.0, 5.5, 6.0]), np.array([-0.4e-3, -0.7e-3, -1.4e-3]))
    models = {"exact": exact, "scaled": lambda f: 1.2 * ph.dispersive_shift(f, fr, g, alpha), "sweep": sweep}
    chi, fig = draw(sq.plots.dispersive_shift, fq, fr, g, alpha, models=models, scale="log", detuning=True,
                    points={"measured": (5.2, -0.5e-3, "data")}, style="folio")
    ax = fig.axes[0]
    assert ax.get_yscale() == "log" and ax.get_legend() is not None
    lines = {ln.get_label(): ln for ln in ax.lines}
    np.testing.assert_allclose(lines["exact"].get_ydata(), np.abs(1e3 * exact))
    np.testing.assert_allclose(lines["exact"].get_xdata(), fq - fr)
    np.testing.assert_allclose(lines["scaled"].get_ydata(), 1.2 * np.abs(chi))
    np.testing.assert_allclose(lines["sweep"].get_xdata(), sweep[0] - fr)
    plt.close(fig)


def test_dispersive_shift_rejects_bad_models_and_scales():
    fq = np.linspace(4.5, 6.4, 40)
    with pytest.raises(ValueError):
        draw(sq.plots.dispersive_shift, fq, 7.0, 0.08, -0.22, models={"short": np.zeros(3)})
    with pytest.raises(ValueError):
        draw(sq.plots.dispersive_shift, fq, 7.0, 0.08, -0.22, scale="loglog")
    plt.close("all")


def test_convergence_error_mode_measures_against_the_last_pass():
    v = 5.0 + 0.4 * np.exp(-np.arange(12) * 1.5)
    out, fig = draw(sq.plots.convergence, np.arange(1, 13), v, error=True, tol=1e-4, floor=1e-6,
                    floor_label="1 kHz", label="error (GHz)")
    ax = fig.axes[0]
    assert ax.get_yscale() == "log" and out["ax2"] is None and out["rel"] is None
    assert np.isnan(ax.lines[0].get_ydata()[-1])                    # the reference pass has no error
    plt.close(fig)
    np.testing.assert_allclose(out["err"], np.abs(v - v[-1]))
    assert out["converged"] == 1 + np.nonzero(np.abs(v - v[-1]) <= 1e-4)[0][0]


def test_convergence_error_mode_with_a_reference_and_edge_cases():
    n, v = np.arange(1, 6), np.array([7.4, 7.2, 7.16, 7.152, 7.1512])
    out, fig = draw(sq.plots.convergence, n, v, error=True, reference=7.1511, style="folio")
    plt.close(fig)
    np.testing.assert_allclose(out["err"], np.abs(v - 7.1511))
    assert out["converged"] is None
    val, fig = draw(sq.plots.convergence, n, v)                     # value mode reports no error
    plt.close(fig)
    assert val["err"] is None and val["ax2"] is not None
    with pytest.raises(ValueError):
        draw(sq.plots.convergence, n, np.full(5, 3.0), error=True)  # every error zero
    with pytest.raises(ValueError):
        draw(sq.plots.convergence, n, v[:3])
    plt.close("all")


# ---- dark matter --------------------------------------------------------------------------------

def test_reach_envelope_limits_and_mass_axis():
    f = np.logspace(0, 1, 50)
    curves = {"a": 1e-12 * f ** -0.5, "b": 2e-13 * f ** 0.3}
    out, fig = draw(sq.plots.reach, f, curves, limits={"run": (5.0, 3e-12)}, excluded="above")
    ax = fig.axes[0]
    assert ax.get_xscale() == ax.get_yscale() == "log" and ax.get_legend() is not None
    np.testing.assert_allclose(out["envelope"], np.minimum(curves["a"], curves["b"]))
    fwd, inv = out["top"]._functions
    assert fwd(1.0) == pytest.approx(4.135667696) and inv(fwd(3.0)) == pytest.approx(3.0)
    assert any(ln.get_marker() == "*" for ln in ax.lines)
    plt.close(fig)


def test_reach_single_curve_below_and_bad_input():
    f = np.linspace(1.0, 8.0, 30)
    e = 1e-13 * f
    out, fig = draw(sq.plots.reach, f, e, excluded="below", mass_axis=False, style="folio")
    assert out["top"] is None and list(out["lines"]) == [None]
    np.testing.assert_allclose(out["envelope"], e)
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.reach, f, e, excluded="left")
    with pytest.raises(ValueError):
        draw(sq.plots.reach, f, e[:5])
    with pytest.raises(ValueError):
        draw(sq.plots.reach, [1.0], [1e-12])
    plt.close("all")


# ---- options for figures that pass their own curves and styles ----------------------------------

def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_points_style_sets_marker_colour_and_label():
    grey, blue = sq.PALETTE["grey"], sq.PALETTE["blue"]
    pts = {"design": (5.0, -4e-4, "sim", dict(marker="o", color=sq.PALETTE["ink"], label=None)),
           "expected": (5.2, -5e-4, "sim", dict(marker="s", color=grey, offset=(4, -6), va="top")),
           "measured": (5.4, -6e-4, "data", dict(marker="s", color=grey)),
           "elsewhere": (5.6, -7e-4, "assumed", dict(color=blue)),       # assumed stays grey
           "plain": (5.8, -8e-4)}
    _, fig = draw(sq.plots.dispersive_shift, np.linspace(4.5, 6.4, 30), 7.0, 0.08, -0.22, points=pts)
    ax = fig.axes[0]
    texts = {t.get_text(): t for t in ax.texts}
    assert "design" not in texts and texts["expected"].get_va() == "top" and texts["expected"].xyann == (4, -6)
    assert {"measured", "elsewhere", "plain"} <= set(texts)
    marks = {ln.get_xdata()[0]: ln for ln in ax.lines if len(ln.get_xdata()) == 1}
    assert marks[5.2].get_marker() == marks[5.4].get_marker() == "s"
    assert marks[5.2].get_markeredgecolor() == grey and marks[5.2].get_markerfacecolor() == "white"
    assert marks[5.4].get_markerfacecolor() == grey                      # measured: filled in field
    assert marks[5.6].get_markeredgecolor() == sq.theme().assumed
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.dispersive_shift, np.linspace(4.5, 6.4, 30), 7.0, 0.08, -0.22,
             points={"bad": (5.0, -4e-4, "sim", {}, "extra")})
    plt.close("all")


def test_dispersive_shift_without_formula_and_with_styled_models():
    fq = np.linspace(4.6, 6.5, 40)
    fr, g, alpha = 7.0, 0.08, -0.2
    full = 1.1 * ph.dispersive_shift(fq, fr, g, alpha)
    models = {"full": (fq, full, dict(color=sq.PALETTE["ink"], ls="-")),
              "ladder": (fq, ph.dispersive_shift(fq, fr, g, alpha), dict(color=sq.PALETTE["orange"], ls="--", lw=0.8))}
    _, fig = draw(sq.plots.dispersive_shift, fq, fr, g, alpha, scale="log", detuning=True, formula=False,
                  models=models, legend_loc=None)
    ax = fig.axes[0]
    curves = {ln.get_label(): ln for ln in ax.lines if len(ln.get_xdata()) > 2}
    assert set(curves) == {"full", "ladder"}                         # no formula curve
    assert curves["ladder"].get_color() == sq.PALETTE["orange"] and curves["ladder"].get_linewidth() == 0.8
    np.testing.assert_allclose(curves["full"].get_ydata(), np.abs(1e3 * full))
    assert ax.get_legend() is None
    assert not any("Delta" in s or "straddling" in s for s in _texts(ax))   # regime and poles off the range
    plt.close(fig)
    _, fig = draw(sq.plots.dispersive_shift, np.linspace(6.0, 7.5, 40), fr, g, alpha)
    assert {r"$\Delta = 0$ ", r" $\Delta = -\alpha$", "straddling"} <= set(_texts(fig.axes[0]))
    plt.close(fig)


def test_flux_tuning_one_colour_and_its_sweet_label():
    c = sq.PALETTE["magenta"]
    out, fig = draw(sq.plots.flux_tuning, 22.0, 0.24, d=(0.0, 0.2), color=c, legend=False,
                    sweet_label="zero-flux point")
    ax = fig.axes[0]
    curves = [ln for ln in ax.lines if len(ln.get_xdata()) > 2]
    assert [ln.get_color() for ln in curves] == [c, c] and curves[0].get_linestyle() == "-"
    assert curves[1].get_linestyle() != "-" and ax.get_legend() is None
    assert "zero-flux point" in _texts(ax)
    sweet = [ln for ln in ax.lines if len(ln.get_xdata()) == 1][0]
    assert sweet.get_markeredgecolor() == c
    plt.close(fig)
    _, fig = draw(sq.plots.flux_tuning, 22.0, 0.24, sweet_label=None)
    assert not fig.axes[0].texts
    plt.close(fig)


def test_dressed_lines_given_branches_and_a_second_coupling():
    fq = np.linspace(-400.0, 400.0, 81)                              # MHz from f_r
    up, lo = ph.dressed_frequencies(fq, 0.0, 115.0)
    out, fig = draw(sq.plots.dressed_lines, fq, 0.0, 115.0, branches=(lo, up), color=sq.PALETTE["grey"],
                    split_label=r"$2g={:.0f}$ MHz", fr_label=r"$f_r$ (bare)")
    ax = fig.axes[0]
    np.testing.assert_allclose(out["lower"], lo)
    assert out["split"] == pytest.approx(230.0) and out["at"] == pytest.approx(0.0)
    assert r"$f_r$ (bare)" in _texts(ax) and r"$2g=230000$ MHz" in _texts(ax)   # gap printed x1e3, as for GHz
    n = len(ax.lines)
    up2, lo2 = ph.dressed_frequencies(fq, 0.0, 80.0)
    sq.plots.dressed_lines(ax, fq, 0.0, 80.0, branches=(lo2, up2), color=sq.PALETTE["ink"], bare=False, split=False,
                           fr_label=None, points={"m": (0.0, -80.0, "data", dict(label=None))})
    assert len(ax.lines) == n + 3                                    # two branches and one point, no bare lines
    assert ax.lines[n].get_color() == sq.PALETTE["ink"]
    plt.close(fig)
    _, fig = draw(sq.plots.dressed_lines, fq, 0.0, 115.0, fr_label=None, split=False)
    assert not fig.axes[0].texts
    plt.close(fig)
    with pytest.raises(ValueError):
        draw(sq.plots.dressed_lines, fq, 0.0, 115.0, branches=(lo[:5], up[:5]))
    plt.close("all")


def test_junction_curve_legend_off_and_frequency_lines_one_colour():
    _, fig = draw(sq.plots.junction_curve, np.linspace(5, 12, 10), {"A": 0.2, "B": 0.25}, gap=1.8e-4, legend=False)
    assert fig.axes[0].get_legend() is None
    plt.close(fig)
    pad = sq.PALETTE["magenta"]
    out, fig = draw(sq.plots.frequency_lines, {"Q1": (4.8, 4.58), "Q2": (5.3, 5.08)}, color=pad)
    assert {ln.get_color() for ln in out["qubits"].values()} == {pad}
    plt.close(fig)


def test_purcell_limit_on_detuning_with_a_mark_label():
    fq = np.linspace(4.6, 7.2, 60)
    out, fig = draw(sq.plots.purcell_limit, fq, 7.6, 0.115, 7.7e-4, detuning=True, loc=None, mark=5.7,
                    mark_label=r"$T_1^\mathrm{{P}}={:.0f}\ \mu$s")
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), fq - 7.6)
    np.testing.assert_allclose(out["bare"], ph.purcell_t1(fq, 7.6, 0.115, 7.7e-4))
    T0 = float(ph.purcell_t1(5.7, 7.6, 0.115, 7.7e-4))
    assert rf"$T_1^\mathrm{{P}}={T0:.0f}\ \mu$s" in _texts(ax) and ax.get_legend() is None
    assert ax.get_xlim() == pytest.approx((fq.min() - 7.6, fq.max() - 7.6))
    plt.close(fig)


# ---- both styles keep the colour rule -----------------------------------------------------------

CASES = [
    (sq.plots.ratio_scaling, ("dispersion",), dict(points={"A": 40.0}, bands={"sensors": (15, 30)})),
    (sq.plots.parity_bands, (12.0, 0.6), {}),
    (sq.plots.flux_tuning, (22.0, 0.24), dict(d=(0.0, 0.2, 0.4), detune=1.0)),
    (sq.plots.flux_tuning, (22.0, 0.24), dict(d=(0.0, 0.2), color=sq.PALETTE["magenta"], sweet_label="zero flux")),
    (sq.plots.purcell_limit, (np.linspace(4.6, 7.2, 30), 7.6, 0.115, 7.7e-4),
     dict(detuning=True, mark=5.7, mark_label="{:.0f} µs", loc=None)),
    (sq.plots.dispersive_shift, (np.linspace(4.5, 6.4, 30), 7.0, 0.08, -0.22),
     dict(formula=False, scale="log", models={"m": (np.linspace(4.5, 6.4, 30), np.full(30, -1e-3),
                                                    dict(color=sq.PALETTE["ink"], ls="-"))},
          points={"p": (5.0, -4e-4, "data", dict(marker="s", color=sq.PALETTE["grey"], offset=(4, -6)))})),
    (sq.plots.dressed_lines, (np.linspace(6, 8, 41), 7.0, 0.08), dict(EC=0.25, mark=6.5)),
    (sq.plots.junction_curve, (np.linspace(5, 12, 20), {"A": 0.2, "B": 0.25}),
     dict(gap=1.8e-4, points={"A": 7.0, "B": 9.0}, spread=0.03, cap_spread=0.05, window=(8, 14), band=(4, 6))),
    (sq.plots.frequency_lines, ({"Q1": (4.8, 4.58), "Q2": (5.3, 5.08)},),
     dict(resonators={"R": 7.0}, spread=0.03, scans={"Q2": (5.3, 5.0)}, windows=[(4.9, 4.95)],
          bands={"filter": (6.8, 7.2)}, guides={"box": 7.6}, floor=4.2, mass_axis=True, xlim=(4.0, 8.0))),
    (sq.plots.design_map, _gain_map(), dict(levels=[1.0, 10.0, 100.0], points={"A": (1e-3, 0.3, "assumed")},
                                            arrows={"A": "up"})),
    (sq.plots.dispersive_shift, (np.linspace(4.5, 6.4, 30), 7.0, 0.08, -0.22),
     dict(models={"x": lambda f: 0.9 * ph.dispersive_shift(f, 7.0, 0.08, -0.22)}, points={"p": (5.0, -4e-4)},
          mark=5.5)),
    (sq.plots.convergence, (np.arange(1, 9), 5 + np.exp(-np.arange(8.0))), dict(error=True, tol=1e-2, floor=1e-3)),
    (sq.plots.reach, (np.logspace(0, 1, 20), {"a": np.logspace(-12, -13, 20), "b": np.logspace(-12.5, -12.8, 20)}),
     dict(excluded="above", limits={"run": (4.0, 2e-12)})),
]


@pytest.mark.parametrize("style", ["field", "folio"])
@pytest.mark.parametrize("fn,args,kw", CASES, ids=[c[0].__name__ for c in CASES])
def test_new_plots_keep_the_colour_rule(fn, args, kw, style):
    _, fig = draw(fn, *args, style=style, **kw)
    bad = violations(fig)
    plt.close(fig)
    assert not bad, [(type(a).__name__, c) for a, c in bad[:8]]
