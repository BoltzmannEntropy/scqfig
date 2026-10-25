"""Comparison and budget charts: return values, edge cases, both styles and argument checks."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.check import violations  # noqa: E402
from scqfig.plots import compare as C  # noqa: E402


@pytest.fixture(params=["field", "folio"])
def style(request):
    with sq.style(request.param) as t:
        yield t
    plt.close("all")


def axes(ncols=1, **kw):
    return sq.figure("page", aspect=0.4, ncols=ncols, **kw)


def clean(fig):
    """Draw the figure and check every colour against the palette rule."""
    fig.canvas.draw()
    bad = violations(fig)
    assert not bad, [(type(a).__name__, c) for a, c in bad[:6]]


def widths(container):
    return np.array([r.get_width() for r in container])


def test_every_function_is_exported():
    for name in C.__all__:
        assert getattr(sq.plots, name) is getattr(C, name)
    assert len(C.__all__) == 11


# ---- stacked_bars -----------------------------------------------------------------------------

def test_stacked_totals_shares_and_columns(style):
    fig, ax = axes()
    seg = {"a": [1.0, 2.0, 0.0], "b": [3.0, 2.0, 0.0]}
    r = sq.plots.stacked_bars(ax, ["x", "y", "z"], seg, totals="total", columns={"note": ["p", "q", "r"]})
    np.testing.assert_allclose(r["totals"], [4.0, 4.0, 0.0])
    np.testing.assert_allclose(widths(r["bars"][1]), [3.0, 2.0, 0.0])
    np.testing.assert_allclose([b.get_x() for b in r["bars"][1]], [1.0, 2.0, 0.0])
    assert [t.get_text() for t in r["texts"][0][:3]] == ["4", "4", "0"]
    assert r["texts"][1][-1].get_text() == "note"
    assert not r["bars"][0][2].get_visible()                 # an empty segment draws no stray edge
    clean(fig)
    tot = r["texts"][0][0].get_window_extent()
    note = r["texts"][1][0].get_window_extent()
    assert note.x0 > tot.x1                                  # the second column starts right of the first


def test_stacked_normalize_gives_shares_and_empty_row_stays_empty(style):
    fig, ax = axes()
    r = sq.plots.stacked_bars(ax, ["x", "y"], {"a": [1.0, 0.0], "b": [3.0, 0.0]}, normalize=True)
    share = widths(r["bars"][0]) + widths(r["bars"][1])
    np.testing.assert_allclose(share, [1.0, 0.0])
    clean(fig)


def test_stacked_log_multiplies_and_skips_zero_factors(style):
    fig, ax = axes()
    r = sq.plots.stacked_bars(ax, ["x", "y"], {"a": [0.5, 0.0], "b": [0.8, 2.0]}, log=True, totals=True,
                              markers={"measured": [0.38, np.nan]})
    np.testing.assert_allclose(r["totals"], [0.4, 2.0])       # the zero factor is left out
    first = r["bars"][0][0]
    np.testing.assert_allclose(sorted([first.get_x(), first.get_x() + first.get_width()]), [0.5, 1.0])
    assert ax.get_xscale() == "log"
    assert len(r["markers"]) == 1
    clean(fig)


def test_stacked_negative_values_stack_below_the_base(style):
    fig, ax = axes()
    r = sq.plots.stacked_bars(ax, ["x"], {"up": [2.0], "down": [-1.0], "more down": [-0.5]})
    b = r["bars"][2][0]
    np.testing.assert_allclose(sorted([b.get_x(), b.get_x() + b.get_width()]), [-1.5, -1.0])
    clean(fig)


def test_stacked_vertical_groups_label_each_run(style):
    fig, ax = axes()
    r = sq.plots.stacked_bars(ax, list("abcde"), {"s": [1, 2, 3, 4, 5]}, horizontal=False, totals=True,
                              groups=["G1", "G1", "G2", "G2", "G2"], legend=False)
    assert [h.get_text() for h in r["headers"]] == ["G1", "G2"]
    assert r["pos"][2] - r["pos"][1] > 1.0                    # a gap between the groups
    assert len(r["texts"]) == 5
    clean(fig)


def test_stacked_empty_and_single_row(style):
    fig, (a, b) = axes(2)
    r = sq.plots.stacked_bars(a, [], {"a": [], "b": []}, totals=True)
    assert r["totals"].size == 0
    r = sq.plots.stacked_bars(b, ["only"], {"a": [np.nan], "b": [2.0]})
    np.testing.assert_allclose(r["totals"], [2.0])
    clean(fig)


def test_stacked_rejects_bad_arguments():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.stacked_bars(ax, ["x", "y"], {"a": [1.0]})
    with pytest.raises(ValueError):
        sq.plots.stacked_bars(ax, ["x"], {"a": [1.0]}, hatches=["", "//"])
    with pytest.raises(ValueError):
        sq.plots.stacked_bars(ax, ["x"], {"a": [1.0]}, log=True, normalize=True)
    with pytest.raises(ValueError):
        sq.plots.stacked_bars(ax, ["x"], {"a": [1.0]}, markers={"m": [1.0, 2.0]})
    with pytest.raises(ValueError):
        sq.plots.stacked_bars(ax, ["x"], {"a": [1.0]}, columns={"c": ["p", "q"]})
    plt.close(fig)


# ---- factor_bars ------------------------------------------------------------------------------

def test_factor_bars_net_is_log_of_product(style):
    fig, ax = axes()
    r = sq.plots.factor_bars(ax, ["a", "b", "c"], {"one": [10.0, 0.01, 1.0], "two": [100.0, 0.1, -3.0]})
    assert r["net"]["one"] == pytest.approx(-1.0)
    assert r["net"]["two"] == pytest.approx(1.0)              # the negative factor is left out
    np.testing.assert_allclose(widths(r["bars"][0]), [1.0, -2.0, 0.0, -1.0])
    assert not r["bars"][1][2].get_visible()
    assert any("10^{-2.0}" in t.get_text() for t in r["texts"])
    assert r["legend"] is not None
    clean(fig)


def test_factor_bars_single_series_colours_by_sign(style):
    fig, ax = axes()
    t = sq.theme()
    r = sq.plots.factor_bars(ax, ["gain", "loss"], [4.0, 0.5], net=False, horizontal=False)
    up, down = r["bars"][0]
    assert up.get_height() > 0 > down.get_height()
    assert up.get_edgecolor() != down.get_edgecolor()
    assert matplotlib.colors.same_color(up.get_edgecolor(), t.sim)
    assert r["net"][None] == pytest.approx(np.log10(2.0))
    clean(fig)


def test_factor_bars_linear_values_and_empty(style):
    fig, (a, b) = axes(2)
    r = sq.plots.factor_bars(a, ["p", "q"], [1.5, -2.5], log10=False)
    np.testing.assert_allclose(widths(r["bars"][0]), [1.5, -2.5, -1.0])
    r = sq.plots.factor_bars(b, [], [], net=False)
    assert len(r["bars"][0]) == 0
    clean(fig)


def test_factor_bars_rejects_mismatch():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.factor_bars(ax, ["a", "b"], [1.0])
    with pytest.raises(ValueError):
        sq.plots.factor_bars(ax, ["a"], {"s": [1.0], "t": [2.0]}, hatches=[""])
    plt.close(fig)


# ---- grouped_bars -----------------------------------------------------------------------------

def test_grouped_bars_layout_and_ratio_labels(style):
    fig, ax = axes()
    r = sq.plots.grouped_bars(ax, ["g1", "g2"], {"A": [1e-5, 2e-4], "B": [5e-5, 0.0]}, log=True, ratios=True)
    assert len(r["bars"]) == 2 and all(len(b) == 2 for b in r["bars"])
    a, b = r["bars"][0][0], r["bars"][1][0]
    assert a.get_x() + a.get_width() == pytest.approx(b.get_x())       # bars of a group touch
    assert [t.get_text() for t in r["texts"]] == ["×5"]                 # no ratio over a zero
    assert not r["bars"][1][1].get_visible()
    assert ax.get_yscale() == "log" and ax.get_ylim()[0] > 0
    clean(fig)


def test_grouped_bars_negative_values_one_group_and_horizontal(style):
    fig, (a, b) = axes(2)
    r = sq.plots.grouped_bars(a, ["only"], {"A": [-2.0], "B": [3.0], "C": [np.nan]}, values=True)
    assert len(r["texts"]) == 2
    r = sq.plots.grouped_bars(b, ["g1", "g2"], {"A": [1.0, 2.0]}, horizontal=True)
    np.testing.assert_allclose(widths(r["bars"][0]), [1.0, 2.0])
    clean(fig)


def test_grouped_bars_rejects_mismatch():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.grouped_bars(ax, ["g1", "g2"], {"A": [1.0, 2.0], "B": [1.0]})
    plt.close(fig)


def test_grouped_bars_colors_fill_each_series_as_given(style):
    from matplotlib.colors import to_rgba
    fig, ax = axes()
    r = sq.plots.grouped_bars(ax, ["R0", "R1"], {"formula": [8.1, 7.9], "solver": [7.6, 7.45]},
                              colors=["light", style.sim])
    assert all(p.get_facecolor() == to_rgba(style.light) for p in r["bars"][0])
    assert all(p.get_facecolor() == to_rgba(style.sim) for p in r["bars"][1])
    assert all(p.get_edgecolor() == to_rgba(style.ink) for p in r["bars"][1])
    handles = r["legend"].get_patches()
    assert handles[0].get_facecolor() == to_rgba(style.light) and handles[1].get_facecolor() == to_rgba(style.sim)
    clean(fig)
    plain = sq.plots.grouped_bars(axes()[1], ["R0"], {"a": [1.0], "b": [2.0]})       # default: series tints
    assert plain["bars"][0][0].get_facecolor() == to_rgba(C._fill(0)["color"])


@pytest.mark.parametrize("colors", [["light"], ["light", "no-such-role"]])
def test_grouped_bars_rejects_bad_colors(colors):
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.grouped_bars(ax, ["g1"], {"A": [1.0], "B": [2.0]}, colors=colors)
    plt.close(fig)


# ---- tornado ----------------------------------------------------------------------------------

def test_tornado_sorts_by_swing_and_reorders_columns(style):
    fig, ax = axes()
    r = sq.plots.tornado(ax, ["small", "large", "mid"], [0.9, 0.1, 0.5], [1.1, 10.0, 1.0],
                         columns={"range": ["s", "l", "m"]}, base_label="base 1")
    assert list(r["order"]) == [1, 2, 0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["large", "mid", "small"]
    assert [t.get_text() for t in r["columns"][0][:3]] == ["l", "m", "s"]
    texts = [t.get_text() for t in r["texts"]]
    assert "1" not in texts                                    # an outcome at the base gets no label
    low = r["bars"][0][0]
    assert low.get_x() + low.get_width() == pytest.approx(0.1)
    clean(fig)


def test_tornado_linear_unsorted_nan_and_empty(style):
    fig, (a, b) = axes(2)
    r = sq.plots.tornado(a, ["p", "q"], [-1.0, np.nan], [2.0, 0.5], base=0.0, log=False, sort=False)
    assert list(r["order"]) == [0, 1]
    assert not r["bars"][0][1].get_visible()
    r = sq.plots.tornado(b, [], [], [])
    assert len(r["order"]) == 0
    clean(fig)


def test_tornado_rejects_mismatch():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.tornado(ax, ["a", "b"], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        sq.plots.tornado(ax, ["a"], [1.0], [2.0], columns={"c": ["x", "y"]})
    plt.close(fig)


# ---- bar_rows ---------------------------------------------------------------------------------

def test_bar_rows_groups_status_marks_and_columns(style):
    fig, ax = axes()
    r = sq.plots.bar_rows(ax, ["a", "b", "c"], [1e-6, 3e-4, 0.0], status=["upper", "simulated", "measured"],
                          groups=["S1", "S1", "S2"], log=True, marks=[np.nan, 1e-3, 2e-4], ref=1e-4,
                          columns={"": ["< 1e-6", "3e-4", "0"]})
    np.testing.assert_allclose(r["pos"], [1.0, 2.0, 4.0])
    assert [h.get_text() for h in r["headers"]] == ["S1", "S2"]
    assert r["bars"][0].get_hatch() == "////"
    assert not r["bars"][2].get_visible()                      # zero on a log axis draws no bar
    lo = ax.get_xlim()[0]
    assert 0 < lo <= 1e-6
    assert len(r["columns"][0]) == 3
    clean(fig)


def test_bar_rows_base_one_and_single_row(style):
    fig, (a, b) = axes(2)
    r = sq.plots.bar_rows(a, ["x", "y"], [40.0, 0.5], base=1.0, log=True)
    np.testing.assert_allclose([p.get_x() for p in r["bars"]], [1.0, 1.0])
    np.testing.assert_allclose(widths(r["bars"]), [39.0, -0.5])
    r = sq.plots.bar_rows(b, ["only"], [2.0])
    assert len(r["bars"]) == 1
    clean(fig)


def test_bar_rows_empty_and_bad_arguments(style):
    fig, ax = axes()
    r = sq.plots.bar_rows(ax, [], [])
    assert len(r["pos"]) == 0
    clean(fig)
    with pytest.raises(ValueError):
        sq.plots.bar_rows(ax, ["a", "b"], [1.0])
    with pytest.raises(ValueError):
        sq.plots.bar_rows(ax, ["a"], [1.0], status=["guessed"])
    with pytest.raises(ValueError):
        sq.plots.bar_rows(ax, ["a"], [1.0], groups=["g", "h"])


# ---- dot_rows ---------------------------------------------------------------------------------

def test_dot_rows_points_ranges_bounds_and_connectors(style):
    fig, ax = axes()
    r = sq.plots.dot_rows(ax, ["a", "b", "c", "d"], [[0.5, 0.6], np.nan, 2.0, 0.0],
                          lo=[np.nan, 0.2, np.nan, np.nan], hi=[np.nan, 0.9, np.nan, np.nan],
                          status=[["simulated", "measured"], "measured", "upper", "measured"],
                          connect=True, ref=1.0, band=(0.8, 1.25), log=True, columns={"value": list("pqrs")})
    assert len(r["points"]) == 3                               # NaN and zero on a log axis are skipped
    assert len(r["ranges"]) == 1 and len(r["arrows"]) == 1
    fill = [p.get_markerfacecolor() for p in r["points"][:2]]
    assert fill[0] != fill[1] or sq.theme().name == "folio"
    clean(fig)
    tail = r["arrows"][0].xy
    head = r["arrows"][0].xyann
    assert tail == (2.0, 2.0) and head[0] < 0                  # an upper bound points to smaller values


def test_dot_rows_groups_and_empty(style):
    fig, (a, b) = axes(2)
    r = sq.plots.dot_rows(a, ["a", "b"], [1.0, 2.0], groups=["G", "H"])
    np.testing.assert_allclose(r["pos"], [1.0, 3.0])
    r = sq.plots.dot_rows(b, [], [])
    assert r["points"] == []
    clean(fig)


def test_dot_rows_rejects_mismatch():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.dot_rows(ax, ["a", "b"], [1.0])
    with pytest.raises(ValueError):
        sq.plots.dot_rows(ax, ["a"], [[1.0, 2.0]], status=[["measured"]])
    with pytest.raises(ValueError):
        sq.plots.dot_rows(ax, ["a"], [1.0], lo=[0.5, 0.6])
    plt.close(fig)


# ---- study_strips -----------------------------------------------------------------------------

def test_study_strips_rows_missing_and_limits(style):
    fig, axs = axes(3, sharey=True)
    cols = {"q1": {"A": 2.0, "B": [(5.0, "simulated"), 3.0]},
            "q2": {"C": 0.0},
            "q3": {"A": (1.0, "measured"), "C": np.nan}}
    r = sq.plots.study_strips(axs, ["A", "B", "C"], cols, lims={"q3": (0.1, 10.0)})
    assert len(r["points"]["q1"]) == 3
    assert r["points"]["q2"] == []                             # zero on a log strip is skipped
    assert [m.get_text() for m in r["missing"]] == ["n.r."] * 4
    assert axs[2].get_xlim() == pytest.approx((0.1, 10.0))
    assert [t.get_text() for t in axs[0].get_yticklabels()] == ["A", "B", "C"]
    clean(fig)


def test_study_strips_linear_and_empty(style):
    fig, axs = axes(2)
    r = sq.plots.study_strips(axs, [], {"a": {}, "b": {}}, log=False)
    assert r["missing"] == []
    clean(fig)


def test_study_strips_rejects_bad_arguments():
    fig, axs = axes(2)
    with pytest.raises(ValueError):
        sq.plots.study_strips(axs, ["A"], {"q": {"A": 1.0}})
    with pytest.raises(ValueError):
        sq.plots.study_strips(axs, ["A"], {"q": {"Z": 1.0}, "r": {}})
    plt.close(fig)


# ---- ratio_strip ------------------------------------------------------------------------------

def test_ratio_strip_is_seeded_and_places_excluded_points_left(style):
    fig, (a, b) = axes(2)
    groups = {"f": [1.01, 0.98, 1.02], "g": [1.2, 0.0, 3.0]}
    status = {"f": ["measured", "fitted", "excluded"], "g": ["simulated", "measured", "lower"]}
    r1 = sq.plots.ratio_strip(a, groups, status=status, band=0.05, rms={"g": 0.1}, legend={"lower": "bound"})
    r2 = sq.plots.ratio_strip(b, groups, status=status)
    np.testing.assert_allclose(r1["pos"]["f"], r2["pos"]["f"])
    assert np.all(np.abs(r1["pos"]["f"][:2] - 0) <= 0.22)
    assert r1["pos"]["f"][2] < -0.22
    assert len(r1["points"]["g"]) == 2                          # the zero ratio is skipped
    labels = [t.get_text() for t in r1["legend"].get_texts()]
    assert "bound" in labels and "fitted" in labels
    clean(fig)


def test_ratio_strip_horizontal_linear_and_empty(style):
    fig, (a, b) = axes(2)
    r = sq.plots.ratio_strip(a, {"x": [0.5, 2.0], "y": [np.nan]}, log=False, horizontal=True, band=(0.9, 1.1))
    assert len(r["points"]["y"]) == 0
    r = sq.plots.ratio_strip(b, {})
    assert r["legend"] is None
    clean(fig)


def test_ratio_strip_rejects_mismatch():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.ratio_strip(ax, {"f": [1.0, 2.0]}, status={"f": ["measured"]})
    with pytest.raises(ValueError):
        sq.plots.ratio_strip(ax, {"f": [1.0]}, status={"g": ["measured"]})
    plt.close(fig)


# ---- paired_stems -----------------------------------------------------------------------------

def test_paired_stems_nan_groups_and_log(style):
    fig, (a, b) = axes(2)
    r = sq.plots.paired_stems(a, ["d1", "d2", "d3", "d4"], [30.0, np.nan, 80.0, 0.0], [20.0, 15.0, 90.0, 5.0],
                              groups=["c1", "c1", "c2", "c2"], log=True)
    assert len(r["stems"].get_segments()) == 2                   # only items with both values get a stem
    assert len(r["open"].get_xdata()) == 2 and len(r["filled"].get_xdata()) == 4
    assert r["x"][2] - r["x"][1] > 1.0
    r = sq.plots.paired_stems(b, ["p", "q"], [100.0, 4.0], [0.0, 1.0], log="symlog")
    assert len(r["filled"].get_xdata()) == 2                     # a count of zero shows on symlog
    assert b.get_ylim()[0] < 0
    clean(fig)


def test_paired_stems_empty_and_mismatch(style):
    fig, ax = axes()
    r = sq.plots.paired_stems(ax, [], [], [])
    assert len(r["x"]) == 0
    clean(fig)
    with pytest.raises(ValueError):
        sq.plots.paired_stems(ax, ["a", "b"], [1.0, 2.0], [1.0])


# ---- timeline ---------------------------------------------------------------------------------

def test_timeline_compresses_early_years_and_spreads_shared_cells(style):
    fig, ax = axes()
    ev = [("R1", 2008, "k1"), ("R1", 2020, "k1"), ("R1", 2020, "k2"), ("R2", 2010, "k2"), ("R2", 2010, "k2")]
    r = sq.plots.timeline(ax, ["R1", "R2"], ev, break_at=2018)
    xm = r["xmap"]
    assert xm(2010) - xm(2008) == pytest.approx(2 * 0.36)
    assert xm(2021) - xm(2020) == pytest.approx(1.0)
    k1 = r["points"]["k1"].get_xdata()
    k2x, k2y = r["points"]["k2"].get_xdata(), r["points"]["k2"].get_ydata()
    assert k1[1] != k2x[0]                                         # same row and year sit side by side
    assert k2x[1] == k2x[2] and k2y[1] != k2y[2]                   # compressed years stack vertically
    assert [t.get_text() for t in r["legend"].get_texts()] == ["k1", "k2"]
    clean(fig)


def test_timeline_plain_axis_kinds_and_empty(style):
    fig, (a, b) = axes(2)
    r = sq.plots.timeline(a, ["R"], [("R", 1.5), ("R", 3.0)], legend=False)
    assert r["xmap"](3.0) == 3.0 and r["legend"] is None
    r = sq.plots.timeline(b, [], [])
    assert r["points"] == {}
    clean(fig)


def test_timeline_rejects_unknown_rows_and_kinds():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.timeline(ax, ["R"], [("S", 2020)])
    with pytest.raises(ValueError):
        sq.plots.timeline(ax, ["R"], [("R", 2020, "k")], kinds=["other"])
    with pytest.raises(ValueError):
        sq.plots.timeline(ax, ["R"], [("R",)])
    plt.close(fig)


# ---- limit_arrow ------------------------------------------------------------------------------

@pytest.mark.parametrize("direction, sign", [("up", (0, 1)), ("down", (0, -1)), ("left", (-1, 0)), ("right", (1, 0))])
def test_limit_arrow_points_the_named_way_on_log_axes(direction, sign):
    with sq.style("field"):
        fig, ax = axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        m, a = sq.plots.limit_arrow(ax, 10.0, 0.5, direction)
        assert m.get_markerfacecolor() in ("white", (1.0, 1.0, 1.0, 1.0))   # a bound is open by default
        dx, dy = a.xyann
        assert np.sign(dx) == sign[0] and np.sign(dy) == sign[1]
        clean(fig)
        plt.close(fig)


def test_limit_arrow_rejects_unknown_direction():
    fig, ax = axes()
    with pytest.raises(ValueError):
        sq.plots.limit_arrow(ax, 1.0, 1.0, "sideways")
    plt.close(fig)
