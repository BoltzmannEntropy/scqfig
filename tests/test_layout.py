"""Text frames and flow diagrams: recipe grids, equation panels, bracketed matrices, Hamiltonian terms on
tiles and flow diagrams, with their layouts, text fitting, failure cases and the colour rule."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.check import violations  # noqa: E402
from scqfig.schematics import flow_diagram, layout, matrix_text, recipe_grid, term_tiles, text_panel  # noqa: E402

STYLES = ["field", "folio"]


@pytest.fixture(autouse=True)
def _field_style():
    with sq.style("field"):
        yield
    plt.close("all")


def _line(ax):
    ax.plot([0, 1], [0, 1], **sq.model_kw())
    ax.set_xlabel("x")


def _steps(n, **kw):
    return [dict(dict(title=f"Step {i + 1}", lines=["Do this.", "Then that."], panel=_line, tools=["pen"]), **kw)
            for i in range(n)]


def _inside(fig, texts, rect_inch, tol=0.02):
    """True if every text's extent lies inside rect (x, y, w, h) given in figure inches."""
    r = fig.canvas.get_renderer()
    x, y, w, h = rect_inch
    for t in texts:
        bb = t.get_window_extent(r)
        if bb.x0 / fig.dpi < x - tol or bb.x1 / fig.dpi > x + w + tol or bb.y0 / fig.dpi < y - tol \
                or bb.y1 / fig.dpi > y + h + tol:
            return False
    return True


def _texts_in(res, rect):
    x, y, w, h = rect
    return [t for t in res["background"].texts if t.get_text().strip()
            and x <= t.get_position()[0] <= x + w and y <= t.get_position()[1] <= y + h]


def _axes_text(ax):
    return [t for t in ax.texts if t.get_text().strip()]


# ---- recipe_grid ------------------------------------------------------------------------------

def test_recipe_row_order_sizes_the_figure_and_returns_one_panel_per_step():
    res = recipe_grid(None, _steps(8), ncol=4, box=(1.5, 2.3), gap=(0.2, 0.3), margin=0.05)
    fig = res["fig"]
    w, h = fig.get_size_inches()
    assert w == pytest.approx(2 * 0.05 + 4 * 1.5 + 3 * 0.2)
    assert h == pytest.approx(2 * 0.05 + 2 * 2.3 + 0.3)
    assert len(res["panels"]) == 8 and all(p is not None for p in res["panels"])
    assert all(p.get_xlabel() == "x" for p in res["panels"])
    xs = [f[0] for f in res["frames"]]
    ys = [f[1] for f in res["frames"]]
    assert xs[:4] == sorted(xs[:4]) and xs[4] == xs[0] and ys[0] > ys[4]
    arrows = [p for p in res["background"].patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == 7
    assert not res["overfull"]


def test_recipe_snake_order_reverses_the_second_row_and_fits_a_between_strip():
    side = [dict(after=5, at="between", title="Optional at step 5:", lines=["resolve the edge fields."])]
    res = recipe_grid(None, _steps(6), ncol=3, order="snake", side=side)
    fx = [f[0] for f in res["frames"]]
    assert fx[3] == fx[2] and fx[4] == fx[1] and fx[5] == fx[0]
    sx, sy, sw, sh = res["side_frames"][0]
    # the down arrow from step 3 to step 4 runs through column 2, so the strip spans columns 0 and 1
    assert sx == pytest.approx(fx[0]) and sx + sw < fx[2]
    assert res["frames"][3][1] + 2.32 < sy < res["frames"][0][1]


def test_recipe_column_order_with_plots_right_and_side_boxes_left():
    steps = _steps(6, tools=["spectroscopy", ("vna", "VNA sweep")])
    side = [dict(after=1, at="left", title="Optional", lines=["ageing"], panel=_line),
            dict(after=3, at="left", title="Optional", lines=["hybrid solve"])]
    res = recipe_grid(None, steps, ncol=2, order="column", plot="right", box=(2.6, 1.4), side=side)
    f = res["frames"]
    assert f[1][0] == f[0][0] and f[1][1] < f[0][1] and f[3][0] > f[0][0] and f[3][1] == f[0][1]
    for pax, fr in zip(res["panels"], f):
        px0 = pax.get_position().x0 * res["size"][0]
        assert px0 > fr[0] + fr[2] / 2                    # the mini-axes sits in the right half
    assert res["side_frames"][0][0] < f[0][0] and res["side_panels"][0] is not None
    assert res["side_panels"][1] is None
    chips = [p for p in res["background"].patches if isinstance(p, FancyBboxPatch)
             and p.get_facecolor()[:3] == matplotlib.colors.to_rgb(sq.theme().tint)]
    assert len(chips) == 6


def test_recipe_one_step_has_no_arrows():
    res = recipe_grid(None, _steps(1), ncol=4)
    assert len(res["frames"]) == 1
    assert res["fig"].get_size_inches()[0] == pytest.approx(2 * 0.04 + 1.58)
    assert not [p for p in res["background"].patches if isinstance(p, FancyArrowPatch)]


def test_recipe_empty_lines_no_tools_and_no_panel():
    res = recipe_grid(None, [dict(title="Bare", lines=[], panel=None), dict(title="", lines=None, panel=False)])
    assert res["panels"] == [None, None]
    assert len(res["fig"].axes) == 1                       # only the background axes
    assert not res["overfull"]


def test_recipe_long_text_wraps_inside_its_box():
    long = "This instruction is far too long for a single line of a narrow recipe box and must wrap."
    res = recipe_grid(None, [dict(title="Wrap", lines=[long], panel=None, tools=["pen"])], box=(1.6, 2.3))
    fig = res["fig"]
    fig.canvas.draw()
    texts = _texts_in(res, res["frames"][0])
    body = [t for t in texts if t.get_text() not in ("Wrap", "1", "You use:", "pen")]
    assert len(body) >= 3
    assert " ".join(t.get_text() for t in body) == long
    assert _inside(fig, texts, res["frames"][0])


def test_recipe_wrap_lines_keeps_each_instruction_and_hangs_its_continuation():
    long = "Extract the Maxwell matrix of every floating node from the solver"
    lines = [long, "Tune the claw"]
    res = recipe_grid(None, [dict(title="Wrap", lines=lines, panel=None)], box=(1.6, 2.3), wrap="lines")
    res["fig"].canvas.draw()
    body = [t for t in _texts_in(res, res["frames"][0]) if t.get_text() not in ("Wrap", "1")]
    assert body[-1].get_text() == "Tune the claw"                    # the short item stays whole, on its own
    assert " ".join(t.get_text() for t in body[:-1]) == long and len(body) >= 3
    x0 = body[0].get_position()[0]
    assert all(t.get_position()[0] == pytest.approx(x0 + layout._HANG) for t in body[1:-1])
    assert body[-1].get_position()[0] == pytest.approx(x0)
    assert _inside(res["fig"], body, res["frames"][0]) and not res["overfull"]
    # the default rewraps the whole paragraph, so the short item joins the long one
    para = recipe_grid(None, [dict(title="Wrap", lines=lines, panel=None)], box=(1.6, 2.3))
    assert "Tune the claw" not in [t.get_text() for t in para["background"].texts]


def test_recipe_wrap_lines_fits_short_lines_and_side_boxes_unchanged():
    side = [dict(after=1, title="Optional", lines=["a short line", "and another"])]
    a = recipe_grid(None, _steps(2), ncol=2, side=side)
    b = recipe_grid(None, _steps(2), ncol=2, side=side, wrap="lines")
    texts = lambda r: [(t.get_text(), t.get_position()) for t in r["background"].texts]
    assert texts(a) == texts(b) and a["size"] == b["size"]
    c = recipe_grid(None, _steps(2), ncol=2, side=side, wrap="shrink")     # short lines keep the default size
    assert texts(a) == texts(c) and c["body"] == a["body"] == sq.theme().size - 0.5


def test_recipe_wrap_shrink_sets_every_line_whole_at_one_smaller_size():
    lines = ["Extract the Maxwell matrix", "Tune the claw"]
    steps = [dict(title="One", lines=lines, panel=_line, panel_left=0.3, tools=["pen"]),
             dict(title="Two", lines=["short"], panel=_line, tools=["pen"])]
    side = [dict(after=2, at="left", title="Side", lines=["a side line"], panel=_line)]
    res = recipe_grid(None, steps, ncol=1, order="column", plot="right", box=(2.6, 1.4), side=side, wrap="shrink")
    fig = res["fig"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert sq.theme().size - 2.0 <= res["body"] < sq.theme().size - 0.5 and not res["overfull"]
    texts = {t.get_text(): t for t in res["background"].texts}
    assert set(lines) | {"short", "a side line"} <= set(texts)      # every instruction stays whole
    body = [texts[s] for s in lines + ["short", "a side line", "pen"]]
    assert {t.get_fontsize() for t in body} == {res["body"]}           # one size for lines and tools
    x, y, w, h = res["frames"][0]
    px0 = res["panels"][0].get_position().x0 * res["size"][0]
    assert texts[lines[0]].get_window_extent(r).x1 / fig.dpi <= px0 - 0.3 + 0.01   # clear of the tick labels


def test_recipe_wrap_shrink_reports_a_line_too_long_at_the_smallest_size():
    long = "a far too long instruction that cannot fit a narrow box even at the smallest size allowed"
    res = recipe_grid(None, [dict(title="x", lines=[long], panel=None)], box=(1.6, 1.4), wrap="shrink")
    assert res["body"] == pytest.approx(sq.theme().size - 2.0)
    assert long in [t.get_text() for t in res["background"].texts]   # still one line, never wrapped
    assert any("wider than the box" in s for s in res["overfull"])


def test_recipe_reports_text_that_overflows_its_budget():
    lines = ["one long line of text that keeps going"] * 6
    res = recipe_grid(None, [dict(title="Too much", lines=lines, panel=_line)], text_lines=2)
    assert any("lines do not fit" in s for s in res["overfull"])


def test_recipe_unicode_and_mathtext():
    steps = [dict(title="Résumé – µs", lines=[r"Set $\chi/2\pi\approx-0.5$ MHz", "Ω, τ and ∆ in text"],
                  panel=None, tools=[("code", "scqubits µ")])]
    res = recipe_grid(None, steps)
    res["fig"].canvas.draw()
    got = " ".join(t.get_text() for t in res["background"].texts)
    assert "Résumé – µs" in got and r"$\chi/2\pi\approx-0.5$" in got and "scqubits µ" in got


def test_recipe_below_side_box_joins_the_lowest_box_of_its_column():
    res = recipe_grid(None, _steps(7), ncol=4, side=[dict(after=2, title="Optional", lines=["x"])])
    sx, sy, sw, sh = res["side_frames"][0]
    assert sy == pytest.approx(0.04) and sx == pytest.approx(res["frames"][1][0])
    assert sy + sh < min(f[1] for f in res["frames"])


@pytest.mark.parametrize("kw, msg", [
    (dict(steps=[]), "at least one step"),
    (dict(order="spiral"), "order"),
    (dict(plot="above"), "plot"),
    (dict(ncol=0), "ncol"),
    (dict(side=[dict(after=9, title="x")]), "after"),
    (dict(side=[dict(after=1, at="top")]), "at="),
    (dict(side=[dict(after=1, at="below", panel=_line)]), "panel"),
    (dict(side=[dict(after=2), dict(after=6)]), "overlap"),
    (dict(box=(1.58, 1.0)), "panels"),
    (dict(steps=[dict(title="x", tools=[("laser", "LASER")])]), "unknown tool icon"),
    (dict(steps=[dict(title="x", colour="red")]), "unknown keys"),
    (dict(steps=_steps(3), side=[dict(after=1, at="between")]), "two rows"),
    (dict(wrap="words"), "wrap"),
])
def test_recipe_rejects_bad_input(kw, msg):
    kw = dict(kw)
    steps = kw.pop("steps", _steps(8))
    kw.setdefault("ncol", 4)
    with pytest.raises(ValueError, match=msg):
        recipe_grid(None, steps, **kw)


def test_recipe_tool_icons_are_all_drawable():
    tools = list(layout.TOOL_ICONS)
    res = recipe_grid(None, [dict(title="Tools", lines=["all icons"], panel=None, tools=tools)], box=(9.0, 1.2))
    labels = {t.get_text() for t in res["background"].texts}
    assert {"VNA", "ADC", "EM", "pen", "fridge"} <= labels and not res["overfull"]


# ---- text_panel -------------------------------------------------------------------------------

def test_text_panel_spreads_lines_and_keeps_a_full_frame():
    with sq.style("folio"):
        fig, ax = plt.subplots(figsize=(2.0, 1.2))
        out = text_panel(ax, [r"$H=\omega a^\dagger a$", ("second", "accent"), "third"])
        assert len(out) == 3
        ys = [t.get_position()[1] for t in out]
        assert ys == sorted(ys, reverse=True) and ys[0] == pytest.approx(1 - 0.5 / 3)
        assert all(sp.get_visible() for sp in ax.spines.values())          # folio hides two spines by default
        assert out[1].get_color() == sq.theme().accent
        assert not ax.get_xticks().size


def test_text_panel_empty_draws_only_the_frame():
    fig, ax = plt.subplots(figsize=(2, 1))
    assert text_panel(ax, []) == [] and not ax.texts
    fig2, ax2 = plt.subplots(figsize=(2, 1))
    assert text_panel(ax2, [], frame=False) == [] and not ax2.axison


def test_text_panel_shrinks_or_wraps_a_long_line_to_fit():
    fig, ax = plt.subplots(figsize=(1.6, 1.4))
    long = "a statement that is much wider than this small framed panel and must be set smaller or wrapped"
    out = text_panel(ax, [long], fs=9)
    fig.canvas.draw()
    assert len(out) > 1 or out[0].get_fontsize() < 9
    assert " ".join(t.get_text() for t in out) == long
    bb = ax.get_window_extent()
    r = fig.canvas.get_renderer()
    assert all(bb.x0 - 1 <= t.get_window_extent(r).x0 and t.get_window_extent(r).x1 <= bb.x1 + 1 for t in out)


def test_text_panel_top_alignment_uses_a_fixed_pitch():
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    out = text_panel(ax, ["a", "b", "c"], valign="top", align="left", frame=False)
    ys = [t.get_position()[1] for t in out]
    assert ys[0] > 0.9 and ys[2] > 0.6
    assert np.diff(ys) == pytest.approx([ys[1] - ys[0]] * 2)


def test_text_panel_unicode_and_colors_list():
    fig, ax = plt.subplots(figsize=(2, 1))
    out = text_panel(ax, ["Ω ≈ 2π × 5 GHz", r"$\sqrt{E_J/E_C}$"], colors=["bus", None])
    assert out[0].get_color() == sq.theme().bus and out[1].get_color() == sq.theme().ink


@pytest.mark.parametrize("kw", [dict(align="justify"), dict(valign="bottom"), dict(colors=["ink"])])
def test_text_panel_rejects_bad_options(kw):
    fig, ax = plt.subplots(figsize=(2, 1))
    with pytest.raises(ValueError):
        text_panel(ax, ["a", "b"], **kw)


def test_text_panel_rejects_broken_mathtext_and_bad_lines():
    fig, ax = plt.subplots(figsize=(2, 1))
    with pytest.raises(ValueError):
        text_panel(ax, [r"$\frac{1}{$"])
    with pytest.raises(ValueError):
        text_panel(ax, [("a", "b", "c")])


# ---- matrix_text ------------------------------------------------------------------------------

def test_matrix_text_places_cells_labels_and_brackets():
    fig, ax = plt.subplots(figsize=(3, 2.5))
    labs = ["$q_0$", "$r_0$", "$q_1$", "$r_1$"]
    t = sq.theme()
    cols = [t.qubits[0], "bus", t.qubits[1], "bus"]
    out = matrix_text(ax, np.arange(16.0).reshape(4, 4), labs, labs, label_colors=cols, fmt="{:.0f}")
    assert len(out["cells"]) == 4 and all(len(r) == 4 for r in out["cells"])
    assert out["cells"][2][3].get_text() == "11"
    assert [x.get_color() for x in out["row_labels"]] == [t.qubits[0], t.bus, t.qubits[1], t.bus]
    assert [x.get_color() for x in out["col_labels"]] == [t.qubits[0], t.bus, t.qubits[1], t.bus]
    assert len(out["brackets"]) == 2 and not ax.axison
    xs = [c.get_position()[0] for c in out["cells"][0]]
    ys = [r[0].get_position()[1] for r in out["cells"]]
    assert xs == sorted(xs) and ys == sorted(ys, reverse=True)


def test_matrix_text_strings_callable_format_table_and_notes():
    fig, ax = plt.subplots(figsize=(3, 3))
    M = np.array([["0", "1", "0"], ["0", "0", r"$\lambda$"], ["0", "0", "0"]], dtype=object)
    out = matrix_text(ax, M, brackets=False, title="operator", notes=["note one", "nöte two"],
                      table=(["mode", "bare"], [["q", 5.2], ["r", 7.1]]), fmt=lambda v: f"{v:.2f}")
    assert out["cells"][1][2].get_text() == r"$\lambda$" and not out["brackets"]
    assert out["title"].get_text() == "operator" and [n.get_text() for n in out["notes"]] == ["note one", "nöte two"]
    assert "5.20" in [x.get_text() for x in out["table"] if hasattr(x, "get_text")]


def test_matrix_text_title_lines_and_text_beside_the_matrix():
    fig, ax = plt.subplots(figsize=(3, 2.5))
    M = np.array([["0", "1"], ["0", r"$\lambda$"]], dtype=object)
    out = matrix_text(ax, M, title=[r"$N=\mathrm{diag}(0,1)$", "operator:"], beside=r"$\lambda=\sqrt{2}$",
                      notes=["note"])
    assert [t.get_text() for t in out["titles"]] == [r"$N=\mathrm{diag}(0,1)$", "operator:"]
    assert out["title"] is out["titles"][0]
    y_t = [t.get_position()[1] for t in out["titles"]]
    y_rows = [r[0].get_position()[1] for r in out["cells"]]
    assert y_t[0] > y_t[1] > max(y_rows)                             # titles stack above the matrix
    (b,) = out["beside"]
    assert b.get_position()[1] == pytest.approx(np.mean(y_rows))     # centred on the rows
    right_bracket = max(max(ln.get_xdata()) for ln in out["brackets"])
    assert b.get_position()[0] > right_bracket
    assert out["notes"][0].get_position()[1] < min(y_rows)
    # a single string title behaves as before
    fig2, ax2 = plt.subplots(figsize=(3, 2.5))
    one = matrix_text(ax2, M, title="operator")
    assert one["title"].get_text() == "operator" and len(one["titles"]) == 1 and one["beside"] == []


def test_matrix_text_shrinks_to_keep_the_beside_text_inside():
    fig, ax = plt.subplots(figsize=(2.0, 1.2))
    out = matrix_text(ax, np.eye(3), beside="the entries in GHz", fs=8)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert out["fs"] < 8
    assert out["beside"][0].get_window_extent(r).x1 <= ax.get_window_extent(r).x1 + 1


def test_matrix_text_one_by_one_and_row_vector():
    fig, ax = plt.subplots(figsize=(1.5, 1))
    assert matrix_text(ax, [[3.0]])["cells"][0][0].get_text() == "3"
    fig, ax = plt.subplots(figsize=(2, 1))
    assert len(matrix_text(ax, [1, 2, 3])["cells"][0]) == 3


def test_matrix_text_shrinks_a_wide_matrix_to_fit_the_axes():
    fig, ax = plt.subplots(figsize=(1.5, 1.2))
    out = matrix_text(ax, np.full((5, 8), 123456.789), fmt="{:.3f}", fs=8)
    assert out["fs"] < 8


@pytest.mark.parametrize("args, kw", [
    ((np.zeros((2, 2, 2)),), {}),
    ((np.zeros((0, 3)),), {}),
    ((np.eye(2), ["a"]), {}),
    ((np.eye(2), None, ["a", "b", "c"]), {}),
    ((np.eye(2),), dict(label_colors=["ink"])),
    ((np.eye(2),), dict(table=(["a", "b"], [["1"]]))),
])
def test_matrix_text_rejects_bad_shapes(args, kw):
    fig, ax = plt.subplots(figsize=(2, 2))
    with pytest.raises(ValueError):
        matrix_text(ax, *args, **kw)


# ---- term_tiles -------------------------------------------------------------------------------

def _hamiltonian():
    return [["$H/h=$", dict(text=r"$5.2\,b^\dagger b$", kind="nonlinear", brace="qubit"), "$+$",
             dict(text=r"$-\dfrac{0.21}{2}b^{\dagger2}b^2$", kind="nonlinear", brace="qubit"), "$+$",
             dict(text=r"$7.1\,a^\dagger a$", kind="inductive", color="bus", brace="resonator")],
            ["$+$", dict(text=r"$0.09\,(a^\dagger b+ab^\dagger)$", kind="coupling", brace="coupling"),
             dict(text="rotating-wave form", color="reference", gap=0.2)]]


def test_term_tiles_lays_terms_on_tiles_and_merges_shared_braces():
    fig, ax = plt.subplots(figsize=(6, 2))
    out = term_tiles(ax, _hamiltonian())
    assert len(out["tiles"]) == 4
    assert [lab.get_text() for lab in out["labels"]] == ["qubit", "resonator", "coupling"]
    assert len(out["braces"]) == 6                         # a bracket and a stem per label
    t = sq.theme()
    assert out["tiles"][0].get_facecolor()[:3] == pytest.approx(matplotlib.colors.to_rgb(t.tiles["nonlinear"]))
    bus = [x for x in out["texts"] if "7.1" in x.get_text()][0]
    assert bus.get_color() == t.bus
    xs = [x.get_position()[0] for x in out["texts"][:6]]
    assert xs == sorted(xs)
    assert out["texts"][6].get_position()[1] < out["texts"][0].get_position()[1]   # second line below


def test_term_tiles_stacks_colliding_labels_instead_of_overlapping_them():
    fig, ax = plt.subplots(figsize=(6, 2))
    lines = [[dict(text="$a$", kind="capacitive", brace="a long label for a short term"),
              dict(text="$b$", kind="inductive", brace="another long label")]]
    out = term_tiles(ax, lines)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    a, b = (lab.get_window_extent(r) for lab in out["labels"])
    assert not (a.x1 > b.x0 and a.x0 < b.x1 and a.y1 > b.y0 and a.y0 < b.y1)


def test_term_tiles_glue_only_and_no_braces():
    fig, ax = plt.subplots(figsize=(3, 1))
    out = term_tiles(ax, [["$H=0$"]], braces=True)
    assert not out["tiles"] and not out["braces"]
    fig, ax = plt.subplots(figsize=(6, 2))
    out = term_tiles(ax, _hamiltonian(), braces=False)
    assert not out["braces"] and not out["labels"] and len(out["tiles"]) == 4


def _term_row(n, note=True):
    items = ["$H/h=$"]
    for k in range(n):
        items += (["$+$"] if k else []) + [dict(text=rf"$5.{k}\,N_{k}$", kind="nonlinear")]
    return items + ([dict(text="Ω drive, 12 MHz", color="drive")] if note else [])


def test_term_tiles_shrinks_a_slightly_long_line():
    fig, ax = plt.subplots(figsize=(4.0, 1.0))
    out = term_tiles(ax, [_term_row(4)])
    assert out["fs"] < sq.theme().size + 0.6
    assert len({x.get_position()[1] for x in out["texts"]}) == 1          # still one line


def test_term_tiles_breaks_a_far_too_long_line_between_terms():
    fig, ax = plt.subplots(figsize=(2.0, 1.6))
    out = term_tiles(ax, [_term_row(8, note=False)])
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = ax.get_window_extent(r)
    assert max(x.get_window_extent(r).x1 for x in out["texts"]) <= bb.x1 + 2
    baselines = sorted({round(x.get_position()[1], 6) for x in out["texts"]}, reverse=True)
    assert len(baselines) > 1
    starts = [min((x for x in out["texts"] if round(x.get_position()[1], 6) == b), key=lambda x: x.get_position()[0])
              for b in baselines[1:]]
    assert all(s.get_text() == "$+$" for s in starts)                     # the "+" moves down with its term


@pytest.mark.parametrize("lines", [
    [[dict(text="$a$", tile="nonlinear")]],
    [[dict(text="$a$", kind="nonsense")]],
    [[dict(kind="nonlinear")]],
    [[42]],
    ["$H$"],
])
def test_term_tiles_rejects_bad_items(lines):
    fig, ax = plt.subplots(figsize=(3, 1))
    with pytest.raises(ValueError):
        term_tiles(ax, lines)


# ---- flow_diagram -----------------------------------------------------------------------------

TAGS = {"S": dict(color="ink", label="specified"), "A": dict(color="accent", label="assumed"),
        "M": dict(color="ink", fill="light", label="measured")}


def _snake():
    nodes = {f"n{i}": dict(cell=(i // 3, i % 3 if (i // 3) % 2 == 0 else 2 - i % 3), title=f"({i}) Step",
                           lines=[("a quantity", "S"), ("an input", "A")]) for i in range(6)}
    edges = [(f"n{i}", f"n{i + 1}") for i in range(5)]
    return nodes, edges


def test_flow_grid_places_cells_top_down_and_routes_straight_arrows():
    fig, ax = plt.subplots(figsize=(7, 3))
    nodes, edges = _snake()
    out = flow_diagram(ax, nodes, edges, grid=dict(w=2.0, h=1.0, gap=(0.3, 0.4)), tags=TAGS)
    b = out["boxes"]
    assert b["n0"] == pytest.approx((0.0, 1.4, 2.0, 1.0)) and b["n3"] == pytest.approx((4.6, 0.0, 2.0, 1.0))
    arrows = [a for a in out["edges"] if isinstance(a, FancyArrowPatch)]
    assert len(arrows) == 5
    (x0, y0), (x1, y1) = arrows[0]._posA_posB
    assert y0 == pytest.approx(y1) and x1 > x0                       # n0 -> n1 runs right
    (x0, y0), (x1, y1) = arrows[2]._posA_posB
    assert x0 == pytest.approx(x1) and y1 < y0                       # n2 -> n3 runs down
    assert sum(len(v) for v in out["badges"].values()) == 12
    assert ax.get_aspect() == 1.0 and not ax.axison


def test_flow_legend_lists_labelled_tags():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    nodes, edges = _snake()
    out = flow_diagram(ax, nodes, edges, grid=dict(w=2.0, h=1.0), tags=TAGS, legend=True)
    labels = [t.get_text() for t in out["legend"]]
    assert labels == ["S", "specified", "A", "assumed", "M", "measured"]
    assert out["extent"][2] < -0.2                                  # the legend row sits under the grid


def test_flow_offset_dashed_edge_and_label_side():
    fig, ax = plt.subplots(figsize=(4, 3))
    nodes = {"a": dict(xy=(0, 1.5), w=2, h=0.8, title="A", lines=["x"]), "b": dict(xy=(0, 0), w=2, h=0.8, text="B")}
    edges = [("a", "b", dict(offset=-0.5)), ("b", "a", dict(ls="dashed", offset=0.5, label="back", label_side="left"))]
    out = flow_diagram(ax, nodes, edges)
    down, up = [a for a in out["edges"] if isinstance(a, FancyArrowPatch)]
    assert down._posA_posB[0][0] == pytest.approx(0.5) and up._posA_posB[0][0] == pytest.approx(1.5)
    assert up.get_linestyle() not in ("-", "solid") and down.get_linestyle() in ("-", "solid")
    assert out["labels"][0].get_ha() == "right"


def test_flow_elbow_route_points_and_explicit_ports():
    fig, ax = plt.subplots(figsize=(5, 3))
    nodes = {"a": dict(xy=(0, 2), w=1, h=0.6, text="A"), "b": dict(xy=(3, 0), w=1, h=0.6, text="B")}
    out = flow_diagram(ax, nodes, [("a", "b"), ("a", "b", dict(route="hv", ls="dotted")),
                                   ("a", "b", dict(ports=("bottom", "top"), at=(0.8, 3.2))),
                                   ((1.5, 2.9), "b", dict(head=False)), dict(path=[(0, 0), (1, 0), (1, 1)])])
    lines = [a for a in out["edges"] if not isinstance(a, FancyArrowPatch)]
    vh = lines[0].get_xydata()
    assert vh[0] == pytest.approx((0.5, 1.97)) and vh[1][0] == pytest.approx(0.5)          # down first
    hv = lines[1].get_xydata()
    assert hv[0] == pytest.approx((1.03, 2.3)) and hv[1][1] == pytest.approx(2.3)           # across first
    straight = [a for a in out["edges"] if isinstance(a, FancyArrowPatch)][2]
    assert straight._posA_posB[0] == pytest.approx((0.8, 1.97)) and straight._posA_posB[1] == pytest.approx((3.2, 0.63))


def test_flow_fans_draw_one_arrow_and_label_per_item():
    fig, ax = plt.subplots(figsize=(6, 3))
    nodes = {"s": dict(xy=(0, 2), w=1.5, h=0.6, text="screened"), "f": dict(xy=(2.5, 0), w=1.2, h=0.6, text="full")}
    fans = [dict(src="s", port="bottom", items=["30 out", "12 dup"]),
            dict(src="f", port="right", items=["a", "b", "c"], header="category"),
            dict(src="f", port="left", items=["z"], at=[0.3]), dict(src="s", items=[])]
    out = flow_diagram(ax, nodes, fans=fans)
    texts = [t.get_text() for t in out["labels"]]
    assert texts == ["30 out", "12 dup", "a", "b", "c", "category", "z"]
    assert sum(isinstance(a, FancyArrowPatch) for a in out["fans"]) == 6
    assert out["labels"][-1].get_ha() == "right"
    bottom = [t for t in out["labels"] if t.get_text() == "12 dup"][0]
    assert bottom.get_position()[1] == pytest.approx(2 - 2 * 0.45)


def test_flow_col_labels_frameless_nodes_and_draw_callback():
    seen = []
    fig, ax = plt.subplots(figsize=(6, 2.5))
    nodes = {"name": dict(cell=(0, 0), text="Direct\nexcitation", frame=False),
             "elem": dict(cell=(0, 1), text="dipole", inset=0.8, draw=lambda a, r: seen.append(r)),
             "rec": dict(cell=(0, 2), text="counts", ls="dashed", shape="square", color="bus", fill="tint")}
    out = flow_diagram(ax, nodes, [("elem", "rec")], grid=dict(widths=[1.2, 2.4, 1.6], h=1.0,
                                                               col_labels=["", "element", "record"]))
    assert seen == [pytest.approx((1.5, 0.0, 0.8, 1.0))]
    assert "name" not in out["frames"] and out["frames"]["rec"].get_edgecolor()[:3] == pytest.approx(
        matplotlib.colors.to_rgb(sq.theme().bus))
    assert [t.get_text() for t in out["labels"]] == ["element", "record"]
    assert out["texts"]["elem"][0].get_position()[0] == pytest.approx(1.5 + 0.8 + (2.4 - 0.8) / 2)


def test_flow_single_node_empty_lines_and_unicode():
    fig, ax = plt.subplots(figsize=(3, 2))
    out = flow_diagram(ax, [dict(id="only", xy=(0, 0), title="Ω résumé", lines=[])])
    assert list(out["boxes"]) == ["only"] and not out["edges"]
    assert out["texts"]["only"][0].get_text() == "Ω résumé"


def test_flow_shrinks_text_that_overflows_its_box():
    fig, ax = plt.subplots(figsize=(3, 1.5))
    long = "a line that is rather too long for this small box"
    out = flow_diagram(ax, {"a": dict(xy=(0, 0), w=2.6, h=0.9, title="Title", lines=[(long, "S")])}, tags=TAGS)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    line = out["texts"]["a"][1]
    assert line.get_fontsize() < sq.theme().size - 0.5
    box = out["frames"]["a"].get_window_extent(r)
    assert line.get_window_extent(r).x1 <= out["badges"]["a"][0].get_window_extent(r).x0 + 1
    assert box.x0 <= line.get_window_extent(r).x0


@pytest.mark.parametrize("align", ["center", "left"])
def test_flow_pitch_sets_the_line_spacing_in_font_sizes(align):
    lines = ["one", "two", "three"]

    def gaps(**kw):
        fig, ax = plt.subplots(figsize=(4, 2))
        node = dict(xy=(0, 0), w=3.0, h=1.6, lines=lines, align=align, **kw)
        out = flow_diagram(ax, {"a": node}, fit=False)
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(-0.2, 1.8)
        fig.canvas.draw()
        ys = [t.get_window_extent(fig.canvas.get_renderer()).y0 for t in out["texts"]["a"]]
        return -np.diff(ys) * 72 / fig.dpi                      # in points
    f = sq.theme().size - 0.5
    default = 1.75 if align == "left" else 1.45
    assert gaps() == pytest.approx([default * f] * 2, abs=0.6)
    assert gaps(pitch=1.2) == pytest.approx([1.2 * f] * 2, abs=0.6)


@pytest.mark.parametrize("nodes, edges, kw, msg", [
    ({"a": dict(xy=(0, 0), pitch=0)}, [], {}, "pitch"),
    ({"a": dict(xy=(0, 0))}, [("a", "zz")], {}, "neither a node"),
    ({"a": dict(xy=(0, 0), lines=[("x", "Q")])}, [], {}, "unknown tag"),
    ({"a": dict(cell=(0, 0))}, [], {}, "need grid"),
    ([dict(id="a", xy=(0, 0)), dict(id="a", xy=(1, 0))], [], {}, "duplicate"),
    ({"a": dict(xy=(0, 0), colour="bus")}, [], {}, "unknown keys"),
    ({"a": dict(xy=(0, 0)), "b": dict(xy=(3, 0))}, [("a", "b", dict(ports=("north", None)))], {}, "ports"),
    ({"a": dict(xy=(0, 0))}, [], dict(fans=[dict(src="nope", items=["x"])]), "fan source"),
    ({"a": dict(xy=(0, 0), w=0)}, [], {}, "positive"),
    ({"a": dict()}, [], {}, "xy="),
    ({"a": dict(xy=(0, 0), text="t", lines=["l"])}, [], {}, "not both"),
    ({"a": dict(xy=(0, 0))}, [("a", "a")], {}, "itself"),
    ({"a": dict(xy=(0, 0))}, [dict(path=[(0, 0)])], {}, "two points"),
])
def test_flow_rejects_bad_input(nodes, edges, kw, msg):
    fig, ax = plt.subplots(figsize=(3, 2))
    with pytest.raises(ValueError, match=msg):
        flow_diagram(ax, nodes, edges, **kw)


# ---- the colour rule --------------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
def test_every_frame_obeys_the_colour_rule(style):
    with sq.style(style):
        steps = _steps(4, tools=["vna", "adc", "stats", "chip"])
        side = [dict(after=2, title="Optional", lines=["x"])]
        figs = [recipe_grid(None, steps, ncol=4, side=side)["fig"]]
        fig, axes = plt.subplots(2, 2, figsize=(7, 4))
        text_panel(axes[0, 0], [("a", "accent"), "b"])
        matrix_text(axes[0, 1], np.eye(2), ["$q$", "$r$"], label_colors=[sq.theme().qubits[0], "bus"])
        term_tiles(axes[1, 0], _hamiltonian())
        nodes, edges = _snake()
        flow_diagram(axes[1, 1], nodes, edges, grid=dict(w=2.0, h=1.0), tags=TAGS, legend=True)
        figs.append(fig)
        for f in figs:
            f.canvas.draw()
            bad = violations(f)
            assert not bad, [(type(a).__name__, c) for a, c in bad[:8]]
