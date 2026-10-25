import dataclasses

import matplotlib as mpl
import pytest

import scqfig as sq
from scqfig.check import allowed
from scqfig.themes import THEMES


@pytest.mark.parametrize("name", sorted(THEMES))
def test_every_theme_colour_obeys_the_rule(name):
    t = THEMES[name]
    for f in dataclasses.fields(t):
        v = getattr(t, f.name)
        vals = v.values() if isinstance(v, dict) else v if isinstance(v, tuple) else [v]
        for c in vals:
            if isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], float):   # colormap stop
                c = c[1]
            if isinstance(c, str) and c.startswith("#"):
                assert allowed(c), f"{name}.{f.name}: {c}"


def test_rule_rejects_green_red_and_cyan():
    for c in ("#2ca02c", "#d62728", "#17becf", "#b86e00", "#8c564b", "#ff0000", "#00ff00"):
        assert not allowed(c), c
    for c in ("#1f5fbf", "#f5e1ee", "#ffffff", "#3d3d3d", sq.lighten("orange", 0.6)):
        assert allowed(c), c


@pytest.mark.parametrize("name", sorted(THEMES))
def test_use_sets_style(name):
    with sq.style(name):
        t = sq.theme()
        assert t.name == name
        assert mpl.rcParams["xtick.direction"] == t.ticks
        assert mpl.rcParams["axes.spines.top"] == t.box
        assert mpl.rcParams["image.cmap"] == t.cmap
        assert t.cmap in mpl.colormaps and t.cmap_div + "_r" in mpl.colormaps
        fig, ax = sq.figure("column", aspect=0.5)
        assert fig.get_size_inches()[0] == pytest.approx(3.4)


def test_style_context_restores_previous():
    sq.use("field")
    with sq.style("folio"):
        assert sq.theme().name == "folio"
    assert sq.theme().name == "field"
    assert mpl.rcParams["xtick.direction"] == "in"


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        sq.use("neon")


def test_marker_grammar_differs_by_style():
    with sq.style("field"):
        assert sq.data_kw()["mfc"] == sq.theme().data
    with sq.style("folio"):
        assert sq.data_kw()["mfc"] == "white"


def test_overrides_replace_theme_fields_for_the_session_only():
    with sq.style("field", size=8.5) as t:
        assert t.size == 8.5 and sq.theme().size == 8.5
        assert mpl.rcParams["xtick.labelsize"] == 8.5 and mpl.rcParams["axes.labelsize"] == 9.0
    assert THEMES["field"].size == 8.0
    with pytest.raises(TypeError):
        sq.use("field", no_such_field=1)


def test_fourth_qubit_never_matches_the_bus():
    for t in THEMES.values():
        assert t.qubits[3] != t.bus


def test_title_draws_letter_alone_or_with_a_grey_finding():
    with sq.style("field"):
        fig, ax = sq.figure("column")
        a = sq.title(ax, "a", "ignored without a finding")
        assert a.get_text() == "(a)" and a.get_ha() == "right"
        head, finding = sq.title(ax, "b", "Couplings", "every model overestimates them")
        assert head.get_text() == "(b)  Couplings" and finding.get_color() == sq.theme().reference
        f = sq.fig_letter(fig, 0.1, 0.9, "c")
        assert f.get_text() == "(c)" and f.get_fontweight() == "bold"
