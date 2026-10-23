"""Shared helpers: colormap rule, matrix cells, colour bars, fit covariance, import-time setup."""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig import fit  # noqa: E402
from scqfig.check import cmap_ok  # noqa: E402
from scqfig.plots import annotated_matrix, colorbar  # noqa: E402
from scqfig.themes import THEMES  # noqa: E402


@pytest.mark.parametrize("name", sorted(THEMES))
def test_theme_colormaps_stay_in_allowed_hues(name):
    t = THEMES[name]
    for cm in (t.cmap, t.cmap_div, t.cmap + "_r"):
        assert cmap_ok(cm), cm


def test_rainbow_and_red_passing_maps_are_rejected():
    for cm in ("plasma", "viridis", "jet", "coolwarm", "RdBu"):
        assert not cmap_ok(cm), cm


def test_colormaps_registered_on_import():
    code = "import matplotlib as mpl, scqfig; print('scq_field' in mpl.colormaps and 'scq_folio_div_r' in mpl.colormaps)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": "src", "MPLBACKEND": "Agg", "PATH": ""}, cwd=str(__import__("pathlib").Path(__file__).parent.parent))
    assert out.stdout.strip() == "True", out.stderr


def test_annotated_matrix_uses_true_minus_and_no_negative_zero():
    with sq.style("field"):
        fig, ax = sq.figure("column")
        annotated_matrix(ax, np.array([[-0.004, -1.25], [0.5, 0.0]]), ["a", "b"], diverging=True)
        cells = [t.get_text() for t in ax.texts]
        plt.close(fig)
    assert "0.00" in cells and "−1.25" in cells
    assert not any(c.startswith("-") or c == "−0.00" for c in cells)


def test_annotated_matrix_accepts_cell_text():
    with sq.style("folio"):
        fig, ax = sq.figure("column")
        annotated_matrix(ax, np.eye(2), ["a", "b"], text=[["one", ""], ["", "two"]])
        cells = sorted(t.get_text() for t in ax.texts)
        plt.close(fig)
    assert cells == ["one", "two"]


def test_colorbar_spans_several_axes():
    with sq.style("field"):
        fig, axes = sq.figure("page", ncols=2)
        m = axes[0].imshow(np.random.default_rng(0).random((4, 4)), cmap=sq.theme().cmap)
        axes[1].imshow(np.zeros((4, 4)), cmap=sq.theme().cmap)
        cb = colorbar(list(axes), m, "z")
        assert cb.ax in fig.axes
        plt.close(fig)


def test_fit_keeps_covariance():
    t = np.linspace(0, 100, 40)
    r = fit.exp_decay(t, 0.9 * np.exp(-t / 30) + 0.05)
    assert r.cov.shape == (3, 3)
    assert np.sqrt(r.cov[1, 1]) == pytest.approx(r.errors["T"])
