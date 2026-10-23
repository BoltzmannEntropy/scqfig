"""Every example builds in both styles without warnings and obeys the colour rule."""
import importlib.util
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.check import violations  # noqa: E402

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("[0-9][0-9]_*.py"))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_examples_exist():
    assert len(EXAMPLES) >= 5


@pytest.mark.parametrize("style", ["field", "folio"])
@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_builds_cleanly(path, style):
    mod = load(path)
    assert (mod.__doc__ or "").strip(), "first docstring line is the gallery caption"
    assert getattr(mod, "TITLE", None) and getattr(mod, "FAMILY", None)
    with sq.style(style), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig = mod.build()
        fig.canvas.draw()
        msgs = [str(x.message) for x in w if "Glyph" in str(x.message) or "layout" in str(x.message).lower()
                or "missing from font" in str(x.message)]
        bad = violations(fig)
        plt.close(fig)
    assert not msgs, msgs
    assert not bad, [(type(a).__name__, c) for a, c in bad[:8]]
