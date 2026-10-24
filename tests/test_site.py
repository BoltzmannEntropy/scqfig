"""The GitHub Pages builder reads every example, writes LaTeX snippets and keeps README blocks in order."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_site", ROOT / "tools" / "build_site.py")
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_every_example_has_title_family_caption_and_a_plot_function(path):
    ex = site.read_example(path)
    assert ex["title"] != path.stem and ex["caption"]
    assert ex["family"] in {f[0] for f in site.FAMILIES}
    assert ex["functions"], "no sq.plots / sq.schematics call found"
    assert ex["code"] == path.read_text()


def test_latex_snippet_picks_column_or_page_width():
    ex = site.read_example(EXAMPLES[0])
    col, page = site.latex_snippet(ex, False), site.latex_snippet(ex, True)
    assert r"\begin{figure}[t]" in col and r"width=\columnwidth" in col
    assert r"\begin{figure*}[t]" in page and r"width=\textwidth" in page
    assert rf"\scqgraphics[width=\columnwidth]{{{ex['id']}}}" in col
    assert r"\usepackage[field,figdir=figures]{scqfig}" in col


def test_latex_snippet_escapes_caption_specials():
    ex = {"id": "99_x", "caption": "50% of T_1 & more"}
    assert r"\caption{50\% of T\_1 \& more}" in site.latex_snippet(ex, False)


def test_readme_blocks_keep_text_and_code_in_order():
    r = site.readme_parts()
    install = r["install"]
    assert "code" in install[0] and "pip install" in install[0]["code"] and install[0]["lang"] == "sh"
    kinds = ["code" if "code" in b else "text" for b in r["quickstart"]]
    assert kinds[0] == "code" and "text" in kinds
    assert all(b.get("text", "x").strip() for b in r["why"])


def test_palettes_cover_both_styles_with_roles_and_maps():
    p = site.palettes()
    assert set(p["styles"]) == {"field", "folio"}
    for t in p["styles"].values():
        assert t["roles"]["model"].startswith("#") and len(t["cmap"]) >= 2 and t["notes"]["frame"]
