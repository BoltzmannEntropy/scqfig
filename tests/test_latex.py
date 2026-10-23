"""The LaTeX package is generated from themes.py and compiles in both styles."""
import pathlib
import shutil
import subprocess

import pytest

from scqfig import latex
from scqfig.themes import PALETTE, THEMES

ROOT = pathlib.Path(__file__).resolve().parent.parent
STY = ROOT / "latex" / "scqfig.sty"


def test_committed_sty_matches_generator():
    assert STY.read_text() == latex.sty_text(), "run: python -m scqfig.latex"


def test_sty_defines_every_palette_colour_and_role():
    s = latex.sty_text()
    for k, v in PALETTE.items():
        assert f"{{{latex.cname(k)}}}{{HTML}}{{{v[1:].upper()}}}" in s
    for t in THEMES.values():
        assert f"scq{t.name.capitalize()}Accent" in s
    assert r"\DeclareVoidOption{folio}" in s and r"\scqgraphics" in s


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="no LaTeX")
@pytest.mark.parametrize("opt", ["field", "folio"])
def test_sty_compiles(tmp_path, opt):
    doc = (r"\documentclass{article}\usepackage{tikz}\usepackage{pgfplots}\pgfplotsset{compat=1.16}"
           rf"\usepackage[{opt},figdir=figs]{{scqfig}}\begin{{document}}\textcolor{{scqAccent}}{{x}}"
           r"\begin{tikzpicture}\begin{axis}[scqfig]\addplot+[scqfig model]{x^2};"
           r"\addplot+[scqfig data] coordinates {(0,0) (1,1)};\end{axis}"
           r"\node[scq tile=capacitive] {C};\end{tikzpicture}\end{document}")
    (tmp_path / "t.tex").write_text(doc)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=tmp_path,
                       capture_output=True, text=True, env={"TEXINPUTS": f"{STY.parent}:", "PATH": "/Library/TeX/texbin:/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout[-1500:]
