"""The LaTeX package is generated from themes.py and compiles in both styles."""
import pathlib
import re
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


def _has(sty):
    if not shutil.which("kpsewhich"):
        return False
    return bool(subprocess.run(["kpsewhich", sty], capture_output=True, text=True).stdout.strip())


@pytest.mark.skipif(not shutil.which("pdflatex") or not _has("quantikz.sty"), reason="no LaTeX or quantikz")
@pytest.mark.parametrize("opt", ["field", "folio"])
def test_quantikz_preset_compiles_without_red_slices(tmp_path, opt):
    doc = (r"\pdfcompresslevel=0\pdfobjcompresslevel=0\documentclass{article}\usepackage{tikz}\usepackage{quantikz}"
           rf"\usepackage[{opt}]{{scqfig}}\begin{{document}}"
           r"\begin{quantikz}[scqfig circuit]"
           r"\lstick{$|0\rangle$} & \gate{H}\gategroup[2,steps=2]{pair}"
           r" \slice[label style=scq slice label]{$\Psi_1$} & \ctrl{1} & \meter{} \\"
           r"\lstick{$|0\rangle$} & \qw & \targ{} & \meter{}"
           r"\end{quantikz}\end{document}")
    (tmp_path / "q.tex").write_text(doc)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "q.tex"], cwd=tmp_path,
                       capture_output=True, text=True, env={"TEXINPUTS": f"{STY.parent}:", "PATH": "/Library/TeX/texbin:/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout[-1500:]
    pdf = (tmp_path / "q.pdf").read_bytes()
    strokes = {tuple(float(v) for v in m.split()) for m in re.findall(rb"([\d.]+ [\d.]+ [\d.]+) RG", pdf)}
    assert (1.0, 0.0, 0.0) not in strokes               # quantikz's red slice is gone, not just painted over
    model = tuple(int(THEMES[opt].model[i:i + 2], 16) / 255 for i in (1, 3, 5))
    assert any(max(abs(a - b) for a, b in zip(s, model)) < 1e-3 for s in strokes), (model, strokes)


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
