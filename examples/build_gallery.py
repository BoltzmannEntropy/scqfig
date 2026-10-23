"""Render every example in both styles into gallery/<style>/, write a contact sheet per style, and
typeset gallery/gallery.pdf with the field and folio versions side by side.

    python examples/build_gallery.py                 # everything
    python examples/build_gallery.py --only 03 30    # examples whose number starts with these
    python examples/build_gallery.py --no-pdf        # skip LaTeX
"""
import argparse
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import scqfig as sq  # noqa: E402

OUT = ROOT / "gallery"
STYLES = ("field", "folio")


def examples(only=None):
    paths = sorted(HERE.glob("[0-9][0-9]_*.py"))
    return [p for p in paths if not only or any(p.stem.startswith(o) for o in only)]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render(paths):
    for style in STYLES:
        for p in paths:
            mod = load(p)
            with sq.style(style):
                sq.save(mod.build(), OUT / style / p.stem, dpi=200)
            print(f"{style:6s} {p.stem}")


def contact_sheet(paths, style, ncol=4):
    nrow = -(-len(paths) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 2.9), squeeze=False)
    with sq.style(style):
        for ax, p in zip(axes.flat, paths):
            ax.imshow(mpimg.imread(OUT / style / f"{p.stem}.png"))
            ax.set_title(f"{p.stem[:2]}  {load(p).TITLE}", fontsize=9, loc="left")
        for ax in axes.flat:
            ax.axis("off")
        fig.savefig(OUT / f"contact_{style}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "<": r"\textless{}", ">": r"\textgreater{}",
        "|": r"\textbar{}", "µ": r"$\mu$", "α": r"$\alpha$", "χ": r"$\chi$", "π": r"$\pi$", "Δ": r"$\Delta$",
        "ψ": r"$\psi$", "φ": r"$\varphi$", "×": r"$\times$", "±": r"$\pm$", "≈": r"$\approx$", "→": r"$\to$"}


def tex_escape(s):
    """Plain text for LaTeX, one character at a time, then a readable |psi(phi)|^2."""
    s = "".join(_TEX.get(ch, ch) for ch in s)
    return s.replace(r"\textbar{}psi(phi)\textbar{}\^{}2", r"$|\psi(\varphi)|^2$")


def write_tex(paths):
    lines = [r"\documentclass[10pt]{article}",
             r"\usepackage[a4paper,margin=16mm]{geometry}",
             r"\usepackage{graphicx}",
             r"\usepackage{pgfplots}\pgfplotsset{compat=1.16}",
             r"\usepackage[field,figdir=.]{scqfig}",
             r"\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",
             r"\setlength{\parindent}{0pt}",
             r"\begin{document}",
             r"{\LARGE\bfseries scqfig gallery}\par\medskip",
             r"Every example in both styles: \textcolor{scqBlue}{\textbf{field}} (left) and "
             r"\textbf{folio} (right). Build with \texttt{python examples/build\_gallery.py}; each panel "
             r"comes from \texttt{examples/NN\_name.py}.\par\bigskip"]
    lines += latex_page()
    family = None
    for p in paths:
        mod = load(p)
        if mod.FAMILY != family:
            family = mod.FAMILY
            lines.append(rf"\section*{{\color{{scqPurple}}{tex_escape(family)}}}")
        cap = tex_escape((mod.__doc__ or "").strip().splitlines()[0])
        lines += [r"\begin{minipage}{\textwidth}",
                  rf"\textbf{{{p.stem[:2]}\quad {tex_escape(mod.TITLE)}}}\hfill\texttt{{{tex_escape(p.stem)}.py}}\par",
                  rf"{{\small {cap}}}\par\smallskip",
                  r"\begin{minipage}[t]{0.49\textwidth}\centering"
                  rf"\includegraphics[width=\linewidth,height=0.36\textheight,keepaspectratio]{{field/{p.stem}.pdf}}"
                  r"\end{minipage}\hfill",
                  r"\begin{minipage}[t]{0.49\textwidth}\centering"
                  rf"\includegraphics[width=\linewidth,height=0.36\textheight,keepaspectratio]{{folio/{p.stem}.pdf}}"
                  r"\end{minipage}",
                  r"\end{minipage}\par\bigskip\medskip"]
    lines.append(r"\end{document}")
    (OUT / "gallery.tex").write_text("\n".join(lines) + "\n")


SWATCH_ROLES = ("Ink", "Data", "Model", "Sim", "Reference", "Accent", "Signal", "Bus", "Wash", "Overlay")


def latex_page():
    """First section: the colour roles of both styles and one pgfplots axis per style."""
    def row(name):
        cells = " & ".join(rf"\fcolorbox{{scqLight}}{{scq{st}{name}}}{{\rule{{0pt}}{{7pt}}\hspace{{16pt}}}}"
                           for st in ("Field", "Folio"))
        return rf"\texttt{{scq{name}}} & {cells} \\"
    names = list(SWATCH_ROLES) + [f"Qubit{c}" for c in "ABCD"] + [f"Level{i}" for i in range(4)] + \
        ["TileNonlinear", "TileCapacitive", "TileInductive", "ChipPad", "ChipResonator", "ChipDrive", "ChipFlux"]
    half = (len(names) + 1) // 2
    tab = lambda ns: ([r"\begin{tabular}[t]{lcc}", r"\textbf{colour} & field & folio \\[2pt]"] + [row(n) for n in ns] +
                      [r"\end{tabular}"])
    plot = lambda st: (rf"\begin{{tikzpicture}}\begin{{axis}}[scqfig {st},width=7.4cm,height=4.6cm,"
                       r"xlabel={free evolution ($\mu$s)},ylabel={$P_1$},ymin=0,ymax=1,xmin=0,xmax=10,"
                       r"legend pos=north east]"
                       rf"\addplot+[scqfig {st} data,samples=41,domain=0:10]"
                       r"{0.5+0.42*exp(-x/4)*cos(deg(2*pi*0.6*x))+0.02*sin(deg(37*x))};\addlegendentry{data}"
                       rf"\addplot+[scqfig {st} model,samples=300,domain=0:10]"
                       r"{0.5+0.42*exp(-x/4)*cos(deg(2*pi*0.6*x))};\addlegendentry{fit}"
                       r"\end{axis}\end{tikzpicture}")
    return ([r"\section*{\color{scqPurple}The LaTeX package}",
             r"\texttt{\textbackslash usepackage[field]\{scqfig\}} or \texttt{[folio]} defines the palette, both "
             r"styles' roles (\texttt{scqFieldAccent}, \texttt{scqFolioAccent}) and the active style's roles under "
             r"plain names (\texttt{scqAccent}). \texttt{\textbackslash scqgraphics\{t1\_decay\}} loads the "
             r"figure of the active style, and the pgfplots styles \texttt{scqfig}, \texttt{scqfig data}, "
             r"\texttt{scqfig model} and \texttt{scqfig ref} match the matplotlib output.\par\medskip",
             r"{\small"] + tab(names[:half]) + [r"\hfill"] + tab(names[half:]) + [r"}\par\bigskip",
             r"\noindent" + plot("field") + r"\hfill" + plot("folio") + r"\par",
             r"{\small pgfplots axes drawn natively by LaTeX with \texttt{[scqfig field]} (left) and "
             r"\texttt{[scqfig folio]} (right).}\par\bigskip", r"\clearpage"])


def write_catalogue(paths):
    """Rewrite the plot catalogue between the markers in README.md from the examples themselves."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    rows = ["| # | Plot | What it shows | Example |", "|---|---|---|---|"]
    for p in paths:
        mod = load(p)
        cap = (mod.__doc__ or "").strip().splitlines()[0].replace("|", "\\|")
        rows.append(f"| {p.stem[:2]} | {mod.TITLE} | {cap} | [`{p.name}`](examples/{p.name}) |")
    text = readme.read_text()
    a, b = "<!-- catalogue:start -->", "<!-- catalogue:end -->"
    if a in text and b in text:
        head, rest = text.split(a, 1)
        readme.write_text(head + a + "\n" + "\n".join(rows) + "\n" + b + rest.split(b, 1)[1])


def typeset():
    env = dict(os.environ, TEXINPUTS=f"{ROOT / 'latex'}:{os.environ.get('TEXINPUTS', '')}")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "gallery.tex"], cwd=OUT,
                           env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    for ext in ("aux", "log"):
        (OUT / f"gallery.{ext}").unlink(missing_ok=True)
    print(f"wrote {OUT / 'gallery.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    paths = examples(a.only)
    render(paths)
    if a.only:
        return
    for style in STYLES:
        contact_sheet(paths, style)
    write_tex(paths)
    write_catalogue(paths)
    if not a.no_pdf and shutil.which("pdflatex"):
        typeset()


if __name__ == "__main__":
    main()
