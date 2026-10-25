"""Build the GitHub Pages site in docs/ from the examples, their gallery renders and the palette.

    python examples/build_gallery.py      # renders every example in both styles (once)
    python tools/build_site.py            # then build the site data from those renders
    python tools/build_site.py --render   # or render missing figures here

Writes:
    docs/data/site.json        examples (source, caption, LaTeX snippet, functions), families,
                               both styles' palettes, the native LaTeX demos
    docs/figures/<style>/      PNG shown on the page and PDF offered for download
    docs/thumbs/<style>/       JPEG textures for the 3D wafer view
    docs/latex/                native pgfplots, TikZ and quantikz renders with their source
    docs/downloads/            scqfig.sty and gallery.pdf
Nothing here is edited by hand: rerun the script after adding an example or changing a theme."""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scqfig.themes import HUES, PALETTE, THEMES, lighten  # noqa: E402

EX = ROOT / "examples"
GAL = ROOT / "gallery"
DOCS = ROOT / "docs"
STYLES = ("field", "folio")
DPI = 200                                   # gallery PNG resolution (examples/build_gallery.py)
PAGE_WIDTH_IN = 4.6                         # wider renders are page-width figures

FAMILIES = [  # name, colour of its marker on the dark UI, one line
    ("Spectrum", lighten("blue", 0.3), "Levels, wavefunctions, charge dispersion, flux spectroscopy, avoided crossings."),
    ("Design", lighten("purple", 0.35), "Frequency plans, collisions, capacitance, loss participation, Purcell and dispersive design."),
    ("Readout", lighten("magenta", 0.15), "Resonator S21, quality factors, dispersive response, single-shot readout."),
    ("Time domain", PALETTE["orange"], "Rabi, chevrons, Ramsey, echo, benchmarking, coherence over time, TLS maps."),
    ("States", lighten("blue", 0.6), "Bloch spheres, Wigner functions, density matrices."),
    ("Noise and errors", lighten("magenta", 0.5), "Noise spectra, error budgets, correlated bursts and correlations."),
    ("Dark matter", lighten("magenta", 0.3), "Coupling reach against frequency, measured limits, dark-photon mass axis."),
    ("Processor", PALETTE["silver"], "Device maps and error distributions across a chip."),
    ("Schematics", lighten("purple", 0.62), "Chips in plan and oblique view, circuits, wiring, pulses, level diagrams, gate circuits, recipe grids, flow diagrams."),
    ("Comparison", lighten("orange", 0.35), "Stacked, grouped and factor bars, tornados, dot rows, ratio strips, paired stems, timelines."),
]

STYLE_NOTES = {
    "field": {"face": "Helvetica (Arial, DejaVu Sans as fallbacks)", "frame": "closed box, ticks inward on four sides",
              "data": "filled black points, blue fits", "series": "colour", "map": "pale magenta to purple to ink",
              "tiles": "pale blue, lilac and amber by element kind", "letters": "bold (a)"},
    "folio": {"face": "STIX Two, to match a Times or STIX body", "frame": "open L frame, ticks outward",
              "data": "open black points, black fits", "series": "line style first, then colour",
              "map": "white to pale blue to blue to ink", "tiles": "greys", "letters": "upright (a)"},
}


# ---- examples -------------------------------------------------------------------------------

def read_example(path):
    src = path.read_text()
    tree = ast.parse(src)
    meta = {n.targets[0].id: n.value.value for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    doc = (ast.get_docstring(tree) or "").strip()
    calls = re.findall(r"sq\.(?:plots|schematics)\.([A-Za-z_]\w*)", src)
    imported = [w.strip() for m in re.findall(r"from scqfig\.(?:plots|schematics)(?:\.\w+)? import ([\w, ]+)", src)
                for w in m.split(",")]
    for w in filter(None, imported):            # an imported module lists the functions called on it
        name, _, alias = w.partition(" as ")
        calls += [f"{name}.{f}" for f in re.findall(rf"\b{alias or name}\.([A-Za-z_]\w*)\(", src)] or [name]
    funcs = sorted(set(calls))
    return {"id": path.stem, "num": path.stem[:2], "title": meta.get("TITLE", path.stem), "family": meta.get("FAMILY", ""),
            "caption": doc.splitlines()[0] if doc else "", "code": src, "functions": funcs}


def latex_snippet(ex, page):
    env, width = ("figure*", r"\textwidth") if page else ("figure", r"\columnwidth")
    label = re.sub(r"^\d\d_", "", ex["id"]).replace("_", "-")
    cap = ex["caption"].replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
    return (f"% preamble: pick the style once for the whole paper\n"
            f"\\usepackage[field,figdir=figures]{{scqfig}}   % or [folio,...]\n\n"
            f"\\begin{{{env}}}[t]\n"
            f"  \\centering\n"
            f"  \\scqgraphics[width={width}]{{{ex['id']}}}   % figures/<style>/{ex['id']}.pdf\n"
            f"  \\caption{{{cap}}}\n"
            f"  \\label{{fig:{label}}}\n"
            f"\\end{{{env}}}\n")


def render_missing(stems):
    sys.path.insert(0, str(EX))
    import build_gallery as bg  # noqa: E402
    paths = [EX / f"{s}.py" for s in stems]
    bg.render(paths)


def copy_figures(ex):
    from PIL import Image
    out = {}
    for st in STYLES:
        png, pdf = GAL / st / f"{ex['id']}.png", GAL / st / f"{ex['id']}.pdf"
        (DOCS / "figures" / st).mkdir(parents=True, exist_ok=True)
        (DOCS / "thumbs" / st).mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, DOCS / "figures" / st / png.name)
        shutil.copy2(pdf, DOCS / "figures" / st / pdf.name)
        im = Image.open(png).convert("RGB")
        w, h = im.size
        th = im.copy()
        th.thumbnail((720, 720), Image.LANCZOS)
        th.save(DOCS / "thumbs" / st / f"{ex['id']}.jpg", quality=86, optimize=True)
        out[st] = {"png": f"figures/{st}/{png.name}", "pdf": f"figures/{st}/{pdf.name}",
                   "thumb": f"thumbs/{st}/{ex['id']}.jpg", "w": w, "h": h, "width_in": round(w / DPI, 2)}
    return out


# ---- native LaTeX demos -----------------------------------------------------------------------

def latex_demos(sty_text):
    demos = [{
        "id": "pgfplots",
        "title": "pgfplots in the house style",
        "text": "Native pgfplots axes take the same frame, ticks, fonts, series and colormap as the matplotlib "
                "figures. Use the styles scqfig, scqfig data, scqfig model and scqfig ref.",
        "preamble": ("\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.16}\n\\usepackage[field]{scqfig}   % or [folio]\n"
                     "\\usepackage{sansmath}   % field only: sans tick numbers"),
        "body": ("\\begin{tikzpicture}\n\\begin{axis}[scqfig, width=7.2cm, height=4.6cm,\n"
                 "    xlabel={free evolution ($\\mu$s)}, ylabel={$P_1$},\n"
                 "    xmin=0, xmax=10, ymin=0, ymax=1, legend pos=north east,\n"
                 "    tick label style={font=\\sansmath\\sffamily\\footnotesize},   % field only\n"
                 "    ]\n"
                 "  \\addplot+[scqfig data, samples=41, domain=0:10]\n"
                 "    {0.5+0.42*exp(-x/4)*cos(deg(2*pi*0.6*x))+0.02*sin(deg(37*x))};\n"
                 "  \\addlegendentry{data}\n"
                 "  \\addplot+[scqfig model, samples=300, domain=0:10]\n"
                 "    {0.5+0.42*exp(-x/4)*cos(deg(2*pi*0.6*x))};\n"
                 "  \\addlegendentry{fit}\n"
                 "\\end{axis}\n\\end{tikzpicture}"),
    }, {
        "id": "tiles",
        "title": "Circuit-element tiles in TikZ",
        "text": "The pale tiles that sit behind capacitors, junctions and inductors in the matplotlib circuits, as "
                "a TikZ style, so Hamiltonian terms in the text can sit on the same tiles.",
        "preamble": "\\usepackage{tikz}\n\\usepackage[field]{scqfig}   % or [folio]",
        "body": ("\\begin{tikzpicture}\n"
                 "  \\node[scq tile=capacitive] at (0,0)   {$4E_C\\hat n^2$};\n"
                 "  \\node[scq tile=nonlinear]  at (2.1,0) {$-E_J\\cos\\hat\\varphi$};\n"
                 "  \\node[scq tile=inductive]  at (4.3,0) {$\\hbar\\omega_r\\hat a^\\dagger\\hat a$};\n"
                 "  \\node[scq tile=coupling]   at (6.6,0) {$\\hbar g\\,\\hat n(\\hat a+\\hat a^\\dagger)$};\n"
                 "\\end{tikzpicture}"),
    }, {
        "id": "colours",
        "title": "Colour roles in running text",
        "text": "Every role of the active style has a plain name, so text and figures agree: scqAccent, scqSignal, "
                "scqBus, scqQubitA, scqLevel0 and the rest.",
        "preamble": "\\usepackage[field]{scqfig}   % or [folio]",
        "body": ("\\begin{minipage}{9cm}\\raggedright\n"
                 "Allowed transitions are \\textcolor{scqAccent}{\\textbf{orange}},\n"
                 "drive photons \\textcolor{scqSignal}{\\textbf{signal}}, the bus\n"
                 "\\textcolor{scqBus}{\\textbf{bus}}, and qubits\n"
                 "\\textcolor{scqQubitA}{\\textbf{Q1}}, \\textcolor{scqQubitB}{\\textbf{Q2}},\n"
                 "\\textcolor{scqQubitC}{\\textbf{Q3}}.\n"
                 "\\end{minipage}"),
    }]
    if "scqfig circuit" in sty_text:
        demos.insert(1, {
            "id": "quantikz",
            "title": "quantikz circuits in the house style",
            "text": "Gate boxes on the nonlinear tile, white meters, slices in the style's model colour with tilted "
                    "labels. quantikz's default red slice is replaced.",
            "preamble": "\\usepackage{tikz}\n\\usepackage{quantikz}\n\\usepackage[field]{scqfig}   % or [folio]",
            "body": ("\\begin{quantikz}[scqfig circuit]\n"
                     "  \\lstick{$|0\\rangle$} & \\gate{H} \\slice[label style=scq slice label]{$\\Psi_1$}\n"
                     "      & \\ctrl{1} \\slice[label style=scq slice label]{$\\Psi_2$}\n"
                     "      & \\ctrl{2} \\slice[label style=scq slice label]{$\\Psi_3$} & \\meter{} \\\\\n"
                     "  \\lstick{$|0\\rangle$} & \\qw & \\targ{} & \\qw & \\meter{} \\\\\n"
                     "  \\lstick{$|0\\rangle$} & \\qw & \\qw & \\targ{} & \\meter{}\n"
                     "\\end{quantikz}"),
        })
    out = []
    env = dict(os.environ, TEXINPUTS=f"{ROOT / 'latex'}:{os.environ.get('TEXINPUTS', '')}")
    (DOCS / "latex").mkdir(parents=True, exist_ok=True)
    for d in demos:
        d = dict(d, images={}, code={})
        for st in STYLES:
            pre = d["preamble"].replace("[field]", f"[{st}]")
            if st == "folio":
                pre = "\n".join(ln for ln in pre.splitlines() if "field only" not in ln).replace("% or [folio]", "% or [field]")
            font = (r"\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
                    if st == "field" else r"\usepackage[T1]{fontenc}\usepackage{stix2}")
            body = d["body"] if st == "field" else "\n".join(ln for ln in d["body"].splitlines()
                                                              if "field only" not in ln)
            d["code"][st] = f"{pre}\n\n% in the document\n{body}\n"
            doc = (f"\\documentclass[border=6pt]{{standalone}}\n{font}\n{pre}\n\\begin{{document}}\n{body}\n"
                   f"\\end{{document}}\n")
            with tempfile.TemporaryDirectory() as tmp:
                (pathlib.Path(tmp) / "d.tex").write_text(doc)
                r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "d.tex"], cwd=tmp,
                                   env=env, capture_output=True, text=True)
                if r.returncode:
                    print(f"latex demo {d['id']} ({st}) failed:\n{r.stdout[-1200:]}")
                    continue
                stem = DOCS / "latex" / f"{d['id']}_{st}"
                subprocess.run(["pdftocairo", "-png", "-r", "240", "-singlefile", f"{tmp}/d.pdf", str(stem)], check=True)
                shutil.copy2(f"{tmp}/d.pdf", f"{stem}.pdf")
            d["images"][st] = {"png": f"latex/{d['id']}_{st}.png", "pdf": f"latex/{d['id']}_{st}.pdf"}
        out.append(d)
    return out


# ---- palettes ---------------------------------------------------------------------------------

def palettes():
    roles = ("ink", "data", "model", "sim", "reference", "accent", "signal", "bus", "wash", "overlay")
    out = {"palette": PALETTE, "hues": list(HUES), "styles": {}}
    for name, t in THEMES.items():
        out["styles"][name] = {
            "roles": {r: getattr(t, r) for r in roles},
            "qubits": list(t.qubits), "levels": list(t.levels), "series": list(t.cycle),
            "tiles": dict(t.tiles), "chip": dict(t.chip),
            "cmap": [[p, c] for p, c in t.cmap_stops], "cmap_div": [[p, c] for p, c in t.cmap_div_stops],
            "notes": STYLE_NOTES[name],
        }
    return out


def readme_parts():
    """Why, Install and Quick start from README.md, as ordered text and code blocks, so the two never drift."""
    text = (ROOT / "README.md").read_text()
    fence = {"python": "py", "bash": "sh", "sh": "sh", "latex": "tex", "tex": "tex"}
    out = {}
    for key, head in (("why", "## Why"), ("install", "## Install"), ("quickstart", "## Quick start")):
        i = text.find(head)
        sec = text[i + len(head):text.find("\n## ", i + 1)] if i >= 0 else ""
        blocks = []
        for part in re.split(r"(```\w*\n.*?```)", sec, flags=re.S):
            m = re.match(r"```(\w*)\n(.*?)```", part, re.S)
            if m:
                blocks.append({"code": m.group(2), "lang": fence.get(m.group(1), "")})
            else:
                blocks += [{"text": t.strip()} for t in part.strip().split("\n\n") if t.strip()]
        out[key] = blocks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="render examples whose gallery files are missing")
    ap.add_argument("--no-latex", action="store_true", help="keep the LaTeX demos of the previous build")
    a = ap.parse_args()
    paths = sorted(EX.glob("[0-9][0-9]_*.py"))
    missing = [p.stem for p in paths if not all((GAL / st / f"{p.stem}.png").exists() for st in STYLES)]
    if missing and a.render:
        render_missing(missing)
    elif missing:
        raise SystemExit(f"missing gallery renders for {missing}; run examples/build_gallery.py or pass --render")
    for sub in ("figures", "thumbs"):
        shutil.rmtree(DOCS / sub, ignore_errors=True)
    examples = []
    for p in paths:
        ex = read_example(p)
        ex["images"] = copy_figures(ex)
        ex["page"] = ex["images"]["field"]["width_in"] > PAGE_WIDTH_IN
        ex["latex"] = latex_snippet(ex, ex["page"])
        examples.append(ex)
        print(f"  {ex['id']}")
    sty = (ROOT / "latex" / "scqfig.sty").read_text()
    old = DOCS / "data" / "site.json"
    demos = json.loads(old.read_text()).get("latex", []) if a.no_latex and old.exists() else latex_demos(sty)
    (DOCS / "downloads").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "latex" / "scqfig.sty", DOCS / "downloads" / "scqfig.sty")
    if (GAL / "gallery.pdf").exists():
        shutil.copy2(GAL / "gallery.pdf", DOCS / "downloads" / "scqfig-gallery.pdf")
    from scqfig import __version__
    stamp = hashlib.sha1(b"".join(f.read_bytes() for f in sorted((DOCS / "assets").glob("*.js")))).hexdigest()[:10]
    data = {"version": __version__, "build": stamp, "families": [{"name": n, "color": c, "text": t} for n, c, t in FAMILIES],
            "examples": examples, "latex": demos, "sty": sty, "readme": readme_parts(), **palettes()}
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "site.json").write_text(json.dumps(data, indent=1))
    (DOCS / ".nojekyll").write_text("")
    print(f"wrote {DOCS / 'data' / 'site.json'}: {len(examples)} examples, {len(demos)} LaTeX demos")


if __name__ == "__main__":
    main()
