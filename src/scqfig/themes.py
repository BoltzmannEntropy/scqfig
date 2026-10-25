"""The palette and the two styles, field and folio.

This module is the single source of truth for every colour, font and size. core.use() turns a Theme
into matplotlib settings, and scqfig.latex writes the same colours into latex/scqfig.sty, so a figure
and the document around it share one palette.

Colour rule: every colour is one of four hues (blue, magenta, purple, orange), a tint of one of them
(the hue mixed with white), or a grey. Nothing reads as green or red, and the four hues stay apart
under simulated protan, deutan and tritan vision. tests/test_palette.py enforces the rule on every
artist of every example in both styles.

field   Colour roles on a sans-serif face. Closed frame with ticks inward on all four sides, measured
        points filled black, fits and simulations blue, pale coloured tiles behind circuit elements, a
        magenta-to-purple sequential map.
folio   Ink first, for greyscale print. Serif face that matches a Times or STIX body, open L frame
        with outward ticks, open markers, series told apart by line style before colour, grey tiles.
"""
from dataclasses import dataclass

from matplotlib.colors import to_hex, to_rgb

PALETTE = {
    "ink": "#000000",
    "blue": "#1f5fbf",
    "magenta": "#c2378f",
    "purple": "#5e2a84",
    "orange": "#e69f00",
    "charcoal": "#3d3d3d",
    "grey": "#6f6f6f",
    "silver": "#9a9a9a",
    "light": "#d4d4d4",
    "tint": "#ececec",
}
HUES = ("blue", "magenta", "purple", "orange")


def lighten(color, f=0.5):
    """Mix a colour (hex, name or PALETTE key) with white: f = 0 keeps it, f = 1 gives white."""
    r, g, b = to_rgb(PALETTE.get(color, color))
    return to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


@dataclass(frozen=True, eq=False)
class Theme:
    name: str
    family: str              # "sans-serif" or "serif"
    fonts: tuple             # preference list within the family
    usetex_preamble: str     # LaTeX preamble when text is set by LaTeX (use(..., usetex=True))
    size: float              # tick-label size in printed points; labels are 0.5 pt larger
    lw: float                # data line width
    axes_lw: float
    box: bool                # closed frame with ticks on all four sides, else an open L frame
    ticks: str               # "in" or "out"
    cycle: tuple             # series colours
    dashes: tuple            # series line styles, same length as cycle
    markers: tuple           # series markers, same length as cycle
    ink: str
    data: str                # measured points
    data_filled: bool
    model: str               # fit and theory curves
    sim: str                 # simulated or designed values (open markers)
    reference: str           # guide lines, bands, secondary text
    assumed: str             # values taken from elsewhere (open markers)
    accent: str              # highlights, allowed transitions, bursts
    signal: str              # drive tones, photons, external fields
    bus: str                 # resonators, buses and feedlines in schematics
    drive: str               # drive and control lines (XY, Z, pump) in circuit schematics
    wash: str                # a uniform field over a region
    light: str
    tint: str
    levels: tuple            # level n = 0, 1, 2, ... in wells and ladders
    qubits: tuple            # one colour per qubit in chips, circuits and multi-qubit plots
    tiles: dict              # pale tiles behind circuit elements, by element kind
    chip: dict               # chip-drawing roles
    cmap: str                # sequential colormap
    cmap_stops: tuple        # (position, colour) stops of the sequential map
    cmap_div: str            # diverging colormap
    cmap_div_stops: tuple
    overlay: str             # model lines drawn over a heatmap (contrasts with cmap)
    letter: str              # panel-letter format
    letter_weight: str


P = PALETTE

FIELD = Theme(
    name="field",
    family="sans-serif",
    fonts=("HelveticaX", "Helvetica", "Arial", "Liberation Sans", "TeX Gyre Heros", "Nimbus Sans",
           "DejaVu Sans"),
    usetex_preamble=r"\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
                    r"\usepackage{sansmath}\sansmath",
    size=8.0,
    lw=1.1,
    axes_lw=0.7,
    box=True,
    ticks="in",
    cycle=(P["blue"], P["magenta"], P["orange"], P["purple"], P["grey"]),
    dashes=("-", "-", "-", "-", "-"),
    markers=("o", "s", "D", "^", "v"),
    ink=P["ink"],
    data=P["ink"],
    data_filled=True,
    model=P["blue"],
    sim=P["blue"],
    reference=P["grey"],
    assumed=P["silver"],
    accent=P["orange"],
    signal=P["magenta"],
    bus=P["purple"],
    drive="#1a1a1a",
    wash=lighten("magenta", 0.87),
    light=P["light"],
    tint=P["tint"],
    levels=(P["purple"], P["blue"], P["magenta"], P["orange"], P["grey"]),
    # The fourth qubit is grey, a parked or auxiliary element, so that it never matches the purple bus.
    qubits=(P["blue"], P["magenta"], P["orange"], "#8c8c8c", P["purple"]),
    tiles={"nonlinear": lighten("purple", 0.82), "capacitive": lighten("blue", 0.84),
           "inductive": lighten("orange", 0.76), "coupling": "#e6e6e6"},
    chip={"substrate": lighten("purple", 0.94), "edge": P["silver"], "pad": P["magenta"],
          "resonator": P["purple"], "feedline": "#1a1a1a", "drive": P["blue"], "flux": P["orange"],
          "coupler": "#6a6a6a", "junction": P["ink"], "ground": "#dcdcdc"},
    cmap="scq_field",
    cmap_stops=((0.0, lighten("magenta", 0.97)), (0.3, lighten("magenta", 0.62)), (0.58, P["magenta"]),
                (0.82, P["purple"]), (1.0, P["ink"])),
    cmap_div="scq_field_div",
    cmap_div_stops=((0.0, P["blue"]), (0.5, "#ffffff"), (1.0, P["magenta"])),
    overlay=P["blue"],
    letter="({})",
    letter_weight="bold",
)

FOLIO = Theme(
    name="folio",
    family="serif",
    fonts=("STIX Two Text", "STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"),
    usetex_preamble=r"\usepackage[T1]{fontenc}\usepackage{stix2}",
    size=8.0,
    lw=0.9,
    axes_lw=0.6,
    box=False,
    ticks="out",
    cycle=(P["ink"], P["blue"], P["grey"], P["orange"], P["purple"]),
    dashes=("-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5, 1, 1.5))),
    markers=("o", "s", "^", "D", "v"),
    ink=P["ink"],
    data=P["ink"],
    data_filled=False,
    model=P["ink"],
    sim=P["blue"],
    reference=P["grey"],
    assumed=P["silver"],
    accent=P["orange"],
    signal=P["purple"],
    bus=P["charcoal"],
    drive=P["ink"],
    wash=P["tint"],
    light=P["light"],
    tint=P["tint"],
    levels=(P["ink"], P["blue"], P["purple"], P["grey"], P["magenta"]),
    qubits=(P["blue"], P["orange"], P["purple"], P["grey"], P["ink"]),
    tiles={"nonlinear": "#e2e2e2", "capacitive": "#eeeeee", "inductive": "#f4f4f4",
           "coupling": "#e8e8e8"},
    chip={"substrate": "#f4f4f4", "edge": P["silver"], "pad": "#8c8c8c", "resonator": P["charcoal"],
          "feedline": P["ink"], "drive": P["blue"], "flux": P["orange"], "coupler": P["grey"],
          "junction": P["ink"], "ground": "#e2e2e2"},
    cmap="scq_folio",
    cmap_stops=((0.0, "#ffffff"), (0.4, lighten("blue", 0.55)), (0.75, P["blue"]), (1.0, P["ink"])),
    cmap_div="scq_folio_div",
    cmap_div_stops=((0.0, P["blue"]), (0.5, "#ffffff"), (1.0, P["orange"])),
    overlay=P["orange"],
    letter="({})",
    letter_weight="normal",
)

THEMES = {"field": FIELD, "folio": FOLIO}
