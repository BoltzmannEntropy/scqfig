"""scqfig: publication figures for superconducting-qubit and chip papers, in two matched styles
(field and folio), with a companion LaTeX package that shares the palette.

    import scqfig as sq
    sq.use("folio")                       # or "field"
    fig, ax = sq.figure("column", aspect=0.7)
    sq.plots.t1_decay(ax, t, p)
    sq.save(fig, "figures/t1")
"""
from . import fit, physics, plots, schematics  # noqa: F401
from .core import (WIDTHS, assumed_kw, data_kw, demo_main, figure, log_minor, model_kw, note,  # noqa: F401
                   panel, pm, ref_kw, save, series, sim_kw, style, theme, use)
from .themes import FIELD, FOLIO, PALETTE, THEMES, lighten  # noqa: F401

__version__ = "0.1.0"
