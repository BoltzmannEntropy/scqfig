"""A generic literature-screening flow chart: reports pass a screening box, the exclusions branch off a trunk, a dotted arrow marks the citation search that adds reports to the full-text set, and the kept reports split into labelled categories (made-up counts)."""
import scqfig as sq

TITLE = "Screening flow"
FAMILY = "Schematics"


def build():
    nodes = {
        "seed": dict(xy=(0.0, 1.75), w=1.25, h=0.55, text="seed register\n120 reports"),
        "added": dict(xy=(0.0, 0.0), w=1.25, h=0.6, text="citation search\n18 added reports"),
        "screen": dict(xy=(1.6, 1.75), w=1.35, h=0.55, text="titles and abstracts\nscreened"),
        "kept": dict(xy=(3.45, 1.75), w=1.15, h=0.55, text="72 seed reports\nkept"),
        "full": dict(xy=(3.45, 0.0), w=1.15, h=0.6, text="full-text reading\n90 reports"),
    }
    edges = [("seed", "screen"), ("screen", "kept"), ("kept", "full"), ("added", "full"),
             ("seed", "added", dict(ls="dotted", label="reference lists"))]
    fans = [dict(src="screen", port="bottom", items=["30 kept as background", "12 out of scope", "6 duplicates"],
                 pitch=0.32, length=0.25),
            dict(src="full", port="right", items=["25 experiments", "20 device designs", "30 theory", "15 reviews"],
                 span=(2.15, 0.25), trunk=4.9, length=0.25, header="category")]
    fig, ax = sq.figure("page", height=2.55)
    sq.schematics.flow_diagram(ax, nodes, edges, fans=fans)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
