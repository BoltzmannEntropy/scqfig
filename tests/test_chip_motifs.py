"""Chip schematics, motifs and symbols added for the tutorial figures: the circular magnifier, claw,
mask-level SQUID, bowtie, 3D cavity plan, coupling-graph glyph, uniform-field wash, dc source, and the
per-wire colours and windows of QCircuit (which must leave existing circuits unchanged)."""
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.check import violations  # noqa: E402
from scqfig.schematics import QCircuit, chip, motifs, symbols  # noqa: E402

STYLES = ["field", "folio"]


@pytest.fixture(autouse=True)
def _field_style():
    with sq.style("field"):
        yield
    plt.close("all")


def _ax():
    fig, ax = plt.subplots()
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect("equal")
    return ax


def _xy(patch):
    """Vertices of a patch in data coordinates."""
    return patch.get_patch_transform().transform(patch.get_path().vertices)


def _new_patches(ax, before):
    return [p for p in ax.patches if id(p) not in before]


# ---- zoom_circle -----------------------------------------------------------------------------

def _square(ax, zoom):
    ax.add_patch(Rectangle((-5.0, -5.0), 10.0, 10.0, fc=sq.theme().qubits[0]))
    ax.plot([0.0, 5.0], [0.0, 0.0], color=sq.theme().ink)


def test_zoom_tangents_touch_both_circles():
    ax = _ax()
    target, rt, centre, r = (0.0, 0.0), 10.0, (300.0, 120.0), 80.0
    out = chip.zoom_circle(ax, target, rt, centre, r, _square)
    assert len(out["tangents"]) == 2
    for ln in out["tangents"]:
        p0, p1 = np.column_stack(ln.get_data())
        assert np.hypot(*(p0 - target)) == pytest.approx(rt)
        assert np.hypot(*(p1 - centre)) == pytest.approx(r)
        d = (p1 - p0) / np.hypot(*(p1 - p0))
        assert (p0 - target) @ d == pytest.approx(0, abs=1e-9)          # radius meets the line at a right angle
        assert (p1 - centre) @ d == pytest.approx(0, abs=1e-9)
    a, b = (np.column_stack(ln.get_data())[0] for ln in out["tangents"])
    assert not np.allclose(a, b)                                          # the two tangents differ


def test_zoom_magnify_maps_the_target_onto_the_disc_to_scale():
    ax = _ax()
    out = chip.zoom_circle(ax, (0.0, 0.0), 10.0, (300.0, 120.0), 80.0, _square, label="SQUID")
    assert out["scale"] == pytest.approx(8.0)
    (rect,) = [a for a in out["content"] if isinstance(a, Rectangle)]
    (line,) = [a for a in out["content"] if isinstance(a, Line2D)]
    T = line.get_transform() - ax.transData                                # chip units -> page units
    assert np.allclose(T.transform([(0.0, 0.0), (5.0, 0.0)]), [(300.0, 120.0), (340.0, 120.0)])
    corner = rect.get_transform().transform((0.0, 0.0))                   # unit-square corner = (-5, -5) in chip units
    assert np.allclose(corner, ax.transData.transform((260.0, 80.0)))
    disc = out["disc"].get_zorder()
    assert all(a.get_zorder() > disc and a.get_clip_path() is not None for a in out["content"])
    assert out["outline"].get_zorder() > max(a.get_zorder() for a in out["content"])
    assert out["label"].get_position()[1] < 120.0 - 80.0 and out["label"].get_text() == "SQUID"


def test_zoom_without_magnify_draws_in_page_units_and_label_above():
    ax = _ax()
    seen = {}

    def inset(ax, zoom):
        seen.update(zoom._asdict())
        ax.plot([zoom.center[0], zoom.center[0] + zoom.r], [zoom.center[1]] * 2)

    out = chip.zoom_circle(ax, (0.0, 0.0), 10.0, (300.0, 0.0), 80.0, inset, magnify=False, label="JJ",
                           label_side="above")
    assert seen["scale"] == pytest.approx(8.0) and seen["center"] == (300.0, 0.0)
    T = out["content"][0].get_transform() - ax.transData
    assert np.allclose(T.transform((300.0, 0.0)), (300.0, 0.0))
    assert out["label"].get_position()[1] > 80.0


@pytest.mark.parametrize("centre", [(0.0, 0.0), (50.0, 0.0), (89.0, 0.0), (90.0, 0.0)])
def test_zoom_rejects_a_target_inside_or_touching_the_disc(centre):
    with pytest.raises(ValueError):
        chip.zoom_circle(_ax(), (0.0, 0.0), 10.0, centre, 80.0, _square)


@pytest.mark.parametrize("rt,r", [(0.0, 80.0), (10.0, 0.0), (-1.0, 80.0)])
def test_zoom_rejects_zero_size_circles(rt, r):
    with pytest.raises(ValueError):
        chip.zoom_circle(_ax(), (0.0, 0.0), rt, (500.0, 0.0), r, _square)


def test_zoom_rejects_unknown_label_side_and_a_missing_callback():
    with pytest.raises(ValueError):
        chip.zoom_circle(_ax(), (0.0, 0.0), 10.0, (500.0, 0.0), 80.0, _square, label="x", label_side="left")
    with pytest.raises(TypeError):
        chip.zoom_circle(_ax(), (0.0, 0.0), 10.0, (500.0, 0.0), 80.0, None)


def test_zoom_with_an_empty_callback_still_draws_the_frame():
    out = chip.zoom_circle(_ax(), (0.0, 0.0), 10.0, (500.0, 0.0), 80.0, lambda ax, zoom: None)
    assert out["content"] == [] and out["outline"].get_radius() == 80.0 and out["label"] is None


# ---- claw ------------------------------------------------------------------------------------

def test_claw_wraps_the_arm_with_its_clearance():
    ax = _ax()
    out = chip.claw(ax, 10.0, 200.0, arm=24.0, clearance=20.0, depth=80.0, w=12.0, gap=8.0)
    assert np.allclose(out["tip"], (10.0, 200.0))
    assert np.allclose(out["attach"], (10.0, 226.0))                     # tip + clearance + w/2
    assert np.allclose(out["left"], (-28.0, 146.0)) and np.allclose(out["right"], (48.0, 146.0))
    metal = ax.patches[-1]                                                # drawn after its gap
    x = _xy(metal)[:, 0]
    inner = np.min(np.abs(x[x > 10.0] - 10.0))                            # inner edge of the right finger
    assert inner == pytest.approx(12.0 + 20.0)                            # arm half-width + clearance
    assert metal.get_facecolor() == to_rgba(sq.theme().chip["resonator"])


def test_claw_rotates_with_angle():
    out = chip.claw(_ax(), 0.0, 0.0, arm=20.0, clearance=10.0, depth=50.0, w=10.0, angle=-90.0)
    assert np.allclose(out["attach"], (15.0, 0.0), atol=1e-9)             # wraps an arm that points right


@pytest.mark.parametrize("kw", [dict(arm=0.0), dict(depth=0.0), dict(w=0.0), dict(clearance=-1.0), dict(gap=-1.0)])
def test_claw_rejects_zero_or_negative_geometry(kw):
    args = dict(arm=24.0, clearance=20.0, depth=80.0) | kw
    with pytest.raises(ValueError):
        chip.claw(_ax(), 0.0, 0.0, **args)


# ---- squid_detail ----------------------------------------------------------------------------

def test_squid_detail_bridges_its_span_with_junctions_to_scale():
    ax = _ax()
    before = {id(p) for p in ax.patches}
    out = chip.squid_detail(ax, 5.0, -182.0, span=24.0, loop=(16.0, 8.0), lead=1.5, jj=0.5)
    assert np.allclose(out["top"], (5.0, -170.0)) and np.allclose(out["bottom"], (5.0, -194.0))
    assert np.allclose(out["left_junction"], (5.0 - 7.25, -182.0))
    new = _new_patches(ax, before)
    assert len(new) == 8                                                  # 2 leads, 4 loop strips, 2 junctions
    jj = [p for p in new if p.get_facecolor() == to_rgba(sq.theme().chip["junction"])]
    assert len(jj) == 2 and all(np.ptp(_xy(p)[:, 0]) == pytest.approx(0.5) for p in jj)
    ys = np.concatenate([_xy(p)[:, 1] for p in new])
    assert ys.max() == pytest.approx(-170.0 + 1.5) and ys.min() == pytest.approx(-194.0 - 1.5)   # overlap


@pytest.mark.parametrize("kw", [dict(loop=(2.0, 8.0)), dict(loop=(16.0, 3.0)), dict(jj=0.0), dict(lead=0.0),
                                dict(span=4.0), dict(overlap=-1.0)])
def test_squid_detail_rejects_degenerate_geometry(kw):
    args = dict(span=24.0, loop=(16.0, 8.0), lead=1.5) | kw
    with pytest.raises(ValueError):
        chip.squid_detail(_ax(), 0.0, 0.0, **args)


# ---- bowtie ----------------------------------------------------------------------------------

def test_bowtie_fins_are_wedges_of_their_radius_and_opening():
    ax = _ax()
    before = {id(p) for p in ax.patches}
    out = chip.bowtie(ax, 0.0, 0.0, fin=110.0, opening=157.5, sep=12.0, gap=40.0)
    assert np.allclose(out["right"], (116.0, 0.0)) and np.allclose(out["left"], (-116.0, 0.0))
    assert out["top"][1] == pytest.approx(110.0 * np.sin(np.radians(78.75)))
    assert out["pocket_top"][1] == pytest.approx(out["top"][1] + 40.0)
    pocket, left, right = _new_patches(ax, before)
    P = _xy(right)[:-1]                                                   # drop the closing vertex
    assert np.allclose(P[0], (6.0, 0.0))                                  # apex
    assert np.allclose(np.hypot(P[1:, 0] - 6.0, P[1:, 1]), 110.0)         # arc of radius fin about the apex
    ang = np.degrees(np.arctan2(P[1:, 1], P[1:, 0] - 6.0))
    assert ang.min() == pytest.approx(-78.75) and ang.max() == pytest.approx(78.75)
    assert np.allclose(_xy(left), _xy(right) * (-1, 1))


def test_bowtie_rotates_vertically():
    out = chip.bowtie(_ax(), 0.0, 0.0, fin=100.0, sep=10.0, angle=90.0)
    assert np.allclose(out["right"], (0.0, 105.0), atol=1e-9)


@pytest.mark.parametrize("kw", [dict(fin=0.0), dict(sep=0.0), dict(opening=0.0), dict(opening=180.0),
                                dict(gap=-1.0)])
def test_bowtie_rejects_degenerate_geometry(kw):
    with pytest.raises(ValueError):
        chip.bowtie(_ax(), 0.0, 0.0, **kw)


# ---- cavity_plan -----------------------------------------------------------------------------

CHAMBERS = {"storage": (0.0, 0.0, 45e3, 40e3), "readout": (51e3, 0.0, 27e3, 40e3)}
CHIP = dict(rect=(41.5e3, 19.2e3, 13e3, 1.6e3), pad=(4e3, 0.9e3), sep=1.6e3)


def test_cavity_plan_draws_chambers_chip_and_port():
    ax = _ax()
    out = chip.cavity_plan(ax, CHAMBERS, trench=(41e3, 17.5e3, 14e3, 5e3), chip=CHIP,
                           port=dict(at=(70e3, 28e3), side="right", label="pin"))
    assert set(out["chambers"]) == {"storage", "readout"}
    b = out["block"]
    assert b.get_x() < 0 and b.get_x() + b.get_width() > 78e3                # the block holds both chambers
    assert np.allclose(out["chip"]["junction"], (48e3, 20e3))
    assert np.allclose(out["chip"]["pads"], [(48e3 - 2.8e3, 20e3), (48e3 + 2.8e3, 20e3)])
    assert out["port"]["wall"][0] == pytest.approx(b.get_x() + b.get_width()) and out["port"]["end"][0] > \
        out["port"]["wall"][0]
    texts = [t.get_text() for t in ax.texts]
    assert {"storage", "readout", "pin", "45 mm", "27 mm", "40 mm"} <= set(texts)
    assert texts.count("40 mm") == 1                                       # equal heights share one dimension
    u0, u1, v0, v1 = out["extent"]
    assert u1 >= out["port"]["end"][0] and v0 < b.get_y()


def test_cavity_plan_accepts_an_unnamed_list_without_dims():
    ax = _ax()
    out = chip.cavity_plan(ax, [(0.0, 0.0, 10e3, 8e3)], dims=False)
    assert list(out["chambers"]) == [0] and not ax.texts


@pytest.mark.parametrize("kw", [dict(chip=dict(CHIP, glue=1)), dict(port=dict(at=(1e3, 1e3), colour="x")),
                                dict(port=dict(at=(1e3, 1e3), side="front")), dict(port=dict(side="top")),
                                dict(chip=dict(pad=(1e3, 1e3)))])
def test_cavity_plan_rejects_unknown_or_missing_options(kw):
    with pytest.raises(ValueError):
        chip.cavity_plan(_ax(), CHAMBERS, **kw)


@pytest.mark.parametrize("chambers", [{}, [], {"a": (0.0, 0.0, 0.0, 5e3)}, {"a": (0.0, 0.0, 5e3)}])
def test_cavity_plan_rejects_missing_or_zero_size_chambers(chambers):
    with pytest.raises(ValueError):
        chip.cavity_plan(_ax(), chambers)


def test_cavity_plan_rejects_pads_that_overrun_the_chip():
    with pytest.raises(ValueError):
        chip.cavity_plan(_ax(), CHAMBERS, chip=dict(rect=(0.0, 0.0, 10e3, 1e3), pad=(4e3, 0.5e3), sep=3e3))


# ---- double_pad options ----------------------------------------------------------------------

def test_double_pad_defaults_keep_pocket_and_single_junction():
    ax = _ax()
    a = chip.double_pad(ax, 0.0, 0.0, pad=(100.0, 200.0), sep=40.0, gap=30.0)
    assert len(ax.patches) == 3 and len(ax.lines) == 3            # pocket, two pads; lead and a cross
    assert a["center"] == (0.0, 0.0) and a["top"] == pytest.approx((0.0, 130.0))


def test_double_pad_without_pocket_or_junction_draws_pads_only():
    ax = _ax()
    chip.double_pad(ax, 0.0, 0.0, pocket=False, junction=None)
    assert len(ax.patches) == 2 and not ax.lines
    assert all(to_hex(p.get_facecolor()) == to_hex(sq.theme().qubits[1]) for p in ax.patches)


@pytest.mark.parametrize("angle", [0.0, 90.0])
def test_double_pad_squid_leads_run_along_the_pad_axis(angle):
    ax = _ax()
    sep = 40.0
    chip.double_pad(ax, 10.0, 20.0, sep=sep, angle=angle, pocket=False, junction="squid")
    assert len(ax.lines) == 7                                      # two leads, a loop, two crosses of two strokes
    lead = np.column_stack(ax.lines[0].get_data())
    axis = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    d = lead[1] - lead[0]
    assert abs(d @ axis) == pytest.approx(np.hypot(*d))           # parallel to the pad axis
    ends = [np.column_stack(ln.get_data()) for ln in ax.lines[:2]]
    reach = max(abs((p - (10.0, 20.0)) @ axis) for e in ends for p in e)
    assert reach == pytest.approx(sep / 2)                         # leads meet the inner pad edges


def test_double_pad_rejects_unknown_junction():
    with pytest.raises(ValueError):
        chip.double_pad(_ax(), 0.0, 0.0, junction="jj")


# ---- coupling_graph --------------------------------------------------------------------------

def test_coupling_graph_draws_nodes_edges_dashed_pairs_and_a_hub():
    ax = _ax()
    t = sq.theme()
    pos = [(0, 0), (1, 0), (0, 1), (1, 1)]
    out = chip.coupling_graph(ax, pos, [(0, 1), (2, 3)], dashed=[(0, 2)], hub=(0.5, 0.5), r=0.2,
                              colors=list(t.qubits[:4]))
    assert len(out["nodes"]) == 4 and len(out["edges"]) == 2 and len(out["dashed"]) == 1 and len(out["links"]) == 4
    assert [n.get_facecolor() for n in out["nodes"]] == [to_rgba(c) for c in t.qubits[:4]]
    assert out["hub"].get_facecolor() == to_rgba(t.bus) and out["edges"][0].get_color() == t.chip["coupler"]
    assert all(n.get_zorder() > e.get_zorder() for n in out["nodes"] for e in out["edges"])


def test_coupling_graph_hub_color_recolours_hub_and_links_only():
    ax = _ax()
    t = sq.theme()
    out = chip.coupling_graph(ax, [(0, 0), (1, 0)], [(0, 1)], hub=(0.5, 0.5), r=0.1, hub_color=t.chip["coupler"])
    assert out["hub"].get_facecolor() == to_rgba(t.chip["coupler"])
    assert all(ln.get_color() == t.chip["coupler"] for ln in out["links"])
    assert out["nodes"][0].get_facecolor() == to_rgba(t.qubits[0])


def test_coupling_graph_single_node_without_edges():
    out = chip.coupling_graph(_ax(), [(0.0, 0.0)])
    assert len(out["nodes"]) == 1 and out["hub"] is None and out["nodes"][0].get_radius() == 0.4


@pytest.mark.parametrize("pos,edges,kw", [([], [], {}), ([(0, 0), (1, 0)], [(0, 2)], {}),
                                          ([(0, 0), (1, 0)], [(1, 1)], {}), ([(0, 0)], [], dict(r=0.0)),
                                          ([(0, 0), (1, 0)], [], dict(colors=["#1f5fbf"])),
                                          ([(0, 0, 0)], [], {})])
def test_coupling_graph_rejects_bad_input(pos, edges, kw):
    with pytest.raises(ValueError):
        chip.coupling_graph(_ax(), pos, edges, **kw)


# ---- wash and dc_source ----------------------------------------------------------------------

@pytest.mark.parametrize("style", STYLES)
def test_wash_is_a_flat_fill_in_the_wash_role(style):
    with sq.style(style):
        ax = _ax()
        p, text = motifs.wash(ax, 1.0, 2.0, 3.0, 4.0, label="uniform field", loc="upper right")
        assert isinstance(p, Rectangle) and p.get_facecolor() == to_rgba(sq.theme().wash)
        assert (p.get_x(), p.get_y(), p.get_width(), p.get_height()) == (1.0, 2.0, 3.0, 4.0)
        assert p.get_zorder() < 1 and text.get_color() == sq.theme().signal
        x, y = text.get_position()
        assert 1.0 < x < 4.0 and 2.0 < y < 6.0 and text.get_ha() == "right" and text.get_va() == "top"


def test_wash_rounded_and_without_label():
    p, text = motifs.wash(_ax(), 0.0, 0.0, 2.0, 1.0, rounded=True)
    assert isinstance(p, FancyBboxPatch) and text is None


@pytest.mark.parametrize("kw", [dict(w=0.0), dict(h=0.0), dict(w=-1.0), dict(loc="middle")])
def test_wash_rejects_zero_size_and_unknown_loc(kw):
    args = dict(x=0.0, y=0.0, w=1.0, h=1.0) | kw
    loc = args.pop("loc", "lower left")
    with pytest.raises(ValueError):
        motifs.wash(_ax(), args["x"], args["y"], args["w"], args["h"], label="x", loc=loc)


@pytest.mark.parametrize("up", [True, False])
def test_dc_source_puts_plus_towards_the_positive_terminal(up):
    ax = _ax()
    ring, (plus_h, plus_v, minus) = symbols.dc_source(ax, 0.0, 0.0, r=0.2, up=up)
    assert isinstance(ring, Circle) and ring.get_radius() == 0.2
    assert np.ptp(plus_v.get_xdata()) == 0 and np.ptp(plus_v.get_ydata()) > 0     # the vertical stroke of +
    assert (plus_h.get_ydata()[0] > 0) == up and (minus.get_ydata()[0] < 0) == up
    pts = np.concatenate([np.column_stack(s.get_data()) for s in (plus_h, plus_v, minus)])
    assert np.all(np.hypot(pts[:, 0], pts[:, 1]) < 0.2)                                     # inside the ring


def test_dc_source_rejects_zero_radius():
    with pytest.raises(ValueError):
        symbols.dc_source(_ax(), 0.0, 0.0, r=0.0)


# ---- QCircuit wire colours and windows -------------------------------------------------------

def _round(v):
    return (np.round(np.asarray(v, float), 4) + 0.0).tolist()


def _signature(ax):
    """Every artist QCircuit drew: kind, z-order, colours, widths and geometry."""
    rows = []
    for a in ax.get_children():
        if isinstance(a, Line2D):
            rows.append(("line", a.get_zorder(), to_hex(a.get_color(), keep_alpha=True), a.get_linewidth(),
                         str(a.get_linestyle()), _round(a.get_xdata()), _round(a.get_ydata())))
        elif isinstance(a, FancyArrowPatch):
            rows.append(("arrow", a.get_zorder(), to_hex(a.get_edgecolor(), keep_alpha=True), a.get_linewidth()))
        elif isinstance(a, Patch) and a is not ax.patch:
            rows.append(("patch", type(a).__name__, a.get_zorder(), to_hex(a.get_facecolor(), keep_alpha=True),
                         to_hex(a.get_edgecolor(), keep_alpha=True), a.get_linewidth(), _round(_xy(a))))
        elif isinstance(a, Text) and a.get_text():
            rows.append(("text", a.get_text(), _round(a.get_position()), to_hex(a.get_color()), a.get_fontsize(),
                         a.get_rotation(), a.get_ha(), a.get_va()))
    return rows


def _ghz(**kw):
    c = QCircuit(3, labels=[r"$|0\rangle$"] * 3, **kw)
    c.gate("H", 0)
    c.slice(r"$\Psi_1$")
    c.cnot(0, 1)
    c.slice(r"$\Psi_2$")
    c.cnot(0, 2)
    c.slice(r"$\Psi_3$")
    c.measure([0, 1, 2])
    c.slice(r"$\Psi_4$")
    return c


def _full(**kw):
    c = QCircuit(3, labels=[r"$|\psi\rangle$", r"$|0\rangle$", r"$|0\rangle$"], outputs=["a", None, "b"], **kw)
    c.gate("H", 1)
    c.cnot(1, 2)
    c.group([1, 2], cols=(0, 1), label="Bell pair")
    c.cnot(0, 1)
    c.gate("H", 0)
    c.swap(0, 1)
    c.cz(0, 2)
    c.controlled("U", [0, 1], 2)
    c.multi("V", [1, 2])
    c.gate(r"$R_x(\pi/2)$", 1)
    c.measure([0, 1])
    c.gate("X", 2, cond=1)
    c.gate("Z", 2, cond=0)
    return c


def _rows(c, style):
    with sq.style(style):
        fig, ax = plt.subplots(figsize=c.figsize())
        g = c.draw(ax)
        fig.canvas.draw()
        rows = _signature(ax)
        plt.close(fig)
    return rows, g


# digests of the artists that QCircuit drew before wire_colors and window existed
BASELINE = {("ghz", "field"): (45, "c59a2c1fbcc13885dc81c46a911768f27db3ebe327881b9d0509603d6b86eca3"),
            ("ghz", "folio"): (45, "fd57d2a0ae87fc4c3e9468fac498ddb905df14d5de61cbe5daac851a31dc35cc"),
            ("full", "field"): (65, "c6ea57f49d7436f1c2517d42b0092644454b9b608ac239e723c13055c4e96aec"),
            ("full", "folio"): (65, "a81e3fcf0b2f13da166e4b07b5c97d3ec2d5d854eb13c9596cc30a420531f7ac")}


@pytest.mark.parametrize("name,style", sorted(BASELINE))
def test_existing_circuits_render_the_same_artists(name, style):
    rows, _ = _rows({"ghz": _ghz, "full": _full}[name](), style)
    assert (len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()) == BASELINE[(name, style)]


@pytest.mark.parametrize("style", STYLES)
def test_all_ink_wire_colours_match_the_default(style):
    ink = sq.THEMES[style].ink
    assert _rows(_full(), style)[0] == _rows(_full(wire_colors=[ink] * 3), style)[0]


def test_wire_colours_reach_wires_labels_and_single_wire_glyphs():
    t = sq.theme()
    cols = list(t.qubits[:3])
    rows, g = _rows(_full(wire_colors=cols), "field")
    lines = [r for r in rows if r[0] == "line"]
    for q, c in enumerate(cols):                                             # each quantum wire from the left edge
        wire = [r for r in lines if r[6] == [-q, -q] and r[5][0] < 0]
        assert wire and wire[0][2] == to_hex(c, keep_alpha=True)
    texts = {r[1]: r[3] for r in rows if r[0] == "text"}
    assert texts[r"$|\psi\rangle$"] == to_hex(cols[0]) and texts["b"] == to_hex(cols[2])
    dots = [c for c in g["controls"]]
    assert {to_hex(d.get_facecolor()) for d in dots} <= {to_hex(c) for c in cols} | {"#ffffff"}
    vertical = [r for r in lines if len(set(r[5])) == 1 and abs(r[6][1] - r[6][0]) > 0.5 and r[4] == "-"
                and r[3] == 0.9]                                             # spans between wires, not glyph strokes
    assert vertical and all(r[2] == to_hex(t.ink, keep_alpha=True) for r in vertical)     # connectors stay ink


def test_window_shades_its_columns_behind_the_wires():
    c = _ghz()
    c.window((1, 2), label="exposure")
    with sq.style("field"):
        fig, ax = plt.subplots(figsize=c.figsize())
        g = c.draw(ax)
        plt.close(fig)
    (p,) = g["windows"]
    x = g["x"]
    assert p.get_x() < x[1] < x[2] < p.get_x() + p.get_width()
    assert p.get_x() > x[0] and p.get_x() + p.get_width() < x[3]
    assert p.get_facecolor() == to_rgba(sq.theme().wash) and p.get_zorder() < 1
    assert "exposure" in [t.get_text() for t in ax.texts]


def test_window_label_adds_headroom_and_fill_overrides_the_wash():
    plain, labelled = QCircuit(2), QCircuit(2)
    for c in (plain, labelled):
        c.gate("H", [0, 1])
    labelled.window(0, label="idle", fill=sq.theme().tint)
    assert labelled.figsize()[1] > plain.figsize()[1] and labelled.figsize()[0] == plain.figsize()[0]
    with sq.style("field"):
        fig, ax = plt.subplots()
        g = labelled.draw(ax)
        plt.close(fig)
    assert g["windows"][0].get_facecolor() == to_rgba(sq.theme().tint)


@pytest.mark.parametrize("cols", [(2, 1), (-1, 0), (0.5, 1), "a"])
def test_window_rejects_bad_columns(cols):
    with pytest.raises(ValueError):
        QCircuit(2).window(cols)


def test_window_past_the_last_column_raises_on_draw_and_bad_wire_colours_raise():
    c = QCircuit(2)
    c.gate("H", 0)
    c.window((0, 3))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        c.draw(ax)
    with pytest.raises(ValueError):
        QCircuit(3, wire_colors=["#000000"] * 2)


# ---- exports and the colour rule -------------------------------------------------------------

def test_new_exports_are_listed_and_callable():
    for name in ("zoom_circle", "claw", "squid_detail", "bowtie", "cavity_plan", "coupling_graph"):
        assert name in chip.__all__ and callable(getattr(chip, name))
    assert callable(motifs.wash) and callable(symbols.dc_source) and callable(QCircuit.window)


@pytest.mark.parametrize("style", STYLES)
def test_new_primitives_obey_the_colour_rule(style):
    with sq.style(style):
        t = sq.theme()
        fig, axes = sq.figure("page", aspect=0.5, ncols=3, layout=None)
        ax = axes[0]
        chip.substrate(ax, 800, 800, u0=-400, v0=-400)
        chip.ground_plane(ax, 800, 800, margin=0, u0=-400, v0=-400)
        q = chip.xmon(ax, 0, 0, arm=150, width=24, squid=False)
        chip.claw(ax, *q["top"], arm=24, clearance=20, depth=60)
        chip.squid_detail(ax, 0, -150 - 15, span=30, loop=(16, 8), lead=1.5)
        chip.bowtie(ax, 250, 250, fin=60, sep=8, gap=20)
        chip.zoom_circle(ax, (0, -165), 20, (300, -200), 90, lambda a, z: chip.squid_detail(
            a, 0, -165, span=30, loop=(16, 8), lead=1.5), label="SQUID")
        motifs.wash(axes[1], 0, 0, 100e3, 60e3, label="uniform field")
        chip.cavity_plan(axes[1], CHAMBERS, trench=(41e3, 17.5e3, 14e3, 5e3), chip=CHIP,
                         port=dict(at=(70e3, 28e3), side="right", label="pin"))
        chip.coupling_graph(axes[2], [(0, 0), (1, 0), (0.5, 1)], [(0, 1)], dashed=[(1, 2)], hub=(0.5, 0.4), r=0.1)
        symbols.dc_source(axes[2], 2.0, 0.0)
        c = QCircuit(2, wire_colors=list(t.qubits[:2]))
        c.gate("H", 0).cnot(0, 1).measure([0, 1]).window(1, label="exposure")
        c.draw(fig.add_axes([0.0, 0.0, 0.3, 0.3]))
        fig.canvas.draw()
        bad = violations(fig)
    assert not bad, [(type(a).__name__, col) for a, col in bad[:8]]
