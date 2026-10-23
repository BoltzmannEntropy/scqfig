"""Schematic primitives and processor plots: chip geometry, the rigid oblique map, pulse envelopes,
the wiring layout, device maps, CDFs, and the colour rule on every primitive in both styles."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402

import scqfig as sq  # noqa: E402
from scqfig.check import violations  # noqa: E402
from scqfig.plots import processor  # noqa: E402
from scqfig.schematics import chip, motifs, pulses, wiring  # noqa: E402

STYLES = ["field", "folio"]


@pytest.fixture(autouse=True)
def _field_style():
    with sq.style("field"):           # registers the colormaps, as sq.figure() would
        yield
    plt.close("all")


# ---- chip geometry ---------------------------------------------------------------------------

def test_rounded_keeps_endpoints_and_fillets_a_right_angle():
    u, v = chip.rounded([(0, 0), (10, 0), (10, 10)], r=2)
    assert (u[0], v[0]) == (0, 0) and (u[-1], v[-1]) == (10, 10)
    arc = (u > 8 + 1e-9) & (v < 2 - 1e-9)
    assert arc.sum() >= 3
    assert np.allclose(np.hypot(u[arc] - 8, v[arc] - 2), 2)


def test_rounded_shrinks_the_radius_on_short_segments():
    u, v = chip.rounded([(0, 0), (2, 0), (2, 2)], r=5)
    assert u.min() >= 0 and v.max() <= 2
    assert np.all(np.diff(u) >= -1e-12) and np.all(np.diff(v) >= -1e-12)


def test_rounded_passes_straight_lines_through():
    u, v = chip.rounded([(0, 0), (5, 0), (9, 0)], r=1)
    assert np.allclose(v, 0) and u[0] == 0 and u[-1] == 9


def test_ribbon_keeps_its_width_on_a_rounded_path():
    u, v = chip.rounded([(0, 0), (20, 0), (20, 20), (0, 20)], r=5, n=24)
    P = np.column_stack([u, v])
    poly = chip._ribbon(P, 1.5)
    left, right = poly[:len(P)], poly[len(P):][::-1]
    assert np.allclose(np.hypot(*(left - right).T), 3.0, rtol=0.01)


def test_ribbon_mitres_a_sharp_corner_and_extends_open_ends():
    P = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    poly = chip._ribbon(P, 1.0, ext1=2.0)
    assert np.allclose(poly[1], (9.0, 1.0))          # inner corner of a 90 degree mitre
    assert np.allclose(poly[2], (9.0, 12.0))         # open end pushed out by ext1


@pytest.mark.parametrize("side,direction", [("left", (1, 0)), ("right", (-1, 0)), ("bottom", (0, 1)),
                                            ("top", (0, -1))])
def test_launcher_attach_point_faces_the_chip_interior(side, direction):
    fig, ax = plt.subplots()
    a = chip.launcher(ax, 100.0, 200.0, side=side, pad=200, taper=150)
    d = np.subtract(a, (100.0, 200.0))
    assert np.allclose(d / np.hypot(*d), direction)
    assert np.hypot(*d) == pytest.approx(250.0)


def test_meander_resonator_runs_from_the_elbow_to_the_requested_end():
    fig, ax = plt.subplots()
    P = chip.meander_resonator(ax, 500.0, 1000.0, 300.0, couple=200, width=300, turns=5)
    assert np.allclose(P[0], (300.0, 1000.0)) and np.allclose(P[-1], (500.0, 300.0))
    assert P[:, 1].max() <= 1000.0 + 1e-9 and P[:, 1].min() >= 300.0 - 1e-9
    assert P[:, 0].min() >= 300.0 - 1e-9 and P[:, 0].max() <= 650.0 + 1e-9


def test_meander_resonator_rejects_a_span_too_short_for_its_turns():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        chip.meander_resonator(ax, 0.0, 100.0, 60.0, turns=8)


def test_xmon_anchors_sit_on_its_arm_tips():
    fig, ax = plt.subplots()
    a = chip.xmon(ax, 50.0, 80.0, arm=100, width=20)
    assert np.allclose(a["top"], (50, 180)) and np.allclose(a["left"], (-50, 80))
    assert a["squid"][1] < a["bottom"][1] and np.isclose(a["squid"][0], 50)
    r = chip.xmon(ax, 0.0, 0.0, arm=100, width=20, angle=90)
    assert np.allclose(r["right"], (0, 100), atol=1e-9)


def test_tunable_coupler_and_double_pad_return_anchors():
    fig, ax = plt.subplots()
    c = chip.tunable_coupler(ax, 0.0, 0.0, length=400, stem=100)
    assert np.allclose(c["left"], (-200, 0)) and np.allclose(c["right"], (200, 0))
    assert c["squid"][1] < -100
    d = chip.double_pad(ax, 0.0, 0.0, pad=(100, 200), sep=40)
    assert np.allclose(d["junction"], (0, 0)) and d["right"][0] == pytest.approx(120)


def test_airbridges_are_spaced_along_the_path():
    fig, ax = plt.subplots()
    n = chip.airbridges(ax, np.array([(0.0, 0.0), (1000.0, 0.0)]), 250)
    assert n == 4


def test_scale_bar_spans_its_length():
    fig, ax = plt.subplots()
    line = chip.scale_bar(ax, 10.0, 5.0, 500)
    x, y = line.get_data()
    assert np.ptp(x) == pytest.approx(500) and np.ptp(y) == 0


def test_coupler_chip_laid_obliquely_is_one_rigid_affine_map():
    fig, ax = plt.subplots()
    ob = motifs.Oblique(x0=100, y0=50, shear=0.45, squash=0.5)
    before = motifs.artists(ax)
    anchors = chip.coupler_chip(ax)
    motifs.onto_plane(ax, ob, before)
    T = ob.transform(ax)
    fig.canvas.draw()
    for a in ax.patches + ax.lines + ax.texts:
        assert np.allclose(Artist.get_transform(a).get_matrix(), T.get_matrix())
    for txt in ax.texts:
        assert txt.get_rotation() == 0
    q = anchors["Q1"]["center"]
    expect = ax.transData.transform(ob(*q))
    assert np.allclose(T.transform(q), expect)


def test_cross_section_marks_the_three_interfaces():
    fig, ax = plt.subplots()
    out = chip.cross_section(ax, [("Si", 525e3, "substrate"), ("Ta", 200, "metal")])
    assert set(out["interfaces"]) == {"MA", "MS", "SA"}
    assert out["metal_top"] > out["surface"]


def test_cross_section_needs_a_substrate_and_a_metal():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        chip.cross_section(ax, [("Ta", 200, "metal")])
    with pytest.raises(ValueError):
        chip.cross_section(ax, [("Si", 525e3, "substrate")])


# ---- pulses ----------------------------------------------------------------------------------

def test_gaussian_starts_and_ends_at_zero_and_peaks_at_its_amplitude():
    t = np.linspace(0, 40, 4001)
    g = pulses.gaussian_pulse(t, 20.0, 5.0, amp=0.8)
    assert g.max() == pytest.approx(0.8, abs=1e-6)
    assert abs(g[np.argmin(abs(t - 10.0))]) < 1e-6 and abs(g[np.argmin(abs(t - 30.0))]) < 1e-6
    assert np.all(g[t < 10] == 0) and np.all(g[t > 30] == 0)


def test_drag_quadrature_is_minus_beta_times_the_derivative():
    t = np.linspace(0, 40, 8001)
    I, Q = pulses.drag_pulse(t, 20.0, 5.0, amp=1.0, beta=2.0)
    dI = np.gradient(I, t)
    inside = (t > 11) & (t < 29)
    assert np.allclose(Q[inside], -2.0 * dI[inside], atol=1e-4)
    assert Q[np.argmin(abs(t - 20))] == pytest.approx(0, abs=1e-9)


def test_square_pulse_rise_is_the_10_to_90_percent_time():
    t = np.linspace(0, 100, 200001)
    p = pulses.square_pulse(t, 20.0, 50.0, amp=1.0, rise=4.0)
    t10, t90 = t[np.argmax(p >= 0.1)], t[np.argmax(p >= 0.9)]
    assert t90 - t10 == pytest.approx(4.0, abs=1e-3)
    assert p[np.argmin(abs(t - 45))] == pytest.approx(1.0, abs=1e-9)


def test_pulse_sequence_maps_time_through_the_cut():
    fig, ax = plt.subplots()
    seq = [dict(row="XY", kind="drag", t0=15, sigma=4, label=r"$\pi$"),
           dict(row="Z", kind="square", t0=40, length=40, amp=-0.6, rise=2),
           dict(row="RO", kind="readout", t0=90, length=900, window=(120, 980))]
    out = pulses.pulse_sequence(ax, ["XY", "Z", "RO"], seq, sync=[30, 90], cut=(300, 800))
    assert list(out["rows"]) == ["XY", "Z", "RO"]
    assert out["rows"]["XY"] > out["rows"]["Z"] > out["rows"]["RO"]
    m = out["map"]
    assert m(100) == pytest.approx(100) and m(900) < 900 and m(900) > m(300)
    assert np.isnan(m(500))


def test_pulse_sequence_rejects_unknown_rows_and_kinds():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pulses.pulse_sequence(ax, ["XY"], [dict(row="Z", kind="drag", t0=10, sigma=3)])
    with pytest.raises(ValueError):
        pulses.pulse_sequence(ax, ["XY"], [dict(row="XY", kind="sawtooth", t0=10)])


# ---- wiring ----------------------------------------------------------------------------------

def test_cryostat_wiring_puts_every_part_inside_its_stage_band():
    fig, ax = plt.subplots()
    out = wiring.cryostat_wiring(ax)
    assert [s for s in out["stages"]] == ["300 K", "50 K", "4 K", "still", "cold plate", "MXC"]
    assert len(out["columns"]) == 4
    assert out["parts"]
    for line, stage, kind, x, y in out["parts"]:
        lo, hi = out["stages"][stage]
        assert lo < y < hi, (line, stage, kind)
        assert x == out["columns"][line]
    kinds = {k for _, _, k, _, _ in out["parts"]}
    assert {"att", "lp", "ir", "dc", "iso", "twpa", "hemt", "amp"} <= kinds


def test_cryostat_wiring_rejects_unknown_parts_and_stages():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        wiring.cryostat_wiring(ax, [dict(name="x", role="drive", parts={"4 K": [("laser",)]})])
    with pytest.raises(ValueError):
        wiring.cryostat_wiring(ax, [dict(name="x", role="drive", parts={"1 K pot": [("att", 10)]})])


# ---- processor -------------------------------------------------------------------------------

def _grid():
    pos = [(c, -r) for r in range(2) for c in range(3)]
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    return pos, edges


def test_device_map_colours_nodes_and_edges_by_their_metrics():
    fig, ax = plt.subplots()
    pos, edges = _grid()
    T1 = np.array([50, 60, 70, 80, 90, 100.0])
    err = np.linspace(0.4, 1.6, len(edges))
    out = processor.device_map(ax, pos, edges, T1, err)
    assert np.allclose(out["nodes"].get_array(), T1)
    assert np.allclose(out["edges"].get_array(), err)
    assert out["node_cbar"] is not None and out["edge_cbar"] is not None
    assert [t.get_text() for t in out["values"]] == ["50", "60", "70", "80", "90", "100"]


def test_device_map_rejects_mismatched_lengths():
    fig, ax = plt.subplots()
    pos, edges = _grid()
    with pytest.raises(ValueError):
        processor.device_map(ax, pos, edges, np.ones(5), np.ones(len(edges)))
    with pytest.raises(ValueError):
        processor.device_map(ax, pos, edges, np.ones(6), np.ones(3))


def test_cdf_returns_and_marks_the_medians():
    fig, ax = plt.subplots()
    rng = np.random.default_rng(1)
    data = {"1Q": 10 ** rng.normal(-3, 0.2, 40), "2Q": 10 ** rng.normal(-2.2, 0.2, 30)}
    med = processor.cdf(ax, data)
    assert med == {k: pytest.approx(np.median(v)) for k, v in data.items()}
    assert ax.get_xscale() == "log"
    with pytest.raises(ValueError):
        processor.cdf(ax, {"bad": [1e-3, -1e-3]})


# ---- exports and the colour rule -------------------------------------------------------------

def test_every_export_is_callable():
    for mod in (chip, pulses, wiring, processor):
        assert mod.__all__
        for name in mod.__all__:
            assert callable(getattr(mod, name)), f"{mod.__name__}.{name}"


def _draw_everything():
    fig, axes = sq.figure("page", aspect=0.9, nrows=2, ncols=3, layout=None)
    ax = axes[0, 0]
    chip.substrate(ax, 3600, 2400)
    chip.ground_plane(ax, 3600, 2400)
    chip.coupler_chip(ax)
    chip.double_pad(ax, 3000, 600)
    chip.scale_bar(ax, 2800, -150, 500)
    chip.cross_section(axes[0, 1], [("Si", 525e3, "substrate"), ("Ta", 200, "metal")])
    wiring.cryostat_wiring(axes[0, 2])
    pulses.pulse_sequence(axes[1, 0], ["XY", "Z", "RO"],
                          [dict(row="XY", kind="gauss", t0=10, sigma=3, label=r"$\pi$"),
                           dict(row="XY", kind="drag", t0=30, sigma=3),
                           dict(row="Z", kind="square", t0=45, length=30, amp=0.5, rise=2),
                           dict(row="RO", kind="readout", t0=80, length=200, window=(90, 270))], sync=[40])
    pos, edges = _grid()
    processor.device_map(axes[1, 1], pos, edges, np.arange(6) * 10 + 40.0, np.linspace(0.3, 1.2, 7),
                         names=[f"Q{i}" for i in range(6)])
    processor.cdf(axes[1, 2], {"a": [1e-3, 2e-3, 3e-3], "b": [1e-2, 2e-2, 4e-2]})
    return fig


@pytest.mark.parametrize("style", STYLES)
def test_every_primitive_obeys_the_colour_rule(style):
    with sq.style(style):
        fig = _draw_everything()
        fig.canvas.draw()
        bad = violations(fig)
    assert not bad, [(type(a).__name__, c) for a, c in bad[:8]]
