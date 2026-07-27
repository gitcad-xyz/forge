"""K3.6b — imported geometry that does not close (#135).

The anti-Parasolid bet, kept at the border: a STEP file whose CLOSED_SHELL
lies (missing face, torn seam) must never silently become a solid with a
confident volume. The importer audits closure BEFORE the orientation flip
(the flip's sign is origin-dependent on an open shell — meaningless), and
refuses with a gap report in user millimetres, or — only as explicit,
recorded intent — heals by exact vertex merge with a certificate. It never
invents geometry: a vertex merge cannot create a face, so a missing wall
always refuses.

Broken files are authored HERE, surgically, from forge's own writer output —
no correct exporter can produce them, which is the point.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from forgekernel.brep import (NonClosedShellError, Polygon, SnapClusterError,
                              Solid, boundary_gap_report,
                              open_boundary_segments, snap_vertices)
from forgekernel.stepio import read_step_planar_solid, write_step_planar_solid

F = Fraction


# -- authored broken files ----------------------------------------------------

def _box_step(dx=10, dy=20, dz=30, origin=None):
    s = Solid.box(dx, dy, dz)
    if origin is not None:
        s = s.translated(tuple(F(c) for c in origin))
    return write_step_planar_solid(s)


def _drop_first_face(text):
    """Remove the first face (the bottom cap) from the CLOSED_SHELL."""
    m = re.search(r"CLOSED_SHELL\('',\(([^)]*)\)\)", text)
    refs = m.group(1).split(",")
    return text.replace(f"CLOSED_SHELL('',({','.join(refs)}))",
                        f"CLOSED_SHELL('',({','.join(refs[1:])}))")


def _tear_at_corner(text):
    """Give one face's use of vertex (10,0,0) a private chain displaced to
    (10.003,0,0) — a 3 µm tear along otherwise-shared edges."""
    cp = re.search(r"#(\d+) = CARTESIAN_POINT\('',\(10\.,0\.,0\.\)\)", text).group(1)
    vp = re.search(r"#(\d+) = VERTEX_POINT\('',#" + cp + r"\)", text).group(1)
    ec = re.search(r"#(\d+) = EDGE_CURVE\('',#" + vp + r",#(\d+),#(\d+),\.T\.\)", text)
    ec_id, vb_id, crv_id = ec.group(1), ec.group(2), ec.group(3)
    maxid = max(int(x) for x in re.findall(r"#(\d+) =", text))
    add = (f"#{maxid + 1} = CARTESIAN_POINT('',(10.003,0.,0.));\n"
           f"#{maxid + 2} = VERTEX_POINT('',#{maxid + 1});\n"
           f"#{maxid + 3} = EDGE_CURVE('',#{maxid + 2},#{vb_id},#{crv_id},.T.);\n")
    oe = re.search(r"#(\d+) = ORIENTED_EDGE\('',\*,\*,#" + ec_id + r",\.T\.\)", text).group(0)
    out = text.replace(oe, oe.replace(f"#{ec_id}", f"#{maxid + 3}"), 1)
    return out.replace("ENDSEC;\nEND-ISO-10303-21;",
                       add + "ENDSEC;\nEND-ISO-10303-21;")


def _bend_a_corner(text):
    """Move one shared corner 3 µm out of plane: edges stay shared (the shell
    still closes) but three quads become exactly non-planar."""
    return text.replace("CARTESIAN_POINT('',(10.,20.,30.))",
                        "CARTESIAN_POINT('',(10.,20.,30.003))")


# -- stage 1: audit at the border ---------------------------------------------

def test_missing_face_refuses_with_mm_gap_report():
    gap = _drop_first_face(_box_step())
    with pytest.raises(NonClosedShellError) as ei:
        read_step_planar_solid(gap)
    rep = ei.value.report
    # the bottom cap of a 10x20 base: its 4 edges, 60 mm of open perimeter,
    # one closed chain (the rim of the absent face), no opposing chain
    assert rep["open_edges"] == 4
    assert rep["open_perimeter_mm"] == pytest.approx(60.0)
    assert rep["chains"] == 1
    assert rep["max_gap_mm"] is None
    # the report speaks millimetres, not carrier-line parameters
    assert "line-dir" not in str(ei.value)
    lengths = sorted(s["length_mm"] for s in rep["segments_mm"])
    assert lengths == [pytest.approx(10.0), pytest.approx(10.0),
                       pytest.approx(20.0), pytest.approx(20.0)]


def test_open_shell_far_from_origin_refuses_instead_of_flipping():
    # The old path oriented by volume sign BEFORE checking closure; on an
    # open shell that sign is origin-dependent, and this exact file used to
    # come back with every face flipped. Now it refuses like any other gap.
    gap = _drop_first_face(_box_step(origin=(-100, -100, -100)))
    with pytest.raises(NonClosedShellError):
        read_step_planar_solid(gap)


def test_closed_box_still_imports_and_orients():
    s = read_step_planar_solid(_box_step())
    assert s.volume() == 6000
    assert s.watertight_violations() == []


def test_multi_solid_reports_the_dropped_rest():
    text = _box_step()
    # duplicate the whole DATA section's solid as a second MANIFOLD_SOLID_BREP
    m = re.search(r"#(\d+) = MANIFOLD_SOLID_BREP\('[^']*',#(\d+)\)", text)
    maxid = max(int(x) for x in re.findall(r"#(\d+) =", text))
    text = text.replace(
        "ENDSEC;\nEND-ISO-10303-21;",
        f"#{maxid + 1} = MANIFOLD_SOLID_BREP('second',#{m.group(2)});\n"
        "ENDSEC;\nEND-ISO-10303-21;")
    report: dict = {}
    s = read_step_planar_solid(text, report=report)
    assert s.volume() == 6000
    assert len(report["dropped"]) == 1
    assert "MANIFOLD_SOLID_BREP" in report["dropped"][0]


def test_bent_quad_refuses_by_name():
    # the reader used to accept a loop that is not exactly planar and hand it
    # to Polygon, which silently planes it through the first three vertices
    bent = _bend_a_corner(_box_step())
    with pytest.raises(ValueError, match="non-planar"):
        read_step_planar_solid(bent)


def test_export_refuses_to_emit_closed_shell_over_an_open_shell():
    open_solid = Solid(Solid.box(10, 20, 30).polys[1:])
    with pytest.raises(NonClosedShellError):
        write_step_planar_solid(open_solid)


# -- stage 2: certified heal (opt-in, exact, never face invention) ------------

def test_torn_shell_refuses_without_heal():
    torn = _tear_at_corner(_box_step())
    with pytest.raises(NonClosedShellError) as ei:
        read_step_planar_solid(torn)
    rep = ei.value.report
    assert rep["open_edges"] >= 2
    assert rep["open_perimeter_mm"] > 0


def test_torn_shell_heals_with_certificate():
    torn = _tear_at_corner(_box_step())
    report: dict = {}
    s = read_step_planar_solid(torn, heal_tolerance="0.01", report=report)
    assert s.volume() == 6000                      # exactly — Fraction == int
    assert s.watertight_violations() == []
    rec = report["healed"]
    assert rec["moved"] == 1
    # the representative is the lexicographic min: (10,0,0) < (10.003,0,0)
    assert rec["max_move_sq"] == F(9, 1000000)     # (3/1000)^2, exact
    assert rec["max_move_mm"] == pytest.approx(0.003)
    # |dV| <= sum(affected face areas) * max_move: the corner touches the
    # 10x20 bottom, the 10x30 front and the 20x30 right walls — measured
    # PRE-heal, so the torn face contributes its true (displaced) area,
    # a hair over nominal 1100
    assert rec["affected_area_mm2"] == pytest.approx(1100.0, abs=0.1)
    assert rec["volume_change_bound_mm3"] == pytest.approx(
        rec["affected_area_mm2"] * 0.003)
    assert 3.29 < rec["volume_change_bound_mm3"] < 3.31


def test_heal_cannot_close_a_missing_face():
    # vertex merge never invents geometry: a missing wall still refuses
    gap = _drop_first_face(_box_step())
    with pytest.raises(NonClosedShellError):
        read_step_planar_solid(gap, heal_tolerance="0.01")


def test_snap_refuses_a_transitive_chain():
    # 0 / 0.008 / 0.016 with tol 0.01: pairwise links (0,1),(1,2) chain into
    # one cluster of diameter 0.016 = 1.6*tol — merging it would move a
    # vertex farther than the promise. Refuse, naming the chain.
    tri = [Polygon([(F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(0), F(1), F(0))], "a"),
           Polygon([(F("0.008"), F(0), F(0)), (F(1), F(1), F(0)), (F(0), F(1), F(1))], "b"),
           Polygon([(F("0.016"), F(0), F(0)), (F(1), F(0), F(1)), (F(0), F(0), F(1))], "c")]
    with pytest.raises(SnapClusterError):
        snap_vertices(tri, F("0.01"))


def test_snap_drops_a_collapsed_face_and_says_so():
    # a sliver triangle whose two near-coincident vertices merge has 2 distinct
    # vertices left: it must be DROPPED and REPORTED, not raise Polygon's raw
    # "collinear points" ValueError and not vanish silently
    polys = [Polygon([(F(5), F(5), F(0)), (F(0), F(0), F(0)),
                      (F("0.003"), F(0), F(0)), (F(5), F(0), F(0))], "sliver.holder"),
             Polygon([(F(0), F(0), F(0)), (F("0.003"), F(0), F(0)),
                      (F(2), F(1), F(0))], "sliver")]
    healed, rec = snap_vertices(polys, F("0.01"))
    # the quad dedupes to a valid triangle and survives; the sliver triangle
    # is left with 2 distinct vertices and is dropped, by name
    assert [p.source for p in healed] == ["sliver.holder"]
    assert len(healed[0].verts) == 3
    assert rec["dropped_faces"] == ["sliver"]
    polys2 = [Polygon([(F(0), F(0), F(0)), (F("0.003"), F(0), F(0)),
                       (F(5), F(5), F(0))], "sliver.tri")]
    healed2, rec2 = snap_vertices(polys2, F("0.01"))
    assert all(p.source != "sliver.tri" for p in healed2)
    assert rec2["dropped_faces"] == ["sliver.tri"]


# -- the gap-report primitives are exact and readable -------------------------

def test_open_boundary_segments_are_exact_3d_points():
    polys = Solid.box(10, 20, 30).polys[1:]         # bottom cap removed
    segs = open_boundary_segments(polys)
    assert len(segs) == 4
    pts = {v for s in segs for v in (s["a"], s["b"])}
    assert pts == {(0, 0, 0), (10, 0, 0), (10, 20, 0), (0, 20, 0)}
    for s in segs:
        assert all(isinstance(c, Fraction) or isinstance(c, int)
                   for v in (s["a"], s["b"]) for c in v)
    rep = boundary_gap_report(segs)
    assert rep["open_perimeter_mm"] == pytest.approx(60.0)


def test_gap_report_measures_the_crack_width_between_chains():
    # two opposing open chains 0.003 apart: the report's gap width answers
    # "how wide is the crack", separate from "how long is it"
    torn = _tear_at_corner(_box_step())
    with pytest.raises(NonClosedShellError) as ei:
        read_step_planar_solid(torn)
    rep = ei.value.report
    if rep["chains"] >= 2:
        assert rep["max_gap_mm"] == pytest.approx(0.003, rel=0.5)
