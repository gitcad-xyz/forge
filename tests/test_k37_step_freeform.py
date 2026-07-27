"""K3.7 — freeform STEP topology import: ADVANCED_FACE → TrimmedShell.

The planar walk refused every non-PLANE surface with "arrives at K3.7".
This is the arrival: a solid whose faces are B-spline surfaces (planar
faces lift to bilinear patches) imports as a :class:`TrimmedShell` —
pcurve chains (SURFACE_CURVE → PCURVE → DEFINITIONAL_REPRESENTATION)
carry each face's exact parameter-space trim, one TrimVertex is minted
per STEP VERTEX_POINT entity, curved edges subdivide PER EDGE ENTITY
(shared breakpoints — a per-face walk mints T-junctions and the pairing
audit refuses), closure is established BEFORE any orientation decision
(the #135 order), and global orientation comes from the certified volume
sign, audit-backed rather than flag-trusted.

Tiers (ADR-0019, never a bare float):

* EXACT (width-0 bracket): polynomial surfaces whose trim pcurves are all
  degree 1 in (u, v) — the certified machinery degenerates to exact via
  the span-safe ``trimmed_patch_flux`` (the interior-knot defect's fix).
* CERTIFIED: curved pcurves bracket through de Casteljau control-hull
  strip boxes (enclosure property — the hull CONTAINS the curve, so the
  bracket provably contains the truth); one-signed rational surfaces go
  through the reciprocal rule.
* REFUSED by name: analytic surfaces in a freeform shell, sign-varying
  weights, edges without pcurves on a B-spline face, VERTEX_LOOP bounds,
  an edge used twice by one face (periodic seam).

Round-trip acceptance: a loft's exact Bézier skin exports via
``write_step_patch_solid`` (seam topology proven by exact control-point
equality), reimports, audits, and its width-0 volume bracket IS the
loft's exact rational volume.
"""

from __future__ import annotations

import re
from fractions import Fraction as F

import pytest

from forgekernel.brep import NonClosedShellError, Solid
from forgekernel.bsolid import PatchSolid, box_patches
from forgekernel.loft import LoftSolid, _derive_seams
from forgekernel.quadric import Cyl
from forgekernel.stepio import (read_step_freeform_solid, read_step_solid,
                                write_step_patch_solid,
                                write_step_planar_solid)
from forgekernel.trimshell import TrimmedShell


def _sq(h):
    return [(-h, -h), (h, -h), (h, h), (-h, h)]


# half-widths 1,2,1 at z = 0,3,6: every Bernstein control value of the
# skin is a terminating rational, so the STEP decimals are LOSSLESS and
# the reimported volume can equal the loft's exact volume, not merely
# approximate it
LOFT = LoftSolid([(_sq(1), 0), (_sq(2), 3), (_sq(1), 6)])


def _loft_step() -> str:
    return write_step_patch_solid(LOFT.to_patches(), name="loft_skin")


# -- round trip: the acceptance test ------------------------------------------


def test_loft_skin_round_trips_to_exact_volume() -> None:
    shell = read_step_freeform_solid(_loft_step())
    assert isinstance(shell, TrimmedShell)
    vol = shell.volume(depth=3)
    v = LOFT.volume()
    assert vol.lo == vol.hi == v          # width-0: the exact tier
    assert v == LOFT.to_patches().volume()
    report = shell.audit(depth=3)
    assert report["faces"] == 10          # 8 walls + 2 caps
    for comp in report["vector_area"]:
        assert comp.lo <= 0 <= comp.hi


def test_bilinear_box_round_trips_width0() -> None:
    patches = box_patches(2, 3, 4)
    ps = PatchSolid(patches, seams=_derive_seams(patches))
    shell = read_step_freeform_solid(write_step_patch_solid(ps))
    vol = shell.volume(depth=3)
    assert vol.lo == vol.hi == 24


def test_router_freeform_to_trimmedshell_planar_to_solid() -> None:
    assert isinstance(read_step_solid(_loft_step()), TrimmedShell)
    s = read_step_solid(write_step_planar_solid(Solid.box(2, 3, 4)))
    assert isinstance(s, Solid)
    assert float(s.volume()) == 24.0


# -- certified tier: a genuinely curved trim (parabolic-roof solid) ------------
#
# The solid under z = 2 + 2x − x²/2 over [0,4]² (roof a degree-(2,1)
# B-spline; volume exactly 160/3). The under-parabola side walls carry
# DEGREE-2 pcurves — a chord polygon is a silently wrong region, so the
# import must bracket through control-hull strip boxes. The y=0 wall is a
# PLANE (exercising the bilinear lift with BOTH a curved pcurve and
# pcurve-less straight edges resolved by exact projection); the y=4 wall
# is a degree-(1,1) B-spline. The roof shares the SAME parabola edges
# through straight border pcurves — its loop must pick up the wall's
# subdivision breakpoints (per-EDGE-ENTITY sharing; a per-face walk would
# refuse at pairing).


def _e(eid: int, body: str) -> str:
    return f"#{eid} = {body};"


def _parabolic_roof_step() -> str:
    L: list[str] = []

    def cp(eid, *coords):
        vals = ",".join(f"{float(c):g}." if float(c) == int(c)
                        else f"{float(c):g}" for c in coords)
        L.append(_e(eid, f"CARTESIAN_POINT('',({vals}))"))
        return eid

    # ---- 3D corner points + vertices (8)
    corners = {  # eid: (x, y, z)
        101: (0, 0, 0), 102: (4, 0, 0), 103: (4, 4, 0), 104: (0, 4, 0),
        105: (0, 0, 2), 106: (4, 0, 2), 107: (4, 4, 2), 108: (0, 4, 2)}
    for eid, (x, y, z) in corners.items():
        cp(eid, x, y, z)
        L.append(_e(eid + 100, f"VERTEX_POINT('',#{eid})"))  # 201..208

    # ---- surfaces
    # bottom z=0: S(u,v) = (u, v, 0), u,v ∈ [0,4]; S_u×S_v = +z (inward)
    cp(301, 0, 0, 0); cp(302, 0, 4, 0); cp(303, 4, 0, 0); cp(304, 4, 4, 0)
    L.append(_e(310, "B_SPLINE_SURFACE_WITH_KNOTS('',1,1,"
                     "((#301,#302),(#303,#304)),.UNSPECIFIED.,.F.,.F.,.F.,"
                     "(2,2),(2,2),(0.,4.),(0.,4.),.UNSPECIFIED.)"))
    # wall x=0: S(u,v) = (0, u, v); S_u×S_v = +x (inward)
    cp(311, 0, 0, 0); cp(312, 0, 0, 2); cp(313, 0, 4, 0); cp(314, 0, 4, 2)
    L.append(_e(320, "B_SPLINE_SURFACE_WITH_KNOTS('',1,1,"
                     "((#311,#312),(#313,#314)),.UNSPECIFIED.,.F.,.F.,.F.,"
                     "(2,2),(2,2),(0.,4.),(0.,2.),.UNSPECIFIED.)"))
    # wall x=4: S(u,v) = (4, u, v); S_u×S_v = +x (outward)
    cp(321, 4, 0, 0); cp(322, 4, 0, 2); cp(323, 4, 4, 0); cp(324, 4, 4, 2)
    L.append(_e(330, "B_SPLINE_SURFACE_WITH_KNOTS('',1,1,"
                     "((#321,#322),(#323,#324)),.UNSPECIFIED.,.F.,.F.,.F.,"
                     "(2,2),(2,2),(0.,4.),(0.,2.),.UNSPECIFIED.)"))
    # wall y=0: a PLANE, frame x-axis=(1,0,0), normal(axis)=(0,-1,0) →
    # plane params (s,t) = (x, z), lift normal −y (outward)
    cp(341, 0, 0, 0)
    L.append(_e(342, "DIRECTION('',(0.,-1.,0.))"))
    L.append(_e(343, "DIRECTION('',(1.,0.,0.))"))
    L.append(_e(344, "AXIS2_PLACEMENT_3D('',#341,#342,#343)"))
    L.append(_e(340, "PLANE('',#344)"))
    # wall y=4: S(u,v) = (u, 4, v); S_u×S_v = −y (inward)
    cp(351, 0, 4, 0); cp(352, 0, 4, 4); cp(353, 4, 4, 0); cp(354, 4, 4, 4)
    L.append(_e(350, "B_SPLINE_SURFACE_WITH_KNOTS('',1,1,"
                     "((#351,#352),(#353,#354)),.UNSPECIFIED.,.F.,.F.,.F.,"
                     "(2,2),(2,2),(0.,4.),(0.,4.),.UNSPECIFIED.)"))
    # roof: degree (2,1), S(u,v) = (u, v, 2+2u−u²/2); S_u×S_v ≈ +z (outward)
    cp(361, 0, 0, 2); cp(362, 0, 4, 2)
    cp(363, 2, 0, 6); cp(364, 2, 4, 6)
    cp(365, 4, 0, 2); cp(366, 4, 4, 2)
    L.append(_e(360, "B_SPLINE_SURFACE_WITH_KNOTS('',2,1,"
                     "((#361,#362),(#363,#364),(#365,#366)),"
                     ".UNSPECIFIED.,.F.,.F.,.F.,"
                     "(3,3),(2,2),(0.,4.),(0.,4.),.UNSPECIFIED.)"))

    ctx2 = 400
    L.append(_e(ctx2, "( GEOMETRIC_REPRESENTATION_CONTEXT(2) "
                      "PARAMETRIC_REPRESENTATION_CONTEXT() "
                      "REPRESENTATION_CONTEXT('2d','') )"))
    nid = [500]

    def emit(body):
        nid[0] += 1
        L.append(_e(nid[0], body))
        return nid[0]

    def pt2(u, v):
        return emit(f"CARTESIAN_POINT('',({float(u):g},{float(v):g}))")

    def pcurve(surf, cps2, degree=1):
        refs = ",".join(f"#{pt2(u, v)}" for (u, v) in cps2)
        m = degree + 1
        c2 = emit(f"B_SPLINE_CURVE_WITH_KNOTS('',{degree},({refs}),"
                  f".UNSPECIFIED.,.F.,.F.,({m},{m}),(0.,4.),.UNSPECIFIED.)")
        dr = emit(f"DEFINITIONAL_REPRESENTATION('',(#{c2}),#{ctx2})")
        return emit(f"PCURVE('',#{surf},#{dr})")

    def curve3(cps3, degree=1):
        refs = []
        for (x, y, z) in cps3:
            refs.append(emit(f"CARTESIAN_POINT('',({float(x):g},{float(y):g},"
                             f"{float(z):g}))"))
        m = degree + 1
        return emit(f"B_SPLINE_CURVE_WITH_KNOTS('',{degree},"
                    f"({','.join('#' + str(r) for r in refs)}),"
                    f".UNSPECIFIED.,.F.,.F.,({m},{m}),(0.,4.),.UNSPECIFIED.)")

    def edge(v1, v2, c3, pcs):
        sc = emit(f"SURFACE_CURVE('',#{c3},"
                  f"({','.join('#' + str(p) for p in pcs)}),.PCURVE_S1.)")
        return emit(f"EDGE_CURVE('',#{v1 + 100},#{v2 + 100},#{sc},.T.)")

    # ---- 12 edges (t ∈ [0,4] everywhere)
    # bottom rectangle
    e_b_y0 = edge(101, 102, curve3([(0, 0, 0), (4, 0, 0)]),
                  [pcurve(310, [(0, 0), (4, 0)])])       # plane wall: projection
    e_b_x4 = edge(102, 103, curve3([(4, 0, 0), (4, 4, 0)]),
                  [pcurve(310, [(4, 0), (4, 4)]), pcurve(330, [(0, 0), (4, 0)])])
    e_b_y4 = edge(103, 104, curve3([(4, 4, 0), (0, 4, 0)]),
                  [pcurve(310, [(4, 4), (0, 4)]), pcurve(350, [(4, 0), (0, 0)])])
    e_b_x0 = edge(104, 101, curve3([(0, 4, 0), (0, 0, 0)]),
                  [pcurve(310, [(0, 4), (0, 0)]), pcurve(320, [(4, 0), (0, 0)])])
    # verticals
    e_v_00 = edge(101, 105, curve3([(0, 0, 0), (0, 0, 2)]),
                  [pcurve(320, [(0, 0), (0, 2)])])       # plane wall: projection
    e_v_40 = edge(102, 106, curve3([(4, 0, 0), (4, 0, 2)]),
                  [pcurve(330, [(0, 0), (0, 2)])])       # plane wall: projection
    e_v_44 = edge(103, 107, curve3([(4, 4, 0), (4, 4, 2)]),
                  [pcurve(330, [(4, 0), (4, 2)]), pcurve(350, [(4, 0), (4, 2)])])
    e_v_04 = edge(104, 108, curve3([(0, 4, 0), (0, 4, 2)]),
                  [pcurve(320, [(4, 0), (4, 2)]), pcurve(350, [(0, 0), (0, 2)])])
    # top: two parabola edges + two straight edges
    e_t_y0 = edge(105, 106, curve3([(0, 0, 2), (2, 0, 6), (4, 0, 2)], degree=2),
                  [pcurve(340, [(0, 2), (2, 6), (4, 2)], degree=2),
                   pcurve(360, [(0, 0), (4, 0)])])
    e_t_y4 = edge(108, 107, curve3([(0, 4, 2), (2, 4, 6), (4, 4, 2)], degree=2),
                  [pcurve(350, [(0, 2), (2, 6), (4, 2)], degree=2),
                   pcurve(360, [(0, 4), (4, 4)])])
    e_t_x0 = edge(105, 108, curve3([(0, 0, 2), (0, 4, 2)]),
                  [pcurve(320, [(0, 2), (4, 2)]), pcurve(360, [(0, 0), (0, 4)])])
    e_t_x4 = edge(106, 107, curve3([(4, 0, 2), (4, 4, 2)]),
                  [pcurve(330, [(0, 2), (4, 2)]), pcurve(360, [(4, 0), (4, 4)])])

    def face(surf, oriented, sense=".T."):
        oes = [emit(f"ORIENTED_EDGE('',*,*,#{e},{'.T.' if f else '.F.'})")
               for (e, f) in oriented]
        lp = emit(f"EDGE_LOOP('',({','.join('#' + str(o) for o in oes)}))")
        b = emit(f"FACE_OUTER_BOUND('',#{lp},.T.)")
        return emit(f"ADVANCED_FACE('',(#{b}),#{surf},{sense})")

    faces = [
        # bottom (outward −z; S_u×S_v = +z → same_sense .F.)
        face(310, [(e_b_y0, True), (e_b_x4, True), (e_b_y4, True),
                   (e_b_x0, True)], ".F."),
        # wall x=0 (outward −x; S_u×S_v = +x → .F.)
        face(320, [(e_b_x0, False), (e_v_04, True), (e_t_x0, False),
                   (e_v_00, False)], ".F."),
        # wall x=4 (outward +x; .T.)
        face(330, [(e_b_x4, False), (e_v_40, True), (e_t_x4, True),
                   (e_v_44, False)], ".T."),
        # wall y=0 (PLANE, outward −y = plane axis; .T.)
        face(340, [(e_b_y0, True), (e_v_40, True), (e_t_y0, False),
                   (e_v_00, False)], ".T."),
        # wall y=4 (outward +y; S_u×S_v = −y → .F.)
        face(350, [(e_b_y4, False), (e_v_44, True), (e_t_y4, False),
                   (e_v_04, False)], ".F."),
        # roof (outward has +z component; .T.)
        face(360, [(e_t_y0, True), (e_t_x4, True), (e_t_y4, False),
                   (e_t_x0, False)], ".T."),
    ]
    sh = emit(f"CLOSED_SHELL('',({','.join('#' + str(f) for f in faces)}))")
    emit(f"MANIFOLD_SOLID_BREP('paraboloid_roof',#{sh})")
    body = "\n".join(L)
    return ("ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
            "ENDSEC;\nDATA;\n" + body + "\nENDSEC;\nEND-ISO-10303-21;\n")


def test_parabolic_roof_certified_bracket_contains_truth() -> None:
    truth = F(160, 3)
    shell = read_step_freeform_solid(_parabolic_roof_step(), trim_subdiv=3)
    vol = shell.volume(depth=4)
    assert vol.lo < truth < vol.hi        # curved trim: certified, not exact
    assert vol.width > 0
    # finer per-edge subdivision → smaller hull boxes → tighter bracket
    shell5 = read_step_freeform_solid(_parabolic_roof_step(), trim_subdiv=5)
    vol5 = shell5.volume(depth=4)
    assert vol5.lo <= truth <= vol5.hi
    assert vol5.width < vol.width


def test_parabolic_roof_survives_audit() -> None:
    shell = read_step_freeform_solid(_parabolic_roof_step(), trim_subdiv=4)
    report = shell.audit(depth=4)
    assert report["faces"] == 6
    for comp in report["vector_area"]:
        assert comp.lo <= 0 <= comp.hi


# -- inner bounds: a through-bored box, holes as FACE_BOUND loops --------------
#
# The planar path refuses holes; the freeform path supports them. A 4×4×4
# box with a square 2×2 bore through z: top and bottom faces carry a
# FACE_BOUND hole loop whose edges pair with the four bore walls. All
# faces are degree-(1,1) B-splines, every pcurve degree 1 → exact tier:
# the volume bracket must be width-0 at exactly 64 − 16 = 48.


def _bored_box_step() -> str:
    L: list[str] = []
    nid = [900]

    def emit(body):
        nid[0] += 1
        L.append(_e(nid[0], body))
        return nid[0]

    def cp(*coords):
        vals = ",".join(f"{float(c):g}." if float(c) == int(c)
                        else f"{float(c):g}" for c in coords)
        return emit(f"CARTESIAN_POINT('',({vals}))")

    # 16 vertices: outer 8, bore rim 8
    V = {}
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    inner = [(1, 1), (3, 1), (3, 3), (1, 3)]
    for i, (x, y) in enumerate(outer):
        V[101 + i] = (x, y, 0)
        V[105 + i] = (x, y, 4)
    for i, (x, y) in enumerate(inner):
        V[111 + i] = (x, y, 0)
        V[115 + i] = (x, y, 4)
    vid = {k: emit(f"VERTEX_POINT('',#{cp(*p)})") for k, p in V.items()}

    def bsurf(p00, p01, p10, p11, ku="(0.,4.)", kv="(0.,4.)"):
        return emit(
            f"B_SPLINE_SURFACE_WITH_KNOTS('',1,1,"
            f"((#{cp(*p00)},#{cp(*p01)}),(#{cp(*p10)},#{cp(*p11)})),"
            f".UNSPECIFIED.,.F.,.F.,.F.,(2,2),(2,2),{ku},{kv},"
            f".UNSPECIFIED.)")

    S = {
        "bot": bsurf((0, 0, 0), (0, 4, 0), (4, 0, 0), (4, 4, 0)),
        "top": bsurf((0, 0, 4), (0, 4, 4), (4, 0, 4), (4, 4, 4)),
        "x0": bsurf((0, 0, 0), (0, 0, 4), (0, 4, 0), (0, 4, 4)),
        "x4": bsurf((4, 0, 0), (4, 0, 4), (4, 4, 0), (4, 4, 4)),
        "y0": bsurf((0, 0, 0), (0, 0, 4), (4, 0, 0), (4, 0, 4)),
        "y4": bsurf((0, 4, 0), (0, 4, 4), (4, 4, 0), (4, 4, 4)),
        "bx1": bsurf((1, 1, 0), (1, 1, 4), (1, 3, 0), (1, 3, 4),
                     ku="(1.,3.)"),
        "bx3": bsurf((3, 1, 0), (3, 1, 4), (3, 3, 0), (3, 3, 4),
                     ku="(1.,3.)"),
        "by1": bsurf((1, 1, 0), (1, 1, 4), (3, 1, 0), (3, 1, 4),
                     ku="(1.,3.)"),
        "by3": bsurf((1, 3, 0), (1, 3, 4), (3, 3, 0), (3, 3, 4),
                     ku="(1.,3.)"),
    }
    ctx2 = emit("( GEOMETRIC_REPRESENTATION_CONTEXT(2) "
                "PARAMETRIC_REPRESENTATION_CONTEXT() "
                "REPRESENTATION_CONTEXT('2d','') )")

    def pc(surf, a, b):
        c2 = emit(f"B_SPLINE_CURVE_WITH_KNOTS('',1,"
                  f"(#{cp(*a)},#{cp(*b)}),.UNSPECIFIED.,.F.,.F.,(2,2),"
                  f"(0.,4.),.UNSPECIFIED.)")
        dr = emit(f"DEFINITIONAL_REPRESENTATION('',(#{c2}),#{ctx2})")
        return emit(f"PCURVE('',#{surf},#{dr})")

    def edge(v1, v2, pcs):
        c3 = emit(f"B_SPLINE_CURVE_WITH_KNOTS('',1,"
                  f"(#{cp(*V[v1])},#{cp(*V[v2])}),.UNSPECIFIED.,.F.,.F.,"
                  f"(2,2),(0.,4.),.UNSPECIFIED.)")
        sc = emit(f"SURFACE_CURVE('',#{c3},"
                  f"({','.join('#' + str(x) for x in pcs)}),.PCURVE_S1.)")
        return emit(f"EDGE_CURVE('',#{vid[v1]},#{vid[v2]},#{sc},.T.)")

    E = {
        # bottom outer rectangle
        "b1": edge(101, 102, [pc(S["bot"], (0, 0), (4, 0)),
                              pc(S["y0"], (0, 0), (4, 0))]),
        "b2": edge(102, 103, [pc(S["bot"], (4, 0), (4, 4)),
                              pc(S["x4"], (0, 0), (4, 0))]),
        "b3": edge(103, 104, [pc(S["bot"], (4, 4), (0, 4)),
                              pc(S["y4"], (4, 0), (0, 0))]),
        "b4": edge(104, 101, [pc(S["bot"], (0, 4), (0, 0)),
                              pc(S["x0"], (4, 0), (0, 0))]),
        # top outer rectangle
        "t1": edge(105, 106, [pc(S["top"], (0, 0), (4, 0)),
                              pc(S["y0"], (0, 4), (4, 4))]),
        "t2": edge(106, 107, [pc(S["top"], (4, 0), (4, 4)),
                              pc(S["x4"], (0, 4), (4, 4))]),
        "t3": edge(107, 108, [pc(S["top"], (4, 4), (0, 4)),
                              pc(S["y4"], (4, 4), (0, 4))]),
        "t4": edge(108, 105, [pc(S["top"], (0, 4), (0, 0)),
                              pc(S["x0"], (4, 4), (0, 4))]),
        # outer verticals
        "v1": edge(101, 105, [pc(S["x0"], (0, 0), (0, 4)),
                              pc(S["y0"], (0, 0), (0, 4))]),
        "v2": edge(102, 106, [pc(S["x4"], (0, 0), (0, 4)),
                              pc(S["y0"], (4, 0), (4, 4))]),
        "v3": edge(103, 107, [pc(S["x4"], (4, 0), (4, 4)),
                              pc(S["y4"], (4, 0), (4, 4))]),
        "v4": edge(104, 108, [pc(S["x0"], (4, 0), (4, 4)),
                              pc(S["y4"], (0, 0), (0, 4))]),
        # bore rims (bottom, top) and verticals
        "r1": edge(111, 112, [pc(S["bot"], (1, 1), (3, 1)),
                              pc(S["by1"], (1, 0), (3, 0))]),
        "r2": edge(112, 113, [pc(S["bot"], (3, 1), (3, 3)),
                              pc(S["bx3"], (1, 0), (3, 0))]),
        "r3": edge(113, 114, [pc(S["bot"], (3, 3), (1, 3)),
                              pc(S["by3"], (3, 0), (1, 0))]),
        "r4": edge(114, 111, [pc(S["bot"], (1, 3), (1, 1)),
                              pc(S["bx1"], (3, 0), (1, 0))]),
        "s1": edge(115, 116, [pc(S["top"], (1, 1), (3, 1)),
                              pc(S["by1"], (1, 4), (3, 4))]),
        "s2": edge(116, 117, [pc(S["top"], (3, 1), (3, 3)),
                              pc(S["bx3"], (1, 4), (3, 4))]),
        "s3": edge(117, 118, [pc(S["top"], (3, 3), (1, 3)),
                              pc(S["by3"], (3, 4), (1, 4))]),
        "s4": edge(118, 115, [pc(S["top"], (1, 3), (1, 1)),
                              pc(S["bx1"], (3, 4), (1, 4))]),
        "w1": edge(111, 115, [pc(S["bx1"], (1, 0), (1, 4)),
                              pc(S["by1"], (1, 0), (1, 4))]),
        "w2": edge(112, 116, [pc(S["bx3"], (1, 0), (1, 4)),
                              pc(S["by1"], (3, 0), (3, 4))]),
        "w3": edge(113, 117, [pc(S["bx3"], (3, 0), (3, 4)),
                              pc(S["by3"], (3, 0), (3, 4))]),
        "w4": edge(114, 118, [pc(S["bx1"], (3, 0), (3, 4)),
                              pc(S["by3"], (1, 0), (1, 4))]),
    }

    def face(surf, loops, sense):
        bounds = []
        for kind, oriented in loops:
            oes = [emit(f"ORIENTED_EDGE('',*,*,#{E[e]},"
                        f"{'.T.' if f else '.F.'})") for (e, f) in oriented]
            lp = emit(f"EDGE_LOOP('',"
                      f"({','.join('#' + str(o) for o in oes)}))")
            bounds.append(emit(f"{kind}('',#{lp},.T.)"))
        blist = ",".join("#" + str(b) for b in bounds)
        return emit(f"ADVANCED_FACE('',({blist}),#{surf},{sense})")

    faces = [
        face(S["bot"],
             [("FACE_OUTER_BOUND", [("b1", True), ("b2", True),
                                    ("b3", True), ("b4", True)]),
              ("FACE_BOUND", [("r1", True), ("r2", True),
                              ("r3", True), ("r4", True)])], ".F."),
        face(S["top"],
             [("FACE_OUTER_BOUND", [("t1", True), ("t2", True),
                                    ("t3", True), ("t4", True)]),
              ("FACE_BOUND", [("s1", True), ("s2", True),
                              ("s3", True), ("s4", True)])], ".T."),
        face(S["x0"], [("FACE_OUTER_BOUND",
                        [("b4", False), ("v4", True), ("t4", True),
                         ("v1", False)])], ".F."),
        face(S["x4"], [("FACE_OUTER_BOUND",
                        [("b2", True), ("v3", True), ("t2", False),
                         ("v2", False)])], ".T."),
        face(S["y0"], [("FACE_OUTER_BOUND",
                        [("b1", True), ("v2", True), ("t1", False),
                         ("v1", False)])], ".T."),
        face(S["y4"], [("FACE_OUTER_BOUND",
                        [("b3", False), ("v3", True), ("t3", True),
                         ("v4", False)])], ".F."),
        face(S["bx1"], [("FACE_OUTER_BOUND",
                         [("r4", True), ("w1", True), ("s4", False),
                          ("w4", False)])], ".T."),
        face(S["bx3"], [("FACE_OUTER_BOUND",
                         [("r2", True), ("w3", True), ("s2", False),
                          ("w2", False)])], ".F."),
        face(S["by1"], [("FACE_OUTER_BOUND",
                         [("r1", True), ("w2", True), ("s1", False),
                          ("w1", False)])], ".F."),
        face(S["by3"], [("FACE_OUTER_BOUND",
                         [("r3", True), ("w4", True), ("s3", False),
                          ("w3", False)])], ".T."),
    ]
    sh = emit(f"CLOSED_SHELL('',({','.join('#' + str(f) for f in faces)}))")
    emit(f"MANIFOLD_SOLID_BREP('bored_box',#{sh})")
    return ("ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
            "ENDSEC;\nDATA;\n" + "\n".join(L)
            + "\nENDSEC;\nEND-ISO-10303-21;\n")


def test_inner_bounds_bored_box_exact_width0() -> None:
    shell = read_step_freeform_solid(_bored_box_step())
    vol = shell.volume(depth=4)
    assert vol.lo == vol.hi == 48         # 64 − 2·2·4, holes as FACE_BOUND
    report = shell.audit(depth=4)
    assert report["faces"] == 10


# -- #135 composition: closure before orientation, gaps in millimetres --------


def test_missing_face_refuses_with_mm_gap_report() -> None:
    step = _loft_step()
    m = re.search(r"CLOSED_SHELL\('[^']*',\((#\d+),", step)
    assert m
    step = step.replace(m.group(0), m.group(0).replace(m.group(1) + ",", "", 1))
    with pytest.raises(NonClosedShellError) as exc:
        read_step_freeform_solid(step)
    rep = exc.value.report
    assert rep["open_edges"] > 0
    assert rep["open_perimeter_mm"] > 0
    assert "mm" in str(exc.value)


def _torn_loft_step() -> tuple[str, int]:
    """The loft STEP with ONE edge's start vertex re-pointed at a duplicate
    VERTEX_POINT 1e-7 away — the classic export tear."""
    step = _loft_step()
    m = re.search(r"#(\d+) = EDGE_CURVE\('',#(\d+),", step)
    assert m
    vid = int(m.group(2))
    vm = re.search(rf"#{vid} = VERTEX_POINT\('',#(\d+)\)", step)
    pm = re.search(rf"#{vm.group(1)} = CARTESIAN_POINT\('',\(([^)]*)\)\)", step)
    x, rest = pm.group(1).split(",", 1)
    dup = (f"#9901 = CARTESIAN_POINT('',({float(x) + 1e-7},{rest}));\n"
           f"#9902 = VERTEX_POINT('',#9901);")
    step = step.replace(m.group(0), m.group(0).replace(f"#{vid},", "#9902,"))
    step = step.replace("ENDSEC;\nEND-ISO", dup + "\nENDSEC;\nEND-ISO")
    return step, vid


def test_tear_refuses_without_heal_and_heals_with_recorded_intent() -> None:
    torn, _ = _torn_loft_step()
    with pytest.raises(NonClosedShellError):
        read_step_freeform_solid(torn)
    rep: dict = {}
    shell = read_step_freeform_solid(torn, heal_tolerance=1e-6, report=rep)
    assert rep["healed"]["moved"] >= 1
    assert rep["healed"]["max_move_mm"] <= 1e-6
    vol = shell.volume(depth=3)
    assert vol.lo == vol.hi == LOFT.volume()   # trims untouched: still exact


def test_heal_cannot_invent_a_missing_face() -> None:
    step = _loft_step()
    m = re.search(r"CLOSED_SHELL\('[^']*',\((#\d+),", step)
    step = step.replace(m.group(0), m.group(0).replace(m.group(1) + ",", "", 1))
    with pytest.raises(NonClosedShellError):
        read_step_freeform_solid(step, heal_tolerance=1e-3)


# -- refusals by name ----------------------------------------------------------


def test_analytic_surface_in_freeform_shell_refuses_by_name() -> None:
    step = write_step_planar_solid(
        Solid.box(20, 20, 5), bores=[Cyl(F(10), F(10), F(3), F(0), F(5))])
    with pytest.raises(ValueError, match="CYLINDRICAL_SURFACE"):
        read_step_solid(step)


def test_missing_pcurve_refuses_by_name() -> None:
    step = _loft_step()
    m = re.search(r"SURFACE_CURVE\('',#\d+,\((#\d+),(#\d+)\)", step)
    assert m
    step = step.replace(m.group(0), m.group(0).replace("," + m.group(2), ""))
    with pytest.raises(ValueError, match="pcurve"):
        read_step_freeform_solid(step)


def test_vertex_loop_refuses_by_name() -> None:
    step = _loft_step()
    vm = re.search(r"#(\d+) = VERTEX_POINT", step)
    lm = re.search(r"FACE_OUTER_BOUND\('',#(\d+),", step)
    step = step.replace(
        "ENDSEC;\nEND-ISO",
        f"#9903 = VERTEX_LOOP('',#{vm.group(1)});\nENDSEC;\nEND-ISO")
    step = step.replace(lm.group(0),
                        lm.group(0).replace("#" + lm.group(1), "#9903"))
    with pytest.raises(ValueError, match="VERTEX_LOOP"):
        read_step_freeform_solid(step)


def test_edge_twice_in_one_face_refuses_as_seam() -> None:
    step = _loft_step()
    lm = re.search(r"#(\d+) = EDGE_LOOP\('',\((#\d+),(#\d+),", step)
    assert lm
    oe = lm.group(3)
    em = re.search(rf"{oe} = ORIENTED_EDGE\('',\*,\*,(#\d+),(\.\w\.)\)", step)
    assert em
    flipped = ".F." if em.group(2) == ".T." else ".T."
    step = step.replace(
        "ENDSEC;\nEND-ISO",
        f"#9904 = ORIENTED_EDGE('',*,*,{em.group(1)},{flipped});\n"
        "ENDSEC;\nEND-ISO")
    step = step.replace(lm.group(0),
                        lm.group(0).replace(oe + ",", oe + ",#9904,"))
    with pytest.raises(ValueError, match="twice|seam"):
        read_step_freeform_solid(step)


def _swap_first_surface_rational(step: str, weights: str) -> str:
    m = re.search(
        r"#(\d+) = B_SPLINE_SURFACE_WITH_KNOTS\('',1,1,"
        r"\(\((#\d+),(#\d+)\),\((#\d+),(#\d+)\)\),\.UNSPECIFIED\.,"
        r"\.F\.,\.F\.,\.F\.,\(2,2\),\(2,2\),\(([^)]*)\),\(([^)]*)\),"
        r"\.UNSPECIFIED\.\);", step)
    assert m, "box writer no longer emits simple bilinear surfaces"
    eid, p00, p01, p10, p11, ku, kv = m.groups()
    complexed = (
        f"#{eid} = ( BOUNDED_SURFACE() B_SPLINE_SURFACE(1,1,"
        f"(({p00},{p01}),({p10},{p11})),.UNSPECIFIED.,.F.,.F.,.F.) "
        f"B_SPLINE_SURFACE_WITH_KNOTS((2,2),(2,2),({ku}),({kv}),"
        f".UNSPECIFIED.) GEOMETRIC_REPRESENTATION_ITEM() "
        f"RATIONAL_B_SPLINE_SURFACE({weights}) SURFACE() );")
    return step.replace(m.group(0), complexed)


def _box_step() -> str:
    # off-origin so every face's flux integrand is nonzero — a rational
    # swap on a through-origin face would show a legitimately width-0
    # bracket (P ≡ 0) and prove nothing
    patches = box_patches(2, 3, 4, origin=(1, 1, 1))
    return write_step_patch_solid(
        PatchSolid(patches, seams=_derive_seams(patches)))


def test_one_signed_rational_face_imports_certified() -> None:
    # varying (one-signed) weights reparameterize the same planar quad,
    # so the true volume is still exactly 24 — but the face's flux now
    # goes through the certified reciprocal/Neumann rule, so the bracket
    # carries genuine width and must CONTAIN the truth
    step = _swap_first_surface_rational(_box_step(), "((0.5,1.),(1.,2.))")
    shell = read_step_freeform_solid(step, depth=4)
    vol = shell.volume(depth=4)
    assert vol.lo <= 24 <= vol.hi
    assert vol.width > 0                  # rational: certified, never "exact"


def test_sign_varying_weights_refuse_by_name() -> None:
    step = _swap_first_surface_rational(_box_step(), "((0.5,-0.5),(0.5,0.5))")
    with pytest.raises(ValueError, match="sign"):
        read_step_freeform_solid(step)


# -- writer guardrails ---------------------------------------------------------


def test_writer_refuses_without_proven_seams() -> None:
    ps = PatchSolid(box_patches(1, 1, 1))          # seams=None
    with pytest.raises(ValueError, match="seam"):
        write_step_patch_solid(ps)


def test_multi_solid_file_drops_into_report() -> None:
    step = _loft_step()
    m = re.search(r"#(\d+) = MANIFOLD_SOLID_BREP\('[^']*',#(\d+)\);", step)
    step = step.replace(
        "ENDSEC;\nEND-ISO",
        f"#9905 = MANIFOLD_SOLID_BREP('extra',#{m.group(2)});\n"
        "ENDSEC;\nEND-ISO")
    rep: dict = {}
    shell = read_step_freeform_solid(step, report=rep)
    assert isinstance(shell, TrimmedShell)
    assert any("multi-solid" in d for d in rep.get("dropped", []))
