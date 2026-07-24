"""K3.3 — SSI: complete branch detection + certified refinement.

Ground-truth cases (constructed so the true branch count is known):
plane z=0 against quadratic sheets whose zero sets are exactly two
lines, one line, nothing, or a tangential line. The tangential case is
the differentiation moment: OCCT's GeomAPI_IntSS returns 0 lines for
it (verified in gitcad's oracle suite); forge finds the branch.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.ssi import BezierPatch, refine_point, ssi, ssi_branches

F = Fraction


def _plane():
    return BezierPatch([[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)]])


def _quad_sheet(b0, b1, b2):
    """z = quadratic in u (Bézier coeffs b0,b1,b2), ruled in v."""
    return BezierPatch([[(0, 0, b0), (0, 1, b0)],
                        [(F(1, 2), 0, b1), (F(1, 2), 1, b1)],
                        [(1, 0, b2), (1, 1, b2)]])


def test_two_branch_case_counts_exactly_two() -> None:
    # z = (u-1/2)^2 - 1/16: zero lines at u = 1/4 and u = 3/4
    r = ssi(_plane(), _quad_sheet(F(3, 16), F(-5, 16), F(3, 16)), depth=5)
    assert r["branches"] == 2
    assert r["uncertified"] == 0
    assert len(r["points"]) > 0


def test_single_branch_case() -> None:
    # z = u - 1/2: one straight line
    lin = BezierPatch([[(0, 0, F(-1, 2)), (0, 1, F(-1, 2))],
                       [(1, 0, F(1, 2)), (1, 1, F(1, 2))]])
    r = ssi(_plane(), lin, depth=5)
    assert r["branches"] == 1
    assert r["uncertified"] == 0


def test_empty_case_is_certified_not_just_unfound() -> None:
    # z = u^2 + 1 never meets z=0; bbox disjointness PROVES it
    r = ssi(_plane(), _quad_sheet(1, 1, 2), depth=5)
    assert r["branches"] == 0
    assert r["empty_certified"] is True


def test_tangent_branch_found_where_float_kernels_miss() -> None:
    # z = (u-1/2)^2 touches z=0 along u=1/2 — a tangential (measure-zero
    # crossing) branch. OCCT GeomAPI_IntSS returns 0 lines for this;
    # forge finds it and certifies points on it.
    r = ssi(_plane(), _quad_sheet(F(1, 4), F(-1, 4), F(1, 4)), depth=5)
    assert r["branches"] == 1
    assert r["uncertified"] == 0
    # every certified point sits at u≈1/2 on the parabola sheet. The
    # certificate is SPATIAL (|A−B| < 1e-10); near a tangency distance
    # grows as (Δu)², so parameters may sit ~1e-5 off while the point is
    # provably 1e-10-close in space.
    for _, _, s, _ in r["points"]:
        assert abs(float(s) - 0.5) < 1e-4


def test_certified_residual_is_exact_rational() -> None:
    plane, sheet = _plane(), _quad_sheet(F(3, 16), F(-5, 16), F(3, 16))
    u, v, s, t, ok, res2 = refine_point(plane.net, sheet.net,
                                        F(1, 4), F(1, 2), F(1, 4), F(1, 2))
    assert ok
    assert isinstance(res2, Fraction)         # the certificate is exact
    assert res2 < F(1, 10 ** 20)


def test_ssi_fuzz_certified_points_lie_on_both_graphs() -> None:
    """Validation gauntlet — fuzz SSI over random bilinear GRAPH patches
    z=f(x,y) over [0,1]². Every certified point must lie on BOTH graphs
    (spatial residual ~0); because a graph's eval(x,y) returns (x,y,·), that
    forces the two parameter pairs to agree (u=s, v=t). A decidable
    correctness net for the SSI core that ssi_curves now builds on.
    Deterministic (seeded) so a failure reproduces exactly."""
    import random

    from forgekernel.ssi import _eval_patch

    rng = random.Random(20260724)

    def graph(z):     # bilinear graph, corner heights z = [z00, z01, z10, z11]
        return BezierPatch([[(0, 0, z[0]), (0, 1, z[1])],
                            [(1, 0, z[2]), (1, 1, z[3])]])

    hits = empties = 0
    for _ in range(40):
        A = graph([F(rng.randint(-3, 3)) for _ in range(4)])
        B = graph([F(rng.randint(-3, 3)) for _ in range(4)])
        r = ssi(A, B, depth=4)
        # SSI is symmetric: swapping operands finds the same branch count
        assert ssi(B, A, depth=4)["branches"] == r["branches"]
        if r["empty_certified"]:
            empties += 1
        if r["points"]:
            hits += 1
        for u, v, s, t in r["points"]:
            pa, pb = _eval_patch(A.net, u, v), _eval_patch(B.net, s, t)
            assert (pa[0], pa[1]) == (u, v)       # graph eval is exact in x,y
            assert (pb[0], pb[1]) == (s, t)
            d2 = sum((pa[c] - pb[c]) ** 2 for c in range(3))
            assert d2 < F(1, 10 ** 18)            # points coincide in space
            assert abs(float(u) - float(s)) < 1e-6 and abs(float(v) - float(t)) < 1e-6
    # the fuzz actually drives both the hit path and the certified-empty path
    assert hits >= 5 and empties >= 3


def test_resolution_semantics_merge_not_miss() -> None:
    # two lines 1/10 apart: z=(u-1/2)^2 - 1/400 → zeros at 1/2 ± 1/20.
    # At depth 3 (cell 1/8) they merge into one reported branch; at depth
    # 6 (cell 1/64) they separate. Never zero — never MISSED.
    b0 = F(1, 4) - F(1, 400)
    b1 = -F(1, 4) - F(1, 400)
    sheet = _quad_sheet(b0, b1, b0)
    shallow, _ = ssi_branches(_plane(), sheet, depth=3)
    deep, _ = ssi_branches(_plane(), sheet, depth=6)
    assert len(shallow) >= 1                   # found (possibly merged)
    assert len(deep) == 2                      # resolved


def test_subdivision_is_exact() -> None:
    # de Casteljau split preserves corner values exactly
    sheet = _quad_sheet(F(3, 16), F(-5, 16), F(3, 16))
    left, right = sheet.split_u()
    assert left.net[0][0] == sheet.net[0][0]
    assert right.net[-1][-1] == sheet.net[-1][-1]
    # midpoint of the split boundary is the exact curve point
    assert left.net[-1][0] == right.net[0][0]
    assert all(isinstance(c, Fraction) for c in left.net[-1][0])


# -- K3.5: Bézier extraction + SSI over B-splines + polylines -----------------

def test_bezier_extraction_is_exact() -> None:
    from forgekernel.nurbs import BSplineCurve, BSplineSurface, \
        bezier_patches, bezier_segments
    from forgekernel.ssi import _dc1, _eval_patch

    c = BSplineCurve(2, [(0, 0, 0), (1, 2, 0), (3, 2, 0), (4, 0, 0)],
                     [0, 0, 0, 1, 2, 2, 2])
    segs = bezier_segments(c)
    assert len(segs) == 2
    for u0, u1, pts in segs:
        for k in range(5):
            t = F(k, 4)
            assert _dc1(pts, t) == c.eval(u0 + t * (u1 - u0))   # bit-equal

    net = [[(x, y, F(x * y, 3)) for y in range(3)] for x in range(4)]
    s = BSplineSurface(2, 2, net, [0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1])
    patches = bezier_patches(s)
    assert len(patches) == 2
    for u0, u1, v0, v1, pn in patches:
        for a in range(3):
            for b in range(3):
                tu, tv = F(a, 2), F(b, 2)
                assert _eval_patch(pn, tu, tv) == \
                    s.eval(u0 + tu * (u1 - u0), v0 + tv * (v1 - v0))


def test_ssi_branch_crosses_patch_boundary_unsplit() -> None:
    from forgekernel.nurbs import BSplineSurface
    from forgekernel.ssi import ssi_surfaces

    plane = BSplineSurface(1, 1, [[(0, 0, 0), (0, 1, 0)],
                                  [(2, 0, 0), (2, 1, 0)]], [0, 0, 2, 2], [0, 0, 1, 1])
    # z = v − 1/2 on a B-spline with interior u-knot at 1: ONE line at
    # v=1/2 running along u THROUGH the Bézier patch boundary — the
    # clustering must not split it into two branches at the seam.
    net = [[(x, 0, F(-1, 2)), (x, 1, F(1, 2))] for x in range(4)]
    s = BSplineSurface(2, 1, net, [0, 0, 0, 1, 2, 2, 2], [0, 0, 1, 1])
    r = ssi_surfaces(plane, s, depth=4)
    assert r["branches"] == 1
    assert r["uncertified"] == 0
    # every certified point sits at y = 1/2 in space
    for u, v, _, _ in r["points"]:
        assert abs(float(plane.eval(u, v)[1]) - 0.5) < 1e-9


def test_polyline_orders_points_monotonically() -> None:
    from forgekernel.ssi import polyline

    pts = [(0.5, 0, 0), (0.1, 0, 0), (0.9, 0, 0), (0.3, 0, 0), (0.7, 0, 0)]
    pl = polyline(pts)
    xs = [p[0] for p in pl]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True)


# -- K3.6: rational patches + planar STEP topology import ---------------------

def test_rational_patch_ssi_finds_the_arc() -> None:
    from forgekernel.nurbs import BSplineSurface
    from forgekernel.ssi import ssi_surfaces

    # rational quadratic arc (weight 3/4 mid) ruled from z=-1/2 to 1/2;
    # plane z=0 cuts it in exactly one branch (the arc at v=1/2)
    arc = [(1, 0), (1, 1), (0, 1)]
    net = [[(x, y, F(-1, 2)), (x, y, F(1, 2))] for (x, y) in arc]
    wts = [[F(1), F(1)], [F(3, 4), F(3, 4)], [F(1), F(1)]]
    s = BSplineSurface(2, 1, net, [0, 0, 0, 1, 1, 1], [0, 0, 1, 1], wts)
    plane = BSplineSurface(1, 1, [[(-1, -1, 0), (-1, 2, 0)],
                                  [(2, -1, 0), (2, 2, 0)]],
                           [0, 0, 3, 3], [0, 0, 3, 3])
    r = ssi_surfaces(plane, s, depth=4)
    assert r["branches"] == 1
    assert r["uncertified"] == 0
    for _, _, _, t in r["points"]:
        assert abs(float(t) - 0.5) < 1e-9        # on the z=0 mid-line


def test_rational_patch_requires_positive_weights() -> None:
    from forgekernel.ssi import BezierPatch

    with pytest.raises(ValueError, match="positive weights"):
        BezierPatch([[(0, 0, 0, 1), (0, 1, 0, -1)],
                     [(1, 0, 0, 1), (1, 1, 0, 1)]])


# -- K7 ordered SSI output: per-branch parameter-space polylines --------------

def test_order_branch_open_line_is_monotonic() -> None:
    from forgekernel.ssi import _order_branch

    # certified points along u at v=1/2, handed in shuffled; s,t placeholders
    us = [F(1, 2), F(1, 10), F(9, 10), F(3, 10), F(7, 10)]
    pts = [(u, F(1, 2), F(0), F(0)) for u in us]
    ordered, closed = _order_branch(pts)
    ou = [float(p[0]) for p in ordered]
    assert ou == sorted(ou) or ou == sorted(ou, reverse=True)
    assert closed is False


def test_order_branch_small_arcs_are_not_falsely_closed() -> None:
    # regression: a 3-point OPEN arc was always reported closed, because for
    # n==3 the median-of-two gaps is the larger gap and the triangle
    # inequality forces wrap <= 2*median. Closure is undecidable at n<4.
    from forgekernel.ssi import _order_branch

    line3 = [(F(0), F(0), F(0), F(0)), (F(1), F(0), F(0), F(0)),
             (F(2), F(0), F(0), F(0))]
    ordered, closed = _order_branch(line3)
    assert closed is False
    ou = [float(p[0]) for p in ordered]
    assert ou == sorted(ou) or ou == sorted(ou, reverse=True)
    # a non-collinear 3-point arc is also open, not a loop
    _, closed3 = _order_branch([(F(0), F(0), F(0), F(0)),
                                (F(1), F(2), F(0), F(0)),
                                (F(3), F(0), F(0), F(0))])
    assert closed3 is False


def test_order_branch_detects_closed_loop() -> None:
    import math

    from forgekernel.ssi import _order_branch

    n = 12
    raw = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        u = F(round(1000 * (0.5 + 0.4 * math.cos(ang))), 1000)
        v = F(round(1000 * (0.5 + 0.4 * math.sin(ang))), 1000)
        raw.append((u, v, F(0), F(0)))
    perm = [0, 5, 2, 9, 4, 11, 6, 1, 8, 3, 10, 7]     # deterministic shuffle
    ordered, closed = _order_branch([raw[i] for i in perm])
    assert closed is True
    # consecutive ordered points are ring neighbours — no jump across the loop
    angs = [math.atan2(float(p[1]) - 0.5, float(p[0]) - 0.5) for p in ordered]
    steps = []
    for i in range(len(angs)):
        d = (angs[(i + 1) % len(angs)] - angs[i]) % (2 * math.pi)
        steps.append(min(d, 2 * math.pi - d))
    assert max(steps) < 2 * (2 * math.pi / n)


def test_ssi_curves_single_open_branch_is_ordered() -> None:
    from forgekernel.nurbs import BSplineSurface
    from forgekernel.ssi import ssi_curves

    plane = BSplineSurface(1, 1, [[(0, 0, 0), (0, 1, 0)],
                                  [(2, 0, 0), (2, 1, 0)]], [0, 0, 2, 2], [0, 0, 1, 1])
    net = [[(x, 0, F(-1, 2)), (x, 1, F(1, 2))] for x in range(4)]
    s = BSplineSurface(2, 1, net, [0, 0, 0, 1, 2, 2, 2], [0, 0, 1, 1])
    r = ssi_curves(plane, s, depth=4)
    assert r["empty_certified"] is False
    assert r["uncertified"] == 0
    assert len(r["curves"]) == 1
    c = r["curves"][0]
    assert c["closed"] is False
    assert len(c["points"]) == len(c["xyz"]) >= 2
    us = [float(p[0]) for p in c["points"]]
    assert us == sorted(us) or us == sorted(us, reverse=True)   # ordered along u
    for xyz in c["xyz"]:
        assert abs(xyz[1] - 0.5) < 1e-9                          # y = 1/2 line


def test_ssi_curves_certifies_empty_for_disjoint() -> None:
    from forgekernel.nurbs import BSplineSurface
    from forgekernel.ssi import ssi_curves

    a = BSplineSurface(1, 1, [[(0, 0, 0), (0, 1, 0)],
                              [(2, 0, 0), (2, 1, 0)]], [0, 0, 2, 2], [0, 0, 1, 1])
    b = BSplineSurface(1, 1, [[(0, 0, 5), (0, 1, 5)],
                              [(2, 0, 5), (2, 1, 5)]], [0, 0, 2, 2], [0, 0, 1, 1])
    r = ssi_curves(a, b, depth=3)
    assert r["empty_certified"] is True
    assert r["curves"] == []


def test_step_planar_solid_import_is_exact() -> None:
    from forgekernel.stepio import read_step_planar_solid

    # a hand-written unit-cube STEP (minimal topology)
    pts = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
           5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1)}
    lines = []
    for i, (x, y, z) in pts.items():
        lines.append(f"#{i} = CARTESIAN_POINT('',({x}.,{y}.,{z}.));")
        lines.append(f"#{i + 10} = VERTEX_POINT('',#{i});")
    # 6 faces as quads (outward windings); plane normals via axis dirs
    faces = [((1, 4, 3, 2), (0, 0, -1)), ((5, 6, 7, 8), (0, 0, 1)),
             ((1, 2, 6, 5), (0, -1, 0)), ((3, 4, 8, 7), (0, 1, 0)),
             ((2, 3, 7, 6), (1, 0, 0)), ((1, 5, 8, 4), (-1, 0, 0))]
    eid = 100
    face_ids = []
    for verts, nrm in faces:
        oes = []
        for a, b in zip(verts, verts[1:] + verts[:1]):
            lines.append(f"#{eid} = EDGE_CURVE('',#{a + 10},#{b + 10},#{a},.T.);")
            lines.append(f"#{eid + 1} = ORIENTED_EDGE('',*,*,#{eid},.T.);")
            oes.append(f"#{eid + 1}")
            eid += 2
        lines.append(f"#{eid} = EDGE_LOOP('',({','.join(oes)}));")
        loop = eid
        eid += 1
        lines.append(f"#{eid} = DIRECTION('',({nrm[0]}.,{nrm[1]}.,{nrm[2]}.));")
        lines.append(f"#{eid + 1} = AXIS2_PLACEMENT_3D('',#1,#{eid},#{eid});")
        lines.append(f"#{eid + 2} = PLANE('',#{eid + 1});")
        lines.append(f"#{eid + 3} = FACE_OUTER_BOUND('',#{loop},.T.);")
        lines.append(f"#{eid + 4} = ADVANCED_FACE('',(#{eid + 3}),#{eid + 2},.T.);")
        face_ids.append(f"#{eid + 4}")
        eid += 5
    lines.append(f"#{eid} = CLOSED_SHELL('',({','.join(face_ids)}));")
    lines.append(f"#{eid + 1} = MANIFOLD_SOLID_BREP('',#{eid});")
    text = "DATA;\n" + "\n".join(lines) + "\nENDSEC;"
    s = read_step_planar_solid(text)
    assert s.volume() == 1                       # exact
    assert not s.watertight_violations()


# -- K5.2: variable-radius (linear-taper) fillets, exact in ℚ[π] --------------

def test_variable_fillet_exact_and_reduces_to_constant() -> None:
    from forgekernel.quadric import FilletedBox, PiVal, VariableFilletedBox

    # r0==r1 must reduce EXACTLY to the constant FilletedBox (self-oracle)
    v_const = VariableFilletedBox((0, 0, 0), (10, 20, 30),
                                  [("z", "max", "max", 2, 2)]).volume()
    assert v_const == FilletedBox((0, 0, 0), (10, 20, 30),
                                  [("z", "max", "max")], 2).volume()
    # genuine taper r0=1→r1=3 on the L=30 z-edge:
    # X = 30(1+3+9)/3 = 130 → V = 6000 − 130 + (130/4)π = 5870 + 65/2·π
    v_taper = VariableFilletedBox((0, 0, 0), (10, 20, 30),
                                  [("z", "max", "max", 1, 3)]).volume()
    assert v_taper == PiVal(5870, F(65, 2))          # exact ℚ[π]


def test_variable_fillet_refuses_adjacent_edges() -> None:
    import pytest

    from forgekernel.quadric import VariableFilletedBox

    with pytest.raises(ValueError, match="K5.3"):
        VariableFilletedBox((0, 0, 0), (10, 10, 10),
                            [("z", "max", "max", 1, 2),
                             ("x", "max", "max", 1, 2)])
