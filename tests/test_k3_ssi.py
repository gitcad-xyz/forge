"""K3.3 — SSI: complete branch detection + certified refinement.

Ground-truth cases (constructed so the true branch count is known):
plane z=0 against quadratic sheets whose zero sets are exactly two
lines, one line, nothing, or a tangential line. The tangential case is
the differentiation moment: OCCT's GeomAPI_IntSS returns 0 lines for
it (verified in gitcad's oracle suite); forge finds the branch.
"""

from __future__ import annotations

from fractions import Fraction

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
