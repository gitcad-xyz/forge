"""K3.0 — certified intervals + the helix/tube family (ADR-0019).

The first transcendental geometry. Nothing here is exact; everything is
*certified* — bracketed by rationals that provably contain the truth.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from forgekernel.curve import Helix, TubeSolid
from forgekernel.interval import CInterval, pi_interval


# -- the certified number kind -----------------------------------------------

def test_pi_enclosure_is_rigorous_and_tight() -> None:
    pi = pi_interval()
    assert pi.lo < pi.hi
    assert pi.width < Fraction(1, 10) ** 55        # very tight
    # brackets the true value (mid agrees with the double to full precision)
    assert abs(pi.to_float() - math.pi) < 1e-15


def test_sqrt_is_a_certified_bracket() -> None:
    for x in (2, 3, Fraction(1, 7), 1000):
        s = CInterval.exact(x).sqrt()
        assert s.lo * s.lo <= x <= s.hi * s.hi      # the enclosure property
        assert abs(s.to_float() - math.sqrt(x)) < 1e-12


def test_sign_certified_only_when_zero_excluded() -> None:
    assert CInterval(Fraction(1, 100), Fraction(2, 100)).sign() == 1
    assert CInterval(Fraction(-2), Fraction(-1)).sign() == -1
    with pytest.raises(ValueError, match="not certified"):
        CInterval(Fraction(-1), Fraction(1)).sign()


def test_interval_arithmetic_preserves_enclosure() -> None:
    a = CInterval(Fraction(1), Fraction(2))
    b = CInterval(Fraction(3), Fraction(4))
    assert (a + b).lo == 4 and (a + b).hi == 6
    assert (b - a).lo == 1 and (b - a).hi == 3
    assert (a * b).lo == 3 and (a * b).hi == 8


# -- the helix + swept tube (coil spring) ------------------------------------

def test_helix_arc_length_certified() -> None:
    # one turn, R=8, pitch=4: L = √((16π)² + 16)
    h = Helix(8, 4, 1)
    L = h.arc_length()
    want = math.sqrt((2 * math.pi * 8) ** 2 + 4 ** 2)
    assert L.lo <= Fraction(want).limit_denominator(10 ** 6) or \
        abs(L.to_float() - want) < 1e-9
    assert L.width < Fraction(1, 10) ** 40          # tight bracket


def test_spring_volume_matches_analytic_and_is_certified() -> None:
    # the corpus spring: R=8, pitch=4, turns=6, wire d=1.5
    tube = TubeSolid(Helix(8, 4, 6), Fraction(3, 4))
    v = tube.volume()
    rho, L = 0.75, 6 * math.sqrt((2 * math.pi * 8) ** 2 + 16)
    want = math.pi * rho * rho * L
    # the bracket is self-consistent and far tighter than a float (its width
    # is below float epsilon, so an independently-computed double lands
    # OUTSIDE it — the interval is the more precise object, not the float)
    assert v.lo <= v.mid <= v.hi
    assert v.width < Fraction(1, 10) ** 40
    assert abs(v.to_float() - want) < 1e-9      # midpoint == analytic value
    assert v.sign() == 1


def test_tube_rejects_self_overlap() -> None:
    # coils merge: 2·wire_radius >= pitch
    with pytest.raises(ValueError, match="self-overlaps"):
        TubeSolid(Helix(8, 1, 3), Fraction(1))       # 2·1 >= 1
    # section bigger than the spine radius of curvature (1/κ ≈ 9.27 here),
    # while still clearing the coil-merge test (2·9.5 = 19 < pitch 20)
    with pytest.raises(ValueError, match="self-overlaps"):
        TubeSolid(Helix(8, 20, 2), Fraction(19, 2))


def test_helix_rejects_bad_input() -> None:
    for bad in [(0, 4, 1), (8, 0, 1), (8, 4, 0)]:
        with pytest.raises(ValueError):
            Helix(*bad)


def test_tube_bbox_and_centroid_are_tight() -> None:
    tube = TubeSolid(Helix(8, 4, 6), Fraction(3, 4))
    (x0, y0, z0), (x1, y1, z1) = tube.bbox_f()
    assert (x0, x1) == (-8.75, 8.75)                 # R + wire radius
    assert z0 == pytest.approx(-0.75) and z1 == pytest.approx(24.75)
    cx, cy, cz = tube.centroid_f()
    assert (cx, cy) == (0.0, 0.0) and cz == pytest.approx(12.0)  # mid-height


# -- K3.1: NURBS / B-spline curve evaluation via de Boor ----------------------

def test_bezier_de_boor_is_exact_rational() -> None:
    from forgekernel.nurbs import bezier

    P = [(0, 0, 0), (1, 3, 0), (3, 3, 0), (4, 0, 0)]
    c = bezier(P)
    # cubic Bézier midpoint = (P0 + 3P1 + 3P2 + P3)/8, exactly
    mid = c.eval(Fraction(1, 2))
    want = tuple((Fraction(P[0][i]) + 3 * Fraction(P[1][i])
                  + 3 * Fraction(P[2][i]) + Fraction(P[3][i])) / 8
                 for i in range(3))
    assert mid == want
    assert all(isinstance(v, Fraction) for v in mid)     # exact, not float
    # clamped endpoints interpolate the first/last control points
    assert c.eval(0) == tuple(Fraction(v) for v in P[0])
    assert c.eval(1) == tuple(Fraction(v) for v in P[3])


def test_bspline_partition_of_unity() -> None:
    from forgekernel.nurbs import BSplineCurve

    # all control points equal → the curve is that point for every t
    pt = (5, 7, 2)
    c = BSplineCurve(2, [pt, pt, pt, pt], [0, 0, 0, 1, 2, 2, 2])
    for t in (Fraction(1, 5), Fraction(1), Fraction(3, 2)):
        assert c.eval(t) == tuple(Fraction(v) for v in pt)


def test_bspline_interior_knot_span() -> None:
    from forgekernel.nurbs import BSplineCurve

    # a quadratic B-spline with one interior knot; a straight collinear
    # control polygon must stay exactly on the line y = x
    c = BSplineCurve(2, [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0)],
                     [0, 0, 0, 1, 2, 2, 2])
    for t in (Fraction(1, 2), Fraction(1), Fraction(3, 2)):
        x, y, _ = c.eval(t)
        assert x == y


def test_nurbs_circle_is_certifiably_on_the_circle() -> None:
    from forgekernel.nurbs import BSplineCurve

    # true quarter circle: rational control points, irrational weight √2/2
    half_s2 = CInterval.exact(2).sqrt() * CInterval.exact(Fraction(1, 2))
    qc = BSplineCurve(2, [(1, 0, 0), (1, 1, 0), (0, 1, 0)], [0, 0, 0, 1, 1, 1],
                      weights=[Fraction(1), half_s2, Fraction(1)])
    # exact eval is unavailable (irrational weight); certified eval is
    with pytest.raises(ValueError, match="irrational weights"):
        qc.eval(Fraction(1, 2))
    px, py, _ = qc.eval_ci(Fraction(1, 2))
    r2 = px * px + py * py
    assert r2.lo <= 1 <= r2.hi                # certifiably on the unit circle
    assert r2.width < Fraction(1, 10) ** 40
    # rational endpoints come back exact
    assert tuple(round(c.to_float(), 12) for c in qc.eval_ci(0)) == (1.0, 0.0, 0.0)
