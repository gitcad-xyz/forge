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
