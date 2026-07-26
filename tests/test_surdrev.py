"""The arc term in a revolved profile's volume, proven before it has a home.

This is the mathematics that blocks four capability cells (sphere x cut
cylinder, cone x shell, cone x fillet(all), cone x chamfer(all)). It is tested
on its own, ahead of the representation that will carry it, because the
arithmetic is the part that goes silently wrong — a wrong Green term produces a
plausible positive volume that no topology check can catch.

The oracle is Steiner-free and independent: a napkin ring (a sphere with a
coaxial bore drilled through) has the closed form

    V = (4/3) * pi * (R^2 - r^2)^(3/2)

which famously does not depend on R and r separately, only on the band height.
"""

import math

import pytest

from forgekernel.surd import SurdVal
from forgekernel.surdrev import contour_r2_dz, napkin_ring_contour

RINGS = [(6, 1), (10, 6), (5, 3), (13, 5), (2, 1)]


@pytest.mark.parametrize("R,r0", RINGS, ids=[f"R{R}r{r}" for R, r in RINGS])
def test_napkin_ring_volume_matches_its_closed_form(R, r0) -> None:
    v3 = contour_r2_dz(napkin_ring_contour(R, r0))
    assert float(v3) * math.pi == pytest.approx(
        (4 / 3) * math.pi * (R * R - r0 * r0) ** 1.5, rel=1e-12)


@pytest.mark.parametrize("R,r0", RINGS, ids=[f"R{R}r{r}" for R, r in RINGS])
def test_the_answer_stays_exact_and_never_becomes_a_float(R, r0) -> None:
    """The point of the exercise. A float here would pass the numeric test
    above while quietly leaving the exact field, which is what ADR-0019
    forbids — and it is invisible unless the TYPE is asserted."""
    v3 = contour_r2_dz(napkin_ring_contour(R, r0))
    assert isinstance(v3, SurdVal), f"left the exact field: {type(v3).__name__}"


def test_a_straight_only_profile_still_agrees_with_RevolveSolid() -> None:
    """The new code must not have redefined the existing term. A cylinder is
    the simplest profile both can express."""
    from fractions import Fraction as F

    from forgekernel.quadric import RevolveSolid

    r, z0, z1 = F(5), F(0), F(12)
    loop = [(r, z0), (r, z1), (F(0), z1), (F(0), z0)]
    edges = [("line", loop[i], loop[(i + 1) % 4]) for i in range(4)]
    assert float(contour_r2_dz(edges)) * math.pi == pytest.approx(
        float(RevolveSolid(loop).volume()), rel=1e-15)
    assert float(contour_r2_dz(edges)) * math.pi == pytest.approx(
        math.pi * 25 * 12, rel=1e-15)


def test_a_fractional_band_height_is_not_truncated_to_an_integer() -> None:
    """Backlog defect class: silent wrong number, found while closing the
    blind-bore cell. `napkin_ring_contour` built its half-height as
    ``SurdVal(0, 1, int(a))`` whenever R² − r² was not an integer — and
    ``int(5/4)`` is 1, so a sphere of R=3/2 bored at r=1 reported √1 where
    √(5/4) belongs: volume 5.7596 against a true 5.8540, 1.6% light, through
    every structural check. The closed form (4/3)π·d^(3/2) with d = 5/4 is
    (5/6)√5·π, written here as a literal."""
    from fractions import Fraction as F

    v3 = contour_r2_dz(napkin_ring_contour(F(3, 2), 1))
    assert v3 == SurdVal(0, F(5, 6), 5), f"got {v3!r}, want (5/6)·√5"
    assert float(v3) * math.pi == pytest.approx(
        (4 / 3) * math.pi * (9 / 4 - 1) ** 1.5, rel=1e-12)


def test_a_bore_as_wide_as_the_sphere_is_refused_not_zeroed() -> None:
    with pytest.raises(ValueError):
        napkin_ring_contour(5, 5)
    with pytest.raises(ValueError):
        napkin_ring_contour(5, 6)


def test_an_unknown_edge_kind_raises_rather_than_being_skipped() -> None:
    """A silently ignored edge is a silently wrong volume."""
    with pytest.raises(ValueError, match="unknown profile edge kind"):
        contour_r2_dz([("bezier", 1, 2, 3)])
