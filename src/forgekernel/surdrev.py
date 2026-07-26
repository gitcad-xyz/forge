"""Green's-theorem volume for a revolved profile whose edges may be ARCS.

`RevolveSolid` handles profiles of straight segments with rational vertices.
Two things it cannot express, and both are needed by the same family of shapes:

  * an ARC edge — the meridian of a sphere or the slant of a rounded lathe;
  * SURD coordinates — the moment an arc meets a straight edge, the crossing
    lands at z = ±√(R²−r²), which is not rational.

This module supplies the exact volume term for that case, and nothing else. It
is deliberately a free function over an edge list rather than a new solid type:
the arithmetic is the part that had to be proven, and proving it separately
means the eventual representation cannot get the number wrong without this
failing first.

    V = π ∮ r² dz

  * straight edge (r₁,z₁)→(r₂,z₂):  (z₂−z₁)(r₁² + r₁r₂ + r₂²)/3
  * arc on r² + z² = R² (centred at the profile origin), z₁→z₂:
        R²(z₂−z₁) − (z₂³ − z₁³)/3

Both terms are polynomial, so they stay inside whatever exact field the
coordinates live in: ℚ for a rational profile, ℚ[√d] once an arc meets a wall.
With the π factor that is ℚ[√d][π] — the field #117 already built. No new
number field is required for this family, which is the whole point.
"""

from fractions import Fraction as F


def contour_r2_dz(edges) -> object:
    """Exact ∮ r² dz over a closed profile. ``edges`` is a sequence of

        ("line", (r1, z1), (r2, z2))
        ("arc",  R2, z1, z2)        # on r² + z² = R², centred at the origin

    Coordinates may be Fraction or SurdVal; the arithmetic is closed over both.
    Multiply the result by π for the volume.
    """
    acc = F(0)
    for e in edges:
        if e[0] == "line":
            (r1, z1), (r2, z2) = e[1], e[2]
            acc = acc + (z2 - z1) * (r1 * r1 + r1 * r2 + r2 * r2) / 3
        elif e[0] == "arc":
            _, r2sq, z1, z2 = e
            acc = acc + r2sq * (z2 - z1) - (z2 * z2 * z2 - z1 * z1 * z1) / 3
        else:
            raise ValueError(f"unknown profile edge kind {e[0]!r}")
    return acc


def napkin_ring_contour(sphere_r, bore_r):
    """The profile of a sphere of radius ``sphere_r`` with a COAXIAL bore of
    radius ``bore_r`` drilled through it — a napkin ring.

    The profile is a lens, not a rectangle: the bore wall at r = bore_r runs
    between z = ∓√(R²−r²), and the sphere's meridian arc closes it. There are
    NO horizontal caps — the sphere's polar caps are removed entirely by the
    bore, which is the detail a rectangle-shaped mental model gets wrong.
    """
    from forgekernel.surd import SurdVal

    R2, r0 = F(sphere_r) ** 2, F(bore_r)
    a = R2 - r0 * r0
    if a <= 0:
        raise ValueError("bore is not narrower than the sphere")
    h = SurdVal(0, 1, int(a)) if F(a).denominator != 1 or not _is_square(a) \
        else F(a) ** F(1, 2)
    if not isinstance(h, SurdVal):
        h = SurdVal(h, 0, 1)
    return [("arc", R2, -h, h),                       # meridian, upward
            ("line", (r0, h), (r0, -h))]              # bore wall, downward


def _is_square(x) -> bool:
    import math

    n = F(x)
    if n.denominator != 1:
        return False
    r = math.isqrt(n.numerator)
    return r * r == n.numerator
