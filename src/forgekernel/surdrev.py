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


class NapkinRing:
    """A sphere with a COAXIAL cylindrical bore drilled clean through it.

    Two faces, and that is the whole solid: one spherical ZONE (the sphere's
    surface between the two circles where the bore exits) and one cylindrical
    bore wall. There are no flat annular caps — the bore removes the sphere's
    polar caps entirely, so nothing planar survives. A rectangle-shaped mental
    model of the profile gets this wrong and invents two caps that are not
    there; the volume it produces is plausible and incorrect.

    Exact in ℚ[√d][π] via `contour_r2_dz` — see this module's docstring. The
    volume famously depends only on the band height, not on R and r
    separately, which makes it an unusually good oracle: two different
    (R, r) pairs with the same R²−r² must give the SAME answer, and that is
    checked in the tests.
    """

    __slots__ = ("R", "r", "cx", "cy", "cz")

    def __init__(self, R, r, cx=0, cy=0, cz=0) -> None:
        from fractions import Fraction as F

        self.R, self.r = F(R), F(r)
        self.cx, self.cy, self.cz = F(cx), F(cy), F(cz)
        if not (0 <= self.r < self.R):
            raise ValueError(
                f"napkin ring needs 0 <= bore {self.r} < sphere {self.R}")

    # --- exact -------------------------------------------------------------
    def _half_height(self):
        from forgekernel.surd import SurdVal

        return SurdVal(0, 1, int(self.R * self.R - self.r * self.r))

    def volume(self):
        from forgekernel.quadric import PiVal

        v3 = contour_r2_dz(napkin_ring_contour(self.R, self.r))
        try:                       # rational band height → stays a PiVal
            return PiVal(0, __import__("fractions").Fraction(v3))
        except (TypeError, ValueError):
            from forgekernel.polypi import PiPoly

            return PiPoly([0, v3])          # a + b·π with b in ℚ[√d]

    def centroid(self):
        """EXACT, and exactly the centre — a napkin ring is symmetric about
        its own centre in all three axes, so no integral is needed and none
        should be invented."""
        return (self.cx, self.cy, self.cz)

    def centroid_f(self) -> tuple[float, float, float]:
        return (float(self.cx), float(self.cy), float(self.cz))

    def bbox(self):
        """The RING's extent, not the sphere's. In z the solid stops at the
        band height ±√(R²−r²), because the polar caps are gone — reporting the
        sphere's ±R here would be a loose bound, and reporting ±r would be an
        unsound one."""
        h = float(self._half_height())
        R, cx, cy, cz = float(self.R), float(self.cx), float(self.cy), float(self.cz)
        return ((cx - R, cy - R, cz - h), (cx + R, cy + R, cz + h))

    def translated(self, x, y, z) -> "NapkinRing":
        return NapkinRing(self.R, self.r,
                          self.cx + x, self.cy + y, self.cz + z)

    # --- display -----------------------------------------------------------
    def tessellate(self, deflection: float = 0.2) -> dict:
        """Floats are legal here — this is meshing (ADR-0019).

        The profile is a LENS: the meridian arc from (r, −h) up over the bulge
        and back to (r, +h), closed by the straight bore wall. Both arc
        endpoints land exactly on r = the bore radius, because
        √(R² − h²) = r by construction — so the loop closes with no cap.
        """
        import math

        from forgekernel.tess import lathe

        R, r0 = float(self.R), float(self.r)
        h = float(self._half_height())
        # sample the arc finely enough that its sagitta is under `deflection`
        span = 2 * math.asin(min(1.0, h / R)) if R else 0.0
        n = max(6, int(math.ceil(span / (2 * math.acos(
            max(-1.0, min(1.0, 1 - deflection / R)))))) if R > deflection else 24)
        prof = []
        for i in range(n + 1):
            z = -h + 2 * h * i / n
            prof.append((math.sqrt(max(0.0, R * R - z * z)), z + float(self.cz)))
        prof.append((r0, h + float(self.cz)))
        prof.append((r0, -h + float(self.cz)))
        return lathe(prof, deflection, float(self.cx), float(self.cy))
