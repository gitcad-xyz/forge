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
    R2, r0 = F(sphere_r) ** 2, F(bore_r)
    a = R2 - r0 * r0
    if a <= 0:
        raise ValueError("bore is not narrower than the sphere")
    h = _band_half_height(a)
    return [("arc", R2, -h, h),                       # meridian, upward
            ("line", (r0, h), (r0, -h))]              # bore wall, downward


def _band_half_height(a, collapse_square: bool = True):
    """Exact √a as a ``SurdVal``, for the height where a bore meets a sphere.

    This used to be ``SurdVal(0, 1, int(a))`` for any non-square a — and
    ``int()`` TRUNCATES, so a fractional band height (sphere R=3/2, bore r=1:
    a = 5/4) was silently rebuilt as √1. Volume 1.6% light, watertight, through
    every structural check: the backlog's own defect class. ``sqrt_rational``
    is exact over all of ℚ≥0 and fixes the fractional case.

    For INTEGER a the two historical representations are preserved bit for
    bit, because ``SurdVal`` equality is representation-sensitive across
    radicands (√45 ≠ 3·√5 to ``__eq__``) and existing napkin-ring bodies,
    documents and test literals carry them: the unnormalised ``(0, 1, a)``
    always, except that ``collapse_square`` (the contour's convention)
    collapses a perfect square to its rational value.

    Always a ``SurdVal`` (possibly with a zero surd part), because callers
    feed it straight into arithmetic that must stay in one representation per
    solid — and test_surdrev pins the type.
    """
    import math

    from forgekernel.surd import SurdVal, sqrt_rational

    a = F(a)
    if a.denominator == 1:
        if collapse_square and _is_square(a):
            return SurdVal(math.isqrt(a.numerator), 0, 1)
        return SurdVal(0, 1, int(a))
    h = sqrt_rational(a)
    return h if isinstance(h, SurdVal) else SurdVal(h, 0, 1)


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
        # via _band_half_height, NOT SurdVal(0, 1, int(...)): int() truncates
        # a fractional band height, and the rims of the canonical body then
        # sit at the wrong z — see test_a_fractional_band_converts_at_the_
        # right_height. collapse_square=False keeps this method's historical
        # representation for every integer band (test literals pin √64 as-is).
        return _band_half_height(self.R * self.R - self.r * self.r,
                                 collapse_square=False)

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

        SEAM NOTE: unused by the seam since #124 — the canonical Body is the
        one mesher there. Kept because the representation promises to draw
        itself (test_the_representation_itself_meshes_inside_its_own_bbox).

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


def blind_bore_contour(sphere_r, bore_r, floor):
    """The profile of a sphere with a coaxial BLIND bore entering from +z.

    ``floor`` is the bore's flat bottom, RELATIVE to the sphere centre, and
    must sit strictly inside the band: floor² < R² − r². The profile is the
    meridian arc from the surviving south pole up to where the bore exits,
    the bore wall down to the floor, the flat floor in to the axis, and the
    axis itself (zero radius, zero contribution — kept so the loop reads as
    closed).

    Unlike the napkin ring there IS a flat face: the tool's own bottom cap.
    An earlier backlog note guessed the floor was a spherical cap — it is a
    PLANE, because the sphere's surface at the floor depth lies outside the
    bore for every admissible floor.
    """
    R, r0, f = F(sphere_r), F(bore_r), F(floor)
    R2 = R * R
    a = R2 - r0 * r0
    if not (0 < r0 < R):
        raise ValueError(
            f"blind bore needs 0 < bore {r0} < sphere {R}")
    if f * f >= a:
        raise ValueError(
            f"blind-bore floor at {f} is not strictly inside the band "
            f"|z| < sqrt({a}) — at the edge the wall vanishes, past it the "
            "topology is a different solid (through ring or untouched cap)")
    # a NEW type carries no legacy representation: fully normalised square-free
    # radicand (√45 is 3·√5 here), unlike the napkin ring's historical form
    from forgekernel.surd import sqrt_rational

    h = sqrt_rational(a)
    return [("arc", R2, -R, h),                       # meridian, pole to rim
            ("line", (r0, h), (r0, f)),               # bore wall, downward
            ("line", (r0, f), (F(0), f)),             # flat floor, inward
            ("line", (F(0), f), (F(0), -R))]          # axis (contributes 0)


class SphereBlindBore:
    """A sphere with a coaxial cylindrical bore that enters from +z and STOPS.

    Three faces, and that is the whole solid:

      * the sphere minus ONE polar cap — a single rim at z = cz + √d
        (d = R² − r²); the south pole survives as a singular point on no
        edge, exactly like a pointed cone's apex;
      * the bore wall, from the floor up to the rim;
      * the FLAT DISK floor at z = cz + f — the tool's own bottom cap, a
        plane and not a spherical cap.

    Exact in ℚ[√d][π] via `contour_r2_dz`:

        V = π · ( 2R³/3 + (2/3)·d·√d + r²·f )

    derived from the removed set (the bore cylinder of height √d − f plus the
    sphere's cap above √d) and verified against Monte-Carlo membership
    sampling: R=6, r=1, f=0 gives π(144 + 70√35/3) = 886.0606404251…, banked
    MC 885.969 ± 1.296 (3σ, 4M samples) and re-measured fresh at seed
    987654321 (16M samples, analytic membership: inside the sphere and not in
    the bore's half-infinite cylinder).

    Only the CENTRED case entering from the top. Off-axis the volume acquires
    elliptic integrals and leaves every algebraic extension; a floor at or
    beyond ±√d is a different topology (no wall left / a through ring / an
    untouched cap). All of those refuse in the constructor — a refusal is the
    finished answer, a near-miss silently answered is not.
    """

    __slots__ = ("R", "r", "f", "cx", "cy", "cz")

    def __init__(self, R, r, f, cx=0, cy=0, cz=0) -> None:
        self.R, self.r, self.f = F(R), F(r), F(f)
        self.cx, self.cy, self.cz = F(cx), F(cy), F(cz)
        blind_bore_contour(self.R, self.r, self.f)      # validates R, r, f

    # --- exact -------------------------------------------------------------
    def _half_height(self):
        # normalised square-free form, matching blind_bore_contour — one
        # representation per solid, so rim arithmetic never mixes radicands
        from forgekernel.surd import sqrt_rational

        return sqrt_rational(self.R * self.R - self.r * self.r)

    def volume(self):
        from forgekernel.quadric import PiVal

        v3 = contour_r2_dz(blind_bore_contour(self.R, self.r, self.f))
        try:                       # rational band height → stays a PiVal
            return PiVal(0, __import__("fractions").Fraction(v3))
        except (TypeError, ValueError):
            from forgekernel.polypi import PiPoly

            return PiPoly([0, v3])          # a + b·π with b in ℚ[√d]
    def centroid(self):
        """EXACT. Not the centre: the bore removes material from the top, so
        z̄ = cz + m_z/(V/π) with (by ∫z dV over the profile of revolution)

            m_z = R²d/2 − d²/4 − R⁴/4 − r²(d − f²)/2,   d = R² − r²

        which is RATIONAL (only even powers of √d appear), divided by the
        ℚ[√d] value V/π. x̄ and ȳ are the axis by symmetry."""
        d = self.R * self.R - self.r * self.r
        mz = (self.R * self.R * d / 2 - d * d / 4
              - self.R ** 4 / 4
              - self.r * self.r * (d - self.f * self.f) / 2)
        v3 = contour_r2_dz(blind_bore_contour(self.R, self.r, self.f))
        return (self.cx, self.cy, self.cz + mz / v3)

    def centroid_f(self) -> tuple[float, float, float]:
        c = self.centroid()
        return (float(c[0]), float(c[1]), float(c[2]))

    def bbox(self):
        """Tight: the north cap is gone, so the top is the rim at cz + √d;
        the south pole survives, so the bottom is cz − R. Floats are legal
        here — a bound, not a topology decision (ADR-0019)."""
        h = float(self._half_height())
        R = float(self.R)
        cx, cy, cz = float(self.cx), float(self.cy), float(self.cz)
        return ((cx - R, cy - R, cz - R), (cx + R, cy + R, cz + h))

    def translated(self, x, y, z) -> "SphereBlindBore":
        return SphereBlindBore(self.R, self.r, self.f,
                               self.cx + x, self.cy + y, self.cz + z)

    # --- display -----------------------------------------------------------
    def tessellate(self, deflection: float = 0.2) -> dict:
        """Floats are legal here — this is meshing (ADR-0019). The seam uses
        the canonical Body's mesher; this exists so the representation can
        draw itself, same as NapkinRing."""
        import math

        from forgekernel.tess import lathe

        R, r0, f = float(self.R), float(self.r), float(self.f)
        h = float(self._half_height())
        cz = float(self.cz)
        # meridian from the south pole up to the rim, sagitta-bounded
        lo_ang, hi_ang = -math.pi / 2, math.asin(max(-1.0, min(1.0, h / R)))
        step = 2 * math.acos(max(-1.0, min(1.0, 1 - deflection / R))) \
            if R > deflection else math.pi / 8
        n = max(12, int(math.ceil((hi_ang - lo_ang) / step)))
        prof = [(0.0, cz - R)]
        for i in range(1, n):
            ang = lo_ang + (hi_ang - lo_ang) * i / n
            prof.append((R * math.cos(ang), cz + R * math.sin(ang)))
        prof.append((r0, cz + h))               # the rim, exactly
        prof.append((r0, cz + f))               # down the bore wall
        prof.append((0.0, cz + f))              # the flat floor
        return lathe(prof, deflection, float(self.cx), float(self.cy))
