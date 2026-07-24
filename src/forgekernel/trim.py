"""K7 — trimmed-patch topology: a surface patch plus parameter-space trim
loops, with **exact** point-in-region classification.

A boolean over freeform solids produces faces that are only *part* of a
surface — the region bounded by intersection curves. We represent that as
the untrimmed patch (any surface exposing ``domain()``/``eval()``) plus a
set of trim loops in the patch's (u, v) parameter domain:

    loops[0]  — the outer boundary
    loops[1:] — holes punched out of it

Trim-loop vertices carry exact rational (u, v) coordinates: they come from
certified SSI points (:func:`forgekernel.ssi.ssi_curves`) and patch-corner
vertices, both already in ℚ. So the point-in-region test — even-odd ray
parity — is decided in ℚ with **no tolerance**. That exactness is the whole
point: which side of a trim boundary a point lies on is a *topological*
decision, and ADR-0019 forbids a float from making it. A point that lands
exactly on the boundary is reported as ``"on"`` (also an exact predicate),
never silently bucketed into "in" or "out".

The even-odd rule classifies correctly regardless of each loop's winding
(a point inside a hole crosses outer+inner = even ⇒ out); loop orientation
matters only for the signed parameter-domain area, not for containment.

Not modeled here (deliberately): the *surface* area/volume of the trimmed
region. Its integrand is the surface Jacobian and the trim curve is only
polyline-approximated in parameter space, so that measure is not exact —
it belongs to the later Green's-over-trim-loops step, not this type.
"""

from __future__ import annotations

from fractions import Fraction as F


def _as_uv(pt):
    return (F(pt[0]), F(pt[1]))


def _orient(a, b, c):
    """Sign of the cross product (b−a)×(c−a): +1 left, −1 right, 0 collinear.
    Exact in ℚ."""
    d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (d > 0) - (d < 0)


def _segments_properly_cross(a, b, c, d) -> bool:
    """Do open segments a→b and c→d cross at an interior point? Exact ℚ, and
    strict: shared endpoints or collinear touching do NOT count (those are
    legal hole-touches-outer contacts, not a boundary crossing)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 \
        and o1 != o2 and o3 != o4


class TrimmedPatch:
    """A surface restricted to the region its trim loops enclose.

    ``surface`` is any object with ``domain()`` -> ((u0,u1),(v0,v1)) and
    ``eval(u, v)``. ``loops`` is a list of loops, each an ordered list of
    (u, v) vertices (the closing edge back to the first vertex is implicit);
    ``loops[0]`` is the outer boundary and the rest are holes."""

    def __init__(self, surface, loops) -> None:
        self.surface = surface
        self.loops = [[_as_uv(p) for p in loop] for loop in loops]
        if not self.loops:
            raise ValueError("trimmed patch needs at least an outer loop")
        for i, loop in enumerate(self.loops):
            if len(loop) < 3:
                raise ValueError(f"loop {i}: a trim loop needs >= 3 vertices")

    # -- exact predicates ------------------------------------------------------

    @staticmethod
    def _on_segment(u, v, a, b) -> bool:
        """Is (u, v) exactly on the closed segment a→b? (exact in ℚ)."""
        cross = (b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0])
        if cross != 0:
            return False
        # collinear — inside the segment's bounding box?
        return (min(a[0], b[0]) <= u <= max(a[0], b[0])
                and min(a[1], b[1]) <= v <= max(a[1], b[1]))

    def classify(self, u, v) -> str:
        """Exact point-in-region: ``"in"`` (strict interior), ``"on"`` (on a
        trim edge), or ``"out"``. All decided in ℚ — no tolerance."""
        u, v = F(u), F(v)
        # boundary first: an on-edge point is neither in nor out
        for loop in self.loops:
            m = len(loop)
            for k in range(m):
                if self._on_segment(u, v, loop[k], loop[(k + 1) % m]):
                    return "on"
        # even-odd parity of a +u ray at height v, over every loop's edges
        inside = False
        for loop in self.loops:
            m = len(loop)
            for k in range(m):
                a, b = loop[k], loop[(k + 1) % m]
                if (a[1] > v) != (b[1] > v):
                    # u of the edge's crossing with the horizontal line at v
                    x_int = a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                    if u < x_int:
                        inside = not inside
        return "in" if inside else "out"

    def contains(self, u, v) -> bool:
        """True iff (u, v) is in the strict interior of the trimmed region."""
        return self.classify(u, v) == "in"

    # -- exact volume contribution (K7 flux over the trim loops) ---------------

    def flux(self):
        """This trimmed face's exact flux contribution
        (1/3)∮∮_D S·(S_u×S_v) du dv to the enclosed solid volume, via Green's
        theorem over the trim loops (:func:`forgekernel.bsolid.trimmed_patch_flux`).
        Requires a polynomial surface. Loops are winding-normalized first so
        the outer/hole orientation is correct. Summing ``flux`` over the
        outward-oriented faces of a closed shell gives the solid volume."""
        from forgekernel.bsolid import trimmed_patch_flux
        n = self.normalized()
        return trimmed_patch_flux(n.surface, n.loops)

    # -- exact measures --------------------------------------------------------

    @staticmethod
    def _loop_signed_area(loop):
        """Shoelace signed area (exact ℚ); CCW positive."""
        m = len(loop)
        s = F(0)
        for k in range(m):
            a, b = loop[k], loop[(k + 1) % m]
            s += a[0] * b[1] - b[0] * a[1]
        return s / 2

    def signed_area(self):
        """Exact signed parameter-domain area = outer + Σ holes, using each
        loop's given winding (outer CCW +, holes CW −). This is the (u, v)
        measure of the region, NOT the surface area — see the module note."""
        return sum((self._loop_signed_area(loop) for loop in self.loops), F(0))

    def area(self):
        """Exact unsigned parameter-domain area: |outer| − Σ|holes|."""
        loops = self.loops
        outer = abs(self._loop_signed_area(loops[0]))
        holes = sum(abs(self._loop_signed_area(h)) for h in loops[1:])
        return outer - holes

    # -- orientation / validation ---------------------------------------------

    def is_ccw(self, index: int = 0) -> bool:
        return self._loop_signed_area(self.loops[index]) > 0

    def normalized(self) -> "TrimmedPatch":
        """Return a copy with canonical winding: outer CCW, holes CW."""
        out = []
        for i, loop in enumerate(self.loops):
            ccw = self._loop_signed_area(loop) > 0
            want_ccw = (i == 0)
            out.append(list(loop) if ccw == want_ccw else list(reversed(loop)))
        return TrimmedPatch(self.surface, out)

    def validate(self) -> None:
        """Structural checks: every vertex inside the surface's parameter
        domain, holes strictly inside the outer boundary."""
        (u0, u1), (v0, v1) = self.surface.domain()
        for i, loop in enumerate(self.loops):
            for (u, v) in loop:
                if not (u0 <= u <= u1 and v0 <= v <= v1):
                    raise ValueError(
                        f"loop {i}: vertex ({u},{v}) outside surface domain")
        outer = TrimmedPatch(self.surface, [self.loops[0]])
        outer_loop = self.loops[0]
        no = len(outer_loop)
        for i, hole in enumerate(self.loops[1:], start=1):
            # WHOLE-polygon containment, not a single sample point: every hole
            # vertex must be in-or-on the outer boundary AND no hole edge may
            # properly cross an outer edge. Testing only the vertex average
            # accepts holes that stick out through the boundary and rejects
            # valid holes whose average lands in a concavity (reviewed 2026-07).
            for (u, v) in hole:
                if outer.classify(u, v) == "out":
                    raise ValueError(f"hole {i} is not inside the outer boundary")
            mh = len(hole)
            for a in range(mh):
                h0, h1 = hole[a], hole[(a + 1) % mh]
                for b in range(no):
                    o0, o1 = outer_loop[b], outer_loop[(b + 1) % no]
                    if _segments_properly_cross(h0, h1, o0, o1):
                        raise ValueError(
                            f"hole {i} crosses the outer boundary")
