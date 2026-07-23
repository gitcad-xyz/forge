"""K2.0 — z-axis cylinders and drilled solids, exact in ℚ[π].

The exactness charter survives curved geometry by extending the number
field: volumes of prisms with cylindrical bores live in ℚ + ℚ·π, so a
drilled plate's volume is EXACTLY ``9600 - 100π`` — an object with
equality, not a float. Floats appear only at the export boundary.

Scope, honestly held: right circular cylinders with +z axes — the
drilled-hole workhorse (plain, blind, and coaxial counterbore stacks).
Every geometric precondition is checked with exact rational predicates
(bore strictly inside the lateral boundary, non-coaxial bores disjoint);
configurations outside the scope refuse with the stage that brings them
(K2.1 general quadric booleans).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from forgekernel.brep import Solid
from forgekernel.exact import F


class PiVal:
    """An exact number a + b·π (a, b rational)."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0) -> None:
        self.a, self.b = F(a), F(b)

    def __add__(self, o: "PiVal | int | Fraction") -> "PiVal":
        o = o if isinstance(o, PiVal) else PiVal(o)
        return PiVal(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, o: "PiVal | int | Fraction") -> "PiVal":
        o = o if isinstance(o, PiVal) else PiVal(o)
        return PiVal(self.a - o.a, self.b - o.b)

    def __eq__(self, o: object) -> bool:
        o = o if isinstance(o, PiVal) else PiVal(o)
        return self.a == o.a and self.b == o.b

    def __float__(self) -> float:
        return float(self.a) + float(self.b) * math.pi

    def __repr__(self) -> str:
        return f"({self.a} + {self.b}·π)"


@dataclass(frozen=True)
class Cyl:
    """A solid right circular cylinder, axis +z through (cx, cy)."""
    cx: Fraction
    cy: Fraction
    r: Fraction
    z0: Fraction
    z1: Fraction

    @classmethod
    def make(cls, r, h) -> "Cyl":
        r, h = F(r), F(h)
        if r <= 0 or h <= 0:
            raise ValueError("cylinder wants positive radius/height")
        return cls(F(0), F(0), r, F(0), h)

    def translated(self, x, y, z) -> "Cyl":
        return Cyl(self.cx + F(x), self.cy + F(y), self.r,
                   self.z0 + F(z), self.z1 + F(z))

    def volume(self) -> PiVal:
        return PiVal(0, self.r * self.r * (self.z1 - self.z0))

    def centroid_f(self) -> tuple[float, float, float]:
        return (float(self.cx), float(self.cy),
                float((self.z0 + self.z1) / 2))

    def bbox(self):
        return ((self.cx - self.r, self.cy - self.r, self.z0),
                (self.cx + self.r, self.cy + self.r, self.z1))


def _dist2_point_seg(px, py, ax, ay, bx, by) -> Fraction:
    """Exact squared distance from point to segment (all rational)."""
    dx, dy = bx - ax, by - ay
    nn = dx * dx + dy * dy
    if nn == 0:
        ex, ey = px - ax, py - ay
        return ex * ex + ey * ey
    t = ((px - ax) * dx + (py - ay) * dy) / nn
    t = max(F(0), min(F(1), t))
    ex, ey = px - (ax + t * dx), py - (ay + t * dy)
    return ex * ex + ey * ey


class DrilledSolid:
    """A planar Solid minus z-axis cylindrical bores — exact composite.

    Preconditions (exact predicates, refusal on violation):
    - each bore's circle stays strictly clear of every non-horizontal
      face of the base in xy (the barrel never crosses a wall);
    - non-coaxial bores are pairwise disjoint (coaxial stacks allowed —
      counterbores); volume of a coaxial stack is the exact z-interval
      union with the largest active radius per interval.
    """

    def __init__(self, base: Solid, bores: list[Cyl]) -> None:
        self.base = base
        self.bores = list(bores)

    def cut(self, c: Cyl) -> "DrilledSolid":
        # clamp to the base z-extent (drilling from above through air is fine)
        (bx0, by0, bz0), (bx1, by1, bz1) = self.base.bbox()
        z0, z1 = max(c.z0, bz0), min(c.z1, bz1)
        if z1 <= z0:
            raise ValueError("bore misses the solid in z (K2.1 for the rest)")
        c = Cyl(c.cx, c.cy, c.r, z0, z1)
        r2 = c.r * c.r
        for p in self.base.polys:
            n = p.plane.n
            if n[0] == 0 and n[1] == 0:
                continue                       # horizontal face: cap, fine
            m = len(p.verts)
            for i in range(m):
                a, b = p.verts[i], p.verts[(i + 1) % m]
                if _dist2_point_seg(c.cx, c.cy, a[0], a[1], b[0], b[1]) <= r2:
                    raise ValueError(
                        "bore crosses a lateral wall — general quadric "
                        "booleans arrive at K2.1")
        for o in self.bores:
            if o.cx == c.cx and o.cy == c.cy:
                continue                       # coaxial stack (counterbore)
            dx, dy = o.cx - c.cx, o.cy - c.cy
            if dx * dx + dy * dy <= (o.r + c.r) ** 2:
                raise ValueError(
                    "bores intersect — general quadric booleans arrive "
                    "at K2.1")
        return DrilledSolid(self.base, self.bores + [c])

    def _bore_union_volume(self) -> PiVal:
        """Exact removed volume: coaxial groups unioned by z-interval with
        the largest active radius per elementary interval."""
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for c in self.bores:
            groups[(c.cx, c.cy)].append(c)
        total = PiVal(0, 0)
        for cyls in groups.values():
            cuts = sorted({t for c in cyls for t in (c.z0, c.z1)})
            for lo, hi in zip(cuts, cuts[1:]):
                rs = [c.r for c in cyls if c.z0 <= lo and hi <= c.z1]
                if rs:
                    rmax = max(rs)
                    total = total + PiVal(0, rmax * rmax * (hi - lo))
        return total

    def volume(self) -> PiVal:
        return PiVal(self.base.volume(), 0) - self._bore_union_volume()

    def centroid_f(self) -> tuple[float, float, float]:
        """Centroid, floated at the boundary (the exact value is a ratio
        of ℚ[π] numbers — outside the field, so floats are honest here)."""
        bv = float(self.base.volume())
        c = self.base.centroid()
        acc = [bv * float(c[0]), bv * float(c[1]), bv * float(c[2])]
        vol = bv
        for cyl in self.bores:
            v = float(cyl.volume())
            cc = cyl.centroid_f()
            for i in range(3):
                acc[i] -= v * cc[i]
            vol -= v
        return (acc[0] / vol, acc[1] / vol, acc[2] / vol)

    def bbox(self):
        return self.base.bbox()

    def watertight_violations(self) -> list[str]:
        return self.base.watertight_violations()

    def cylinder_faces(self) -> list[dict]:
        """OCCT-shaped descriptors, one per bore — feature recognition and
        hole callouts read these keys."""
        return [{"surface": "cylinder", "radius": float(c.r),
                 "axis_dir": [0.0, 0.0, 1.0],
                 "axis_origin": [float(c.cx), float(c.cy), float(c.z0)]}
                for c in self.bores]


@dataclass(frozen=True)
class Sphere:
    """Solid sphere centered (cx, cy, cz)."""
    cx: Fraction
    cy: Fraction
    cz: Fraction
    r: Fraction

    @classmethod
    def make(cls, r) -> "Sphere":
        r = F(r)
        if r <= 0:
            raise ValueError("sphere wants positive radius")
        return cls(F(0), F(0), F(0), r)

    def translated(self, x, y, z) -> "Sphere":
        return Sphere(self.cx + F(x), self.cy + F(y), self.cz + F(z), self.r)


@dataclass(frozen=True)
class Cone:
    """Right conical frustum, axis +z through (cx, cy), r1 at z0, r2 at z1."""
    cx: Fraction
    cy: Fraction
    r1: Fraction
    r2: Fraction
    z0: Fraction
    z1: Fraction

    @classmethod
    def make(cls, r1, r2, h) -> "Cone":
        r1, r2, h = F(r1), F(r2), F(h)
        if h <= 0 or r1 < 0 or r2 < 0 or (r1 == 0 and r2 == 0):
            raise ValueError("cone wants positive height and a radius")
        return cls(F(0), F(0), r1, r2, F(0), h)

    def translated(self, x, y, z) -> "Cone":
        return Cone(self.cx + F(x), self.cy + F(y), self.r1, self.r2,
                    self.z0 + F(z), self.z1 + F(z))


class _Quad:
    """Exact quadratic q(z) = a z^2 + b z + c — the r^2 profile of every
    K2 primitive (cylinder: constant; cone: squared linear; sphere:
    R^2 - (z-c)^2)."""

    __slots__ = ("a", "b", "c")

    def __init__(self, a, b, c) -> None:
        self.a, self.b, self.c = F(a), F(b), F(c)

    def at(self, z: Fraction) -> Fraction:
        return self.a * z * z + self.b * z + self.c

    def integral(self, lo: Fraction, hi: Fraction) -> Fraction:
        return (self.a * (hi ** 3 - lo ** 3) / 3
                + self.b * (hi ** 2 - lo ** 2) / 2 + self.c * (hi - lo))

    def z_integral(self, lo: Fraction, hi: Fraction) -> Fraction:
        """Integral of z*q(z)."""
        return (self.a * (hi ** 4 - lo ** 4) / 4
                + self.b * (hi ** 3 - lo ** 3) / 3
                + self.c * (hi ** 2 - lo ** 2) / 2)

    def rational_roots_between(self, lo: Fraction, hi: Fraction,
                               other: "_Quad") -> list[Fraction] | None:
        """Roots of (self - other) strictly inside (lo, hi): the list when
        every such root is rational, None when an IRRATIONAL crossover may
        exist in range (the caller refuses — exactness is never faked)."""
        a, b, c = self.a - other.a, self.b - other.b, self.c - other.c
        if a == 0:
            if b == 0:
                return []
            z = -c / b
            return [z] if lo < z < hi else []
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        num, den = disc.numerator, disc.denominator
        rn, rd = math.isqrt(num), math.isqrt(den)
        if rn * rn != num or rd * rd != den:
            # irrational roots: exact interval sign analysis decides if any
            # lie in (lo, hi); vertex of the difference is at -b/2a
            vz = -b / (2 * a)
            s_lo = a * lo * lo + b * lo + c
            s_hi = a * hi * hi + b * hi + c
            crosses = (s_lo < 0) != (s_hi < 0)
            if not crosses and lo < vz < hi:
                s_v = a * vz * vz + b * vz + c
                crosses = (s_v < 0) != (s_lo < 0) and s_v != 0
            return None if crosses else []
        sq = Fraction(rn, rd)
        roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        return [z for z in roots if lo < z < hi]


def _segments_of(prim) -> list:
    if isinstance(prim, Cyl):
        return [(prim.z0, prim.z1, _Quad(0, 0, prim.r * prim.r))]
    if isinstance(prim, Cone):
        h = prim.z1 - prim.z0
        k = (prim.r2 - prim.r1) / h
        a = k * k
        b = 2 * prim.r1 * k - 2 * k * k * prim.z0
        c = (prim.r1 - k * prim.z0) ** 2
        return [(prim.z0, prim.z1, _Quad(a, b, c))]
    if isinstance(prim, Sphere):
        return [(prim.cz - prim.r, prim.cz + prim.r,
                 _Quad(-1, 2 * prim.cz, prim.r * prim.r - prim.cz * prim.cz))]
    raise TypeError(f"not an axis primitive: {type(prim).__name__}")


class AxisStack:
    """Union of coaxial z-axis primitives — exact in the field of a+b*pi.

    Concentric circles make the union area pi*max_i r_i(z)^2, and every
    r^2 profile is a rational quadratic, so the union integrates exactly
    piecewise. Profile crossovers must land on rational z; a possible
    irrational crossover refuses honestly (K2.2 brings the algebraic
    extension)."""

    def __init__(self, cx, cy, prims: list) -> None:
        self.cx, self.cy = F(cx), F(cy)
        self.prims = list(prims)

    def fuse(self, prim) -> "AxisStack":
        if getattr(prim, "cx", None) != self.cx or \
           getattr(prim, "cy", None) != self.cy:
            raise ValueError("union of non-coaxial quadrics arrives at K2.2")
        return AxisStack(self.cx, self.cy, self.prims + [prim])

    def _pieces(self) -> list:
        segs = [s for p in self.prims for s in _segments_of(p)]
        cuts = {t for lo, hi, _ in segs for t in (lo, hi)}
        for i, (lo1, hi1, q1) in enumerate(segs):
            for lo2, hi2, q2 in segs[i + 1:]:
                lo, hi = max(lo1, lo2), min(hi1, hi2)
                if lo < hi:
                    roots = q1.rational_roots_between(lo, hi, q2)
                    if roots is None:
                        raise ValueError(
                            "irrational profile crossover arrives at K2.2 "
                            "(algebraic extension)")
                    cuts.update(roots)
        ordered = sorted(cuts)
        out = []
        for lo, hi in zip(ordered, ordered[1:]):
            mid = (lo + hi) / 2
            live = [q for slo, shi, q in segs if slo <= lo and hi <= shi]
            if not live:
                continue
            best = max(live, key=lambda q: q.at(mid))
            out.append((lo, hi, best))
        return out

    def volume(self) -> PiVal:
        return PiVal(0, sum((q.integral(lo, hi)
                             for lo, hi, q in self._pieces()), F(0)))

    def centroid_f(self) -> tuple[float, float, float]:
        pieces = self._pieces()
        v = sum((q.integral(lo, hi) for lo, hi, q in pieces), F(0))
        zbar = sum((q.z_integral(lo, hi) for lo, hi, q in pieces), F(0)) / v
        return (float(self.cx), float(self.cy), float(zbar))

    def bbox(self):
        pieces = self._pieces()
        z0 = min(lo for lo, _, _ in pieces)
        z1 = max(hi for _, hi, _ in pieces)
        r2max = F(0)
        for lo, hi, q in pieces:
            cands = [q.at(lo), q.at(hi)]
            if q.a != 0:
                vz = -q.b / (2 * q.a)
                if lo <= vz <= hi:
                    cands.append(q.at(vz))
            r2max = max(r2max, *cands)
        r = math.sqrt(float(r2max))
        return ((float(self.cx) - r, float(self.cy) - r, float(z0)),
                (float(self.cx) + r, float(self.cy) + r, float(z1)))


class RevolveSolid:
    """A closed line-segment profile in the (r, z) half-plane revolved
    360 degrees about the z axis. Green gives exact metrics:
    V = pi * contour_integral(r^2 dz), each edge contributing
    (z2-z1)(r1^2 + r1 r2 + r2^2)/3."""

    def __init__(self, loop_rz: list) -> None:
        self.loop = [(F(r), F(z)) for r, z in loop_rz]
        if any(r < 0 for r, _ in self.loop):
            raise ValueError("revolve profile must stay at r >= 0")
        if self._v3() < 0:
            self.loop = list(reversed(self.loop))

    def _edges(self):
        n = len(self.loop)
        for i in range(n):
            yield self.loop[i], self.loop[(i + 1) % n]

    def _v3(self) -> Fraction:
        acc = F(0)
        for (r1, z1), (r2, z2) in self._edges():
            acc += (z2 - z1) * (r1 * r1 + r1 * r2 + r2 * r2)
        return acc

    def volume(self) -> PiVal:
        return PiVal(0, self._v3() / 3)

    def centroid_f(self) -> tuple[float, float, float]:
        num = F(0)
        for (r1, z1), (r2, z2) in self._edges():
            dz = z2 - z1
            num += dz * (z1 * (3 * r1 * r1 + 2 * r1 * r2 + r2 * r2)
                         + z2 * (r1 * r1 + 2 * r1 * r2 + 3 * r2 * r2)) / 12
        v3 = self._v3()
        return (0.0, 0.0, float(num / v3 * 3) if v3 else 0.0)

    def bbox(self):
        rmax = max(r for r, _ in self.loop)
        zs = [z for _, z in self.loop]
        return ((-float(rmax), -float(rmax), float(min(zs))),
                (float(rmax), float(rmax), float(max(zs))))
