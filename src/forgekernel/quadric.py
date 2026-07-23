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
