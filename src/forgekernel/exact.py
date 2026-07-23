"""Exact rational linear algebra — the numerical substrate of K1.

Every coordinate is a ``fractions.Fraction``; every predicate (side of
plane, collinearity, orientation) is an exact sign computation. There
are NO epsilons anywhere in this package: two points are equal iff
their coordinates are equal, a point is on a plane iff the incidence
expression is exactly zero. Approximation exists only at the export
boundary (floats for tessellation/metrics), never inside a decision.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Iterable

Vec = tuple[Fraction, Fraction, Fraction]


def F(x) -> Fraction:
    """Exact conversion. Floats convert via Fraction(float) — the exact
    binary value, no decimal guessing and no denominator snapping — so
    every decision made afterward is exact relative to the given inputs.
    Slow denominators are ref's price for being the spec."""
    return x if isinstance(x, Fraction) else Fraction(x)


def vec(x, y, z) -> Vec:
    return (F(x), F(y), F(z))


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def neg(a: Vec) -> Vec:
    return (-a[0], -a[1], -a[2])


def smul(s: Fraction, a: Vec) -> Vec:
    return (s * a[0], s * a[1], s * a[2])


def dot(a: Vec, b: Vec) -> Fraction:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def is_zero(a: Vec) -> bool:
    return a[0] == 0 and a[1] == 0 and a[2] == 0


class Plane:
    """Oriented plane ``normal · x == d``. The normal is rational and
    unnormalized; orientation carries meaning (outward = positive side)."""

    __slots__ = ("n", "d")

    def __init__(self, n: Vec, d: Fraction) -> None:
        if is_zero(n):
            raise ValueError("degenerate plane (zero normal)")
        self.n = n
        self.d = d

    @classmethod
    def from_points(cls, a: Vec, b: Vec, c: Vec) -> "Plane":
        n = cross(sub(b, a), sub(c, a))
        if is_zero(n):
            raise ValueError("collinear points do not define a plane")
        return cls(n, dot(n, a))

    def side(self, p: Vec) -> int:
        """Exact classification: +1 front, -1 back, 0 on."""
        s = dot(self.n, p) - self.d
        return (s > 0) - (s < 0)

    def flipped(self) -> "Plane":
        return Plane(neg(self.n), -self.d)

    def canonical(self) -> tuple:
        """Hashable canonical form: normal scaled to coprime integers with a
        sign convention — coplanarity comparisons become tuple equality."""
        nums = [self.n[0], self.n[1], self.n[2], self.d]
        den = 1
        for v in nums:
            den = den * v.denominator // gcd(den, v.denominator)
        ints = [int(v * den) for v in nums]
        g = 0
        for v in ints:
            g = gcd(g, abs(v))
        if g:
            ints = [v // g for v in ints]
        for v in ints[:3]:
            if v != 0:
                if v < 0:
                    ints = [-w for w in ints]
                break
        return tuple(ints)

    def coplanar_key(self) -> tuple:
        """Canonical form ignoring orientation (for adjacency grouping)."""
        c = self.canonical()
        return min(c, tuple(-v for v in c))


def centroid(points: Iterable[Vec]) -> Vec:
    pts = list(points)
    k = Fraction(1, len(pts))
    acc = (Fraction(0), Fraction(0), Fraction(0))
    for p in pts:
        acc = add(acc, p)
    return smul(k, acc)
