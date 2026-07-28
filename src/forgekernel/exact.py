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


def F(x):
    """Exact conversion. Floats convert via Fraction(float) — the exact
    binary value, no decimal guessing and no denominator snapping — so
    every decision made afterward is exact relative to the given inputs.
    Slow denominators are ref's price for being the spec.

    WIDER EXACT FIELDS PASS THROUGH UNTOUCHED. A value already living in
    ℚ[√d] or ℚ[√d][π] is exact; forcing it to Fraction would either raise or,
    worse, lose the radical. Solids whose faces meet at an irrational height —
    a napkin ring's bore circles at z = ±√(R²−r²), a shelled cone's offset
    profile — cannot be expressed at all while this only admits ℚ.

    The guard that matters is unchanged: a FLOAT still converts to its exact
    binary Fraction rather than being waved through, so nothing silently
    becomes inexact. Widening the domain is not the same as dropping the
    check, and this function is the single place either could happen.
    """
    if isinstance(x, Fraction):
        return x
    if getattr(x, "is_exact_scalar", False):
        return x
    return Fraction(x)


def as_fraction(x):
    """``x`` as a ``Fraction`` if its VALUE is rational, else None.

    The companion to ``F``. ``F`` widens — it lets ℚ[√d] and ℚ(√p,√q) through
    untouched — and everything downstream that wants to do integer arithmetic
    (a perfect-square test, a numerator) then has to ask whether what it got
    is actually rational. Asking is the whole job, and it has to be asked in
    ONE place, because getting it wrong has gone both ways here:

    * assuming rational — ``nn.numerator`` on a rotated plane's |n|², which is
      a ``SurdVal`` even when its value is 2, crashed chamfer and shell with
      ``AttributeError`` on the most ordinary modelling step there is;
    * assuming irrational — reading ``.b == 0`` as "rational" on a ``BiSurd``,
      which carries ``.a`` and ``.b`` too, silently dropped ``c√q + e√pq`` and
      reported 960 for 960 − 20√3.

    A value from a wider field must PROVE it is rational: ask ``demote()`` for
    the smallest type that holds it, and refuse to guess at coefficients we do
    not understand.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    dem = getattr(x, "demote", None)
    if dem is not None:                          # BiSurd, and any wider field
        x = dem()                                # -> Fraction | SurdVal | self
        if isinstance(x, Fraction):
            return x
        if isinstance(x, int):
            return Fraction(x)
    b = getattr(x, "b", None)                    # SurdVal: a + b√d
    if b is not None and b == 0 and not getattr(x, "c", 0) \
            and not getattr(x, "e", 0):
        return Fraction(x.a)
    return None


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
        """Hashable canonical form: (normal, d) divided through by the normal's
        first nonzero component, so any scalar multiple ±λ·(n, d) maps to the
        same tuple — coplanarity becomes tuple equality. Works over ℚ and over
        ℚ[√d] (an exactly-rotated solid carries surd face normals)."""
        nums = (self.n[0], self.n[1], self.n[2], self.d)
        lead = None
        for v in nums[:3]:
            if v != 0:
                lead = v
                break
        return tuple(v / lead for v in nums)

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
