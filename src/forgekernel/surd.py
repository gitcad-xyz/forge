"""Exact quadratic-surd arithmetic — the field ℚ[√d] (K3.0 groundwork).

A SurdVal is a + b·√d with rational a, b and a fixed square-free d.
Values with different radicals cannot be combined (that needs a larger
field — refused honestly). This is the number field mitered sweeps live
in: a 45°-cornered path has length in ℚ[√2], so the swept volume is an
EXACT object with equality, not a float — the model OCCT fails to build.
"""

from __future__ import annotations

import math
from fractions import Fraction

from forgekernel.exact import F


def _squarefree(n: int) -> tuple[int, int]:
    """Factor n = m^2 * k with k square-free; return (m, k). n >= 0."""
    if n == 0:
        return (0, 1)
    m, k, i = 1, n, 2
    while i * i <= k:
        while k % (i * i) == 0:
            k //= i * i
            m *= i
        i += 1
    return m, k


def sqrt_rational(x) -> "SurdVal":
    """Exact √x for rational x >= 0, as m·√k with k square-free."""
    x = F(x)
    if x < 0:
        raise ValueError("sqrt of a negative rational")
    mn, kn = _squarefree(x.numerator)
    md, kd = _squarefree(x.denominator)
    # √(num/den) = (mn/md)·√(kn·kd)/kd  ... normalise via √(kn*kd)
    mk, kk = _squarefree(kn * kd)
    coeff = Fraction(mn, md) * Fraction(mk, kd)
    if kk == 1:
        return SurdVal(coeff, 0, 1)
    return SurdVal(0, coeff, kk)


class SurdVal:
    """a + b·√d, exact. d is square-free (d==1 means purely rational)."""

    __slots__ = ("a", "b", "d")

    def __init__(self, a=0, b=0, d=1) -> None:
        self.a, self.b = F(a), F(b)
        self.d = 1 if self.b == 0 else int(d)

    def _co(self, o: "SurdVal | int | Fraction") -> "SurdVal":
        return o if isinstance(o, SurdVal) else SurdVal(o, 0, 1)

    def _radical(self, o: "SurdVal") -> int:
        if self.b == 0:
            return o.d
        if o.b == 0:
            return self.d
        if self.d != o.d:
            raise ValueError(
                f"mixed radicals √{self.d} and √{o.d} — a bigger field "
                "arrives at K3.1")
        return self.d

    def __add__(self, o) -> "SurdVal":
        o = self._co(o)
        return SurdVal(self.a + o.a, self.b + o.b, self._radical(o))

    __radd__ = __add__

    def __sub__(self, o) -> "SurdVal":
        o = self._co(o)
        return SurdVal(self.a - o.a, self.b - o.b, self._radical(o))

    def __mul__(self, o) -> "SurdVal":
        o = self._co(o)
        if self.b == 0 or o.b == 0:
            d = self._radical(o)
            return SurdVal(self.a * o.a, self.a * o.b + self.b * o.a, d)
        if self.d != o.d:
            raise ValueError("mixed radicals in product — K3.1")
        # (a+b√d)(c+e√d) = (ac + be d) + (ae + bc)√d
        return SurdVal(self.a * o.a + self.b * o.b * self.d,
                       self.a * o.b + self.b * o.a, self.d)

    __rmul__ = __mul__

    def __eq__(self, o: object) -> bool:
        o = self._co(o)
        if self.b == 0 and o.b == 0:
            return self.a == o.a
        return self.a == o.a and self.b == o.b and self.d == o.d

    def __float__(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(self.d)

    def __repr__(self) -> str:
        if self.b == 0:
            return f"{self.a}"
        return f"({self.a} + {self.b}·√{self.d})"
