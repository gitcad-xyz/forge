"""Exact biquadratic-surd arithmetic — the field ℚ(√p, √q) (K3.1).

A ``BiSurd`` is ``a + b·√p + c·√q + e·√(pq)`` with rational coefficients and a
fixed pair of radicands ``p < q``, both square-free, coprime. That is the
degree-4 field ℚ(√p, √q); ``{1, √p, √q, √(pq)}`` is a ℚ-basis, so equality and
zero-testing are coefficient comparisons — exact, no numerics anywhere. The
closure that names the field is ``√p · √q = √(pq)``: products of the two
generators land on the third basis radical, never outside.

The consumer that forced this widening: ``chamfered box × fillet(all)``. The
eroded polytope's chamfer planes have √2 normals (the √2), and its
chamfer–chamfer edges run along (1,±1,±1) with sweep angle π/3 (the √3, as
2√6·π after the edge-length × angle product). One square-free ``d`` cannot
hold both — see ``FilletedChamferedBox`` (quadric.py).

Only ORDER costs anything: the sign of a nonzero element is decided by
narrowing PROVEN rational enclosures of the radicals (integer square root, no
float consulted) until the interval sum excludes zero — the same discipline as
``PiPoly.sign``. Termination is guaranteed because a nonzero element is a
fixed nonzero real. Zero itself is decided structurally, by the basis.

Radicand pairs with ``gcd(p, q) > 1`` refuse, and the reason is NORMAL FORM,
not span: {1,√2,√6,√12} is a perfectly good basis of ℚ(√2,√3), but √12 = 2√3
so the same value reaches ``_co`` tagged 3, not 12, and coercion matches on the
tag. Requiring coprime radicands makes the tag canonical. ``SurdVal._promote``
therefore changes generators before it gets here — √2 with √6 enters as
``BiSurd(·,·,·,·, 2, 3)``, the same field under its canonical names. The pairs
that stay out are the ones with no coprime generators at all: ℚ(√6,√10) has
radicals 6 = 2·3, 10 = 2·5 and 15 = 3·5, which pairwise share a prime.
"""

from __future__ import annotations

import math
from fractions import Fraction

from forgekernel.surd import MixedRadicals, SurdVal, _squarefree


def _enc(d: int, digits: int) -> tuple[Fraction, Fraction]:
    """A proven rational enclosure of √d, by integer square root."""
    scale = 10 ** digits
    root = math.isqrt(d * scale * scale)          # floor(√d · 10^digits)
    return Fraction(root, scale), Fraction(root + 1, scale)


def _term_bounds(coeff: Fraction, lo: Fraction, hi: Fraction):
    return (coeff * lo, coeff * hi) if coeff >= 0 else (coeff * hi, coeff * lo)


class BiSurd:
    is_exact_scalar = True   # exact beyond ℚ; exact.F() must not coerce it away

    __slots__ = ("a", "b", "c", "e", "p", "q")

    def __init__(self, a=0, b=0, c=0, e=0, p=2, q=3) -> None:
        p, q = int(p), int(q)
        if not 1 < p < q:
            raise ValueError(
                f"ℚ(√{p},√{q}) is not in canonical form — radicands must "
                "satisfy 1 < p < q (equal or rational radicands collapse to "
                "a smaller field)")
        if _squarefree(p)[0] != 1 or _squarefree(q)[0] != 1:
            raise ValueError(
                f"radicand not square-free in ℚ(√{p},√{q}) — factor the "
                "square out first (sqrt_rational does)")
        if math.gcd(p, q) != 1:
            raise MixedRadicals(
                p, q,
                message=f"radicands √{p} and √{q} share a factor, so √{p * q} "
                        f"is not in square-free form and the same value can "
                        f"reach coercion under two different tags — hand the "
                        f"field its canonical coprime generators instead "
                        f"(SurdVal._promote does)")
        self.a, self.b = Fraction(a), Fraction(b)
        self.c, self.e = Fraction(c), Fraction(e)
        self.p, self.q = p, q

    # -- coercion --------------------------------------------------------------

    def _co(self, o):
        """Coerce into THIS field, refuse honestly, or defer (NotImplemented)."""
        if isinstance(o, BiSurd):
            if (o.p, o.q) == (self.p, self.q):
                return o
            o = o.demote()                # a foreign tag may name a value we hold
            if isinstance(o, BiSurd):
                raise MixedRadicals(
                    self.p, self.q, o.p, o.q,
                    message=f"mixed biquadratic fields ℚ(√{self.p},√{self.q}) "
                            f"and ℚ(√{o.p},√{o.q}) — a degree-8 tower arrives "
                            "at K3.2")
        if isinstance(o, SurdVal):
            if o.b == 0 or o.d == 1:
                return BiSurd(o.a, 0, 0, 0, self.p, self.q)
            if o.d == self.p:
                return BiSurd(o.a, o.b, 0, 0, self.p, self.q)
            if o.d == self.q:
                return BiSurd(o.a, 0, o.b, 0, self.p, self.q)
            if o.d == self.p * self.q:
                return BiSurd(o.a, 0, 0, o.b, self.p, self.q)
            raise MixedRadicals(
                self.p, self.q, o.d,
                message=f"√{o.d} is not in ℚ(√{self.p},√{self.q}) — a bigger "
                        "tower arrives at K3.2")
        if isinstance(o, (int, float, Fraction)):
            return BiSurd(Fraction(o), 0, 0, 0, self.p, self.q)
        return NotImplemented

    def _is_rational(self) -> bool:
        return self.b == 0 and self.c == 0 and self.e == 0

    def demote(self):
        """The value in the SMALLEST standard type that holds it — Fraction,
        SurdVal, or self. Widening a field must be invisible to everything
        that did not need it (the ``exact_sqrt`` rule)."""
        if self._is_rational():
            return self.a
        if self.c == 0 and self.e == 0:
            return SurdVal(self.a, self.b, self.p)
        if self.b == 0 and self.e == 0:
            return SurdVal(self.a, self.c, self.q)
        if self.b == 0 and self.c == 0:
            return SurdVal(self.a, self.e, self.p * self.q)
        return self

    # -- ring ------------------------------------------------------------------

    def __add__(self, o):
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        return BiSurd(self.a + o.a, self.b + o.b, self.c + o.c,
                      self.e + o.e, self.p, self.q)

    __radd__ = __add__

    def __neg__(self) -> "BiSurd":
        return BiSurd(-self.a, -self.b, -self.c, -self.e, self.p, self.q)

    def __sub__(self, o):
        o = self._co(o)
        return NotImplemented if o is NotImplemented else self + (-o)

    def __rsub__(self, o):
        o = self._co(o)
        return NotImplemented if o is NotImplemented else o + (-self)

    def __mul__(self, o):
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        p, q = self.p, self.q
        a1, b1, c1, e1 = self.a, self.b, self.c, self.e
        a2, b2, c2, e2 = o.a, o.b, o.c, o.e
        # (√p)² = p, (√q)² = q, √p·√q = √pq, √p·√pq = p√q, √q·√pq = q√p
        return BiSurd(
            a1 * a2 + b1 * b2 * p + c1 * c2 * q + e1 * e2 * p * q,
            a1 * b2 + b1 * a2 + (c1 * e2 + e1 * c2) * q,
            a1 * c2 + c1 * a2 + (b1 * e2 + e1 * b2) * p,
            a1 * e2 + e1 * a2 + b1 * c2 + c1 * b2,
            p, q)

    __rmul__ = __mul__

    def _inverse(self) -> "BiSurd":
        if self.sign() == 0:
            raise ZeroDivisionError("division by zero in ℚ(√p,√q)")
        # multiply by the conjugate over √p, landing in ℚ(√q); then by the
        # conjugate over √q, landing in ℚ — a rational to divide by
        c1 = BiSurd(self.a, -self.b, self.c, -self.e, self.p, self.q)
        y = self * c1                              # y.b == y.e == 0 by algebra
        c2 = BiSurd(y.a, 0, -y.c, 0, self.p, self.q)
        z = y * c2                                 # rational: a² − c²q
        num = c1 * c2
        return BiSurd(num.a / z.a, num.b / z.a, num.c / z.a, num.e / z.a,
                      self.p, self.q)

    def __truediv__(self, o):
        if isinstance(o, (int, Fraction)):
            r = Fraction(o)
            if r == 0:
                raise ZeroDivisionError("division by zero in ℚ(√p,√q)")
            return BiSurd(self.a / r, self.b / r, self.c / r, self.e / r,
                          self.p, self.q)
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        return self * o._inverse()

    def __rtruediv__(self, o):
        o = self._co(o)
        return NotImplemented if o is NotImplemented else o * self._inverse()

    def __abs__(self) -> "BiSurd":
        return -self if self.sign() < 0 else self

    # -- exact sign ------------------------------------------------------------

    def sign(self) -> int:
        """-1, 0 or +1 — exact.

        Zero is decided WITHOUT numerics: {1, √p, √q, √pq} is a ℚ-basis of a
        degree-4 field, so the element is zero iff every coefficient is. For a
        nonzero element the radical enclosures are narrowed until the interval
        sum excludes zero, which must happen because the value is a fixed
        nonzero real.
        """
        if self._is_rational():
            return (self.a > 0) - (self.a < 0)
        digits = 20
        while digits <= 1_000_000:
            lo = hi = self.a
            for coeff, d in ((self.b, self.p), (self.c, self.q),
                             (self.e, self.p * self.q)):
                if coeff == 0:
                    continue
                t0, t1 = _term_bounds(coeff, *_enc(d, digits))
                lo += t0
                hi += t1
            if lo > 0:
                return 1
            if hi < 0:
                return -1
            digits *= 2
        raise ArithmeticError(
            "could not decide the sign of a ℚ(√p,√q) value within 10^6 "
            "digits — it is indistinguishable from zero at that precision")

    def bounds(self, digits: int) -> tuple[Fraction, Fraction]:
        """A proven rational enclosure at the given precision — the hook
        ``PiPoly``'s interval evaluation uses for coefficients beyond ℚ."""
        lo = hi = self.a
        for coeff, d in ((self.b, self.p), (self.c, self.q),
                         (self.e, self.p * self.q)):
            if coeff == 0:
                continue
            t0, t1 = _term_bounds(coeff, *_enc(d, digits))
            lo += t0
            hi += t1
        return lo, hi

    # -- order and equality ----------------------------------------------------

    def _cmp(self, o):
        o = self._co(o)
        return None if o is NotImplemented else (self - o).sign()

    def __lt__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c < 0

    def __le__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c <= 0

    def __gt__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c > 0

    def __ge__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c >= 0

    def __eq__(self, o) -> bool:
        try:
            o = self._co(o)
        except ValueError:
            return False                     # a value in a foreign field
        if o is NotImplemented:
            return NotImplemented
        return (self.a == o.a and self.b == o.b
                and self.c == o.c and self.e == o.e)

    def __hash__(self) -> int:
        # equal values must hash equally across the tower: a pure rational
        # hashes as its Fraction, a single-radical value as the SurdVal it
        # demotes to (whose own rule is hash((a, b, d)))
        d = self.demote()
        if isinstance(d, Fraction):
            return hash(d)
        if isinstance(d, SurdVal):
            return hash(d)
        return hash((self.a, self.b, self.c, self.e, self.p, self.q))

    # -- the float boundary ----------------------------------------------------

    def __float__(self) -> float:
        return (float(self.a) + float(self.b) * math.sqrt(self.p)
                + float(self.c) * math.sqrt(self.q)
                + float(self.e) * math.sqrt(self.p * self.q))

    def __repr__(self) -> str:
        parts = [f"{self.a}"]
        for coeff, d in ((self.b, self.p), (self.c, self.q),
                         (self.e, self.p * self.q)):
            if coeff:
                parts.append(f"{coeff}·√{d}")
        return "(" + " + ".join(parts) + ")" if len(parts) > 1 else f"{self.a}"
