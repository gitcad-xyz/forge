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


class MixedRadicals(ValueError):
    """Radicands with no common home in the current number field.

    ℚ[√2] and ℚ[√3] used to be such a pair; K3.1 built ℚ(√p,√q) and
    ``SurdVal._promote`` now lifts any pair that a coprime square-free pair of
    generators can name, so this fires only where the tower really stops:
    √6 with √10 (6 = 2·3, 10 = 2·5, 15 = 3·5 — no two of the field's radicals
    are coprime), or a value from outside a ``BiSurd``'s field.

    This is a boundary of the number field, not a bug and not an unwritten
    algorithm, so a caller needs to tell it apart from both. It was a bare
    ``ValueError`` and a bbox comparison inside ``boolean`` let it out of the
    Kernel seam raw — by the capability bench's own taxonomy a raw exception
    escaping the seam is a CRASH, i.e. a defect, so the one wall that is
    genuinely about exactness was also the one the seam failed to report
    honestly. Named so callers can catch it (every sibling refusal in forge
    already is: ``BooleanUnsupported``, ``TrimLoopUnstitchable``,
    ``ShellAuditUncertified``, …).

    ``.radicals`` is every radicand involved, in the order they were met — two
    for a ``SurdVal`` pair, three when a value meets a ``BiSurd`` field it is
    outside of, four for two different biquadratic fields.

    Still a ``ValueError``, so existing ``except ValueError`` handlers around
    surd arithmetic keep working unchanged.
    """

    def __init__(self, *radicands: int, message: str | None = None):
        self.radicals = tuple(int(d) for d in radicands)
        super().__init__(message or (
            "mixed radicals "
            + " and ".join(f"√{d}" for d in self.radicals)
            + " — no coprime square-free pair of generators names one field "
              "holding them all"))


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
    """Exact √x for rational x >= 0, as m·√k with k square-free.

    ``F`` WIDENS — it lets ℚ[√d] and ℚ(√p,√q) through untouched — so `F(x)`
    here does not guarantee a Fraction, and the `.numerator` two lines down
    then raised AttributeError straight through the seam. That happened four
    separate times in one day, from four different callers, because a value
    whose TYPE is SurdVal and whose VALUE is rational is the normal residue of
    any exact rotation. Ask `as_fraction`, and when the argument really is
    irrational say so by name instead of by traceback: √(a+b√d) is a NESTED
    radical, a field above this one.
    """
    from forgekernel.exact import as_fraction

    q = as_fraction(x)
    if q is None:
        raise MixedRadicals(
            message=f"sqrt of {x!r} needs a nested radical — √(a+b√d) is not "
                    "in ℚ[√d], and the tower that holds it arrives at K3.2")
    x = q
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


def exact_sqrt(x):
    """Exact √x in the SMALLEST field that holds it: a ``Fraction`` when x is
    a perfect rational square, a ``SurdVal`` otherwise.

    ``sqrt_rational`` always hands back a ``SurdVal``, which is right when the
    caller is already in ℚ[√d] and wrong when it is not: a value typed
    ``SurdVal(4, 0, 1)`` is equal to 4 but is not a ``Fraction``, and code that
    shares one offset routine between rectilinear and slanted profiles would
    then retype every existing rational answer. Widening a field must be
    invisible to everything that did not need it — the same rule ``_pi_value``
    follows when it returns ``PiVal`` rather than ``PiPoly``.
    """
    v = sqrt_rational(x)
    return v.a if v.b == 0 else v


class SurdVal:
    is_exact_scalar = True   # exact beyond Q; F() must not coerce it away
    """a + b·√d, exact. d is square-free (d==1 means purely rational)."""

    __slots__ = ("a", "b", "d")

    def __init__(self, a=0, b=0, d=1) -> None:
        self.a, self.b = F(a), F(b)
        self.d = 1 if self.b == 0 else int(d)

    def _co(self, o: "SurdVal | int | Fraction"):
        """Coerce a plain number, or DEFER. Wrapping ``SurdVal(o, 0, 1)``
        around ANY object silently nested wider exact types (a ``BiSurd``, a
        ``PiVal``) inside the rational slot — ``F()`` passes exact scalars
        through, so ``self.a`` became a non-Fraction and every predicate
        downstream lied. Returning ``NotImplemented`` instead lets Python
        reflect to the wider type's own arithmetic, which knows how to hold a
        ``SurdVal`` — or raises an honest ``TypeError`` when nothing does."""
        if isinstance(o, SurdVal):
            return o
        if isinstance(o, (int, float, Fraction)):
            return SurdVal(o, 0, 1)
        return NotImplemented

    def _radical(self, o: "SurdVal") -> int:
        if self.b == 0:
            return o.d
        if o.b == 0:
            return self.d
        if self.d != o.d:
            raise MixedRadicals(self.d, o.d)
        return self.d

    def _promote(self, o):
        """Both operands lifted into ℚ(√p,√q), or None when that field does
        not exist for this pair (#127).

        ``BiSurd`` was already general in (p, q) and complete — arithmetic,
        comparison, sign, inverse — but nothing ever CONNECTED the two, so
        every mixed-radical expression refused although its exact home was
        already in the kernel. A frustum's chamfer is the case that exposed it:
        the face normal lives in ℚ[√37] and the edge direction in ℚ[√38].

        ``BiSurd`` names its field by a COPRIME square-free pair, because its
        coercion matches on the radicand tag and √(pq) must carry the tag pq.
        Two radicands that share a factor can still be inside one biquadratic
        field, just under different generators: √2 and √6 both live in
        ℚ(√2,√3), where √6 is the third basis radical √(pq). So when one
        radicand divides the other, change generators to (p, q/p) rather than
        refuse — otherwise a body with √2 face normals and √6 edge directions
        is turned away from a field it is already inside.

        Returns None (leaving the caller to refuse) when no coprime pair of
        generators exists. √6 with √10 is the honest case: the field is
        ℚ(√6,√10) = ℚ(√6,√15), degree 4 and perfectly ordinary, but its
        radicals are 6 = 2·3, 10 = 2·5 and 15 = 3·5, which pairwise share a
        prime — no two of them are coprime, so this normal form cannot name it.
        """
        from math import gcd

        if not isinstance(o, SurdVal) or self.b == 0 or o.b == 0:
            return None
        if self.d == o.d:
            return None                       # already one field
        p, q = (self.d, o.d) if self.d < o.d else (o.d, self.d)
        if p < 2:
            return None
        if _squarefree(p)[0] != 1 or _squarefree(q)[0] != 1:
            return None
        if gcd(p, q) != 1:
            if q % p:
                return None                   # e.g. √6 with √10
            p, q = sorted((p, q // p))         # √p, √(pq) -> generators p, q/p
            if p < 2 or gcd(p, q) != 1 or _squarefree(q)[0] != 1:
                return None
        from forgekernel.bisurd import BiSurd

        def lift(v):
            if v.d == p:
                return BiSurd(v.a, v.b, 0, 0, p, q)
            if v.d == q:
                return BiSurd(v.a, 0, v.b, 0, p, q)
            return BiSurd(v.a, 0, 0, v.b, p, q)      # v.d == p·q

        return lift(self), lift(o)

    def __add__(self, o) -> "SurdVal":
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        pr = self._promote(o)
        if pr is not None:
            return pr[0] + pr[1]
        return SurdVal(self.a + o.a, self.b + o.b, self._radical(o))

    __radd__ = __add__

    def __sub__(self, o) -> "SurdVal":
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        pr = self._promote(o)
        if pr is not None:
            return pr[0] - pr[1]
        return SurdVal(self.a - o.a, self.b - o.b, self._radical(o))

    def __rsub__(self, o) -> "SurdVal":     # o − self, for rational/int on the left
        o = self._co(o)
        return NotImplemented if o is NotImplemented else o - self

    def __mul__(self, o) -> "SurdVal":
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        if self.b == 0 or o.b == 0:
            d = self._radical(o)
            return SurdVal(self.a * o.a, self.a * o.b + self.b * o.a, d)
        pr = self._promote(o)
        if pr is not None:
            return pr[0] * pr[1]
        if self.d != o.d:
            raise MixedRadicals(self.d, o.d)
        # (a+b√d)(c+e√d) = (ac + be d) + (ae + bc)√d
        return SurdVal(self.a * o.a + self.b * o.b * self.d,
                       self.a * o.b + self.b * o.a, self.d)

    __rmul__ = __mul__

    def __neg__(self) -> "SurdVal":
        return SurdVal(-self.a, -self.b, self.d)

    def __abs__(self) -> "SurdVal":
        """``abs()`` over ℚ[√d] — exact, via the sign predicate.

        Without it every ``abs()`` in the kernel is a landmine that only goes
        off once a solid has been ROTATED: an exact 30/45/90° turn puts surds
        in the coordinates, and three separate call sites then died with a
        bare ``TypeError`` rather than any honest refusal. An ordered field
        that can compare but cannot take a magnitude is a half-built type.
        """
        return -self if self._sign() < 0 else self

    def __truediv__(self, o) -> "SurdVal":
        if isinstance(o, (int, Fraction)):
            r = F(o)
            return SurdVal(self.a / r, self.b / r, self.d)
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        if o.b == 0:                             # divide by a rational
            return SurdVal(self.a / o.a, self.b / o.a, self.d)
        # DIVISION PROMOTES TOO. It was the one arithmetic route #127 left
        # behind: the conjugate trick below builds `self * conj`, which
        # promotes to a BiSurd the moment the two live in different quadratic
        # fields, and then reading `num.d` off it raised AttributeError. Found
        # by rotating a sphere about an oblique axis, where the centre lands in
        # ℚ(√2,√3) — BiSurd divides perfectly well, it was simply never asked.
        pr = self._promote(o)
        if pr is not None:
            return pr[0] / pr[1]
        # divide by (c+e√d) via the conjugate: ·(c−e√d)/(c²−e²d)
        denom = o.a * o.a - o.b * o.b * o.d      # rational, nonzero
        num = self * SurdVal(o.a, -o.b, o.d)
        return SurdVal(num.a / denom, num.b / denom, num.d)

    def __rtruediv__(self, o) -> "SurdVal":
        o = self._co(o)
        return NotImplemented if o is NotImplemented else o / self

    def __pow__(self, n) -> "SurdVal":
        """``x ** n`` for integer ``n`` — exact, by square-and-multiply.

        ℚ[√d] is closed under multiplication, so an integer power never leaves
        the field and there was never a reason for this to be missing. The
        same lesson ``__abs__`` records: a field that can multiply but cannot
        raise to a power is a half-built type, and the gap shows up only after
        a solid has been ROTATED, when surds first appear in coordinates.

        gitcad #49: a body placed through a non-``z`` sketch plane carries an
        exact 120° axis permutation, and drilling it reached
        ``_seg_dist2``'s ``(px - qx) ** 2`` — which raised a bare TypeError
        straight through the seam's refusal wrapper. A distance predicate is
        exactly where an exact field must not have holes.

        A non-integer exponent leaves the field, so it DEFERS
        (``NotImplemented``) rather than rounding through float.
        """
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            # __truediv__ raises ZeroDivisionError for a zero denominator,
            # which is the honest answer for 0 ** -1.
            return SurdVal(1) / (self ** -n)
        out, base, e = SurdVal(1), self, n
        while e:
            if e & 1:
                out = out * base
            e >>= 1
            if e:
                base = base * base
        return out

    def _sign(self) -> int:
        """Exact sign of a + b√d (d ≥ 1, √d > 0) — decides comparisons."""
        if self.b == 0:
            return (self.a > 0) - (self.a < 0)
        if self.a == 0:
            return (self.b > 0) - (self.b < 0)
        sa = (self.a > 0) - (self.a < 0)
        sb = (self.b > 0) - (self.b < 0)
        if sa == sb:                             # both terms same sign
            return sa
        # opposite signs: compare magnitudes a² vs b²·d (squaring is monotone)
        da, db = self.a * self.a, self.b * self.b * self.d
        if da == db:
            return 0
        return sa if da > db else sb

    def _diff_sign(self, o) -> int:
        """Sign of (self − o), whichever field the difference lands in.

        ``__sub__`` now PROMOTES a mixed-radical pair to ``BiSurd`` (#127),
        and BiSurd spells its sign ``sign()`` rather than ``_sign()`` — so the
        comparisons have to ask the object, not assume the type.
        """
        diff = self - o
        return diff._sign() if isinstance(diff, SurdVal) else diff.sign()

    def __lt__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else self._diff_sign(o) < 0

    def __le__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else self._diff_sign(o) <= 0

    def __gt__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else self._diff_sign(o) > 0

    def __ge__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else self._diff_sign(o) >= 0

    def __eq__(self, o: object) -> bool:
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        if self.b == 0 and o.b == 0:
            return self.a == o.a
        return self.a == o.a and self.b == o.b and self.d == o.d

    def __hash__(self) -> int:
        if self.b == 0:                          # a pure rational hashes as itself
            return hash(self.a)
        return hash((self.a, self.b, self.d))

    def __float__(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(self.d)

    def __repr__(self) -> str:
        if self.b == 0:
            return f"{self.a}"
        return f"({self.a} + {self.b}·√{self.d})"
