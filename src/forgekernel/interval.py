"""Certified intervals — the K3 number kind (ADR-0019).

A ``CInterval`` is a pair of exact rationals ``[lo, hi]`` that *provably*
brackets a real value. Arithmetic only ever widens the bracket; it never
loses the enclosure. Rational ``+ - *`` are exact (an interval widens
only because its inputs already had width), so a bracket grows *only* at
the genuinely irrational steps — ``pi`` and ``sqrt`` — and by a bounded,
reportable amount.

This is not a float with error bars. The bounds are the primitive and
they are rigorous: ``pi`` enters through a digit-verified rational
enclosure; ``sqrt(x)`` returns ``[a, b]`` with ``a*a <= x <= b*b``.

A topological decision may consult ``sign()`` only when the interval
strictly excludes zero (``certified``); otherwise the caller tightens or
refuses — never guesses.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

# pi to 60 decimal places (a widely tabulated, digit-verified constant).
# Stored as the integer floor(pi * 10**60); truncation is a rigorous lower
# bound and +1 ulp a rigorous upper bound, so [lo, hi] certainly brackets pi.
_PI_NUM = 3141592653589793238462643383279502884197169399375105820974944
_PI_SCALE = 10 ** 60


def pi_interval() -> "CInterval":
    """A certified rational enclosure of pi (width 1e-60)."""
    lo = Fraction(_PI_NUM, _PI_SCALE)
    return CInterval(lo, lo + Fraction(1, _PI_SCALE))


def _sqrt_low(x: Fraction, scale: int) -> Fraction:
    """Largest r = m/scale with r*r <= x (rational lower bound of sqrt(x))."""
    if x < 0:
        raise ValueError("sqrt of a negative certified interval")
    m = isqrt((x.numerator * scale * scale) // x.denominator)
    r = Fraction(m, scale)
    # isqrt floor guarantees r*r <= x; nudge defensively (never loops in practice)
    while r * r > x:
        m -= 1
        r = Fraction(m, scale)
    return r


def _sqrt_high(x: Fraction, scale: int) -> Fraction:
    """Smallest r = m/scale with r*r >= x (rational upper bound of sqrt(x))."""
    lo = _sqrt_low(x, scale)
    r = lo + Fraction(1, scale)
    while r * r < x:            # at most a couple of steps
        r += Fraction(1, scale)
    return r


class CInterval:
    """A certified real: lo <= true value <= hi, both exact rationals."""

    __slots__ = ("lo", "hi")

    # width of the rational sqrt bracket (1e-50): far below any float epsilon
    _SQRT_SCALE = 10 ** 50

    def __init__(self, lo, hi=None) -> None:
        lo = lo if isinstance(lo, Fraction) else Fraction(lo)
        if hi is None:
            hi = lo
        else:
            hi = hi if isinstance(hi, Fraction) else Fraction(hi)
        if lo > hi:
            raise ValueError(f"degenerate interval [{lo}, {hi}]")
        self.lo = lo
        self.hi = hi

    # -- construction ---------------------------------------------------------

    @staticmethod
    def exact(x) -> "CInterval":
        """A zero-width interval around an exact rational."""
        f = x if isinstance(x, Fraction) else Fraction(x)
        return CInterval(f, f)

    # -- arithmetic (enclosure-preserving) ------------------------------------

    def __add__(self, o: "CInterval") -> "CInterval":
        o = _as_ci(o)
        return CInterval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, o: "CInterval") -> "CInterval":
        o = _as_ci(o)
        return CInterval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o) -> "CInterval":
        return _as_ci(o).__sub__(self)

    def __mul__(self, o: "CInterval") -> "CInterval":
        o = _as_ci(o)
        prods = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return CInterval(min(prods), max(prods))

    __rmul__ = __mul__

    def sqrt(self) -> "CInterval":
        s = self._SQRT_SCALE
        return CInterval(_sqrt_low(self.lo, s), _sqrt_high(self.hi, s))

    # -- certified queries ----------------------------------------------------

    def sign(self) -> int:
        """+1/-1 if the interval strictly excludes zero; else raise — the
        sign is not certified and the caller must tighten or refuse."""
        if self.lo > 0:
            return 1
        if self.hi < 0:
            return -1
        raise ValueError("sign not certified: interval straddles zero")

    @property
    def mid(self) -> Fraction:
        return (self.lo + self.hi) / 2

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def to_float(self) -> float:
        """Reported value: the midpoint. The true value is within
        ``width/2`` of it, and ``width`` is available for the report."""
        return float(self.mid)

    def __repr__(self) -> str:
        return f"CInterval({float(self.lo):.6g}~{float(self.hi):.6g})"


def _as_ci(x) -> CInterval:
    return x if isinstance(x, CInterval) else CInterval.exact(x)
