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
from functools import lru_cache
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


def ci_reciprocal(x: CInterval) -> CInterval:
    """1/x for an interval that strictly excludes zero (certified).
    Enclosure-preserving: 1/t is monotone on either side of zero, so the
    reciprocal of the endpoints brackets the reciprocal of every point."""
    x.sign()                                        # raises if straddles 0
    lo, hi = Fraction(1) / x.hi, Fraction(1) / x.lo
    return CInterval(min(lo, hi), max(lo, hi))


# -- certified trigonometry ---------------------------------------------------
#
# WHY THIS EXISTS, because the gap it fills is a roadmap decision and not a
# missing convenience. Every exact rung that measures an ARC is pinned to
# twelfths: the area of a circular segment is r^2(pi - t + sin t cos t) with
# t = arccos(h/r), and by Niven t lands in Q*pi only at multiples of 30
# degrees. So `flat.py` answers a flat milled on a bar at FIVE depths out of a
# continuum, and `notch.py` decides its crossings the same way. Measured on a
# radius-5 bar over 33 rational depths: 3 exact. The same sweep against a
# sphere cap, whose closed form is a polynomial and carries no angle, is 33 of
# 33. That is the whole difference between the two rungs, and it is the reason
# the composed grid's "42 cells need only lines and circles" is a much weaker
# statement than it sounds: a cell is scored green by ONE representative
# parameter, and the admissible set behind it can be measure zero.
#
# ADR-0019 already decided what to do about this. A topological decision needs
# a CERTIFIED SIGN, and certified includes "proven by a bracket" — an arc angle
# has no algebraic home but it has a perfectly good rational enclosure. What
# was missing was only the primitive.

#: Terms of cos's series needed for a remainder below ~1e-30 on |t| <= pi:
#: pi^40/40! is about 1e-28 and pi^44/44! about 1e-33, so 22 is ample and 30 was
#: simply generous. It matters because these are EXACT rationals — a bracket
#: endpoint has a ~25-digit denominator, so t^(2N) carries ~2N*25 digits and the
#: cost is superlinear in N. Trimming 30 to 22 is a third off the largest powers.
_COS_TERMS = 22


def cos_rational(t: Fraction, terms: int = _COS_TERMS) -> CInterval:
    """A certified enclosure of cos(t) for rational ``t``.

    Alternating series with the first omitted term as the remainder bound —
    rigorous, not asymptotic. That bound is valid only while the omitted TAIL
    is decreasing in magnitude, and the term ratio is t^2/((2k+1)(2k+2)), so
    the precondition is exactly ``t^2 < (2N+1)(2N+2)`` at the first omitted
    index N. It increases with k, so checking it at N covers the whole tail.

    The domain is therefore DERIVED from ``terms`` rather than fixed. An
    earlier version capped |t| at a hard-coded 4, which was far more
    conservative than the mathematics required and, worse, was not the real
    precondition: it made `sin` — which evaluates cos(pi/2 - t) — refuse over
    part of its own natural range for a reason that had nothing to do with
    whether the bound held. Outside the derived domain this refuses, because
    a remainder bound that silently stops bounding is worse than none.
    """
    t = t if isinstance(t, Fraction) else Fraction(t)
    t2 = t * t
    if t2 >= (2 * terms + 1) * (2 * terms + 2):
        raise ValueError(
            f"cos_rational: |t| = {float(abs(t)):.4g} is too large for "
            f"{terms} terms to bound the remainder; the alternating tail is "
            "not yet decreasing there")
    term = Fraction(1)                              # k = 0
    total = Fraction(0)
    for k in range(terms):
        total += term if k % 2 == 0 else -term
        term = term * t2 / ((2 * k + 1) * (2 * k + 2))
    # `term` is now the magnitude of the FIRST OMITTED term, which bounds the
    # remainder of an alternating series with decreasing terms
    return CInterval(total - term, total + term)


#: Requested half-width for a float-proposed arccos bracket.
#:
#: THE FLOAT'S OWN ERROR IS THE FLOOR HERE, and that is the honest trade-off in
#: this file. The bracket is centred on `math.acos`, which is accurate to ~1e-16,
#: so asking for 1e-25 does not produce 1e-25 — verification fails, the loop
#: widens, and it settles near 1e-16 anyway after several wasted rounds.
#: Requesting 1e-15 succeeds on the first try instead.
#:
#: Tighter is available and costs real time: bisecting down from here needs one
#: exact cos evaluation per halving, and those are rationals whose denominators
#: double each step, so t^44 grows past a thousand digits. Bisecting to 1e-40 is
#: what made a single flat-on-a-bar volume take 51 SECONDS. A caller who needs a
#: narrower bracket passes `width=` and pays for it.
#:
#: 1e-15 on an angle is ~1e-10 absolute on the volume of a Ø10 bar — about 1e-13
#: relative, comparable to a double's own error but PROVEN to enclose rather than
#: hoped. ADR-0019 asks for a certified sign and a reportable half-width; this
#: gives both with room to spare.
_ARCCOS_WIDTH = Fraction(1, 10 ** 15)


@lru_cache(maxsize=4096)
def arccos_rational(v: Fraction, width: Fraction = _ARCCOS_WIDTH
                    ) -> CInterval:
    """A certified enclosure of arccos(v) for -1 <= v <= 1.

    MEMOISED, and that is a real speedup rather than tidiness: measuring one
    flat-on-a-bar volume showed ~80 calls for ~12 distinct arguments, because a
    body's faces share their arc endpoints — the same twelfth point bounds a cap
    and the band that meets it. Pure function of (v, width), so the cache cannot
    change an answer.

    A FLOAT PROPOSES AND EXACT ARITHMETIC DISPOSES — the same discipline as
    `body.certified_bracket`, and here it is also what makes this usable at all.
    `math.acos` supplies a candidate, and the bracket is accepted only when
    exact arithmetic PROVES it encloses: cos is strictly decreasing on [0, pi],
    so it is enough that cos(lo) >= v >= cos(hi), each established from a
    certified `cos_rational`. The float decides nothing; if it lied, verification
    fails and the bracket widens.

    WHY NOT BISECTION, which is what this did first. Bisecting to 1e-40 takes
    ~133 exact-rational cos evaluations whose denominators compound every step,
    and each evaluation raises a huge fraction to the 60th power. One certified
    volume for a flat on a bar took **51 seconds**. This needs exactly TWO cos
    evaluations at a denominator the float fixes, and it is milliseconds.

    The default width is 1e-25 rather than 1e-40 — still tighter than a double
    by nine orders of magnitude, and ADR-0019 only ever needs enough to certify
    a sign. `bisect_arccos_rational` keeps the old route for anyone who needs a
    guaranteed answer without trusting a float to propose one.
    """
    import math

    v = v if isinstance(v, Fraction) else Fraction(v)
    if not (-1 <= v <= 1):
        raise ValueError(f"arccos outside [-1, 1]: {v}")
    if v == 1:
        return CInterval(Fraction(0), Fraction(0))
    if v == -1:
        return pi_interval()
    hi_pi = pi_interval().hi
    try:
        approx = Fraction(math.acos(float(v)))
    except (ValueError, OverflowError):                 # pragma: no cover
        return bisect_arccos_rational(v, width)
    w = width
    for _ in range(14):
        lo = max(Fraction(0), approx - w)
        hi = min(hi_pi, approx + w)
        # cos(lo) >= v >= cos(hi) proves the root is in [lo, hi], by
        # monotonicity on [0, pi]. Certified bounds on each side, so this is a
        # proof and not a comparison of estimates.
        if cos_rational(lo).lo >= v and cos_rational(hi).hi <= v:
            return CInterval(lo, hi)
        w *= 1000
    return bisect_arccos_rational(v, width)


def bisect_arccos_rational(v: Fraction,
                           width: Fraction = _ARCCOS_WIDTH
                           ) -> CInterval:
    """arccos by pure bisection — no float anywhere, at the cost of ~133 exact
    cos evaluations. Kept as the fallback `arccos_rational` drops to when a
    proposed bracket cannot be verified, so a float is never load-bearing.
    """
    v = v if isinstance(v, Fraction) else Fraction(v)
    if not (-1 <= v <= 1):
        raise ValueError(f"arccos outside [-1, 1]: {v}")
    # THE TWO ENDPOINTS ARE TANGENCIES and the bisection below cannot see
    # them. cos is flat where it meets +-1, so at v = -1 the test "cos(m) < v"
    # is never true, b never moves, and a climbs past pi into the region where
    # cos is INCREASING again — the invariant quietly dies and the answer came
    # back just above pi, excluding the very value it was asked for.
    if v == 1:
        return CInterval(Fraction(0), Fraction(0))
    if v == -1:
        return pi_interval()
    a = Fraction(0)                                 # cos(0) = 1 > v
    b = pi_interval().hi
    # ...and the same tangency makes the STARTING bracket a claim that has to
    # be checked rather than assumed: b must satisfy cos(b) < v for the walk
    # to be sound, and pi's upper bound only does so for v clear of -1. Refuse
    # rather than return a bracket whose enclosure was never established.
    if not cos_rational(b).hi < v:
        raise ValueError(
            f"arccos({v}) is inside the tangency at -1; not certifiable here")
    while b - a > width:
        m = (a + b) / 2
        c = cos_rational(m)
        if c.lo > v:
            a = m                                   # cos(m) > v: root is right
        elif c.hi < v:
            b = m                                   # cos(m) < v: root is left
        else:
            break                                   # not certified; stop here
    return CInterval(a, b)


def arccos(x, width: Fraction = _ARCCOS_WIDTH) -> CInterval:
    """arccos of a certified interval — the enclosure of every value in it.

    arccos is DECREASING, so the bracket flips: the low end comes from the
    high end of the argument. An argument bracket that pokes marginally
    outside [-1, 1] (a sqrt widening, say) is clamped, which is sound because
    the true value was inside it and inside [-1, 1] both.
    """
    x = _as_ci(x)
    lo = max(Fraction(-1), min(Fraction(1), x.lo))
    hi = max(Fraction(-1), min(Fraction(1), x.hi))
    return CInterval(arccos_rational(hi, width).lo,
                     arccos_rational(lo, width).hi)


def cos(x) -> CInterval:
    """cos of a certified interval, via the 1-Lipschitz bound.

    |cos u - cos v| <= |u - v| everywhere, so the midpoint's enclosure widened
    by the argument's half-width contains every value. Sound for any argument
    within the series domain, and it needs no case analysis over which
    extrema the interval happens to span.
    """
    x = _as_ci(x)
    c = cos_rational(x.mid)
    half = x.width / 2
    return CInterval(c.lo - half, c.hi + half)


def sin(x) -> CInterval:
    """sin of a certified interval — cos shifted by a certified pi/2.

    The shift is itself an interval, so the result carries pi's enclosure as
    well as the argument's. Stated this way rather than with a second series
    so there is one remainder-bound argument in this file, not two.
    """
    half_pi = pi_interval() * CInterval.exact(Fraction(1, 2))
    return cos(half_pi - _as_ci(x))
