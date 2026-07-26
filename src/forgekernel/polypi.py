"""ℚ[π] as a polynomial ring — the number field a fillet needs.

``PiVal`` is ``a + bπ``, which covers every volume the kernel has computed so
far: a cylinder is πr²h, a sphere (4/3)πr³, a cone (π/3)h(r₁²+r₁r₂+r₂²). A
FILLET breaks it. Revolving a circular arc sweeps a torus, and Pappus gives

    V = 2πR · πa² = 2π²Ra

so a filleted rim needs π². More generally, Green's theorem over a profile
containing arcs produces ``∫cos²θ dθ = θ/2 + sin2θ/4``, and with θ a multiple
of π/2 that θ/2 is another power of π. There is no bound on the power in
principle, so the honest structure is the whole polynomial ring rather than one
more hand-added term.

π is TRANSCENDENTAL, so ℚ[π] is a free polynomial ring: two elements are equal
exactly when their coefficients are, and an element is zero exactly when every
coefficient is. That makes equality and zero-testing exact with no numerics at
all — the property that matters, since the charter forbids a float deciding
anything topological. Only ORDER needs evaluation, and that is done by
narrowing a rational enclosure of π until the sign is decided, which always
terminates for a nonzero element.
"""

from __future__ import annotations

from fractions import Fraction as F

# π to 100 significant digits, as an exact rational enclosure. Comparison
# narrows from here; the digits themselves are never used as a float.
_PI_DIGITS = (
    "3141592653589793238462643383279502884197169399375105820974944592307816"
    "40628620899862803482534211706798214808651328230664709384460955058223172"
    "535940812848111745028410270193852110555964462294895493038"
)


def _pi_bounds(digits: int) -> tuple[F, F]:
    """A rational interval containing π, using `digits` decimal places."""
    digits = min(digits, len(_PI_DIGITS) - 1)
    scale = 10 ** digits
    lo = F(int(_PI_DIGITS[: digits + 1]), scale)
    return lo, lo + F(1, scale)


def _sqrt_bounds(d: int, digits: int) -> tuple[F, F]:
    """A rational interval containing √d, to `digits` decimal places.

    By integer square root, so the enclosure is PROVEN rather than believed —
    no float is consulted anywhere in it.
    """
    import math

    scale = 10 ** digits
    root = math.isqrt(int(d) * scale * scale)    # floor(√d · 10^digits)
    return F(root, scale), F(root + 1, scale)


def _coef_bounds(x, digits: int) -> tuple[F, F]:
    """A rational enclosure of a coefficient, whatever field it lives in.

    A ``Fraction`` is its own enclosure. A ``SurdVal`` ``a + b√d`` is bracketed
    from a proven enclosure of √d, ends swapped when b is negative. This is the
    ONLY place ℚ[√d] costs anything here: equality and arithmetic on the ring
    are already exact over it, because π is transcendental over any algebraic
    extension of ℚ just as it is over ℚ itself.
    """
    if isinstance(x, F):
        return x, x
    if hasattr(x, "bounds"):                     # BiSurd (and any wider field)
        return x.bounds(digits)                  # provides its own proven hook
    if getattr(x, "b", 0) == 0:
        return F(x.a), F(x.a)
    lo, hi = _sqrt_bounds(x.d, digits)
    return ((x.a + x.b * lo, x.a + x.b * hi) if x.b > 0
            else (x.a + x.b * hi, x.a + x.b * lo))


def _coef(x):
    """Accept a coefficient into the ring: ℚ as a Fraction, ℚ[√d] as itself."""
    if isinstance(x, float):
        if x != int(x):
            raise ValueError(
                f"a float ({x!r}) cannot enter ℚ[π] — pass a Fraction, or "
                "the exactness the ring exists for is already gone")
        return F(int(x))
    if isinstance(x, (int, F)):
        return F(x)
    if hasattr(x, "demote"):                     # BiSurd: a + b√p + c√q + e√pq
        x = x.demote()                           # smallest field that holds it
        if isinstance(x, F):
            return x
        if not hasattr(x, "d"):
            return x                             # a genuinely biquadratic value
    if hasattr(x, "a") and hasattr(x, "d"):      # SurdVal: a + b√d
        return F(x.a) if x.b == 0 else x         # collapse a pure rational
    raise ValueError(f"{type(x).__name__} is not an exact coefficient for ℚ[π]")


class PiPoly:
    is_exact_scalar = True   # exact beyond Q; F() must not coerce it away
    """Exact ``Σ cₖ πᵏ`` over ℚ — or over ℚ[√d], which costs nothing.

    π is transcendental over any ALGEBRAIC extension of ℚ, not merely over ℚ,
    so the ring stays free when the coefficients are surds: equality is still
    coefficient equality and zero is still all-coefficients-zero, with no
    numerics. Only ORDER has to look at values, and it brackets each coefficient
    from a proven enclosure of its radical.

    That widening is what a napkin ring needs — a sphere bored coaxially has
    volume (140/3)π√35 — and what a cone's chamfer needs, since its slant is
    irrational.
    """

    __slots__ = ("c",)

    def __init__(self, coeffs) -> None:
        if (isinstance(coeffs, (int, F)) or hasattr(coeffs, "d")
                or getattr(coeffs, "is_exact_scalar", False)):
            coeffs = [coeffs]
        c = [_coef(x) for x in coeffs]
        while len(c) > 1 and c[-1] == 0:
            c.pop()
        self.c = tuple(c)

    # -- construction ---------------------------------------------------------

    @classmethod
    def rational(cls, q) -> "PiPoly":
        return cls([F(q)])

    @classmethod
    def term(cls, coeff, power: int) -> "PiPoly":
        if power < 0:
            raise ValueError("ℚ[π] has no negative powers of π")
        return cls([F(0)] * power + [F(coeff)])

    @classmethod
    def from_pival(cls, v) -> "PiPoly":
        """Lift the legacy ``a + bπ`` representation.

        The coefficients go through ``_coef`` (via ``__init__``), NOT through
        the stdlib ``Fraction``. That distinction is the whole bug: ``PiVal``'s
        own constructor uses the widened ``exact.F``, so since #122 a ``PiVal``
        can legitimately carry a ℚ[√d] coefficient — a napkin ring's volume is
        ``(140/3)√35·π``. Forcing it back through ``Fraction`` raised
        "argument should be a string or a Rational instance", and because
        ``PiVal._cmp`` swallows ``TypeError`` that surfaced not as an error but
        as ``NotImplemented``: ``volume(body) <= 0`` in the answer audit became
        an unorderable-types ``TypeError`` and the audit could not check the
        one thing it exists to check.

        ``_coef`` admits ℚ and ℚ[√d] and still REFUSES a non-integral float, so
        this widens the field without opening the door ADR-0019 closes.
        """
        return cls([v.a, v.b])

    # -- ring -----------------------------------------------------------------

    def _co(self, o):
        if isinstance(o, PiPoly):
            return o
        if isinstance(o, (int, F)):
            return PiPoly([F(o)])
        # PiVal (a + b*pi) names a SUBSET of these numbers, and volume() now
        # returns whichever fits — so the two meet constantly. Without this,
        # comparing a filleted part's volume with an unfilleted one raised
        # TypeError on two values that are perfectly comparable.
        if hasattr(o, "a") and hasattr(o, "b") and not hasattr(o, "c"):
            return PiPoly.from_pival(o)
        return NotImplemented

    def __add__(self, o):
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        n = max(len(self.c), len(o.c))
        return PiPoly([self[i] + o[i] for i in range(n)])

    __radd__ = __add__

    def __neg__(self) -> "PiPoly":
        return PiPoly([-x for x in self.c])

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
        out = [F(0)] * (len(self.c) + len(o.c) - 1)
        for i, a in enumerate(self.c):
            if a:
                for j, b in enumerate(o.c):
                    out[i + j] += a * b
        return PiPoly(out)

    __rmul__ = __mul__

    def __truediv__(self, o):
        """Division by a RATIONAL only: ℚ[π] is a ring, not a field, and
        1/π is not in it. Dividing by a polynomial would leave the field
        silently, which is exactly what the charter forbids."""
        if isinstance(o, PiPoly):
            if len(o.c) != 1:
                raise ValueError(
                    "ℚ[π] is a ring: dividing by a polynomial in π leaves it "
                    "(1/π is not in ℚ[π])")
            o = o.c[0]
        if not isinstance(o, (int, F)):
            return NotImplemented
        if o == 0:
            raise ZeroDivisionError("division by zero in ℚ[π]")
        return PiPoly([x / F(o) for x in self.c])

    def __pow__(self, n: int) -> "PiPoly":
        if not isinstance(n, int) or n < 0:
            raise ValueError("ℚ[π] supports non-negative integer powers only")
        out = PiPoly([F(1)])
        for _ in range(n):
            out = out * self
        return out

    def __getitem__(self, i: int) -> F:
        return self.c[i] if 0 <= i < len(self.c) else F(0)

    @property
    def degree(self) -> int:
        return len(self.c) - 1

    def is_rational(self) -> bool:
        return len(self.c) == 1

    # -- equality is EXACT: π is transcendental, so no reduction exists -------

    def __eq__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else self.c == o.c

    def __hash__(self) -> int:
        # equal values must hash equally: PiPoly.rational(3) == 3 is True, so
        # a set or dict keyed on one must find the other
        return hash(self.c[0]) if self.is_rational() else hash(self.c)

    # -- order needs evaluation, and narrows until the sign is decided --------

    def sign(self) -> int:
        """-1, 0 or +1 — exact.

        Zero is decided WITHOUT any numerics (all coefficients zero, by
        transcendence). For a nonzero element the enclosure of π is narrowed
        until the value's interval excludes zero, which must happen because the
        value is a fixed nonzero real.
        """
        if all(x == 0 for x in self.c):
            return 0
        digits = 20
        while digits < len(_PI_DIGITS):
            lo, hi = _pi_bounds(digits)
            # RIGOROUS interval evaluation, term by term. Evaluating at the two
            # endpoints and taking min/max would only bound a MONOTONE
            # polynomial, and nothing here promises monotonicity — an interior
            # extremum would make the "enclosure" exclude the true value and
            # the sign could come back confidently wrong. Since lo > 0 every
            # power is increasing, so πᵏ ∈ [loᵏ, hiᵏ] and each term's bounds
            # follow from its coefficient's sign.
            v0 = v1 = F(0)
            plo = phi = F(1)
            for coeff in self.c:
                # each coefficient contributes an INTERVAL: exact for a
                # rational, a proven bracket for a surd. Combining interval
                # coefficients with interval powers needs all four corners,
                # because either can be negative.
                clo, chi = _coef_bounds(coeff, digits)
                ends = (clo * plo, clo * phi, chi * plo, chi * phi)
                v0 += min(ends)
                v1 += max(ends)
                plo *= lo
                phi *= hi
            if v0 > 0:
                return 1
            if v1 < 0:
                return -1
            digits *= 2
        raise ArithmeticError(
            "could not decide the sign of a ℚ[π] value within the stored "
            "digits of π "
            "— it is indistinguishable from zero at that precision")

    def __lt__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else (self - o).sign() < 0

    def __le__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else (self - o).sign() <= 0

    def __gt__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else (self - o).sign() > 0

    def __ge__(self, o) -> bool:
        o = self._co(o)
        return NotImplemented if o is NotImplemented else (self - o).sign() >= 0

    # -- the float boundary ---------------------------------------------------

    def __float__(self) -> float:
        import math

        acc, p = 0.0, 1.0
        for coeff in self.c:
            acc += float(coeff) * p
            p *= math.pi
        return acc

    def __repr__(self) -> str:
        if self.is_rational():
            return f"PiPoly({self.c[0]})"
        parts = [f"{x}" if k == 0 else
                 (f"{x}*pi" if k == 1 else f"{x}*pi^{k}")
                 for k, x in enumerate(self.c) if x]
        return "PiPoly(" + " + ".join(parts or ["0"]) + ")"
