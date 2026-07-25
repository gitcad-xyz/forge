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
    "5359408128481117450284102701938521105559644622948954930382"
)


def _pi_bounds(digits: int) -> tuple[F, F]:
    """A rational interval containing π, using `digits` decimal places."""
    digits = min(digits, len(_PI_DIGITS) - 1)
    scale = 10 ** digits
    lo = F(int(_PI_DIGITS[: digits + 1]), scale)
    return lo, lo + F(1, scale)


class PiPoly:
    """Exact ``Σ cₖ πᵏ`` with rational coefficients."""

    __slots__ = ("c",)

    def __init__(self, coeffs) -> None:
        if isinstance(coeffs, (int, F)):
            coeffs = [F(coeffs)]
        c = [F(x) for x in coeffs]
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
        """Lift the legacy ``a + bπ`` representation."""
        return cls([F(v.a), F(v.b)])

    # -- ring -----------------------------------------------------------------

    def _co(self, o):
        if isinstance(o, PiPoly):
            return o
        if isinstance(o, (int, F)):
            return PiPoly([F(o)])
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
        return hash(self.c)

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
                if coeff >= 0:
                    v0 += coeff * plo
                    v1 += coeff * phi
                else:
                    v0 += coeff * phi
                    v1 += coeff * plo
                plo *= lo
                phi *= hi
            if v0 > 0:
                return 1
            if v1 < 0:
                return -1
            digits *= 2
        raise ArithmeticError(
            "could not decide the sign of a ℚ[π] value within 100 digits of π "
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
