"""ℚ[π] as a polynomial ring (the number field a fillet needs).

``PiVal`` is ``a + bπ``, which covers every volume the kernel computed before
blends: πr²h, (4/3)πr³, (π/3)h(r1²+r1r2+r2²). A fillet breaks it — revolving an
arc sweeps a torus, and Pappus gives V = 2πR·πa² = 2π²Ra.

π is transcendental, so ℚ[π] is FREE: equality is coefficient equality and
zero-testing needs no numerics at all. That is the property the charter cares
about, because it means no float ever decides whether two volumes are equal.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

from forgekernel.polypi import PiPoly

PI = PiPoly.term(1, 1)


def test_the_ring_operations_are_exact() -> None:
    a = PiPoly([1, 2, 3])                       # 1 + 2pi + 3pi^2
    b = PiPoly([0, 1])                          # pi
    assert (a + b).c == (F(1), F(3), F(3))
    assert (a - b).c == (F(1), F(1), F(3))
    assert (a * b).c == (F(0), F(1), F(2), F(3))
    assert (a / 2).c == (F(1, 2), F(1), F(3, 2))
    assert (b ** 3).c == (F(0), F(0), F(0), F(1))


def test_a_torus_volume_needs_pi_squared_and_gets_it() -> None:
    """Pappus: V = 2 pi R * pi a^2. This is the number PiVal cannot hold."""
    R, a = F(10), F(2)
    v = PiPoly.term(2 * R * a * a, 2)
    assert v.degree == 2
    assert float(v) == pytest.approx(2 * math.pi ** 2 * 10 * 4)


def test_equality_and_zero_are_decided_without_any_numerics() -> None:
    """Transcendence is what makes this exact: no combination of powers of pi
    with rational coefficients can cancel unless every coefficient is zero. A
    kernel that compared volumes by float would call these equal."""
    assert PiPoly([0, 1]) != PiPoly([F(355, 113)])      # pi vs a good rational
    assert PiPoly([0, 0, 1]) != PiPoly([F(9869604401089358618834, 10 ** 21)])
    assert (PiPoly([1, 2, 3]) - PiPoly([1, 2, 3])).sign() == 0
    assert PiPoly([0, 0, 0]) == PiPoly.rational(0)
    assert hash(PiPoly([2])) == hash(PiPoly.rational(2))


SIGNS = [
    ([1], 1), ([-1], -1), ([0], 0),
    ([0, 1], 1), ([0, -1], -1),
    # pi^2 - 9.8696044 ... : positive by a hair, and no float would know
    ([F(-98696044010893586188344909998761511353, 10 ** 37), 0, 1], 1),
    ([F(-98696044010893586188344909998761511354, 10 ** 37), 0, 1], -1),
    # 22/7 - pi > 0, 333/106 - pi < 0 — the classic near-misses
    ([F(22, 7), -1], 1), ([F(333, 106), -1], -1),
]


@pytest.mark.parametrize("coeffs,want", SIGNS)
def test_sign_narrows_the_enclosure_until_it_is_certain(coeffs, want) -> None:
    """Order is the only operation that needs to look at pi's value, and it
    narrows a rational enclosure until the interval excludes zero — which must
    happen for any nonzero element."""
    assert PiPoly(coeffs).sign() == want


def test_comparison_follows_the_sign() -> None:
    assert PiPoly([0, 1]) > 3
    assert PiPoly([0, 1]) < 4
    assert PiPoly([F(22, 7)]) > PiPoly([0, 1])
    assert PiPoly([F(333, 106)]) < PiPoly([0, 1])
    assert PiPoly([1, 1]) >= PiPoly([1, 1])


def test_dividing_by_a_polynomial_refuses_rather_than_leaving_the_ring() -> None:
    """ℚ[π] is a RING, not a field: 1/π is not in it. Silently returning a
    float here is exactly the charter violation this type exists to prevent."""
    with pytest.raises(ValueError, match="ring"):
        PiPoly([1]) / PiPoly([0, 1])
    with pytest.raises(ZeroDivisionError):
        PiPoly([1]) / 0


def test_negative_powers_are_refused() -> None:
    with pytest.raises(ValueError, match="negative powers"):
        PiPoly.term(1, -1)
    with pytest.raises(ValueError, match="non-negative"):
        PiPoly([1, 1]) ** -2


def test_it_lifts_the_legacy_representation() -> None:
    from forgekernel.quadric import PiVal

    lifted = PiPoly.from_pival(PiVal(F(7), F(3)))
    assert lifted.c == (F(7), F(3))
    assert float(lifted) == pytest.approx(7 + 3 * math.pi)


def test_trailing_zero_coefficients_do_not_change_identity() -> None:
    assert PiPoly([2, 0, 0]) == PiPoly([2])
    assert PiPoly([2, 0, 0]).degree == 0
    assert hash(PiPoly([2, 0, 0])) == hash(PiPoly([2]))


def test_float_is_only_the_reporting_boundary() -> None:
    v = PiPoly([1, 2, 3])
    assert float(v) == pytest.approx(1 + 2 * math.pi + 3 * math.pi ** 2)


def test_it_meets_the_legacy_type_in_both_directions() -> None:
    """volume() returns PiVal when the answer fits and PiPoly when it needs
    pi^2, so the two meet constantly. Neither could coerce the other, and
    comparing a filleted part's volume with an unfilleted one raised TypeError
    on two values that name the same number."""
    from forgekernel.quadric import PiVal

    assert PiPoly([1, 2]) == PiVal(F(1), F(2))
    assert PiVal(F(1), F(2)) == PiPoly([1, 2])
    assert PiPoly([1, 2]) + PiVal(F(1), F(2)) == PiPoly([2, 4])
    assert PiVal(F(1), F(2)) + PiPoly([1, 2]) == PiPoly([2, 4])
    assert PiPoly([1, 1]) < PiVal(F(1), F(2))


def test_equal_values_hash_equally() -> None:
    """PiPoly.rational(3) == 3 was True while hashing differently, so a set
    held both and a dict lookup missed."""
    assert len({PiPoly.rational(3), 3}) == 1
    assert {PiPoly.rational(3): "x"}.get(3) == "x"


def test_a_float_cannot_enter_the_exact_ring() -> None:
    with pytest.raises(ValueError, match="float"):
        PiPoly([0.1])
    PiPoly([2.0])                       # integral floats are unambiguous


def test_the_stored_digits_really_do_enclose_pi() -> None:
    """The last stored digit was ROUNDED UP, so at full precision `lo` was
    ABOVE pi and the 'enclosure' excluded it. Unreachable from sign() as the
    loop is written, but the file's whole claim is rigour."""
    import decimal
    from forgekernel.polypi import _PI_DIGITS, _pi_bounds

    decimal.getcontext().prec = len(_PI_DIGITS) + 20
    pi = decimal.Decimal(
        "3.14159265358979323846264338327950288419716939937510582097494459230"
        "78164062862089986280348253421170679821480865132823066470938446095505"
        "8223172535940812848111745028410270193852110555964462294895493038196")
    lo, hi = _pi_bounds(len(_PI_DIGITS))
    d = lambda q: decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator)
    assert d(lo) <= pi <= d(hi)


PIVAL_ORDER = [
    ("pi is positive", (0, 1), ">", 0, True),
    ("pi is not <= 0", (0, 1), "<=", 0, False),
    ("pi < 4", (0, 1), "<", 4, True),
    ("22/7 does NOT exceed pi from below", (0, 1), ">", F(22, 7), False),
    ("pi > 333/106", (0, 1), ">", F(333, 106), True),
    ("zero volume is not positive", (0, 0), ">", 0, False),
    ("a negative sweeps negative", (-1, 0), "<", 0, True),
]


@pytest.mark.parametrize("label,ab,op,rhs,want", PIVAL_ORDER,
                         ids=[p[0] for p in PIVAL_ORDER])
def test_pival_can_be_ordered_exactly(label, ab, op, rhs, want) -> None:
    """PiVal could be added, subtracted and compared for EQUALITY but not
    ordered — so the one question anyone asks of a volume, "is it positive?",
    could only be answered by ``float(v) > 0``. That is a float deciding
    whether a solid is valid, which ADR-0019 forbids."""
    from forgekernel.quadric import PiVal

    v = PiVal(F(ab[0]), F(ab[1]))
    got = {">": v > rhs, "<": v < rhs, ">=": v >= rhs, "<=": v <= rhs}[op]
    assert got is want


def test_the_boundary_a_float_gets_wrong() -> None:
    """A true volume a hair above zero rounds to 0.0 and the solid reads as
    inside-out. This is the case the exact comparison exists for."""
    from forgekernel.quadric import PiVal

    tiny = PiVal(F(1, 10 ** 400), F(0))
    assert tiny > 0
    assert float(tiny) == 0.0, "…which is exactly what float() would have said"


def test_pival_sign_agrees_with_pipoly() -> None:
    from forgekernel.quadric import PiVal

    for a, b in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (-22, 7), (22, -7)):
        v = PiVal(F(a), F(b))
        assert v.sign() == PiPoly.from_pival(v).sign()


# --- ℚ[√d][π]: π is transcendental over ALGEBRAIC extensions too -------------

def _s(a, b, d):
    from forgekernel.surd import SurdVal

    return SurdVal(F(a), F(b), d)


def test_a_napkin_ring_volume_lives_in_the_ring() -> None:
    """A sphere bored coaxially is a napkin ring: V = (π/6)h³ with
    h = 2√(R²−r²). For R=6, r=1 that is (140/3)π√35 — a π coefficient that is
    NOT rational, which is precisely what ℚ[π] could not hold."""
    v = PiPoly([F(0), _s(0, F(140, 3), 35)])
    assert v.degree == 1
    assert float(v) == pytest.approx(math.pi / 6 * (2 * math.sqrt(35)) ** 3)
    assert v.sign() == 1 and v > 0


def test_equality_stays_exact_with_no_numerics() -> None:
    """The ring is still FREE. π is transcendental over any algebraic extension
    of ℚ, not merely over ℚ, so equality is coefficient equality and zero is
    all-coefficients-zero — the property the whole type exists for."""
    a = PiPoly([F(0), _s(0, F(140, 3), 35)])
    assert a == PiPoly([F(0), _s(0, F(140, 3), 35)])
    assert (a - a).sign() == 0
    # (140/3)√35 = 276.0837…, and a rational that close must NOT compare equal
    assert a != PiPoly([F(0), F(276)])
    assert a != PiPoly([F(0), F(2760837232113, 10 ** 10)])


SURD_ORDER = [
    ("above a rational just below it", F(276), 1),
    ("below a rational just above it", F(277), -1),
    ("above a very close rational", F(2760837232113, 10 ** 10), 1),
    ("below a very close rational", F(2760837232114, 10 ** 10), -1),
]


@pytest.mark.parametrize("label,rhs,want", SURD_ORDER,
                         ids=[s[0] for s in SURD_ORDER])
def test_order_brackets_the_radical_until_it_is_certain(label, rhs, want) -> None:
    """Order is the only operation that looks at values, and the bracket comes
    from an integer square root — proven, not believed. These rationals differ
    from (140/3)√35 in the tenth decimal."""
    a = PiPoly([F(0), _s(0, F(140, 3), 35)])
    assert (a - PiPoly([F(0), rhs])).sign() == want


def test_the_sqrt_enclosure_really_does_enclose() -> None:
    """The file's whole claim is rigour, so check the bracket rather than
    trusting isqrt's off-by-one."""
    import decimal

    from forgekernel.polypi import _sqrt_bounds

    decimal.getcontext().prec = 60
    for d in (2, 3, 5, 35, 109, 12345):
        lo, hi = _sqrt_bounds(d, 25)
        assert lo * lo <= d <= hi * hi
        exact = decimal.Decimal(d).sqrt()
        assert decimal.Decimal(lo.numerator) / lo.denominator <= exact
        assert exact <= decimal.Decimal(hi.numerator) / hi.denominator


def test_a_negative_surd_coefficient_brackets_the_other_way() -> None:
    """b < 0 swaps the ends of the interval. Getting this backwards would give
    a confidently wrong sign, which is the failure mode the ring must not have."""
    v = PiPoly([F(0), _s(0, -1, 2)])            # −√2·π
    assert v.sign() == -1 and v < 0
    assert PiPoly([_s(3, -2, 2)]).sign() == 1   # 3 − 2√2 = 0.1715…
    assert PiPoly([_s(1, -1, 2)]).sign() == -1  # 1 − √2 = −0.414…


def test_a_surd_that_is_really_rational_collapses() -> None:
    """SurdVal(3, 0, 1) IS 3, and must hash and compare as 3 — otherwise a set
    holds both and a dict lookup misses."""
    assert PiPoly([_s(3, 0, 1)]) == PiPoly([F(3)])
    assert hash(PiPoly([_s(3, 0, 1)])) == hash(PiPoly([F(3)]))
    assert len({PiPoly([_s(3, 0, 1)]), PiPoly([F(3)]), 3}) == 1


def test_mixed_radicals_still_refuse() -> None:
    """ℚ[√2, √3] is a bigger field than either, and inventing it silently is
    exactly what the charter forbids (K3.1)."""
    with pytest.raises(ValueError, match="mixed radicals"):
        PiPoly([_s(0, 1, 2)]) * PiPoly([_s(0, 1, 3)])


def test_a_float_still_cannot_enter() -> None:
    with pytest.raises(ValueError, match="float"):
        PiPoly([0.1])
    with pytest.raises(ValueError, match="not an exact coefficient"):
        PiPoly(["7/2"])
