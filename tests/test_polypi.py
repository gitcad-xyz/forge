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
