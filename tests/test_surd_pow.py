"""SurdVal ** n — the third hole in a half-built ordered field.

``SurdVal`` could add, subtract, multiply, divide, negate, compare and (since
the rotation crashes that motivated ``__abs__``) take a magnitude — but not
raise itself to an integer power. ℚ[√d] is closed under multiplication, so
``x ** 2`` is exactly ``x * x``; there was never a mathematical reason for the
gap, only a missing dunder.

The cost was a CRASH, not a refusal. gitcad #49: any body built through a
non-``z`` sketch plane is placed with an exact 120° axis-permutation rotation,
which leaves surds in its vertex coordinates. Drilling it reached

    forgekernel/quadric.py::_seg_dist2
        return (px - qx) ** 2 + (py - qy) ** 2

and died with a bare ``TypeError: unsupported operand type(s) for ** or pow():
'SurdVal' and 'int'``. That escaped gitcad's refusal wrapper entirely (it
catches GitcadError/ValueError/KeyError/NotImplementedError), so the caller got
a traceback instead of either geometry or a named wall.

This pins the operator itself, exactly, including the identities that make it
safe to use inside a distance predicate.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.surd import SurdVal


R2 = SurdVal(0, 1, 2)                    # √2
HALF_R3 = SurdVal(0, F(1, 2), 3)         # √3/2
X = SurdVal(F(3, 2), F(5, 7), 5)         # 3/2 + (5/7)√5


def test_square_of_root_two_is_exactly_two() -> None:
    assert R2 ** 2 == 2
    assert (R2 ** 2).b == 0               # collapses to a pure rational


def test_square_of_half_root_three_is_three_quarters() -> None:
    assert HALF_R3 ** 2 == F(3, 4)


def test_power_agrees_with_repeated_multiplication() -> None:
    for n in range(0, 7):
        want = SurdVal(1)
        for _ in range(n):
            want = want * X
        assert X ** n == want, n


def test_zeroth_power_is_one_and_first_is_self() -> None:
    assert X ** 0 == 1
    assert X ** 1 == X
    assert SurdVal(0) ** 0 == 1


def test_negative_power_is_the_exact_reciprocal() -> None:
    assert (R2 ** -1) * R2 == 1
    assert X ** -2 == SurdVal(1) / (X * X)


def test_zero_to_a_negative_power_refuses() -> None:
    with pytest.raises(ZeroDivisionError):
        SurdVal(0) ** -1


def test_non_integer_exponent_defers_rather_than_lying() -> None:
    """A half power leaves ℚ[√d]; returning NotImplemented gives an honest
    TypeError instead of a silently rounded float."""
    with pytest.raises(TypeError):
        X ** 0.5
    with pytest.raises(TypeError):
        X ** F(1, 2)


def test_squares_are_non_negative_so_a_distance_predicate_is_safe() -> None:
    """The property _seg_dist2 relies on: a squared surd is comparable to a
    rational radius and never negative."""
    for v in (R2, -R2, HALF_R3, X, -X, SurdVal(0)):
        assert v ** 2 >= 0


def test_bool_is_not_an_exponent() -> None:
    """bool is an int subclass; True ** would quietly mean 1. Accept it only
    as the integer it is, never as a stray flag that changes the answer."""
    assert X ** True == X


def test_seg_dist2_accepts_surd_coordinates() -> None:
    """gitcad #49's exact crash site, at the forge seam."""
    from forgekernel.quadric import _seg_dist2

    d2 = _seg_dist2(R2, R2, SurdVal(0), SurdVal(0), SurdVal(4), SurdVal(0))
    assert d2 == 2                        # point (√2,√2) to the x-axis: (√2)²
