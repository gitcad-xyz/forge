"""BiSurd — the biquadratic field ℚ(√p, √q) (K3.1's first real tower widening).

The consumer that forced it: ``chamfered box × fillet(all)``. The blend volume
is V = 8040 − 456√2 + (166/3 − 6√2 + 2√6)π — the π-coefficient generates
ℚ(√2,√6) = ℚ(√2,√3), degree 4 over ℚ, one surd wider than SurdVal's single
square-free d. The load-bearing closure is (√2)(√3) = √6: products of the two
generators land on the third basis radical, never outside the field.

Every identity here is checked EXACTLY (coefficient equality) and the float
image is compared against independent ``math.sqrt`` arithmetic — the exact
value must project onto the right real number, not merely be self-consistent.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr

import pytest

from forgekernel.bisurd import BiSurd
from forgekernel.exact import F
from forgekernel.surd import SurdVal


def bs(a=0, b=0, c=0, e=0):
    return BiSurd(a, b, c, e, 2, 3)


S2 = bs(0, 1)
S3 = bs(0, 0, 1)
S6 = bs(0, 0, 0, 1)


def approx(x: BiSurd) -> float:
    return (float(x.a) + float(x.b) * math.sqrt(2)
            + float(x.c) * math.sqrt(3) + float(x.e) * math.sqrt(6))


# -- the closure that names the field ---------------------------------------

def test_sqrt2_times_sqrt3_is_sqrt6():
    assert S2 * S3 == S6
    assert S3 * S2 == S6


def test_generator_squares_are_rational():
    assert S2 * S2 == 2
    assert S3 * S3 == 3
    assert S6 * S6 == 6


def test_sqrt6_products_fold_back():
    assert S2 * S6 == 2 * S3          # √2·√6 = 2√3
    assert S3 * S6 == 3 * S2          # √3·√6 = 3√2


# -- exhaustive arithmetic over the whole basis -----------------------------

BASIS = [bs(1), S2, S3, S6]
COEFFS = [Fr(-3, 2), Fr(0), Fr(1), Fr(5, 3)]


def test_multiplication_matches_float_on_a_grid():
    vals = [bs(a, b, c, e)
            for a in (-2, 0, Fr(1, 2)) for b in (-1, 0, Fr(2, 3))
            for c in (0, 1, Fr(-3, 4)) for e in (0, Fr(1, 5), 2)]
    for x in vals[::7]:
        for y in vals[::11]:
            for got, want in (
                    (x + y, approx(x) + approx(y)),
                    (x - y, approx(x) - approx(y)),
                    (x * y, approx(x) * approx(y))):
                assert isinstance(got, BiSurd)
                assert math.isclose(approx(got), want,
                                    rel_tol=1e-12, abs_tol=1e-12)


def test_division_is_exact_inverse():
    x = bs(Fr(1, 2), -3, Fr(2, 7), 1)
    y = bs(-1, Fr(5, 3), 2, Fr(-1, 4))
    q = x / y
    assert q * y == x
    assert math.isclose(approx(q), approx(x) / approx(y), rel_tol=1e-12)
    # division by each pure radical, and by a rational
    for d in (bs(7), S2, S3, S6):
        assert (x / d) * d == x
    assert x / 2 == bs(Fr(1, 4), Fr(-3, 2), Fr(1, 7), Fr(1, 2))


def test_division_by_zero_refuses():
    with pytest.raises(ZeroDivisionError):
        bs(1) / bs(0)


# -- exact sign and order ---------------------------------------------------

def test_sign_zero_iff_all_coefficients_zero():
    # {1, √2, √3, √6} is a ℚ-basis: no nonzero combination vanishes
    assert bs(0).sign() == 0
    for a in COEFFS:
        for b in COEFFS:
            for c in COEFFS:
                for e in COEFFS:
                    v = bs(a, b, c, e)
                    got = v.sign()
                    if a == b == c == e == 0:
                        assert got == 0
                    else:
                        fl = approx(v)
                        # every grid value is far from zero in float terms
                        assert got == (1 if fl > 0 else -1)


def test_sign_decides_a_tight_cancellation():
    # 1 + √2 − √3 + ... values within 1e-3 of zero, sign still exact:
    # √2 + √3 − √6 − 1/1000000 vs the same + 2/1000000... use the honest
    # near-cancellation 3√2 − 2√3 − (float would need care): 3√2 = 4.2426,
    # 2√3 = 3.4641; instead pit √6 against its convergent 4899/2001... keep
    # it simple and PROVEN: x − y where y is x with the last coefficient
    # nudged by 10^-12 must recover the nudge's sign.
    x = bs(Fr(1, 3), Fr(-2, 7), Fr(5, 11), Fr(-3, 13))
    eps = Fr(1, 10**12)
    assert (x - (x + bs(0, 0, 0, eps))).sign() == -1
    assert (x - (x - bs(eps))).sign() == 1
    assert (x - x).sign() == 0


def test_order_operators():
    assert S2 < S3 < 2 < S6
    assert S6 > S3 > S2 > 1
    assert bs(0, 1, 1) > bs(3)          # √2 + √3 = 3.146… > 3
    assert bs(0, 1, 1) < bs(Fr(63, 20))  # … < 3.15
    assert abs(bs(0, -1)) == S2
    assert -S2 < 0 < S2


# -- coercion: rationals, SurdVal, and refusals -----------------------------

def test_surdval_coerces_by_radical():
    assert bs(1, 2) == SurdVal(1, 2, 2)
    assert bs(1, 0, 2) == SurdVal(1, 2, 3)
    assert bs(1, 0, 0, 2) == SurdVal(1, 2, 6)
    assert bs(5) == SurdVal(5, 0, 1)
    assert S2 + SurdVal(0, 1, 3) == bs(0, 1, 1)
    assert SurdVal(0, 1, 3) + S2 == bs(0, 1, 1)   # reflected path
    assert SurdVal(0, 1, 2) * S3 == S6            # reflected mul
    assert S3 * SurdVal(0, 1, 2) == S6


def test_foreign_radical_refuses():
    with pytest.raises(ValueError):
        S2 + SurdVal(0, 1, 5)
    with pytest.raises(ValueError):
        S2 + BiSurd(0, 1, 1, 0, 2, 5)     # √2 + √5 needs a three-radical tower
    # but a foreign TAG holding a value this field contains coerces fine:
    assert S2 + BiSurd(0, 1, 0, 0, 2, 5) == 2 * S2
    assert BiSurd(0, 1, 0, 0, 2, 5) == S2


def test_field_spec_guards():
    with pytest.raises(ValueError):
        BiSurd(0, 1, 0, 0, 4, 3)      # 4 is not square-free
    with pytest.raises(ValueError):
        BiSurd(0, 1, 0, 0, 3, 2)      # p < q required (one canonical name)
    with pytest.raises(ValueError):
        BiSurd(0, 1, 0, 0, 2, 2)      # distinct radicals
    with pytest.raises(ValueError):
        BiSurd(0, 1, 0, 0, 2, 6)      # gcd > 1: √2·√6 = 2√3 leaves the basis
    with pytest.raises(ValueError):
        BiSurd(0, 1, 0, 0, 1, 3)      # p > 1: √1 is rational


def test_a_float_cannot_smuggle_in_approximation():
    # floats convert to their EXACT binary Fraction (the F() contract) — the
    # value is exact relative to the given input, never a hidden epsilon
    assert bs(0.5) == bs(Fr(1, 2))


# -- hashing and demotion ---------------------------------------------------

def test_hash_consistent_with_equality_across_types():
    assert hash(bs(3)) == hash(Fr(3))
    assert hash(bs(1, 2)) == hash(SurdVal(1, 2, 2))
    assert hash(bs(1, 0, 0, 2)) == hash(SurdVal(1, 2, 6))
    assert bs(3) == 3 and 3 == bs(3)


def test_demote_returns_the_smallest_field():
    assert isinstance(bs(3).demote(), Fr)
    d = bs(1, 2).demote()
    assert isinstance(d, SurdVal) and d.d == 2
    d = bs(1, 0, 0, 5).demote()
    assert isinstance(d, SurdVal) and d.d == 6
    assert isinstance(bs(1, 1, 1).demote(), BiSurd)


def test_exact_F_passes_bisurd_through():
    v = bs(1, 1, 1, 1)
    assert F(v) is v


# -- ℚ(√2,√3)[π]: PiPoly must carry BiSurd coefficients ---------------------

def test_pipoly_with_bisurd_coefficients():
    from forgekernel.polypi import PiPoly

    # the chamfered-box fillet's π-coefficient: 166/3 − 6√2 + 2√6
    v = PiPoly([bs(8040, -456), bs(Fr(166, 3), -6, 0, 2)])
    assert v.degree == 1
    assert v.sign() == 1
    assert (v - v).sign() == 0
    truth = (8040 - 456 * math.sqrt(2)
             + (166 / 3 - 6 * math.sqrt(2) + 2 * math.sqrt(6)) * math.pi)
    assert math.isclose(float(v), truth, rel_tol=1e-12)
    # order against a rational bracket, decided exactly
    assert v > 7557 and v < 7558
    # arithmetic keeps the field: multiplying by π shifts the polynomial
    from forgekernel.polypi import PiPoly as P
    w = v * P.term(1, 1)
    assert w.degree == 2 and w[1] == bs(8040, -456)


def test_pipoly_demotes_pure_rational_bisurd_coefficients():
    from forgekernel.polypi import PiPoly

    v = PiPoly([bs(3), bs(0, 1, 1)])
    assert v[0] == Fr(3) and isinstance(v[0], Fr)
