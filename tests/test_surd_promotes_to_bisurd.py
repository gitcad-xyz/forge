"""Mixing two quadratic fields PROMOTES to the biquadratic one (#127).

``SurdVal`` held ℚ[√d] for one d and raised ``MixedRadicals`` the moment two
different radicands met. ``BiSurd`` held ℚ(√p,√q) and was already general in
(p, q) — square-free, coprime, p < q — with full arithmetic, comparison, sign
and inverse. The two were simply never connected, so every mixed-radical
expression refused even when its exact home already existed in the kernel.

Found by tracing why a tapered chamfer refuses: a frustum's slanted face
normal lives in ℚ[√37] and its slanted EDGE direction in ℚ[√38], so the wedge
construction needs ℚ(√37,√38). The refusal was real; the field was too.

Promotion is only valid when the pair generates a genuine biquadratic field.
√6 and √10 do NOT: gcd is 2, and ℚ(√6,√10) = ℚ(√2,√3,√5) has degree 8, so
{1,√6,√10,√60} is not a basis. Those still refuse, and the message says why.
"""

from fractions import Fraction as F

import pytest

from forgekernel.bisurd import BiSurd
from forgekernel.surd import MixedRadicals, SurdVal, sqrt_rational


def s(a, b, d):
    return SurdVal(F(a), F(b), d)


# -- promotion happens, and the answer is right ------------------------------

def test_adding_two_radicals_lands_in_the_biquadratic_field():
    out = s(0, 1, 2) + s(0, 1, 3)                 # √2 + √3
    assert isinstance(out, BiSurd)
    assert float(out) == pytest.approx(2 ** 0.5 + 3 ** 0.5, rel=1e-12)


def test_the_case_that_blocked_a_tapered_chamfer():
    """√37 (face normal) with √38 (edge direction)."""
    out = s(0, 1, 37) + s(0, 1, 38)
    assert isinstance(out, BiSurd)
    assert float(out) == pytest.approx(37 ** 0.5 + 38 ** 0.5, rel=1e-12)


@pytest.mark.parametrize("op,fn", [
    ("+", lambda x, y: x + y),
    ("-", lambda x, y: x - y),
    ("*", lambda x, y: x * y),
])
def test_every_arithmetic_route_promotes(op, fn):
    a, b = s(1, 2, 2), s(3, 4, 3)                 # 1+2√2, 3+4√3
    out = fn(a, b)
    assert isinstance(out, BiSurd)
    expect = fn(1 + 2 * 2 ** 0.5, 3 + 4 * 3 ** 0.5)
    assert float(out) == pytest.approx(expect, rel=1e-12)


@pytest.mark.parametrize("cmp,fn", [
    ("<", lambda x, y: x < y),
    ("<=", lambda x, y: x <= y),
    (">", lambda x, y: x > y),
    (">=", lambda x, y: x >= y),
])
def test_comparisons_promote_rather_than_refuse(cmp, fn):
    a, b = s(0, 1, 2), s(0, 1, 3)                 # √2 < √3
    assert fn(a, b) == fn(2 ** 0.5, 3 ** 0.5)


def test_the_product_of_two_radicals_is_the_cross_term():
    """√2·√3 = √6 — the fourth basis element, not a collapse to ℚ."""
    out = s(0, 1, 2) * s(0, 1, 3)
    assert float(out) == pytest.approx(6 ** 0.5, rel=1e-12)


def test_a_shared_radical_still_stays_in_the_smaller_field():
    """Promotion must not fire when it is not needed: ℚ[√2] is closed."""
    out = s(1, 1, 2) + s(2, 3, 2)
    assert isinstance(out, SurdVal)
    assert out == s(3, 4, 2)


def test_a_rational_surd_mixes_with_anything_without_promoting():
    """b == 0 carries no radical, so there is nothing to mix."""
    out = SurdVal(F(5), F(0), 1) + s(0, 1, 7)
    assert isinstance(out, SurdVal) and out == s(5, 1, 7)


# -- the boundary: pairs that do NOT generate a biquadratic field ------------

def test_non_coprime_radicands_still_refuse():
    """√6 and √10 share a factor: ℚ(√6,√10) = ℚ(√2,√3,√5), degree 8, so
    {1,√6,√10,√60} is not a basis. Refusing is the correct answer, and the
    message must not pretend otherwise."""
    with pytest.raises((MixedRadicals, ValueError)) as exc:
        s(0, 1, 6) + s(0, 1, 10)
    assert "factor" in str(exc.value) or "radical" in str(exc.value).lower()


def test_promotion_survives_a_round_trip_through_sqrt_rational():
    """The real callers build their surds with sqrt_rational, not literals."""
    out = sqrt_rational(F(37)) * sqrt_rational(F(38))
    assert float(out) == pytest.approx((37 * 38) ** 0.5, rel=1e-12)
