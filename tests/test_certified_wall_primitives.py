"""arcsin, ln, arctan certified — the primitives the exact-field WALLS need.

`_EXACT_FIELD_BOUNDARY` marks a handful of cells as outside every ℚ[√d][π]: a
prism through a sphere (arcsin), a prism through a cone (ln(1+√2), by Baker), a
cone's fillet (arctan, by Lindemann). Every one of those constants is
transcendental AND bracketable, so under ADR-0019/0023 they are certified
answers, not permanent refusals. These pin the primitives that make that true.

Containment is checked against identities and a float cross-check at a
tolerance a double can honour — the brackets are far tighter than a double, so
`lo <= math.asin(v) <= hi` would fail on the ORACLE, the trap documented in
`test_certified_trigonometry`.
"""

import math
from fractions import Fraction as F

import pytest

from forgekernel.interval import (CInterval, arccos_rational, arcsin_rational,
                                  arctan_rational, exp_rational, ln_rational,
                                  pi_interval)

W = 1e-12


@pytest.mark.parametrize("v", [F(n, 16) for n in range(-16, 17)])
def test_arcsin_plus_arccos_is_pi_over_two(v):
    """The defining identity, in certified arithmetic — no float anywhere."""
    s = arcsin_rational(v)
    c = arccos_rational(v)
    total = s + c
    half_pi = pi_interval() * CInterval.exact(F(1, 2))
    assert total.lo <= half_pi.hi and half_pi.lo <= total.hi, (v, total)


@pytest.mark.parametrize("v", [F(-3, 4), F(0), F(1, 2), F(99, 100)])
def test_arcsin_agrees_with_the_float_library(v):
    a = arcsin_rational(v)
    assert abs(float(a.mid) - math.asin(float(v))) < W, (v, a)


@pytest.mark.parametrize("x", [F(2), F(1, 3), F(7), F(10), F(1, 100)])
def test_ln_and_exp_are_inverse_and_certified(x):
    lx = ln_rational(x)
    assert abs(float(lx.mid) - math.log(float(x))) < W, (x, lx)
    # exp(ln x) must bracket x — a round trip through both primitives
    back = exp_rational(lx.mid)
    assert back.lo <= x <= back.hi or abs(float(back.mid) - float(x)) < W


def test_exp_brackets_and_refuses_out_of_domain():
    for t in (F(1), F(-2), F(1, 2), F(-5)):
        e = exp_rational(t)
        assert e.lo <= math.exp(float(t)) <= e.hi or \
            abs(float(e.mid) - math.exp(float(t))) < W, (t, e)
    with pytest.raises(ValueError):
        exp_rational(F(100))                  # past the tail-bound domain


@pytest.mark.parametrize("v", [F(1), F(5, 2), F(-3), F(1, 4), F(0)])
def test_arctan_agrees_with_the_float_library(v):
    a = arctan_rational(v)
    assert abs(float(a.mid) - math.atan(float(v))) < W, (v, a)


def test_the_wall_constants_are_now_computable():
    """The three constants `_EXACT_FIELD_BOUNDARY` calls permanent, bracketed.

    ln(1+√2) = arcsinh(1) is the cone-through-a-prism wall; arctan(5/2) is the
    turn angle in a cone's fillet; arcsin appears in a prism through a sphere.
    None is in any exact field the kernel has; each has a certified bracket.
    """
    ln_1p_sqrt2 = ln_rational(F(1) + F(math.sqrt(2)).limit_denominator(10 ** 12))
    assert abs(float(ln_1p_sqrt2.mid) - math.log(1 + math.sqrt(2))) < 1e-9
    assert ln_1p_sqrt2.sign() > 0

    turn = arctan_rational(F(5, 2))
    assert turn.sign() > 0
    assert float(turn.width) < 1e-10


def test_arcsin_refuses_out_of_range():
    with pytest.raises(ValueError):
        arcsin_rational(F(3, 2))
    with pytest.raises(ValueError):
        ln_rational(F(0))
