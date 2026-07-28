"""The sanctioned exact→decimal boundary for STEP (bucket-B, 2026-07-28).

``Fraction(x)`` refuses an exact scalar on purpose: ``SurdVal``/``PiPoly``
carry ``is_exact_scalar = True`` so a value beyond ℚ can never be coerced away
by accident. STEP Part 21 is a DECIMAL format, so the conversion has to happen
somewhere — ``export_rational`` is the one place it is allowed, and it is
reached only when writing a numeral.

Before it existed, a 45°-rotated box could not be exported at all: its
coordinates live in ℚ[√2], ``Fraction()`` raised a TypeError, and the seam
reported "export_step on Solid yet — K3.7 (canonical B-rep)" as though the
canonical form lacked a capability. It did not; the numeral did.
"""

from fractions import Fraction as F

import pytest

from forgekernel.stepio import export_rational
from forgekernel.surd import SurdVal, sqrt_rational


# -- values already in ℚ must not be approximated at all ---------------------

@pytest.mark.parametrize("value", [0, 23, -7, F(3, 7), F(-22, 5), F(10 ** 20)])
def test_a_rational_passes_through_exactly(value):
    """The common case must not degrade. An integer 23 becoming
    22.999999999999999 in a STEP file would be a wrong number in the one
    artifact a shop reads."""
    out = export_rational(value)
    assert out == F(value)
    assert isinstance(out, F)


# -- values beyond ℚ get a bounded approximation ------------------------------

def test_sqrt2_is_accurate_to_the_requested_digits():
    approx = export_rational(sqrt_rational(2), digits=25)
    # proven bound WITHOUT floats: squaring must straddle 2 tightly
    err_sq = abs(approx * approx - 2)
    assert err_sq < F(1, 10 ** 24)


def test_the_result_is_a_rational_not_a_false_claim_of_exactness():
    approx = export_rational(sqrt_rational(2))
    assert isinstance(approx, F)
    assert approx * approx != 2, "√2 is irrational; an exact hit is a bug"


@pytest.mark.parametrize("d", [2, 3, 5, 29])
def test_it_brackets_correctly_for_several_radicals(d):
    r = sqrt_rational(d)
    approx = export_rational(r, digits=20)
    lo, hi = approx - F(1, 10 ** 19), approx + F(1, 10 ** 19)
    # the ONLY comparisons used are the exact ones the type provides
    assert lo <= r <= hi


def test_negative_and_mixed_surds():
    v = SurdVal(F(-5), F(3), 2)                 # -5 + 3√2
    approx = export_rational(v, digits=20)
    assert approx < 0 or approx > 0             # a real number either way
    assert approx - F(1, 10 ** 19) <= v <= approx + F(1, 10 ** 19)


def test_a_float_seed_cannot_make_the_answer_wrong():
    """The bracket is SEEDED from a float and then verified exactly, so a bad
    seed costs iterations, never correctness (ADR-0019). Exercised by asking
    for a value whose float is far from where a naive guess would start."""
    big = SurdVal(F(10 ** 15), F(1), 2)         # 1e15 + √2
    approx = export_rational(big, digits=10)
    assert approx - F(1, 10 ** 9) <= big <= approx + F(1, 10 ** 9)


def test_pi_bearing_values_convert_too():
    """The tower is not just surds — a lathe's coordinates reach ℚ[π]. The
    converter uses only ordered-field comparisons, so it needs no per-type
    code; this pins that claim rather than assuming it."""
    pytest.importorskip("forgekernel.polypi")
    from forgekernel.polypi import PiPoly

    v = PiPoly([F(0), F(1)])                    # π itself
    approx = export_rational(v, digits=18)
    assert approx - F(1, 10 ** 17) <= v <= approx + F(1, 10 ** 17)
    assert F(314, 100) < approx < F(315, 100)
