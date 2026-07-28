"""Two radicals meet: promote if there is a home, refuse by NAME if not.

The original version of this file pinned a refusal — ℚ[√2] and ℚ[√3] had no
common home, so ``√2 + √3`` raised. That was true of the kernel, never of the
mathematics: ℚ(√2,√3) is a perfectly ordinary biquadratic field, and ``BiSurd``
had held it (arithmetic, comparison, sign, inverse) the whole time. The two
types were simply never connected. #127 connected them, so the pin came down.

What survives is the part that was actually about correctness: when a pair has
no ``BiSurd`` home, the kernel must say so by name rather than invent a field.
``SurdVal(0,1,6) + SurdVal(0,1,10)`` is that case. ``BiSurd`` requires
``gcd(p, q) == 1`` because its coercion matches on the radicand TAG, and
√(6·10) = √60 carries the tag 15 once its square is factored out. Usually that
is only a naming problem — √2 with √6 is ℚ(√2,√3) under other names, and
``_promote`` changes generators to reach it. ℚ(√6,√10) has no such escape: its
radicals are 6 = 2·3, 10 = 2·5 and 15 = 3·5, so no two of them are coprime and
this normal form cannot name the field at all.

Callers have to tell "outside the field" apart from "bug" and from "unwritten
algorithm", and a bare ``ValueError`` gives them nothing to catch on. Every
sibling refusal in forge is named (``BooleanUnsupported``,
``TrimLoopUnstitchable``, ``ShellAuditUncertified``, ``NotchRefused``, …);
``surd`` was the exception. Found via gitcad, where a bbox comparison inside
``boolean`` let this out of the Kernel seam raw — a crash, by the capability
bench's own taxonomy.
"""

import math

import pytest

from forgekernel.bisurd import BiSurd
from forgekernel.surd import MixedRadicals, SurdVal, sqrt_rational


def _r2():
    return sqrt_rational(2)


def _r3():
    return sqrt_rational(3)


# --------------------------------------------------------------------------
# coprime square-free pair -> promotes, and to the RIGHT number
# --------------------------------------------------------------------------


def test_a_coprime_pair_promotes_to_the_biquadratic_field():
    s = _r2() + _r3()
    assert isinstance(s, BiSurd)
    assert float(s) == pytest.approx(math.sqrt(2) + math.sqrt(3), abs=1e-12)


@pytest.mark.parametrize(
    "op,truth",
    [
        ("add", math.sqrt(2) + math.sqrt(3)),
        ("sub", math.sqrt(2) - math.sqrt(3)),
        ("mul", math.sqrt(6)),
    ],
)
def test_every_arithmetic_route_promotes(op, truth):
    a, b = _r2(), _r3()
    fns = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}
    assert float(fns[op]()) == pytest.approx(truth, abs=1e-12)


def test_comparison_across_the_two_fields_is_exact():
    """√2 < √3 needs the SIGN of a difference that lives in neither field
    alone — the comparison route was the one that leaked a bare ValueError
    into gitcad's bbox early-out."""
    assert _r2() < _r3()
    assert _r2() <= _r3()
    assert not (_r3() < _r2())
    assert _r3() > _r2()
    # equal values across the promotion path must not compare as unequal
    assert not (_r2() < _r2() + SurdVal(0, 0, 3))


# --------------------------------------------------------------------------
# no home -> named refusal, still a ValueError
# --------------------------------------------------------------------------


def _r6():
    return sqrt_rational(6)


def _r10():
    return sqrt_rational(10)


def test_a_pair_with_no_biquadratic_home_raises_the_named_error():
    with pytest.raises(MixedRadicals) as exc:
        _r6() + _r10()
    assert exc.value.radicals == (6, 10)
    assert "√6" in str(exc.value) and "√10" in str(exc.value)


def test_it_is_still_a_valueerror():
    """Existing `except ValueError` handlers around surd arithmetic must keep
    working — naming was a widening, not a contract change."""
    with pytest.raises(ValueError):
        _r6() - _r10()


@pytest.mark.parametrize("op", ["add", "sub", "mul", "le", "lt"])
def test_every_arithmetic_route_names_it(op):
    a, b = _r6(), _r10()
    fns = {
        "add": lambda: a + b,
        "sub": lambda: a - b,
        "mul": lambda: a * b,
        "le": lambda: a <= b,
        "lt": lambda: a < b,
    }
    with pytest.raises(MixedRadicals):
        fns[op]()


def test_a_shared_factor_changes_generators_instead_of_refusing():
    """√2 and √6 share a factor, so the pair (2,6) is not a legal ``BiSurd``
    tag — but √6 is exactly ℚ(√2,√3)'s third basis radical √(pq), so the field
    is right there under different names. Promotion changes generators to
    (2, 6/2) rather than turn away a body it can already hold.

    A body with √2 face normals and √6 edge directions is not exotic; it is a
    45°-rotated prism meeting a (1,1,2) edge.
    """
    for x, y in ((2, 6), (3, 6), (5, 10), (3, 15)):
        s = sqrt_rational(x) + sqrt_rational(y)
        assert float(s) == pytest.approx(math.sqrt(x) + math.sqrt(y), abs=1e-12)
    # and the result is EXACT, not merely close: (√2+√6)² = 8 + 4√3
    root = sqrt_rational(2) + sqrt_rational(6)
    assert (root * root).demote() == SurdVal(8, 4, 3)


def test_a_square_inside_the_radicand_never_reaches_promotion():
    """12 = 4·3, so √3 + √12 is √3 + 2√3 = 3√3 and never left ℚ[√3].
    ``sqrt_rational`` factors the square out at construction."""
    assert sqrt_rational(3) + sqrt_rational(12) == SurdVal(0, 3, 3)


# --------------------------------------------------------------------------
# one field up: BiSurd's own boundary is named the same way
# --------------------------------------------------------------------------


def test_a_value_outside_a_bisurd_field_is_named_not_bare():
    """√5 really is not in ℚ(√2,√3), and the refusal has to be catchable by
    the same handler — it left the seam as a bare ValueError otherwise, which
    is precisely the defect this whole file exists for, one field higher."""
    with pytest.raises(MixedRadicals) as exc:
        BiSurd(0, 1, 0, 0, 2, 3) + sqrt_rational(5)
    assert exc.value.radicals == (2, 3, 5)
    assert "ℚ(√2,√3)" in str(exc.value)


def test_two_different_biquadratic_fields_are_named():
    """√5 + √7 cannot demote out of ℚ(√5,√7) — it needs both generators — so
    this is the genuine field-vs-field case, degree 8 and out of reach."""
    with pytest.raises(MixedRadicals) as exc:
        BiSurd(0, 1, 0, 0, 2, 3) + BiSurd(0, 1, 1, 0, 5, 7)
    assert exc.value.radicals == (2, 3, 5, 7)


def test_a_foreign_tag_that_names_a_value_we_hold_still_coerces():
    """BiSurd(0,1,0,0,5,7) IS √5 — a one-radical value wearing a wide tag.
    Demotion has to strip the tag before the field check, or ordinary values
    would refuse for cosmetic reasons."""
    assert BiSurd(0, 1, 0, 0, 5, 11) + BiSurd(0, 0, 1, 0, 3, 5) == \
        sqrt_rational(5) * 2


def test_an_illegal_field_tag_is_named():
    with pytest.raises(MixedRadicals) as exc:
        BiSurd(0, 1, 0, 0, 6, 10)
    assert exc.value.radicals == (6, 10)


# --------------------------------------------------------------------------
# things that must not have changed
# --------------------------------------------------------------------------


def test_a_rational_surd_still_combines_with_anything():
    """b == 0 means the value is rational and carries no radical, so it must
    NOT trip the guard — otherwise ordinary integers would refuse."""
    plain = SurdVal(5, 0, 1)
    assert plain + _r2() == SurdVal(5, 1, 2)
    assert plain + _r3() == SurdVal(5, 1, 3)
    assert (_r2() * 0 + 7) + _r3() == SurdVal(7, 1, 3)
    assert plain + _r6() == SurdVal(5, 1, 6)


def test_same_radical_is_exact_as_before():
    assert _r2() + _r2() == SurdVal(0, 2, 2)
    assert (_r2() * _r2()) == SurdVal(2, 0, 1)
    assert _r6() + _r6() == SurdVal(0, 2, 6)
