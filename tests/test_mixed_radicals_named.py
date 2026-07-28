"""Leaving the quadratic field raises a NAMED exception, not a bare one.

ℚ[√2] and ℚ[√3] have no common home until the biquadratic tower (K3.1), so
combining values from the two is a permanent boundary of the number field —
not a bug, not an unwritten algorithm. Callers have to tell those three
apart, and a bare ``ValueError`` gives them nothing to catch on.

Every sibling refusal in forge is already named (``BooleanUnsupported``,
``TrimLoopUnstitchable``, ``ShellAuditUncertified``, ``NotchRefused``, …);
``surd`` was the exception. Found via gitcad, where a bbox comparison inside
``boolean`` let this out of the Kernel seam raw — a crash, by the capability
bench's own taxonomy.
"""

import pytest

from forgekernel.surd import MixedRadicals, SurdVal, sqrt_rational


def _r2():
    return sqrt_rational(2)


def _r3():
    return sqrt_rational(3)


def test_mixing_radicals_raises_the_named_error():
    with pytest.raises(MixedRadicals) as exc:
        _r2() + _r3()
    assert exc.value.radicals == (2, 3)
    assert "√2" in str(exc.value) and "√3" in str(exc.value)
    assert "K3.1" in str(exc.value)


def test_it_is_still_a_valueerror():
    """Existing `except ValueError` handlers around surd arithmetic must
    keep working — this is a naming change, not a contract change."""
    with pytest.raises(ValueError):
        _r2() - _r3()


@pytest.mark.parametrize("op", ["add", "sub", "mul", "le", "lt"])
def test_every_arithmetic_route_names_it(op):
    a, b = _r2(), _r3()
    fns = {
        "add": lambda: a + b,
        "sub": lambda: a - b,
        "mul": lambda: a * b,
        "le": lambda: a <= b,
        "lt": lambda: a < b,
    }
    with pytest.raises(MixedRadicals):
        fns[op]()


def test_a_rational_surd_still_combines_with_anything():
    """b == 0 means the value is rational and carries no radical, so it
    must NOT trip the guard — otherwise ordinary integers would refuse."""
    plain = SurdVal(5, 0, 1)
    assert plain + _r2() == SurdVal(5, 1, 2)
    assert plain + _r3() == SurdVal(5, 1, 3)
    assert (_r2() * 0 + 7) + _r3() == SurdVal(7, 1, 3)


def test_same_radical_is_exact_as_before():
    assert _r2() + _r2() == SurdVal(0, 2, 2)
    assert (_r2() * _r2()) == SurdVal(2, 0, 1)
