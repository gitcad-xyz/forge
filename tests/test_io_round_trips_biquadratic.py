"""A shape the kernel can BUILD, the text format must be able to SAY.

ADR-0004: text is source, geometry is a build artifact. A body whose
coordinates the kernel produces but ``dumps_body`` cannot write is a hole in
that rule — the shape exists and has no source.

``_num`` knew ``Fraction`` and ``SurdVal`` and fell through to
``_fr(Fraction(v))`` for anything else, which raises TypeError on a ``BiSurd``.
Unreachable until #127: a boolean across a 45° body (ℚ[√2]) and a 30° one
(ℚ[√3]) lands in ℚ(√2,√3), and before promotion that combination refused
before any coordinate was ever built. The moment it computed, saving it
crashed.

Same defect class as ``body._as_fraction`` (which read a BiSurd as its
rational part) and ``notch.py`` (which coerced one with ``Fraction``): code
written when there was exactly one field above ℚ, meeting a second one.
"""

from __future__ import annotations

import pytest

from forgekernel.bisurd import BiSurd
from forgekernel.io import _num, _unnum
from forgekernel.surd import SurdVal, sqrt_rational


CASES = [
    BiSurd(0, 1, 1, 0, 2, 3),                    # √2 + √3
    BiSurd(960, 0, -20, 0, 2, 3),                # the volume that was truncated
    BiSurd(0, 0, 0, 1, 2, 3),                    # √6 alone, in the wide tag
    BiSurd(-7, 3, -5, 11, 5, 7),
    sqrt_rational(2) + sqrt_rational(6),         # promotion via generator change
]


@pytest.mark.parametrize("v", CASES, ids=lambda v: repr(v)[:28])
def test_a_biquadratic_coordinate_round_trips(v):
    s = _num(v)
    back = _unnum(s)
    assert back == v, f"{s!r} came back as {back!r}"


@pytest.mark.parametrize("v", CASES, ids=lambda v: repr(v)[:28])
def test_the_encoding_is_a_string_not_a_crash(v):
    assert isinstance(_num(v), str)


def test_a_wide_tag_on_a_narrow_value_spells_it_NARROW():
    """ADR-0004's byte-canonical rule. BiSurd(3,0,0,0,·,·) IS the rational 3
    and BiSurd(0,1,0,0,2,3) IS √2 — writing them in the wide form would make a
    round trip through a wider field rewrite a file for a geometric no-op,
    which is the exact defect docket S1 recorded for SurdVal."""
    assert _num(BiSurd(3, 0, 0, 0, 2, 3)) == _num(3)
    assert _num(BiSurd(0, 1, 0, 0, 2, 3)) == _num(SurdVal(0, 1, 2))
    assert _num(BiSurd(5, 0, 2, 0, 2, 3)) == _num(SurdVal(5, 2, 3))
    assert _num(BiSurd(0, 0, 0, 1, 2, 3)) == _num(SurdVal(0, 1, 6))


def test_the_existing_encodings_are_untouched():
    from fractions import Fraction

    assert _num(Fraction(3, 4)) == _num(Fraction(3, 4))
    assert _unnum(_num(Fraction(3, 4))) == Fraction(3, 4)
    assert _unnum(_num(SurdVal(1, 2, 5))) == SurdVal(1, 2, 5)
    assert _unnum(_num(SurdVal(7, 0, 5))) == 7          # rational stays narrow
    assert _num(SurdVal(7, 0, 5)) == _num(7)
