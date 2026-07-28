"""A rotated plane's normal must not crash the unit-normal predicate.

``_unit_normal`` asks "is there a RATIONAL unit normal here", and answers by
testing whether |n|² is a perfect square. It read ``nn.numerator`` directly,
which assumes ``nn`` is a ``Fraction`` — but a rotated body's plane normal has
``SurdVal`` components, so ``nn`` is a ``SurdVal`` even when its VALUE is
rational: a 45°-turned box canonicalises to n = (1,−1,0), nn = 2, and 2 arrives
wearing the ℚ[√2] tag it acquired on the way.

Result: ``AttributeError: 'SurdVal' object has no attribute 'numerator'`` —
which gitcad turned into a KernelError whose entire message was that sentence,
and which crashed ``shell`` outright one caller over. Reachable by rotating a
box 45° and chamfering it, which is not an exotic thing to ask.

Same lesson as ``body._as_fraction``, one module over: a value from a wider
exact field must be ASKED whether it is rational, never assumed either way.
The predicate's contract is unchanged — a rational unit normal or None — and
``_unit_normal_exact`` still picks up the ℚ[√d] case behind it, which is what
lets a 45° box chamfer at all.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

from forgekernel.brep import Solid, _unit_normal, _unit_normal_exact
from forgekernel.exact import Plane
from forgekernel.kernel import rotate
from forgekernel.surd import SurdVal


def _planes(solid):
    return list({p.plane.canonical(): p.plane for p in solid.polys}.values())


@pytest.mark.parametrize("deg", [30, 45, 60, 90, 135, 180])
def test_no_rotation_crashes_the_predicate(deg):
    s = rotate(Solid.box(20, 20, 10), (0, 0, 1), deg)
    for pl in _planes(s):
        _unit_normal(pl)                     # must not raise


@pytest.mark.parametrize("deg", [30, 45, 60, 90, 135, 180])
def test_the_exact_unit_normal_is_a_unit_vector(deg):
    """The point of the predicate: whatever comes back has length 1."""
    s = rotate(Solid.box(20, 20, 10), (0, 0, 1), deg)
    for pl in _planes(s):
        u = _unit_normal_exact(pl)
        assert u is not None
        n2 = sum(float(c) * float(c) for c in u)
        assert n2 == pytest.approx(1.0, abs=1e-12), f"|n|² = {n2} at {deg}°"


def test_a_rational_value_wearing_a_surd_tag_is_still_rational():
    """The exact shape of the bug: nn = SurdVal(2, 0, 2) IS 2."""
    pl = Plane((SurdVal(1, 0, 2), SurdVal(-1, 0, 2), SurdVal(0, 0, 2)),
               SurdVal(0, 0, 2))
    assert _unit_normal(pl) is None          # |n| = √2, no RATIONAL unit
    u = _unit_normal_exact(pl)               # but ℚ[√2] holds it exactly
    assert sum(float(c) * float(c) for c in u) == pytest.approx(1.0, abs=1e-12)


def test_an_axis_aligned_normal_still_comes_back_rational():
    """The control: nothing about the widening may change the ℚ answer."""
    pl = Plane((F(0), F(0), F(2)), F(6))
    assert _unit_normal(pl) == (F(0), F(0), F(1))
    pl2 = Plane((F(3), F(4), F(0)), F(5))    # 3-4-5, |n| = 5 exactly
    assert _unit_normal(pl2) == (F(3, 5), F(4, 5), F(0))


def test_a_genuinely_irrational_normal_declines_rationally():
    """|n|² = 3 is not a perfect square, so there is no rational unit normal
    and the predicate must say so rather than guess one."""
    pl = Plane((F(1), F(1), F(1)), F(0))
    assert _unit_normal(pl) is None
    u = _unit_normal_exact(pl)
    assert sum(float(c) * float(c) for c in u) == pytest.approx(1.0, abs=1e-12)
    assert float(u[0]) == pytest.approx(1 / math.sqrt(3), abs=1e-12)
