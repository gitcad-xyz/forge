"""Certified trig must ENCLOSE, not approximate (ADR-0019).

The value of a CInterval is the guarantee that the true value is inside it. A
bracket that is merely close is a float wearing a costume, so what is tested
here is CONTAINMENT.

HOW IT IS TESTED, because the obvious way is wrong. The first draft of this
file asserted ``lo <= math.cos(t) <= hi`` and failed almost everywhere — not
because the brackets were wrong but because they are ~1e-40 wide while a
double carries ~1e-16 of error. The oracle was outside the bracket, and the
bracket was right. A double simply cannot witness an enclosure this tight.

So the enclosure is checked against IDENTITIES the true values must satisfy,
evaluated in certified arithmetic end to end — cos^2 + sin^2 = 1, the double
angle, and the exact twelfths the kernel already ships — plus the defining
property of arccos, that its bracket straddles the root under our own
certified cos. `math` is kept only as a loose cross-check at a tolerance a
double can honour, which catches a gross error without pretending to witness
the fine one.
"""

import math
from fractions import Fraction as F

import pytest

from forgekernel.interval import (CInterval, arccos, arccos_rational, cos,
                                  cos_rational, pi_interval, sin)

#: what a double can actually witness; nothing finer is asserted against it
FLOAT_WITNESS = 1e-12


def _ratios():
    """h/r over a whole diameter — the twelfths, and mostly not the twelfths."""
    return [F(n, 16) for n in range(-15, 16)]


# -- the defining property, in certified arithmetic only ----------------------

@pytest.mark.parametrize("v", _ratios())
def test_the_arccos_bracket_straddles_the_root(v):
    """cos(lo) >= v >= cos(hi): the bracket contains a solution of cos t = v.

    This is the enclosure claim itself, checked with no float anywhere. cos is
    strictly decreasing on [0, pi], so a bracket whose endpoints' cosines
    surround v must contain the unique root.
    """
    a = arccos_rational(v)
    assert cos_rational(a.lo).hi >= v, (v, a)
    assert cos_rational(a.hi).lo <= v, (v, a)


@pytest.mark.parametrize("v", _ratios())
def test_cos_undoes_arccos_within_the_bracket(v):
    """cos(arccos(v)) must contain v — a round trip through both primitives."""
    back = cos(arccos_rational(v))
    assert back.lo <= v <= back.hi, (v, back)


@pytest.mark.parametrize("num", range(-30, 31, 5))
def test_pythagoras_holds_inside_the_brackets(num):
    t = CInterval.exact(F(num, 10))
    one = cos(t) * cos(t) + sin(t) * sin(t)
    assert one.lo <= 1 <= one.hi, (num, one)


@pytest.mark.parametrize("num", range(-15, 16, 3))
def test_the_double_angle_identity_holds(num):
    """cos 2t = 2cos^2 t - 1, both sides certified — an independent route to
    the same value, so an error in the series would have to be conspiratorial
    to survive it."""
    t = F(num, 10)
    lhs = cos_rational(2 * t)
    rhs = CInterval.exact(F(2)) * cos_rational(t) * cos_rational(t) - 1
    assert lhs.lo <= rhs.hi and rhs.lo <= lhs.hi, (t, lhs, rhs)


def test_the_exact_twelfths_are_contained():
    """Where the exact path already HAS an answer, the certified one must
    agree with it. This is what lets an angle-bearing rung route through
    certified arithmetic without contradicting results it already ships."""
    pi = pi_interval()
    for cosv, mult in ((F(1), F(0)), (F(1, 2), F(1, 3)), (F(0), F(1, 2)),
                       (F(-1, 2), F(2, 3)), (F(-1), F(1))):
        got = arccos_rational(cosv)
        want = pi * CInterval.exact(mult)
        assert got.lo <= want.hi and want.lo <= got.hi, (cosv, got, want)
    # and the cosines of those angles come back where they started
    third = pi * CInterval.exact(F(1, 3))
    assert cos(third).lo <= F(1, 2) <= cos(third).hi


# -- the tangencies, which the first implementation got wrong -----------------

def test_the_endpoints_enclose_their_answers():
    """arccos(1) = 0 and arccos(-1) = pi.

    cos is FLAT where it meets +-1, so bisection cannot see these: at v = -1
    the "cos(m) < v" branch never fires, b never moves, and a climbs past pi
    into the region where cos increases again. The first implementation
    returned a bracket lying just ABOVE pi — excluding the value asked for.
    """
    zero = arccos_rational(F(1))
    assert zero.lo <= 0 <= zero.hi
    at_pi = arccos_rational(F(-1))
    pi = pi_interval()
    assert at_pi.lo <= pi.hi and pi.lo <= at_pi.hi, at_pi
    # cos of that bracket must come back to -1
    assert cos(at_pi).lo <= -1


def test_it_refuses_rather_than_returning_an_unfounded_bracket():
    """Outside the series domain the remainder bound stops bounding, and
    inside the tangency the starting bracket's premise is false. Both must
    refuse: a bound that silently stops bounding is worse than none, because
    everything downstream still reads as certified."""
    with pytest.raises(ValueError):
        # past where 30 terms can bound the tail: the precondition is
        # t^2 < (2N+1)(2N+2), so the cap is derived, not decreed
        cos_rational(F(100))
    with pytest.raises(ValueError):
        arccos_rational(F(3, 2))                    # outside [-1, 1]


def test_the_domain_follows_the_term_count():
    """The cap is a statement about the remainder bound, not a magic number:
    ask for more terms and more of the real line becomes certifiable."""
    with pytest.raises(ValueError):
        cos_rational(F(20), terms=5)
    wide = cos_rational(F(20), terms=60)
    assert wide.lo <= math.cos(20.0) + FLOAT_WITNESS
    assert wide.hi >= math.cos(20.0) - FLOAT_WITNESS


# -- interval arguments -------------------------------------------------------

def test_arccos_of_an_interval_covers_every_point_in_it():
    x = CInterval(F(1, 4), F(3, 4))
    a = arccos(x)
    for k in range(11):
        v = F(1, 4) + (F(3, 4) - F(1, 4)) * F(k, 10)
        inner = arccos_rational(v)
        assert a.lo <= inner.lo and inner.hi <= a.hi, (v, a, inner)


def test_a_wider_argument_gives_a_wider_answer_not_a_wrong_one():
    tight = arccos(CInterval.exact(F(1, 2)))
    loose = arccos(CInterval(F(49, 100), F(51, 100)))
    assert loose.width > tight.width
    assert loose.lo <= tight.lo and tight.hi <= loose.hi


# -- the width claim, kept separate from the enclosure claim ------------------

def test_the_bracket_is_narrow_enough_to_certify_a_sign():
    """A bracket is only useful if it can decide something. ADR-0019's rule is
    that a topological decision needs an interval that strictly excludes zero,
    so the test is whether two genuinely different depths come back as two
    intervals a caller can order."""
    a = arccos_rational(F(3, 10))
    b = arccos_rational(F(3001, 10000))
    assert (b - a).sign() == -1, (a, b)


# -- a loose cross-check, at a tolerance a double can honour ------------------

@pytest.mark.parametrize("v", _ratios())
def test_it_agrees_with_the_float_library_to_a_doubles_precision(v):
    a = arccos_rational(v)
    assert abs(float(a.mid) - math.acos(float(v))) < FLOAT_WITNESS, (v, a)


# -- the reason this exists ---------------------------------------------------

def test_the_segment_area_is_computable_at_a_depth_with_no_exact_answer():
    """The measurement that motivated the primitive.

    A flat milled on a bar of radius 5 at depth h removes a circular segment
    of area r^2(t - sin t cos t), t = arccos(h/r) — an ARC ANGLE, so by Niven
    the exact path answers only where t is a rational multiple of pi. h = 1 is
    not such a depth: on a radius-5 bar the exact rungs answer at 3 of 33
    rational depths, and a certified one answers at all of them.
    """
    r, h = F(5), F(1)
    ratio = CInterval.exact(h / r)
    t = arccos(ratio)
    sin_t = (CInterval.exact(F(1)) - ratio * ratio).sqrt()
    area = CInterval.exact(r * r) * (t - ratio * sin_t)

    ft = math.acos(float(h / r))
    expect = float(r * r) * (ft - math.cos(ft) * math.sin(ft))
    assert abs(float(area.mid) - expect) < FLOAT_WITNESS, (area, expect)
    # and it is a genuine certification, not a vacuous one
    # see `_ARCCOS_WIDTH`: the float-proposed bracket floors near 1e-16, so a
    # segment area on r=5 lands near 1e-14. Certified, and ample.
    assert area.width < F(1, 10 ** 10), area.width
