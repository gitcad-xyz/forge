"""A plane cutting a sphere — the second rung of the quadric boolean (K2.x).

Easier than the flat on a bar and exact in strictly more places. A plane meets
a sphere in a CIRCLE, always — there is no oblique case to classify — and the
removed cap's volume carries no trigonometry:

    cap of height a on radius r:  V = pi a^2 (3r - a) / 3

so every RATIONAL cut height is exact, with none of the twelfth constraint the
flat needed. That difference is the point of doing the two rungs separately:
they fail for different reasons and it would be easy to assume the flat's
restriction is a property of "quadric booleans" in general. It is not; it is a
property of arcs.

The kernel already had the representation. ``SphereS.pole`` exists precisely
because a one-rim spherical face is ambiguous between the cap and the
pole-containing remainder, so this rung is mostly a matter of asking for it.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

from forgekernel import body as B
from forgekernel.quadric import Sphere
from forgekernel.sphercut import SphereCutRefused, cut_sphere_at_z

R = F(5)


def _ball():
    return Sphere(F(0), F(0), F(0), R)


def _closed_form(zc) -> float:
    """Kept volume when everything above z = zc is removed."""
    a = float(R) - float(zc)                      # the cap's height
    return (4 / 3 * math.pi * float(R) ** 3
            - math.pi * a * a * (3 * float(R) - a) / 3)


#: rational heights, deliberately including thirds — a NON-twelfth, to pin
#: that this rung carries no such restriction
HEIGHTS = [F(4), F(5, 2), F(7, 3), F(0), F(-5, 2), F(-4), F(-9, 2)]


def test_the_closed_form_is_right_about_the_case_anyone_can_check():
    """Validate the oracle first: cutting through the centre leaves half."""
    assert _closed_form(F(0)) == pytest.approx(
        4 / 3 * math.pi * float(R) ** 3 / 2, abs=1e-9)


@pytest.mark.parametrize("zc", HEIGHTS, ids=str)
def test_the_volume_matches_the_closed_form(zc):
    assert float(B.volume(cut_sphere_at_z(_ball(), zc))) == pytest.approx(
        _closed_form(zc), abs=1e-9)


@pytest.mark.parametrize("zc", HEIGHTS, ids=str)
def test_the_shell_is_closed(zc):
    out = cut_sphere_at_z(_ball(), zc)
    assert len(out.faces) == 2                # the ball's remainder, and a disc
    assert B.manifold_violations(out) == []


def test_a_non_twelfth_height_is_fine_here():
    """The flat on a bar refuses off a twelfth because its area carries an arc
    angle. A cap carries none, so 7/3 is exact — and asserting that stops
    anyone 'fixing' this rung by copying the flat's restriction across."""
    v = float(B.volume(cut_sphere_at_z(_ball(), F(7, 3))))
    assert v == pytest.approx(_closed_form(F(7, 3)), abs=1e-9)


def test_deeper_cuts_leave_less():
    vs = [float(B.volume(cut_sphere_at_z(_ball(), z))) for z in HEIGHTS]
    assert vs == sorted(vs, reverse=True), vs


def test_keeping_the_other_side_is_the_complement():
    """The two halves of a cut must sum to the whole ball, exactly. This is
    the check that catches a pole flag set the wrong way — each side would
    still look plausible alone."""
    for zc in (F(5, 2), F(0), F(-5, 2)):
        lo = float(B.volume(cut_sphere_at_z(_ball(), zc, keep_below=True)))
        hi = float(B.volume(cut_sphere_at_z(_ball(), zc, keep_below=False)))
        assert lo + hi == pytest.approx(4 / 3 * math.pi * float(R) ** 3,
                                        abs=1e-9), zc


def test_an_offset_sphere_cuts_the_same_way():
    """The cut height is relative to the CENTRE, so a translated ball must
    give a translated answer — a term reading absolute z would pass every
    test above on a centred ball."""
    off = Sphere(F(3), F(-2), F(7), R)
    v = float(B.volume(cut_sphere_at_z(off, F(7) + F(5, 2))))
    assert v == pytest.approx(_closed_form(F(5, 2)), abs=1e-9)


@pytest.mark.parametrize("zc", [R, -R, F(6), F(-11, 2)])
def test_a_plane_that_misses_refuses(zc):
    with pytest.raises(SphereCutRefused, match="does not cut"):
        cut_sphere_at_z(_ball(), zc)


def test_it_refuses_a_shape_that_is_not_a_sphere():
    from forgekernel.quadric import Cyl

    with pytest.raises(SphereCutRefused, match="wants a Sphere"):
        cut_sphere_at_z(Cyl(F(0), F(0), F(5), F(0), F(10)), F(0))


# -- the centroid, which is what held this rung back (#143) --------------------

def test_a_hemisphere_centroid_is_the_textbook_value():
    """3r/8 from the flat face. A whole number anyone can look up, and the
    reason to check it FIRST: `body.volume` handled a pole-trimmed cap all
    along and `body.centroid` did not, so this rung computed an exact volume
    and then refused its centre of mass."""
    c = B.centroid(cut_sphere_at_z(_ball(), F(0)))
    assert float(c[2]) == pytest.approx(-3 * float(R) / 8, abs=1e-12)
    assert float(c[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(c[1]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("zc", HEIGHTS, ids=str)
def test_the_centroid_lies_inside_the_solid(zc):
    c = B.centroid(cut_sphere_at_z(_ball(), zc))
    x, y, z = (float(v) for v in c)
    assert x * x + y * y + z * z < float(R) ** 2, "inside the ball"
    assert z < float(zc), "below the cut"


def test_an_off_centre_cap_centroid_translates_with_the_sphere():
    """The check a centred ball cannot make: every c_k term in the moment
    integral is invisible at the origin, so a sign error there passes each
    centred case and fails only once the sphere moves."""
    cx, cy, cz = F(3), F(-2), F(7)
    off = Sphere(cx, cy, cz, R)
    a = B.centroid(cut_sphere_at_z(off, cz + F(5, 2)))
    b = B.centroid(cut_sphere_at_z(_ball(), F(5, 2)))
    for i, off_i in enumerate((cx, cy, cz)):
        assert float(a[i]) == pytest.approx(float(b[i]) + float(off_i),
                                            abs=1e-9)


def test_a_whole_sphere_centroid_is_untouched():
    """The branch this shares. A whole sphere's centre of mass is its centre,
    off-origin included, and widening the sphere case must not move it."""
    whole = B.to_body(Sphere(F(3), F(-2), F(7), R))
    assert [float(v) for v in B.centroid(whole)] == pytest.approx(
        [3.0, -2.0, 7.0], abs=1e-9)
