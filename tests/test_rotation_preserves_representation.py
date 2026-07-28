"""A rotation a family ABSORBS must not change what kind of thing a shape is.

The mirror argument with a different matrix, and it lands on a bigger gap. A
z-axis cylinder turned about z is still a z-axis cylinder — only (cx, cy) move.
A sphere absorbs EVERY rotation, being round. A frustum turned 180° about x is
a frustum standing on its other end, radii swapped.

None of that was used: ``RefKernel.transform`` sent every rotated quadric
through the canonical ``Body``, and since the feature and boolean paths
dispatch on representation, "a transformed solid lands in the canonical B-rep
and boolean has no path over it" became one of the largest composed-tier gap
families. 36 (shape, op) pairs work upright and stop working once rotated.

``rotated`` returns None — not an error — when the family does NOT absorb the
rotation (a genuine tilt), because falling through to the canonical form is
the right answer there, not a refusal.

Everything here is exact: the rotation matrix is already ℚ[√d] for multiples of
30° and 45°, and absorbing it costs one 2×2 product on the axis position.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.quadric import (AxisStack, Cone, Cyl, DisjointUnion,
                                 DrilledSolid, RevolveSolid, RoundedBox,
                                 Sphere)

Z = (0, 0, 1)
X = (1, 0, 0)
#: every angle the exact rotation table holds
ANGLES = (30, 45, 60, 90, 120, 135, 150, 180, 270, 330)


def _measurable(s):
    if isinstance(s, (Sphere, Cone)):
        return AxisStack(s.cx, s.cy, [s])
    return s


def _vol(s) -> float:
    v = _measurable(s).volume()
    return float(v.to_float() if hasattr(v, "to_float") else float(v))


def _shapes():
    return {
        "cyl": Cyl(F(3), F(-2), F(5), F(1), F(9)),
        "sphere": Sphere(F(2), F(-3), F(4), F(5)),
        "cone": Cone(F(1), F(2), F(6), F(2), F(-1), F(7)),
        "axis stack": AxisStack(F(3), F(-2),
                                [Cyl(F(3), F(-2), F(5), F(0), F(4)),
                                 Cyl(F(3), F(-2), F(2), F(4), F(9))]),
        "revolve": RevolveSolid([(0, 0), (4, 0), (4, 8), (0, 8)], cx=2, cy=-1),
    }


@pytest.mark.parametrize("deg", ANGLES)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_a_z_rotation_is_absorbed(name, deg):
    """Every family here has a +z axis, so turning about z only moves the
    axis position — the shape is unchanged in kind."""
    s = _shapes()[name]
    out = s.rotated(Z, deg)
    assert out is not None, f"{name} should absorb a z-rotation"
    assert type(out) is type(s)
    assert _vol(out) == pytest.approx(_vol(s), rel=1e-12)


@pytest.mark.parametrize("deg", ANGLES)
def test_a_sphere_absorbs_every_rotation(deg):
    """Being round, about any axis at all."""
    s = _shapes()["sphere"]
    for axis in (Z, X, (0, 1, 0), (1, 1, 0), (1, 2, 3)):
        out = s.rotated(axis, deg)
        assert out is not None and type(out) is Sphere
        assert out.r == s.r
        assert _vol(out) == pytest.approx(_vol(s), rel=1e-12)


@pytest.mark.parametrize("name", sorted(_shapes()))
def test_a_full_turn_is_the_identity(name):
    """360° must return the ORIGINAL, field for field — the check that catches
    a matrix applied to the wrong pair of coordinates."""
    s = _shapes()[name]
    out = s.rotated(Z, 90).rotated(Z, 90).rotated(Z, 90).rotated(Z, 90)
    assert type(out) is type(s)
    assert _vol(out) == pytest.approx(_vol(s), rel=1e-12)
    lo0, hi0 = _measurable(s).bbox()
    lo1, hi1 = _measurable(out).bbox()
    assert [float(v) for v in lo0] == pytest.approx([float(v) for v in lo1],
                                                    abs=1e-12)
    assert [float(v) for v in hi0] == pytest.approx([float(v) for v in hi1],
                                                    abs=1e-12)


@pytest.mark.parametrize("name", sorted(_shapes()))
def test_the_axis_actually_moves(name):
    """A no-op that kept the type would pass every test above. A 90° turn
    about z sends (cx, cy) to (−cy, cx)."""
    s = _shapes()[name]
    out = s.rotated(Z, 90)
    assert (out.cx, out.cy) == (-s.cy, s.cx)


def test_a_tilt_declines_rather_than_lying():
    """A cylinder turned about x is NOT a z-axis cylinder any more. None means
    "use the canonical form" — falling through is right, refusing is not."""
    c = _shapes()["cyl"]
    assert c.rotated(X, 90) is None
    assert c.rotated((1, 1, 0), 45) is None
    assert _shapes()["cone"].rotated((0, 1, 0), 30) is None


def test_a_half_turn_about_x_stands_a_frustum_on_its_other_end():
    """180° about x flips the +z axis to −z, which Cyl and Cone can express:
    the same argument as `mirrored`, and the radii travel with their ends."""
    c = Cone(F(0), F(0), F(6), F(2), F(0), F(10))
    out = c.rotated(X, 180)
    assert out is not None and type(out) is Cone
    assert (out.r1, out.r2) == (F(2), F(6))
    assert (out.z0, out.z1) == (F(-10), F(0))
    assert _vol(out) == pytest.approx(_vol(c), rel=1e-12)


def test_an_unrepresentable_angle_declines():
    """15° is not in the exact table, so there is no exact rotation to
    absorb — and inventing a float one is what the charter forbids."""
    assert _shapes()["cyl"].rotated(Z, 15) is None


def test_a_composite_rotates_member_by_member():
    box_cyl = DisjointUnion._unchecked(
        [Cyl(F(0), F(0), F(3), F(0), F(2)), Cyl(F(40), F(0), F(2), F(0), F(5))])
    out = box_cyl.rotated(Z, 90)
    assert out is not None and type(out) is DisjointUnion
    assert [(m.cx, m.cy) for m in out.members] == [(F(0), F(0)), (F(0), F(40))]


def test_a_drilled_solid_carries_its_bores_round():
    from forgekernel.brep import Solid

    d = DrilledSolid(Solid.box(20, 20, 10),
                     [Cyl(F(10), F(5), F(3), F(0), F(10))])
    out = d.rotated(Z, 90)
    assert out is not None and type(out) is DrilledSolid
    assert (out.bores[0].cx, out.bores[0].cy) == (F(-5), F(10))
    assert _vol(out) == pytest.approx(_vol(d), rel=1e-12)


def test_a_rounded_box_takes_quarter_turns_only():
    """Its extents are axis-aligned, so 90° permutes them exactly and 45°
    genuinely leaves the family."""
    rb = RoundedBox(10, 8, 6, 2, origin=(1, 2, 3))
    out = rb.rotated(Z, 90)
    assert out is not None and type(out) is RoundedBox
    assert (out.a, out.b, out.c) == (F(8), F(10), F(6))
    assert _vol(out) == pytest.approx(_vol(rb), rel=1e-12)
    assert rb.rotated(Z, 45) is None
