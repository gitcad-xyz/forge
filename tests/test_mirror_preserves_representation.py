"""A reflection must not change WHAT KIND OF THING a shape is.

Every quadric family here is closed under an axis reflection — a mirrored
cylinder is a cylinder, a mirrored frustum is a frustum with its radii swapped
— so a reflection that hands back the generic canonical ``Body`` instead has
thrown away structure for nothing.

That loss is not cosmetic. The feature paths (chamfer, fillet, shell) dispatch
on representation, so ``mirror`` downgrading a ``Cyl`` to a ``Body`` made every
one of them refuse a shape they handle perfectly well upright. It is the whole
of gitcad's 34-pair mirror-asymmetry list: 30 of the 34 are a curved or
composite representation that survives the operation one way up and cannot even
be offered it reflected.

The invariant behind it: for an isometry σ, ``op(σS) ≡ σ op(S)``. A kernel
that answers one and refuses the other is not wrong about geometry, it is
wrong about itself.

``mirrored`` is exact and structural — negate a coordinate, swap an interval's
ends — so nothing here costs arithmetic.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.brep import Solid
from forgekernel.quadric import (AxisStack, Cone, Cyl, DisjointUnion,
                                 DrilledSolid, RevolveSolid, RoundedBox,
                                 Sphere)

AXES = ("x", "y", "z")


def _measurable(s):
    """A bare Sphere or Cone carries no volume()/bbox() of its own — the
    kernel measures them wrapped in a one-member AxisStack, and so does this."""
    if isinstance(s, (Sphere, Cone)):
        return AxisStack(s.cx, s.cy, [s])
    return s


def _vol(s) -> float:
    v = _measurable(s).volume()
    return float(v.to_float() if hasattr(v, "to_float") else float(v))


def _bbox_f(s):
    lo, hi = _measurable(s).bbox()
    return ([float(x) for x in lo], [float(x) for x in hi])


def _shapes():
    box = Solid.box(10, 6, 4).translated((F(1), F(2), F(3)))
    cyl = Cyl(F(3), F(-2), F(5), F(1), F(9))
    return {
        "cyl": cyl,
        "sphere": Sphere(F(2), F(-3), F(4), F(5)),
        "cone": Cone(F(1), F(2), F(6), F(2), F(-1), F(7)),
        "rounded box": RoundedBox(10, 8, 6, 2, origin=(1, 2, 3)),
        "drilled": DrilledSolid(Solid.box(20, 20, 10),
                                [Cyl(F(10), F(10), F(3), F(0), F(10))]),
        "axis stack": AxisStack(F(3), F(-2),
                                [Cyl(F(3), F(-2), F(5), F(0), F(4)),
                                 Cyl(F(3), F(-2), F(2), F(4), F(9))]),
        "disjoint union": DisjointUnion._unchecked(
            [box, Cyl(F(40), F(0), F(2), F(0), F(5))]),
        "revolve": RevolveSolid([(0, 0), (4, 0), (4, 8), (0, 8)], cx=2, cy=-1),
    }


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_a_mirror_returns_the_same_representation(name, axis):
    s = _shapes()[name]
    out = s.mirrored(axis)
    assert type(out) is type(s), (
        f"mirror {axis} turned a {type(s).__name__} into a "
        f"{type(out).__name__}")


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_a_mirror_preserves_volume(name, axis):
    """A reflection is an isometry. Any volume change is a wrong answer."""
    s = _shapes()[name]
    assert _vol(s.mirrored(axis)) == pytest.approx(_vol(s), rel=1e-12)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_a_mirror_reflects_the_bounding_box(name, axis):
    """The shape must actually MOVE the way a reflection moves it — a no-op
    that keeps the type would pass the two tests above and be nonsense."""
    i = AXES.index(axis)
    lo, hi = _bbox_f(_shapes()[name])
    mlo, mhi = _bbox_f(_shapes()[name].mirrored(axis))
    for j in range(3):
        if j == i:
            assert mlo[j] == pytest.approx(-hi[j], abs=1e-12)
            assert mhi[j] == pytest.approx(-lo[j], abs=1e-12)
        else:
            assert mlo[j] == pytest.approx(lo[j], abs=1e-12)
            assert mhi[j] == pytest.approx(hi[j], abs=1e-12)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_mirroring_twice_is_the_identity(name, axis):
    """σ² = 1, so the round trip must return the ORIGINAL, field for field.
    This is what catches a sign convention applied to the wrong end of an
    interval — a frustum whose radii swap once and not back."""
    s = _shapes()[name]
    back = s.mirrored(axis).mirrored(axis)
    assert type(back) is type(s)
    assert _vol(back) == pytest.approx(_vol(s), rel=1e-12)
    assert _bbox_f(back) == pytest.approx(_bbox_f(s), abs=1e-12)


def test_a_frustum_swaps_its_radii_about_the_xy_plane():
    """The one case where "negate a coordinate" is not enough: r1 lives at z0,
    so reflecting z has to carry it to the other end."""
    c = Cone(F(0), F(0), F(6), F(2), F(0), F(10))
    m = c.mirrored("z")
    assert (m.r1, m.r2) == (F(2), F(6))
    assert (m.z0, m.z1) == (F(-10), F(0))
    # and NOT about x, where the radii keep their ends
    mx = c.mirrored("x")
    assert (mx.r1, mx.r2) == (F(6), F(2))
    assert (mx.z0, mx.z1) == (F(0), F(10))


def test_a_bad_axis_is_refused():
    with pytest.raises(ValueError):
        Cyl(F(0), F(0), F(1), F(0), F(1)).mirrored("w")
