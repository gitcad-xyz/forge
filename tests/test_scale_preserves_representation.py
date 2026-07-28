"""A UNIFORM scale must not change what kind of thing a shape is.

The third member of the family, after ``mirrored`` and ``rotated``, and the
argument is the same one: a similarity transform maps every one of these
representations to itself. A cylinder scaled ×2 is a cylinder. A frustum is a
frustum. A rounded box is a rounded box, radius and all.

``k.scale(s, 2)`` was sending all of them through the canonical ``Body``, and
because the feature and boolean paths dispatch on representation, ``scale then
cut`` was 8 of the composed grid's 34 "a transformed solid landed in the
canonical B-rep" cells.

NON-uniform scale is genuinely NOT closed — squash a cylinder in x and it is an
elliptic cylinder, which no family here holds — so ``scaled`` returns None for
it and the canonical form takes over. That is the honest answer, not a refusal.

Volume scales by f³ exactly, which is the cheapest possible oracle and needs
nothing but arithmetic.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.quadric import (AxisStack, Cone, Cyl, DisjointUnion,
                                 DrilledSolid, RevolveSolid, RoundedBox,
                                 Sphere)

FACTORS = (2, 3, F(1, 2), F(5, 3), 10)


def _measurable(s):
    if isinstance(s, (Sphere, Cone)):
        return AxisStack(s.cx, s.cy, [s])
    return s


def _vol(s) -> float:
    v = _measurable(s).volume()
    return float(v.to_float() if hasattr(v, "to_float") else float(v))


def _shapes():
    from forgekernel.brep import Solid

    return {
        "cyl": Cyl(F(3), F(-2), F(5), F(1), F(9)),
        "sphere": Sphere(F(2), F(-3), F(4), F(5)),
        "cone": Cone(F(1), F(2), F(6), F(2), F(-1), F(7)),
        "rounded box": RoundedBox(10, 8, 6, 2, origin=(1, 2, 3)),
        "axis stack": AxisStack(F(3), F(-2),
                                [Cyl(F(3), F(-2), F(5), F(0), F(4)),
                                 Cyl(F(3), F(-2), F(2), F(4), F(9))]),
        "revolve": RevolveSolid([(0, 0), (4, 0), (4, 8), (0, 8)], cx=2, cy=-1),
        "drilled": DrilledSolid(Solid.box(20, 20, 10),
                                [Cyl(F(10), F(10), F(3), F(0), F(10))]),
        "disjoint union": DisjointUnion._unchecked(
            [Cyl(F(0), F(0), F(3), F(0), F(2)),
             Cyl(F(40), F(0), F(2), F(0), F(5))]),
    }


@pytest.mark.parametrize("f", FACTORS)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_a_uniform_scale_keeps_the_representation(name, f):
    s = _shapes()[name]
    out = s.scaled(f)
    assert out is not None, f"{name} should absorb a uniform scale"
    assert type(out) is type(s)


@pytest.mark.parametrize("f", FACTORS)
@pytest.mark.parametrize("name", sorted(_shapes()))
def test_volume_scales_by_the_cube(name, f):
    """The oracle is arithmetic: a similarity of ratio f multiplies volume by
    f³, exactly, for every shape there is."""
    s = _shapes()[name]
    assert _vol(s.scaled(f)) == pytest.approx(_vol(s) * float(f) ** 3,
                                              rel=1e-12)


@pytest.mark.parametrize("name", sorted(_shapes()))
def test_scaling_back_is_the_identity(name):
    s = _shapes()[name]
    back = s.scaled(4).scaled(F(1, 4))
    assert type(back) is type(s)
    assert _vol(back) == pytest.approx(_vol(s), rel=1e-12)
    lo0, hi0 = _measurable(s).bbox()
    lo1, hi1 = _measurable(back).bbox()
    assert [float(v) for v in lo0] == pytest.approx([float(v) for v in lo1],
                                                    abs=1e-12)
    assert [float(v) for v in hi0] == pytest.approx([float(v) for v in hi1],
                                                    abs=1e-12)


@pytest.mark.parametrize("name", sorted(_shapes()))
def test_the_bounding_box_scales_about_the_origin(name):
    """A scale that kept the type but did not MOVE anything would pass the
    two tests above on a shape centred at the origin."""
    s = _shapes()[name]
    lo0, hi0 = _measurable(s).bbox()
    lo1, hi1 = _measurable(s.scaled(3)).bbox()
    assert [float(v) for v in lo1] == pytest.approx(
        [float(v) * 3 for v in lo0], abs=1e-12)
    assert [float(v) for v in hi1] == pytest.approx(
        [float(v) * 3 for v in hi0], abs=1e-12)


def test_a_degenerate_scale_is_refused():
    for bad in (0, -1, F(-3, 2)):
        with pytest.raises(ValueError):
            _shapes()["cyl"].scaled(bad)


def test_a_rounded_box_scales_its_radius_too():
    """The trap: a rounded box has FOUR lengths, and forgetting the fillet
    radius yields a box of the right size with the wrong corners."""
    rb = RoundedBox(10, 8, 6, 2, origin=(1, 2, 3))
    out = rb.scaled(3)
    assert (out.a, out.b, out.c, out.r) == (F(30), F(24), F(18), F(6))
    assert out.origin == (F(3), F(6), F(9))


def test_a_drilled_solid_scales_its_bores_too():
    from forgekernel.brep import Solid

    d = DrilledSolid(Solid.box(20, 20, 10),
                     [Cyl(F(10), F(5), F(3), F(0), F(10))])
    out = d.scaled(2)
    b = out.bores[0]
    assert (b.cx, b.cy, b.r, b.z0, b.z1) == (F(20), F(10), F(6), F(0), F(20))
    assert _vol(out) == pytest.approx(_vol(d) * 8, rel=1e-12)
