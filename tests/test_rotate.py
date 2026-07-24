"""Exact rotation over ℚ[√d] — the K2.2 circular-pattern / diagonal-sketch-plane
enabler. Rotation is a rigid motion, so volume is preserved EXACTLY (rational),
and representable angles (multiples of 30°/45°) keep coordinates exact."""

from __future__ import annotations

import math

import pytest

from forgekernel import kernel as fk
from forgekernel.surd import SurdVal, sqrt_rational


def test_rotation_preserves_volume_exactly() -> None:
    b = fk.box(10, 4, 3)
    for axis, deg in [((0, 0, 1), 60), ((0, 0, 1), 45), ((1, 0, 0), 30),
                      ((1, 1, 1), 120)]:
        r = fk.rotate(b, axis, deg)
        assert r.volume() == 120           # exact rational, not a float


def test_six_sixty_degree_rotations_are_the_identity() -> None:
    b = fk.box(7, 5, 2)
    acc = b
    for _ in range(6):
        acc = fk.rotate(acc, (0, 0, 1), 60)
    orig = {tuple(float(c) for c in v) for p in b.polys for v in p.verts}
    back = {tuple(float(c) for c in v) for p in acc.polys for v in p.verts}
    assert orig == back                    # 6·60° = 360° back to start, exactly


def test_diagonal_120_is_a_rational_axis_permutation() -> None:
    p = fk.rotate(fk.box(3, 5, 7), (1, 1, 1), 120)
    # 120° about the body diagonal cyclically permutes the axes — the surds
    # cancel, leaving purely rational coordinates.
    assert all(not isinstance(c, SurdVal) or c.b == 0
               for poly in p.polys for v in poly.verts for c in v)
    assert p.volume() == 105


def test_unrepresentable_angle_is_refused() -> None:
    with pytest.raises(ValueError):
        fk.rotate(fk.box(2, 2, 2), (0, 0, 1), 37)


def test_boolean_union_of_rotated_copies_is_watertight_and_exact() -> None:
    # the circular-pattern path: union disjoint rotated copies (surd coords)
    seed = fk.translate(fk.box(5, 2, 3), 10, -1, 0)
    out = seed
    for i in range(1, 6):
        out = fk.boolean("union", out, fk.rotate(seed, (0, 0, 1), 60 * i))
    assert out.volume() == 6 * 5 * 2 * 3   # disjoint -> exact sum, no leaks
    assert out.watertight_violations() == []


def test_overlapping_boolean_mixes_rational_and_surd_coords() -> None:
    # the case disjoint patterns miss: an axis-aligned (ℚ) body booleaned with
    # an OVERLAPPING rotated (ℚ[√3]) body — BSP clipping makes mixed-type
    # polygons, so Fraction−SurdVal must resolve (needs SurdVal.__rsub__).
    from fractions import Fraction as F

    a = fk.box(2, 2, 2)
    b = fk.rotate(fk.translate(fk.box(2, 2, 2), F(1, 2), F(1, 2), 0), (0, 0, 1), 30)
    vu = float(fk.boolean("union", a, b).volume())
    vi = float(fk.boolean("intersect", a, b).volume())
    vc = float(fk.boolean("cut", a, b).volume())
    assert vi > 0                                   # they genuinely overlap
    assert abs(vu + vi - 16) < 1e-9                 # |A∪B| + |A∩B| = |A| + |B|
    assert abs(vc - (8 - vi)) < 1e-9                # |A∖B| = |A| − |A∩B|
    for op in ("union", "cut", "intersect"):
        assert fk.boolean(op, a, b).watertight_violations() == []


def test_surd_is_an_ordered_field() -> None:
    import random
    random.seed(0)
    for _ in range(2000):
        d = random.choice([2, 3, 5, 6])
        from fractions import Fraction as F
        x = SurdVal(F(random.randint(-9, 9), random.randint(1, 4)),
                    F(random.randint(-9, 9), random.randint(1, 4)), d)
        y = SurdVal(F(random.randint(-9, 9), random.randint(1, 4)),
                    F(random.randint(-9, 9), random.randint(1, 4)), d)
        if abs(float(x) - float(y)) > 1e-9:
            assert (x < y) == (float(x) < float(y))
        assert (x / y) * y == x if float(y) != 0 else True
