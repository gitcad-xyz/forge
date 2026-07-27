"""VariableFilletedBox: the disc-sweep linear-taper fillet (#134).

The semantic is the load-bearing fact: disc-sweep (perpendicular slices are
2D fillet arcs of radius r(t)) is exact in Q[pi]; the rolling-ball envelope
(Parasolid/SolidWorks) is transcendental for every rational nonzero taper
(cos du = b^2, Niven). This file pins the exact volume against the closed
form, the centroid against slice quadrature, and both guards.
"""
from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

from forgekernel.quadric import VariableFilletedBox


def test_volume_matches_the_polynomial_closed_form() -> None:
    vfb = VariableFilletedBox((0, 0, 0), (10, 10, 10),
                              [("z", "min", "min", 1, 2)])
    v = vfb.volume()
    # 1000 - (1 - pi/4) * 10*(1+2+4)/3  =  2930/3 + 35pi/6
    assert (v.a, v.b) == (F(2930, 3), F(35, 6))


def test_centroid_matches_slice_quadrature() -> None:
    """Independent check: integrate the removed region slice by slice.
    Each slice at height t removes the corner square r(t)^2 minus the
    quarter disc, whose area and first moments have elementary forms."""
    r0, r1, L = 1.0, 2.0, 10.0
    n = 200000
    vrem = mx = mz = 0.0
    for i in range(n):
        t = (i + 0.5) * L / n
        r = r0 + (r1 - r0) * t / L
        a = r * r * (1 - math.pi / 4)
        cstar = r ** 3 * (5 / 6 - math.pi / 4) / a
        vrem += a
        mx += a * cstar
        mz += a * t
    h = L / n
    vrem, mx, mz = vrem * h, mx * h, mz * h
    vtot = 1000 - vrem
    ex = (5000 - mx) / vtot
    ez = (5000 - mz) / vtot
    cx, cy, cz = VariableFilletedBox(
        (0, 0, 0), (10, 10, 10), [("z", "min", "min", 1, 2)]).centroid_f()
    assert cx == pytest.approx(ex, abs=1e-9)
    assert cy == pytest.approx(ex, abs=1e-9)
    assert cz == pytest.approx(ez, abs=1e-9)


def test_max_side_edge_centroid_is_mirrored() -> None:
    a = VariableFilletedBox((0, 0, 0), (10, 10, 10),
                            [("z", "min", "min", 1, 2)]).centroid_f()
    b = VariableFilletedBox((0, 0, 0), (10, 10, 10),
                            [("z", "max", "max", 1, 2)]).centroid_f()
    assert b[0] == pytest.approx(10 - a[0], abs=1e-12)
    assert b[1] == pytest.approx(10 - a[1], abs=1e-12)
    assert b[2] == pytest.approx(a[2], abs=1e-12)


def test_adjacent_edges_refuse() -> None:
    with pytest.raises(ValueError, match="corner patch"):
        VariableFilletedBox((0, 0, 0), (10, 10, 10),
                            [("z", "min", "min", 1, 2),
                             ("x", "min", "min", 1, 2)])


def test_overflow_refuses() -> None:
    with pytest.raises(ValueError, match="half-width"):
        VariableFilletedBox((0, 0, 0), (10, 10, 10),
                            [("z", "min", "min", 1, 6)])


def test_mesh_refuses_by_name_not_by_crash() -> None:
    vfb = VariableFilletedBox((0, 0, 0), (10, 10, 10),
                              [("z", "min", "min", 1, 2)])
    with pytest.raises(NotImplementedError, match="oblique-cone"):
        vfb.tessellate()


def test_the_docstring_names_the_semantic() -> None:
    """The disc-sweep vs rolling-ball distinction is the contract; losing it
    from the doc reopens the silent-16%-at-b=0.8 comparison trap."""
    doc = VariableFilletedBox.__doc__
    assert "DISC-SWEEP" in doc or "disc-sweep" in doc
    assert "ROLLING BALL" in doc or "rolling ball" in doc
