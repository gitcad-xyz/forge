"""K7 — trimmed-patch topology + exact point-in-region classification.

The classification is a topological predicate decided entirely in ℚ:
in / on / out with no tolerance (ADR-0019 — no float decides which side of
a boundary a point is on). Ground-truth cases: squares, a square-with-hole,
an L-shaped concave region, and a diamond whose vertices sit exactly on the
test ray (the half-open convention must not double-count them).
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.nurbs import BSplineSurface
from forgekernel.trim import TrimmedPatch


def _plane(side=10):
    """A flat BSplineSurface over [0,side] x [0,side] (z = 0)."""
    return BSplineSurface(1, 1, [[(0, 0, 0), (0, side, 0)],
                                 [(side, 0, 0), (side, side, 0)]],
                          [0, 0, side, side], [0, 0, side, side])


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]           # CCW


def test_square_containment_in_on_out() -> None:
    tp = TrimmedPatch(_plane(), [SQUARE])
    assert tp.classify(5, 5) == "in"
    assert tp.classify(15, 5) == "out"
    assert tp.classify(-1, -1) == "out"
    assert tp.classify(5, 0) == "on"       # on the bottom edge
    assert tp.classify(0, 0) == "on"       # exactly a vertex
    assert tp.contains(5, 5) and not tp.contains(5, 0)


def test_square_with_hole() -> None:
    hole = [(3, 3), (3, 7), (7, 7), (7, 3)]             # CW (a hole)
    tp = TrimmedPatch(_plane(), [SQUARE, hole])
    assert tp.classify(5, 5) == "out"      # inside the hole -> not material
    assert tp.classify(2, 5) == "in"       # in the material ring
    assert tp.classify(5, 3) == "on"       # on the hole boundary
    # even-odd is orientation-independent: a CCW hole classifies identically
    hole_ccw = list(reversed(hole))
    assert TrimmedPatch(_plane(), [SQUARE, hole_ccw]).classify(5, 5) == "out"


def test_exact_parameter_area() -> None:
    hole = [(3, 3), (3, 7), (7, 7), (7, 3)]
    tp = TrimmedPatch(_plane(), [SQUARE, hole])
    assert tp.area() == F(84)              # 100 - 16, exact
    assert tp.signed_area() == F(84)       # outer CCW (+100) + hole CW (-16)
    assert isinstance(tp.area(), F)


def test_normalized_fixes_winding_without_changing_measure() -> None:
    outer_cw = list(reversed(SQUARE))                    # wrong: outer CW
    hole_ccw = [(3, 3), (7, 3), (7, 7), (3, 7)]          # wrong: hole CCW
    tp = TrimmedPatch(_plane(), [outer_cw, hole_ccw]).normalized()
    assert tp.is_ccw(0) is True                          # outer now CCW
    assert tp.is_ccw(1) is False                         # hole now CW
    assert tp.signed_area() == F(84)
    assert tp.area() == F(84)


def test_concave_L_region() -> None:
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    tp = TrimmedPatch(_plane(), [L])
    assert tp.classify(7, 7) == "out"      # the notch
    assert tp.classify(2, 7) == "in"       # vertical arm
    assert tp.classify(7, 2) == "in"       # horizontal arm


def test_ray_through_vertices_not_double_counted() -> None:
    diamond = [(5, 0), (10, 5), (5, 10), (0, 5)]         # vertices at v=5
    tp = TrimmedPatch(_plane(), [diamond])
    assert tp.classify(5, 5) == "in"       # centre
    assert tp.classify(-1, 5) == "out"     # ray grazes two vertices at v=5
    assert tp.classify(0, 5) == "on"       # exactly the left vertex


def test_validate_catches_bad_topology() -> None:
    outer = [(0, 0), (5, 0), (5, 5), (0, 5)]
    hole_outside = [(6, 6), (6, 8), (8, 8), (8, 6)]      # in domain, not in outer
    with pytest.raises(ValueError, match="not inside the outer"):
        TrimmedPatch(_plane(), [outer, hole_outside]).validate()
    with pytest.raises(ValueError, match="outside surface domain"):
        TrimmedPatch(_plane(), [[(0, 0), (11, 0), (11, 5), (0, 5)]]).validate()
    with pytest.raises(ValueError, match=">= 3 vertices"):
        TrimmedPatch(_plane(), [[(0, 0), (1, 1)]])
