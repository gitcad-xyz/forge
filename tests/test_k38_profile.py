"""K3.8 — spline sketch profiles: exact area via Green's theorem."""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.profile2d import SplinePrism, exact_area

F = Fraction


def test_polygon_area_exact() -> None:
    sq = [{"kind": "line", "to": [10, 0]}, {"kind": "line", "to": [10, 10]},
          {"kind": "line", "to": [0, 10]}, {"kind": "line", "to": [0, 0]}]
    assert exact_area([0, 0], sq) == 100
    tri = [{"kind": "line", "to": [4, 0]}, {"kind": "line", "to": [0, 3]},
           {"kind": "line", "to": [0, 0]}]
    assert exact_area([0, 0], tri) == 6


def test_spline_profile_area_and_prism_exact() -> None:
    d = [{"kind": "line", "to": [10, 0]},
         {"kind": "spline", "to": [0, 0], "ctrl": [[12, 7], [-2, 7]]}]
    a = exact_area([0, 0], d)
    assert isinstance(a, Fraction)
    assert a == F(231, 5)                         # exact via Green's theorem
    prism = SplinePrism([0, 0], d, 5)
    assert prism.volume() == 231                  # A·h exact


def test_spline_prism_rejects_arc_segments() -> None:
    with pytest.raises(ValueError, match="arc"):
        exact_area([0, 0], [{"kind": "line", "to": [4, 0]},
                            {"kind": "arc", "to": [0, 0], "via": [2, 2]}])
