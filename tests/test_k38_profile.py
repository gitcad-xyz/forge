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


def test_self_intersecting_profile_is_rejected_clearly() -> None:
    import pytest

    from forgekernel.profile2d import SplinePrism

    # a bow-tie: opposite lobes cancel to zero signed area — must be
    # rejected AS self-intersecting, not mis-diagnosed as "zero area"
    bowtie = [{"kind": "line", "to": [2, 2]}, {"kind": "line", "to": [2, 0]},
              {"kind": "line", "to": [0, 2]}, {"kind": "line", "to": [0, 0]}]
    with pytest.raises(ValueError, match="self-intersect"):
        SplinePrism([0, 0], bowtie, 5)


def test_open_spline_profile_auto_closes() -> None:
    # a profile whose last point != start must be closed for Green's area
    from forgekernel.profile2d import SplinePrism
    open_prof = [{"kind": "line", "to": [10, 0]},
                 {"kind": "spline", "to": [3, 3], "ctrl": [[12, 7], [-2, 7]]}]
    assert SplinePrism([0, 0], open_prof, 5).volume() > 0


def test_splineprism_guards() -> None:
    import pytest

    from forgekernel.profile2d import SplinePrism
    sq = [{"kind": "line", "to": [4, 0]}, {"kind": "line", "to": [4, 4]},
          {"kind": "line", "to": [0, 4]}, {"kind": "line", "to": [0, 0]}]
    with pytest.raises(ValueError, match="height"):
        SplinePrism([0, 0], sq, 0)
    # base_z offset is honored in bbox
    pr = SplinePrism([0, 0], sq, 2, base_z=5)
    (_, _, z0), (_, _, z1) = pr.bbox_f()
    assert (z0, z1) == (5.0, 7.0)


def test_splineprism_centroid_is_exact_polygon_centroid() -> None:
    # right triangle (0,0)-(6,0)-(0,3): area centroid exactly (2,1); z at h/2
    tri = [{"kind": "line", "to": [6, 0]}, {"kind": "line", "to": [0, 3]}]
    pr = SplinePrism([0, 0], tri, 5)
    assert pr.centroid() == (F(2), F(1), F(5, 2))
    assert all(isinstance(c, Fraction) for c in pr.centroid())


def test_splineprism_centroid_beats_bbox_for_asymmetric_bezier() -> None:
    # a curved profile that is NOT vertically centred in its bbox — the true
    # area centroid (y = 51/22) differs sharply from the bbox centre (y=3.5),
    # which is exactly the approximation the old centroid_f returned.
    prof = [{"kind": "line", "to": [10, 0]},
            {"kind": "spline", "to": [0, 0], "ctrl": [[12, 7], [-2, 7]]}]
    pr = SplinePrism([0, 0], prof, 5)
    cx, cy, cz = pr.centroid()
    assert cx == F(5) and cz == F(5, 2)          # x symmetric, z mid-height
    assert cy == F(51, 22)                        # exact area centroid ≈ 2.318
    (x0, y0, _), (x1, y1, _) = pr.bbox_f()
    assert cy != F(y0 + y1) / 2                    # ≠ bbox centre (3.5)
