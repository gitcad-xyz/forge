"""K3.7 — smooth multi-section loft, exact rational volume (natural cubic
spline through the sections)."""

from __future__ import annotations

from fractions import Fraction

from forgekernel.loft import LoftSolid, natural_spline_M

F = Fraction


def _sq(half):
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


def test_natural_spline_solve_is_rational() -> None:
    M = natural_spline_M([F(0), F(1), F(0), F(1), F(3)])
    assert all(isinstance(x, Fraction) for x in M)
    assert M[0] == 0 and M[-1] == 0                      # natural end conditions


def test_constant_section_loft_is_the_prism() -> None:
    # a constant square section lofted straight up is just the prism —
    # the smooth spline degenerates to a line, volume exact
    loft = LoftSolid([(_sq(2), 0), (_sq(2), 3), (_sq(2), 6)])
    assert loft.volume() == 16 * 6                        # 96, exact


def test_smooth_hourglass_volume_is_exact_rational() -> None:
    # 4×4 → 2×2 → 4×4: the natural-cubic waist gives an exact ℚ volume
    # distinct from the ruled prismatoid stack (56)
    loft = LoftSolid([(_sq(2), 0), (_sq(1), 3), (_sq(2), 6)])
    v = loft.volume()
    assert isinstance(v, Fraction)
    assert v == F(1668, 35)                              # exact
    assert v != 56                                        # smooth ≠ ruled


def test_loft_requires_equal_section_counts() -> None:
    import pytest

    with pytest.raises(ValueError, match="equal vertex count"):
        LoftSolid([([(0, 0), (1, 0), (1, 1)], 0),
                   ([(0, 0), (1, 0)], 1)])


def _sq_at(half, cx, cy):
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


def test_centroid_is_exact_and_not_the_bbox_centre() -> None:
    # a linear frustum (2 sections) half 2 -> 1 over z 0..3: mass sits toward
    # the wider base, so the TRUE z-centroid (33/28) is below the bbox centre
    # (3/2). The old bbox-centre approximation was simply wrong here.
    fr = LoftSolid([(_sq(2), 0), (_sq(1), 3)])
    assert fr.volume() == 28
    cx, cy, cz = fr.centroid()
    assert (cx, cy) == (0, 0)                    # symmetric in x, y
    assert cz == F(33, 28)                       # exact, hand-verified
    assert cz < F(3, 2)                          # strictly below the bbox centre
    assert all(isinstance(c, Fraction) for c in (cx, cy, cz))


def test_offset_prism_centroid_recovers_the_box_centre() -> None:
    # constant square section at (5,7), z 0..6 -> a box; centroid exact (5,7,3)
    pr = LoftSolid([(_sq_at(2, 5, 7), 0), (_sq_at(2, 5, 7), 3),
                    (_sq_at(2, 5, 7), 6)])
    assert pr.centroid() == (F(5), F(7), F(3))


def test_symmetric_hourglass_centroid_is_the_axis_midpoint() -> None:
    hg = LoftSolid([(_sq(2), 0), (_sq(1), 3), (_sq(2), 6)])
    assert hg.volume() == F(1668, 35)            # volume unchanged by refactor
    assert hg.centroid() == (F(0), F(0), F(3))   # symmetry
    assert hg.centroid_f() == (0.0, 0.0, 3.0)    # float view agrees
