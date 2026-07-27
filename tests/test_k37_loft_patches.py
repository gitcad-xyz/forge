"""K3.7/K7 gap 8 — ``LoftSolid.to_patches()``: the exact Bézier skin.

A smooth loft's boundary is already piecewise-polynomial: each wall
(one section-polygon edge × one spline segment) is a ruled surface
between two per-vertex natural-spline curves — a degree (1, 3) Bézier
patch whose control points come straight from the spline coefficients —
and the caps are planar. So the loft has an EXACT patch skin, and the
divergence-theorem volume of that skin must equal the loft's own exact
``V = ∫ A(v) z'(v) dv`` — two independent exact pipelines, one rational
number. That equality is the acceptance oracle for this converter.

The skin is what unlocks the loft × loft boolean: ``to_patches`` also
emits the operand's SEAM TOPOLOGY (which face sides are glued, with
exact 3D control-point equality), which the boolean assembly needs to
pair trim edges that leave a face through its border.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import PatchSolid, patch_flux
from forgekernel.loft import LoftSolid


def _sq(half):
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


HOURGLASS = LoftSolid([(_sq(2), 0), (_sq(1), 3), (_sq(2), 6)])   # 1668/35
BARREL = LoftSolid([(_sq(F(3, 2)), 0), (_sq(F(5, 2)), 2),
                    (_sq(F(3, 2)), 4)])                          # 2582/35


def _skin_flux(solid: PatchSolid):
    return sum((patch_flux(p) for p in solid.patches), F(0))


# -- the acceptance oracle: two exact pipelines, one rational ------------------

def test_skin_volume_equals_the_loft_volume_exactly() -> None:
    for loft in (HOURGLASS, BARREL):
        skin = loft.to_patches()
        assert isinstance(skin, PatchSolid)
        assert _skin_flux(skin) == loft.volume()


def test_barrel_volume_is_the_hand_derived_closed_form() -> None:
    # w(s) = 3/2 + (3/2)s − (1/2)s³ per segment (natural spline through
    # 3/2, 5/2, 3/2), V = 16·∫₀¹(w²)ds per two symmetric segments = 2582/35
    assert BARREL.volume() == F(2582, 35)
    assert _skin_flux(BARREL.to_patches()) == F(2582, 35)


def test_constant_loft_skin_is_the_prism() -> None:
    prism = LoftSolid([(_sq(2), 0), (_sq(2), 3), (_sq(2), 6)])
    assert _skin_flux(prism.to_patches()) == 96


# -- structure: walls degree (1,3), caps planar, outward, single-span ----------

def test_wall_faces_are_degree_1x3_and_caps_close_the_skin() -> None:
    skin = BARREL.to_patches()
    m, n = 4, 3
    walls = skin.patches[:m * (n - 1)]
    caps = skin.patches[m * (n - 1):]
    assert len(walls) == m * (n - 1) and len(caps) == 2
    for w in walls:
        assert (w.p, w.q) == (1, 3)
    for c in caps:
        zs = {pt[2] for row in c.cp for pt in row}
        assert len(zs) == 1                       # planar horizontal cap


def test_skin_passes_the_boolean_operand_preflight() -> None:
    from forgekernel.bsolid import _operand_faces
    faces = _operand_faces("A", BARREL.to_patches())
    assert len(faces) == 10


def test_cw_sections_are_normalized_to_an_outward_skin() -> None:
    cw = LoftSolid([(list(reversed(_sq(2))), 0),
                    (list(reversed(_sq(1))), 3),
                    (list(reversed(_sq(2))), 6)])
    skin = cw.to_patches()
    assert _skin_flux(skin) == F(1668, 35)        # positive: outward


def test_mixed_orientation_sections_refuse_by_name() -> None:
    bad = LoftSolid([(_sq(2), 0), (list(reversed(_sq(1))), 3), (_sq(2), 6)])
    with pytest.raises(ValueError, match="orientation"):
        bad.to_patches()


# -- seam topology: exact, complete, control-point-verified --------------------

def test_seams_cover_every_side_exactly_once() -> None:
    skin = BARREL.to_patches()
    sides = {(fi, s) for fi in range(len(skin.patches))
             for s in ("u0", "u1", "v0", "v1")}
    seen = set()
    for (a, b, flip) in skin.seams:
        assert isinstance(flip, bool)
        for side in (a, b):
            assert side in sides
            assert side not in seen, "side glued twice"
            seen.add(side)
    assert seen == sides                          # closed skin: no free side


def _side_curve(patch, side):
    if side == "u0":
        return [patch.cp[0][j] for j in range(patch.nv)]
    if side == "u1":
        return [patch.cp[-1][j] for j in range(patch.nv)]
    if side == "v0":
        return [patch.cp[i][0] for i in range(patch.nu)]
    return [patch.cp[i][-1] for i in range(patch.nu)]


def test_seam_sides_share_their_3d_curve_control_points_exactly() -> None:
    skin = BARREL.to_patches()
    for ((fi, si), (fj, sj), flip) in skin.seams:
        ci = _side_curve(skin.patches[fi], si)
        cj = _side_curve(skin.patches[fj], sj)
        if flip:
            cj = list(reversed(cj))
        if len(ci) != len(cj):                    # degree-elevate the linear side
            lo, hi = (ci, cj) if len(ci) < len(cj) else (cj, ci)
            assert len(lo) == 2 and len(hi) == 4
            a, b = lo
            lo = [a,
                  tuple(a[c] + (b[c] - a[c]) / 3 for c in range(3)),
                  tuple(a[c] + 2 * (b[c] - a[c]) / 3 for c in range(3)),
                  b]
            ci, cj = lo, hi
        assert ci == cj


# -- non-quad sections: exact skin still, no seams (fan caps) ------------------

def test_pentagon_loft_skin_volume_is_exact_but_carries_no_seams() -> None:
    pent = [(2, 0), (0, 2), (-2, 1), (-2, -1), (0, -2)]
    pent2 = [(1, 0), (0, 1), (-1, F(1, 2)), (-1, F(-1, 2)), (0, -1)]
    loft = LoftSolid([(pent, 0), (pent2, 2), (pent, 4)])
    skin = loft.to_patches()
    assert _skin_flux(skin) == loft.volume()
    assert skin.seams is None
