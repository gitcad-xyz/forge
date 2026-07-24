"""K7 — exact flux volume of a TRIMMED polynomial patch (Green's theorem
over the trim loops), and the boolean-assembly reduction.

The trimmed flux is validated three ways, all exact ℚ:
  * over the full parameter rectangle it equals the independent
    tensor-quadrature ``patch_flux`` — two different exact methods agree;
  * it is additive over a triangulation of the region (tri1 + tri2 == full);
  * summed over the closed, outward-oriented faces of a box it recovers the
    box volume, and that volume is invariant when a face is re-trimmed
    (split by an added edge) — the boolean-assembly invariant.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import (PatchSolid, box_patches, patch_flux,
                                trimmed_patch_flux, trimmed_solid_volume)
from forgekernel.nurbs import bezier_surface

# a curved biquadratic patch offset away from the origin → nonzero flux
_NET = [[(x + 2, y + 3, F((x * y) % 3) + F(x, 2) + 5) for y in range(3)]
        for x in range(3)]
_S = bezier_surface(_NET)
_FULL = [(F(0), F(0)), (F(1), F(0)), (F(1), F(1)), (F(0), F(1))]     # CCW


def test_trimmed_full_rectangle_matches_tensor_patch_flux() -> None:
    pf = patch_flux(_S)
    assert pf != 0                                  # a meaningful (nonzero) case
    assert trimmed_patch_flux(_S, [_FULL]) == pf    # two exact methods agree


def test_trimmed_flux_is_additive_over_a_triangulation() -> None:
    tri1 = [(F(0), F(0)), (F(1), F(0)), (F(1), F(1))]
    tri2 = [(F(0), F(0)), (F(1), F(1)), (F(0), F(1))]
    a, b = trimmed_patch_flux(_S, [tri1]), trimmed_patch_flux(_S, [tri2])
    assert a != 0 and b != 0
    assert a + b == trimmed_patch_flux(_S, [_FULL])   # exact additivity


def test_trimmed_flux_subregion_is_a_proper_fraction() -> None:
    quarter = [(F(0), F(0)), (F(1, 2), F(0)), (F(1, 2), F(1, 2)), (F(0), F(1, 2))]
    whole = abs(trimmed_patch_flux(_S, [_FULL]))
    part = abs(trimmed_patch_flux(_S, [quarter]))
    assert 0 < part < whole


def test_hole_is_subtracted_via_orientation() -> None:
    hole_ccw = [(F(1, 4), F(1, 4)), (F(1, 2), F(1, 4)),
                (F(1, 2), F(1, 2)), (F(1, 4), F(1, 2))]
    hole_cw = list(reversed(hole_ccw))
    with_hole = trimmed_patch_flux(_S, [_FULL, hole_cw])
    assert with_hole == trimmed_patch_flux(_S, [_FULL]) \
        - trimmed_patch_flux(_S, [hole_ccw])


def test_boolean_assembly_box_volume_is_exact() -> None:
    faces = [(p, [_FULL]) for p in box_patches(3, 4, 5)]
    assert trimmed_solid_volume(faces) == 60           # 3·4·5, exact
    assert trimmed_solid_volume(faces) == PatchSolid(box_patches(3, 4, 5)).volume()


def test_volume_invariant_when_a_face_is_re_trimmed() -> None:
    # the boolean-assembly invariant: splitting one boundary face into two
    # trimmed pieces (as a boolean adds an intersection edge) leaves the
    # enclosed volume unchanged — exactly.
    patches = box_patches(3, 4, 5)
    faces = [(p, [_FULL]) for p in patches]
    tri1 = [(F(0), F(0)), (F(1), F(0)), (F(1), F(1))]
    tri2 = [(F(0), F(0)), (F(1), F(1)), (F(0), F(1))]
    refaced = ([(patches[0], [tri1]), (patches[0], [tri2])]
               + [(p, [_FULL]) for p in patches[1:]])
    assert trimmed_solid_volume(refaced) == trimmed_solid_volume(faces) == 60


def test_trimmedpatch_flux_method_matches_the_free_function() -> None:
    from forgekernel.trim import TrimmedPatch
    tp = TrimmedPatch(_S, [_FULL])
    assert tp.flux() == patch_flux(_S)
    # a CW-authored loop is normalized before fluxing → same magnitude
    tp_cw = TrimmedPatch(_S, [list(reversed(_FULL))])
    assert tp_cw.flux() == tp.flux()


def test_trimmed_flux_rejects_rational_patch() -> None:
    from forgekernel.nurbs import BSplineSurface
    wts = [[F(1), F(1)], [F(3, 4), F(3, 4)], [F(1), F(1)]]
    arc = BSplineSurface(2, 1, [[(1, 0, 0), (1, 0, 1)], [(1, 1, 0), (1, 1, 1)],
                                [(0, 1, 0), (0, 1, 1)]],
                         [0, 0, 0, 1, 1, 1], [0, 0, 1, 1], wts)
    with pytest.raises(ValueError, match="polynomial patches only"):
        trimmed_patch_flux(arc, [_FULL])
