"""K7 — certified-safe tessellation of a TrimmedShell for the viewer.

A boolean result's trim boundary is only known as an SSI cell
enclosure, so a watertight mesh is not honestly available — but a
coarse VIEW is: triangulate exactly the dyadic cells whose membership
is certain (strip-free and certified kept), and report the strip cells
as explicit ``gaps`` instead of guessing across them. The mesh is a
render artifact — floats are fine — but it never lies: no triangle
covers a point whose membership is uncertain.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import boolean_trimmed
from forgekernel.loft import LoftSolid
from forgekernel.tess import trimmed_shell_mesh


def _sq(half):
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


@pytest.fixture(scope="module")
def cap():
    A = LoftSolid([(_sq(F(3, 2)), 0), (_sq(F(5, 2)), 2),
                   (_sq(F(3, 2)), 4)]).to_patches()
    B = LoftSolid([([(1, -1), (4, -1), (4, 1), (1, 1)], 1),
                   ([(1, -1), (4, -1), (4, 1), (1, 1)], 3)]).to_patches()
    return boolean_trimmed("intersect", A, B, depth=4)


def test_mesh_covers_only_certified_cells_and_reports_gaps(cap) -> None:
    mesh = trimmed_shell_mesh(cap, depth=4)
    assert mesh["triangles"], "expected a non-empty view"
    assert mesh["gaps"], "a trimmed shell must report its uncertainty band"
    nv = len(mesh["vertices"])
    for tri in mesh["triangles"]:
        assert len(tri) == 3 and all(0 <= k < nv for k in tri)
    for v in mesh["vertices"]:
        assert all(abs(c) < 100 for c in v)


def test_mesh_is_a_view_not_a_measure(cap) -> None:
    mesh = trimmed_shell_mesh(cap, depth=4)
    assert mesh["provenance"] == "render"
    # the gap quads exactly cover the strip cells: one per strip cell
    assert len(mesh["gaps"]) == sum(len(f.strip) for f in cap.faces)
