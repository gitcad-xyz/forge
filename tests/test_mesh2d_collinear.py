"""keep_collinear must survive float noise (#131).

A face of a solid carries mid-edge vertices that are T-junction seams shared
with a neighbouring face. They are EXACTLY collinear in the b-rep, but the
mesher's 2D projection is float, so by the time earcut sees them they sit
~1e-15 off the line — on either side, depending on the face's basis.

When the noise lands on the convex side, the old ear test accepted the
vertex as a near-zero-area "ear" and emitted a SLIVER triangle whose long
edge is the chord skipping the seam vertex. The neighbouring face (noise on
the other side) kept the vertex and meshed the two halves. Undirected edge
counts then LOOK paired — the sliver launders the T-junction — but the mesh
carries overlapping degenerate triangles and, once the sliver is discarded
(as any downstream consumer of an STL will), the shell is torn: chamfering a
shelled box produced exactly this, 4 bad edges at the cavity corners with
validate().ok=True throughout.

The contract pinned here, for both noise signs and exact zero: no sliver
triangles, every boundary edge used exactly once (so the seam vertex stays
stitched into the triangulation), and the triangle areas sum to the polygon
area.
"""

from __future__ import annotations

import pytest

from forgekernel.mesh2d import triangulate

SQUARE_WITH_SEAM = [(0.0, 0.0), (4.0, 0.0), (10.0, 0.0),
                    (10.0, 10.0), (0.0, 10.0)]


def _audit(outer, holes=()):
    pts, tris = triangulate(outer, list(holes), keep_collinear=True)

    def area(t):
        (ax, ay), (bx, by), (cx, cy) = (pts[t[0]], pts[t[1]], pts[t[2]])
        return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2

    from collections import Counter
    seen = Counter()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            seen[tuple(sorted((a, b)))] += 1
    return pts, tris, area, seen


@pytest.mark.parametrize("noise", [0.0, 1e-15, -1e-15],
                         ids=["exact", "reflex-side", "convex-side"])
def test_a_noisy_collinear_seam_vertex_is_kept_not_eaten(noise) -> None:
    outer = list(SQUARE_WITH_SEAM)
    outer[1] = (4.0, noise)
    pts, tris, area, seen = _audit(outer)
    # no slivers: a near-zero-area triangle is a tear wearing a pairing count
    assert all(area(t) > 1e-9 for t in tris), [t for t in tris if area(t) <= 1e-9]
    # every boundary edge used exactly once — the seam vertex stays stitched
    n = len(outer)
    for i in range(n):
        e = tuple(sorted((i, (i + 1) % n)))
        assert seen.get(e, 0) == 1, f"boundary edge {e} used {seen.get(e, 0)}x"
    # area preserved
    assert sum(area(t) for t in tris) == pytest.approx(100.0, abs=1e-6)


def test_a_seam_vertex_next_to_a_hole_still_pairs() -> None:
    """The hole-bridge path runs the same linked list; the seam contract must
    hold with a hole in play too."""
    outer = [(0.0, 0.0), (5.0, -1e-15), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hole = [(4.0, 4.0), (4.0, 6.0), (6.0, 6.0), (6.0, 4.0)]
    pts, tris, area, seen = _audit(outer, [hole])
    assert all(area(t) > 1e-9 for t in tris)
    for i in range(5):
        e = tuple(sorted((i, (i + 1) % 5)))
        assert seen.get(e, 0) == 1
    assert sum(area(t) for t in tris) == pytest.approx(96.0, abs=1e-6)
