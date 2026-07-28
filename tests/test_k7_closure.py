"""Wave-2b — exact branch topology for SSI: resolved-cell-graph clustering
and the certified closed-vs-open decision.

Two defects pinned here, both reproduced by execution against the prior
code before the fix was written (Rule 7):

(b) BRANCH MERGING — branches were connected components of surviving
    cells in A's parameter domain ONLY. Two provably disjoint
    intersection curves that pass within one cell of each other in A's
    domain (a hairpin-folded sheet crossing a plane in two nearby lines)
    were welded into ONE branch — wrong loop count, no warning. The fix
    clusters in the RESOLVED cell graph: two cells are neighbours only
    if they touch in A's domain AND some surviving B-cell of one touches
    some surviving B-cell of the other. A connected curve is covered by
    surviving leaf PAIRS whose 4-dim boxes overlap consecutively, so the
    resolved graph can never split a genuine branch; it separates
    islands the moment EITHER parameter domain resolves the gap.

(a) CLOSED-VS-OPEN — the ``closed`` flag came from a float median-gap
    heuristic over the greedy chain (wrong in both directions: the
    hairpin's two merged OPEN lines reported closed=True at depth 5).
    The fix decides closure from the branch's certified cell strip,
    exactly: closed iff the union of the branch's A-cells encloses a
    hole in A's parameter domain (complement flood-fill from the domain
    border over the exact-ℚ coordinate-compressed grid; foreground
    8-adjacency / background 4-adjacency). ``closed=True`` is now a
    certificate at the stated resolution; a genuinely closed loop whose
    interior hole is finer than the cells reports open (tighten by
    raising depth) rather than being guessed closed.

Ground truths in this file are hand-derived closed forms, never read
back from the code under test.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.nurbs import BSplineSurface
from forgekernel.ssi import ssi_curves, ssi_surfaces

F = Fraction


def _plane_z(c):
    """Plane z = c as a bilinear patch over (u, v) = (x, y) in [0,1]^2."""
    return BSplineSurface(1, 1, [[(0, 0, c), (0, 1, c)], [(1, 0, c), (1, 1, c)]],
                          [0, 0, 1, 1], [0, 0, 1, 1])


# Hairpin-folded sheet, ruled in y: profile (x(s), z(s)) with
#   x = Bezier2[2/5, 17/40, 9/20]  (x stays inside the band [2/5, 9/20])
#   z = Bezier2[-1, 3, -1]         => z(s) = -1 + 8 s(1-s)
# z = 0 at s(1-s) = 1/8, i.e. s = (1 ± sqrt(1/2))/2 ~ 0.14645 / 0.85355.
# TRUE intersection with the plane z=0: TWO disjoint vertical lines
# x = x(s1) ~ 0.4066 and x = x(s2) ~ 0.4434 — 0.037 apart in the plane's
# u = x (sub-cell up to depth 5), but 0.707 apart in the hairpin's s.
_XS = [F(2, 5), F(17, 40), F(9, 20)]
_ZS = [F(-1), F(3), F(-1)]


def _hairpin():
    net = [[(_XS[i], 0, _ZS[i]), (_XS[i], 1, _ZS[i])] for i in range(3)]
    return BSplineSurface(2, 1, net, [0, 0, 0, 1, 1, 1], [0, 0, 1, 1])


# Same fold, z modulated by a bump in t: z = -1 + 9 s(1-s) * 4 t(1-t).
# Max 5/4 at (1/2, 1/2) > 0, corners -1 < 0: the zero set is ONE
# genuinely connected closed loop, squeezed into the same thin x-band.
def _thin_loop():
    ws = [F(0), F(1, 2), F(0)]              # B2 coeffs of s(1-s)
    wt = [F(0), F(2), F(0)]                 # B2 coeffs of 4t(1-t)
    ts = [F(0), F(1, 2), F(1)]
    net = [[(_XS[i], ts[j], F(-1) + 9 * ws[i] * wt[j]) for j in range(3)]
           for i in range(3)]
    return BSplineSurface(2, 2, net, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])


# Round dome loop: z = 6u(1-u) * 4v(1-v), max 3/2 at the centre, cut at
# z=1 — one closed loop around the centre (the standard transversal case).
def _dome():
    P, q = [F(0), F(3), F(0)], [F(0), F(2), F(0)]
    g = [F(0), F(1, 2), F(1)]
    net = [[(g[i], g[j], P[i] * q[j]) for j in range(3)] for i in range(3)]
    return BSplineSurface(2, 2, net, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])


# -- (b) the merge defect: disjoint islands must not be welded ----------------

@pytest.mark.parametrize("depth", [4, 5])
def test_hairpin_two_disjoint_lines_are_two_branches(depth) -> None:
    """TRUTH: z(s) = -1 + 8s(1-s) vanishes at exactly two s values, so the
    hairpin meets z=0 in exactly two disjoint lines. The prior A-domain
    union-find returned ONE branch at depths 4 and 5 (and flagged it
    closed=True at depth 5). The resolved cell graph must report two."""
    r = ssi_curves(_plane_z(0), _hairpin(), depth=depth)
    assert len(r["curves"]) == 2
    # each branch stays on its own fold arm: s entirely below or above 1/2
    arms = []
    for c in r["curves"]:
        ss = [float(p[2]) for p in c["points"]]
        assert max(ss) < 0.5 or min(ss) > 0.5      # never straddles the fold
        arms.append(min(ss) > 0.5)
        # a line across the domain is OPEN — never a closed loop
        assert c["closed"] is False
    assert sorted(arms) == [False, True]           # one branch per arm
    # branch count is consistent across the surface-level entry point
    assert ssi_surfaces(_plane_z(0), _hairpin(), depth=depth)["branches"] == 2


def test_hairpin_branches_resolved_at_depth_6_stay_two() -> None:
    """At depth 6 the A-cells separate on their own; the resolved graph
    must agree with the already-correct answer (regression guard)."""
    r = ssi_curves(_plane_z(0), _hairpin(), depth=6)
    assert len(r["curves"]) == 2
    assert all(c["closed"] is False for c in r["curves"])


# -- the no-split guard: a connected branch must never be divided -------------

@pytest.mark.parametrize("depth", [4, 5, 6])
def test_thin_loop_is_never_split(depth) -> None:
    """TRUTH: z = -1 + 9s(1-s)·4t(1-t) = 0 is one connected closed loop
    (max 5/4 at the centre, negative on every domain edge). Its two long
    sides are ~0.04 apart in the plane's u — sub-cell through depth 5 —
    and the resolved-graph clustering must keep it ONE branch. This is
    the soundness half: separating islands must not split loops."""
    r = ssi_curves(_plane_z(0), _thin_loop(), depth=depth)
    assert len(r["curves"]) == 1


def test_thin_loop_closure_certifies_at_depth_6() -> None:
    """At depth 6 the loop's interior hole is wider than a cell, so the
    strip encloses a hole and closure is certified."""
    r = ssi_curves(_plane_z(0), _thin_loop(), depth=6)
    assert len(r["curves"]) == 1
    assert r["curves"][0]["closed"] is True


# -- (a) closed is a certificate, open is honest ------------------------------

@pytest.mark.parametrize("depth", [3, 4, 5])
def test_dome_loop_closure_is_certified(depth) -> None:
    """The round dome loop's strip is an annulus at every tested depth:
    closure must certify (and did under the old heuristic too — this
    guards the standard case through the semantics change)."""
    r = ssi_curves(_dome(), _plane_z(1), depth=depth)
    assert len(r["curves"]) == 1
    assert r["curves"][0]["closed"] is True


def test_open_line_stays_open() -> None:
    """A single straight intersection line across the domain: no hole,
    closed=False (guards against the hole test ever going trivially True)."""
    net = [[(x, 0, F(-1, 2)), (x, 1, F(1, 2))] for x in range(4)]
    s = BSplineSurface(2, 1, net, [0, 0, 0, 1, 2, 2, 2], [0, 0, 1, 1])
    plane = BSplineSurface(1, 1, [[(0, 0, 0), (0, 1, 0)],
                                  [(2, 0, 0), (2, 1, 0)]], [0, 0, 2, 2],
                           [0, 0, 1, 1])
    r = ssi_curves(plane, s, depth=4)
    assert len(r["curves"]) == 1
    assert r["curves"][0]["closed"] is False


# -- unit tests of the exact hole test ----------------------------------------

def _boxes(spec, w=F(1, 8)):
    """Cells at integer grid positions (i, j) scaled by w."""
    return {(i * w, (i + 1) * w, j * w, (j + 1) * w) for i, j in spec}


def test_strip_hole_annulus_is_closed() -> None:
    from forgekernel.ssi import _branch_is_closed

    ring = [(i, j) for i in range(1, 5) for j in range(1, 5)
            if i in (1, 4) or j in (1, 4)]
    dom = ((0, 1), (0, 1))
    assert _branch_is_closed(_boxes(ring), dom) is True


def test_strip_hole_broken_ring_is_open() -> None:
    from forgekernel.ssi import _branch_is_closed

    ring = [(i, j) for i in range(1, 5) for j in range(1, 5)
            if i in (1, 4) or j in (1, 4)]
    ring.remove((4, 2))                    # one cell missing: flood leaks in
    dom = ((0, 1), (0, 1))
    assert _branch_is_closed(_boxes(ring), dom) is False


def test_strip_hole_straight_and_u_shape_are_open() -> None:
    from forgekernel.ssi import _branch_is_closed

    dom = ((0, 1), (0, 1))
    line = [(i, 3) for i in range(8)]
    assert _branch_is_closed(_boxes(line), dom) is False
    ushape = [(1, j) for j in range(1, 5)] + [(4, j) for j in range(1, 5)] \
        + [(i, 1) for i in (2, 3)]
    assert _branch_is_closed(_boxes(ushape), dom) is False


def test_strip_hole_ring_against_domain_border_is_closed() -> None:
    """A loop pressed against the border still encloses its hole; the
    flood must enter only through genuine gaps, not through the border
    cells themselves."""
    from forgekernel.ssi import _branch_is_closed

    ring = [(i, j) for i in range(0, 4) for j in range(0, 4)
            if i in (0, 3) or j in (0, 3)]
    dom = ((0, 1), (0, 1))
    assert _branch_is_closed(_boxes(ring), dom) is True


def test_strip_hole_mixed_cell_sizes_exact() -> None:
    """Cells of two different widths (two Bézier spans of unequal length)
    must compose exactly — the grid is coordinate-compressed ℚ, never a
    raster."""
    from forgekernel.ssi import _branch_is_closed

    w1, w2 = F(1, 8), F(1, 16)
    left = {(F(1, 8), F(2, 8), F(j, 16), F(j + 1, 16)) for j in range(2, 14)}
    right = {(F(5, 8), F(11, 16), F(j, 16), F(j + 1, 16)) for j in range(2, 14)}
    top = {(F(i, 16), F(i + 1, 16), F(12, 16), F(14, 16)) for i in range(2, 11)}
    bot = {(F(i, 16), F(i + 1, 16), F(2, 16), F(4, 16)) for i in range(2, 11)}
    dom = ((0, 1), (0, 1))
    assert _branch_is_closed(left | right | top | bot, dom) is True
    assert _branch_is_closed(left | top | bot, dom) is False
