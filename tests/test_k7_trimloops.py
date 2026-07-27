"""K7 gaps 2+4 — SSI points become trim loops; trim loops partition a face.

``ssi_trim_loops`` turns certified SSI points into closed trim loops on
BOTH parameter domains: closed branches directly, OPEN branches stitched
to certified domain-border crossings and closed with border segments and
patch corners. A branch that cannot be stitched refuses by name
(``TrimLoopUnstitchable``) — never a guessed loop.

``split_trim_region`` partitions a face's parameter domain by those loops
in exact ℚ (polygon-with-holes per region), with an internal exact
area-conservation audit: Σ region areas == domain area.

Verification targets:
  * the derivation's dome case — dome ∩ half-space above z=1, one CLOSED
    transversal loop, cap volume converging to the midpoint-rule
    reference 1.079472473 (k7_probe5, second-order in depth);
  * an OPEN-branch case — parabolic ridge × plane z=1, two branches that
    cross the patch edges; the stitched loops carry exact border
    endpoints and the assembled volume matches the closed form 16√3/9
    to 1e-6 (the trim curves are straight lines in parameter space, so
    the only error is the certified points' 1e-10 residual);
  * an unstitchable case — a trim curve ending strictly interior to one
    domain refuses by name.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.nurbs import bezier_surface
from forgekernel.ssi import TrimLoopUnstitchable, ssi_trim_loops
from forgekernel.trim import TrimmedPatch, split_trim_region

# the derivation's dome: z = 24 u(1-u) v(1-v) over [0,4]², peak 1.5
DOME = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                       [(2, 0, 0), (2, 2, 6), (2, 4, 0)],
                       [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
PLANE = bezier_surface([[(0, 0, 1), (0, 4, 1)], [(4, 0, 1), (4, 4, 1)]])
# parabolic ridge z = 6u(1-u), constant in v — the OPEN-branch operand:
# the z=1 level lines u = (3±√3)/6 cross the full v extent of the domain
RIDGE = bezier_surface([[(0, 0, 0), (0, 4, 0)],
                        [(2, 0, 3), (2, 4, 3)],
                        [(4, 0, 0), (4, 4, 0)]])

CAP_REF = 1.079472473          # midpoint 2000×2000 reference (k7_probe5)
STRIP_REF = 16 * 3 ** 0.5 / 9  # exact: 16·∫(6u(1-u)-1)du between roots
WALL = 16 * 3 ** 0.5 / 27      # (1/3)·y·Area of the y=4 wall, exact form

UNIT = ((F(0), F(1)), (F(0), F(1)))


def _sq(a, b):
    return [(F(a), F(a)), (F(b), F(a)), (F(b), F(b)), (F(a), F(b))]


def _area(region):
    outer = TrimmedPatch._loop_signed_area(region[0])
    holes = sum(abs(TrimmedPatch._loop_signed_area(h)) for h in region[1:])
    return abs(outer) - holes


def _region_containing(regions, surface, u, v):
    for reg in regions:
        if TrimmedPatch(surface, reg).classify(u, v) == "in":
            return reg
    raise AssertionError(f"no region contains ({u},{v})")


# -- split_trim_region: exact partition ---------------------------------------

def test_split_interior_loop_two_regions() -> None:
    regions = split_trim_region(UNIT, [_sq(F(1, 4), F(3, 4))])
    assert len(regions) == 2
    areas = sorted(_area(r) for r in regions)
    assert areas == [F(1, 4), F(3, 4)]              # exact ℚ
    assert sum(areas) == F(1)
    # the small region is the loop itself; the big one carries it as a hole
    small = min(regions, key=_area)
    big = max(regions, key=_area)
    assert len(small) == 1 and len(big) == 2


def test_split_nested_loops_three_regions() -> None:
    regions = split_trim_region(
        UNIT, [_sq(F(1, 8), F(7, 8)), _sq(F(3, 8), F(5, 8))])
    assert len(regions) == 3
    assert sorted(_area(r) for r in regions) == [F(1, 16), F(7, 16), F(1, 2)]


def test_split_border_touching_loop() -> None:
    # a stitched-style loop sharing edges and corners with the border
    loop = [(F(1, 4), F(0)), (F(1, 4), F(1)), (F(0), F(1)), (F(0), F(0))]
    regions = split_trim_region(UNIT, [loop])
    assert len(regions) == 2
    assert sorted(_area(r) for r in regions) == [F(1, 4), F(3, 4)]


def test_split_two_chains_three_regions() -> None:
    left = [(F(1, 4), F(0)), (F(1, 4), F(1)), (F(0), F(1)), (F(0), F(0))]
    right = [(F(3, 4), F(0)), (F(1), F(0)), (F(1), F(1)), (F(3, 4), F(1))]
    regions = split_trim_region(UNIT, [left, right])
    assert len(regions) == 3
    assert sorted(_area(r) for r in regions) == [F(1, 4), F(1, 4), F(1, 2)]


def test_split_crossing_loops_refuse_by_name() -> None:
    with pytest.raises(ValueError, match="cross"):
        split_trim_region(UNIT, [_sq(F(1, 8), F(5, 8)), _sq(F(3, 8), F(7, 8))])


def test_split_vertex_outside_domain_refuses() -> None:
    with pytest.raises(ValueError, match="outside"):
        split_trim_region(UNIT, [_sq(F(1, 2), F(3, 2))])


def test_split_accepts_surface_object() -> None:
    regions = split_trim_region(DOME, [_sq(F(1, 4), F(3, 4))])
    assert len(regions) == 2


# -- the derivation's dome case: one CLOSED transversal loop ------------------

def test_dome_closed_loop_consistent_between_domains() -> None:
    out = ssi_trim_loops(DOME, PLANE, depth=4)
    assert len(out["loops"]) == 1
    br = out["loops"][0]
    assert br["closed"] is True
    assert len(br["loop_a"]) == len(br["loop_b"]) == len(br["points"])
    # the loops are the SAME certified chain read in each domain
    for (u, v, s, t), (la, lb) in zip(br["points"],
                                      zip(br["loop_a"], br["loop_b"])):
        assert (u, v) == la and (s, t) == lb
    # dome and plane share the parameterization x=4u, y=4v here, so the
    # certified chain must satisfy s≈u, t≈v to the residual scale
    for (u, v, s, t) in br["points"]:
        assert abs(float(u - s)) < 1e-9 and abs(float(v - t)) < 1e-9


def test_dome_cap_volume_converges_to_reference() -> None:
    errs = {}
    for depth in (4, 5):
        out = ssi_trim_loops(DOME, PLANE, depth=depth)
        br = out["loops"][0]
        assert br["closed"]
        rd = split_trim_region(DOME, [br["loop_a"]])
        rp = split_trim_region(PLANE, [br["loop_b"]])
        assert len(rd) == 2 and len(rp) == 2
        cap_d = _region_containing(rd, DOME, F(1, 2), F(1, 2))
        cap_p = _region_containing(rp, PLANE, F(1, 2), F(1, 2))
        fd = TrimmedPatch(DOME, cap_d).flux()
        fp = TrimmedPatch(PLANE, cap_p).flux()
        cands = [abs(a + b) for a in (fd, -fd) for b in (fp, -fp)]
        best = min(cands, key=lambda c: abs(float(c) - CAP_REF))
        errs[depth] = abs(float(best) - CAP_REF)
    assert errs[4] < 2.5e-2                       # measured ~1.0e-2
    assert errs[5] < 6e-3                         # measured ~2.2e-3
    assert errs[5] <= errs[4]                     # second-order refinement


# -- the OPEN-branch case: patch edge crossings -------------------------------

def test_ridge_open_branches_stitch_to_certified_border_points() -> None:
    out = ssi_trim_loops(RIDGE, PLANE, depth=4)
    assert len(out["loops"]) == 2
    for br in out["loops"]:
        assert br["closed"] is False
        chain = br["points"]
        # every open chain ends at a certified border crossing, EXACTLY
        # on the border in both domains (v and t borders here: y=4v=4t)
        for (u, v, s, t) in (chain[0], chain[-1]):
            assert v in (F(0), F(1))
            assert t in (F(0), F(1))
        assert chain[0][1] != chain[-1][1]        # opposite v borders
        # the stitched loops close with border segments + patch corners
        assert len(br["loop_a"]) >= len(chain) + 2
        assert len(br["loop_b"]) >= len(chain) + 2
        corners_a = [p for p in br["loop_a"]
                     if p[0] in (F(0), F(1)) and p[1] in (F(0), F(1))]
        assert len(corners_a) >= 2                # walked past patch corners


def test_ridge_partition_is_exact_in_both_domains() -> None:
    out = ssi_trim_loops(RIDGE, PLANE, depth=4)
    la = [br["loop_a"] for br in out["loops"]]
    lb = [br["loop_b"] for br in out["loops"]]
    ra = split_trim_region(RIDGE, la)
    rb = split_trim_region(PLANE, lb)
    assert len(ra) == 3 and len(rb) == 3
    # exact ℚ area conservation in BOTH parameter domains
    assert sum(_area(r) for r in ra) == F(1)
    assert sum(_area(r) for r in rb) == F(1)


def test_ridge_open_volume_matches_closed_form() -> None:
    out = ssi_trim_loops(RIDGE, PLANE, depth=4)
    ra = split_trim_region(RIDGE, [br["loop_a"] for br in out["loops"]])
    rb = split_trim_region(PLANE, [br["loop_b"] for br in out["loops"]])
    mid_a = _region_containing(ra, RIDGE, F(1, 2), F(1, 2))
    mid_b = _region_containing(rb, PLANE, F(1, 2), F(1, 2))
    fa = TrimmedPatch(RIDGE, mid_a).flux()
    fb = TrimmedPatch(PLANE, mid_b).flux()
    # solid: ridge cap above z=1. Faces: ridge strip + plane strip +
    # wall at y=4 (the y=0 wall has zero flux: S·n = -y = 0 there).
    cands = [abs(sa * float(fa) + sb * float(fb) + sw * WALL)
             for sa in (1, -1) for sb in (1, -1) for sw in (1, -1)]
    best = min(cands, key=lambda c: abs(c - STRIP_REF))
    # trim lines are straight in parameter space: the only error is the
    # certified points' 1e-10 residual, so this is sharp
    assert abs(best - STRIP_REF) < 1e-6


# -- unstitchable: refuse by name ---------------------------------------------

def test_branch_ending_interior_refuses_by_name() -> None:
    # A: plane z=0 over [0,4]²; B: tilted patch z = x-2 over x,y ∈ [1,3].
    # Intersection x=2, y∈[1,3] ends at B's t-borders — strictly interior
    # to A's domain. No trim loop exists on A without B's edge curves
    # (boolean-assembly work, not this stage): refuse by name.
    A = bezier_surface([[(0, 0, 0), (0, 4, 0)], [(4, 0, 0), (4, 4, 0)]])
    B = bezier_surface([[(1, 1, -1), (1, 3, -1)], [(3, 1, 1), (3, 3, 1)]])
    with pytest.raises(TrimLoopUnstitchable):
        ssi_trim_loops(A, B, depth=4)
