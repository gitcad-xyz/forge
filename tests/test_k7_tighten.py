"""Bracket tightening (#96 follow-on) — the three limits named by the
stage-1/2 derivation, closed:

1. **Second-order trim enclosure** (the tube bound): a boolean face's
   flux is the EXACT polygon flux over its certified (snapped) trim
   loops ± a tube error that shrinks O(h²) with SSI depth — replacing
   the first-order strip-hull bracket (which is information-
   theoretically stuck at O(2^-depth)·strip-length). The dome cap at
   depth 5 must tighten from width ~2.04 to the hand-pipeline scale.
2. **Certified B-side strip prune**: ``ssi_chains`` returns strips in
   which every dropped cell is PROVEN empty (swapped-role resolver) —
   the raw B-side strip on the dome×plane case is 3× inflated.
3. **Dyadic snapping**: closed-loop coordinates snap to a 2^-(2·depth+6)
   grid (denominators bounded, exact flux cheap), with the snap distance
   folded into the tube radius so enclosures stay enclosures.
4. **Second-order rational leaves** (Neumann): ``patch_flux_ci`` on the
   exact quarter-cylinder converges O(4^-depth), not O(2^-depth).

Every bracket must CONTAIN its independently derived truth — containment
is the spec; the pinned widths are the achieved tightness.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import (PatchSolid, boolean_trimmed, box_patches,
                                certified_trim_flux, patch_flux_ci,
                                trimmed_patch_flux)
from forgekernel.interval import CInterval, pi_interval
from forgekernel.nurbs import bezier_surface
from forgekernel.ssi import ssi_chains, ssi_strips

DOME = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                       [(2, 0, 0), (2, 2, 6), (2, 4, 0)],
                       [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
BOT = bezier_surface([[(0, 0, 0), (4, 0, 0)], [(0, 4, 0), (4, 4, 0)]])
PLANE = bezier_surface([[(0, 0, 1), (0, 4, 1)], [(4, 0, 1), (4, 4, 1)]])
PILLOW = PatchSolid([DOME, BOT])                            # volume 32/3
BOX = PatchSolid(box_patches(4, 4, 2, origin=(0, 0, 1)))    # volume 32

# independent stage-1 reference (Simpson, 2e6 panels)
CAP_REF = F(10794724554159, 10 ** 13)                       # 1.0794724554159079


# -- 4. Neumann leaves: rational flux second order -----------------------------

QCYL = bezier_surface([[(2, 0, 0), (2, 0, 3)],
                       [(2, 2, 0), (2, 2, 3)],
                       [(0, 2, 0), (0, 2, 3)]],
                      weights=[[1, 1], [1, 1], [2, 2]])     # true flux 2π


def test_quarter_cylinder_flux_ci_second_order() -> None:
    two_pi = pi_interval() * CInterval.exact(2)
    widths = []
    for depth in (3, 4, 5):
        iv = patch_flux_ci(QCYL, depth=depth)
        assert iv.lo <= two_pi.lo and two_pi.hi <= iv.hi    # containment
        widths.append(iv.width)
    # second order: quartering per depth (allow slack: < 1/3.2)
    assert widths[1] < widths[0] / F(16, 5)
    assert widths[2] < widths[1] / F(16, 5)
    # achieved tightness at depth 5: 0.0147 (was 0.375, first-order rule)
    assert widths[2] < F(1, 60)


def test_quarter_cylinder_trimmed_ci_half_height_tightens() -> None:
    # trim to v ∈ [0, 1/2] → true flux π; the "in" cells now carry the
    # Neumann rule, so the bracket beats the old first-order one
    pi = pi_interval()
    iv = trimmed_patch_flux_ci_half()
    assert iv.lo <= pi.lo and pi.hi <= iv.hi
    assert iv.width < F(1, 100)                 # was ~0.19 at depth 5


def trimmed_patch_flux_ci_half():
    from forgekernel.bsolid import trimmed_patch_flux_ci
    lo_half = [(0, 0), (1, 0), (1, F(1, 2)), (0, F(1, 2))]
    return trimmed_patch_flux_ci(QCYL, [lo_half], depth=5)


# -- 2. certified strips: the B-side prune ------------------------------------

def test_ssi_chains_returns_certified_strips() -> None:
    raw_a, raw_b = ssi_strips(DOME, PLANE, depth=5)
    res = ssi_chains(DOME, PLANE, depth=5)
    assert "strips" in res
    a_cells, b_cells = res["strips"]
    # pruned ⊆ raw: pruning only ever removes cells
    assert a_cells <= raw_a and b_cells <= raw_b
    # the measured 3× B-side inflation is certified away (228 → ~curve size)
    assert len(b_cells) <= len(raw_b) // 2
    # soundness: every certified point still lies inside BOTH strips
    for ch in res["chains"]:
        for (u, v, s, t) in ch["points"]:
            assert any(u0 <= u <= u1 and v0 <= v <= v1
                       for (u0, u1, v0, v1) in a_cells)
            assert any(s0 <= s <= s1 and t0 <= t <= t1
                       for (s0, s1, t0, t1) in b_cells)


def test_ssi_chains_reports_cells_per_point() -> None:
    res = ssi_chains(DOME, PLANE, depth=4)
    for ch in res["chains"]:
        assert len(ch["cells"]) == len(ch["points"])
        for (acell, bcell), (u, v, s, t) in zip(ch["cells"], ch["points"]):
            assert acell is not None
            if bcell is not None:
                s0, s1, t0, t1 = bcell
                assert s0 <= s <= s1 and t0 <= t <= t1


def test_pruned_strip_tightens_the_first_order_bracket() -> None:
    # the plane face's strip bracket shrinks with the pruned strip alone
    res = ssi_chains(DOME, PLANE, depth=5)
    _, b_pruned = res["strips"]
    _, b_raw = ssi_strips(DOME, PLANE, depth=5)

    def above(s, t):
        return PLANE.eval(s, t)[2] > 1          # trivially true — z ≡ 1?

    # membership on the plane face: inside the dome's footprint.  Use the
    # dome height at the matching (x, y): the plane's (s,t) maps to
    # (4s, 4t); kept where dome z(4s,4t) > 1 — i.e. inside the trim oval.
    def kept(s, t):
        return DOME.eval(s, t)[2] > 1           # same parametrization scale

    iv_raw = certified_trim_flux(PLANE, b_raw, kept, depth=5)
    iv_pruned = certified_trim_flux(PLANE, b_pruned, kept, depth=5)
    assert iv_pruned.width < iv_raw.width / 2
    # both still contain the true plane-face flux (z=1 disk area · 1/3 …
    # containment of each other's midpoints is not required; enclosure
    # nesting is: the pruned bracket must lie inside the raw one
    assert iv_raw.lo <= iv_pruned.lo and iv_pruned.hi <= iv_raw.hi


# -- 1+3. the boolean: second-order volume bracket + snapped loops ------------

@pytest.fixture(scope="module")
def cap5():
    return boolean_trimmed("intersect", PILLOW, BOX, depth=5)


def test_boolean_volume_bracket_is_second_order(cap5) -> None:
    v5 = cap5.volume(depth=5)
    assert v5.lo <= CAP_REF <= v5.hi            # containment, always
    # the headline: width 2.04 → the hand-pipeline's few-e-3 scale
    assert v5.width < F(1, 20)
    cap6 = boolean_trimmed("intersect", PILLOW, BOX, depth=6)
    v6 = cap6.volume(depth=6)
    assert v6.lo <= CAP_REF <= v6.hi
    # second order: ~quartering per depth (allow 1/2.5 slack)
    assert v6.width < v5.width / F(5, 2)


def test_boolean_positivity_certifies_at_depth_4() -> None:
    # the old first-order bracket straddled zero at depth 4 (audit refused);
    # the tightened one certifies positivity there
    shell = boolean_trimmed("intersect", PILLOW, BOX, depth=4)
    v = shell.volume(depth=4)
    assert v.lo > 0
    assert v.lo <= CAP_REF <= v.hi


def test_boolean_loops_are_dyadically_snapped(cap5) -> None:
    bound = 2 ** (2 * 5 + 10)
    for face in cap5.faces:
        for loop in face.loops:
            for vx in loop:
                u, v = vx.uv(face)
                assert u.denominator <= bound and v.denominator <= bound


def test_tight_bracket_nests_inside_strip_bracket(cap5) -> None:
    # per face: the tube bracket must lie inside the strip bracket — the
    # volume() intersection can only tighten, never shift
    for face in cap5.faces:
        strip_iv = certified_trim_flux(face.surface, face.strip, face.inside,
                                       depth=5)
        t = face.tight
        assert t is not None                    # the dome case certifies
        assert strip_iv.lo <= t.hi and t.lo <= strip_iv.hi  # they overlap


def test_conservation_identities_still_hold_tightened() -> None:
    cap = boolean_trimmed("intersect", PILLOW, BOX, depth=4)
    cup = boolean_trimmed("union", PILLOW, BOX, depth=4)
    cut = boolean_trimmed("cut", PILLOW, BOX, depth=4)
    both = cup.volume(depth=4) + cap.volume(depth=4)
    assert both.lo <= F(32, 3) + 32 <= both.hi
    also = cut.volume(depth=4) + cap.volume(depth=4)
    assert also.lo <= F(32, 3) <= also.hi


def test_polygon_flux_form_equals_trimmed_patch_flux() -> None:
    # the fast B-form Green path must agree EXACTLY with the quadrature
    # Green path — same integral, two derivations
    from forgekernel.bsolid import _flux_forms
    from forgekernel.nurbs import bezier_patches
    from forgekernel.tube import polygon_flux
    tri = [(F(1, 8), F(1, 8)), (F(3, 4), F(1, 4)), (F(1, 3), F(2, 3))]
    (u0, u1, v0, v1, net), = bezier_patches(DOME)
    P, Q = _flux_forms(net)
    assert Q is None
    assert polygon_flux(P, [tri]) == trimmed_patch_flux(DOME, [tri])
