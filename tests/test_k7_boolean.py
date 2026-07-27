"""K7 gap 5 — ``boolean_trimmed``: two patch solids in, one audited shell out.

The assembly under test: per face-pair SSI → certified trim loops →
exact domain partition → certified point-membership classification of
every strip-free fragment → stitch + orient into a
:class:`~forgekernel.trimshell.TrimmedShell` → full three-oracle audit.

**Orientation is principled, not enumerated.** Every operand face is
outward-oriented for its own solid (checked: Σ patch flux > 0), and the
result's outward normal follows from membership monotonicity alone:
union and intersection are monotone *increasing* in both operands, so a
kept operand face keeps its own outward normal (sense +1); a cut
``A − B = A ∩ ¬B`` is monotone *decreasing* in B, so kept B-faces flip
(sense −1). No sign search — the derivation that had to try all four
sign combinations by hand is retired.

Acceptance (the mission's stated oracles):

* dome ∩ half-space and dome ∪ half-space end to end — as the pillow
  solid (dome + flat bottom, exact volume 32/3) against a box
  [0,4]²×[1,3] (exact volume 32), whose intersection is exactly the
  stage-1 cap (independent Simpson reference 1.0794724554159079);
* the conservation identity vol(A)+vol(B) ∈ vol(A∪B)+vol(A∩B) *in
  intervals* (and vol(A) ∈ vol(A−B)+vol(A∩B));
* MC membership spot-checks with certified signs: for sample points,
  ``in(result) == op(in(A), in(B))`` where every membership is a
  certified ray-parity verdict, never a float guess.

What stage 2 refuses BY NAME, a wrong shell being the only failure:
tangent contact (no transversal certificate), open/stitched branches
(their border segments would need pairing with the operand's own seam
edges — unbuilt), rational and multi-span operands (untested), operands
that are not outward-oriented, and an empty result (no shell to build).
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import (BooleanUnsupported, PatchSolid,
                                boolean_trimmed, box_patches, patch_flux,
                                untrimmed_shell)
from forgekernel.interval import CInterval
from forgekernel.nurbs import BSplineSurface, bezier_surface
from forgekernel.raycast import classify_point_in_shell
from forgekernel.ssi import SsiCellUncertified, TrimLoopUnstitchable
from forgekernel.trimshell import TrimmedShell

# the stage-1 dome: z = 24 u(1-u) v(1-v) over the [0,4]² footprint, peak 3/2
DOME = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                       [(2, 0, 0), (2, 2, 6), (2, 4, 0)],
                       [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
# flat bottom, outward -z: closes the dome into the "pillow" solid
BOT = bezier_surface([[(0, 0, 0), (4, 0, 0)], [(0, 4, 0), (4, 4, 0)]])

PILLOW = PatchSolid([DOME, BOT])                 # exact volume 32/3
PILLOW_VOL = F(32, 3)
BOX = PatchSolid(box_patches(4, 4, 2, origin=(0, 0, 1)))   # [0,4]²×[1,3]
BOX_VOL = F(32)

# independent stage-1 reference for the cap volume (Simpson, 2·10^6 panels)
CAP_REF = F(10794724554159, 10 ** 13)            # 1.0794724554159079


@pytest.fixture(scope="module")
def cap():
    """pillow ∩ box — exactly the stage-1 dome∩half-space cap."""
    return boolean_trimmed("intersect", PILLOW, BOX, depth=5)


@pytest.fixture(scope="module")
def cup():
    """pillow ∪ box."""
    return boolean_trimmed("union", PILLOW, BOX, depth=4)


@pytest.fixture(scope="module")
def cut():
    """pillow − box: the pillow with its cap sliced off."""
    return boolean_trimmed("cut", PILLOW, BOX, depth=4)


# -- operand sanity (the numbers every oracle below leans on) ------------------

def test_operand_volumes_are_the_hand_derived_exact_values() -> None:
    assert sum(patch_flux(p) for p in PILLOW.patches) == PILLOW_VOL
    assert sum(patch_flux(p) for p in BOX.patches) == BOX_VOL


# -- acceptance 1: dome ∩ half-space, end to end ------------------------------

def test_intersect_is_an_audited_certified_shell(cap) -> None:
    assert isinstance(cap, TrimmedShell)
    assert cap.provenance == "certified"
    assert len(cap.faces) == 2               # dome cap region + plane disk
    report = cap.audit(depth=5)
    assert report["volume"].lo > 0
    for comp in report["vector_area"]:
        assert comp.lo <= 0 <= comp.hi


def test_intersect_volume_bracket_contains_the_stage1_reference(cap) -> None:
    vol = cap.volume(depth=5)
    assert isinstance(vol, CInterval)
    assert vol.lo <= CAP_REF <= vol.hi
    assert vol.lo > 0


# -- acceptance 2: dome ∪ half-space, end to end ------------------------------

def test_union_is_an_audited_shell_with_the_untrimmed_faces_kept(cup) -> None:
    assert isinstance(cup, TrimmedShell)
    # dome (trimmed) + pillow bottom + box bottom (trimmed) + 4 sides + top
    assert len(cup.faces) == 8


def test_union_volume_bracket_contains_the_exact_truth(cup, cap) -> None:
    # truth: vol(A) + vol(B) − vol(A∩B); CAP_REF is accurate to ~1e-13,
    # far inside any bracket width here
    truth = PILLOW_VOL + BOX_VOL - CAP_REF
    vol = cup.volume(depth=4)
    assert vol.lo <= truth <= vol.hi


# -- acceptance 3: conservation identities, in intervals ----------------------

def test_conservation_vol_a_plus_vol_b_in_union_plus_intersect(cup, cap) -> None:
    both = cup.volume(depth=4) + cap.volume(depth=5)
    assert both.lo <= PILLOW_VOL + BOX_VOL <= both.hi


def test_conservation_vol_a_in_cut_plus_intersect(cut, cap) -> None:
    both = cut.volume(depth=4) + cap.volume(depth=5)
    assert both.lo <= PILLOW_VOL <= both.hi


# -- acceptance 4: MC membership spot-check, certified signs ------------------

def _op_truth(op, in_a, in_b):
    return {"union": in_a or in_b,
            "intersect": in_a and in_b,
            "cut": in_a and not in_b}[op]


# generic rational points, chosen off every surface and trim curve
_SAMPLES = ((2, 2, F(5, 4)),        # in pillow, in box  (inside the cap)
            (2, 2, F(1, 2)),        # in pillow, out of box
            (F(1, 2), F(1, 2), F(9, 8)),   # out of pillow, in box
            (2, 2, 4),              # out of both
            (3, 1, F(9, 8)))        # near the trim curve's cylinder, in box


def test_mc_membership_matches_op_of_certified_operand_memberships(
        cap, cup, cut) -> None:
    shell_a = untrimmed_shell(PILLOW)
    shell_b = untrimmed_shell(BOX)
    for p in _SAMPLES:
        in_a = classify_point_in_shell(shell_a, p) == "in"
        in_b = classify_point_in_shell(shell_b, p) == "in"
        for op, shell in (("intersect", cap), ("union", cup), ("cut", cut)):
            want = "in" if _op_truth(op, in_a, in_b) else "out"
            assert classify_point_in_shell(shell, p) == want, (op, p)


# -- the principled orientation rule ------------------------------------------

def test_senses_follow_membership_monotonicity_not_a_sign_search(
        cap, cup, cut) -> None:
    box_surfaces = set(map(id, BOX.patches))

    def senses(shell):
        return {("B" if id(f.surface) in box_surfaces else "A", f.sense)
                for f in shell.faces}

    assert senses(cap) == {("A", 1), ("B", 1)}       # intersect: keep outward
    assert senses(cup) == {("A", 1), ("B", 1)}       # union: keep outward
    assert senses(cut) == {("A", 1), ("B", -1)}      # cut: B flips (A ∩ ¬B)


def test_cut_of_a_disjoint_tool_is_exactly_the_pillow() -> None:
    far = PatchSolid(box_patches(1, 1, 1, origin=(9, 9, 9)))
    shell = boolean_trimmed("cut", PILLOW, far, depth=2)
    vol = shell.volume(depth=2)
    # nothing is trimmed, so the certified bracket collapses to exact
    assert vol.lo == vol.hi == PILLOW_VOL


# -- refusals by name: a wrong shell is the only failure ----------------------

def test_tangent_contact_refuses_never_a_wrong_shell() -> None:
    # box bottom at z = 3/2 touches the dome exactly at its peak (2,2,3/2)
    graze = PatchSolid(box_patches(4, 4, 1, origin=(0, 0, F(3, 2))))
    with pytest.raises((SsiCellUncertified, TrimLoopUnstitchable,
                        BooleanUnsupported)):
        boolean_trimmed("intersect", PILLOW, graze, depth=3)


def test_open_branch_refuses_by_name() -> None:
    # a half-width box: the z=1 circle leaves the tool's bottom face
    # through its border — open branches on both meeting faces
    half = PatchSolid(box_patches(2, 4, 2, origin=(0, 0, 1)))
    with pytest.raises((BooleanUnsupported, TrimLoopUnstitchable)) as e:
        boolean_trimmed("intersect", PILLOW, half, depth=4)
    assert ("open" in str(e.value)) or ("stitch" in str(e.value))


def test_rational_operand_refuses_by_name() -> None:
    quad = bezier_surface([[(0, 0, 0), (0, 2, 0)], [(2, 0, 0), (2, 2, 2)]],
                          weights=[[1, 1], [1, 2]])
    with pytest.raises(BooleanUnsupported, match="rational_operand"):
        boolean_trimmed("union", PatchSolid([quad]), BOX, depth=2)


def test_multispan_operand_refuses_by_name() -> None:
    two_span = BSplineSurface(
        1, 1,
        [[(0, 0, 0), (0, 2, 0)], [(1, 0, 1), (1, 2, 1)], [(2, 0, 0), (2, 2, 0)]],
        [0, 0, F(1, 2), 1, 1], [0, 0, 1, 1])
    with pytest.raises(BooleanUnsupported, match="multispan_operand"):
        boolean_trimmed("union", PatchSolid([two_span]), BOX, depth=2)


def test_inward_oriented_operand_refuses_by_name() -> None:
    # u/v transposed nets flip S_u×S_v: the same pillow, inside out
    rdome = bezier_surface([[(0, 0, 0), (2, 0, 0), (4, 0, 0)],
                            [(0, 2, 0), (2, 2, 6), (4, 2, 0)],
                            [(0, 4, 0), (2, 4, 0), (4, 4, 0)]])
    rbot = bezier_surface([[(0, 0, 0), (0, 4, 0)], [(4, 0, 0), (4, 4, 0)]])
    with pytest.raises(BooleanUnsupported, match="operand_not_outward"):
        boolean_trimmed("union", PatchSolid([rdome, rbot]), BOX, depth=2)


def test_empty_intersection_refuses_rather_than_fabricating_a_shell() -> None:
    far = PatchSolid(box_patches(1, 1, 1, origin=(9, 9, 9)))
    with pytest.raises(BooleanUnsupported, match="empty_result"):
        boolean_trimmed("intersect", PILLOW, far, depth=2)


def test_unknown_op_refuses() -> None:
    with pytest.raises(ValueError, match="union|intersect|cut"):
        boolean_trimmed("xor", PILLOW, BOX)
