"""K7 flagship — the loft × loft boolean, end to end.

Two z-fibered solids can NEVER meet in a closed loop on a single face
pair (their walls share the horizontal fibration, so every per-face
intersection piece leaves through a border), so this boolean is the one
that forces the named-unbuilt piece of stage 2: OPEN branches, whose
border crossings must pair with the operand's own seam edges. The
machinery: ``to_patches`` emits exact seam topology, certified border
crossings are CANONICALIZED (one solve per crossing, transferred across
the seam by exact control-point equality), chains glue across the other
operand's seams, each face's domain is partitioned by the resulting
border-anchored paths, and kept faces carry full border loops so the
exact pairing audit closes over seam edges too.

The acceptance case (hand-derivable closed forms):

* A — the "barrel": square sections of half-width 3/2, 5/2, 3/2 at
  z = 0, 2, 4; walls bulge as w(s) = 3/2 + (3/2)s − (1/2)s³ per
  segment; exact volume 2582/35.
* B — a slab loft: [1,4]×[−1,1] quad sections at z = 1 and z = 3;
  exact volume 12.
* A ∩ B = {z∈[1,3], y∈[−1,1], x∈[1, w(z)]}: V = ∫₁³ 2(w−1) dz
  = 2·(153/32) − 4 = **89/16** exactly.

Oracles: certified volume brackets containing the closed forms, the
interval conservation identities, certified-MC membership spot checks,
the principled sense rule, and named refusals when an operand carries
no seam topology.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import (BooleanUnsupported, PatchSolid,
                                boolean_trimmed, untrimmed_shell)
from forgekernel.interval import CInterval
from forgekernel.loft import LoftSolid
from forgekernel.raycast import classify_point_in_shell
from forgekernel.trimshell import TrimmedShell


def _sq(half):
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


BARREL = LoftSolid([(_sq(F(3, 2)), 0), (_sq(F(5, 2)), 2), (_sq(F(3, 2)), 4)])
SLAB = LoftSolid([([(1, -1), (4, -1), (4, 1), (1, 1)], 1),
                  ([(1, -1), (4, -1), (4, 1), (1, 1)], 3)])

A = BARREL.to_patches()          # 10 faces, seams proven
B = SLAB.to_patches()            # 6 faces, seams proven

VOL_A = F(2582, 35)
VOL_B = F(12)
VOL_AB = F(89, 16)               # ∫₁³ 2(w(z)−1) dz, hand-derived


@pytest.fixture(scope="module")
def cap():
    return boolean_trimmed("intersect", A, B, depth=4)


@pytest.fixture(scope="module")
def cup():
    return boolean_trimmed("union", A, B, depth=4)


@pytest.fixture(scope="module")
def cut():
    return boolean_trimmed("cut", A, B, depth=4)


def test_operand_volumes_are_the_closed_forms() -> None:
    assert BARREL.volume() == VOL_A
    assert SLAB.volume() == VOL_B


# -- acceptance: two lofted solids in, one audited certified shell out --------

def test_intersect_is_an_audited_certified_shell(cap) -> None:
    assert isinstance(cap, TrimmedShell)
    assert cap.provenance == "certified"
    report = cap.audit(depth=4)
    assert report["volume"].lo > 0
    for comp in report["vector_area"]:
        assert comp.lo <= 0 <= comp.hi


def test_intersect_volume_bracket_contains_89_16(cap) -> None:
    vol = cap.volume(depth=4)
    assert isinstance(vol, CInterval)
    assert vol.lo <= VOL_AB <= vol.hi
    assert vol.lo > 0


def test_union_volume_bracket_contains_the_closed_form(cup) -> None:
    truth = VOL_A + VOL_B - VOL_AB
    vol = cup.volume(depth=4)
    assert vol.lo <= truth <= vol.hi


def test_cut_volume_bracket_contains_the_closed_form(cut) -> None:
    truth = VOL_A - VOL_AB
    vol = cut.volume(depth=4)
    assert vol.lo <= truth <= vol.hi


# -- the interval conservation identities -------------------------------------

def test_conservation_vol_a_plus_vol_b_in_union_plus_intersect(
        cup, cap) -> None:
    both = cup.volume(depth=4) + cap.volume(depth=4)
    assert both.lo <= VOL_A + VOL_B <= both.hi


def test_conservation_vol_a_in_cut_plus_intersect(cut, cap) -> None:
    both = cut.volume(depth=4) + cap.volume(depth=4)
    assert both.lo <= VOL_A <= both.hi


# -- certified-MC membership spot checks --------------------------------------

def _op_truth(op, in_a, in_b):
    return {"union": in_a or in_b,
            "intersect": in_a and in_b,
            "cut": in_a and not in_b}[op]


# rational points off every surface and trim curve: w(2) = 5/2, w(1) = 35/16
_SAMPLES = ((2, 0, 2),                    # in A (x<5/2), in B
            (0, 0, 2),                    # in A, out of B (x<1)
            (3, 0, 2),                    # out of A (x>5/2), in B
            (2, F(3, 2), 2),              # in A, out of B (y>1)
            (F(7, 2), F(1, 2), 2),        # out of A, in B
            (5, 5, 5))                    # out of both


def test_mc_membership_matches_op_of_certified_operand_memberships(
        cap, cup, cut) -> None:
    shell_a = untrimmed_shell(A)
    shell_b = untrimmed_shell(B)
    for p in _SAMPLES:
        in_a = classify_point_in_shell(shell_a, p) == "in"
        in_b = classify_point_in_shell(shell_b, p) == "in"
        for op, shell in (("intersect", cap), ("union", cup), ("cut", cut)):
            want = "in" if _op_truth(op, in_a, in_b) else "out"
            assert classify_point_in_shell(shell, p) == want, (op, p)


# -- the principled sense rule stays principled --------------------------------

def test_senses_follow_membership_monotonicity(cap, cup, cut) -> None:
    b_surfaces = set(map(id, B.patches))

    def senses(shell):
        return {("B" if id(f.surface) in b_surfaces else "A", f.sense)
                for f in shell.faces}

    assert senses(cap) == {("A", 1), ("B", 1)}
    assert senses(cup) == {("A", 1), ("B", 1)}
    assert senses(cut) == {("A", 1), ("B", -1)}


# -- seam-edge pairing is real: border loops close the audit -------------------

def test_untouched_kept_face_carries_its_border_loop(cap) -> None:
    # B's x=1 wall is entirely inside A: kept whole, and in an open-branch
    # boolean it must still carry a full border loop so its edges pair
    # with the trimmed caps and y-walls
    whole = [f for f in cap.faces if not f.strip]
    assert whole, "expected at least one untouched kept face"
    assert all(f.loops for f in whole)


# -- refusals stay named -------------------------------------------------------

def test_open_branch_without_seam_topology_refuses_by_name() -> None:
    bare = PatchSolid(list(A.patches))            # same skin, seams stripped
    with pytest.raises(BooleanUnsupported, match="open_branch"):
        boolean_trimmed("intersect", bare, B, depth=4)


def test_open_branch_with_one_seamless_operand_refuses_by_name() -> None:
    bare_b = PatchSolid(list(B.patches))
    with pytest.raises(BooleanUnsupported, match="open_branch"):
        boolean_trimmed("intersect", A, bare_b, depth=4)
