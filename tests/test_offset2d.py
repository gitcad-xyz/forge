"""Exact inward offset, with the topology repair that needs.

Offsetting every edge by t and reconnecting corners is right only while nothing
runs out. When a region is thinner than 2t its walls cross and the naive answer
is a polygon turned inside out — which is how a lathe with a 1 mm base, inset
1 mm, produced a void whose floor sat below its ceiling.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.offset2d import OffsetError, inset_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
T = [(0, 0), (9, 0), (9, 3), (6, 3), (6, 9), (3, 9), (3, 3), (0, 3)]
U = [(0, 0), (12, 0), (12, 10), (9, 10), (9, 3), (3, 3), (3, 10), (0, 10)]
TUBE = [(0, 3), (4, 3), (4, 9), (1, 9), (1, 4), (0, 4)]     # 1 mm base


def _area(poly) -> F:
    n = len(poly)
    return abs(sum(F(poly[i][0]) * F(poly[(i + 1) % n][1])
                   - F(poly[(i + 1) % n][0]) * F(poly[i][1])
                   for i in range(n))) / 2


CASES = [
    ("square", SQUARE, 2, 36),
    ("L-bracket", L, 1, 28),
    ("T-section", T, 1, 13),
    ("U-channel", U, 1, 24),
    ("3-4-5 triangle", [(0, 0), (4, 0), (4, 3)], F(1, 2), F(3, 2)),
]


@pytest.mark.parametrize("label,poly,t,area", CASES, ids=[c[0] for c in CASES])
def test_the_inset_is_exact(label, poly, t, area) -> None:
    """Non-convex profiles included: the L, T and U all have reflex corners,
    which is where a naive per-edge offset starts producing nonsense."""
    assert _area(inset_polygon(poly, t)) == area


def test_every_vertex_really_is_t_from_the_original_wall() -> None:
    """The property that defines an erosion, checked directly rather than
    inferred from the construction."""
    from forgekernel.offset2d import _seg_dist2

    for poly in (SQUARE, L, T, U):
        out = inset_polygon(poly, 1)
        for p in out:
            for i in range(len(poly)):
                a = (F(poly[i][0]), F(poly[i][1]))
                b = (F(poly[(i + 1) % len(poly)][0]),
                     F(poly[(i + 1) % len(poly)][1]))
                assert _seg_dist2(a, b, p) >= 1


EATEN = [("exactly consumed", SQUARE, 5), ("over-eaten", SQUARE, 6)]


@pytest.mark.parametrize("label,poly,t", EATEN, ids=[e[0] for e in EATEN])
def test_nothing_left_is_an_answer_not_a_crash(label, poly, t) -> None:
    """A thickness that consumes a CONVEX profile returns the empty polygon.
    The old code raised "the inset collapses", which reads like a failure when
    it is simply the truth."""
    assert inset_polygon(poly, t) == []


COMB = [(0, 0), (14, 0), (14, 8), (12, 8), (12, 3), (9, 3), (9, 8), (7, 8),
        (7, 3), (4, 3), (4, 8), (2, 8), (2, 3), (0, 3)]


def test_empty_is_only_concluded_where_it_is_SOUND_to_conclude_it() -> None:
    """A comb's teeth are 2 mm wide, so at t=1 they vanish and two opposite
    walls meet — but its SPINE is 3 mm tall and survives as a 1 mm strip.
    Concluding "nothing left" from a local collapse would throw that away, and
    a wrong answer is worse than a refusal.

    The erosion of a CONVEX set is convex, so there a collapse really does
    settle it globally. That is the whole distinction."""
    with pytest.raises(OffsetError, match="may survive"):
        inset_polygon(COMB, 1)


def test_two_opposite_walls_meeting_is_refused_by_name() -> None:
    """THE case that motivated this: the tube's base is 1 mm thick, so at t=1
    its floor and ceiling meet without any CORNER collapsing. No adjacent-edge
    event fires, the simulation sails past it, and the clearance check is what
    catches it — validating the answer rather than trusting the bookkeeping."""
    with pytest.raises(OffsetError, match="split event"):
        inset_polygon(TUBE, 1)
    # …and while the base is thick enough, the same profile is fine
    thick = [(0, 3), (4, 3), (4, 9), (1, 9), (1, 6), (0, 6)]
    assert _area(inset_polygon(thick, 1)) > 0


def test_an_irrational_edge_normal_refuses_rather_than_floats() -> None:
    """An edge's offset line is n·x = d + t with n the unit normal, so t enters
    linearly and every event time stays rational — but only while n does
    (ADR-0019)."""
    with pytest.raises(OffsetError, match="irrational unit normal"):
        inset_polygon([(0, 0), (3, 0), (0, 2)], 1)


def test_degenerate_input_refuses() -> None:
    with pytest.raises(OffsetError, match="three distinct"):
        inset_polygon([(0, 0), (1, 1)], 1)
    with pytest.raises(OffsetError, match="zero-area"):
        inset_polygon([(0, 0), (2, 0), (4, 0)], 1)
    with pytest.raises(OffsetError, match="dilation"):
        inset_polygon(SQUARE, -1)


def test_a_zero_inset_is_the_profile_itself() -> None:
    assert _area(inset_polygon(L, 0)) == _area(L)
