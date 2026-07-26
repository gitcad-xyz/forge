"""FilletedChamferedBox — chamfer(box, d) with every edge blended at radius r.

The volume lives in ℚ(√2,√3)[π] (biquadratic — BiSurd's first consumer). The
closed form asserted here was derived and verified INDEPENDENTLY of this code
(backlog §1.2), for box 20³, d = 2, r = 1:

    fillet(all, r) on a convex solid is the opening P₋ᵣ ⊕ B_r. The eroded
    polytope P₋ᵣ is the chamfered box of dims L−2r with setback
    t' = d − (2−√2)r, and Steiner gives

    V(P₋₁) = 5616 + 12√2          (8a³ − 12at'² + 6t'³, a = 9, t' = √2)
    S(P₋₁) = 2424 − 468√2         (6 squares (18−2√2)² + 12 hexagons 36−3√2)
    edges  = (54 − 6√2)π + 2√6·π  (24 × (18−2√2) × π/4, 24 × (√6/2) × π/3)
    corners= 4π/3                 (convex: 32 spherical patches sum to a ball)

    V = 8040 − 456√2 + (166/3 − 6√2 + 2√6)π = 7557.6867093895…

Verified against qhull (V, S, edge lengths to ≤ 1e-12) and by Monte-Carlo
membership dist(q, P₋₁) ≤ 1: seed 20260726 40M → 7557.99 ± 0.29 (z +1.05),
seed 987654321 60M → 7557.74 ± 0.24 (z +0.23); fresh seed 1357911 (this
session, before wiring) — recorded in the gitcad golden test.

The corner patches are individually NOT in any ℚ[√d][π] (their solid angles
carry arccos(1/3) and arccos(1/√3)); they cancel only in aggregate to 4π.
The implementation must therefore use the Steiner/opening decomposition and
never per-corner patches — asserted here by exactness itself: a per-patch
accumulation cannot produce these coefficients.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr

import pytest

from forgekernel.bisurd import BiSurd
from forgekernel.polypi import PiPoly
from forgekernel.quadric import FilletedChamferedBox


def bs(a=0, b=0, c=0, e=0):
    return BiSurd(a, b, c, e, 2, 3)


def test_probe_cell_closed_form_exact():
    f = FilletedChamferedBox((0, 0, 0), (20, 20, 20), 2, 1)
    v = f.volume()
    assert v == PiPoly([bs(8040, -456), bs(Fr(166, 3), -6, 0, 2)])
    assert math.isclose(float(v), 7557.6867093895, rel_tol=1e-12)


def test_result_fills_the_original_bbox_and_is_centred():
    f = FilletedChamferedBox((1, 2, 3), (20, 20, 20), 2, 1)
    lo, hi = f.bbox()
    assert tuple(map(float, lo)) == (1.0, 2.0, 3.0)
    assert tuple(map(float, hi)) == (21.0, 22.0, 23.0)
    # centrally symmetric solid: centroid is the box centre, exactly
    assert f.centroid_f() == (11.0, 12.0, 13.0)


def test_non_cube_box_volume_matches_independent_float_assembly():
    A, B, C, d, r = 30.0, 20.0, 10.0, 2.0, 1.0
    f = FilletedChamferedBox((0, 0, 0), (30, 20, 10), 2, 1)
    # independent float Steiner assembly (written from the derivation, not
    # from the class): eroded dims and setback
    s2, s3, s6 = math.sqrt(2), math.sqrt(3), math.sqrt(6)
    Ae, Be, Ce, t = A - 2 * r, B - 2 * r, C - 2 * r, d - (2 - s2) * r
    v0 = Ae * Be * Ce - 2 * t * t * (Ae + Be + Ce) + 6 * t ** 3
    s0 = (2 * ((Ae - 2 * t) * (Be - 2 * t) + (Be - 2 * t) * (Ce - 2 * t)
               + (Ae - 2 * t) * (Ce - 2 * t))
          + 4 * s2 * t * (Ae + Be + Ce) - 18 * s2 * t * t)
    edges = (math.pi / 4) * (r * r / 2) * 8 * (Ae + Be + Ce - 6 * t) \
        + (math.pi / 3) * (r * r / 2) * 24 * (s3 / 2) * t
    want = v0 + s0 * r + edges + 4 * math.pi * r ** 3 / 3
    assert math.isclose(float(f.volume()), want, rel_tol=1e-12)


def test_guard_ball_bridges_the_chamfer():
    # d ≤ (2−√2)r: the eroded polytope keeps no chamfer facet; blends from
    # the three edges at a corner collide — refuse by name, never guess
    with pytest.raises(ValueError, match="bridges the chamfer"):
        FilletedChamferedBox((0, 0, 0), (20, 20, 20), Fr(1, 2), 1)


def test_guard_blends_meet_across_a_thin_face():
    # C − 2d − (2√2−2)r must stay positive: C = 4.5, d = 2, r = 1 gives
    # 4.5 − 4 = 0.5 < 2√2 − 2 = 0.8284…, an exact-irrational comparison a
    # float epsilon could not be trusted with
    with pytest.raises(ValueError, match="across the"):
        FilletedChamferedBox((0, 0, 0), (20, 20, Fr(9, 2)), 2, 1)


def test_guard_degenerate_inputs():
    with pytest.raises(ValueError):
        FilletedChamferedBox((0, 0, 0), (20, 20, 20), 2, 0)
    with pytest.raises(ValueError):
        FilletedChamferedBox((0, 0, 0), (20, 20, 20), 0, 1)
    with pytest.raises(ValueError):
        FilletedChamferedBox((0, 0, 0), (20, 20, 8), 5, 1)   # 2d ≥ min dim
