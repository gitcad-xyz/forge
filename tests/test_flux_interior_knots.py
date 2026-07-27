"""``patch_flux`` / ``trimmed_patch_flux`` on interior-knot B-splines.

The exact quadrature in both functions assumed the integrand
F = S·(S_u×S_v) is ONE polynomial of degree (3p−1, 3q−1) over the whole
domain. On a B-spline surface with interior knots F is only PIECEWISE
polynomial — the global rule integrates the wrong polynomial and
returns a silently wrong exact-looking Fraction (ADR-0019's worst
case: a wrong number wearing an exact costume). Every imported
``B_SPLINE_SURFACE_WITH_KNOTS`` has interior knots, so this was
reachable today via ``TrimmedPatch.flux``.

The fix splits exactly at the knots (Boehm insertion via
``bezier_patches`` for the full-domain flux; per-span segmentation of
the antiderivative and per-edge knot-crossing splits for the trimmed
contour integral) — still exact ℚ, and bitwise identical on
single-span (Bézier) input where the old rule was already exact.

Hand-derived oracle — the "tent": p=q=1, U=[0,0,1/2,1,1], V=[0,0,1,1],
x-fiber (0,1,2), z-fiber (0,1,0), y ∈ [0,2].  In global (u,v):

* u ≤ 1/2:  S=(2u,2v,2u)   → S·(S_u×S_v) = 0
* u ≥ 1/2:  S=(2u,2v,2−2u) → S·(S_u×S_v) = 8

Flux = (1/3)·(0·(1/2) + 8·(1/2)) = **4/3**; over u∈[1/4,3/4]: **2/3**;
over the triangle (0,0),(1,0),(1,1) (region v<u): 8·(3/8)/3 = **1**.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.bsolid import patch_flux, trimmed_patch_flux
from forgekernel.nurbs import BSplineSurface, bezier_patches, bezier_surface


def _tent():
    cp = [[(F(0), F(0), F(0)), (F(0), F(2), F(0))],
          [(F(1), F(0), F(1)), (F(1), F(2), F(1))],
          [(F(2), F(0), F(0)), (F(2), F(2), F(0))]]
    return BSplineSurface(1, 1, cp,
                          [F(0), F(0), F(1, 2), F(1), F(1)],
                          [F(0), F(0), F(1), F(1)])


def _tent_v():
    """The same tent with the interior knot in v instead of u."""
    cp = [[(F(0), F(0), F(0)), (F(1), F(0), F(1)), (F(2), F(0), F(0))],
          [(F(0), F(2), F(0)), (F(1), F(2), F(1)), (F(2), F(2), F(0))]]
    return BSplineSurface(1, 1, cp,
                          [F(0), F(0), F(1), F(1)],
                          [F(0), F(0), F(1, 2), F(1), F(1)])


def test_patch_flux_interior_knot_matches_hand_integral():
    assert patch_flux(_tent()) == F(4, 3)


def test_patch_flux_interior_knot_agrees_with_per_bezier_sum():
    tent = _tent()
    per_bezier = sum((patch_flux(bezier_surface(net))
                      for *_, net in bezier_patches(tent)), F(0))
    assert per_bezier == F(4, 3)            # the independent oracle itself
    assert patch_flux(tent) == per_bezier


def test_patch_flux_interior_knot_in_v_direction():
    # u↔v swap flips the normal: same magnitude, opposite sign.
    assert patch_flux(_tent_v()) == -F(4, 3)


def test_trimmed_flux_full_domain_equals_patch_flux():
    tent = _tent()
    full = [[(F(0), F(0)), (F(1), F(0)), (F(1), F(1)), (F(0), F(1))]]
    assert trimmed_patch_flux(tent, full) == F(4, 3)
    assert trimmed_patch_flux(tent, full) == patch_flux(tent)


def test_trimmed_flux_rectangle_straddling_the_knot():
    tent = _tent()
    rect = [[(F(1, 4), F(0)), (F(3, 4), F(0)),
             (F(3, 4), F(1)), (F(1, 4), F(1))]]
    assert trimmed_patch_flux(tent, rect) == F(2, 3)


def test_trimmed_flux_diagonal_edge_crossing_the_knot_line():
    tent = _tent()
    tri = [[(F(0), F(0)), (F(1), F(0)), (F(1), F(1))]]
    assert trimmed_patch_flux(tent, tri) == F(1)


def test_trimmed_flux_v_knot_with_diagonal_edge():
    tent_v = _tent_v()
    tri = [[(F(0), F(0)), (F(1), F(0)), (F(1), F(1))]]
    # region v<u of the transposed tent: by the u↔v swap this is the
    # mirror region u<v of the original ⇒ −(4/3 − 1) = −1/3.
    assert trimmed_patch_flux(tent_v, tri) == -F(1, 3)


def test_single_span_bezier_results_are_unchanged():
    """The fix must be invisible on single-span input — the old rule was
    already exact there (strictly-stronger replacement, ADR/rule 7)."""
    net = [[(F(0), F(0), F(3)), (F(0), F(2), F(3))],
           [(F(2), F(0), F(3)), (F(2), F(2), F(3))]]
    ramp = bezier_surface(net)              # flat z=3 plate, S·(S_u×S_v)=12
    assert patch_flux(ramp) == F(4)
    full = [[(F(0), F(0)), (F(1), F(0)), (F(1), F(1)), (F(0), F(1))]]
    assert trimmed_patch_flux(ramp, full) == F(4)
