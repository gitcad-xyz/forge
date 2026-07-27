"""K7.1 + K7.0c — the certified flux tier (ADR-0019).

Two capabilities that were refusals/silent approximations before:

1. ``patch_flux_ci`` / ``trimmed_patch_flux_ci`` — flux of a RATIONAL
   patch as a real interval bracket (CInterval), where the exact path
   refuses by name (K7.1). Validated on the exact quarter-cylinder
   rational patch (x²+y²==4 exactly at rational parameters), whose true
   flux is known in closed form: π·r²·h/6.

2. ``certified_trim_flux`` — the trimmed flux of a boolean face bracketed
   through the surviving SSI subdivision cells (the cells provably
   enclose the true trim curve), so the boolean volume comes out as
   "certified ± e", never a bare float. Validated on the derivation's
   dome ∩ half-space case against an independent high-accuracy reference.

Every bracket here must CONTAIN an independently derived truth — the
brackets are the product; containment is the spec.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.interval import CInterval, ci_reciprocal, pi_interval
from forgekernel.nurbs import bezier_surface
from forgekernel.bsolid import (patch_flux, patch_flux_ci,
                                trimmed_patch_flux_ci, certified_trim_flux)
from forgekernel.ssi import ssi_strips

F = Fraction


# -- ci_reciprocal moved to interval.py (was nurbs._ci_reciprocal) ------------

def test_ci_reciprocal_lives_in_interval() -> None:
    iv = ci_reciprocal(CInterval(2, 4))
    assert iv.lo == F(1, 4) and iv.hi == F(1, 2)
    neg = ci_reciprocal(CInterval(-4, -2))
    assert neg.lo == F(-1, 2) and neg.hi == F(-1, 4)
    with pytest.raises(ValueError):
        ci_reciprocal(CInterval(-1, 1))                 # straddles zero


def test_nurbs_still_uses_the_shared_reciprocal() -> None:
    # the curve eval_ci path (irrational-weight circles) must keep working
    from forgekernel.nurbs import bezier
    w = CInterval(F(70, 99), F(71, 100))                # ~ √2/2 bracket
    arc = bezier([(1, 0, 0), (1, 1, 0), (0, 1, 0)], weights=[F(1), w, F(1)])
    x, y, z = arc.eval_ci(F(1, 2))
    # the point stays a certified enclosure (nonempty, finite intervals)
    assert x.lo <= x.hi and y.lo <= y.hi and z.lo == z.hi == 0


# -- the exact quarter-cylinder rational patch --------------------------------

R, H = 2, 3


def quarter_cylinder() -> "bezier_surface":
    """Quarter cylinder wall, radius 2 about the z-axis, height 3.
    Rational quadratic in u (weights 1,1,2 — the exact-ℚ quarter circle:
    x(u)=r(1−u²)/(1+u²), y(u)=2ru/(1+u²), so x²+y²==r² exactly), linear
    in v (z from 0 to 3). True flux = (1/3)·h·r²·(π/2) = 2π."""
    net = [[(R, 0, 0), (R, 0, H)],
           [(R, R, 0), (R, R, H)],
           [(0, R, 0), (0, R, H)]]
    return bezier_surface(net, weights=[[1, 1], [1, 1], [2, 2]])


def test_quarter_cylinder_is_exactly_on_the_cylinder() -> None:
    cyl = quarter_cylinder()
    for t in (F(1, 3), F(1, 7), F(2, 5)):
        x, y, _ = cyl.eval(t, F(1, 2))
        assert x * x + y * y == R * R


def test_patch_flux_still_refuses_rational_by_name() -> None:
    with pytest.raises(ValueError, match="K7.1"):
        patch_flux(quarter_cylinder())


def test_patch_flux_ci_refuses_polynomial_exact_stays_default() -> None:
    # ADR-0019: never silently downgrade an exact answer to an interval
    poly = bezier_surface([[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 1)]])
    with pytest.raises(ValueError, match="patch_flux"):
        patch_flux_ci(poly)


def test_patch_flux_ci_brackets_two_pi() -> None:
    cyl = quarter_cylinder()
    two_pi = pi_interval() * CInterval.exact(2)
    widths = []
    for depth in (3, 4, 5):
        iv = patch_flux_ci(cyl, depth=depth)
        assert isinstance(iv, CInterval)
        # bracket must CONTAIN the true value 2π (π's own bracket is
        # width 1e-60 — containment of it is containment of the truth)
        assert iv.lo <= two_pi.lo and two_pi.hi <= iv.hi
        widths.append(iv.width)
    # tighten-or-refuse: deeper subdivision must tighten the bracket
    assert widths[2] < widths[1] < widths[0]


def test_trimmed_patch_flux_ci_refuses_polynomial() -> None:
    poly = bezier_surface([[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 1)]])
    with pytest.raises(ValueError, match="trimmed_patch_flux"):
        trimmed_patch_flux_ci(poly, [[(0, 0), (1, 0), (1, 1), (0, 1)]])


def test_trimmed_patch_flux_ci_half_height_brackets_pi() -> None:
    # trim to v ∈ [0, 1/2] — half the wall height → true flux = π
    cyl = quarter_cylinder()
    lo_half = [(0, 0), (1, 0), (1, F(1, 2)), (0, F(1, 2))]
    pi = pi_interval()
    iv = trimmed_patch_flux_ci(cyl, [lo_half], depth=5)
    assert iv.lo <= pi.lo and pi.hi <= iv.hi


def test_trimmed_patch_flux_ci_full_domain_matches_patch_flux_ci() -> None:
    cyl = quarter_cylinder()
    full = [(0, 0), (1, 0), (1, 1), (0, 1)]
    a = trimmed_patch_flux_ci(cyl, [full], depth=4)
    b = patch_flux_ci(cyl, depth=4)
    assert a.lo == b.lo and a.hi == b.hi


def test_trimmed_patch_flux_ci_split_sum_brackets_two_pi() -> None:
    # re-trim additivity, certified flavor: bottom half + top half is a
    # bracket for the whole (interval sum preserves enclosure)
    cyl = quarter_cylinder()
    bot = [(0, 0), (1, 0), (1, F(1, 2)), (0, F(1, 2))]
    top = [(0, F(1, 2)), (1, F(1, 2)), (1, 1), (0, 1)]
    s = trimmed_patch_flux_ci(cyl, [bot], depth=4) \
        + trimmed_patch_flux_ci(cyl, [top], depth=4)
    two_pi = pi_interval() * CInterval.exact(2)
    assert s.lo <= two_pi.lo and two_pi.hi <= s.hi


# -- the derivation's boolean case: dome ∩ half-space z ≥ 1 -------------------

def dome_and_plane():
    dome = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                           [(2, 0, 0), (2, 2, 6), (2, 4, 0)],
                           [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
    plane = bezier_surface([[(0, 0, 1), (0, 4, 1)], [(4, 0, 1), (4, 4, 1)]])
    return dome, plane


# High-accuracy independent reference for the cap volume, from the exact
# 1D reduction V = (8/3)·∫ a(u)·s(u)³ du, a = 24u(1−u), s = √(1−4/a),
# over u where a > 4 — Simpson at 2·10⁶ points: 1.0794724554159079.
# (The derivation's cruder 2000² midpoint rule gave 1.079472473.)
CAP_REF = F(10794724554159, 10 ** 13)


def test_ssi_strips_enclose_the_trim_curve_on_both_sides() -> None:
    dome, plane = dome_and_plane()
    a_cells, b_cells = ssi_strips(dome, plane, depth=4)
    assert a_cells and b_cells
    h = F(1, 2 ** 4)
    for (u0, u1, v0, v1) in list(a_cells) + list(b_cells):
        assert u1 - u0 == h and v1 - v0 == h        # dyadic at 2^-depth
    # every certified SSI point lies inside the strip it came from
    from forgekernel.ssi import ssi_curves
    out = ssi_curves(dome, plane, depth=4)
    for (u, v, s, t) in out["curves"][0]["points"]:
        assert any(u0 <= u <= u1 and v0 <= v <= v1
                   for (u0, u1, v0, v1) in a_cells)
        assert any(s0 <= s <= s1 and t0 <= t <= t1
                   for (s0, s1, t0, t1) in b_cells)


def test_certified_trim_flux_empty_strip_is_exact_patch_flux() -> None:
    dome, _ = dome_and_plane()
    iv = certified_trim_flux(dome, [], lambda u, v: True, depth=3)
    assert iv.width == 0
    assert iv.lo == patch_flux(dome)


def test_dome_half_space_volume_certified_bracket() -> None:
    """The ADR-0019 headline: the boolean volume as certified ± e.

    The hand pipeline (probe 5) got 1.069 at depth 4 — a silent 1e-2 off
    the truth with no error bar. Here the same depth yields an interval
    that provably CONTAINS the truth, and deepening tightens it."""
    dome, plane = dome_and_plane()

    def above(u, v):                     # exact membership vs the half-space
        return dome.eval(u, v)[2] > 1

    widths = []
    for depth in (4, 5):
        a_cells, b_cells = ssi_strips(dome, plane, depth=depth)
        f_dome = certified_trim_flux(dome, a_cells, above, depth=depth)
        f_plane = certified_trim_flux(plane, b_cells, above, depth=depth)
        # dome's Su×Sv points +z (outward for the cap top); the plane's
        # +z orientation points INTO the cap, so its face enters negated
        vol = f_dome - f_plane
        # containment of the independent reference — the truth test
        assert vol.lo <= CAP_REF <= vol.hi
        widths.append(vol.width)
    assert widths[1] < widths[0]         # deeper strip → tighter bracket
    # pin the achieved tightness (measured 3.98 / 2.04; the bound is O(2^-d)
    # — first order is information-theoretically forced by a cell enclosure)
    assert widths[0] < F(41, 10)
    assert widths[1] < F(21, 10)
