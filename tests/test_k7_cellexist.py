"""K7 gap item 9 — certified per-cell existence: no silent drops.

Detection (K3.3 subdivision) guarantees every intersection branch is HIT
by surviving cells; certification then lands one residual-certified point
per cell. The gap this file closes: a cell whose Newton refinement failed
was silently dropped — measured 25/42 (depth 3) and 18/54 (depth 4) cells
on wavy bicubic multi-knot pairs — which would punch holes in the trim
loops a K7 boolean builds from those points.

The contract now: every surviving cell ends in exactly one of three
states, and the third is loud —

* a certified point (existence: exact rational residual < 1e-20),
* proven empty (exclusion: deeper subdivision prunes the cell's every
  descendant pair — the same bbox-disjointness proof detection uses), or
* ``SsiCellUncertified`` — a named refusal carrying the cells that could
  be neither certified nor excluded (tangential contact, or resolution
  budget), instead of a bare count the caller cannot act on.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.nurbs import BSplineSurface, bezier_surface
from forgekernel.ssi import (BezierPatch, SsiCellUncertified,
                             _resolve_cell_pairs, refine_point, ssi,
                             ssi_curves, ssi_surfaces)

F = Fraction


def _wavy(z0, amp, shift=0):
    """Wavy bicubic with an interior knot — the measured degradation case."""
    net = [[(i + shift, j, z0 + (amp if (i + j) % 2 else -amp) * F(1, 2))
            for j in range(5)] for i in range(5)]
    k = [0, 0, 0, 0, F(1, 2), 1, 1, 1, 1]
    return BSplineSurface(3, 3, net, k, k)


def test_wavy_bicubic_every_cell_is_classified() -> None:
    # Before the existence test: depth 3 dropped 25 of 42 cells, depth 4
    # dropped 18 of 54 — silently, as a bare "uncertified" count.
    Wa = _wavy(F(0), F(2))
    Wb = _wavy(F(1, 4), F(2), shift=F(1, 3))
    for depth in (3, 4):
        out = ssi_curves(Wa, Wb, depth=depth)
        npts = sum(len(c["points"]) for c in out["curves"])
        assert out["uncertified"] == 0
        # full accounting: every surviving cell is a point or proven empty
        assert out["cells"] == npts + out["empty_cells"]
        assert npts > 0
        assert not out["empty_certified"]


def test_wavy_bicubic_points_still_certified() -> None:
    # tightened cells must meet the SAME certificate as first-try cells
    Wa = _wavy(F(0), F(2))
    Wb = _wavy(F(1, 4), F(2), shift=F(1, 3))
    out = ssi_curves(Wa, Wb, depth=3)
    assert out["tightened"] > 0                  # the resolver actually ran
    for c in out["curves"]:
        for (u, v, s, t) in c["points"]:
            pa, pb = Wa.eval(u, v), Wb.eval(s, t)
            r2 = sum((pa[k] - pb[k]) ** 2 for k in range(3))
            assert r2 < F(1, 10 ** 20)           # exact rational certificate


def test_point_tangency_is_found_and_certified() -> None:
    # plane through the exact peak: before, ssi_curves returned ZERO curves
    # and uncertified=4 — a genuine (tangential, single-point) contact
    # silently dropped. The resolver's deepened retries now land points
    # meeting the standard spatial certificate (|A−B| < 1e-10; near a
    # tangency a whole (Δu)² neighbourhood qualifies — the same semantics
    # test_tangent_branch_found_where_float_kernels_miss pins).
    bump = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                           [(2, 0, 0), (2, 2, 2), (2, 4, 0)],
                           [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
    plane = bezier_surface([[(0, 0, F(1, 2)), (0, 4, F(1, 2))],
                            [(4, 0, F(1, 2)), (4, 4, F(1, 2))]])
    out = ssi_curves(bump, plane, depth=4)
    npts = sum(len(c["points"]) for c in out["curves"])
    assert out["uncertified"] == 0
    assert out["cells"] == npts + out["empty_cells"]
    assert npts > 0                              # the contact is REPORTED
    for c in out["curves"]:
        for (u, v, s, t) in c["points"]:
            pa, pb = bump.eval(u, v), plane.eval(s, t)
            assert sum((pa[k] - pb[k]) ** 2 for k in range(3)) < F(1, 10 ** 20)
            # and it really is the peak contact
            assert abs(float(u) - 0.5) < 1e-3 and abs(float(v) - 0.5) < 1e-3


def _tangent_sheet_and_plane(offset):
    """z = u² (ruled in v) against its tangent plane at the NON-dyadic
    u = 1/3, vertically shifted by ``offset``: gap (u − 1/3)² + offset."""
    sheet = bezier_surface([[(0, 0, 0), (0, 1, 0)],
                            [(F(1, 2), 0, 0), (F(1, 2), 1, 0)],
                            [(1, 0, 1), (1, 1, 1)]])
    z0, z1 = -F(1, 9) + offset, F(5, 9) + offset
    plane = bezier_surface([[(0, 0, z0), (0, 1, z0)],
                            [(1, 0, z1), (1, 1, z1)]])
    return sheet, plane


def test_line_tangency_refuses_by_name_not_silently() -> None:
    # exact tangential line at u = 1/3: Newton cannot land a certifiable
    # point (non-dyadic double root) and pruning cannot separate touching
    # surfaces — before, the cells were dropped as a bare count; now the
    # pair refuses by name, carrying WHERE in both parameter domains.
    sheet, plane = _tangent_sheet_and_plane(F(0))
    with pytest.raises(SsiCellUncertified) as ei:
        ssi_curves(sheet, plane, depth=3)
    assert "ssi_cell_uncertified" in str(ei.value)
    assert "tangen" in str(ei.value)             # names the likely cause
    assert ei.value.cells
    for (abox, bbox_) in ei.value.cells:
        assert len(abox) == 4 and len(bbox_) == 4


def test_subresolution_gap_refuses_rather_than_lies() -> None:
    # gap (u-1/3)² + 1e-8: NO intersection, but the gap sits in the dead
    # zone — too wide to certify a point (needs < 1e-10), too narrow for
    # subdivision to prune within budget. The only honest answers are
    # "proven empty" (unreachable here) or a named refusal; claiming
    # either an intersection or a certified emptiness would be a lie.
    sheet, plane = _tangent_sheet_and_plane(-F(1, 10 ** 8))
    with pytest.raises(SsiCellUncertified):
        ssi_curves(sheet, plane, depth=3)


def test_near_tangency_split_resolves_both_ways() -> None:
    # shifting the tangent plane INTO the sheet by 1e-8 creates two real
    # intersection lines at u = 1/3 ± 1e-4. Cells between/near the lines
    # fail first-try Newton; the resolver must certify the real ones and
    # PROVE the spurious ones empty — the both-directions workout.
    sheet, plane = _tangent_sheet_and_plane(F(1, 10 ** 8))
    out = ssi_curves(sheet, plane, depth=4)
    npts = sum(len(c["points"]) for c in out["curves"])
    assert out["uncertified"] == 0
    assert out["cells"] == npts + out["empty_cells"]
    assert npts > 0 and out["empty_cells"] > 0   # both verdicts exercised
    for c in out["curves"]:
        for (u, v, s, t) in c["points"]:
            pa, pb = sheet.eval(u, v), plane.eval(s, t)
            assert sum((pa[k] - pb[k]) ** 2 for k in range(3)) < F(1, 10 ** 20)


def test_transversal_control_is_unchanged() -> None:
    # dome × plane z=1/4: closed transversal loop, no resolution needed
    bump = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                           [(2, 0, 0), (2, 2, 2), (2, 4, 0)],
                           [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
    plane = bezier_surface([[(0, 0, F(1, 4)), (0, 4, F(1, 4))],
                            [(4, 0, F(1, 4)), (4, 4, F(1, 4))]])
    out = ssi_curves(bump, plane, depth=4)
    assert out["uncertified"] == 0
    assert out["empty_cells"] == 0               # nothing needed resolving
    assert out["tightened"] == 0
    npts = sum(len(c["points"]) for c in out["curves"])
    assert out["cells"] == npts > 0


def test_resolver_proves_a_spurious_pair_empty() -> None:
    # z = (u-1/2)^2 + 1/64 never meets z = 0, but at the ROOT level the
    # control-net boxes overlap (Bernstein coeff -15/64 < 0). The resolver
    # must turn that surviving pair into a certified exclusion.
    plane = BezierPatch([[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)]])
    sheet = BezierPatch([[(0, 0, F(17, 64)), (0, 1, F(17, 64))],
                         [(F(1, 2), 0, F(-15, 64)), (F(1, 2), 1, F(-15, 64))],
                         [(1, 0, F(17, 64)), (1, 1, F(17, 64))]])

    def newton(um, vm, sm, tm, iters=12):
        return refine_point(plane.net, sheet.net, um, vm, sm, tm, iters)

    verdict, pt = _resolve_cell_pairs(
        [(plane, sheet)], (F(0), F(1), F(0), F(1)), newton)
    assert verdict == "empty" and pt is None


def test_resolver_finds_the_point_when_one_exists() -> None:
    # z = u - 1/2 crosses z = 0: the resolver's tighten path must land a
    # certified point inside the cell, not report empty or give up.
    plane = BezierPatch([[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)]])
    lin = BezierPatch([[(0, 0, F(-1, 2)), (0, 1, F(-1, 2))],
                       [(1, 0, F(1, 2)), (1, 1, F(1, 2))]])

    def newton(um, vm, sm, tm, iters=12):
        return refine_point(plane.net, lin.net, um, vm, sm, tm, iters)

    verdict, pt = _resolve_cell_pairs(
        [(plane, lin)], (F(0), F(1), F(0), F(1)), newton)
    assert verdict == "point"
    u, v, s, t = pt
    assert F(0) <= u <= F(1) and F(0) <= v <= F(1)
    pa = (u, v, F(0))
    pb = (s, t, s - F(1, 2))
    assert sum((pa[c] - pb[c]) ** 2 for c in range(3)) < F(1, 10 ** 20)


def test_ssi_surfaces_accounting_matches_curves() -> None:
    Wa = _wavy(F(0), F(2))
    Wb = _wavy(F(1, 4), F(2), shift=F(1, 3))
    r = ssi_surfaces(Wa, Wb, depth=3)
    assert r["uncertified"] == 0
    assert r["cells"] == len(r["points"]) + r["empty_cells"]


def test_bezier_level_ssi_reports_accounting() -> None:
    # the Bézier-pair entry point carries the same contract
    plane = BezierPatch([[(0, 0, 0), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)]])
    lin = BezierPatch([[(0, 0, F(-1, 2)), (0, 1, F(-1, 2))],
                       [(1, 0, F(1, 2)), (1, 1, F(1, 2))]])
    r = ssi(plane, lin, depth=5)
    assert r["uncertified"] == 0
    assert r["cells"] == len(r["points"]) + r["empty_cells"]
    assert r["branches"] == 1
