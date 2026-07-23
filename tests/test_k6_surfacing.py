"""K6.0 — surfacing: exact Gaussian curvature, Coons patches, G1 proofs."""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.nurbs import bezier, bezier_surface
from forgekernel.surfacing import (coons_patch, g1_certify,
                                   gaussian_curvature, mean_curvature)

F = Fraction

_B = [0, 0, 1]          # Bézier coefficients of t²


def _paraboloid():
    return bezier_surface([[(F(i, 2), F(j, 2), _B[i] + _B[j])
                            for j in range(3)] for i in range(3)])


def test_gaussian_curvature_is_exactly_rational() -> None:
    # z = u² + v²:  K = 4/(1 + 4u² + 4v²)², exact at rational params
    s = _paraboloid()
    assert gaussian_curvature(s, 0, 0) == 4
    assert gaussian_curvature(s, F(1, 2), F(1, 2)) == F(4, 9)
    assert isinstance(gaussian_curvature(s, F(1, 3), F(1, 7)), Fraction)


def test_developable_has_exactly_zero_curvature() -> None:
    # a parabolic cylinder is developable: K ≡ 0 — and forge says
    # EXACTLY zero, not 1e-17
    cyl = bezier_surface([[(F(i, 2), j, _B[i]) for j in range(2)]
                          for i in range(3)])
    for uv in ((F(1, 3), F(2, 7)), (F(4, 5), F(1, 9))):
        assert gaussian_curvature(cyl, *uv) == 0


def test_mean_curvature_certified() -> None:
    H = mean_curvature(_paraboloid(), 0, 0)      # analytic H = 2
    assert H.lo <= 2 <= H.hi
    assert H.width < F(1, 10 ** 30)


def test_coons_interpolates_all_boundaries_exactly() -> None:
    c0 = [(0, 0, 0), (1, 0, 1), (2, 0, 0)]
    c1 = [(0, 2, 0), (1, 2, 1), (2, 2, 0)]
    d0 = [(0, 0, 0), (0, 1, 1), (0, 2, 0)]
    d1 = [(2, 0, 0), (2, 1, 1), (2, 2, 0)]
    cp = coons_patch(c0, c1, d0, d1)
    for t in (F(0), F(1, 3), F(1, 2), F(4, 5), F(1)):
        assert cp.eval(t, 0) == bezier(c0).eval(t)
        assert cp.eval(t, 1) == bezier(c1).eval(t)
        assert cp.eval(0, t) == bezier(d0).eval(t)
        assert cp.eval(1, t) == bezier(d1).eval(t)


def test_coons_corner_mismatch_refuses() -> None:
    with pytest.raises(ValueError, match="corner"):
        coons_patch([(0, 0, 0), (1, 0, 0)], [(0, 1, 0), (1, 1, 0)],
                    [(9, 9, 9), (0, 1, 0)], [(1, 0, 0), (1, 1, 0)])


def test_g1_certification_is_a_proof_not_a_heuristic() -> None:
    # A: z = x² over x∈[0,1]; B continues z = x² over [1,2].
    # At the seam x=1 both have slope 2 → G1 holds; nudging B's middle
    # control row breaks the tangent and must be DETECTED.
    A = bezier_surface([[(0, 0, 0), (0, 1, 0)],
                        [(F(1, 2), 0, 0), (F(1, 2), 1, 0)],
                        [(1, 0, 1), (1, 1, 1)]])
    B = bezier_surface([[(1, 0, 1), (1, 1, 1)],
                        [(F(3, 2), 0, 2), (F(3, 2), 1, 2)],
                        [(2, 0, 4), (2, 1, 4)]])
    assert g1_certify(A, B)
    B_bad = bezier_surface([[(1, 0, 1), (1, 1, 1)],
                            [(F(3, 2), 0, 1), (F(3, 2), 1, 1)],
                            [(2, 0, 4), (2, 1, 4)]])
    assert not g1_certify(A, B_bad)
    # G0 breakage (seam positions differ) also detected
    B_gap = bezier_surface([[(1, 0, 2), (1, 1, 2)],
                            [(F(3, 2), 0, 2), (F(3, 2), 1, 2)],
                            [(2, 0, 4), (2, 1, 4)]])
    assert not g1_certify(A, B_gap)


# -- K6.1: rational 2nd partials, G2 proofs, curvature combs ------------------

def test_rational_second_partials_consistent() -> None:
    from forgekernel.nurbs import BSplineSurface, surface_partials2

    arc = [(1, 0), (1, 1), (0, 1)]
    net = [[(x, y, F(-1, 2)), (x, y, F(1, 2))] for (x, y) in arc]
    wts = [[F(1), F(1)], [F(3, 4), F(3, 4)], [F(1), F(1)]]
    s = BSplineSurface(2, 1, net, [0, 0, 0, 1, 1, 1], [0, 0, 1, 1], wts)
    # first partials from the quotient-rule partials2 must equal the
    # independently implemented (and OCCT-oracle-tested) partials()
    S, Su, Sv = s.partials(F(1, 3), F(2, 5))
    S2, Su2, Sv2, Suu, Suv, Svv = surface_partials2(s, F(1, 3), F(2, 5))
    assert (S, Su, Sv) == (S2, Su2, Sv2)
    assert all(isinstance(c, Fraction) for c in Suu + Suv + Svv)


def test_g2_certifies_curvature_continuity() -> None:
    from forgekernel.surfacing import g1_certify, g2_certify

    # A: z = x² on [0,1]; B continues z = x² on [1,2] → C∞ → G2 holds
    A = bezier_surface([[(0, 0, 0), (0, 1, 0)],
                        [(F(1, 2), 0, 0), (F(1, 2), 1, 0)],
                        [(1, 0, 1), (1, 1, 1)]])
    B = bezier_surface([[(1, 0, 1), (1, 1, 1)],
                        [(F(3, 2), 0, 2), (F(3, 2), 1, 2)],
                        [(2, 0, 4), (2, 1, 4)]])
    assert g2_certify(A, B)
    # z = 2(x−1)² + 2(x−1) + 1: SAME tangent at the seam (G1 true) but
    # curvature 4 vs 2 — the G2 certifier must catch what G1 cannot
    B2 = bezier_surface([[(1, 0, 1), (1, 1, 1)],
                         [(F(3, 2), 0, 2), (F(3, 2), 1, 2)],
                         [(2, 0, 5), (2, 1, 5)]])
    assert g1_certify(A, B2)          # tangent-continuous...
    assert not g2_certify(A, B2)      # ...but curvature breaks


def test_curvature_comb_peaks_at_the_apex() -> None:
    from forgekernel.surfacing import curve_curvature_comb

    comb = curve_curvature_comb(bezier([(0, 0, 0), (1, 2, 0), (2, 0, 0)]), n=8)
    lens = [((t[0] - p[0]) ** 2 + (t[1] - p[1]) ** 2) ** 0.5
            for p, t in comb]
    assert max(lens) == lens[len(lens) // 2]      # parabola: κ max at apex
    # the comb is a float VIEWER artifact (unlike the exact curvature
    # values) — symmetric ends agree to float precision, not bitwise
    assert lens[0] == pytest.approx(lens[-1])


# -- K6.2: the G1 blend strip — self-certifying fill --------------------------

def test_blend_strip_certifies_g1_to_both_neighbours() -> None:
    from forgekernel.surfacing import g1_blend_strip, g1_certify

    # A: z=x² over [0,1]; B: z=(x−3)² over [3,4] — a gap from x=1 to 3.
    A = bezier_surface([[(0, 0, 0), (0, 1, 0)],
                        [(F(1, 2), 0, 0), (F(1, 2), 1, 0)],
                        [(1, 0, 1), (1, 1, 1)]])
    B = bezier_surface([[(3, 0, 1), (3, 1, 1)],
                        [(F(7, 2), 0, 0), (F(7, 2), 1, 0)],
                        [(4, 0, 0), (4, 1, 0)]])
    strip = g1_blend_strip(A, B, a_edge="u1", b_edge="u0")
    # the construction is Hermite; the PROOF is the polynomial-identity
    # certifier — the strip must pass it against both neighbours
    assert g1_certify(A, strip, a_edge="u1", b_edge="v0")
    assert g1_certify(strip, B, a_edge="v1", b_edge="u0")
    # boundary interpolation is exact
    assert strip.eval(0, 0) == (1, 0, 1)
    assert strip.eval(0, 1) == (3, 0, 1)
