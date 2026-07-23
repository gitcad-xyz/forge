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
