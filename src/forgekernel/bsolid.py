"""K7 — boundary-represented freeform solids (NURBS patches).

The keystone result: the volume of a solid bounded by **polynomial**
Bézier patches is *exactly rational*. By the divergence theorem

    V = (1/3) ∮∮_∂Ω  S · (S_u × S_v)  du dv   (summed over patches),

and the integrand ``S·(S_u×S_v)`` is a polynomial in (u,v). A polynomial
integrates exactly over [0,1]², so V ∈ ℚ — no epsilon, not even ℚ[π].
OCCT can only Gauss-quadrature the same flux to a tolerance.

Exact integration uses an interpolatory rational quadrature: n = 3p
distinct rational nodes give weights (integrals of Lagrange bases) that
are exact rationals and integrate any degree ≤ 3p−1 polynomial exactly
— which the flux is, in each variable. Rational (non-polynomial) patches
give a rational integrand and a **certified interval** volume instead
(ADR-0019); that is K7.1.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.nurbs import BSplineSurface, bezier_surface, surface_partials2

F = Fraction


def _lagrange_weights(nodes):
    """Interpolatory quadrature weights on ``nodes`` ⊂ [0,1]: w_k =
    ∫₀¹ ∏_{m≠k}(x−x_m)/(x_k−x_m) dx — exact rationals. Exact for any
    polynomial of degree ≤ len(nodes)−1."""
    n = len(nodes)
    weights = []
    for k in range(n):
        # Lagrange basis numerator ∏_{m≠k}(x − x_m) as power-basis coeffs
        coeffs = [F(1)]                 # polynomial "1"
        denom = F(1)
        for m in range(n):
            if m == k:
                continue
            # multiply by (x − x_m)
            new = [F(0)] * (len(coeffs) + 1)
            for i, c in enumerate(coeffs):
                new[i + 1] += c            # x·c
                new[i] += -nodes[m] * c    # −x_m·c
            coeffs = new
            denom *= (nodes[k] - nodes[m])
        # integrate power series ∫₀¹ Σ c_i x^i = Σ c_i/(i+1)
        integ = sum(c / (i + 1) for i, c in enumerate(coeffs))
        weights.append(integ / denom)
    return weights


def _nodes(n):
    """n distinct rationals in (0,1): the Chebyshev-like split k+1/(n+1)."""
    return [F(k + 1, n + 1) for k in range(n)]


def _triple(a, b, c):
    """Scalar triple product a·(b×c), exact."""
    cx = (b[1] * c[2] - b[2] * c[1],
          b[2] * c[0] - b[0] * c[2],
          b[0] * c[1] - b[1] * c[0])
    return a[0] * cx[0] + a[1] * cx[1] + a[2] * cx[2]


def patch_flux(surface: BSplineSurface) -> Fraction:
    """(1/3)∮∮ S·(S_u×S_v) du dv over the surface's parameter domain —
    the patch's contribution to the enclosed signed volume. EXACT ℚ for
    polynomial surfaces (raises for rational — K7.1 certified path)."""
    if any(w != F(1) for row in surface.w for w in row):
        raise ValueError("exact flux: polynomial patches only (K7.1)")
    p, q = surface.p, surface.q
    (u0, u1), (v0, v1) = surface.domain()
    u0, u1, v0, v1 = F(u0), F(u1), F(v0), F(v1)
    du, dv = u1 - u0, v1 - v0
    # degree of S·(S_u×S_v) is (3p−1, 3q−1) → need 3p, 3q nodes
    un, vn = _nodes(3 * p), _nodes(3 * q)
    uw, vw = _lagrange_weights(un), _lagrange_weights(vn)
    total = F(0)
    for i, uu in enumerate(un):
        for j, vv in enumerate(vn):
            U, Su, Sv = surface_partials2(surface, u0 + uu * du, v0 + vv * dv)[:3]
            # chain rule: dS/dū = Su·du, dS/dv̄ = Sv·dv (ū,v̄ ∈ [0,1])
            t = _triple(U, tuple(du * s for s in Su), tuple(dv * s for s in Sv))
            total += uw[i] * vw[j] * t
    return total / 3


class PatchSolid:
    """A closed solid whose boundary is a list of outward-oriented
    polynomial Bézier patches. Volume exact in ℚ via the flux theorem."""

    provenance = "exact"

    def __init__(self, patches) -> None:
        self.patches = list(patches)
        if not self.patches:
            raise ValueError("PatchSolid needs at least one boundary patch")

    def volume(self) -> Fraction:
        v = sum((patch_flux(p) for p in self.patches), F(0))
        return abs(v)

    def bbox_f(self):
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        for patch in self.patches:
            for row in patch.cp:
                for pt in row:
                    for c in range(3):
                        lo[c] = min(lo[c], float(pt[c]))
                        hi[c] = max(hi[c], float(pt[c]))
        return tuple(lo), tuple(hi)


def box_patches(dx, dy, dz, origin=(0, 0, 0)):
    """Six outward flat Bézier patches forming a box (degree 1×1) — the
    hand-checkable sanity solid: volume must be exactly dx·dy·dz."""
    ox, oy, oz = (F(v) for v in origin)
    dx, dy, dz = F(dx), F(dy), F(dz)
    x0, y0, z0 = ox, oy, oz
    x1, y1, z1 = ox + dx, oy + dy, oz + dz

    def patch(p00, p10, p01, p11):
        return bezier_surface([[p00, p01], [p10, p11]])
    # each patch oriented so S_u×S_v points OUTWARD
    return [
        patch((x0, y0, z0), (x0, y1, z0), (x1, y0, z0), (x1, y1, z0)),   # z0 (−z out): check sign via abs
        patch((x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)),   # z1 (+z)
        patch((x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (x1, y0, z1)),   # y0
        patch((x0, y1, z0), (x0, y1, z1), (x1, y1, z0), (x1, y1, z1)),   # y1
        patch((x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1)),   # x0
        patch((x1, y0, z0), (x1, y1, z0), (x1, y0, z1), (x1, y1, z1)),   # x1
    ]
