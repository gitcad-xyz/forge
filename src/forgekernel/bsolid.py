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


def trimmed_patch_flux(surface: BSplineSurface, loops) -> Fraction:
    """(1/3)∮∮_D S·(S_u×S_v) du dv over the TRIMMED parameter region D of a
    polynomial patch — D bounded by polygonal ``loops`` in the surface's
    (u, v) domain (outer CCW, holes CW; use ``TrimmedPatch.normalized()``).

    Green's theorem turns the area integral into a contour integral over the
    loop edges: ∫∫_D F du dv = ∮_∂D G dv with G(u,v)=∫_{u0}^{u} F(u',v) du'.
    F = S·(S_u×S_v) is a polynomial (u-degree 3p−1, v-degree 3q−1), so the
    inner u-antiderivative is an exact 3p-node quadrature and the outer
    edge integral (G is degree 3(p+q)−1 along a straight edge) an exact
    3(p+q)-node one — the whole result is exact ℚ.

    Exactness holds for polynomial patches AND polygonal trim loops. When
    the loops are the polyline sampling of a curved SSI trim boundary, the
    result is exact for THAT polygon — i.e. it carries the boundary's
    discretization error, not a rounding one (the honest K7 caveat)."""
    if any(w != F(1) for row in surface.w for w in row):
        raise ValueError("exact trimmed flux: polynomial patches only (K7.1)")
    p, q = surface.p, surface.q
    (ud0, _), _ = surface.domain()
    ud0 = F(ud0)
    inn = _nodes(3 * p)
    inw = _lagrange_weights(inn)
    otn = _nodes(3 * (p + q))
    otw = _lagrange_weights(otn)

    def Fpt(u, v):
        S, Su, Sv = surface_partials2(surface, u, v)[:3]
        return _triple(S, Su, Sv)

    def Gpt(u, v):                 # ∫_{ud0}^{u} F(u',v) du' via σ∈[0,1] map
        span = u - ud0
        return span * sum(w * Fpt(ud0 + s * span, v) for w, s in zip(inw, inn))

    total = F(0)
    for loop in loops:
        pts = [(F(a), F(b)) for a, b in loop]
        m = len(pts)
        for k in range(m):
            (ua, va), (ub, vb) = pts[k], pts[(k + 1) % m]
            dvv = vb - va
            if dvv == 0:           # a horizontal edge adds nothing to ∮ G dv
                continue
            duu = ub - ua
            edge = F(0)
            for w, tau in zip(otw, otn):
                edge += w * Gpt(ua + tau * duu, va + tau * dvv)
            total += dvv * edge
    return total / 3


def trimmed_solid_volume(faces) -> Fraction:
    """Volume of a solid whose closed, outward-oriented boundary is a set of
    TRIMMED polynomial patches — Σ per-face flux, exact ℚ. ``faces`` is a
    list of ``(surface, loops)``. This is the boolean-assembly reduction:
    a boolean re-trims faces and adds intersection-curve loops, but the
    enclosed volume is just the sum of the trimmed-face fluxes."""
    return abs(sum((trimmed_patch_flux(s, loops) for s, loops in faces), F(0)))


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


# -- K7.0b: exact inertia tensor (same flux trick, one degree higher) ---------

def _flux_moment(surface: BSplineSurface, fx, fy, fz):
    """(1/1) ∮∮ (fx,fy,fz)·(S_u×S_v) du dv where fx,fy,fz are callables
    of the point S — used to lift a volume integral to a surface flux.
    Polynomial integrand ⇒ exact ℚ."""
    if any(w != F(1) for row in surface.w for w in row):
        raise ValueError("exact moments: polynomial patches only (K7.1)")
    p, q = surface.p, surface.q
    (u0, u1), (v0, v1) = surface.domain()
    u0, u1, v0, v1 = F(u0), F(u1), F(v0), F(v1)
    du, dv = u1 - u0, v1 - v0
    # A moment ∮ f(S)·n with f of coordinate-degree m has integrand degree
    # m·p + (2p−1) = (m+2)p−1 in u. The heaviest moment used here is the
    # SECOND moment (m=3 → 5p−1), so 5p nodes are needed — NOT 3p+2, which
    # only coincides at p=1 (the trap that let degree-1 box tests pass while
    # degree≥2 patches returned a wrong, non-exact inertia tensor).
    un, vn = _nodes(5 * p), _nodes(5 * q)
    uw, vw = _lagrange_weights(un), _lagrange_weights(vn)
    total = F(0)
    for i, uu in enumerate(un):
        for j, vv in enumerate(vn):
            S, Su, Sv = surface_partials2(surface, u0 + uu * du, v0 + vv * dv)[:3]
            nx = (du * Su[1]) * (dv * Sv[2]) - (du * Su[2]) * (dv * Sv[1])
            ny = (du * Su[2]) * (dv * Sv[0]) - (du * Su[0]) * (dv * Sv[2])
            nz = (du * Su[0]) * (dv * Sv[1]) - (du * Su[1]) * (dv * Sv[0])
            total += uw[i] * vw[j] * (fx(S) * nx + fy(S) * ny + fz(S) * nz)
    return total


def mass_properties(solid: "PatchSolid") -> dict:
    """Exact volume, centroid, and inertia tensor (about the centroid) of
    a Bézier-patch solid — every entry an exact ``Fraction``.

    Divergence theorem lifts each volume integral to a boundary flux of a
    polynomial: V=∮(x,·,·)·n, ∫x=∮(x²/2,·,·)·n, ∫x²=∮(x³/3,·,·)·n,
    ∫xy=∮(x²y/2,·,·)·n, …"""
    zero = lambda S: F(0)
    V = sum((_flux_moment(p, lambda S: S[0], zero, zero)
             for p in solid.patches), F(0))
    sign = 1 if V >= 0 else -1
    V *= sign

    def moment(fx):
        return sign * sum((_flux_moment(p, fx, zero, zero)
                           for p in solid.patches), F(0))

    mx = moment(lambda S: S[0] * S[0] / 2)
    my = sign * sum((_flux_moment(p, zero, lambda S: S[1] * S[1] / 2, zero)
                     for p in solid.patches), F(0))
    mz = sign * sum((_flux_moment(p, zero, zero, lambda S: S[2] * S[2] / 2)
                     for p in solid.patches), F(0))
    cx, cy, cz = mx / V, my / V, mz / V
    Ixx_o = moment(lambda S: S[0] ** 3 / 3)              # ∫x² dV
    Iyy_o = sign * sum((_flux_moment(p, zero, lambda S: S[1] ** 3 / 3, zero)
                        for p in solid.patches), F(0))
    Izz_o = sign * sum((_flux_moment(p, zero, zero, lambda S: S[2] ** 3 / 3)
                        for p in solid.patches), F(0))
    Ixy_o = moment(lambda S: S[0] * S[0] * S[1] / 2)     # ∫xy dV
    Iyz_o = sign * sum((_flux_moment(p, zero, lambda S: S[1] * S[1] * S[2] / 2,
                                     zero) for p in solid.patches), F(0))
    Izx_o = sign * sum((_flux_moment(p, zero, zero,
                                     lambda S: S[2] * S[2] * S[0] / 2)
                        for p in solid.patches), F(0))
    # inertia tensor about the CENTROID (parallel-axis, exact)
    Ixx = (Iyy_o + Izz_o) - V * (cy * cy + cz * cz)
    Iyy = (Izz_o + Ixx_o) - V * (cz * cz + cx * cx)
    Izz = (Ixx_o + Iyy_o) - V * (cx * cx + cy * cy)
    Ixy = -(Ixy_o - V * cx * cy)
    Iyz = -(Iyz_o - V * cy * cz)
    Izx = -(Izx_o - V * cz * cx)
    return {"volume": V, "centroid": (cx, cy, cz),
            "inertia": ((Ixx, Ixy, Izx), (Ixy, Iyy, Iyz), (Izx, Iyz, Izz))}


# -- K7.0d: exact mass properties of a planar Solid (reuse the flux) ----------

def solid_to_patches(solid):
    """Convert a planar forge ``Solid`` to flat degenerate Bézier patches
    (one per boundary triangle) so the exact flux machinery applies. A
    triangle (v0,v1,v2) becomes the collapsed bilinear patch
    [[v0,v0],[v1,v2]] — its S_u×S_v is the triangle's outward area
    vector, exactly."""
    patches = []
    for poly in solid.polys:
        vs = [tuple(F(c) for c in v) for v in poly.verts]
        for i in range(1, len(vs) - 1):        # fan triangulation
            a, b, c = vs[0], vs[i], vs[i + 1]
            patches.append(bezier_surface([[a, a], [b, c]]))
    return patches


def polyhedron_mass_properties(solid) -> dict:
    """Exact volume + centroid + inertia tensor of a planar ``Solid``,
    every entry a Fraction — via the same divergence-theorem flux used
    for freeform solids."""
    return mass_properties(PatchSolid(solid_to_patches(solid)))
