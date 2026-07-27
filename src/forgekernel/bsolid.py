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
(ADR-0019); that is K7.1 — built below as ``patch_flux_ci`` /
``trimmed_patch_flux_ci``, alongside ``certified_trim_flux``, which
brackets a boolean face's flux through the SSI cell enclosure of its
trim curve so the boolean volume reports "certified ± e".
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
        raise ValueError("exact flux: polynomial patches only — rational "
                         "patches take the certified bracket patch_flux_ci "
                         "(K7.1)")
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
        raise ValueError("exact trimmed flux: polynomial patches only — "
                         "rational patches take the certified bracket "
                         "trimmed_patch_flux_ci (K7.1)")
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


# =============================================================================
# K7.1 / K7.0c — the CERTIFIED flux tier (ADR-0019)
#
# Where the exact tier must stop, the answer is a real interval bracket
# (``CInterval``), never a bare float:
#
#   * ``patch_flux_ci`` / ``trimmed_patch_flux_ci`` — RATIONAL patches.
#     The flux integrand of a rational patch is the rational function
#     F = det(A, A_u, A_v) / w³ (A, w the homogeneous numerator/weight),
#     both of whose tensor Bernstein forms have exact ℚ coefficients.
#     Per subdivision cell, when P := det(A,A_u,A_v) is one-signed the
#     cell's integral is enclosed by (exact ∫P)·[1/Q_max, 1/Q_min]
#     (convex-hull bounds of Q := w³ > 0); otherwise by range×area. The
#     sum over cells provably brackets the true flux — "certified ± e".
#
#   * ``certified_trim_flux`` — a boolean face's flux bracketed through
#     the surviving SSI subdivision cells. The cells provably enclose
#     the true trim curve (:mod:`forgekernel.ssi`'s completeness
#     guarantee), so every cell OFF the strip is wholly in or out of the
#     trimmed region (decided by one exact membership test), and every
#     cell ON the strip contributes its hull-bounded range as
#     uncertainty. The boolean volume then comes out "certified ± e"
#     instead of carrying the trim polyline's silent discretization
#     error (the honest K7 caveat in ``trimmed_patch_flux``).
#
# Exact stays the default: the ci entry points REFUSE polynomial input
# for which the exact functions above apply — ADR-0019 forbids silently
# downgrading an exact answer to an interval.
# =============================================================================

# -- tensor Bernstein forms: grid[i][j] exact ℚ coeffs, degrees (m, n) --------

def _bf_mul(A, B):
    """Product of two tensor Bernstein forms (exact ℚ)."""
    from math import comb
    m1, n1 = len(A) - 1, len(A[0]) - 1
    m2, n2 = len(B) - 1, len(B[0]) - 1
    M, N = m1 + m2, n1 + n2
    out = [[F(0)] * (N + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(N + 1):
            acc = F(0)
            for k in range(max(0, i - m2), min(m1, i) + 1):
                cu = F(comb(m1, k) * comb(m2, i - k), comb(M, i))
                for l in range(max(0, j - n2), min(n1, j) + 1):
                    cv = F(comb(n1, l) * comb(n2, j - l), comb(N, j))
                    acc += cu * cv * A[k][l] * B[i - k][j - l]
            out[i][j] = acc
    return out


def _bf_sub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _bf_add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _bf_du(A):
    """d/du of a B-form (u ∈ [0,1] local coordinates)."""
    m = len(A) - 1
    if m == 0:
        return [[F(0)] * len(A[0])]
    return [[m * (A[i + 1][j] - A[i][j]) for j in range(len(A[0]))]
            for i in range(m)]


def _bf_dv(A):
    n = len(A[0]) - 1
    if n == 0:
        return [[F(0)] for _ in range(len(A))]
    return [[n * (A[i][j + 1] - A[i][j]) for j in range(n)]
            for i in range(len(A))]


def _bf_mean(A):
    """Mean of the form over [0,1]² (= its exact integral): each basis
    function B_i^m·B_j^n integrates to 1/((m+1)(n+1))."""
    return sum(sum(r, F(0)) for r in A) / (len(A) * len(A[0]))


def _bf_hull(A):
    """(min, max) coefficient hull — bounds the form's range on [0,1]²
    (convex-hull property of the Bernstein basis)."""
    return (min(min(r) for r in A), max(max(r) for r in A))


def _split1d(col):
    """De Casteljau split of a coefficient list at t = 1/2 (exact)."""
    b = list(col)
    left, right = [b[0]], [b[-1]]
    for _ in range(1, len(b)):
        b = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
        left.append(b[0])
        right.insert(0, b[-1])
    return left, right


def _bf_split_u(A):
    per_col = [_split1d([A[i][j] for i in range(len(A))])
               for j in range(len(A[0]))]
    L = [[per_col[j][0][i] for j in range(len(A[0]))] for i in range(len(A))]
    R = [[per_col[j][1][i] for j in range(len(A[0]))] for i in range(len(A))]
    return L, R


def _bf_split_v(A):
    L, R = [], []
    for row in A:
        l, r = _split1d(row)
        L.append(l)
        R.append(r)
    return L, R


def _flux_forms(net):
    """(P, Q) tensor Bernstein forms of the flux integrand F = P/Q over a
    Bézier net in LOCAL [0,1]² coordinates. ``net`` entries are 3-tuples
    (polynomial → Q is None, F = det(S,S_u,S_v)) or homogeneous 4-tuples
    (rational → P = det(A,A_u,A_v), Q = w³; the identity F = P/w³ follows
    from S = A/w and det(A,·A,·) = 0). The LOCAL integral of F equals
    the patch's global flux contribution (parametrization invariance of
    the flux), so no domain-span factors are needed."""
    dim = len(net[0][0])
    comps = [[[F(pt[c]) for pt in row] for row in net] for c in range(dim)]
    # comps[c][i][j]; u runs along i, v along j
    Ax, Ay, Az = comps[0], comps[1], comps[2]
    Aux, Auy, Auz = _bf_du(Ax), _bf_du(Ay), _bf_du(Az)
    Avx, Avy, Avz = _bf_dv(Ax), _bf_dv(Ay), _bf_dv(Az)
    cx = _bf_sub(_bf_mul(Auy, Avz), _bf_mul(Auz, Avy))
    cy = _bf_sub(_bf_mul(Auz, Avx), _bf_mul(Aux, Avz))
    cz = _bf_sub(_bf_mul(Aux, Avy), _bf_mul(Auy, Avx))
    P = _bf_add(_bf_add(_bf_mul(Ax, cx), _bf_mul(Ay, cy)), _bf_mul(Az, cz))
    if dim == 3:
        return P, None
    W = comps[3]
    Q = _bf_mul(W, _bf_mul(W, W))
    qmn, _ = _bf_hull(Q)
    if qmn <= 0:
        raise ValueError(
            "certified flux: weight form is not provably one-signed "
            "(sign-varying weights arrive at K3.7)")
    return P, Q


def _is_rational(surface) -> bool:
    return any(w != F(1) for row in surface.w for w in row)


def _patch_forms(surface):
    """Per exact Bézier sub-patch of ``surface``: (global parameter box,
    P, Q) with F = P/Q in the sub-patch's local [0,1]² coordinates."""
    from forgekernel.nurbs import bezier_patches
    out = []
    for (u0, u1, v0, v1, net) in bezier_patches(surface):
        P, Q = _flux_forms(net)
        out.append(((F(u0), F(u1), F(v0), F(v1)), P, Q))
    return out


def _leaf_mean_ci(P, Q):
    """(lo, hi) bracketing the MEAN of F = P/Q over a leaf cell.
    Q is None → polynomial → exact (zero-width). Otherwise the per-cell
    reciprocal rule: P one-signed → (exact mean P)·[1/Qmax, 1/Qmin]."""
    if Q is None:
        m = _bf_mean(P)
        return m, m
    pmn, pmx = _bf_hull(P)
    qmn, qmx = _bf_hull(Q)
    if pmn >= 0:
        I = _bf_mean(P)
        return I / qmx, I / qmn
    if pmx <= 0:
        I = _bf_mean(P)
        return I / qmn, I / qmx
    return min(pmn / qmn, pmn / qmx), max(pmx / qmn, pmx / qmx)


def _leaf_range(P, Q):
    """(Fmin, Fmax) hull bound of F = P/Q on a leaf cell."""
    pmn, pmx = _bf_hull(P)
    if Q is None:
        return pmn, pmx
    qmn, qmx = _bf_hull(Q)
    cands = (pmn / qmn, pmn / qmx, pmx / qmn, pmx / qmx)
    return min(cands), max(cands)


def _accumulate(P, Q, box, depth, classify):
    """(lo, hi) bracketing the MEAN over ``box`` of the integrand
    restricted to the trimmed region. ``classify(box)`` returns:

      "out"   — box provably outside the region (contributes 0),
      "in"    — box provably inside (full integrand),
      "split" — the region boundary may cross the box.

    A "split" verdict at depth 0 becomes the hull boundary bound
    [min(0, Fmin), max(0, Fmax)] — the unknown covered fraction of the
    cell is bracketed by 0..1, which is what makes the result a rigorous
    enclosure rather than an estimate."""
    verdict = classify(box)
    if verdict == "out":
        return F(0), F(0)
    if verdict == "in":
        if Q is None:
            m = _bf_mean(P)                      # polynomial: exact, prune
            return m, m
        if depth == 0:
            return _leaf_mean_ci(P, Q)
    else:                                        # boundary
        if depth == 0:
            fmn, fmx = _leaf_range(P, Q)
            return min(F(0), fmn), max(F(0), fmx)
    u0, u1, v0, v1 = box
    um, vm = (u0 + u1) / 2, (v0 + v1) / 2
    PL, PR = _bf_split_u(P)
    QL = QR = None
    if Q is not None:
        QL, QR = _bf_split_u(Q)
    lo = hi = F(0)
    for (Ph, Qh, ub) in ((PL, QL, (u0, um)), (PR, QR, (um, u1))):
        PB, PT = _bf_split_v(Ph)
        QB = QT = None
        if Qh is not None:
            QB, QT = _bf_split_v(Qh)
        for (Pq, Qq, vb) in ((PB, QB, (v0, vm)), (PT, QT, (vm, v1))):
            l, h = _accumulate(Pq, Qq, (ub[0], ub[1], vb[0], vb[1]),
                               depth - 1, classify)
            lo += l
            hi += h
    return lo / 4, hi / 4


def _sum_ci(surface, depth, classify):
    from forgekernel.interval import CInterval
    lo = hi = F(0)
    for (box, P, Q) in _patch_forms(surface):
        l, h = _accumulate(P, Q, box, depth, classify)
        lo += l
        hi += h
    return CInterval(lo / 3, hi / 3)


def patch_flux_ci(surface: BSplineSurface, depth: int = 4):
    """Certified flux of a RATIONAL patch: a ``CInterval`` that provably
    brackets (1/3)∮∮ S·(S_u×S_v) du dv — the K7.1 answer where
    :func:`patch_flux` refuses. Deeper ``depth`` tightens the bracket
    (per-cell reciprocal rule; width is O(2^-depth)).

    Refuses polynomial patches: the exact tier applies there and
    ADR-0019 forbids silently downgrading exact to an interval."""
    if not _is_rational(surface):
        raise ValueError(
            "polynomial patch: exact patch_flux applies — refusing to "
            "downgrade an exact answer to an interval (ADR-0019)")
    return _sum_ci(surface, depth, lambda box: "in")


def _parity_in(loops, u, v) -> bool:
    """Even-odd parity of a +u ray — exact ℚ (the trim.classify rule)."""
    inside = False
    for loop in loops:
        m = len(loop)
        for k in range(m):
            a, b = loop[k], loop[(k + 1) % m]
            if (a[1] > v) != (b[1] > v):
                x = a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if u < x:
                    inside = not inside
    return inside


def _seg_cuts_box(a, b, box) -> bool:
    """May the closed segment a→b pass through the OPEN box? Exact for
    axis-parallel segments; conservative (never a false "no") for
    general ones. A segment lying on the box's own gridlines does not
    cut the open box — so trim edges on cell skeleta cost nothing."""
    u0, u1, v0, v1 = box
    (au, av), (bu, bv) = a, b
    if au == bu:                                  # vertical: exact
        return u0 < au < u1 and min(av, bv) < v1 and max(av, bv) > v0
    if av == bv:                                  # horizontal: exact
        return v0 < av < v1 and min(au, bu) < u1 and max(au, bu) > u0
    if max(au, bu) <= u0 or min(au, bu) >= u1:    # provably beyond a face
        return False
    if max(av, bv) <= v0 or min(av, bv) >= v1:
        return False
    signs = set()
    for (cu, cv) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        d = (bu - au) * (cv - av) - (bv - av) * (cu - au)
        signs.add((d > 0) - (d < 0))
    if 0 not in signs and len(signs) == 1:
        return False                              # box strictly one side
    return True


def trimmed_patch_flux_ci(surface: BSplineSurface, loops, depth: int = 5):
    """Certified trimmed flux of a RATIONAL patch over the region bounded
    by polygonal ``loops`` (even-odd, exact ℚ vertices) — the K7.1
    counterpart of :func:`trimmed_patch_flux`, which refuses rational
    surfaces. Returns a ``CInterval`` bracket: cells wholly inside use
    the per-cell reciprocal rule, boundary-straddling cells contribute
    their hull range as uncertainty (width is O(2^-depth)).

    Refuses polynomial patches (exact tier applies, ADR-0019)."""
    if not _is_rational(surface):
        raise ValueError(
            "polynomial patch: exact trimmed_patch_flux applies — "
            "refusing to downgrade an exact answer to an interval "
            "(ADR-0019)")
    lq = [[(F(u), F(v)) for (u, v) in loop] for loop in loops]

    def classify(box):
        for loop in lq:
            m = len(loop)
            for k in range(m):
                if _seg_cuts_box(loop[k], loop[(k + 1) % m], box):
                    return "split"
        u0, u1, v0, v1 = box
        um, vm = (u0 + u1) / 2, (v0 + v1) / 2
        return "in" if _parity_in(lq, um, vm) else "out"

    return _sum_ci(surface, depth, classify)


def _strip_classify(strip, inside):
    """The cell classifier a certified boolean-face bracket runs on:
    boxes overlapping the SSI strip are "split" (their covered fraction
    is unknown), boxes off the strip are decided by one exact membership
    test at the midpoint — sound because membership is constant on every
    strip-free connected region (the strip encloses the whole trim
    boundary)."""
    sq = [tuple(F(c) for c in cell) for cell in strip]

    def classify(box):
        u0, u1, v0, v1 = box
        for (s0, s1, t0, t1) in sq:
            if s0 < u1 and u0 < s1 and t0 < v1 and v0 < t1:
                return "split"                    # strip overlaps interior
        um, vm = (u0 + u1) / 2, (v0 + v1) / 2
        return "in" if inside(um, vm) else "out"

    return classify


def certified_trim_flux(surface: BSplineSurface, strip, inside,
                        depth: int = 4):
    """Flux of a boolean face bracketed through the SSI strip — the
    certified answer to "what is the trimmed flux when the trim boundary
    is only known as an enclosure?" (ADR-0019: "certified ± e", never a
    polyline's silent discretization error).

    ``strip`` is the set of surviving SSI subdivision cells on THIS
    surface's parameter domain (``forgekernel.ssi.ssi_strips``), as
    (u0, u1, v0, v1) boxes; the true trim curve provably lies inside
    their union (subdivision completeness). ``inside(u, v)`` is an exact
    membership predicate for points OFF the strip (e.g. which side of
    the other solid); it must be constant on every strip-free connected
    region — which the enclosure property guarantees whenever the strip
    covers the whole trim boundary in the open domain (transversal
    closed loops; open/tangent branches must refuse upstream).

    Cells off the strip are integrated exactly (polynomial) or by the
    reciprocal rule (rational); cells touching the strip contribute
    their hull-bounded range as uncertainty. Bracket width is
    O(2^-depth)·|F|·(strip length). Works for polynomial AND rational
    surfaces — nothing exact exists for a strip-bounded region, so
    there is no exact tier to defend here."""
    return _sum_ci(surface, depth, _strip_classify(strip, inside))


def _normal_forms(net):
    """((Px, Py, Pz), Q) tensor Bernstein forms with S_u×S_v = P/Q in
    LOCAL [0,1]² coordinates. Polynomial nets: the cross-product forms
    directly, Q None. Rational nets (homogeneous 4-tuples): S = A/w
    gives S_u×S_v = [(A_u w − A w_u) × (A_v w − A w_v)] / w⁴, so P is
    that cross form and Q = w⁴ (provably positive, checked)."""
    dim = len(net[0][0])
    comps = [[[F(pt[c]) for pt in row] for row in net] for c in range(dim)]
    if dim == 3:
        Du = [_bf_du(c) for c in comps[:3]]
        Dv = [_bf_dv(c) for c in comps[:3]]
        Q = None
    else:
        W = comps[3]
        Wu, Wv = _bf_du(W), _bf_dv(W)
        Du = [_bf_sub(_bf_mul(_bf_du(c), W), _bf_mul(c, Wu))
              for c in comps[:3]]
        Dv = [_bf_sub(_bf_mul(_bf_dv(c), W), _bf_mul(c, Wv))
              for c in comps[:3]]
        W2 = _bf_mul(W, W)
        Q = _bf_mul(W2, W2)
        qmn, _ = _bf_hull(Q)
        if qmn <= 0:
            raise ValueError(
                "certified vector area: weight form is not provably "
                "one-signed (sign-varying weights arrive at K3.7)")
    P = (_bf_sub(_bf_mul(Du[1], Dv[2]), _bf_mul(Du[2], Dv[1])),
         _bf_sub(_bf_mul(Du[2], Dv[0]), _bf_mul(Du[0], Dv[2])),
         _bf_sub(_bf_mul(Du[0], Dv[1]), _bf_mul(Du[1], Dv[0])))
    return P, Q


def certified_vector_area(surface: BSplineSurface, strip, inside,
                          depth: int = 4):
    """∮∮_D S_u×S_v du dv over a boolean face's trimmed region D,
    bracketed through the SSI strip — a triple of ``CInterval``.

    This is the Σ-flux orientation oracle for a closed shell of trimmed
    faces (:class:`forgekernel.trimshell.TrimmedShell`): ∮ n̂ dA over ANY
    closed surface is exactly zero (divergence theorem on the constant
    fields x̂, ŷ, ẑ), so the per-face brackets summed with the faces'
    senses must CONTAIN (0, 0, 0) — an interval that certifiably
    excludes zero is a proven orientation or membership error, while a
    true shell can never fail (each bracket contains its true value).
    Same cell rules as :func:`certified_trim_flux`: strip-free cells are
    exact (polynomial) or reciprocal-ruled (rational), strip cells
    contribute their hull range as uncertainty."""
    from forgekernel.interval import CInterval
    from forgekernel.nurbs import bezier_patches

    classify = _strip_classify(strip, inside)
    spans = []
    for (u0, u1, v0, v1, net) in bezier_patches(surface):
        P, Q = _normal_forms(net)
        spans.append(((F(u0), F(u1), F(v0), F(v1)), P, Q))
    out = []
    for c in range(3):
        lo = hi = F(0)
        for (box, P, Q) in spans:
            l, h = _accumulate(P[c], Q, box, depth, classify)
            lo += l
            hi += h
        out.append(CInterval(lo, hi))
    return tuple(out)
