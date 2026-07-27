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
    polynomial Bézier patches. Volume exact in ℚ via the flux theorem.

    ``seams`` (optional) is the operand's exact side-gluing topology:
    a tuple of ``((fi, si), (fj, sj), flip)`` records meaning side
    ``si`` of face ``fi`` is the SAME 3D curve as side ``sj`` of face
    ``fj`` (control points exactly equal, reversed when ``flip``); a
    side is one of ``"u0" | "u1" | "v0" | "v1"``. The boolean assembly
    needs it to pair trim edges that leave a face through its border
    (open SSI branches); a converter that knows its topology exactly
    (:meth:`forgekernel.loft.LoftSolid.to_patches`) emits it, and
    ``None`` simply means open-branch booleans refuse on this operand."""

    provenance = "exact"

    def __init__(self, patches, seams=None) -> None:
        self.patches = list(patches)
        self.seams = seams
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


# -- K7 gap 5: boolean assembly — two patch solids to one audited shell -------
#
# boolean_trimmed(op, A, B): per face-pair SSI → certified trim loops →
# exact partition of each face's parameter domain → certified
# point-membership classification of every strip-free fragment → stitch
# and orient into a TrimmedShell → full three-oracle audit. Every
# topological decision is certified or refused by name; a wrong shell is
# the only failure mode this module is not allowed to have.
#
# ORIENTATION IS PRINCIPLED, NOT ENUMERATED. Each operand face is
# outward-oriented for its own solid (preflighted: Σ patch flux > 0
# exactly). The result's outward normal on a kept fragment follows from
# the monotonicity of the boolean's membership function alone:
#
#   union      m = m_A ∨ m_B    monotone increasing in m_A and m_B
#   intersect  m = m_A ∧ m_B    monotone increasing in m_A and m_B
#   cut        m = m_A ∧ ¬m_B   increasing in m_A, DECREASING in m_B
#
# A kept fragment of ∂A bounds the result where m_B is locally constant,
# so stepping along A's outward normal flips m_A true→false and (by
# monotonicity in m_A) exits the result: the result's outward normal is
# A's own — sense +1, for all three ops. On a kept fragment of ∂B the
# same step flips m_B true→false; for union/intersect that exits the
# result (sense +1), for cut it ENTERS it (m = m_A ∧ ¬m_B rises), so the
# face flips — sense −1. That is the whole sign rule; the derivation
# that had to try all four sign combinations by hand is retired.
#
# The loop ROLE (does the kept region lie inside the loop — outer, CCW —
# or outside it — hole, CW) orients the pairing audit's traversal. A
# role mistake can never ship a wrong shell: the certified measures
# (flux, vector area) never read roles, and the pairing audit fails on a
# single flipped role while a consistent double flip reverses both
# traversals and stays opposite — refusal or no-op, never a lie.

class BooleanUnsupported(ValueError):
    """Structured refusal: a boolean configuration stage 2 cannot turn
    into a *certified* shell. ``predicate`` names the guard that fired:

    * ``rational_operand`` / ``multispan_operand`` — untested operand
      representations (certified flux exists per face — K7.1 — but the
      assembly over them is unproven);
    * ``operand_not_outward`` — Σ patch flux ≤ 0: the operand's faces
      are not consistently outward-oriented, so the sign rule above has
      no premise to stand on;
    * ``open_branch`` — an intersection curve leaves a face through its
      border and an operand it crosses carries NO seam topology
      (``PatchSolid.seams is None``): border segments can only be paired
      with the operand's own seam edges when those seams are declared
      and proven (:meth:`forgekernel.loft.LoftSolid.to_patches` emits
      them);
    * ``seam_topology_invalid`` / ``seam_mismatch`` — a declared seam
      set does not cover every face side exactly once, or a record's two
      sides do not carry exactly equal 3D control points (the equality
      the certified-point transfer leans on);
    * ``seam_crossing_unmatched`` / ``seam_crossing_ambiguous`` — the
      two per-face-pair solves of one seam crossing cannot be identified
      1:1 (count mismatch, different continuing face, or crossings
      closer than the detection resolution can separate);
    * ``assembly_degenerate`` — two distinct trim vertices landed on the
      same exact parameter point of one face;
    * ``region_unresolved`` / ``nested_loops_unresolved`` — the dyadic
      grid at this depth has no strip-free witness cell inside a trim
      loop, or mixed certified memberships inside one loop (nested
      curves); raise the depth;
    * ``trim_loops_cross`` — the exact domain partition refused the
      loop arrangement;
    * ``empty_result`` — no face fragment survives; the empty solid has
      no TrimmedShell representation.

    Tangent contact and uncertifiable cells refuse upstream with their
    own names (:class:`forgekernel.ssi.SsiCellUncertified`,
    :class:`forgekernel.ssi.TrimLoopUnstitchable`,
    :class:`forgekernel.raycast.PointClassifyUncertified`)."""

    def __init__(self, predicate: str, detail: str) -> None:
        self.predicate = predicate
        super().__init__(f"boolean_unsupported[{predicate}]: {detail}")


# op → (keep A-fragments inside B?, keep B-fragments inside A?)
_BOOL_KEEP_INSIDE = {"union": (False, False),
                     "intersect": (True, True),
                     "cut": (False, True)}
# op → sense of kept B-faces (A-faces are always +1); see the derivation
_BOOL_B_SENSE = {"union": 1, "intersect": 1, "cut": -1}


def untrimmed_shell(solid):
    """A :class:`~forgekernel.bsolid.PatchSolid` (or bare patch list)
    wrapped as a :class:`~forgekernel.trimshell.TrimmedShell` of whole,
    untrimmed faces — the certified ray-parity membership target
    (:func:`forgekernel.raycast.classify_point_in_shell`) for an operand
    during boolean assembly, and for MC membership oracles in tests."""
    from forgekernel.trimshell import ShellFace, TrimmedShell

    faces = list(solid.patches) if hasattr(solid, "patches") else list(solid)
    return TrimmedShell([ShellFace(f, 1, set(), lambda u, v: True)
                         for f in faces])


def _operand_faces(name: str, solid):
    """Preflight one operand: polynomial single-span patches whose flux
    sum is certifiably outward (> 0, exact ℚ). Anything else refuses by
    name — the assembly is only proven over this class."""
    from forgekernel.nurbs import bezier_patches

    faces = list(solid.patches)
    for i, f in enumerate(faces):
        if _is_rational(f):
            raise BooleanUnsupported(
                "rational_operand",
                f"operand {name} face {i} has rational weights — the "
                f"certified flux exists per face (K7.1) but the boolean "
                f"assembly over rational operands is untested; refusing "
                f"rather than risk a wrong shell")
        if len(bezier_patches(f)) != 1:
            raise BooleanUnsupported(
                "multispan_operand",
                f"operand {name} face {i} has multiple Bézier spans — "
                f"the SSI strip grid and trim-loop bookkeeping are only "
                f"proven over single-span faces; split the operand into "
                f"per-span faces first")
    total = sum((patch_flux(f) for f in faces), F(0))
    if total <= 0:
        raise BooleanUnsupported(
            "operand_not_outward",
            f"operand {name}: Σ patch flux = {total} ≤ 0 — the faces are "
            f"not consistently outward-oriented, so the boolean sign rule "
            f"has no premise; fix the operand's face orientations")
    return faces


class _FaceGrid:
    """Strip-free connected components of one face's parameter domain on
    the dyadic 2^-depth grid — the exact scaffolding a face's certified
    membership predicate memoizes over.

    The SSI strip encloses every trim curve on the face (subdivision
    completeness), so membership in the other solid is CONSTANT on each
    connected component of the strip's complement: one certified
    ray-parity verdict at a deep-interior representative decides the
    whole component. 4-adjacency is used for connectivity — conservative
    (a corner-touching component may split in two), which costs at most
    one extra certified raycast and can never mislabel a point."""

    def __init__(self, surface, strip, depth: int) -> None:
        (u0, u1), (v0, v1) = surface.domain()
        self.u0, self.v0 = F(u0), F(v0)
        n = 1 << depth
        self.n = n
        self.wu = (F(u1) - F(u0)) / n
        self.wv = (F(v1) - F(v0)) / n
        marked = set()
        for (s0, s1, t0, t1) in strip:
            iu = (F(s0) - self.u0) / self.wu
            iv = (F(t0) - self.v0) / self.wv
            if iu.denominator != 1 or iv.denominator != 1 \
                    or F(s1) - F(s0) != self.wu or F(t1) - F(t0) != self.wv:
                raise AssertionError(
                    "SSI strip cell is not aligned to the dyadic grid — "
                    "strip depth and grid depth must match")
            marked.add((int(iu), int(iv)))
        self.marked = marked

        comp: dict = {}
        cid = 0
        for i in range(n):
            for j in range(n):
                if (i, j) in marked or (i, j) in comp:
                    continue
                stack = [(i, j)]
                comp[(i, j)] = cid
                while stack:
                    ci, cj = stack.pop()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (ci + di, cj + dj)
                        if 0 <= q[0] < n and 0 <= q[1] < n \
                                and q not in marked and q not in comp:
                            comp[q] = cid
                            stack.append(q)
                cid += 1
        self.comp = comp
        self.ncomp = cid

        # deep-interior representative per component: the cell furthest
        # (in grid BFS steps) from the strip — the sample a certified
        # raycast is least likely to refuse on
        from collections import deque

        dist: dict = {}
        dq = deque()
        for c in marked:
            dist[c] = 0
            dq.append(c)
        while dq:
            ci, cj = dq.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (ci + di, cj + dj)
                if 0 <= q[0] < n and 0 <= q[1] < n and q not in dist:
                    dist[q] = dist[(ci, cj)] + 1
                    dq.append(q)
        best: dict = {}
        by_comp: dict = {}
        for cell, c in comp.items():
            d = dist.get(cell, 2 * n)
            by_comp.setdefault(c, []).append((d, cell))
            if c not in best or d > best[c][0]:
                best[c] = (d, cell)
        self.rep = {c: cell for c, (d, cell) in best.items()}
        self._by_comp = {c: [cell for _, cell in sorted(lst, reverse=True)]
                         for c, lst in by_comp.items()}

    def candidates(self, c, k: int = 4):
        """Up to ``k`` witness cells of component ``c``, deepest-interior
        first — fallbacks for a certified raycast that refuses on a
        degenerate first sample (any strip-free cell of the component is
        an equally valid witness: membership is constant on it)."""
        return self._by_comp[c][:k]

    def cell_mid(self, cell):
        i, j = cell
        return (self.u0 + self.wu * F(2 * i + 1, 2),
                self.v0 + self.wv * F(2 * j + 1, 2))

    def comp_of(self, u, v):
        """Component id of the grid cell containing (u, v), or ``None``
        for a strip-marked cell. A query on a gridline takes the
        upper-right cell — the right convention for the midpoints the
        flux accumulator asks about (their cell lies inside the queried
        strip-free box)."""
        iu = min(self.n - 1, int((F(u) - self.u0) / self.wu))
        iv = min(self.n - 1, int((F(v) - self.v0) / self.wv))
        return self.comp.get((iu, iv))


def _assemble_face(surface, sense, strip, chains, keep_inside, other_shell,
                   depth, raycast_depth):
    """One operand face → its kept fragment as a
    :class:`~forgekernel.trimshell.ShellFace`, or ``None`` if the whole
    face is discarded. ``chains`` is ``[(verts, uvs), …]`` — the face's
    certified trim loops as shared TrimVertex objects plus this face's
    own exact (u, v) coordinates."""
    from forgekernel.raycast import (PointClassifyUncertified,
                                     classify_point_in_shell)
    from forgekernel.trim import split_trim_region
    from forgekernel.trimshell import ShellFace

    def kept_at(u, v):
        p = surface.eval(u, v)
        verdict = classify_point_in_shell(other_shell, p,
                                          depth=raycast_depth)
        return (verdict == "in") == keep_inside

    def kept_at_any(cands):
        """First certified verdict over equally-valid witness points —
        membership is constant on the region they sample, so any
        certified answer is THE answer; refuse only when all refuse."""
        last = None
        for (u, v) in cands:
            try:
                return kept_at(u, v)
            except PointClassifyUncertified as exc:
                last = exc
        raise last

    if not strip:
        # no intersection touches this face: it is entirely in or out of
        # the other solid — one certified verdict at an interior point
        # (midpoint first; asymmetric prime-ratio fallbacks break the
        # degeneracies a symmetric sample can hit against aligned faces)
        (u0, u1), (v0, v1) = surface.domain()
        du, dv = F(u1) - F(u0), F(v1) - F(v0)
        cands = [(F(u0) + du * a, F(v0) + dv * b)
                 for a, b in ((F(1, 2), F(1, 2)), (F(9, 17), F(8, 19)),
                              (F(7, 23), F(13, 29)), (F(19, 31), F(11, 37)))]
        if not kept_at_any(cands):
            return None
        return ShellFace(surface, sense, set(), lambda u, v: True)

    # structural cross-check: the loops must form a legal arrangement —
    # exact partition of the domain, refusing crossings and degeneracies
    try:
        split_trim_region(surface, [uvs for _, uvs in chains])
    except ValueError as exc:
        raise BooleanUnsupported("trim_loops_cross", str(exc)) from exc

    grid = _FaceGrid(surface, strip, depth)
    verdicts: dict = {}

    def comp_kept(c):
        if c not in verdicts:
            verdicts[c] = kept_at_any(
                [grid.cell_mid(cell) for cell in grid.candidates(c)])
        return verdicts[c]

    # loop roles: the kept side of each loop, certified per component
    roles = []
    for _, uvs in chains:
        inside_vals = set()
        for c in range(grid.ncomp):
            um, vm = grid.cell_mid(grid.rep[c])
            if _parity_in([uvs], um, vm):
                inside_vals.add(comp_kept(c))
        if not inside_vals:
            raise BooleanUnsupported(
                "region_unresolved",
                f"no strip-free grid cell inside a trim loop at depth "
                f"{depth} — the loop's interior is thinner than the grid; "
                f"raise the depth")
        if len(inside_vals) > 1:
            raise BooleanUnsupported(
                "nested_loops_unresolved",
                "mixed certified memberships inside one trim loop — "
                "nested intersection curves are untested; refusing "
                "rather than guess the arrangement")
        roles.append(inside_vals.pop())

    if not any(comp_kept(c) for c in range(grid.ncomp)):
        return None

    point_memo: dict = {}

    def inside(u, v):
        c = grid.comp_of(u, v)
        if c is not None:
            return comp_kept(c)
        # a query finer than the grid whose cell is strip-marked (e.g. a
        # raycast leaf midpoint): certify the point itself, memoized
        key = (F(u), F(v))
        if key not in point_memo:
            point_memo[key] = kept_at(*key)
        return point_memo[key]

    face = ShellFace(surface, sense, strip, inside)
    for (verts, uvs), role in zip(chains, roles):
        face.add_loop(list(zip(verts, uvs)), outer=role)
    return face


# -- open branches: seam topology, crossing canonicalization, chain gluing ----
#
# A curve that leaves a face through its border continues on the
# adjacent face of the SAME operand. Each side detects its own certified
# border crossing (a separate Newton solve), so the same 3D crossing
# exists twice with slightly different exact coordinates. The assembly
# CANONICALIZES: one solve is kept, and its coordinates are transferred
# to the adjacent face through the seam's exact control-point equality —
# S_adj(mapped uv) == S_own(uv) EXACTLY, so the kept residual
# certificate (|S_own(uv) − S_other(s,t)|² < 1e-20) applies verbatim to
# the adjacent face. Identity becomes the object: one TrimVertex per
# crossing, bound to three faces.

_SIDE_NAMES = ("u0", "u1", "v0", "v1")


def _side_corner_bits(side, which):
    """Corner (u-bit, v-bit) at the ``which`` end (0 = low border
    parameter, 1 = high) of a face side."""
    return {"u0": (0, which), "u1": (1, which),
            "v0": (which, 0), "v1": (which, 1)}[side]


class _SeamTopology:
    """One operand's validated seam topology for a boolean assembly.

    Construction PROVES the declared records against the control nets:
    every face side glued exactly once, and each record's two side
    curves carry exactly equal control points (reversed under ``flip``).
    Provides the exact affine border-parameter maps, and the operand's
    shared corner vertices (union-find over seam-end incidences — one
    :class:`~forgekernel.trimshell.TrimVertex` per geometric corner)."""

    def __init__(self, name, faces, seams) -> None:
        from forgekernel.loft import _side_cps
        from forgekernel.trimshell import TrimVertex

        self.name = name
        self.faces = faces
        self.side_map: dict = {}
        for (ka, kb, flip) in seams:
            for k, other in ((ka, kb), (kb, ka)):
                if k in self.side_map:
                    raise BooleanUnsupported(
                        "seam_topology_invalid",
                        f"operand {name}: side {k} appears in more than "
                        f"one seam record")
                self.side_map[k] = (other, bool(flip))
        for fi in range(len(faces)):
            for s in _SIDE_NAMES:
                if (fi, s) not in self.side_map:
                    raise BooleanUnsupported(
                        "seam_topology_invalid",
                        f"operand {name}: side ({fi}, {s!r}) is glued to "
                        f"nothing — a closed seamed skin covers every "
                        f"side exactly once")
        for (ka, kb, flip) in seams:
            ca = _side_cps(faces[ka[0]], ka[1])
            cb = _side_cps(faces[kb[0]], kb[1])
            if flip:
                cb = tuple(reversed(cb))
            if tuple(ca) != tuple(cb):
                raise BooleanUnsupported(
                    "seam_mismatch",
                    f"operand {name}: seam {ka} ↔ {kb} sides do not carry "
                    f"exactly equal 3D control points — the certified "
                    f"transfer across this seam has no exact premise")

        # shared corner vertices: union-find over seam end incidences
        parent: dict = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (ka, kb, flip) in seams:
            for which in (0, 1):
                mate_which = which if not flip else 1 - which
                union((ka[0], _side_corner_bits(ka[1], which)),
                      (kb[0], _side_corner_bits(kb[1], mate_which)))
        self._find = find
        self._corner_vx: dict = {}
        self._TrimVertex = TrimVertex

    def _dom(self, fi):
        (a, b), (c, d) = self.faces[fi].domain()
        return F(a), F(b), F(c), F(d)

    def corner_uv(self, fi, bits):
        u0, u1, v0, v1 = self._dom(fi)
        return ((u0, u1)[bits[0]], (v0, v1)[bits[1]])

    def corner_vertex(self, fi, bits):
        root = self._find((fi, bits))
        vx = self._corner_vx.get(root)
        if vx is None:
            vx = self._corner_vx[root] = self._TrimVertex()
        return vx

    def side_range(self, fi, side):
        u0, u1, v0, v1 = self._dom(fi)
        return (v0, v1) if side in ("u0", "u1") else (u0, u1)

    def side_tau(self, side, uv):
        return uv[1] if side in ("u0", "u1") else uv[0]

    def side_point(self, fi, side, tau):
        u0, u1, v0, v1 = self._dom(fi)
        return {"u0": (u0, tau), "u1": (u1, tau),
                "v0": (tau, v0), "v1": (tau, v1)}[side]

    def mate(self, fi, side, tau):
        """The glued side and the EXACT corresponding border parameter."""
        (fj, sj), flip = self.side_map[(fi, side)]
        lo, hi = self.side_range(fi, side)
        t = (F(tau) - lo) / (hi - lo)
        if flip:
            t = 1 - t
        lo2, hi2 = self.side_range(fj, sj)
        return (fj, sj), lo2 + t * (hi2 - lo2)

    def canon_key(self, fi, side):
        other, _ = self.side_map[(fi, side)]
        a, b = (fi, side), other
        return (a, b) if a <= b else (b, a)

    def canon_tau(self, fi, side, tau):
        """Border parameter normalized into the seam's slot-0 side."""
        lo, hi = self.side_range(fi, side)
        t = (F(tau) - lo) / (hi - lo)
        key = self.canon_key(fi, side)
        if (fi, side) != key[0]:
            _, flip = self.side_map[(fi, side)]
            if flip:
                t = 1 - t
        return t


def _canonicalize_crossings(open_recs, topos, depth: int) -> None:
    """Identify the two per-face-pair solves of every seam crossing and
    merge them into ONE canonical vertex + coordinates (mutates the
    chain records in place).

    The slot-0 solve is kept; the slot-1 chain's endpoint is rewritten
    to the seam-mapped coordinates (exact) and to the slot-0 solve's
    other-operand coordinates, and its endpoint vertex is replaced by
    the slot-0 vertex object. Soundness: the seam sides carry exactly
    equal control points, so S_slot1(mapped uv) == S_slot0(uv) exactly
    and the slot-0 residual certificate transfers verbatim. Matching is
    1:1 within each seam, refusing count mismatches, different
    continuing faces, matches wider than the detection resolution, and
    crossings the resolution cannot separate."""
    for X, own in (("A", 0), ("B", 1)):
        topo = topos[X]
        if topo is None:
            continue
        oth = 1 - own
        entries: dict = {}
        for rec in open_recs:
            for e in (0, 1):
                end = rec["ends"][e]
                if end is None or end[0] != X:
                    continue
                fi = rec["pair"][own]
                pt = rec["pts"][0 if e == 0 else -1]
                uv = (pt[0], pt[1]) if X == "A" else (pt[2], pt[3])
                side = end[1]
                tau = topo.side_tau(side, uv)
                key = topo.canon_key(fi, side)
                slot = 0 if (fi, side) == key[0] else 1
                entries.setdefault(key, []).append(
                    (topo.canon_tau(fi, side, tau), slot, rec, e, tau,
                     fi, side))
        w = F(1, 1 << depth)
        for key, ents in entries.items():
            s0 = sorted((x for x in ents if x[1] == 0), key=lambda x: x[0])
            s1 = sorted((x for x in ents if x[1] == 1), key=lambda x: x[0])
            if len(s0) != len(s1):
                raise BooleanUnsupported(
                    "seam_crossing_unmatched",
                    f"operand {X} seam {key}: {len(s0)} crossing(s) "
                    f"certified from one side, {len(s1)} from the other "
                    f"— the curve's continuation across the seam was not "
                    f"detected; raise the depth")
            for k, (e0, e1) in enumerate(zip(s0, s1)):
                if abs(e0[0] - e1[0]) > 2 * w:
                    raise BooleanUnsupported(
                        "seam_crossing_unmatched",
                        f"operand {X} seam {key}: nearest crossing "
                        f"candidates are {float(abs(e0[0] - e1[0])):.3g} "
                        f"apart in the seam parameter — beyond the "
                        f"2·2^-{depth} identification bound")
                if k + 1 < len(s0) and (
                        s0[k + 1][0] - e0[0] <= 4 * w
                        or s1[k + 1][0] - e1[0] <= 4 * w):
                    raise BooleanUnsupported(
                        "seam_crossing_ambiguous",
                        f"operand {X} seam {key}: two crossings closer "
                        f"than 4·2^-{depth} in the seam parameter — the "
                        f"1:1 identification is not certified; raise "
                        f"the depth")
                _, _, rec0, i0, tau0, f0, side0 = e0
                _, _, rec1, i1, _, f1, side1 = e1
                if rec0["pair"][oth] != rec1["pair"][oth]:
                    raise BooleanUnsupported(
                        "seam_crossing_unmatched",
                        f"operand {X} seam {key}: matched crossing ends "
                        f"lie on different faces of the other operand "
                        f"({rec0['pair'][oth]} vs {rec1['pair'][oth]})")
                (fj, sj), tau_m = topo.mate(f0, side0, tau0)
                if (fj, sj) != (f1, side1):
                    raise BooleanUnsupported(
                        "seam_crossing_unmatched",
                        f"operand {X} seam {key}: slot-1 end sits on side "
                        f"({f1}, {side1!r}), expected ({fj}, {sj!r})")
                uvj = topo.side_point(f1, side1, tau_m)
                p0 = rec0["pts"][0 if i0 == 0 else -1]
                if X == "A":
                    newpt = (uvj[0], uvj[1], p0[2], p0[3])
                else:
                    newpt = (p0[0], p0[1], uvj[0], uvj[1])
                idx = 0 if i1 == 0 else -1
                rec1["pts"][idx] = newpt
                rec1["verts"][idx] = rec0["verts"][0 if i0 == 0 else -1]


def _face_paths(side_idx: int, face_index: int, recs):
    """One face's trim curves, glued: chains of the face concatenated at
    shared canonical endpoint vertices (crossings of the OTHER operand's
    seams, interior to this face). Returns ``(paths, cycles)``, each
    entry ``(verts, uvs)`` in this face's own parameters — paths end on
    THIS operand's borders, cycles are closed."""
    X = "A" if side_idx == 0 else "B"
    mine = [r for r in recs if r["pair"][side_idx] == face_index]

    def coords(r):
        return [(p[0], p[1]) if side_idx == 0 else (p[2], p[3])
                for p in r["pts"]]

    cycles = []
    open_items = []
    for r in mine:
        if r["closed"]:
            cycles.append((list(r["verts"]), coords(r)))
        else:
            open_items.append(r)
    if not open_items:
        return [], cycles

    by_id = {id(r): r for r in open_items}
    glue: dict = {}
    for r in open_items:
        for e in (0, 1):
            if r["ends"][e][0] != X:            # other operand's seam:
                vid = id(r["verts"][0 if e == 0 else -1])
                glue.setdefault(vid, []).append((id(r), e))
    for vid, uses in glue.items():
        if len(uses) != 2:
            raise BooleanUnsupported(
                "seam_crossing_unmatched",
                f"a glued crossing vertex is used by {len(uses)} chain "
                f"end(s) on one face, not 2 — the curve's continuation "
                f"across the other operand's seam is incomplete")

    used: set = set()

    def walk(rid, e, stop_vid=None):
        """Traverse from end ``e`` of chain ``rid`` gluing across shared
        vertices; stops at a terminal (own-border) end or when the walk
        returns to ``stop_vid`` (cycle). Returns (verts, uvs, closed)."""
        seq_v: list = []
        seq_p: list = []
        while True:
            r = by_id[rid]
            used.add(rid)
            pts = coords(r)
            verts = list(r["verts"])
            if e == 1:
                pts = list(reversed(pts))
                verts = list(reversed(verts))
            if seq_p and seq_p[-1] == pts[0]:
                pts, verts = pts[1:], verts[1:]
            seq_p += pts
            seq_v += verts
            exit_e = 1 - e
            end = r["ends"][exit_e]
            exit_vert = r["verts"][0 if exit_e == 0 else -1]
            if end[0] == X:
                return seq_v, seq_p, False
            if stop_vid is not None and id(exit_vert) == stop_vid:
                # cycle closed: drop the repeated closing vertex
                if seq_v and seq_v[-1] is seq_v[0]:
                    seq_v, seq_p = seq_v[:-1], seq_p[:-1]
                return seq_v, seq_p, True
            nxt = [u for u in glue[id(exit_vert)] if u != (rid, exit_e)]
            rid2, e2 = nxt[0]
            if rid2 in used and stop_vid is None:
                raise BooleanUnsupported(
                    "seam_crossing_unmatched",
                    "chain gluing revisited a chain while walking from a "
                    "border terminal — inconsistent crossing topology")
            rid, e = rid2, e2

    paths = []
    for r in open_items:
        if id(r) in used:
            continue
        term = next((e for e in (0, 1) if r["ends"][e][0] == X), None)
        if term is not None:
            verts, uvs, _ = walk(id(r), term)
            paths.append((verts, uvs))
    for r in open_items:
        if id(r) not in used:
            start_vid = id(r["verts"][0])
            verts, uvs, closed = walk(id(r), 0, stop_vid=start_vid)
            if not closed:
                raise BooleanUnsupported(
                    "seam_crossing_unmatched",
                    "leftover glued chains neither reach a border nor "
                    "close a cycle")
            cycles.append((verts, uvs))
    return paths, cycles


def _assemble_face_open(surface, sense, strip, paths, cycles, keep_inside,
                        other_shell, depth, raycast_depth, topo, fi):
    """Open-branch counterpart of :func:`_assemble_face`: the face's
    domain is partitioned by border-anchored trim paths (plus closed
    loops), every region is certified kept/discarded through its
    strip-free grid components, and kept faces carry FULL border loops —
    corner vertices and border sub-edges shared through the operand's
    seam topology, so the exact pairing audit closes over seam edges."""
    from forgekernel.raycast import (PointClassifyUncertified,
                                     classify_point_in_shell)
    from forgekernel.trim import _classify_in_loops, split_domain_by_paths
    from forgekernel.trimshell import ShellFace

    def kept_at(u, v):
        p = surface.eval(u, v)
        verdict = classify_point_in_shell(other_shell, p,
                                          depth=raycast_depth)
        return (verdict == "in") == keep_inside

    def kept_at_any(cands):
        last = None
        for (u, v) in cands:
            try:
                return kept_at(u, v)
            except PointClassifyUncertified as exc:
                last = exc
        raise last

    corner_items = [(topo.corner_vertex(fi, bits), topo.corner_uv(fi, bits))
                    for bits in ((0, 0), (1, 0), (1, 1), (0, 1))]   # CCW

    if not strip:
        (u0, u1), (v0, v1) = surface.domain()
        du, dv = F(u1) - F(u0), F(v1) - F(v0)
        cands = [(F(u0) + du * a, F(v0) + dv * b)
                 for a, b in ((F(1, 2), F(1, 2)), (F(9, 17), F(8, 19)),
                              (F(7, 23), F(13, 29)), (F(19, 31), F(11, 37)))]
        if not kept_at_any(cands):
            return None
        face = ShellFace(surface, sense, set(), lambda u, v: True)
        face.add_loop(corner_items, outer=True)
        return face

    vmap: dict = {}

    def register(vx, uv):
        key = (F(uv[0]), F(uv[1]))
        old = vmap.get(key)
        if old is not None and old is not vx:
            raise BooleanUnsupported(
                "assembly_degenerate",
                f"two distinct trim vertices at the same exact parameter "
                f"point ({key[0]}, {key[1]}) of one face")
        vmap[key] = vx

    for verts, uvs in list(paths) + list(cycles):
        for vx, uv in zip(verts, uvs):
            register(vx, uv)
    for vx, uv in corner_items:
        register(vx, uv)

    regions = split_domain_by_paths(surface, [uvs for _, uvs in paths],
                                    [uvs for _, uvs in cycles])
    grid = _FaceGrid(surface, strip, depth)
    verdicts: dict = {}

    def comp_kept(c):
        if c not in verdicts:
            verdicts[c] = kept_at_any(
                [grid.cell_mid(cell) for cell in grid.candidates(c)])
        return verdicts[c]

    region_comps: list = [[] for _ in regions]
    for c in range(grid.ncomp):
        placed = False
        for cell in grid.candidates(c, k=8):
            um, vm = grid.cell_mid(cell)
            hit = None
            for ri, reg in enumerate(regions):
                cls = _classify_in_loops(reg, um, vm)
                if cls == "in":
                    hit = ri
                    break
                if cls == "on":
                    hit = "on"
                    break
            if isinstance(hit, int):
                region_comps[hit].append(c)
                placed = True
                break
        if not placed:
            raise BooleanUnsupported(
                "region_unresolved",
                f"a strip-free component's witness points all landed on "
                f"trim-region boundaries at depth {depth}; raise the "
                f"depth")

    kept_regions = []
    for ri, reg in enumerate(regions):
        comps = region_comps[ri]
        if not comps:
            raise BooleanUnsupported(
                "region_unresolved",
                f"no strip-free grid cell inside a trim region at depth "
                f"{depth} — the region is thinner than the grid; raise "
                f"the depth")
        vals = {comp_kept(c) for c in comps}
        if len(vals) > 1:
            raise BooleanUnsupported(
                "region_unresolved",
                "mixed certified memberships inside one trim region — "
                "the arrangement disagrees with the certified verdicts; "
                "raise the depth")
        if vals.pop():
            kept_regions.append(ri)
    if not kept_regions:
        return None

    point_memo: dict = {}

    def inside(u, v):
        c = grid.comp_of(u, v)
        if c is not None:
            return comp_kept(c)
        key = (F(u), F(v))
        if key not in point_memo:
            point_memo[key] = kept_at(*key)
        return point_memo[key]

    face = ShellFace(surface, sense, strip, inside)

    def vertex_of(uv):
        key = (F(uv[0]), F(uv[1]))
        vx = vmap.get(key)
        if vx is None:
            raise BooleanUnsupported(
                "assembly_degenerate",
                f"trim-region vertex ({key[0]}, {key[1]}) is neither a "
                f"chain point, a border crossing, nor a corner")
        return vx

    for ri in kept_regions:
        reg = regions[ri]
        face.add_loop([(vertex_of(uv), uv) for uv in reg[0]], outer=True)
        for hole in reg[1:]:
            face.add_loop([(vertex_of(uv), uv) for uv in hole],
                          outer=False)
    return face


def boolean_trimmed(op: str, A, B, depth: int = 5, audit_depth=None,
                    raycast_depth: int = 5, use_rust=None):
    """Boolean of two closed patch solids as an audited, certified
    :class:`~forgekernel.trimshell.TrimmedShell` — K7 gap 5.

    ``op`` is ``"union"``, ``"intersect"`` or ``"cut"`` (A − B); ``A``
    and ``B`` are :class:`PatchSolid`-shaped operands (outward-oriented
    polynomial single-span Bézier faces — anything else refuses by
    name). Pipeline: per face-pair SSI (every surviving cell certified
    or refused), certified chains on both parameter domains, exact
    partition of each trimmed face's domain, certified ray-parity
    membership of every strip-free fragment, senses from the
    monotonicity rule (module comment), shared-vertex loop topology, and
    the full three-oracle shell audit before anything is returned. The
    result's volume is a ``CInterval`` ("certified ± e", ADR-0019) —
    exact (zero-width) whenever nothing was trimmed.

    OPEN branches — curves that leave a face through its border, the
    unavoidable shape of a loft × loft boolean (two z-fibered solids
    can never meet in a closed single-face-pair loop) — are assembled
    when every operand they cross carries proven seam topology
    (``PatchSolid.seams``, emitted by
    :meth:`forgekernel.loft.LoftSolid.to_patches`): each seam crossing
    is canonicalized to ONE certified vertex transferred across the
    seam by exact control-point equality, chains glue across the other
    operand's seams, and kept faces carry full border loops so the
    exact pairing audit closes over seam edges too.

    Refuses by name (never a wrong shell): tangent contact
    (:class:`~forgekernel.ssi.SsiCellUncertified`), unstitchable chains
    (:class:`~forgekernel.ssi.TrimLoopUnstitchable`), open branches over
    a seamless operand (``open_branch``), invalid/unmatched seam
    topology, rational/multi-span/inward operands, unresolvable regions,
    and the empty result (:class:`BooleanUnsupported`)."""
    from forgekernel.ssi import ssi_chains, ssi_strips
    from forgekernel.trimshell import TrimmedShell, TrimVertex

    if op not in _BOOL_KEEP_INSIDE:
        raise ValueError(
            f"boolean op must be union|intersect|cut, got {op!r}")
    fa = _operand_faces("A", A)
    fb = _operand_faces("B", B)

    strips: dict = {("A", i): set() for i in range(len(fa))}
    strips.update({("B", j): set() for j in range(len(fb))})
    recs: list = []
    for i, sa in enumerate(fa):
        for j, sb in enumerate(fb):
            a_cells, b_cells = ssi_strips(sa, sb, depth, use_rust=use_rust)
            if not a_cells:
                continue                      # certified non-intersection
            res = ssi_chains(sa, sb, depth, use_rust=use_rust)
            for ch in res["chains"]:
                recs.append({"pair": (i, j), "pts": list(ch["points"]),
                             "verts": [TrimVertex() for _ in ch["points"]],
                             "closed": ch["closed"], "ends": ch["ends"]})
            strips[("A", i)] |= a_cells
            strips[("B", j)] |= b_cells

    open_recs = [r for r in recs if not r["closed"]]
    topo_a = topo_b = None
    if open_recs:
        needs = {X: any(end is not None and end[0] == X
                        for r in open_recs for end in r["ends"])
                 for X in ("A", "B")}
        for X, solid, flist in (("A", A, fa), ("B", B, fb)):
            if not needs[X]:
                continue
            if getattr(solid, "seams", None) is None:
                raise BooleanUnsupported(
                    "open_branch",
                    f"an intersection curve leaves a face of operand {X} "
                    f"through its border, and the operand carries no "
                    f"seam topology (PatchSolid.seams is None) — border "
                    f"segments can only be paired with declared, "
                    f"control-point-proven seam edges (LoftSolid."
                    f"to_patches emits them); refusing rather than ship "
                    f"an unauditable shell")
            topo = _SeamTopology(X, flist, solid.seams)
            if X == "A":
                topo_a = topo
            else:
                topo_b = topo
        _canonicalize_crossings(open_recs, {"A": topo_a, "B": topo_b},
                                depth)

    keep_a, keep_b = _BOOL_KEEP_INSIDE[op]
    shell_a = untrimmed_shell(fa)
    shell_b = untrimmed_shell(fb)
    faces = []
    for side_idx, side, flist, keep_inside, sense, other, topo in (
            (0, "A", fa, keep_a, 1, shell_b, topo_a),
            (1, "B", fb, keep_b, _BOOL_B_SENSE[op], shell_a, topo_b)):
        for i, f in enumerate(flist):
            paths, cycles = _face_paths(side_idx, i, recs)
            if topo is None:
                # closed-loop mode: identical to the landed stage-2 path
                sf = _assemble_face(f, sense, strips[(side, i)], cycles,
                                    keep_inside, other, depth,
                                    raycast_depth)
            else:
                sf = _assemble_face_open(f, sense, strips[(side, i)],
                                         paths, cycles, keep_inside,
                                         other, depth, raycast_depth,
                                         topo, i)
            if sf is not None:
                faces.append(sf)
    if not faces:
        raise BooleanUnsupported(
            "empty_result",
            f"{op}: no face fragment survives — the result is the empty "
            f"solid, which has no TrimmedShell representation; test "
            f"emptiness upstream if empty is an expected answer")

    shell = TrimmedShell(faces)              # exact pairing audit runs here
    shell.audit(depth=depth if audit_depth is None else audit_depth)
    return shell
