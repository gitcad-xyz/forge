"""K6.0 — surfacing groundwork: Coons patches, exact curvature, G1.

The exactness thesis extends into differential geometry:

- **Gaussian curvature is exactly rational.** With the unnormalized
  normal ``n = S_u × S_v``, the second-fundamental terms ``L' = S_uu·n``
  etc. carry a factor |n| each, and

      K = (L'·N' − M'²) / (EG − F²)²

  — every |n| cancels, so K at a rational parameter of a polynomial
  surface is an exact ``Fraction``. A float kernel evaluates the same
  formula through several rounded square roots.
- **Mean curvature needs one √** (the (EG−F²)^{3/2}) and comes back as
  a certified interval (ADR-0019).
- **G1 continuity is certified by polynomial identity.** Two patches
  meet G1 along a shared boundary iff the cross product of their
  transversal tangents with the boundary tangent vanishes identically —
  a polynomial in the boundary parameter. Checking it at more sample
  points than its degree bound is a PROOF (a nonzero polynomial of
  degree d has at most d roots), not a heuristic.
- **Coons patches** (bilinearly blended) from four Bézier boundaries:
  ruled(u) + ruled(v) − bilinear, assembled exactly via Bézier degree
  elevation and control-net arithmetic.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.interval import CInterval
from forgekernel.nurbs import BSplineSurface, bezier_surface, surface_partials2

F = Fraction


# -- Bézier net algebra --------------------------------------------------------

def _elevate_row(row):
    """Degree-elevate a Bézier control row by one (exact convex comb.)."""
    p = len(row) - 1
    out = [row[0]]
    for i in range(1, p + 1):
        a = F(i, p + 1)
        out.append(tuple(a * row[i - 1][c] + (1 - a) * row[i][c]
                         for c in range(3)))
    out.append(row[-1])
    return out


def _elevate_u(net, times):
    for _ in range(times):
        cols = list(map(list, zip(*net)))
        cols = [_elevate_row(c) for c in cols]
        net = list(map(list, zip(*cols)))
    return net


def _elevate_v(net, times):
    for _ in range(times):
        net = [_elevate_row(r) for r in net]
    return net


def coons_patch(c0, c1, d0, d1) -> BSplineSurface:
    """Bilinearly blended Coons patch from four Bézier boundary curves
    (given as control-point lists): ``c0``/``c1`` run along u at v=0/1;
    ``d0``/``d1`` run along v at u=0/1. Corners must agree exactly.

    S = Ruled_v(c0,c1) + Ruled_u(d0,d1) − Bilinear(corners), assembled
    as one exact Bézier net via degree elevation."""
    c0 = [tuple(F(x) for x in pt) for pt in c0]
    c1 = [tuple(F(x) for x in pt) for pt in c1]
    d0 = [tuple(F(x) for x in pt) for pt in d0]
    d1 = [tuple(F(x) for x in pt) for pt in d1]
    if len(c0) != len(c1) or len(d0) != len(d1):
        raise ValueError("opposite boundaries must share degree (K6.1)")
    # corner compatibility, exact
    if not (c0[0] == d0[0] and c0[-1] == d1[0]
            and c1[0] == d0[-1] and c1[-1] == d1[-1]):
        raise ValueError("Coons boundaries disagree at a corner")
    p = len(c0) - 1                     # u-degree
    q = len(d0) - 1                     # v-degree
    # ruled between c0 and c1: net rows along u, 2 columns in v
    ruled_v = [[c0[i], c1[i]] for i in range(p + 1)]          # (p+1)×2
    ruled_u = [[d0[j] for j in range(q + 1)],
               [d1[j] for j in range(q + 1)]]                 # 2×(q+1)
    bilin = [[c0[0], c1[0]], [c0[-1], c1[-1]]]                # 2×2
    # elevate all three to degree (p, q)
    A = _elevate_v(ruled_v, q - 1)
    B = _elevate_u(ruled_u, p - 1)
    C = _elevate_u(_elevate_v(bilin, q - 1), p - 1)
    net = [[tuple(A[i][j][c] + B[i][j][c] - C[i][j][c] for c in range(3))
            for j in range(q + 1)] for i in range(p + 1)]
    return bezier_surface(net)


# -- curvature -----------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def gaussian_curvature(surface: BSplineSurface, u, v) -> Fraction:
    """K at (u, v) — an EXACT rational for a polynomial surface.
    Every |n| in the classical formula cancels:
    K = ((S_uu·n)(S_vv·n) − (S_uv·n)²) / (EG − F²)²  with n = S_u×S_v."""
    _, su, sv, suu, suv, svv = surface_partials2(surface, u, v)
    n = _cross(su, sv)
    E, Ff, G = _dot(su, su), _dot(su, sv), _dot(sv, sv)
    denom = (E * G - Ff * Ff) ** 2
    if denom == 0:
        raise ValueError("degenerate tangent plane (singular parameterization)")
    return (_dot(suu, n) * _dot(svv, n) - _dot(suv, n) ** 2) / denom


def mean_curvature(surface: BSplineSurface, u, v) -> CInterval:
    """H at (u, v) — certified (one genuine √ in (EG−F²)^{3/2}).
    H = (E·N' − 2F·M' + G·L') / (2 (EG−F²)^{3/2}) with primes · n̂|n|."""
    _, su, sv, suu, suv, svv = surface_partials2(surface, u, v)
    n = _cross(su, sv)
    E, Ff, G = _dot(su, su), _dot(su, sv), _dot(sv, sv)
    W = E * G - Ff * Ff                      # = |n|²
    if W == 0:
        raise ValueError("degenerate tangent plane")
    num = E * _dot(svv, n) - 2 * Ff * _dot(suv, n) + G * _dot(suu, n)
    w32 = (CInterval.exact(W) * CInterval.exact(W) * CInterval.exact(W)).sqrt()
    lo, hi = F(num) / w32.hi / 2, F(num) / w32.lo / 2
    return CInterval(min(lo, hi), max(lo, hi))


# -- G1 continuity certification -----------------------------------------------

def g1_certify(A: BSplineSurface, B: BSplineSurface, *,
               a_edge: str = "u1", b_edge: str = "u0",
               samples: int | None = None) -> bool:
    """Certify G1 continuity along a shared boundary by polynomial
    identity testing: tangent planes agree along the seam iff
    (t_A × t_B) · anything vanishes — concretely, the normals of A and
    B are parallel at every boundary point, i.e. n_A × n_B ≡ 0, a
    vector polynomial in the seam parameter. Its degree is bounded by
    4·(p+q); checking MORE sample points than the bound and finding
    zero every time is a proof, not a heuristic."""
    def at(surface, edge, t):
        (u0d, u1d), (v0d, v1d) = surface.domain()
        u0d, u1d = F(u0d), F(u1d)
        v0d, v1d = F(v0d), F(v1d)
        if edge == "u1":
            return F(u1d), v0d + t * (v1d - v0d)
        if edge == "u0":
            return F(u0d), v0d + t * (v1d - v0d)
        if edge == "v1":
            return u0d + t * (u1d - u0d), F(v1d)
        return u0d + t * (u1d - u0d), F(v0d)

    bound = 4 * (A.p + A.q + B.p + B.q) + 1
    n = samples or bound
    for k in range(n + 1):
        t = F(k, n)
        ua, va = at(A, a_edge, t)
        ub, vb = at(B, b_edge, t)
        # positions must coincide exactly (G0 first)
        if A.eval(ua, va) != B.eval(ub, vb):
            return False
        na = A.normal(ua, va)
        nb = B.normal(ub, vb)
        if _cross(na, nb) != (0, 0, 0):
            return False
    return True


# -- K6.1: G2 certification + curvature combs ----------------------------------

def g2_certify(A: BSplineSurface, B: BSplineSurface, *,
               a_edge: str = "u1", b_edge: str = "u0",
               samples: int | None = None) -> bool:
    """Certify G2 (curvature continuity) along a shared seam, exactly.

    Requires G1 first (checked). Then the transversal normal curvature
    must agree at every seam point. With unnormalized normals the
    comparison cross-multiplies into pure rationals:

        κ_n = (S_tt·n̂)/|S_t'|²-ish terms → compare
        (S_tt^A·n_A)²·W_B == (S_tt^B·n_B)²·W_A   and matching sign,

    where W = |n|² = EG−F² is rational. Zero at more samples than the
    polynomial degree bound ⇒ identically zero: a proof."""
    from forgekernel.nurbs import surface_partials2

    if not g1_certify(A, B, a_edge=a_edge, b_edge=b_edge, samples=samples):
        return False

    def at(surface, edge, t):
        (u0d, u1d), (v0d, v1d) = surface.domain()
        u0d, u1d, v0d, v1d = F(u0d), F(u1d), F(v0d), F(v1d)
        if edge == "u1":
            return F(u1d), v0d + t * (v1d - v0d)
        if edge == "u0":
            return F(u0d), v0d + t * (v1d - v0d)
        if edge == "v1":
            return u0d + t * (u1d - u0d), F(v1d)
        return u0d + t * (u1d - u0d), F(v0d)

    def transversal_kn(surface, edge, uu, vv):
        """(S_tt·n, W) for the TRANSVERSAL parameter curve at the seam
        (the u-direction for u-edges, v-direction for v-edges)."""
        _, su, sv, suu, suv, svv = surface_partials2(surface, uu, vv)
        n = _cross(su, sv)
        W = _dot(n, n)
        stt = suu if edge in ("u0", "u1") else svv
        return _dot(stt, n), W, n

    bound = 8 * (A.p + A.q + B.p + B.q) + 1
    ns = samples or bound
    for k in range(ns + 1):
        t = F(k, ns)
        ua, va = at(A, a_edge, t)
        ub, vb = at(B, b_edge, t)
        Xa, Wa, na = transversal_kn(A, a_edge, ua, va)
        Xb, Wb, nb = transversal_kn(B, b_edge, ub, vb)
        if Wa == 0 or Wb == 0:
            return False
        # sign of n_B relative to n_A (G1 already guarantees parallel)
        s = 1 if _dot(na, nb) > 0 else -1
        # κ_n^A == κ_n^B in A's orientation, cross-multiplied (exact):
        if Xa * Xa * Wb != Xb * Xb * Wa:
            return False
        # signs must agree too (curvature bowing the same way)
        if (Xa > 0) != (s * Xb > 0) and not (Xa == 0 and Xb == 0):
            return False
    return True


def curve_curvature_comb(curve, n: int = 32, scale: float = 1.0):
    """Curvature comb for a Bézier/B-spline curve — the visual surfacing
    audit tool. Returns [(point, tip), ...] where tip = point + scale·κ·N̂.
    κ = |c'×c''|/|c'|³ has one √ per sample: magnitudes are certified-
    exact ratios evaluated to float only for the viewer."""
    import math

    from forgekernel.nurbs import _deboor4, _hodo_list

    p, U = curve.p, curve.U
    pts = [tuple(pt) for pt in curve.cp]
    (t0, t1) = curve.domain()
    out = []
    p1, U1, D1 = _hodo_list(p, U, pts)
    p2, U2, D2 = _hodo_list(p1, U1, D1) if p1 >= 1 else (-1, [], [])
    for k in range(n + 1):
        t = F(t0) + F(k, n) * (F(t1) - F(t0))
        c = curve.eval(t)
        d1 = _deboor4(p1, U1, D1, t) if D1 else (F(0),) * 3
        d2 = _deboor4(p2, U2, D2, t) if p2 >= 0 and D2 else (F(0),) * 3
        cx = _cross(d1, d2)
        num2 = _dot(cx, cx)                    # |c'×c''|², exact
        den2 = _dot(d1, d1)                    # |c'|², exact
        if den2 == 0:
            continue
        kappa = math.sqrt(float(num2)) / float(den2) ** 1.5
        # normal direction: component of c'' orthogonal to c'
        proj = _dot(d2, d1)
        nvec = tuple(float(d2[c]) - float(proj / den2) * float(d1[c])
                     for c in range(3))
        nlen = math.sqrt(sum(x * x for x in nvec)) or 1.0
        pt = tuple(float(x) for x in c)
        tip = tuple(pt[c] + scale * kappa * nvec[c] / nlen for c in range(3))
        out.append((pt, tip))
    return out


# -- K6.2: G1 blend strip (fill between two patches, proven smooth) -----------

def _edge_data(surface: BSplineSurface, edge: str):
    """(boundary curve net, transversal derivative net) along an edge —
    exact Bézier control rows extracted from the surface net.

    For a Bézier patch the boundary curve is the edge row/column of the
    control net, and the transversal derivative along that edge is
    p·(row₁ − row₀) (a Bézier curve of the edge parameter)."""
    net = surface.cp
    p, q = surface.p, surface.q
    if any(w != F(1) for row in surface.w for w in row):
        raise ValueError("blend strip: polynomial patches only (K6.3)")
    if edge == "u1":        # u = 1 edge, parameter runs along v
        c = net[-1]
        d = [tuple(F(p) * (net[-1][j][k] - net[-2][j][k]) for k in range(3))
             for j in range(q + 1)]
    elif edge == "u0":
        c = net[0]
        d = [tuple(F(p) * (net[1][j][k] - net[0][j][k]) for k in range(3))
             for j in range(q + 1)]
    elif edge == "v1":
        c = [net[i][-1] for i in range(p + 1)]
        d = [tuple(F(q) * (net[i][-1][k] - net[i][-2][k]) for k in range(3))
             for i in range(p + 1)]
    else:
        c = [net[i][0] for i in range(p + 1)]
        d = [tuple(F(q) * (net[i][1][k] - net[i][0][k]) for k in range(3))
             for i in range(p + 1)]
    return [tuple(x) for x in c], d


def g1_blend_strip(A: BSplineSurface, B: BSplineSurface, *,
                   a_edge: str = "u1", b_edge: str = "u0") -> BSplineSurface:
    """Fill the gap between an edge of ``A`` and an edge of ``B`` with a
    cubic Hermite strip that meets BOTH with G1 continuity — by
    construction, and certifiable by :func:`g1_certify` afterwards.

    The strip is cubic transversally: Hermite data (position + outward
    transversal derivative) at each end, converted to an exact Bézier
    net via the Hermite→Bézier map  [P0, P0+m0/3, P1−m1/3, P1]."""
    ca, da = _edge_data(A, a_edge)
    cb, db = _edge_data(B, b_edge)
    if len(ca) != len(cb):
        raise ValueError("blend strip: edges must share degree (K6.3)")
    # outward direction: A's edge derivative points OUT of A when the
    # edge is u1/v1, INTO A when u0/v0 (flip). Likewise B's must point
    # INTO the strip (i.e. out of B flipped at u1/v1... B's edge faces
    # the strip, so B's outward transversal at u0/v0 is already inward).
    sa = 1 if a_edge in ("u1", "v1") else -1
    sb = 1 if b_edge in ("u1", "v1") else -1
    n = len(ca)
    net = []
    for i in range(n):
        p0 = ca[i]
        p1 = cb[i]
        m0 = tuple(sa * da[i][k] for k in range(3))       # d/dv at strip v=0
        m1 = tuple(-sb * db[i][k] for k in range(3))      # d/dv at strip v=1
        net.append([
            p0,
            tuple(p0[k] + m0[k] / 3 for k in range(3)),
            tuple(p1[k] - m1[k] / 3 for k in range(3)),
            p1,
        ])
    return bezier_surface(net)
