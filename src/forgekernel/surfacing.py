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
