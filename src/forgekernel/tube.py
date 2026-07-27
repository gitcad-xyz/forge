"""Second-order trim enclosures — the tube bound (#96 bracket tightening).

The first-order strip bracket (:func:`forgekernel.bsolid.
certified_trim_flux`) prices every SSI strip cell at its full hull range:
width O(2^-depth)·|F|·strip-length — information-theoretically forced
when the trim boundary is only known as a cell enclosure. This module
uses MORE of what the SSI machinery already certifies — the residual-
certified points and the hodograph control nets — to build a tube of
half-width O(h²) around the certified polyline, so a boolean face's flux
becomes

    EXACT polygon flux over the (snapped) trim loops  ±  ∫_tube |F|,

second order in the SSI depth. Everything here is exact ℚ; a bound that
cannot be certified refuses (:class:`TubeUncertified`) and the caller
falls back to the first-order strip bracket — never a wrong bound.

The three certified ingredients, per chord between consecutive certified
points p_k → p_{k+1}:

1. **Anchor** (δ): the certified point carries an exact residual
   |S_A − S_B| < 1e-10, and a true intersection point q_k provably lies
   within δ = 1e-10/σ of it in parameter space, where σ is a certified
   lower bound on the smallest singular value of the 3×4 SSI Jacobian
   over a tiny box around the point (steepest-descent argument: |∇|Φ||
   ≥ σ_min(J), so the descent path reaches Φ = 0 within |Φ|/σ without
   leaving the box). σ_min² = λ_min(JJ^T) ≥ 4·det/trace² for a 3×3 PSD
   matrix (det = λ₁λ₂λ₃ ≤ λ_min·(trace/2)²), with det and trace
   evaluated in interval arithmetic over Bernstein hodograph hulls.

2. **Cone** (ρ): the intersection-curve tangent in this face's (u, v) is
   (up to positive scale) D = (det(T, S_v, N), det(S_u, T, N)) with
   N = S_u×S_v and T = N_own×N_other. Expanding the determinants,
   D_u = n_oth·H_u and D_v = n_oth·H_v where H_u = (S_v×N)×N and
   H_v = (N×S_u)×N are EXACT polynomial B-forms of this face alone —
   so the chord components s = D·d and w = D×d are computed as exact
   forms (s = n_oth·(H_u·dx + H_v·dy), w = n_oth·(H_v·dx − H_u·dy))
   hulled per covered cell, which preserves the near-cancellation in w
   that plain interval cross-products destroy; only the other surface's
   normal enters as an interval (exact for a plane). If s is one-signed,
   progress along the chord is monotone and the perpendicular drift rate
   is at most ρ = max|w|/min|s| — the whole arc stays inside a rational
   quad around the chord with half-width r = pads + ρ·(len + pads),
   where the pads fold in δ and the dyadic snap distance (snapping stays
   an enclosure). Coverage is a certified fixpoint: every strip cell the
   quad overlaps must be in the hull-union the cone was computed over.
   The PRICED area is sharper than the quad: the arc leaves the chord at
   rate ≤ ρ from BOTH anchored ends, so the band mass is at most
   2·maxpad·X + ρX²/2 (X the padded chord length) — and when w itself is
   one-signed the perpendicular offset is monotone between the anchors,
   so the band collapses to 2·maxpad·X.

3. **Band audit**: the symmetric difference between the true kept region
   and the polygon region has its boundary inside Band = ∪ quads (true
   curves by the cone bound, polygon chords by construction), so it is a
   union of connected components of Domain∖Band. A subcell-level flood
   fill enumerates those components; each is either *verified* (a
   witness point where exact polygon parity equals the face's certified
   membership — the component carries no difference) or *priced* (its
   whole |F| mass joins the error — sound, and normally empty).

Premise inherited from the chain machinery (documented, same residual
class as SSI chain ordering): within one segment's covered cells the
branch passes once. A violation is guarded by the one-signed-cone and
coverage-fixpoint checks and by the intersection with the independent
first-order bracket, but is not separately certified.
"""

from __future__ import annotations

from fractions import Fraction

F = Fraction

#: exact residual bound carried by every certified SSI point
#: (refine_point certifies |S_A − S_B|² < 1e-20)
EPS_RES = F(1, 10 ** 10)

#: half-size of the σ evaluation box around a certified point (~1e-6)
SIGMA_BOX = F(1, 2 ** 20)

_SQ_SCALE = 10 ** 30


class TubeUncertified(ValueError):
    """Structured refusal: a second-order tube enclosure could not be
    certified for this face (the reason names the failing predicate).
    Callers fall back to the first-order strip bracket — a wider answer,
    never a wrong one."""

    def __init__(self, predicate: str, detail: str) -> None:
        self.predicate = predicate
        super().__init__(f"{predicate}: {detail}")


# -- rational interval scalars/vectors: (lo, hi) pairs ------------------------

def _iv(x):
    return (x, x)


def _iv_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _iv_sub(a, b):
    return (a[0] - b[1], a[1] - b[0])


def _iv_mul(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(p), max(p))


def _iv_scale(a, c):
    return (a[0] * c, a[1] * c) if c >= 0 else (a[1] * c, a[0] * c)


def _iv_mag(a):
    return max(abs(a[0]), abs(a[1]))


def _v_cross(u, v):
    return (_iv_sub(_iv_mul(u[1], v[2]), _iv_mul(u[2], v[1])),
            _iv_sub(_iv_mul(u[2], v[0]), _iv_mul(u[0], v[2])),
            _iv_sub(_iv_mul(u[0], v[1]), _iv_mul(u[1], v[0])))


def _v_dot(u, v):
    s = _iv(F(0))
    for c in range(3):
        s = _iv_add(s, _iv_mul(u[c], v[c]))
    return s


def _v_union(u, v):
    return tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(u, v))


# -- exact Bézier net clipping + hodograph hulls ------------------------------

def _dc_split_at(pts, t):
    """De Casteljau split of a control list (tuples) at rational t."""
    pts = [tuple(p) for p in pts]
    left, right = [pts[0]], [pts[-1]]
    n = len(pts)
    for r in range(1, n):
        pts = [tuple((1 - t) * pts[i][c] + t * pts[i + 1][c]
                     for c in range(len(pts[i])))
               for i in range(n - r)]
        left.append(pts[0])
        right.insert(0, pts[-1])
    return left, right


def _net_clip(net, box):
    """Clip a Bézier net over [0,1]² down to the sub-box ``box`` — exact
    generalized de Casteljau (net rows run along u, columns along v)."""
    u0, u1, v0, v1 = (F(c) for c in box)
    cols = list(map(list, zip(*net)))                    # rows along u
    if u0 > 0:
        cols = [_dc_split_at(col, u0)[1] for col in cols]
    if u1 < 1:
        t = (u1 - u0) / (1 - u0)
        cols = [_dc_split_at(col, t)[0] for col in cols]
    rows = list(map(list, zip(*cols)))
    if v0 > 0:
        rows = [_dc_split_at(row, v0)[1] for row in rows]
    if v1 < 1:
        t = (v1 - v0) / (1 - v0)
        rows = [_dc_split_at(row, t)[0] for row in rows]
    return rows


def _hodo_hulls(net, box):
    """(S_u hull, S_v hull) interval 3-vectors of the clipped net over
    ``box`` — hodograph coefficient hulls, GLOBAL parameter scaling."""
    sub = _net_clip(net, box)
    wu = F(box[1]) - F(box[0])
    wv = F(box[3]) - F(box[2])
    p = len(sub) - 1
    q = len(sub[0]) - 1
    su = []
    sv = []
    for c in range(3):
        if p:
            vals = [F(p) * (sub[i + 1][j][c] - sub[i][j][c]) / wu
                    for i in range(p) for j in range(q + 1)]
            su.append((min(vals), max(vals)))
        else:
            su.append((F(0), F(0)))
        if q:
            vals = [F(q) * (sub[i][j + 1][c] - sub[i][j][c]) / wv
                    for i in range(p + 1) for j in range(q)]
            sv.append((min(vals), max(vals)))
        else:
            sv.append((F(0), F(0)))
    return tuple(su), tuple(sv)


# -- the anchor: certified distance from a point to the true curve ------------

def _det3_iv(M):
    def mul(a, b):
        return _iv_mul(a, b)
    a = mul(M[0][0], _iv_sub(mul(M[1][1], M[2][2]), mul(M[1][2], M[2][1])))
    b = mul(M[0][1], _iv_sub(mul(M[1][0], M[2][2]), mul(M[1][2], M[2][0])))
    c = mul(M[0][2], _iv_sub(mul(M[1][0], M[2][1]), mul(M[1][1], M[2][0])))
    return _iv_add(_iv_sub(a, b), c)


def _sqrt_lo(x):
    from forgekernel.interval import _sqrt_low
    return _sqrt_low(x, _SQ_SCALE)


def _sqrt_hi(x):
    from forgekernel.interval import _sqrt_high
    return _sqrt_high(x, _SQ_SCALE)


def _tiny_box(x, y):
    g = SIGMA_BOX
    return (max(F(0), F(x) - g), min(F(1), F(x) + g),
            max(F(0), F(y) - g), min(F(1), F(y) + g))


def point_delta(own_net, oth_net, own_xy, oth_xy):
    """Certified parameter-space radius δ around a certified SSI point
    that provably contains a true intersection point, or ``None`` when
    the transversality bound fails. See the module docstring (anchor)."""
    ba = _tiny_box(*own_xy)
    bb = _tiny_box(*oth_xy)
    su_a, sv_a = _hodo_hulls(own_net, ba)
    su_b, sv_b = _hodo_hulls(oth_net, bb)
    cols = (su_a, sv_a, su_b, sv_b)
    M = [[_iv(F(0))] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = _iv(F(0))
            for col in cols:
                s = _iv_add(s, _iv_mul(col[i], col[j]))
            M[i][j] = s
    det = _det3_iv(M)
    if det[0] <= 0:
        return None
    trace_hi = sum(M[c][c][1] for c in range(3))
    if trace_hi <= 0:
        return None
    lam_lo = 4 * det[0] / (trace_hi * trace_hi)
    sigma = _sqrt_lo(lam_lo)
    if sigma <= 0:
        return None
    delta = EPS_RES / sigma
    # the descent ball must stay inside the boxes the hulls are valid on
    margin = min(F(own_xy[0]) - ba[0], ba[1] - F(own_xy[0]),
                 F(own_xy[1]) - ba[2], ba[3] - F(own_xy[1]),
                 F(oth_xy[0]) - bb[0], bb[1] - F(oth_xy[0]),
                 F(oth_xy[1]) - bb[2], bb[3] - F(oth_xy[1]))
    if delta > margin:
        return None
    return delta


# -- quad geometry (exact, conservative where stated) -------------------------

def _quad_box_overlap(quad, box) -> bool:
    """May the convex quad meet the closed box? Conservative-exact SAT:
    ``False`` only on a proven separating axis."""
    u0, u1, v0, v1 = box
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if max(xs) < u0 or min(xs) > u1 or max(ys) < v0 or min(ys) > v1:
        return False
    corners = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    m = len(quad)
    for k in range(m):
        ax, ay = quad[k]
        bx, by = quad[(k + 1) % m]
        ex, ey = bx - ax, by - ay
        # side of the quad interior: sign of the other vertices
        ref = 0
        for (cx, cy) in quad:
            d = ex * (cy - ay) - ey * (cx - ax)
            if d != 0:
                ref = 1 if d > 0 else -1
                break
        if ref == 0:
            continue                              # degenerate edge: skip
        if all((ex * (cy - ay) - ey * (cx - ax)) * ref < 0
               for (cx, cy) in corners):
            return False                          # box beyond this edge
    return True


def _quad_area(quad):
    s = F(0)
    m = len(quad)
    for k in range(m):
        (ax, ay), (bx, by) = quad[k], quad[(k + 1) % m]
        s += ax * by - bx * ay
    return abs(s) / 2


# -- flux-form hulls per dyadic cell ------------------------------------------

class FormHullCache:
    """Coefficient hulls of a tensor Bernstein form restricted to dyadic
    square cells of [0,1]² — exact de Casteljau descent, memoized along
    the quadtree path (convex-hull property: the hull bounds the form's
    range on the cell)."""

    def __init__(self, P) -> None:
        self._forms = {(F(0), F(1), F(0), F(1)): P}
        self._hulls: dict = {}

    def _form(self, box):
        cached = self._forms.get(box)
        if cached is not None:
            return cached
        from forgekernel.bsolid import _bf_split_u, _bf_split_v
        u0, u1, v0, v1 = box
        w = u1 - u0
        pw = 2 * w
        qu = (u0 / pw)
        qv = (v0 / pw)
        pu0 = (qu.numerator // qu.denominator) * pw
        pv0 = (qv.numerator // qv.denominator) * pw
        parent = (pu0, pu0 + pw, pv0, pv0 + pw)
        pf = self._form(parent)
        left, right = _bf_split_u(pf)
        half = left if u0 == pu0 else right
        bottom, top = _bf_split_v(half)
        quarter = bottom if v0 == pv0 else top
        self._forms[box] = quarter
        return quarter

    def hull(self, cell):
        cached = self._hulls.get(cell)
        if cached is None:
            from forgekernel.bsolid import _bf_hull
            cached = _bf_hull(self._form(tuple(F(c) for c in cell)))
            self._hulls[cell] = cached
        return cached

    def mag(self, cell):
        lo, hi = self.hull(cell)
        return max(abs(lo), abs(hi))


# -- exact polygon flux from the flux B-form ----------------------------------

def _bern_to_power_1d(coeffs):
    """Bernstein → power basis: c_k = C(m,k)·Σ_{i≤k} (−1)^{k−i} C(k,i) b_i."""
    from math import comb
    m = len(coeffs) - 1
    return [F(comb(m, k)) * sum((-1) ** (k - i) * comb(k, i) * coeffs[i]
                                for i in range(k + 1))
            for k in range(m + 1)]


def polygon_flux(P, loops):
    """(1/3)∬_D F du dv over the even-odd region D of polygonal
    ``loops`` in [0,1]², where F is the flux integrand given as the
    exact tensor B-form ``P`` — Green's theorem on the exact B-form
    u-antiderivative, each edge integrated in CLOSED FORM (the
    antiderivative composed with the linear edge is a 1D power-basis
    polynomial whose exact integral is a coefficient sum — no
    quadrature nodes). Same integral as
    :func:`forgekernel.bsolid.trimmed_patch_flux`, exact ℚ."""
    m = len(P) - 1                                 # u-degree
    n = len(P[0]) - 1                              # v-degree
    # u-antiderivative: AA has u-degree m+1, AA(0,·) = 0
    AA = [[F(0)] * (n + 1)]
    for j in range(1, m + 2):
        AA.append([sum(P[i][col] for i in range(j)) / (m + 1)
                   for col in range(n + 1)])
    # tensor power basis: rows (v) then columns (u)
    AAp_rows = [_bern_to_power_1d(row) for row in AA]
    AAp = [_bern_to_power_1d([AAp_rows[i][j] for i in range(m + 2)])
           for j in range(n + 1)]                  # AAp[j][i]: u^i v^j
    total = F(0)
    for loop in loops:
        pts = [(F(a), F(b)) for a, b in loop]
        cnt = len(pts)
        for k in range(cnt):
            (ua, va), (ub, vb) = pts[k], pts[(k + 1) % cnt]
            dvv = vb - va
            if dvv == 0:
                continue
            duu = ub - ua
            # powers of the two linear edge polynomials in τ
            upow = [[F(1)]]
            for _ in range(m + 1):
                prev = upow[-1]
                nxt = [F(0)] * (len(prev) + 1)
                for d, c in enumerate(prev):
                    nxt[d] += c * ua
                    nxt[d + 1] += c * duu
                upow.append(nxt)
            vpow = [[F(1)]]
            for _ in range(n):
                prev = vpow[-1]
                nxt = [F(0)] * (len(prev) + 1)
                for d, c in enumerate(prev):
                    nxt[d] += c * va
                    nxt[d + 1] += c * dvv
                vpow.append(nxt)
            g = [F(0)] * (m + n + 3)
            for j in range(n + 1):
                row = AAp[j]
                # W(τ) = Σ_i row[i]·U(τ)^i, then g += W ⊛ V^j
                W = [F(0)] * (m + 2)
                for i in range(m + 2):
                    ci = row[i]
                    if ci:
                        for d, c in enumerate(upow[i]):
                            W[d] += ci * c
                vj = vpow[j]
                for d1, c1 in enumerate(W):
                    if c1:
                        for d2, c2 in enumerate(vj):
                            g[d1 + d2] += c1 * c2
            # ∮ G dv over the edge = dvv · ∫₀¹ g(τ) dτ, exactly
            total += dvv * sum(c / (d + 1) for d, c in enumerate(g))
    return total / 3


# -- the cone bound: exact direction forms of one face ------------------------

def _net_forms(net):
    """Component B-forms of a polynomial net (rows along u)."""
    return [[[F(pt[c]) for pt in row] for row in net] for c in range(3)]


def _form_cross(A, B):
    from forgekernel.bsolid import _bf_mul, _bf_sub
    return (_bf_sub(_bf_mul(A[1], B[2]), _bf_mul(A[2], B[1])),
            _bf_sub(_bf_mul(A[2], B[0]), _bf_mul(A[0], B[2])),
            _bf_sub(_bf_mul(A[0], B[1]), _bf_mul(A[1], B[0])))


def face_direction_forms(net):
    """(H_u, H_v): exact B-form 3-vectors of a [0,1]² face with
    D_u = n_other·H_u, D_v = n_other·H_v the parameter-space tangent
    representation of an intersection curve on the face:
    H_u = (S_v×N)×N, H_v = (N×S_u)×N, N = S_u×S_v.  Derivation:
    D_u = det(T, S_v, N) and D_v = det(S_u, T, N) with T = N×n_other,
    and det(N, n, G) = n·(G×N)."""
    from forgekernel.bsolid import _bf_du, _bf_dv, _bf_mul
    comps = _net_forms(net)
    Su = [_bf_du(c) for c in comps]
    Sv = [_bf_dv(c) for c in comps]
    N = _form_cross(Su, Sv)
    Hu = _form_cross(_form_cross(Sv, N), N)
    Hv = _form_cross(_form_cross(N, Su), N)
    # degree-elevate to a common degree (coefficientwise combination in
    # _combo_hull needs matching Bernstein bases); multiplying by an
    # all-ones form of the complementary degree IS degree elevation
    mu = max(len(Hu[0]) - 1, len(Hv[0]) - 1)
    nv = max(len(Hu[0][0]) - 1, len(Hv[0][0]) - 1)

    def lift(A):
        du = mu - (len(A) - 1)
        dv = nv - (len(A[0]) - 1)
        if du == 0 and dv == 0:
            return A
        ones = [[F(1)] * (dv + 1) for _ in range(du + 1)]
        return _bf_mul(A, ones)

    Hu = [lift(c) for c in Hu]
    Hv = [lift(c) for c in Hv]
    # scale ALL six component forms by one common integer: D is a
    # tangent representation up to positive scale, so ρ and every sign
    # test are invariant — and integer coefficients keep the dyadic
    # subdivision (denominators 2^k) cheap
    from math import lcm
    L = 1
    for c in Hu + Hv:
        for row in c:
            for x in row:
                L = lcm(L, x.denominator)
    scale = F(L)
    Hu = [[[x * scale for x in row] for row in c] for c in Hu]
    Hv = [[[x * scale for x in row] for row in c] for c in Hv]
    return Hu, Hv


def _cell_int_coeffs(caches, cell):
    """Per component: (den, [Hu coeffs·den], [Hv coeffs·den]) with a
    common integer denominator — flattened, memoized. Lets the combo
    hull below run in pure integer arithmetic."""
    memo = caches.setdefault("int", {})
    got = memo.get(cell)
    if got is None:
        from math import lcm
        got = []
        for c in range(3):
            fu = [x for row in caches["hu"][c]._form(cell) for x in row]
            fv = [x for row in caches["hv"][c]._form(cell) for x in row]
            den = 1
            for x in fu:
                den = lcm(den, x.denominator)
            for x in fv:
                den = lcm(den, x.denominator)
            got.append((den,
                        [int(x * den) for x in fu],
                        [int(x * den) for x in fv]))
        memo[cell] = got
    return got


def _combo_hull(caches, cell, cu, cv):
    """Interval 3-vector hull over ``cell`` of the exact form
    H_u·cu + H_v·cv, combined per component from the cached clipped
    forms — the combination is done at coefficient level, so the
    Bernstein hull keeps the cancellation between the two terms.
    Integer fast path: cu, cv share a denominator (dyadic snapping), so
    each combined coefficient is (a·nu + b·nv)/(den·dc) with integer
    a, b, nu, nv — hulled as integers, divided once."""
    dc = (cu.denominator * cv.denominator
          // __import__("math").gcd(cu.denominator, cv.denominator))
    nu = int(cu * dc)
    nv = int(cv * dc)
    out = []
    for (den, au, av) in _cell_int_coeffs(caches, cell):
        vals = [a * nu + b * nv for a, b in zip(au, av)]
        scale = F(1, den * dc)
        out.append((min(vals) * scale, max(vals) * scale))
    return tuple(out)


def _strip_index(own_strip, depth):
    """{(i, j): cell} for grid-aligned strip cells — bbox candidate
    lookup instead of full-strip scans."""
    nn = 1 << depth
    idx = {}
    for cell in own_strip:
        idx[(int(cell[0] * nn), int(cell[2] * nn))] = cell
    return idx


def _overlapping_strip(idx, depth, quad):
    """Strip cells possibly meeting the quad (conservative-exact)."""
    nn = 1 << depth
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    i0 = max(0, int(min(xs) * nn) - 1)
    i1 = min(nn - 1, int(max(xs) * nn) + 1)
    j0 = max(0, int(min(ys) * nn) - 1)
    j1 = min(nn - 1, int(max(ys) * nn) + 1)
    out = set()
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            cell = idx.get((i, j))
            if cell is not None and _quad_box_overlap(quad, cell):
                out.add(cell)
    return out


def _segment_quad(p0, p1, a0, a1, own_strip, own2oth, caches, oth_net,
                  hull_memo):
    """The certified quad containing the true arc between the anchors of
    two consecutive (snapped) loop vertices — cone bound + coverage
    fixpoint. Returns (quad, covered, band_area) where ``band_area``
    bounds the area between arc and chord (the two-ended ramp bound; the
    monotone-offset bound when the perpendicular rate is one-signed).
    Raises :class:`TubeUncertified`."""
    dx, dy = F(p1[0]) - F(p0[0]), F(p1[1]) - F(p0[1])
    if dx == 0 and dy == 0:
        raise TubeUncertified("degenerate_chord",
                              "consecutive loop vertices coincide")
    l2 = dx * dx + dy * dy
    l_hi = _sqrt_hi(l2)
    l_lo = _sqrt_lo(l2)

    def oth_normal(cells):
        key = ("n", tuple(sorted(cells)))
        n = hull_memo.get(key)
        if n is None:
            osu = osv = None
            for c in cells:
                h = hull_memo.get(("t", c))
                if h is None:
                    h = _hodo_hulls(oth_net, c)
                    hull_memo[("t", c)] = h
                hu, hv = h
                osu = hu if osu is None else _v_union(osu, hu)
                osv = hv if osv is None else _v_union(osv, hv)
            n = _v_cross(osu, osv)
            hull_memo[key] = n
        return n

    # seed: the strip cells nearest the chord endpoints
    idx = caches.get("stripidx")
    depth = caches["depth"]
    covered = _overlapping_strip(idx, depth, [p0, p1])
    if not covered:
        raise TubeUncertified("chord_off_strip",
                              "a loop chord touches no strip cell")
    for _ in range(6):
        oth_cells = set()
        for c in covered:
            oth_cells.update(own2oth.get(c, ()))
        if not oth_cells:
            raise TubeUncertified("no_pair_cells",
                                  "covered cells map to no other-domain "
                                  "cells")
        n_oth = oth_normal(oth_cells)
        s_iv = w_iv = None
        for c in covered:
            sh = _combo_hull(caches, c, dx, dy)      # H_u·dx + H_v·dy
            wh = _combo_hull(caches, c, -dy, dx)     # H_v·dx − H_u·dy
            s_c = _v_dot(n_oth, sh)
            w_c = _v_dot(n_oth, wh)
            s_iv = s_c if s_iv is None else (min(s_iv[0], s_c[0]),
                                             max(s_iv[1], s_c[1]))
            w_iv = w_c if w_iv is None else (min(w_iv[0], w_c[0]),
                                             max(w_iv[1], w_c[1]))
        if s_iv[0] <= 0 <= s_iv[1]:
            raise TubeUncertified(
                "tangent_cone",
                "curve tangent not certifiably transversal to the chord "
                "over the covered cells")
        smin = s_iv[0] if s_iv[0] > 0 else -s_iv[1]
        rho = _iv_mag(w_iv) / smin
        maxpad = max(a0, a1)
        X = l_hi + a0 + a1
        r = a0 + a1 + rho * X
        e0 = a0 / l_lo
        e1 = a1 / l_lo
        h = r / l_lo
        px, py = -dy * h, dx * h
        q0 = (F(p0[0]) - dx * e0, F(p0[1]) - dy * e0)
        q1 = (F(p1[0]) + dx * e1, F(p1[1]) + dy * e1)
        quad = [(q0[0] - px, q0[1] - py), (q1[0] - px, q1[1] - py),
                (q1[0] + px, q1[1] + py), (q0[0] + px, q0[1] + py)]
        needed = _overlapping_strip(idx, depth, quad)
        if needed <= covered:
            if w_iv[0] >= 0 or w_iv[1] <= 0:
                band = 2 * maxpad * X                # monotone offset
            else:
                band = 2 * maxpad * X + rho * X * X / 2   # two-ended ramp
            return quad, covered, band
        covered |= needed
        if len(covered) > 12:
            raise TubeUncertified(
                "tube_spread",
                f"quad coverage grew past 12 cells (chord {float(l_hi):.3g})")
    raise TubeUncertified("tube_unstable",
                          "coverage fixpoint did not stabilize")


def loop_tube_error(pts_snap, pts_cert, own_first, own_net,
                    oth_net, own_strip, own2oth, eps, fcache, depth,
                    hull_memo, delta_memo, caches):
    """Σ (band area)·max|F|/3 over one closed loop's segments — the
    certified flux error the loop's polygon can carry — plus the quads
    (the Band pieces the audit floods against). Raises
    :class:`TubeUncertified` when any ingredient fails."""
    n = len(pts_snap)
    deltas = []
    for k in range(n):
        pt = pts_cert[k]
        own_xy = (pt[0], pt[1]) if own_first else (pt[2], pt[3])
        oth_xy = (pt[2], pt[3]) if own_first else (pt[0], pt[1])
        d = delta_memo.get(pt)
        if d is None and pt not in delta_memo:
            d = point_delta(own_net, oth_net, own_xy, oth_xy)
            delta_memo[pt] = d
        if d is None:
            raise TubeUncertified(
                "anchor_uncertified",
                "no certified transversality bound at a loop vertex")
        # the anchor also pays the snap displacement of THIS vertex
        sx = abs(F(pts_snap[k][0]) - F(own_xy[0]))
        sy = abs(F(pts_snap[k][1]) - F(own_xy[1]))
        deltas.append(d + sx + sy)
    err = F(0)
    quads = []
    nn = 1 << depth
    wc = F(1, nn)
    caches["stripidx"] = _strip_index(own_strip, depth)
    caches["depth"] = depth
    for k in range(n):
        k2 = (k + 1) % n
        quad, covered, band = _segment_quad(
            pts_snap[k], pts_snap[k2], deltas[k], deltas[k2],
            own_strip, own2oth, caches, oth_net, hull_memo)
        # |F| over every grid cell the quad may touch (strip or not)
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        i0 = max(0, int(min(xs) / wc) - 1)
        i1 = min(nn - 1, int(max(xs) / wc) + 1)
        j0 = max(0, int(min(ys) / wc) - 1)
        j1 = min(nn - 1, int(max(ys) / wc) + 1)
        fmax = F(0)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                cell = (i * wc, (i + 1) * wc, j * wc, (j + 1) * wc)
                if _quad_box_overlap(quad, cell):
                    fmax = max(fmax, fcache.mag(cell))
        err += band * fmax / 3
        quads.append(quad)
    return err, quads


# -- the band audit: components of Domain∖Band --------------------------------

def band_audit(depth, strip, quads, fcache, parity_fn, kept_of_comp,
               comp_of_cell):
    """Certify XOR(kept region, polygon region) ⊆ Band ∪ priced pockets.

    Floods the sub-refined grid (factor 2) outside the Band quads; each
    connected component must contain a witness subcell in a NON-strip
    coarse cell whose exact polygon parity equals the face's certified
    membership (``kept_of_comp(comp_of_cell(cell))``) — a mismatch is a
    proven bookkeeping error and refuses. Components with no such
    witness (pockets) are PRICED: their whole |F| mass joins the
    returned extra error — sound, and normally zero."""
    nn = 1 << depth
    sub = 2
    N = nn * sub
    ws = F(1, N)
    strip_idx = set()
    for (u0, u1, v0, v1) in strip:
        strip_idx.add((int(u0 * nn), int(v0 * nn)))
    # subcells possibly meeting the Band: only within cells quads touch
    quad_cells: dict = {}
    for quad in quads:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        i0 = max(0, int(min(xs) * nn) - 1)
        i1 = min(nn - 1, int(max(xs) * nn) + 1)
        j0 = max(0, int(min(ys) * nn) - 1)
        j1 = min(nn - 1, int(max(ys) * nn) + 1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                quad_cells.setdefault((i, j), []).append(quad)
    marked = set()
    for (i, j), qs in quad_cells.items():
        for si in range(sub):
            for sj in range(sub):
                I, J = i * sub + si, j * sub + sj
                box = (I * ws, (I + 1) * ws, J * ws, (J + 1) * ws)
                if any(_quad_box_overlap(q, box) for q in qs):
                    marked.add((I, J))
    # flood fill the unmarked subgrid
    seen = set()
    extra = F(0)
    for I0 in range(N):
        for J0 in range(N):
            if (I0, J0) in marked or (I0, J0) in seen:
                continue
            stack = [(I0, J0)]
            seen.add((I0, J0))
            comp_cells = []
            witness = None
            while stack:
                I, J = stack.pop()
                parent = (I // sub, J // sub)
                if witness is None and parent not in strip_idx:
                    witness = (I, J, parent)
                comp_cells.append((I, J))
                for dI, dJ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (I + dI, J + dJ)
                    if 0 <= q[0] < N and 0 <= q[1] < N \
                            and q not in marked and q not in seen:
                        seen.add(q)
                        stack.append(q)
            if witness is not None:
                I, J, parent = witness
                mid = ((2 * I + 1) * ws / 2, (2 * J + 1) * ws / 2)
                comp = comp_of_cell(parent)
                if comp is None:
                    raise TubeUncertified(
                        "band_witness_lost",
                        "witness cell has no strip-free component id")
                if parity_fn(mid) != kept_of_comp(comp):
                    raise TubeUncertified(
                        "band_witness_mismatch",
                        "polygon parity disagrees with certified "
                        "membership on a Band-free component — "
                        "bookkeeping error, refusing the tight bracket")
            else:
                # pocket: price its |F| mass instead of guessing
                for (I, J) in comp_cells:
                    parent = (I // sub, J // sub)
                    cell = (parent[0] * F(1, nn), (parent[0] + 1) * F(1, nn),
                            parent[1] * F(1, nn), (parent[1] + 1) * F(1, nn))
                    extra += ws * ws * fcache.mag(cell) / 3
    return extra
