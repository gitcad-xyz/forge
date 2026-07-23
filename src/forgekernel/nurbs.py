"""K3.1 — NURBS / B-spline curve evaluation via de Boor (ADR-0018/0019).

The groundwork for free-form geometry: STEP AP214 curves, general
sweeps/lofts, and — the crown jewel — surface–surface intersection.

The exactness charter reaches further here than one might expect. A
B-spline with **rational** control points, knots, and weights, evaluated
at a **rational** parameter, comes out *exactly rational*: de Boor's
recurrence is nothing but convex combinations — ``+``, ``×``, and ``÷``
by rationals — so no irrationality enters. Such a curve is ``exact``,
not merely certified, and ``eval`` returns a point in ℚ³ that OCCT can
only approximate.

The certified-interval path (ADR-0019) is reserved for the genuinely
irrational cases: an irrational weight (the √2/2 of a true circular
NURBS arc) or an irrational parameter. There ``eval_ci`` carries each
homogeneous coordinate as a ``CInterval`` and the division by the
weight stays enclosure-preserving.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.interval import CInterval

F = Fraction


class BSplineCurve:
    """A NURBS curve: degree ``p``, control points, clamped/open knot
    vector, optional weights (default 1 → a polynomial B-spline).

    Control points are 3-tuples; knots and weights are scalars. Anything
    rational stays exact through evaluation."""

    def __init__(self, degree: int, control_points, knots, weights=None) -> None:
        self.p = int(degree)
        self.cp = [tuple(F(c) for c in pt) for pt in control_points]
        self.U = [F(u) for u in knots]
        n = len(self.cp)
        if weights is None:
            self.w = [F(1)] * n
            self.rational = False
            self.exact_weights = True
        else:
            # weights may be rational (exact eval available) or CInterval
            # (a genuinely irrational weight, e.g. √2/2 for a true circle —
            # only the certified path eval_ci applies)
            self.w = [x if isinstance(x, CInterval) else F(x) for x in weights]
            self.exact_weights = all(not isinstance(x, CInterval) for x in self.w)
            self.rational = not self.exact_weights or \
                any(x != self.w[0] for x in self.w)
        if len(self.w) != n:
            raise ValueError("weights and control points differ in count")
        if len(self.U) != n + self.p + 1:
            raise ValueError(
                f"knot vector must have n+p+1={n + self.p + 1} entries, "
                f"got {len(self.U)}")

    # -- span location --------------------------------------------------------

    def _span(self, u: F) -> int:
        """Knot span index k with U[k] <= u < U[k+1] (clamped to the last
        non-empty span at the right end). Exact rational comparisons."""
        n = len(self.cp) - 1
        if u >= self.U[n + 1]:
            return n
        if u <= self.U[self.p]:
            return self.p
        lo, hi = self.p, n + 1
        while hi - lo > 1:                  # binary search, exact
            mid = (lo + hi) // 2
            if u < self.U[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    # -- exact rational evaluation (de Boor in homogeneous coords) ------------

    def eval(self, t):
        """Exact point in ℚ³ at parameter ``t`` (rational in, rational out).
        Raises if any weight is irrational — use :meth:`eval_ci` for that."""
        if not self.exact_weights:
            raise ValueError(
                "curve has irrational weights — use eval_ci (certified)")
        u = F(t)
        k = self._span(u)
        p = self.p
        # homogeneous control points (w·x, w·y, w·z, w) for the active span
        d = []
        for j in range(p + 1):
            i = k - p + j
            wi = self.w[i]
            x, y, z = self.cp[i]
            d.append([wi * x, wi * y, wi * z, wi])
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = k - p + j
                denom = self.U[i + p - r + 1] - self.U[i]
                a = F(0) if denom == 0 else (u - self.U[i]) / denom
                b = 1 - a
                d[j] = [b * d[j - 1][c] + a * d[j][c] for c in range(4)]
        hx, hy, hz, hw = d[p]
        return (hx / hw, hy / hw, hz / hw)

    # -- certified evaluation (for irrational weights/parameters) -------------

    def eval_ci(self, t):
        """Certified point (three ``CInterval``s) — the enclosure-preserving
        path for irrational weights or parameters. Accepts rational or
        ``CInterval`` inputs and never loses the bracket."""
        u = t if isinstance(t, CInterval) else CInterval.exact(F(t))
        # span from the interval midpoint (a location choice, not a decision
        # that affects the certified value — the recurrence is continuous)
        k = self._span(u.mid)
        p = self.p
        d = []
        for j in range(p + 1):
            i = k - p + j
            wi = _ci(self.w[i])
            x, y, z = (_ci(v) for v in self.cp[i])
            d.append([wi * x, wi * y, wi * z, wi])
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = k - p + j
                denom = self.U[i + p - r + 1] - self.U[i]
                if denom == 0:
                    continue
                a = (u - _ci(self.U[i])) * _ci(F(1) / denom)
                b = _ci(1) - a
                d[j] = [b * d[j - 1][c] + a * d[j][c] for c in range(4)]
        hx, hy, hz, hw = d[p]
        inv = _ci_reciprocal(hw)
        return (hx * inv, hy * inv, hz * inv)

    # -- float evaluation (tessellation) --------------------------------------

    def eval_f(self, t: float):
        x, y, z = self.eval(F(t).limit_denominator(10 ** 9))
        return (float(x), float(y), float(z))

    def domain(self):
        return (float(self.U[self.p]), float(self.U[len(self.cp)]))


# -- constructors -------------------------------------------------------------

def bezier(control_points, weights=None) -> BSplineCurve:
    """A Bézier curve as a clamped B-spline on [0, 1]."""
    n = len(control_points)
    p = n - 1
    knots = [F(0)] * (p + 1) + [F(1)] * (p + 1)
    return BSplineCurve(p, control_points, knots, weights)


def _ci(x) -> CInterval:
    return x if isinstance(x, CInterval) else CInterval.exact(F(x))


def _ci_reciprocal(x: CInterval) -> CInterval:
    """1/x for an interval that strictly excludes zero (certified)."""
    s = x.sign()                                    # raises if straddles 0
    lo, hi = (F(1) / x.hi, F(1) / x.lo) if s > 0 else (F(1) / x.hi, F(1) / x.lo)
    return CInterval(min(lo, hi), max(lo, hi))


# -- K3.2: tensor-product NURBS surfaces --------------------------------------

def _deboor4(p: int, U, pts, u):
    """De Boor on homogeneous 4-vectors (exact rational). ``pts`` spans one
    curve; returns the 4-vector at parameter ``u``."""
    n = len(pts) - 1
    # span
    if u >= U[n + 1]:
        k = n
    elif u <= U[p]:
        k = p
    else:
        lo, hi = p, n + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if u < U[mid]:
                hi = mid
            else:
                lo = mid
        k = lo
    d = [list(pts[k - p + j]) for j in range(p + 1)]
    dim = len(d[0])
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = k - p + j
            denom = U[i + p - r + 1] - U[i]
            a = F(0) if denom == 0 else (u - U[i]) / denom
            b = 1 - a
            d[j] = [b * d[j - 1][c] + a * d[j][c] for c in range(dim)]
    return tuple(d[p])


def _hodograph4(p: int, U, pts):
    """Derivative curve of a (homogeneous) B-spline: degree p-1, control
    points p·(P[i+1]−P[i])/(U[i+p+1]−U[i+1]), knots U[1:-1]. Exact."""
    D = []
    for i in range(len(pts) - 1):
        denom = U[i + p + 1] - U[i + 1]
        s = F(0) if denom == 0 else F(p) / denom
        D.append(tuple(s * (pts[i + 1][c] - pts[i][c]) for c in range(4)))
    return p - 1, U[1:-1], D


class BSplineSurface:
    """A tensor-product NURBS surface: degrees (p, q), an nu×nv control
    net, clamped knot vectors U (nu+p+1) and V (nv+q+1), optional
    weights. Rational data at rational parameters evaluates exactly."""

    def __init__(self, degree_u: int, degree_v: int, control_net,
                 knots_u, knots_v, weights=None) -> None:
        self.p = int(degree_u)
        self.q = int(degree_v)
        self.cp = [[tuple(F(c) for c in pt) for pt in row] for row in control_net]
        self.nu = len(self.cp)
        self.nv = len(self.cp[0])
        if any(len(r) != self.nv for r in self.cp):
            raise ValueError("ragged control net")
        self.U = [F(u) for u in knots_u]
        self.V = [F(v) for v in knots_v]
        if len(self.U) != self.nu + self.p + 1:
            raise ValueError(f"knots_u wants {self.nu + self.p + 1} entries")
        if len(self.V) != self.nv + self.q + 1:
            raise ValueError(f"knots_v wants {self.nv + self.q + 1} entries")
        if weights is None:
            self.w = [[F(1)] * self.nv for _ in range(self.nu)]
        else:
            self.w = [[F(x) for x in row] for row in weights]
        # homogeneous net
        self.H = [[(self.w[i][j] * self.cp[i][j][0],
                    self.w[i][j] * self.cp[i][j][1],
                    self.w[i][j] * self.cp[i][j][2],
                    self.w[i][j]) for j in range(self.nv)]
                  for i in range(self.nu)]

    # -- exact evaluation ------------------------------------------------------

    def _eval_h(self, u, v):
        """Homogeneous 4-vector at (u, v): de Boor down v then across u."""
        u, v = F(u), F(v)
        col = [_deboor4(self.q, self.V, row, v) for row in self.H]
        return _deboor4(self.p, self.U, col, u)

    def eval(self, u, v):
        """Exact surface point in ℚ³ (rational in → rational out)."""
        hx, hy, hz, hw = self._eval_h(u, v)
        return (hx / hw, hy / hw, hz / hw)

    def eval_f(self, u: float, v: float):
        x, y, z = self.eval(F(u).limit_denominator(10 ** 9),
                            F(v).limit_denominator(10 ** 9))
        return (float(x), float(y), float(z))

    # -- exact partial derivatives (quotient rule in homogeneous space) -------

    def _partials_h(self, u, v):
        """(H, H_u, H_v) homogeneous 4-vectors, all exact."""
        u, v = F(u), F(v)
        col = [_deboor4(self.q, self.V, row, v) for row in self.H]
        H = _deboor4(self.p, self.U, col, u)
        pu, Uu, Du = _hodograph4(self.p, self.U, col)
        H_u = _deboor4(pu, Uu, Du, u) if pu >= 0 and Du else (F(0),) * 4
        # v-partial: hodograph each row in v, evaluate, then de Boor in u
        rows_dv = []
        for row in self.H:
            qv, Vv, Dv = _hodograph4(self.q, self.V, row)
            rows_dv.append(_deboor4(qv, Vv, Dv, v) if qv >= 0 and Dv
                           else (F(0),) * 4)
        H_v = _deboor4(self.p, self.U, rows_dv, u)
        return H, H_u, H_v

    def partials(self, u, v):
        """(S, S_u, S_v) in ℚ³ — the exact tangent plane data SSI needs.
        Quotient rule: S_u = (A_u·w − A·w_u)/w² with H = (A, w)."""
        H, H_u, H_v = self._partials_h(u, v)
        w, wu, wv = H[3], H_u[3], H_v[3]
        S = tuple(H[c] / w for c in range(3))
        S_u = tuple((H_u[c] * w - H[c] * wu) / (w * w) for c in range(3))
        S_v = tuple((H_v[c] * w - H[c] * wv) / (w * w) for c in range(3))
        return S, S_u, S_v

    def normal(self, u, v):
        """Exact (unnormalized) normal S_u × S_v in ℚ³."""
        _, su, sv = self.partials(u, v)
        return (su[1] * sv[2] - su[2] * sv[1],
                su[2] * sv[0] - su[0] * sv[2],
                su[0] * sv[1] - su[1] * sv[0])

    def domain(self):
        return ((float(self.U[self.p]), float(self.U[self.nu])),
                (float(self.V[self.q]), float(self.V[self.nv])))


def bezier_surface(control_net, weights=None) -> BSplineSurface:
    """A Bézier patch as a clamped B-spline surface on [0,1]²."""
    nu, nv = len(control_net), len(control_net[0])
    p, q = nu - 1, nv - 1
    ku = [F(0)] * (p + 1) + [F(1)] * (p + 1)
    kv = [F(0)] * (q + 1) + [F(1)] * (q + 1)
    return BSplineSurface(p, q, control_net, ku, kv, weights)


# -- K3.5: exact Bézier extraction (knot insertion to full multiplicity) ------

def _insert_knot_once(p: int, U, pts, u):
    """Boehm insertion of knot ``u`` into a (homogeneous or cartesian)
    control sequence. Exact: convex combinations only. Returns (U', pts')."""
    n = len(pts) - 1
    # span k: U[k] <= u < U[k+1]
    k = p
    while k < n + 1 and not (U[k] <= u < U[k + 1]):
        k += 1
    if k == n + 1:
        k = n
    dim = len(pts[0])
    out = [pts[0]]
    for i in range(1, len(pts) + 1):
        if i <= k - p:
            out.append(pts[i] if i < len(pts) else pts[-1])
        elif i <= k:
            denom = U[i + p] - U[i]
            a = F(0) if denom == 0 else (u - U[i]) / denom
            out.append(tuple((1 - a) * pts[i - 1][c] + a * pts[i][c]
                             for c in range(dim)))
        else:
            out.append(pts[i - 1])
    newU = sorted(list(U) + [u])
    return newU, out[:n + 2]


def bezier_segments(curve: "BSplineCurve"):
    """Split a (polynomial) B-spline curve into exact Bézier segments:
    [(u0, u1, [P0..Pp]), ...]. Knot insertion to full multiplicity."""
    if not curve.exact_weights or curve.rational:
        raise ValueError("bezier extraction: polynomial curves only (K3.6)")
    p = curve.p
    U = list(curve.U)
    pts = [tuple(pt) for pt in curve.cp]
    # insert every interior knot up to multiplicity p
    lo, hi = U[p], U[len(pts)]
    interior = sorted({u for u in U if lo < u < hi})
    for u in interior:
        while U.count(u) < p:
            U, pts = _insert_knot_once(p, U, pts, u)
    breaks = [lo] + interior + [hi]
    segs = []
    for j in range(len(breaks) - 1):
        segs.append((breaks[j], breaks[j + 1], pts[j * p: j * p + p + 1]))
    return segs


def bezier_patches(surface: "BSplineSurface"):
    """Split a B-spline surface into exact Bézier patches:
    [(u0, u1, v0, v1, net), ...] — insertion along u then along v.

    Polynomial surfaces yield 3-tuple cartesian nets; rational surfaces
    yield homogeneous 4-tuple nets (wx, wy, wz, w) — knot insertion is
    the same convex-combination recurrence in either space."""
    rational = any(w != F(1) for row in surface.w for w in row)
    p, q = surface.p, surface.q
    # --- u direction: treat each v-column of the net as a u-curve
    U = list(surface.U)
    src = surface.H if rational else surface.cp
    cols = [[src[i][j] for i in range(surface.nu)]
            for j in range(surface.nv)]
    lo_u, hi_u = U[p], U[surface.nu]
    int_u = sorted({u for u in U if lo_u < u < hi_u})
    for u in int_u:
        while U.count(u) < p:
            newU = None
            for j in range(len(cols)):
                nu2, cols[j] = _insert_knot_once(p, U, cols[j], u)
                newU = nu2
            U = newU
    # --- v direction on the refined net
    V = list(surface.V)
    rows = [[cols[j][i] for j in range(len(cols))]
            for i in range(len(cols[0]))]
    lo_v, hi_v = V[q], V[surface.nv]
    int_v = sorted({v for v in V if lo_v < v < hi_v})
    for v in int_v:
        while V.count(v) < q:
            newV = None
            for i in range(len(rows)):
                nv2, rows[i] = _insert_knot_once(q, V, rows[i], v)
                newV = nv2
            V = newV
    ub = [lo_u] + int_u + [hi_u]
    vb = [lo_v] + int_v + [hi_v]
    patches = []
    for a in range(len(ub) - 1):
        for b in range(len(vb) - 1):
            net = [[rows[a * p + i][b * q + j] for j in range(q + 1)]
                   for i in range(p + 1)]
            patches.append((ub[a], ub[a + 1], vb[b], vb[b + 1], net))
    return patches


# -- K6.0: second partials (polynomial surfaces; rational → K6.1) -------------

def _hodo_list(p, U, pts):
    """Hodograph as plain lists (degree p-1, knots U[1:-1])."""
    D = []
    for i in range(len(pts) - 1):
        denom = U[i + p + 1] - U[i + 1]
        s = F(0) if denom == 0 else F(p) / denom
        D.append(tuple(s * (pts[i + 1][c] - pts[i][c])
                       for c in range(len(pts[0]))))
    return p - 1, U[1:-1], D


def surface_partials2(surface: "BSplineSurface", u, v):
    """(S, S_u, S_v, S_uu, S_uv, S_vv) — all exact ℚ³ for a POLYNOMIAL
    surface (weights 1). Rational second partials arrive at K6.1."""
    if any(w != F(1) for row in surface.w for w in row):
        raise ValueError("second partials: polynomial surfaces only (K6.1)")
    u, v = F(u), F(v)
    p, q, U, V = surface.p, surface.q, surface.U, surface.V

    def eval_curve(pp, UU, pts, t):
        return _deboor4(pp, UU, pts, t) if pp >= 0 and pts else \
            (F(0),) * len(surface.cp[0][0])

    # rows evaluated in v, then u-curves (S, S_u, S_uu from u-hodographs)
    rows_v = [tuple(x) for x in
              [_deboor4(q, V, [tuple(pt) for pt in row], v)
               for row in [[pt for pt in r] for r in surface.cp]]]
    S = _deboor4(p, U, rows_v, u)
    pu, Uu, Du = _hodo_list(p, U, rows_v)
    S_u = eval_curve(pu, Uu, Du, u)
    puu, Uuu, Duu = _hodo_list(pu, Uu, Du) if pu >= 1 else (-1, [], [])
    S_uu = eval_curve(puu, Uuu, Duu, u)
    # v-hodograph rows → S_v, S_vv; and u-hodograph of S_v rows → S_uv
    rows_dv, rows_dvv = [], []
    for r in surface.cp:
        qv, Vv, Dv = _hodo_list(q, V, [tuple(pt) for pt in r])
        rows_dv.append(eval_curve(qv, Vv, Dv, v))
        if qv >= 1:
            qvv, Vvv, Dvv = _hodo_list(qv, Vv, Dv)
            rows_dvv.append(eval_curve(qvv, Vvv, Dvv, v))
        else:
            rows_dvv.append((F(0),) * 3)
    S_v = _deboor4(p, U, rows_dv, u)
    S_vv = _deboor4(p, U, rows_dvv, u)
    puv, Uuv, Duv = _hodo_list(p, U, rows_dv)
    S_uv = eval_curve(puv, Uuv, Duv, u)
    return S, S_u, S_v, S_uu, S_uv, S_vv
