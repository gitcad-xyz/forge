"""K3.3 — surface–surface intersection with complete branch detection.

The crown jewel of K3 (coverage plan): where float kernels genuinely
miss branches, this finds them all — provably, at a stated resolution.

Structure (all geometry exact rational; Bézier patches, weights 1):

1. **Subdivision branch detection.** De Casteljau subdivision of both
   patches (exact — convex combinations only) with bounding-box pruning
   via the convex-hull property: a Bézier patch lies inside the bbox of
   its control net, so a pair whose control-net boxes are disjoint
   *provably* does not intersect. Pruning therefore never discards a
   real intersection: at subdivision depth ``d`` every intersection
   branch is guaranteed to be hit by at least one surviving leaf pair —
   completeness at resolution 2^-d, stated, not hoped.

2. **Branch counting.** Surviving leaves are cells in A's parameter
   square; connected components (8-neighbour union-find) = branches.

3. **Certified refinement.** Per cell, a float Newton solve lands a
   point on the intersection; the *certificate* is exact: the rational
   residual |A(u,v) − B(s,t)|² is computed in ℚ and must be below tol².
   Points that fail to certify are dropped and reported honestly.

The empty case is a genuine differentiator: bbox-disjointness at the
top level *proves* non-intersection — a float sampler can only fail to
find what it never proves absent.
"""

from __future__ import annotations

from fractions import Fraction

F = Fraction


class BezierPatch:
    """A Bézier patch: (p+1)×(q+1) exact rational control net over a
    parameter box [u0,u1]×[v0,v1] of the original surface.

    Net entries are 3-tuples (polynomial) or homogeneous 4-tuples
    (wx, wy, wz, w) for a rational patch. The convex-hull property that
    bbox pruning relies on holds for rational patches with POSITIVE
    weights over the *cartesian* control points — enforced here."""

    __slots__ = ("net", "u0", "u1", "v0", "v1", "dim")

    def __init__(self, net, u0=F(0), u1=F(1), v0=F(0), v1=F(1)) -> None:
        self.net = [[tuple(F(c) for c in pt) for pt in row] for row in net]
        self.dim = len(self.net[0][0])
        if self.dim == 4:
            for row in self.net:
                for pt in row:
                    if pt[3] <= 0:
                        raise ValueError(
                            "rational patch: convex-hull pruning needs "
                            "positive weights (K3.7 for sign-varying)")
        self.u0, self.u1, self.v0, self.v1 = F(u0), F(u1), F(v0), F(v1)

    def bbox(self):
        """Cartesian control-net box — contains the patch (convex hull
        property; for rational nets, over points x/w with w>0)."""
        if self.dim == 3:
            xs = [pt for row in self.net for pt in row]
        else:
            xs = [(pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3])
                  for row in self.net for pt in row]
        lo = tuple(min(p[c] for p in xs) for c in range(3))
        hi = tuple(max(p[c] for p in xs) for c in range(3))
        return lo, hi

    def _split_rows(self, rows):
        """De Casteljau at 1/2 along a list of control rows; exact.
        Dimension-agnostic: homogeneous 4-vectors subdivide identically."""
        dim = self.dim
        left, right = [], []
        for row in rows:
            pts = [list(p) for p in row]
            lo = [tuple(pts[0])]
            hi = [tuple(pts[-1])]
            n = len(pts)
            for r in range(1, n):
                for i in range(n - r):
                    pts[i] = [(pts[i][c] + pts[i + 1][c]) / 2 for c in range(dim)]
                lo.append(tuple(pts[0]))
                hi.append(tuple(pts[n - r - 1]))
            left.append(lo)
            right.append(list(reversed(hi)))
        return left, right

    def split_u(self):
        """Split at the u-midpoint (rows of the net run along u)."""
        cols = list(map(list, zip(*self.net)))          # transpose: u-rows
        l, r = self._split_rows(cols)
        um = (self.u0 + self.u1) / 2
        return (BezierPatch(list(map(list, zip(*l))), self.u0, um, self.v0, self.v1),
                BezierPatch(list(map(list, zip(*r))), um, self.u1, self.v0, self.v1))

    def split_v(self):
        l, r = self._split_rows(self.net)
        vm = (self.v0 + self.v1) / 2
        return (BezierPatch(l, self.u0, self.u1, self.v0, vm),
                BezierPatch(r, self.u0, self.u1, vm, self.v1))

    def split4(self):
        a, b = self.split_u()
        return a.split_v() + b.split_v()


def _boxes_overlap(a, b) -> bool:
    (alo, ahi), (blo, bhi) = a, b
    return all(alo[c] <= bhi[c] and blo[c] <= ahi[c] for c in range(3))


def ssi_branches(A: BezierPatch, B: BezierPatch, depth: int = 5):
    """Complete branch detection at resolution 2^-depth.

    Returns ``(branches, leaf_pairs)`` where ``branches`` is a list of
    cell-sets on A's parameter square (each a connected component = one
    intersection branch) and ``leaf_pairs`` the surviving (cellA, cellB)
    parameter boxes. Empty list = *certified* non-intersection."""
    pairs = [(A, B)]
    for _ in range(depth):
        nxt = []
        for a, b in pairs:
            if not _boxes_overlap(a.bbox(), b.bbox()):
                continue                                # proven disjoint
            for sa in a.split4():
                ba = sa.bbox()
                for sb in b.split4():
                    if _boxes_overlap(ba, sb.bbox()):
                        nxt.append((sa, sb))
        pairs = nxt
        if not pairs:
            return [], []                               # certified empty

    # cluster surviving A-cells into branches (8-neighbour union-find)
    cells = {}
    for a, _ in pairs:
        cells.setdefault((a.u0, a.u1, a.v0, a.v1), []).append(a)
    keys = list(cells)
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def touch(k1, k2):
        return (k1[0] <= k2[1] and k2[0] <= k1[1]
                and k1[2] <= k2[3] and k2[2] <= k1[3])

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if touch(keys[i], keys[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)
    return list(groups.values()), pairs


# -- certified point refinement ------------------------------------------------

def _eval_patch(net, u, v):
    """De Casteljau evaluation of a Bézier net at exact (u, v) → ℚ³."""
    rows = [_dc1(row, v) for row in net]
    return _dc1(rows, u)


def _dc1(pts, t):
    pts = [tuple(p) for p in pts]
    n = len(pts)
    for r in range(1, n):
        pts = [tuple((1 - t) * pts[i][c] + t * pts[i + 1][c] for c in range(3))
               for i in range(n - r)]
    return pts[0]


def _partials_patch(net, u, v):
    """(S, S_u, S_v) of a Bézier net, exact (hodograph differences)."""
    rows = [_dc1(row, v) for row in net]
    S = _dc1(rows, u)
    p = len(rows) - 1
    du = [tuple(F(p) * (rows[i + 1][c] - rows[i][c]) for c in range(3))
          for i in range(p)]
    S_u = _dc1(du, u) if du else (F(0),) * 3
    q = len(net[0]) - 1
    dv_rows = []
    for row in net:
        dv_rows.append([tuple(F(q) * (row[j + 1][c] - row[j][c]) for c in range(3))
                        for j in range(q)])
    cols = [_dc1(r, v) for r in dv_rows]
    S_v = _dc1(cols, u) if cols and cols[0] else (F(0),) * 3
    return S, S_u, S_v


def refine_point(Anet, Bnet, u, v, s, t, iters: int = 12):
    """Float Newton on A(u,v) − B(s,t) = 0 (3 eqs, 4 unknowns; smallest-
    norm step via normal equations), then an EXACT residual certificate.

    Returns (u, v, s, t, ok, res2) — ``ok`` iff the exact rational
    |A−B|² is below 1e-20 (distance < 1e-10)."""
    import itertools

    uf, vf, sf, tf = (float(x) for x in (u, v, s, t))
    for _ in range(iters):
        ur, vr = F(uf).limit_denominator(10 ** 12), F(vf).limit_denominator(10 ** 12)
        sr, tr = F(sf).limit_denominator(10 ** 12), F(tf).limit_denominator(10 ** 12)
        Sa, Au, Av = _partials_patch(Anet, ur, vr)
        Sb, Bu, Bv = _partials_patch(Bnet, sr, tr)
        r = [float(Sa[c] - Sb[c]) for c in range(3)]
        if max(abs(x) for x in r) < 1e-14:
            break
        # J is 3x4: [Au Av -Bu -Bv]; solve J J^T y = -r, step = J^T y
        J = [[float(Au[c]), float(Av[c]), -float(Bu[c]), -float(Bv[c])]
             for c in range(3)]
        JJT = [[sum(J[i][k] * J[j][k] for k in range(4)) for j in range(3)]
               for i in range(3)]
        y = _solve3f(JJT, [-x for x in r])
        if y is None:
            break
        step = [sum(J[i][k] * y[i] for i in range(3)) for k in range(4)]
        uf, vf, sf, tf = uf + step[0], vf + step[1], sf + step[2], tf + step[3]
        uf = min(1.0, max(0.0, uf)); vf = min(1.0, max(0.0, vf))
        sf = min(1.0, max(0.0, sf)); tf = min(1.0, max(0.0, tf))
    ur, vr = F(uf).limit_denominator(10 ** 12), F(vf).limit_denominator(10 ** 12)
    sr, tr = F(sf).limit_denominator(10 ** 12), F(tf).limit_denominator(10 ** 12)
    pa = _eval_patch(Anet, ur, vr)
    pb = _eval_patch(Bnet, sr, tr)
    res2 = sum((pa[c] - pb[c]) ** 2 for c in range(3))   # EXACT rational
    return ur, vr, sr, tr, res2 < F(1, 10 ** 20), res2


def _solve3f(M, b):
    import copy
    a = [row[:] + [b[i]] for i, row in enumerate(copy.deepcopy(M))]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-30:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(3):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [a[r][k] - f * a[col][k] for k in range(4)]
    return [a[i][3] / a[i][i] for i in range(3)]


def ssi(A: BezierPatch, B: BezierPatch, depth: int = 5):
    """Full SSI: branches + one certified point per surviving cell.

    Returns {"branches": n, "points": [(u,v,s,t)...] certified,
    "uncertified": count, "empty_certified": bool}."""
    branches, pairs = ssi_branches(A, B, depth)
    if not pairs:
        return {"branches": 0, "points": [], "uncertified": 0,
                "empty_certified": True}
    pts, bad = [], 0
    seen = set()
    for a, b in pairs:
        key = (a.u0, a.v0)
        if key in seen:
            continue
        seen.add(key)
        um, vm = (a.u0 + a.u1) / 2, (a.v0 + a.v1) / 2
        sm, tm = (b.u0 + b.u1) / 2, (b.v0 + b.v1) / 2
        u, v, s, t, ok, _ = refine_point(A.net, B.net, um, vm, sm, tm)
        if ok:
            pts.append((u, v, s, t))
        else:
            bad += 1
    return {"branches": len(branches), "points": pts, "uncertified": bad,
            "empty_certified": False}


# -- K3.5: SSI over B-spline surfaces + ordered polylines ---------------------

def _cluster(keys):
    """Union-find over parameter boxes (closed-touch adjacency)."""
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def touch(k1, k2):
        return (k1[0] <= k2[1] and k2[0] <= k1[1]
                and k1[2] <= k2[3] and k2[2] <= k1[3])

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if touch(keys[i], keys[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)
    return list(groups.values())


def ssi_surfaces(A, B, depth: int = 4):
    """SSI between two (polynomial) B-spline surfaces: exact Bézier
    extraction, pairwise subdivision detection, branch clustering in the
    ORIGINAL parameter squares (branches crossing patch boundaries stay
    one branch), certified points in global parameters."""
    from forgekernel.nurbs import bezier_patches

    pa = [BezierPatch(net, u0, u1, v0, v1)
          for u0, u1, v0, v1, net in bezier_patches(A)]
    pb = [BezierPatch(net, u0, u1, v0, v1)
          for u0, u1, v0, v1, net in bezier_patches(B)]
    all_pairs = []
    for a in pa:
        ba = a.bbox()
        for b in pb:
            if not _boxes_overlap(ba, b.bbox()):
                continue                        # proven disjoint
            _, pairs = ssi_branches(a, b, depth)
            all_pairs.extend(pairs)
    if not all_pairs:
        return {"branches": 0, "points": [], "uncertified": 0,
                "empty_certified": True}
    cells = {}
    for a, b in all_pairs:
        cells.setdefault((a.u0, a.u1, a.v0, a.v1), (a, b))
    branches = _cluster(list(cells))
    pts, bad = [], 0
    for key, (a, b) in cells.items():
        um, vm = (a.u0 + a.u1) / 2, (a.v0 + a.v1) / 2
        sm, tm = (b.u0 + b.u1) / 2, (b.v0 + b.v1) / 2
        # local params inside the leaf patches are recovered by refine on
        # the ORIGINAL patch nets: find the top patches containing these
        # cells — the leaves carry global boxes, so evaluate via A directly
        u, v, s, t, ok, _ = _refine_global(A, B, um, vm, sm, tm)
        if ok:
            pts.append((u, v, s, t))
        else:
            bad += 1
    return {"branches": len(branches), "points": pts, "uncertified": bad,
            "empty_certified": False}


def _refine_global(A, B, u, v, s, t, iters: int = 12):
    """refine_point against full B-spline surfaces (global parameters)."""
    uf, vf, sf, tf = (float(x) for x in (u, v, s, t))
    (au0, au1), (av0, av1) = A.domain()
    (bu0, bu1), (bv0, bv1) = B.domain()
    for _ in range(iters):
        ur = F(uf).limit_denominator(10 ** 12)
        vr = F(vf).limit_denominator(10 ** 12)
        sr = F(sf).limit_denominator(10 ** 12)
        tr = F(tf).limit_denominator(10 ** 12)
        Sa, Au, Av = A.partials(ur, vr)
        Sb, Bu, Bv = B.partials(sr, tr)
        r = [float(Sa[c] - Sb[c]) for c in range(3)]
        if max(abs(x) for x in r) < 1e-14:
            break
        J = [[float(Au[c]), float(Av[c]), -float(Bu[c]), -float(Bv[c])]
             for c in range(3)]
        JJT = [[sum(J[i][k] * J[j][k] for k in range(4)) for j in range(3)]
               for i in range(3)]
        y = _solve3f(JJT, [-x for x in r])
        if y is None:
            break
        step = [sum(J[i][k] * y[i] for i in range(3)) for k in range(4)]
        uf = min(au1, max(au0, uf + step[0]))
        vf = min(av1, max(av0, vf + step[1]))
        sf = min(bu1, max(bu0, sf + step[2]))
        tf = min(bv1, max(bv0, tf + step[3]))
    ur = F(uf).limit_denominator(10 ** 12)
    vr = F(vf).limit_denominator(10 ** 12)
    sr = F(sf).limit_denominator(10 ** 12)
    tr = F(tf).limit_denominator(10 ** 12)
    pa_ = A.eval(ur, vr)
    pb_ = B.eval(sr, tr)
    res2 = sum((pa_[c] - pb_[c]) ** 2 for c in range(3))
    return ur, vr, sr, tr, res2 < F(1, 10 ** 20), res2


def polyline(points_xyz):
    """Order 3D points into a polyline by greedy nearest-neighbour
    chaining from an extreme point (float; a render/report artifact —
    the certified objects are the points themselves)."""
    if not points_xyz:
        return []
    pts = [tuple(float(c) for c in p) for p in points_xyz]
    start = max(range(len(pts)),
                key=lambda i: sum((pts[i][c] - pts[0][c]) ** 2 for c in range(3)))
    todo = set(range(len(pts)))
    order = [start]
    todo.discard(start)
    while todo:
        last = pts[order[-1]]
        nxt = min(todo, key=lambda i: sum((pts[i][c] - last[c]) ** 2
                                          for c in range(3)))
        order.append(nxt)
        todo.discard(nxt)
    return [pts[i] for i in order]
