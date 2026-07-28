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

2. **Branch counting.** At detection level, surviving leaves are cells
   in A's parameter square and connected components (8-neighbour
   union-find) = branches. The RESOLVED entry points (``ssi``,
   ``ssi_surfaces``, ``ssi_curves``, ``ssi_trim_loops``, ``ssi_chains``)
   cluster in the resolved cell graph instead: two point-bearing cells
   are neighbours only if their A-boxes touch AND some surviving B-cell
   of one touches some surviving B-cell of the other. A connected curve
   is covered by surviving leaf PAIRS whose 4-dim (u,v,s,t) boxes
   overlap consecutively, so the resolved graph never splits a genuine
   branch; it separates disjoint curves the moment EITHER parameter
   domain resolves the gap (A-only clustering welded curves that pass
   close in A's domain but far apart in B's — the double-bump defect).
   The ``closed`` flag is likewise exact: a branch is closed iff the
   union of its cells encloses a hole in A's parameter domain
   (:func:`_branch_is_closed`), a certificate at the stated resolution,
   replacing the float median-gap heuristic.

3. **Certified refinement.** Per cell, a float Newton solve lands a
   point on the intersection; the *certificate* is exact: the rational
   residual |A(u,v) − B(s,t)|² is computed in ℚ and must be below tol².

4. **Per-cell existence resolution.** A cell whose refinement fails to
   certify is never silently dropped (that punches holes in the trim
   loops a boolean would build). The resolver deepens the subdivision of
   exactly that cell's pairs until one of three certified outcomes:
   a residual-certified point INSIDE the cell (existence), every
   descendant pair pruned (exclusion — the same bbox-disjointness proof
   detection uses), or a named refusal ``SsiCellUncertified`` carrying
   the cells (tangential contact, or resolution budget).

The empty case is a genuine differentiator: bbox-disjointness at the
top level *proves* non-intersection — a float sampler can only fail to
find what it never proves absent.
"""

from __future__ import annotations

from fractions import Fraction

F = Fraction

# resolution budget for the per-cell existence test: extra subdivision
# levels tried on a cell whose first refinement failed to certify
_RESOLVE_LEVELS = 6


class SsiCellUncertified(ValueError):
    """Structured refusal: surviving subdivision cell(s) could be neither
    certified (no residual-certified point found inside the cell) nor
    proven empty (subdivision pruning never exhausted the cell's
    descendant pairs) within the resolution budget.

    Tangential contact is the canonical cause — bbox pruning cannot
    separate touching surfaces, and Newton's normal equations are
    singular where the tangent planes coincide, so no transversal
    certificate exists to find. Refusing by name is the finished answer
    (ADR-0019); the alternative was a silent drop that punches holes in
    the trim loops a boolean builds from these points.

    ``cells`` — [(a_box, b_box), …]: the unresolved cells as
    (u0, u1, v0, v1) Fraction boxes in BOTH surfaces' parameter domains.
    """

    def __init__(self, cells, depth: int, extra: int) -> None:
        self.cells = list(cells)
        self.depth = depth
        self.extra = extra
        super().__init__(
            f"ssi_cell_uncertified: {len(self.cells)} surviving cell(s) "
            f"neither certified nor proven empty at depth {depth} plus "
            f"{extra} resolution levels — tangential or near-tangential "
            f"contact has no transversal certificate; raise depth only "
            f"if the contact is believed transversal")


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


def _resolve_cell_pairs(pairs, cell, newton, extra: int = _RESOLVE_LEVELS,
                        cap: int = 256, tries: int = 8, constrained=None):
    """Certified existence/exclusion for ONE surviving cell whose first
    Newton refinement failed to certify — the test that turns a silent
    drop into a classification.

    ``pairs`` are the surviving (A-leaf, B-leaf) ``BezierPatch`` pairs
    sharing this A-cell, ``cell`` its (u0, u1, v0, v1) box, and
    ``newton(um, vm, sm, tm)`` the certified refinement in the caller's
    parameter convention. Deepens the subdivision of exactly these pairs,
    retrying certification from the refined midpoints. Returns one of

    * ``("point", (u, v, s, t))`` — a residual-certified point INSIDE the
      cell (exact ℚ containment; a certified point elsewhere on the curve
      does not classify THIS cell);
    * ``("empty", None)`` — every descendant pair pruned by control-net
      bbox disjointness: the cell provably contains no intersection.
      Sound only because no pair is ever discarded — growth past ``cap``
      returns unresolved instead of truncating;
    * ``("unresolved", None)`` — budget exhausted with pairs alive and no
      certificate either way. Tangential contact lands exactly here:
      pruning cannot separate touching surfaces and Newton's normal
      equations go singular where the tangent planes coincide."""
    u0, u1, v0, v1 = cell
    cur = list(pairs)
    # Matched-corner probes first: a contact sitting exactly ON a cell
    # corner (domain edges meeting, or the curve grazing a dyadic corner)
    # is invisible to midpoint-started Newton — the iteration stalls
    # against the domain clamp — but certifies directly AT the corner,
    # residual exactly 0, so the certificate alone (0 Newton iterations)
    # decides and costs one exact evaluation per corner.
    for a, b in cur:
        for (ua, va), (ub, vb) in (((a.u0, a.v0), (b.u0, b.v0)),
                                   ((a.u1, a.v0), (b.u1, b.v0)),
                                   ((a.u0, a.v1), (b.u0, b.v1)),
                                   ((a.u1, a.v1), (b.u1, b.v1))):
            u, v, s, t, ok, _ = newton(ua, va, ub, vb, 0)
            if ok and u0 <= u <= u1 and v0 <= v <= v1:
                return ("point", (u, v, s, t))
    for _ in range(extra):
        nxt = []
        for a, b in cur:
            for sa in a.split4():
                ba = sa.bbox()
                for sb in b.split4():
                    if _boxes_overlap(ba, sb.bbox()):
                        nxt.append((sa, sb))
        if not nxt:
            return ("empty", None)                  # certified exclusion
        if len(nxt) > cap:
            return ("unresolved", None)             # never truncate
        step = max(1, len(nxt) // tries)            # spread the retries
        for a, b in nxt[::step][:tries]:
            um, vm = (a.u0 + a.u1) / 2, (a.v0 + a.v1) / 2
            sm, tm = (b.u0 + b.u1) / 2, (b.v0 + b.v1) / 2
            u, v, s, t, ok, _ = newton(um, vm, sm, tm)
            if ok and u0 <= u <= u1 and v0 <= v <= v1:
                return ("point", (u, v, s, t))      # certified existence
        cur = nxt
    # Last resort, tried only AFTER the whole budget (so every input that
    # resolved before still resolves to the identical answer): a curve
    # endpoint sitting exactly ON a domain border — e.g. the border cell
    # of one surface touching a dyadic gridline of the other — stalls
    # unconstrained Newton against the clamp and is no matched corner.
    # Pin each hugged border EXACTLY (``constrained`` wraps
    # :func:`_refine_constrained`) and let the exact residual decide.
    if constrained is not None and cur:
        seen = set()
        for a, b in cur[:tries]:
            hugged = constrained["hugged"]((a, b))
            mid = ((a.u0 + a.u1) / 2, (a.v0 + a.v1) / 2,
                   (b.u0 + b.u1) / 2, (b.v0 + b.v1) / 2)
            cands = [dict(hugged)] if len(hugged) > 1 else []
            cands += [{i: val} for i, val in hugged.items()]
            for fixed in cands:
                key = tuple(sorted(fixed.items()))
                if not fixed or key in seen:
                    continue
                seen.add(key)
                pt, ok, _ = constrained["solve"](mid, fixed)
                if ok and u0 <= pt[0] <= u1 and v0 <= pt[1] <= v1:
                    return ("point", pt)
    return ("unresolved", None)


def ssi(A: BezierPatch, B: BezierPatch, depth: int = 5):
    """Full SSI: branches + one certified point per surviving cell, with
    every cell CLASSIFIED — a certified point, proven empty by the
    resolver, or the pair refuses (:class:`SsiCellUncertified`). Never a
    silent drop.

    Returns {"branches": n (over point-bearing cells), "points":
    [(u,v,s,t)...] certified, "uncertified": 0, "empty_certified": bool,
    "cells": surviving-cell count, "empty_cells": cells proven empty,
    "tightened": cells certified only after deepening}. ``uncertified``
    is kept for compatibility and is 0 by construction — an unresolved
    cell raises instead of being counted."""
    _, pairs = ssi_branches(A, B, depth)
    if not pairs:
        return {"branches": 0, "points": [], "uncertified": 0,
                "empty_certified": True, "cells": 0, "empty_cells": 0,
                "tightened": 0}
    groups: dict = {}
    for a, b in pairs:
        groups.setdefault((a.u0, a.u1, a.v0, a.v1), []).append((a, b))

    def newton(um, vm, sm, tm, iters=12):
        return refine_point(A.net, B.net, um, vm, sm, tm, iters)

    pts, alive, unresolved = [], [], []
    empty = tightened = 0
    for cell, grp in groups.items():
        a0, b0 = grp[0]
        um, vm = (a0.u0 + a0.u1) / 2, (a0.v0 + a0.v1) / 2
        sm, tm = (b0.u0 + b0.u1) / 2, (b0.v0 + b0.v1) / 2
        u, v, s, t, ok, _ = newton(um, vm, sm, tm)
        if ok:
            pts.append((u, v, s, t))
            alive.append(cell)
            continue
        verdict, pt = _resolve_cell_pairs(grp, cell, newton)
        if verdict == "point":
            pts.append(pt)
            alive.append(cell)
            tightened += 1
        elif verdict == "empty":
            empty += 1
        else:
            unresolved.append((cell, (b0.u0, b0.u1, b0.v0, b0.v1)))
    if unresolved:
        raise SsiCellUncertified(unresolved, depth, _RESOLVE_LEVELS)
    bmap = {cell: [(b.u0, b.u1, b.v0, b.v1) for _, b in groups[cell]]
            for cell in alive}
    return {"branches": len(_cluster_resolved(alive, bmap)), "points": pts,
            "uncertified": 0, "empty_certified": not pts,
            "cells": len(groups), "empty_cells": empty,
            "tightened": tightened}


# -- K3.5: SSI over B-spline surfaces + ordered polylines ---------------------

def _cluster_resolved(cells, bmap):
    """Union-find over point-bearing A-cells in the RESOLVED cell graph:
    two cells are neighbours iff their A-boxes touch (closed touch, exact
    ℚ) AND some surviving B-cell of one touches some surviving B-cell of
    the other. ``bmap`` maps each A-cell to its list of surviving B-side
    parameter boxes.

    Soundness (never splits a genuine branch): a connected intersection
    curve is covered by surviving leaf pairs, and any two points of the
    curve close along it lie in pairs whose 4-dim (u,v,s,t) boxes
    overlap — which is exactly "A-boxes touch and B-boxes touch". Every
    A-cell containing curve is point-bearing (it cannot be proven empty,
    and an unresolved cell raises), so the covering survives into this
    graph. Completeness of separation is resolution-bounded the other
    way: two disjoint curves closer than one cell in BOTH domains still
    merge (merge-not-miss, as at detection level) — but curves welded
    only by A-domain proximity now separate, because their B-cells are
    far apart."""
    keys = list(cells)
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys)):
        bi = bmap[keys[i]]
        for j in range(i + 1, len(keys)):
            if not _boxes_touch_2d(keys[i], keys[j]):
                continue
            bj = bmap[keys[j]]
            if any(_boxes_touch_2d(b1, b2) for b1 in bi for b2 in bj):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)
    return list(groups.values())


def _branch_is_closed(cells, dom) -> bool:
    """Exact closure certificate for one branch: ``True`` iff the union
    of the branch's A-cells encloses a HOLE in A's parameter domain — a
    complement component with no path to the domain border.

    A closed intersection curve encloses area in the parameter domain,
    and at adequate resolution its cell strip is an annulus whose
    uncovered core is that hole; an open curve's strip deformation-
    retracts to an arc and encloses nothing. The test is combinatorial
    over exact ℚ coordinates (coordinate-compressed grid, never a
    raster): strip squares are those covered by some cell; the
    complement flood-fills inward from the domain border with
    4-adjacency, the topological dual of the strip's 8-adjacency
    clustering. Any unreached uncovered square is a hole.

    This replaces the float median-gap heuristic of the old
    ``_order_branch`` (wrong in both directions under execution) and the
    end-cell-adjacency rule (which called two merged open curves a loop).
    Resolution caveat, stated rather than hidden: a hole finer than one
    cell is invisible at this depth, so a genuinely closed loop whose
    interior is sub-cell reports ``False`` — raise the depth (tighten-
    or-refuse), the flag never guesses. The converse — an OPEN curve
    whose two interior endpoints seal a hole — needs curve endpoints
    strictly inside both domains, i.e. a tangential retreat, which
    refuses upstream as :class:`SsiCellUncertified` before reaching
    here."""
    import bisect

    (du0, du1), (dv0, dv1) = dom
    us = {F(du0), F(du1)}
    vs = {F(dv0), F(dv1)}
    for c in cells:
        us.add(F(c[0]))
        us.add(F(c[1]))
        vs.add(F(c[2]))
        vs.add(F(c[3]))
    us, vs = sorted(us), sorted(vs)
    nu, nv = len(us) - 1, len(vs) - 1
    if nu < 3 or nv < 3:
        return False                      # no room for an interior hole
    covered = [[False] * nv for _ in range(nu)]
    for c in cells:
        i0 = bisect.bisect_left(us, F(c[0]))
        i1 = bisect.bisect_left(us, F(c[1]))
        j0 = bisect.bisect_left(vs, F(c[2]))
        j1 = bisect.bisect_left(vs, F(c[3]))
        for i in range(i0, i1):
            for j in range(j0, j1):
                covered[i][j] = True
    seen = [[False] * nv for _ in range(nu)]
    stack = []
    for i in range(nu):
        for j in (0, nv - 1):
            if not covered[i][j] and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    for j in range(nv):
        for i in (0, nu - 1):
            if not covered[i][j] and not seen[i][j]:
                seen[i][j] = True
                stack.append((i, j))
    while stack:
        i, j = stack.pop()
        for i2, j2 in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= i2 < nu and 0 <= j2 < nv and not covered[i2][j2] \
                    and not seen[i2][j2]:
                seen[i2][j2] = True
                stack.append((i2, j2))
    return any(not covered[i][j] and not seen[i][j]
               for i in range(nu) for j in range(nv))


def _rstr(x) -> str:
    f = F(x)
    return f"{f.numerator}/{f.denominator}"


def _net_str(net):
    return [[[_rstr(c) for c in pt] for pt in row] for row in net]


def _ssi_boxes(A, B, depth: int, use_rust: bool | None):
    """Raw subdivision detection between two B-spline surfaces: exact
    Bézier extraction + pairwise subdivision, returning every surviving
    ``(a_box, b_box)`` leaf pair as Fraction 4-tuples. ``[]`` means
    certified-empty. The union of the a-boxes (resp. b-boxes) provably
    encloses the true intersection curve in A's (resp. B's) parameter
    domain — subdivision pruning never discards a real intersection.

    Detection — the hot loop — runs in the Rust port (``ssi_pairs``,
    oracle-verified bit-identical) when the extension is built; the Python
    path is the fallback and stays the executable spec."""
    from forgekernel.nurbs import bezier_patches

    rs = None
    if use_rust is not False:
        try:
            import forgekernel_rs as rs
        except ImportError:
            if use_rust is True:
                raise
            rs = None

    pa = [(u0, u1, v0, v1, net) for u0, u1, v0, v1, net in bezier_patches(A)]
    pb = [(u0, u1, v0, v1, net) for u0, u1, v0, v1, net in bezier_patches(B)]
    # (a_box, b_box) surviving leaf pairs, as Fraction 4-tuples
    boxes: list[tuple[tuple, tuple]] = []
    for au0, au1, av0, av1, na in pa:
        patch_a = BezierPatch(na, au0, au1, av0, av1)
        ba = patch_a.bbox()
        for bu0, bu1, bv0, bv1, nb in pb:
            patch_b = BezierPatch(nb, bu0, bu1, bv0, bv1)
            if not _boxes_overlap(ba, patch_b.bbox()):
                continue                        # proven disjoint
            if rs is not None:
                rows = rs.ssi_pairs(
                    _net_str(na), _net_str(nb), depth,
                    [_rstr(au0), _rstr(au1), _rstr(av0), _rstr(av1)],
                    [_rstr(bu0), _rstr(bu1), _rstr(bv0), _rstr(bv1)])
                for row in rows:
                    vals = [F(x) for x in row]
                    boxes.append((tuple(vals[:4]), tuple(vals[4:])))
            else:
                _, pairs = ssi_branches(patch_a, patch_b, depth)
                boxes.extend(((p.u0, p.u1, p.v0, p.v1),
                              (q.u0, q.u1, q.v0, q.v1)) for p, q in pairs)
    return boxes


def _clip_dyadic(p: BezierPatch, box):
    """Clip a Bézier span patch down to a dyadic descendant box by de
    Casteljau splits — exact, the same arithmetic the detection pass used
    to produce the cell in the first place."""
    u0, u1, v0, v1 = box
    for _ in range(64):
        if (p.u0, p.u1) == (u0, u1):
            break
        lo, hi = p.split_u()
        p = lo if u1 <= lo.u1 else hi
    else:
        raise AssertionError("cell box is not a dyadic descendant (u)")
    for _ in range(64):
        if (p.v0, p.v1) == (v0, v1):
            break
        lo, hi = p.split_v()
        p = lo if v1 <= lo.v1 else hi
    else:
        raise AssertionError("cell box is not a dyadic descendant (v)")
    return p


def _cell_patch(spans, box):
    """The exact Bézier sub-patch of a surface for one surviving cell:
    find the Bézier span containing the (dyadic-within-span) box, clip
    down to it."""
    u0, u1, v0, v1 = box
    for su0, su1, sv0, sv1, net in spans:
        if su0 <= u0 and u1 <= su1 and sv0 <= v0 and v1 <= sv1:
            return _clip_dyadic(BezierPatch(net, su0, su1, sv0, sv1), box)
    raise AssertionError("surviving cell lies in no Bézier span")


def _ssi_resolved(A, B, depth: int, use_rust: bool | None):
    """Shared certified-classification core for the surface SSI entry
    points. Every surviving A-cell ends classified: a certified point,
    proven empty (:func:`_resolve_cell_pairs`), or the whole pair refuses
    with :class:`SsiCellUncertified` — never a silent drop.

    Returns ``None`` for detection-level certified emptiness, else
    ``(points, branches, stats, bcells)`` where ``points`` maps each
    point-bearing A-cell box to its certified (u, v, s, t), ``branches``
    clusters those cells (connected components in A's parameter domain),
    ``stats`` is {"cells", "empty_cells", "tightened"}, and ``bcells``
    maps each point-bearing A-cell to the LIST of its surviving B-side
    cell boxes (border-proximity tolerances in B's domain; entry 0 is
    the historical representative)."""
    boxes = _ssi_boxes(A, B, depth, use_rust)
    if not boxes:
        return None
    groups: dict = {}
    for abox, bbox_ in boxes:
        groups.setdefault(abox, []).append(bbox_)

    def newton(um, vm, sm, tm, iters=12):
        return _refine_global(A, B, um, vm, sm, tm, iters)

    adom, bdom = A.domain(), B.domain()
    bounds = (tuple(map(F, adom[0])), tuple(map(F, adom[1])),
              tuple(map(F, bdom[0])), tuple(map(F, bdom[1])))

    def hugged(pair):
        """Domain borders a surviving leaf pair's boxes sit exactly on:
        {param index: exact border value} — the pin candidates for a
        border-constrained certification probe."""
        a, b = pair
        box = (a.u0, a.u1, a.v0, a.v1, b.u0, b.u1, b.v0, b.v1)
        out = {}
        for i in range(4):
            lo, hi = bounds[i]
            if box[2 * i] == lo:
                out[i] = lo
            elif box[2 * i + 1] == hi:
                out[i] = hi
        return out

    constrained = {"hugged": hugged,
                   "solve": lambda pt, fixed: _refine_constrained(
                       A, B, pt, fixed)}

    spans = None
    points: dict = {}
    bcells: dict = {}
    unresolved = []
    empty = tightened = 0
    for abox, bbs in groups.items():
        b0 = bbs[0]
        um, vm = (abox[0] + abox[1]) / 2, (abox[2] + abox[3]) / 2
        sm, tm = (b0[0] + b0[1]) / 2, (b0[2] + b0[3]) / 2
        u, v, s, t, ok, _ = newton(um, vm, sm, tm)
        if ok:
            points[abox] = (u, v, s, t)
            bcells[abox] = list(bbs)
            continue
        if spans is None:
            from forgekernel.nurbs import bezier_patches
            spans = (list(bezier_patches(A)), list(bezier_patches(B)))
        grp = [(_cell_patch(spans[0], abox), _cell_patch(spans[1], bb))
               for bb in bbs]
        verdict, pt = _resolve_cell_pairs(grp, abox, newton,
                                          constrained=constrained)
        if verdict == "point":
            points[abox] = pt
            bcells[abox] = list(bbs)
            tightened += 1
        elif verdict == "empty":
            empty += 1
        else:
            unresolved.append((abox, b0))
    if unresolved:
        raise SsiCellUncertified(unresolved, depth, _RESOLVE_LEVELS)
    branches = _cluster_resolved(list(points), bcells)
    return points, branches, {"cells": len(groups), "empty_cells": empty,
                              "tightened": tightened}, bcells


def ssi_strips(A, B, depth: int = 4, use_rust: bool | None = None):
    """The certified trim-curve ENCLOSURE on both parameter domains: the
    deduplicated sets of surviving subdivision cells, ``(a_cells,
    b_cells)``, each a set of (u0, u1, v0, v1) Fraction boxes at
    resolution 2^-depth (within each Bézier span).

    The true intersection curve provably lies inside the union of the
    a-cells in A's domain AND inside the union of the b-cells in B's
    domain — bbox pruning never discards a real intersection. This is
    the raw detection-level enclosure; the boolean assembly reads the
    certified-PRUNED strips from :func:`ssi_chains` instead (every
    dropped cell proven empty — the B side here is ~3× inflated by
    3D-box smear), and brackets flux through them
    (:func:`forgekernel.bsolid.certified_trim_flux`). The chained
    polyline of :func:`ssi_curves` is only a float rendering aid.
    ``(set(), set())`` means certified non-intersection."""
    boxes = _ssi_boxes(A, B, depth, use_rust)
    return ({abox for abox, _ in boxes}, {bbox_ for _, bbox_ in boxes})


def ssi_surfaces(A, B, depth: int = 4, use_rust: bool | None = None):
    """SSI between two B-spline surfaces: certified points in global
    parameters plus the branch count, every surviving cell classified
    (point / proven-empty / :class:`SsiCellUncertified` — see
    :func:`_resolve_cell_pairs`). See :func:`ssi_curves` for the same
    points chained into ordered per-branch polylines."""
    res = _ssi_resolved(A, B, depth, use_rust)
    if res is None:
        return {"branches": 0, "points": [], "uncertified": 0,
                "empty_certified": True, "cells": 0, "empty_cells": 0,
                "tightened": 0}
    points, branches, stats, _ = res
    return {"branches": len(branches), "points": list(points.values()),
            "uncertified": 0, "empty_certified": not points, **stats}


def ssi_curves(A, B, depth: int = 4, use_rust: bool | None = None):
    """Ordered SSI output: the certified intersection points chained into
    one parameter-space polyline **per branch** (a branch = a connected
    component of the RESOLVED cell graph — A-boxes touching AND surviving
    B-cells touching, see :func:`_cluster_resolved`). Within a branch the
    points are ordered along the curve; ``closed`` is the exact strip
    certificate of :func:`_branch_is_closed` (the branch's cells enclose
    a hole in A's parameter domain), never a float heuristic.

    Returns ``{"curves": [{"points": [(u,v,s,t)…] (exact ℚ),
    "xyz": [(x,y,z)…] (float, for render), "closed": bool}, …],
    "uncertified": 0, "empty_certified": bool, "cells": n,
    "empty_cells": n, "tightened": n}``. The points are the certified
    objects (residual < 1e-20); the ordering is a float convenience
    layered on top. Every surviving cell is classified — a cell that can
    be neither certified nor proven empty raises
    :class:`SsiCellUncertified` instead of being dropped."""
    res = _ssi_resolved(A, B, depth, use_rust)
    if res is None:
        return {"curves": [], "uncertified": 0, "empty_certified": True,
                "cells": 0, "empty_cells": 0, "tightened": 0}
    points, branches, stats, _ = res
    adom = A.domain()
    curves = []
    for group in branches:
        items = _order_chain([(abox, points[abox]) for abox in group])
        ordered = [pt for _, pt in items]
        closed = _branch_is_closed(group, adom)
        xyz = [tuple(float(c) for c in A.eval(p[0], p[1])) for p in ordered]
        curves.append({"points": ordered, "xyz": xyz, "closed": closed})
    return {"curves": curves, "uncertified": 0,
            "empty_certified": not points, **stats}


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


# -- K7 gap 2: SSI branches → trim loops on BOTH parameter domains ------------

class TrimLoopUnstitchable(ValueError):
    """Structured refusal: an SSI branch could not be turned into a
    closed trim loop on both parameter domains.

    Causes, each named in the message:

    * ``open_interior`` — an open branch's endpoint lies strictly inside
      a domain and no domain-border crossing certifies near it. The
      curve genuinely ends mid-face (it leaves the OTHER patch through
      that patch's edge); closing the loop there needs the other
      surface's edge curves — boolean-assembly work, refused here rather
      than guessed.
    * ``border_uncertified`` — an endpoint sits near a border but the
      constrained Newton solve found no residual-certified crossing
      (near-tangential exit).
    * ``resolution`` — too few certified points to form a loop at this
      depth; raise the depth.

    Never returns a partial or guessed loop: a wrong trim loop builds a
    wrong shell (ADR-0019)."""

    def __init__(self, branch: int, reason: str, detail: str = "") -> None:
        self.branch = branch
        self.reason = reason
        msg = f"trim_loop_unstitchable[{reason}]: branch {branch}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


def _solve_lsqf(M, b):
    """Small dense float solve (k ≤ 3) by Gaussian elimination with
    partial pivoting; None if singular. Float is fine: the answer is only
    a Newton step — the certificate downstream is exact."""
    k = len(b)
    a = [list(M[i]) + [b[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-30:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(k):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [a[r][c] - f * a[col][c] for c in range(k + 1)]
    return [a[i][k] / a[i][i] for i in range(k)]


def _refine_constrained(A, B, pt, fixed, iters: int = 16):
    """Newton on A(u,v) − B(s,t) = 0 with some parameters FIXED exactly
    (typically at domain-border values), least-squares over the free
    ones, then the same EXACT residual certificate as :func:`refine_point`.

    ``fixed`` maps parameter index (0=u, 1=v, 2=s, 3=t) → exact ℚ value.
    Returns ``((u, v, s, t), ok, res2)`` — the fixed entries are carried
    through exactly, so a border-constrained solution lies EXACTLY on
    the border; ``ok`` iff the exact |A−B|² < 1e-20."""
    (au0, au1), (av0, av1) = A.domain()
    (bu0, bu1), (bv0, bv1) = B.domain()
    bounds = ((au0, au1), (av0, av1), (bu0, bu1), (bv0, bv1))
    vals = [float(x) for x in pt]
    for i, val in fixed.items():
        vals[i] = float(val)
    free = [i for i in range(4) if i not in fixed]
    if not free:
        # All four parameters pinned: the candidate point is fully
        # determined — there is nothing to SOLVE, only the exact residual
        # certificate to check (the matched-corner-probe idiom, 0 Newton
        # iterations). This arises when a corner cell of one surface's
        # domain pairs with a corner cell of the other's (e.g. exactly
        # coincident faces sharing a parameter-domain corner); raising
        # here used to escape the seam as a raw ValueError (the released
        # 0.9.8 coplanar-loft crash) instead of letting the caller's
        # structured refusal machinery classify the cell.
        out = tuple(fixed[i] for i in range(4))
        pa = A.eval(out[0], out[1])
        pb = B.eval(out[2], out[3])
        res2 = sum((pa[c] - pb[c]) ** 2 for c in range(3))
        return out, res2 < F(1, 10 ** 20), res2

    def _rat():
        return tuple(fixed[i] if i in fixed
                     else F(vals[i]).limit_denominator(10 ** 12)
                     for i in range(4))

    for _ in range(iters):
        ur, vr, sr, tr = _rat()
        Sa, Au, Av = A.partials(ur, vr)
        Sb, Bu, Bv = B.partials(sr, tr)
        r = [float(Sa[c] - Sb[c]) for c in range(3)]
        if max(abs(x) for x in r) < 1e-14:
            break
        cols = (Au, Av, Bu, Bv)
        J = [[(1 if i < 2 else -1) * float(cols[i][c]) for i in free]
             for c in range(3)]
        k = len(free)
        JtJ = [[sum(J[c][a_] * J[c][b_] for c in range(3)) for b_ in range(k)]
               for a_ in range(k)]
        Jtr = [sum(J[c][a_] * r[c] for c in range(3)) for a_ in range(k)]
        step = _solve_lsqf(JtJ, [-x for x in Jtr])
        if step is None:
            break
        for j, i in enumerate(free):
            lo, hi = bounds[i]
            vals[i] = min(float(hi), max(float(lo), vals[i] + step[j]))
    out = _rat()
    pa = A.eval(out[0], out[1])
    pb = B.eval(out[2], out[3])
    res2 = sum((pa[c] - pb[c]) ** 2 for c in range(3))
    return out, res2 < F(1, 10 ** 20), res2


def _order_chain(items):
    """Greedy nearest-neighbour ordering of ``[(a_cell, point), …]`` in
    A's (u, v), started from an endpoint via the double-sweep diameter
    heuristic (furthest point from an arbitrary sample is an endpoint of
    an open arc), carrying the cells along. Float ordering over exact
    points (a report artifact; the points stay the certified data —
    topology is decided by :func:`_cluster_resolved` and
    :func:`_branch_is_closed`, never by this ordering)."""
    n = len(items)
    if n <= 1:
        return list(items)
    uv = [(float(p[0]), float(p[1])) for _, p in items]

    def d2(i, j):
        return (uv[i][0] - uv[j][0]) ** 2 + (uv[i][1] - uv[j][1]) ** 2

    start = max(range(n), key=lambda i: d2(i, 0))
    todo = set(range(n))
    todo.discard(start)
    order = [start]
    while todo:
        last = order[-1]
        nxt = min(todo, key=lambda i: d2(i, last))
        order.append(nxt)
        todo.discard(nxt)
    return [items[i] for i in order]


def _boxes_touch_2d(k1, k2) -> bool:
    """Closed-touch adjacency of two parameter boxes — exact ℚ."""
    return (k1[0] <= k2[1] and k2[0] <= k1[1]
            and k1[2] <= k2[3] and k2[2] <= k1[3])


def _border_position(dom, p):
    """Arc-length position of ``p`` along the domain rectangle's border,
    counterclockwise from (u0, v0); ``None`` if p is not exactly on the
    border. Exact ℚ."""
    (u0, u1), (v0, v1) = dom
    u, v = p
    if v == v0 and u0 <= u <= u1:
        return u - u0
    W, H = u1 - u0, v1 - v0
    if u == u1 and v0 <= v <= v1:
        return W + (v - v0)
    if v == v1 and u0 <= u <= u1:
        return W + H + (u1 - u)
    if u == u0 and v0 <= v <= v1:
        return 2 * W + H + (v1 - v)
    return None


def _border_corners(dom, p_from, p_to):
    """The domain corners passed when walking the border counterclockwise
    from ``p_from`` to ``p_to`` (both exactly on the border), in walk
    order, endpoints excluded. Exact ℚ."""
    (u0, u1), (v0, v1) = dom
    P = 2 * ((u1 - u0) + (v1 - v0))
    a = _border_position(dom, p_from)
    b = _border_position(dom, p_to)
    span = (b - a) % P
    out = []
    for c in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        cp = (_border_position(dom, c) - a) % P
        if 0 < cp < span:
            out.append((cp, c))
    return [c for _, c in sorted(out)]


def _cell_hits_border(cell, dom) -> bool:
    """Does the parameter box touch the domain rectangle's border?"""
    (lo0, hi0), (lo1, hi1) = dom
    c0, c1, c2, c3 = cell
    return c0 <= lo0 or c1 >= hi0 or c2 <= lo1 or c3 >= hi1


def _extend_end(A, B, pt, acell, bcell):
    """Certified border crossing near an open chain's endpoint: try
    constrained Newton solves with parameters pinned EXACTLY to nearby
    domain borders — larger constraint sets first, so an endpoint that
    exits both domains at once (shared edge geometry) lands exactly on
    BOTH borders. Returns the certified (u, v, s, t) or ``None``.

    Candidate borders are those within 2 cells of the endpoint (the cell
    widths are the detection resolution); the filter is float-flavored
    but harmless — the accepted answer carries an exact residual
    certificate and exact in-domain checks."""
    from itertools import combinations

    (au0, au1), (av0, av1) = A.domain()
    (bu0, bu1), (bv0, bv1) = B.domain()
    bounds = ((au0, au1), (av0, av1), (bu0, bu1), (bv0, bv1))
    widths = (acell[1] - acell[0] or F(1), acell[3] - acell[2] or F(1),
              bcell[1] - bcell[0] or F(1), bcell[3] - bcell[2] or F(1))
    near = []
    for i in range(4):
        for bord in bounds[i]:
            d = abs(pt[i] - bord) / widths[i]
            if d <= 2:
                near.append((d, i, bord))
    near.sort()
    cand_sets = []
    for size in (3, 2, 1):
        for combo in combinations(near, size):
            if len({c[1] for c in combo}) == size:
                cand_sets.append(combo)
    for combo in cand_sets:
        fixed = {i: bord for (_, i, bord) in combo}
        newpt, ok, _ = _refine_constrained(A, B, pt, fixed)
        if not ok:
            continue
        if not all(bounds[i][0] <= newpt[i] <= bounds[i][1] for i in range(4)):
            continue                                # exact in-domain check
        if all(abs(float(newpt[i] - pt[i])) <= 4 * float(widths[i])
               for i in range(4)):                  # stay near THIS endpoint
            return newpt
    return None


def ssi_trim_loops(A, B, depth: int = 4, use_rust: bool | None = None):
    """SSI branches as closed TRIM LOOPS on BOTH parameter domains — the
    input :func:`forgekernel.trim.split_trim_region` partitions a face by.

    Closed branches (the exact strip certificate of
    :func:`_branch_is_closed` — the branch's cells enclose a hole in A's
    parameter domain) become loops directly. OPEN branches are stitched:
    each chain end is extended to a residual-certified domain-border
    crossing (:func:`_refine_constrained` with the border value pinned
    exactly), then the loop is closed with border segments and the patch
    corners passed on a counterclockwise border walk. A branch that
    cannot be stitched — an endpoint strictly interior to either domain,
    or an uncertifiable border crossing — refuses by name
    (:class:`TrimLoopUnstitchable`), never a guessed loop.

    Consistency between the domains is by construction: ``loop_a`` and
    ``loop_b`` read the SAME certified chain (residual² < 1e-20, so both
    describe the same 3D curve) in A's (u, v) and B's (s, t); only the
    border-walk segments are domain-local. For an open branch, which
    side the stitched loop encloses depends on the chain's orientation —
    callers that need the full partition (both sides) should hand the
    loops to ``split_trim_region``, whose face tracing is independent of
    that choice.

    Returns ``{"loops": [{"closed": bool, "points": [(u,v,s,t), …],
    "loop_a": [(u,v), …], "loop_b": [(s,t), …]}, …], "empty_certified":
    bool, "cells": n, "empty_cells": n, "tightened": n}``; every
    surviving SSI cell is classified upstream (certified point / proven
    empty / :class:`SsiCellUncertified`)."""
    res = _ssi_resolved(A, B, depth, use_rust)
    if res is None:
        return {"loops": [], "empty_certified": True, "cells": 0,
                "empty_cells": 0, "tightened": 0}
    points, branches, stats, bcells = res
    adom, bdom = A.domain(), B.domain()
    out = []
    for bi, group in enumerate(branches):
        items = _order_chain([(abox, points[abox]) for abox in group])
        chain = [pt for _, pt in items]
        if len(chain) < 3:
            raise TrimLoopUnstitchable(
                bi, "resolution",
                f"only {len(chain)} certified point(s) at depth {depth} — "
                f"too few to form a loop; raise depth")
        if _branch_is_closed(group, adom):
            # exact strip certificate: the branch's cells enclose a hole
            out.append({"closed": True, "points": chain,
                        "loop_a": [p[0:2] for p in chain],
                        "loop_b": [p[2:4] for p in chain]})
            continue
        ends = (items[0], items[-1])
        ends_touch_border = any(
            _cell_hits_border(acell, adom)
            or _cell_hits_border(bcells[acell][0], bdom)
            for acell, _ in ends)
        stitched = None
        if ends_touch_border:
            p0 = _extend_end(A, B, chain[0], ends[0][0],
                             bcells[ends[0][0]][0])
            p1 = _extend_end(A, B, chain[-1], ends[1][0],
                             bcells[ends[1][0]][0])
            if p0 is not None and p1 is not None:
                stitched = (p0, p1)
            else:
                raise TrimLoopUnstitchable(
                    bi, "border_uncertified",
                    "open chain end near a domain border but no "
                    "residual-certified border crossing was found")
        if stitched is not None:
            p0, p1 = stitched
            full = [p0] + chain + [p1]
            full = [p for i, p in enumerate(full)
                    if i == 0 or p != full[i - 1]]   # drop exact repeats
            loops_ab = []
            for dom, lo, hi in ((adom, 0, 2), (bdom, 2, 4)):
                pts2 = [p[lo:hi] for p in full]
                for name, p in (("first", pts2[0]), ("last", pts2[-1])):
                    if _border_position(dom, p) is None:
                        which = "A" if lo == 0 else "B"
                        raise TrimLoopUnstitchable(
                            bi, "open_interior",
                            f"{name} chain end {tuple(map(str, p))} is "
                            f"strictly interior to {which}'s parameter "
                            f"domain — the curve ends mid-face; closing "
                            f"it needs the other surface's edge curves "
                            f"(boolean assembly), refusing to guess")
                loops_ab.append(pts2 + _border_corners(dom, pts2[-1], pts2[0]))
            out.append({"closed": False, "points": full,
                        "loop_a": loops_ab[0], "loop_b": loops_ab[1]})
            continue
        raise TrimLoopUnstitchable(
            bi, "open_interior",
            "branch is not certifiably closed (its cell strip encloses "
            "no hole in A's parameter domain) and its chain ends touch "
            "no domain border — open branch with no certified closure; "
            "raise depth or treat as tangency")
    return {"loops": out, "empty_certified": not out, **stats}


def _pair_group_prunable(grp, levels: int = 4, cap: int = 256) -> bool:
    """Certified EXCLUSION for a group of surviving (BezierPatch,
    BezierPatch) leaf pairs by deeper subdivision + control-net bbox
    pruning alone (no Newton): ``True`` iff every descendant pair is
    proven disjoint — the cell carries no intersection. ``False`` means
    "could not prove empty" (never "nonempty"); callers must keep the
    cell. Sound because no pair is ever discarded — growth past ``cap``
    gives up rather than truncating."""
    cur = list(grp)
    for _ in range(levels):
        nxt = []
        for a, b in cur:
            for sa in a.split4():
                ba = sa.bbox()
                for sb in b.split4():
                    if _boxes_overlap(ba, sb.bbox()):
                        nxt.append((sa, sb))
        if not nxt:
            return True                          # certified exclusion
        if len(nxt) > cap:
            return False
        cur = nxt
    return False


def _bside_certified_strip(A, B, points, bcells, use_rust=None):
    """The certified B-side strip: from the surviving pairs of the
    point-bearing A-cells, drop every B-cell PROVEN empty and keep the
    rest — the swapped-role analogue of the A-side classification.

    Soundness of the enclosure: any true intersection point survives in
    the leaf pair containing it, whose A-cell is therefore point-bearing
    — so the union of the kept B-cells still encloses the curve in B's
    domain. A cell is dropped only on a subdivision-pruning proof of
    emptiness (deeper subdivision + control-net bbox disjointness — the
    Rust ``ssi_pairs`` hot loop when built, else
    :func:`_pair_group_prunable`, the executable spec); "could not
    prove" keeps the cell. This kills the measured ~3× spurious B-side
    inflation (an A-cell's 3D box overlaps a smear of B-cells whose
    parameter boxes never meet the curve)."""
    from forgekernel.nurbs import bezier_patches

    owners: dict = {}
    for ac, bbs in bcells.items():
        for bb in bbs:
            owners.setdefault(bb, set()).add(ac)
    keep = set()
    todo = []
    for bb in sorted(owners):
        acs = owners[bb]
        if any(bb[0] <= points[ac][2] <= bb[1]
               and bb[2] <= points[ac][3] <= bb[3] for ac in acs):
            keep.add(bb)                 # certified point inside: curve-bearing
        else:
            todo.append((bb, acs))
    if not todo:
        return keep
    rs = None
    if use_rust is not False:
        try:
            import forgekernel_rs as rs
        except ImportError:
            if use_rust is True:
                raise
            rs = None
    spans = (list(bezier_patches(A)), list(bezier_patches(B)))
    patch_memo: dict = {}

    def cellpatch(side, box):
        key = (side, box)
        if key not in patch_memo:
            patch_memo[key] = _cell_patch(spans[side], box)
        return patch_memo[key]

    for bb, acs in todo:
        grp = [(cellpatch(0, ac), cellpatch(1, bb)) for ac in sorted(acs)]
        if rs is not None:
            empty = all(not rs.ssi_pairs(
                _net_str(a.net), _net_str(b.net), 4,
                [_rstr(a.u0), _rstr(a.u1), _rstr(a.v0), _rstr(a.v1)],
                [_rstr(b.u0), _rstr(b.u1), _rstr(b.v0), _rstr(b.v1)])
                for a, b in grp)
        else:
            empty = _pair_group_prunable([(b, a) for a, b in grp])
        if not empty:
            keep.add(bb)
    return keep


def _border_sides(dom, p):
    """The side names ("u0"|"u1"|"v0"|"v1") of the domain rectangle that
    point ``p`` lies EXACTLY on — [] for an interior point, two names for
    a corner. Exact ℚ."""
    (u0, u1), (v0, v1) = dom
    u, v = F(p[0]), F(p[1])
    out = []
    if u == F(u0):
        out.append("u0")
    if u == F(u1):
        out.append("u1")
    if v == F(v0):
        out.append("v0")
    if v == F(v1):
        out.append("v1")
    return out


def ssi_chains(A, B, depth: int = 4, use_rust: bool | None = None):
    """SSI branches as certified CHAINS — the boolean assembly's input.

    Unlike :func:`ssi_trim_loops` (which must close every branch on both
    domains and therefore refuses a curve that leaves the OTHER surface
    through its border), this returns each branch as it is: a closed
    chain, or an open chain whose two ends are residual-certified border
    crossings. Each end is classified exactly: on A's border, on B's
    border (the side is named), or refused —

    * an end on BOTH domains' borders at once refuses
      (``seam_corner`` — a double crossing needs corner topology);
    * an end strictly interior to both domains refuses
      (``open_interior`` — tangential retreat, not a crossing);
    * an end near a border with no certified crossing refuses
      (``border_uncertified``).

    Returns ``{"chains": [{"closed": bool, "points": [(u,v,s,t), …],
    "cells": [(a_cell, b_cell|None), …], "ends": (end0, end1)}, …],
    "empty_certified": bool, "cells": n, "empty_cells": n,
    "tightened": n, "strips": (a_cells, b_cells), "maps": {"a2b", "b2a",
    "a_set", "b_set"}}`` where each end is ``None`` (for a closed chain)
    or ``("A"|"B", side)``. Every surviving SSI cell is classified
    upstream (certified / proven empty / :class:`SsiCellUncertified`).

    ``strips`` are the CERTIFIED trim-curve enclosures on both parameter
    domains — like :func:`ssi_strips` but with every cell that the
    resolver PROVED empty pruned away (A side: the point-bearing cells;
    B side: :func:`_bside_certified_strip`). Still rigorous enclosures —
    only proven-empty cells are dropped — and substantially smaller on
    the B side (measured ~3× on the dome × plane case). Per chain point,
    ``cells`` carries the point's own A-cell and the surviving B-cell
    its (s, t) lies in (``None`` when no listed B-cell contains it, or
    for border-extension points)."""
    res = _ssi_resolved(A, B, depth, use_rust)
    if res is None:
        return {"chains": [], "empty_certified": True, "cells": 0,
                "empty_cells": 0, "tightened": 0,
                "strips": (set(), set()),
                "maps": {"a2b": {}, "b2a": {}, "a_set": set(),
                         "b_set": set()}}
    points, branches, stats, bcells = res
    b_strip = _bside_certified_strip(A, B, points, bcells, use_rust)
    b2a: dict = {}
    for ac, bbs in bcells.items():
        for bb in bbs:
            b2a.setdefault(bb, set()).add(ac)
    maps = {"a2b": {ac: list(bbs) for ac, bbs in bcells.items()},
            "b2a": {bb: sorted(acs) for bb, acs in b2a.items()},
            "a_set": set(points), "b_set": set(b_strip)}

    def cell_pair(acell, pt):
        if acell is None:
            return (None, None)
        s, t = pt[2], pt[3]
        for bb in bcells[acell]:
            if bb[0] <= s <= bb[1] and bb[2] <= t <= bb[3]:
                return (acell, bb)
        return (acell, None)

    adom, bdom = A.domain(), B.domain()
    out = []
    for bi, group in enumerate(branches):
        items = _order_chain([(abox, points[abox]) for abox in group])
        chain = [pt for _, pt in items]
        cellsq = [ab for ab, _ in items]
        if len(chain) < 3:
            raise TrimLoopUnstitchable(
                bi, "resolution",
                f"only {len(chain)} certified point(s) at depth {depth} — "
                f"too few to form a chain; raise depth")
        # exact strip certificate: closed iff the branch's cells enclose
        # a hole in A's parameter domain (never chain-end adjacency)
        is_closed = _branch_is_closed(group, adom)
        ends = []
        for e, (acell, pt) in enumerate((items[0], items[-1])):
            a_on = _border_sides(adom, pt[0:2])
            b_on = _border_sides(bdom, pt[2:4])
            if not a_on and not b_on and not is_closed:
                touches = (_cell_hits_border(acell, adom)
                           or any(_cell_hits_border(b, bdom)
                                  for b in bcells[acell]))
                if touches:
                    p2 = _extend_end(A, B, pt, acell, bcells[acell][0])
                    if p2 is not None:
                        if e == 0:
                            chain = [p2] + chain
                            cellsq = [None] + cellsq
                        else:
                            chain = chain + [p2]
                            cellsq = cellsq + [None]
                        a_on = _border_sides(adom, p2[0:2])
                        b_on = _border_sides(bdom, p2[2:4])
                    else:
                        raise TrimLoopUnstitchable(
                            bi, "border_uncertified",
                            "open chain end near a domain border but no "
                            "residual-certified border crossing was found")
            if a_on and b_on:
                raise TrimLoopUnstitchable(
                    bi, "seam_corner",
                    f"chain end lies on A's border ({a_on[0]}) AND B's "
                    f"border ({b_on[0]}) at once — a double crossing "
                    f"needs corner topology; refusing to guess")
            if len(a_on) > 1 or len(b_on) > 1:
                raise TrimLoopUnstitchable(
                    bi, "seam_corner",
                    "chain end lies exactly on a domain corner")
            if a_on:
                ends.append(("A", a_on[0]))
            elif b_on:
                ends.append(("B", b_on[0]))
            else:
                ends.append(None)
        # drop exact consecutive repeats introduced by border extension
        keep_idx = [i for i, p in enumerate(chain)
                    if i == 0 or p != chain[i - 1]]
        chain = [chain[i] for i in keep_idx]
        cellsq = [cellsq[i] for i in keep_idx]
        cpairs = [cell_pair(ab, p) for ab, p in zip(cellsq, chain)]
        if ends[0] is None and ends[1] is None:
            if not is_closed:
                raise TrimLoopUnstitchable(
                    bi, "open_interior",
                    "branch is not certifiably closed (its cell strip "
                    "encloses no hole in A's parameter domain) and its "
                    "chain ends touch no domain border — no certified "
                    "closure; raise depth or treat as tangency")
            out.append({"closed": True, "points": chain, "cells": cpairs,
                        "ends": (None, None)})
            continue
        if ends[0] is None or ends[1] is None:
            raise TrimLoopUnstitchable(
                bi, "open_interior",
                "one chain end crosses a border but the other ends "
                "strictly interior to both domains — the curve ends "
                "mid-face; refusing to guess a closure")
        out.append({"closed": False, "points": chain, "cells": cpairs,
                    "ends": (ends[0], ends[1])})
    return {"chains": out, "empty_certified": not out, **stats,
            "strips": (set(points), b_strip), "maps": maps}


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
