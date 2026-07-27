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
                        cap: int = 256, tries: int = 8):
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
    return {"branches": len(_cluster(alive)), "points": pts,
            "uncertified": 0, "empty_certified": not pts,
            "cells": len(groups), "empty_cells": empty,
            "tightened": tightened}


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
    ``(points, branches, stats)`` where ``points`` maps each point-bearing
    A-cell box to its certified (u, v, s, t), ``branches`` clusters those
    cells (connected components in A's parameter domain), and ``stats``
    is {"cells", "empty_cells", "tightened"}."""
    boxes = _ssi_boxes(A, B, depth, use_rust)
    if not boxes:
        return None
    groups: dict = {}
    for abox, bbox_ in boxes:
        groups.setdefault(abox, []).append(bbox_)

    def newton(um, vm, sm, tm, iters=12):
        return _refine_global(A, B, um, vm, sm, tm, iters)

    spans = None
    points: dict = {}
    unresolved = []
    empty = tightened = 0
    for abox, bbs in groups.items():
        b0 = bbs[0]
        um, vm = (abox[0] + abox[1]) / 2, (abox[2] + abox[3]) / 2
        sm, tm = (b0[0] + b0[1]) / 2, (b0[2] + b0[3]) / 2
        u, v, s, t, ok, _ = newton(um, vm, sm, tm)
        if ok:
            points[abox] = (u, v, s, t)
            continue
        if spans is None:
            from forgekernel.nurbs import bezier_patches
            spans = (list(bezier_patches(A)), list(bezier_patches(B)))
        grp = [(_cell_patch(spans[0], abox), _cell_patch(spans[1], bb))
               for bb in bbs]
        verdict, pt = _resolve_cell_pairs(grp, abox, newton)
        if verdict == "point":
            points[abox] = pt
            tightened += 1
        elif verdict == "empty":
            empty += 1
        else:
            unresolved.append((abox, b0))
    if unresolved:
        raise SsiCellUncertified(unresolved, depth, _RESOLVE_LEVELS)
    branches = _cluster(list(points))
    return points, branches, {"cells": len(groups), "empty_cells": empty,
                              "tightened": tightened}


def ssi_strips(A, B, depth: int = 4, use_rust: bool | None = None):
    """The certified trim-curve ENCLOSURE on both parameter domains: the
    deduplicated sets of surviving subdivision cells, ``(a_cells,
    b_cells)``, each a set of (u0, u1, v0, v1) Fraction boxes at
    resolution 2^-depth (within each Bézier span).

    The true intersection curve provably lies inside the union of the
    a-cells in A's domain AND inside the union of the b-cells in B's
    domain — bbox pruning never discards a real intersection. This is
    the rigorous object a boolean's certified volume bracket is built
    from (:func:`forgekernel.bsolid.certified_trim_flux`); the chained
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
    points, branches, stats = res
    return {"branches": len(branches), "points": list(points.values()),
            "uncertified": 0, "empty_certified": not points, **stats}


def _order_branch(points):
    """Chain a branch's certified points into an ordered polyline in A's
    (u, v) parameter space by greedy nearest-neighbour, started from an
    endpoint via the double-sweep diameter heuristic (furthest point from
    an arbitrary sample is an endpoint of an open arc). Reports whether the
    branch closes on itself. Float ordering over exact points — a
    render/report artifact; the points themselves are the certified data."""
    n = len(points)
    if n <= 2:
        return list(points), False
    uv = [(float(p[0]), float(p[1])) for p in points]

    def d2(i, j):
        return (uv[i][0] - uv[j][0]) ** 2 + (uv[i][1] - uv[j][1]) ** 2

    start = max(range(n), key=lambda i: d2(i, 0))   # an extreme end
    todo = set(range(n))
    todo.discard(start)
    order = [start]
    gaps = []
    while todo:
        last = order[-1]
        nxt = min(todo, key=lambda i: d2(i, last))
        gaps.append(d2(nxt, last) ** 0.5)
        order.append(nxt)
        todo.discard(nxt)
    ordered = [points[i] for i in order]
    # Closure needs enough samples to be decidable: with only 3 points the
    # median-of-two collapses to the LARGER gap, and the triangle inequality
    # then forces wrap ≤ g1+g2 ≤ 2·median for ANY three points — so the test
    # would call every 3-point arc closed. A genuine median needs ≥3 gaps
    # (n ≥ 4); below that, report open (closure is undecidable from the
    # samples) rather than let a float heuristic mis-decide the topology.
    if n < 4:
        return ordered, False
    wrap = d2(order[0], order[-1]) ** 0.5
    med = sorted(gaps)[len(gaps) // 2]
    closed = med > 0 and wrap <= 2.0 * med
    return ordered, closed


def ssi_curves(A, B, depth: int = 4, use_rust: bool | None = None):
    """Ordered SSI output: the certified intersection points chained into
    one parameter-space polyline **per branch** (a branch = a connected
    component of surviving cells in A's parameter domain). Within a branch
    the points are ordered along the curve; a branch that returns to its
    start is flagged ``closed``.

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
    points, branches, stats = res
    branch_of = {abox: bi for bi, group in enumerate(branches) for abox in group}
    per_branch: dict[int, list] = {bi: [] for bi in range(len(branches))}
    for abox, pt in points.items():
        per_branch[branch_of[abox]].append(pt)
    curves = []
    for bi in range(len(branches)):
        ordered, closed = _order_branch(per_branch[bi])
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
