"""K7 gap 3 — certified point-in-shell membership by ray×patch parity.

A point p is inside a closed shell iff a ray from p crosses the boundary
an odd number of times. For freeform faces nothing here may be a float
guess (ADR-0019), so every ingredient of the count is certified:

* **Candidate pruning.** The ray {p + τd} is cut out by two linear
  forms f1 = e1·(x − p), f2 = e2·(x − p) with e1·d = e2·d = 0. Composed
  with a Bézier patch both are tensor Bernstein forms with exact ℚ
  coefficients, so a subdivision cell whose coefficient hull excludes
  zero *provably* contains no ray hit (convex-hull property). Rational
  patches multiply through by the (positive) weight form — sign
  unchanged, still exact.

* **Parity per surviving cell = topological degree.** On a leaf cell
  whose boundary provably avoids the zero of (f1, f2), the number of
  ray crossings inside is congruent mod 2 to the winding number of
  (f1, f2) around the origin along the cell boundary — and the winding
  is computed EXACTLY: signed crossings of {f1 = 0, f2 > 0} along the
  four edges by 1D Bernstein sign-variation root isolation (Descartes:
  one sign change in the coefficients = exactly one root). No
  transversality certificate is needed for parity — the local degree of
  a grazing contact is 0 or ±2, exactly its crossing parity.

* **Direction gate.** The τ-form (d·(x − p) composed with the patch)
  prunes cells provably behind the origin; a surviving cell that cannot
  be certified strictly in front refuses (a point ON the shell refuses
  on every ray — the honest answer).

* **Trim gate.** A crossing counts only if it lies in the face's kept
  region. A leaf cell that overlaps the face's SSI trim strip REFUSES —
  the hit may sit on either side of the trim boundary — and the
  classifier retries with a jittered ray rather than guessing. A
  strip-free cell has constant membership (the strip encloses the whole
  trim boundary), decided by one exact test at the cell midpoint.

A ray refusal is not an answer about p: any *certified* ray gives the
true parity, so the classifier walks a deterministic list of rational
directions and returns the first certified verdict. When every ray in
the budget refuses — p on or near the boundary, or pathologically
aligned geometry — it raises :class:`PointClassifyUncertified` with the
per-ray reasons, never a guess.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.bsolid import _bf_hull, _bf_split_u, _bf_split_v, _split1d

F = Fraction


class PointClassifyUncertified(ValueError):
    """Structured refusal: no ray in the budget produced a certified
    crossing parity for this point. Typical causes: the point lies on
    (or within resolution of) the shell boundary, or every tried ray
    passes through a trim-strip cell. ``reasons`` maps each tried ray
    direction to why it refused."""

    def __init__(self, point, reasons) -> None:
        self.point = tuple(point)
        self.reasons = list(reasons)
        why = "; ".join(f"{d}: {r}" for d, r in self.reasons[:4])
        super().__init__(
            f"point_classify_uncertified: {len(self.reasons)} ray(s) "
            f"refused for point {tuple(map(str, self.point))} — {why} — "
            f"the point may lie on the boundary; raise max_rays/depth "
            f"only if it is believed strictly off the shell")


class _RayUncertified(Exception):
    """Internal: this ray cannot be certified — jitter and retry."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _perp2(d):
    """Two ℚ vectors e1, e2 with e1·d = e2·d = 0, linearly independent —
    the pencil of planes whose intersection is the ray line. Built from
    generic reference vectors (not axis-aligned) so a tilted ray gets
    planes with all three components live — an axis-degenerate pencil
    (e.g. z-free planes) would put every point of a vertical line on the
    same f1 = 0 set and refuse spuriously at dyadic parameter columns."""
    if d == (0, 0, 0):
        raise ValueError("zero ray direction")
    for r in ((F(3, 7), F(5, 11), F(1, 13)), (F(1), F(0), F(0))):
        e1 = _cross(d, r)
        if e1 != (0, 0, 0):
            return e1, _cross(d, e1)
    raise AssertionError("no reference vector independent of the ray")


def _lin_form(net, e, c):
    """Tensor B-form of e·A − c·w over the net (w ≡ 1 for polynomial
    3-nets) — this is w·(e·S − c), same sign as e·S − c since w > 0."""
    out = []
    for row in net:
        orow = []
        for pt in row:
            w = F(pt[3]) if len(pt) == 4 else F(1)
            orow.append(F(pt[0]) * e[0] + F(pt[1]) * e[1]
                        + F(pt[2]) * e[2] - c * w)
        out.append(orow)
    return out


def _sign_changes(coeffs) -> int:
    signs = [1 if x > 0 else -1 for x in coeffs if x != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _edge_crossings(c1, c2, budget: int = 40) -> int:
    """Signed crossings of the curve (f1(t), f2(t)), t ∈ [0, 1], with the
    positive-f2 half-axis: +1 where f1 passes + → − with f2 > 0, −1 for
    − → +. Exact, by Bernstein sign-variation isolation; refuses when a
    root sits exactly on a subdivision point or fails to isolate."""
    if c1[0] == 0 or c1[-1] == 0:
        raise _RayUncertified("f1 vanishes at a cell-boundary vertex")

    def rec(a1, a2, budget):
        lo1, hi1 = min(a1), max(a1)
        if lo1 > 0 or hi1 < 0:
            return 0
        sc = _sign_changes(a1)
        if sc == 0:
            return 0                       # variation-diminishing: no root
        if sc == 1:                        # exactly one simple root
            lo2, hi2 = min(a2), max(a2)
            if hi2 < 0:
                return 0                   # crossing below the axis
            if lo2 > 0:
                return 1 if a1[0] > 0 else -1
        if budget == 0:
            raise _RayUncertified("edge root isolation budget exhausted")
        L1, R1 = _split1d(a1)
        L2, R2 = _split1d(a2)
        if L1[-1] == 0:                    # exact root at the split point
            raise _RayUncertified("f1 vanishes at a subdivision point")
        return rec(L1, L2, budget - 1) + rec(R1, R2, budget - 1)

    return rec(list(c1), list(c2), budget)


def _winding(P1, P2) -> int:
    """Winding number of (f1, f2) around the origin along the cell
    boundary (counterclockwise in (u, v)) — the topological degree,
    whose parity is the cell's ray-crossing parity. Exact; refuses if
    (f1, f2) cannot be certified nonzero on the boundary."""
    m, n = len(P1) - 1, len(P1[0]) - 1
    total = 0
    # bottom (v=0, u 0→1), right (u=1, v 0→1), top (v=1, u 1→0),
    # left (u=0, v 1→0) — a closed CCW walk
    edges = (
        ([P1[i][0] for i in range(m + 1)], [P2[i][0] for i in range(m + 1)]),
        ([P1[m][j] for j in range(n + 1)], [P2[m][j] for j in range(n + 1)]),
        ([P1[i][n] for i in range(m, -1, -1)],
         [P2[i][n] for i in range(m, -1, -1)]),
        ([P1[0][j] for j in range(n, -1, -1)],
         [P2[0][j] for j in range(n, -1, -1)]),
    )
    for c1, c2 in edges:
        total += _edge_crossings(c1, c2)
    return total


def _overlaps_strip(box, strip) -> bool:
    u0, u1, v0, v1 = box
    for (s0, s1, t0, t1) in strip:
        if s0 < u1 and u0 < s1 and t0 < v1 and v0 < t1:
            return True
    return False


def _cell_parity(P1, P2, T, box, depth, strip, inside) -> int:
    """Certified crossing count (for parity) of the ray with the face
    piece over ``box``. 0 is a *certificate* (pruned or even-and-decided);
    anything undecidable raises :class:`_RayUncertified`."""
    h1 = _bf_hull(P1)
    if h1[0] > 0 or h1[1] < 0:
        return 0                            # f1 ≠ 0: no hit in this cell
    h2 = _bf_hull(P2)
    if h2[0] > 0 or h2[1] < 0:
        return 0
    ht = _bf_hull(T)
    if ht[1] < 0:
        return 0                            # provably behind the origin
    if depth > 0:
        u0, u1, v0, v1 = box
        um, vm = (u0 + u1) / 2, (v0 + v1) / 2
        P1L, P1R = _bf_split_u(P1)
        P2L, P2R = _bf_split_u(P2)
        TL, TR = _bf_split_u(T)
        total = 0
        for (p1h, p2h, th, ub) in ((P1L, P2L, TL, (u0, um)),
                                   (P1R, P2R, TR, (um, u1))):
            p1b, p1t = _bf_split_v(p1h)
            p2b, p2t = _bf_split_v(p2h)
            tb, tt = _bf_split_v(th)
            total += _cell_parity(p1b, p2b, tb, (ub[0], ub[1], v0, vm),
                                  depth - 1, strip, inside)
            total += _cell_parity(p1t, p2t, tt, (ub[0], ub[1], vm, v1),
                                  depth - 1, strip, inside)
        return total
    # leaf: the cell may contain a hit
    if _overlaps_strip(box, strip):
        raise _RayUncertified(
            "ray passes through a trim-strip cell — which side of the "
            "trim boundary the hit lies on is not decidable")
    if ht[0] <= 0:
        raise _RayUncertified(
            "hit cell cannot be certified strictly in front of the ray "
            "origin (the point may lie on the shell)")
    w = _winding(P1, P2)
    if w == 0:
        return 0                            # even parity, membership moot
    u0, u1, v0, v1 = box
    return abs(w) if inside((u0 + u1) / 2, (v0 + v1) / 2) else 0


def _face_crossings(face, p, d, depth: int) -> int:
    """Certified crossing count of the ray with one trimmed face."""
    from forgekernel.nurbs import bezier_patches

    e1, e2 = _perp2(d)
    c1 = sum(a * b for a, b in zip(e1, p))
    c2 = sum(a * b for a, b in zip(e2, p))
    ct = sum(a * b for a, b in zip(d, p))
    total = 0
    for (u0, u1, v0, v1, net) in bezier_patches(face.surface):
        if len(net[0][0]) == 4:
            for row in net:
                for pt in row:
                    if pt[3] <= 0:
                        raise ValueError(
                            "certified raycast: convex-hull pruning needs "
                            "positive weights (K3.7 for sign-varying)")
        P1 = _lin_form(net, e1, c1)
        P2 = _lin_form(net, e2, c2)
        T = _lin_form(net, d, ct)
        total += _cell_parity(P1, P2, T, (F(u0), F(u1), F(v0), F(v1)),
                              depth, face.strip, face.inside)
    return total


# deterministic ray directions: +z first, then rational jitters of
# growing tilt — enough spread that a trim-strip graze clears. The tail
# uses PRIME denominators: against axis-aligned faces a ray from a
# symmetric rational point crosses at parameters like ½ + ¾·d_x, which
# for small round d_x lands exactly on a dyadic gridline and exhausts
# edge root isolation on every early direction (measured: all eight
# refused for (2,2,0) against a [0,4]²×[1,3] box); a prime denominator
# cannot cancel into a dyadic and clears the degeneracy.
_DIRECTIONS = (
    (F(0), F(0), F(1)),
    (F(1, 3), F(1, 5), F(1)),
    (F(-2, 7), F(1, 3), F(1)),
    (F(3, 8), F(-2, 5), F(1)),
    (F(-1, 4), F(-3, 7), F(1)),
    (F(1), F(1, 3), F(1, 2)),
    (F(-1, 2), F(1), F(2, 3)),
    (F(2, 3), F(-1), F(1, 2)),
    (F(5, 17), F(3, 23), F(1)),
    (F(-7, 19), F(5, 13), F(1)),
    (F(11, 29), F(-4, 31), F(1)),
    (F(-3, 37), F(-8, 41), F(1)),
)


def classify_point_in_shell(shell, p, depth: int = 5, max_rays: int = 12):
    """Certified membership of point ``p`` in a closed
    :class:`~forgekernel.trimshell.TrimmedShell`: ``"in"`` or ``"out"``,
    or a :class:`PointClassifyUncertified` refusal — never a guess.

    Casts rays from ``p`` along a deterministic list of rational
    directions; each ray's boundary-crossing parity is certified end to
    end (hull pruning, exact winding numbers, τ-front gate, trim-strip
    refusal — see the module docstring). The first certified ray decides:
    parity odd ⇒ inside. A ray through a trim-strip cell refuses and the
    next, jittered direction is tried."""
    p = tuple(F(c) for c in p)
    reasons = []
    for d in _DIRECTIONS[:max_rays]:
        try:
            total = 0
            for face in shell.faces:
                total += _face_crossings(face, p, d, depth)
            return "in" if total % 2 else "out"
        except _RayUncertified as exc:
            reasons.append((tuple(float(x) for x in d), exc.reason))
    raise PointClassifyUncertified(p, reasons)
