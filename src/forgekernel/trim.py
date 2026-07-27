"""K7 — trimmed-patch topology: a surface patch plus parameter-space trim
loops, with **exact** point-in-region classification.

A boolean over freeform solids produces faces that are only *part* of a
surface — the region bounded by intersection curves. We represent that as
the untrimmed patch (any surface exposing ``domain()``/``eval()``) plus a
set of trim loops in the patch's (u, v) parameter domain:

    loops[0]  — the outer boundary
    loops[1:] — holes punched out of it

Trim-loop vertices carry exact rational (u, v) coordinates: they come from
certified SSI points (:func:`forgekernel.ssi.ssi_curves`) and patch-corner
vertices, both already in ℚ. So the point-in-region test — even-odd ray
parity — is decided in ℚ with **no tolerance**. That exactness is the whole
point: which side of a trim boundary a point lies on is a *topological*
decision, and ADR-0019 forbids a float from making it. A point that lands
exactly on the boundary is reported as ``"on"`` (also an exact predicate),
never silently bucketed into "in" or "out".

The even-odd rule classifies correctly regardless of each loop's winding
(a point inside a hole crosses outer+inner = even ⇒ out); loop orientation
matters only for the signed parameter-domain area, not for containment.

Not modeled here (deliberately): the *surface* area/volume of the trimmed
region. Its integrand is the surface Jacobian and the trim curve is only
polyline-approximated in parameter space, so that measure is not exact —
it belongs to the later Green's-over-trim-loops step, not this type.
"""

from __future__ import annotations

from fractions import Fraction as F
from functools import cmp_to_key


def _as_uv(pt):
    return (F(pt[0]), F(pt[1]))


def _orient(a, b, c):
    """Sign of the cross product (b−a)×(c−a): +1 left, −1 right, 0 collinear.
    Exact in ℚ."""
    d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (d > 0) - (d < 0)


def _segments_properly_cross(a, b, c, d) -> bool:
    """Do open segments a→b and c→d cross at an interior point? Exact ℚ, and
    strict: shared endpoints or collinear touching do NOT count (those are
    legal hole-touches-outer contacts, not a boundary crossing)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 \
        and o1 != o2 and o3 != o4


class TrimmedPatch:
    """A surface restricted to the region its trim loops enclose.

    ``surface`` is any object with ``domain()`` -> ((u0,u1),(v0,v1)) and
    ``eval(u, v)``. ``loops`` is a list of loops, each an ordered list of
    (u, v) vertices (the closing edge back to the first vertex is implicit);
    ``loops[0]`` is the outer boundary and the rest are holes."""

    def __init__(self, surface, loops) -> None:
        self.surface = surface
        self.loops = [[_as_uv(p) for p in loop] for loop in loops]
        if not self.loops:
            raise ValueError("trimmed patch needs at least an outer loop")
        for i, loop in enumerate(self.loops):
            if len(loop) < 3:
                raise ValueError(f"loop {i}: a trim loop needs >= 3 vertices")

    # -- exact predicates ------------------------------------------------------

    @staticmethod
    def _on_segment(u, v, a, b) -> bool:
        """Is (u, v) exactly on the closed segment a→b? (exact in ℚ)."""
        cross = (b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0])
        if cross != 0:
            return False
        # collinear — inside the segment's bounding box?
        return (min(a[0], b[0]) <= u <= max(a[0], b[0])
                and min(a[1], b[1]) <= v <= max(a[1], b[1]))

    def classify(self, u, v) -> str:
        """Exact point-in-region: ``"in"`` (strict interior), ``"on"`` (on a
        trim edge), or ``"out"``. All decided in ℚ — no tolerance."""
        u, v = F(u), F(v)
        # boundary first: an on-edge point is neither in nor out
        for loop in self.loops:
            m = len(loop)
            for k in range(m):
                if self._on_segment(u, v, loop[k], loop[(k + 1) % m]):
                    return "on"
        # even-odd parity of a +u ray at height v, over every loop's edges
        inside = False
        for loop in self.loops:
            m = len(loop)
            for k in range(m):
                a, b = loop[k], loop[(k + 1) % m]
                if (a[1] > v) != (b[1] > v):
                    # u of the edge's crossing with the horizontal line at v
                    x_int = a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                    if u < x_int:
                        inside = not inside
        return "in" if inside else "out"

    def contains(self, u, v) -> bool:
        """True iff (u, v) is in the strict interior of the trimmed region."""
        return self.classify(u, v) == "in"

    # -- exact volume contribution (K7 flux over the trim loops) ---------------

    def flux(self):
        """This trimmed face's exact flux contribution
        (1/3)∮∮_D S·(S_u×S_v) du dv to the enclosed solid volume, via Green's
        theorem over the trim loops (:func:`forgekernel.bsolid.trimmed_patch_flux`).
        Requires a polynomial surface. Loops are winding-normalized first so
        the outer/hole orientation is correct. Summing ``flux`` over the
        outward-oriented faces of a closed shell gives the solid volume."""
        from forgekernel.bsolid import trimmed_patch_flux
        n = self.normalized()
        return trimmed_patch_flux(n.surface, n.loops)

    # -- exact measures --------------------------------------------------------

    @staticmethod
    def _loop_signed_area(loop):
        """Shoelace signed area (exact ℚ); CCW positive."""
        m = len(loop)
        s = F(0)
        for k in range(m):
            a, b = loop[k], loop[(k + 1) % m]
            s += a[0] * b[1] - b[0] * a[1]
        return s / 2

    def signed_area(self):
        """Exact signed parameter-domain area = outer + Σ holes, using each
        loop's given winding (outer CCW +, holes CW −). This is the (u, v)
        measure of the region, NOT the surface area — see the module note."""
        return sum((self._loop_signed_area(loop) for loop in self.loops), F(0))

    def area(self):
        """Exact unsigned parameter-domain area: |outer| − Σ|holes|."""
        loops = self.loops
        outer = abs(self._loop_signed_area(loops[0]))
        holes = sum(abs(self._loop_signed_area(h)) for h in loops[1:])
        return outer - holes

    # -- orientation / validation ---------------------------------------------

    def is_ccw(self, index: int = 0) -> bool:
        return self._loop_signed_area(self.loops[index]) > 0

    def normalized(self) -> "TrimmedPatch":
        """Return a copy with canonical winding: outer CCW, holes CW."""
        out = []
        for i, loop in enumerate(self.loops):
            ccw = self._loop_signed_area(loop) > 0
            want_ccw = (i == 0)
            out.append(list(loop) if ccw == want_ccw else list(reversed(loop)))
        return TrimmedPatch(self.surface, out)

    def validate(self) -> None:
        """Structural checks: every vertex inside the surface's parameter
        domain, holes strictly inside the outer boundary."""
        (u0, u1), (v0, v1) = self.surface.domain()
        for i, loop in enumerate(self.loops):
            for (u, v) in loop:
                if not (u0 <= u <= u1 and v0 <= v <= v1):
                    raise ValueError(
                        f"loop {i}: vertex ({u},{v}) outside surface domain")
        outer = TrimmedPatch(self.surface, [self.loops[0]])
        outer_loop = self.loops[0]
        no = len(outer_loop)
        for i, hole in enumerate(self.loops[1:], start=1):
            # WHOLE-polygon containment, not a single sample point: every hole
            # vertex must be in-or-on the outer boundary AND no hole edge may
            # properly cross an outer edge. Testing only the vertex average
            # accepts holes that stick out through the boundary and rejects
            # valid holes whose average lands in a concavity (reviewed 2026-07).
            for (u, v) in hole:
                if outer.classify(u, v) == "out":
                    raise ValueError(f"hole {i} is not inside the outer boundary")
            mh = len(hole)
            for a in range(mh):
                h0, h1 = hole[a], hole[(a + 1) % mh]
                for b in range(no):
                    o0, o1 = outer_loop[b], outer_loop[(b + 1) % no]
                    if _segments_properly_cross(h0, h1, o0, o1):
                        raise ValueError(
                            f"hole {i} crosses the outer boundary")


# -- K7 gap 4: exact partition of a parameter domain by trim loops ------------

def _on_closed_segment(p, a, b) -> bool:
    """Is p exactly on the closed segment a→b? Exact ℚ."""
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if cross != 0:
        return False
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _classify_in_loops(loops, u, v) -> str:
    """Even-odd point-in-region over a list of loops — the exact rule
    :meth:`TrimmedPatch.classify` uses, available on bare loop lists."""
    for loop in loops:
        m = len(loop)
        for k in range(m):
            if _on_closed_segment((u, v), loop[k], loop[(k + 1) % m]):
                return "on"
    inside = False
    for loop in loops:
        m = len(loop)
        for k in range(m):
            a, b = loop[k], loop[(k + 1) % m]
            if (a[1] > v) != (b[1] > v):
                x_int = a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if u < x_int:
                    inside = not inside
    return "in" if inside else "out"


def _angle_cmp(o):
    """Exact counterclockwise angular comparator for directions out of
    ``o``: half-plane split (angle in [0,π) before [π,2π)), then the
    cross-product sign within a half. No trigonometry, pure ℚ."""
    def cmp(p, q):
        dpx, dpy = p[0] - o[0], p[1] - o[1]
        dqx, dqy = q[0] - o[0], q[1] - o[1]
        hp = 0 if (dpy > 0 or (dpy == 0 and dpx > 0)) else 1
        hq = 0 if (dqy > 0 or (dqy == 0 and dqx > 0)) else 1
        if hp != hq:
            return hp - hq
        cr = dpx * dqy - dpy * dqx
        return -1 if cr > 0 else (1 if cr < 0 else 0)
    return cmp


def _cycle_signed_area(cyc):
    s = F(0)
    m = len(cyc)
    for k in range(m):
        a, b = cyc[k], cyc[(k + 1) % m]
        s += a[0] * b[1] - b[0] * a[1]
    return s / 2


def split_trim_region(surface, loops):
    """Exact-ℚ partition of a face's parameter domain by trim loops —
    the polygon-with-holes split a boolean's face classification runs on.

    ``surface`` is anything with ``domain()`` (or a bare
    ``((u0,u1),(v0,v1))`` pair); ``loops`` are closed trim loops with
    exact rational vertices — e.g. the ``loop_a`` / ``loop_b`` sides of
    :func:`forgekernel.ssi.ssi_trim_loops`. Loops may share segments and
    corners with the domain border (stitched open branches); they must
    not properly cross each other or themselves — a crossing refuses by
    name rather than guessing an arrangement.

    Method: the domain border and every loop edge are split at all
    incident vertices, deduplicated into an exact planar subdivision,
    and the faces are traced with exact orientation predicates (angular
    order by half-plane + cross-product sign — no floats anywhere).
    Counterclockwise (positive-area) cycles are region outers; each
    clockwise cycle is attached as a hole to the smallest region that
    strictly contains it; the unbounded face is discarded.

    Returns a list of regions, each a loop list ``[outer, holes…]`` in
    the :class:`TrimmedPatch` convention (outer CCW, holes CW). The
    regions PARTITION the domain: Σ region areas == domain area, checked
    in exact ℚ before returning — a failed audit raises rather than
    shipping a wrong partition."""
    dom = surface.domain() if hasattr(surface, "domain") else surface
    (u0, u1), (v0, v1) = ((F(dom[0][0]), F(dom[0][1])),
                          (F(dom[1][0]), F(dom[1][1])))
    lps = [[_as_uv(p) for p in lp] for lp in loops]
    for i, lp in enumerate(lps):
        if len(lp) < 3:
            raise ValueError(f"split_trim_region: loop {i} needs >= 3 vertices")
        for (u, v) in lp:
            if not (u0 <= u <= u1 and v0 <= v <= v1):
                raise ValueError(
                    f"split_trim_region: loop {i} vertex ({u},{v}) lies "
                    f"outside the parameter domain")

    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    raw = [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    for lp in lps:
        m = len(lp)
        for k in range(m):
            a, b = lp[k], lp[(k + 1) % m]
            if a != b:
                raw.append((a, b))
    return _partition_raw(((u0, u1), (v0, v1)), raw)


def split_domain_by_paths(surface, paths, loops=()):
    """Exact-ℚ partition of a face's parameter domain by OPEN trim paths
    (each anchored on the domain border at both ends) plus optional
    closed loops — the arrangement an open-branch boolean face runs on.

    ``paths`` are polylines with exact rational vertices whose first and
    last vertex lie EXACTLY on the domain border (certified border
    crossings from :func:`forgekernel.ssi.ssi_chains`, canonicalized by
    the boolean assembly); ``loops`` are closed as in
    :func:`split_trim_region`. The domain border is part of the
    subdivision, so the traced regions PARTITION the domain — Σ region
    areas == domain area, audited in exact ℚ. Paths and loops must not
    properly cross; a crossing refuses by name."""
    dom = surface.domain() if hasattr(surface, "domain") else surface
    (u0, u1), (v0, v1) = ((F(dom[0][0]), F(dom[0][1])),
                          (F(dom[1][0]), F(dom[1][1])))
    pts_paths = [[_as_uv(p) for p in path] for path in paths]
    for i, path in enumerate(pts_paths):
        if len(path) < 2:
            raise ValueError(
                f"split_domain_by_paths: path {i} needs >= 2 vertices")
        for (u, v) in path:
            if not (u0 <= u <= u1 and v0 <= v <= v1):
                raise ValueError(
                    f"split_domain_by_paths: path {i} vertex ({u},{v}) "
                    f"lies outside the parameter domain")
        for p in (path[0], path[-1]):
            on = (p[0] == u0 or p[0] == u1 or p[1] == v0 or p[1] == v1)
            if not on:
                raise ValueError(
                    f"split_domain_by_paths: path {i} end ({p[0]},{p[1]}) "
                    f"is not on the domain border — an open trim path "
                    f"must enter and leave through the border")
    lps = [[_as_uv(p) for p in lp] for lp in loops]
    for i, lp in enumerate(lps):
        if len(lp) < 3:
            raise ValueError(
                f"split_domain_by_paths: loop {i} needs >= 3 vertices")
        for (u, v) in lp:
            if not (u0 <= u <= u1 and v0 <= v <= v1):
                raise ValueError(
                    f"split_domain_by_paths: loop {i} vertex ({u},{v}) "
                    f"lies outside the parameter domain")

    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    raw = [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    for path in pts_paths:
        for a, b in zip(path, path[1:]):
            if a != b:
                raw.append((a, b))
    for lp in lps:
        m = len(lp)
        for k in range(m):
            a, b = lp[k], lp[(k + 1) % m]
            if a != b:
                raw.append((a, b))
    return _partition_raw(((u0, u1), (v0, v1)), raw)


def _partition_raw(dom, raw):
    """Shared arrangement core: split raw segments at all incident
    vertices, dedupe, refuse proper crossings and dangling edges, trace
    faces with exact predicates, attach holes, audit Σ areas == domain
    area. See :func:`split_trim_region` for the contract."""
    (u0, u1), (v0, v1) = dom
    verts = set()
    for a, b in raw:
        verts.add(a)
        verts.add(b)
    edges = set()
    for a, b in raw:
        d = (b[0] - a[0], b[1] - a[1])
        mids = [p for p in verts
                if p != a and p != b and _on_closed_segment(p, a, b)]
        mids.sort(key=lambda p: (p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1])
        chain = [a] + mids + [b]
        for x, y in zip(chain, chain[1:]):
            if x != y:
                edges.add((x, y) if x < y else (y, x))
    edges = sorted(edges)

    for i in range(len(edges)):
        a, b = edges[i]
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            if a in (c, d) or b in (c, d):
                continue
            if _segments_properly_cross(a, b, c, d):
                raise ValueError(
                    "split_trim_region: trim loops cross — refusing to "
                    "guess the arrangement")

    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for o, lst in adj.items():
        if len(lst) < 2:
            raise ValueError(
                f"split_trim_region: dangling trim edge at ({o[0]},{o[1]})")
        lst.sort(key=cmp_to_key(_angle_cmp(o)))
    idx = {o: {p: i for i, p in enumerate(lst)} for o, lst in adj.items()}

    # face tracing: arriving a→b, leave b along the clockwise-next edge
    # from the reversal — every directed edge lands in exactly one cycle
    visited = set()
    cycles = []
    for a in adj:
        for b in adj[a]:
            if (a, b) in visited:
                continue
            cyc = []
            e = (a, b)
            while e not in visited:
                visited.add(e)
                cyc.append(e[0])
                x, y = e
                lst = adj[y]
                e = (y, lst[(idx[y][x] - 1) % len(lst)])
            cycles.append(cyc)

    pos, neg = [], []
    for cyc in cycles:
        ar = _cycle_signed_area(cyc)
        if ar > 0:
            pos.append([ar, cyc, []])
        elif ar < 0:
            neg.append(cyc)
        else:
            raise ValueError(
                "split_trim_region: zero-area face cycle — degenerate "
                "trim loop")
    for cyc in neg:
        best = None
        for rec in pos:
            verdicts = {_classify_in_loops([rec[1]], p[0], p[1]) for p in cyc}
            if "out" in verdicts or "in" not in verdicts:
                continue
            if best is None or rec[0] < best[0]:
                best = rec
        if best is not None:
            best[2].append(cyc)              # a hole (already clockwise)
        # else: the unbounded face — dropped

    regions = []
    total = F(0)
    for ar, cyc, holes in pos:
        regions.append([cyc] + holes)
        total += ar + sum(_cycle_signed_area(h) for h in holes)
    if total != (u1 - u0) * (v1 - v0):
        raise ValueError(
            f"split_trim_region: partition audit failed — region areas "
            f"sum to {total}, domain area is {(u1 - u0) * (v1 - v0)}")
    return regions
