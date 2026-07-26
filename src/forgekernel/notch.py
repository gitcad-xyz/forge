"""A box tool whose footprint STRADDLES a bore (#123), cut on the canonical Body.

The last boolean family the capability matrix refused. A prism whose footprint
crosses a bore wall is not expressible in the feature representations — a
``DrilledSolid`` cannot hold a bore that meets a lateral wall, and its guard
that says so is correct — but it is perfectly expressible as a ``Body``, so
this is written ONCE against the canonical form (ADR-0021) and serves every
representation that converts into one.

WHAT IS ACTUALLY HARD, and it is not the arithmetic. The removed cross-section
is NOT the tool's rectangle. Classify the tool's footprint corners against the
bore and the sides that lie inside it contribute no face at all: for a Ø4 bore
with the tool corner on its axis the notch floor has THREE edges, not four, and
emitting the fourth gives a self-intersecting shell with a plausible volume.
The same code run against a Ø2 bore gives FIVE. So nothing here is templated:
the boundary is classified, every sub-segment whose interior lies strictly
inside the bore is dropped, and the bore arc is spliced across the gap.

EXACTNESS. Every crossing of a tool wall with the bore is decided by comparing
the wall's offset from the axis, a/r, against the cosines of the twelfths —
{0, ±1/2, ±√3/2, ±1}. Anywhere else the removed area's arc term is the sine of
an angle with no algebraic value this kernel can hold (Niven), so the answer
leaves ℚ[√3][π] and this refuses by name rather than rounding. No atan2, no
tolerance, nowhere: there is not one float in this file, which is why its
docstring does NOT claim the charter's float exemption — the lint scans it.

THE SPLIT DECOMPOSITION. The bore wall over the notch is a plain trimmed band
and below it a plain full band — never one L-shaped face. The two are measured
correctly by the stock band term today; an L-shaped face in (θ, z) would need a
new quadric measure that 500+ existing faces also depend on.
"""

from __future__ import annotations

from fractions import Fraction as F

from forgekernel import body as B

UP = (F(0), F(0), F(1))
DN = (F(0), F(0), F(-1))
XR = (F(1), F(0), F(0))


class NotchRefused(ValueError):
    """Not this family — the caller should try its other paths.

    Distinguished from :class:`NotchOutsideField` on purpose. A caller that
    reported every one of these would hide the guards downstream that are
    correct for shapes this builder simply does not cover.
    """


class NotchOutsideField(NotchRefused):
    """This IS a straddling cut, and its answer leaves every exact field the
    kernel has. A permanent boundary, not a gap — say so, by name."""


def _vertical(d) -> bool:
    return d[0] == 0 and d[1] == 0 and d[2] != 0


def _z_range(face):
    """(lo, hi) axial extent of a cylindrical face from its circular rims."""
    zs = [e.curve.c[2] for lp in face.loops for e in lp.edges
          if isinstance(e.curve, B.Circle)]
    if len(zs) < 2:
        raise NotchRefused("a cylindrical face without two circular rims")
    return min(zs), max(zs)


def _cos_twelfth(a):
    """The twelfth index t in 0..6 with cos(t·π/6) == a, or None.

    This is the whole exact-or-refuse gate for #123: a is the tool wall's
    offset from the bore axis divided by the radius, and the removed area stays
    in ℚ[√3][π] exactly when a is one of these seven values. Exact equality on
    exact types — an ``abs(a - c) < eps`` here would be a float deciding how
    many faces the answer has, which ADR-0019 forbids outright.
    """
    cs = B._sin_cos_twelfths()
    for t in range(7):
        if cs[t][0] == a:
            return t
    return None


def _half_plane_arc(a, shift: int, keep_upper: bool, what: str) -> set:
    """The twelfths of the circle satisfying one of the tool's four walls.

    ``a`` is the wall's signed offset from the axis over the radius. With
    ``shift`` = 0 the constraint is on cos θ (an x wall), with 3 it is on
    sin θ (a y wall), because sin θ == cos(θ − 3 twelfths).
    """
    if a >= 1:
        # cos θ ≤ a is everything; cos θ ≥ a holds only at θ = 0, and only when
        # the wall is exactly tangent
        return set(range(12)) if keep_upper else ({shift % 12} if a == 1 else set())
    if a <= -1:
        return ({(6 + shift) % 12} if a == -1 else set()) if keep_upper \
            else set(range(12))
    t = _cos_twelfth(a)
    if t is None:
        raise NotchOutsideField(
            f"the tool's {what} wall meets the bore at an angle that is not a "
            "multiple of 30°, so the removed area leaves ℚ[√3][π] (Niven's "
            "theorem: those are the only rational angles with algebraic sine "
            "this kernel can hold). Offset the wall to 0, r/2 or r·√3/2 from "
            "the bore axis, or clear the bore entirely")
    # cos θ ≤ a  ⟺  θ ∈ [t, 12−t];   cos θ ≥ a  ⟺  θ ∈ [−t, t]
    rng = range(t, 12 - t + 1) if keep_upper else range(-t, t + 1)
    return {(k + shift) % 12 for k in rng}


def _run(ks: set):
    """A set of twelfths as one CONTIGUOUS cyclic run (start, span), or None."""
    if not ks or len(ks) == 12:
        return None
    for start in sorted(ks):
        if (start - 1) % 12 not in ks:
            span = 0
            while (start + span + 1) % 12 in ks:
                span += 1
            return (start, span) if span + 1 == len(ks) else None
    return None


def _twelfth_point(cx, cy, z, r, k):
    co, si = B._sin_cos_twelfths()[k % 12]
    return (cx + r * co, cy + r * si, z)


def _inside_rect(p, rect) -> bool:
    x0, x1, y0, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def _rect_chain(rect, cx, cy, r, pa, pb):
    """The tool footprint's boundary MINUS whatever lies inside the bore.

    Returns the CCW chain of (p, q) segments from ``pa`` round to ``pb``. This
    is the pass that deletes the two inner walls when the tool's corner sits on
    the bore axis, and KEEPS their outer halves when the bore is smaller — one
    code path, two different face counts, no template.
    """
    x0, x1, y0, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    nodes = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        nodes.append((a[0], a[1]))
        # the two crossings lie ON the footprint boundary; insert each into the
        # side it belongs to, ordered along that side
        on = [p for p in (pa, pb)
              if (a[0] == b[0] == p[0] and min(a[1], b[1]) < p[1] < max(a[1], b[1]))
              or (a[1] == b[1] == p[1] and min(a[0], b[0]) < p[0] < max(a[0], b[0]))]
        # x*x, never x**2: a crossing at an odd twelfth has SurdVal
        # coordinates, and SurdVal implements multiplication, not __pow__ —
        # the ADR-0019 lint's own defect class (backlog 1.4)
        on.sort(key=lambda p: (p[0] - a[0]) * (p[0] - a[0])
                + (p[1] - a[1]) * (p[1] - a[1]))
        nodes.extend((p[0], p[1]) for p in on)
    key = [(pa[0], pa[1]), (pb[0], pb[1])]
    idx = [i for i, nd in enumerate(nodes) if nd in key]
    if len(idx) != 2:
        raise NotchRefused(
            "the bore meets the tool footprint somewhere other than its "
            "boundary — the tool does not straddle this bore")
    i, j = idx
    chains = [nodes[i:j + 1], nodes[j:] + nodes[:i + 1]]
    keep = []
    for ch in chains:
        segs = [(ch[t], ch[t + 1]) for t in range(len(ch) - 1)]
        if not segs:
            continue
        if all((((p[0] + q[0]) / 2 - cx) * ((p[0] + q[0]) / 2 - cx)
                + ((p[1] + q[1]) / 2 - cy) * ((p[1] + q[1]) / 2 - cy))
               >= r * r for p, q in segs):
            keep.append(segs)
    if len(keep) != 1:
        raise NotchRefused(
            "the tool footprint is not split into one inside and one outside "
            "arc by this bore — a footprint that swallows the bore, or misses "
            "it, is a different topology")
    segs = keep[0]
    if (segs[0][0][0], segs[0][0][1]) != (pa[0], pa[1]):
        segs = [(q, p) for p, q in reversed(segs)]
    return segs


def _line(a, b):
    return B.Edge(B.Line(a, tuple(b[i] - a[i] for i in range(3))), a, b)


def notch_cut(body: B.Body, rect, za, zb) -> B.Body:
    """``body`` minus an axis-aligned box tool [x0,x1]×[y0,y1]×[za,zb] whose
    footprint straddles exactly one bore.

    Refuses, by name, anything outside that family. A refusal is a legitimate
    answer here; a body that is not the difference is not.
    """
    x0, x1, y0, y1 = (F(v) for v in rect)
    za, zb = F(za), F(zb)
    rect = (x0, x1, y0, y1)
    if not (x0 < x1 and y0 < y1 and za < zb):
        raise NotchRefused("the tool box is degenerate")

    # -- 1. find the ONE bore the footprint straddles ------------------------
    bores = []
    for f in body.faces:
        s = f.surface
        if not isinstance(s, B.Cylinder) or not _vertical(s.d):
            continue
        cx, cy = s.p[0], s.p[1]
        d2 = [((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
              for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        rr = s.r * s.r
        if min(d2) < rr < max(d2) or (min(d2) < rr and any(x == rr for x in d2)):
            bores.append(f)
    if len(bores) != 1:
        raise NotchRefused(
            f"the tool footprint straddles {len(bores)} cylindrical walls — "
            "exactly one is supported")
    bore = bores[0]
    if bore.sense:
        raise NotchRefused(
            "the wall the tool straddles is an outward boss, not a bore — the "
            "notch would open to the outside")
    cyl = bore.surface
    if cyl.d[2] <= 0:
        raise NotchRefused("the bore's axis runs the wrong way for this cut")
    cx, cy, r = cyl.p[0], cyl.p[1], cyl.r
    zlo, ztop = _z_range(bore)
    if not zlo < za < ztop:
        raise NotchRefused(
            "the tool floor is not strictly inside the bore's own extent — a "
            "notch that starts below the bore, or above it, is a different "
            "topology")
    if zb < ztop:
        raise NotchRefused(
            "the tool stops inside the bore, so the notch needs a ceiling face "
            "— a BLIND straddling pocket is not supported")

    # -- 2. the removed sector, exactly -------------------------------------
    ks = (_half_plane_arc((x1 - cx) / r, 0, True, "+x")
          & _half_plane_arc((x0 - cx) / r, 0, False, "−x")
          & _half_plane_arc((y1 - cy) / r, 3, True, "+y")
          & _half_plane_arc((y0 - cy) / r, 3, False, "−y"))
    run = _run(ks)
    if run is None:
        raise NotchRefused(
            "the tool covers all of the bore, or touches it at a single point "
            "— neither is a straddle")
    ka, span = run
    kb = (ka + span) % 12
    if span == 0:
        raise NotchRefused("the tool is tangent to the bore, so nothing opens")
    kept = 12 - span

    # -- 3. everything else must stay clear of the notch column -------------
    for f in body.faces:
        s = f.surface
        if isinstance(s, B.Plane):
            if s.n[0] == 0 and s.n[1] == 0:
                h = s.d / s.n[2]
                if za < h < ztop:
                    raise NotchRefused(
                        "a horizontal face sits inside the notch's depth, so "
                        "the cross-section is not constant there")
            continue
        if f is bore:
            continue
        if isinstance(s, B.Cylinder) and _vertical(s.d):
            lo, hi = _z_range(f)
            if hi <= za or lo >= ztop:
                continue
            d2 = [((p[0] - s.p[0]) ** 2 + (p[1] - s.p[1]) ** 2)
                  for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            if not (max(d2) < s.r * s.r and f.sense):
                raise NotchRefused(
                    "another cylindrical wall runs through the notch's depth")
            continue
        raise NotchRefused(
            f"a {type(s).__name__} face is not supported in this cut yet")

    # -- 4. the face at the bore's mouth, whose hole the notch grows into ----
    tops = [f for f in body.faces
            if isinstance(f.surface, B.Plane) and f.surface.n[0] == 0
            and f.surface.n[1] == 0 and f.surface.d / f.surface.n[2] == ztop
            and any(len(lp.edges) == 1 and isinstance(lp.edges[0].curve, B.Circle)
                    and lp.edges[0].curve.r == r
                    and lp.edges[0].curve.c[0] == cx
                    and lp.edges[0].curve.c[1] == cy for lp in f.loops)]
    if len(tops) != 1:
        raise NotchRefused(
            "the bore does not open into exactly one flat face at its mouth")
    top = tops[0]
    if not (top.surface.n[2] > 0 and top.sense):
        raise NotchRefused(
            "the bore's mouth face does not look upward, so the notch would "
            "not open into free space")
    outer = top.loops[0]
    ocirc = B._loop_is_circle(outer)
    if ocirc is None:
        pts = [e.v0 for e in outer.edges]
        if any(isinstance(e.curve, B.Circle) for e in outer.edges):
            raise NotchRefused("the mouth face's outer bound mixes arcs and lines")
        # The outer bound must BE an axis-aligned rectangle, not merely have a
        # bounding box that clears the tool. A bbox test on an L-shaped outline
        # passes while the tool pokes past the reentrant corner into empty
        # space, and the notch then removes material that does not exist — a
        # plausible exact wrong number that only the mesh audit happened to
        # catch. Containment in a general polygon is a real capability (exact
        # point-in-polygon plus edge clearance); until someone builds and
        # proves it, a non-rectangular outline is OUT of this family.
        xs = sorted({p[0] for p in pts})
        ys = sorted({p[1] for p in pts})
        rectangular = (
            len(pts) == 4 and len(xs) == 2 and len(ys) == 2
            and all(e.v0[0] == e.v1[0] or e.v0[1] == e.v1[1]
                    for e in outer.edges))
        if not rectangular:
            raise NotchRefused(
                "the bore's mouth face is not bounded by a circle or an "
                "axis-aligned rectangle — near a reentrant outline the notch "
                "could break out of the part unseen")
        clear = xs[0] < x0 and x1 < xs[1] and ys[0] < y0 and y1 < ys[1]
    else:
        clear = all(((p[0] - ocirc.c[0]) ** 2 + (p[1] - ocirc.c[1]) ** 2)
                    < ocirc.r * ocirc.r
                    for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    if not clear:
        raise NotchRefused(
            "the tool reaches the outer bound of the bore's mouth face — the "
            "notch would break out of the part")
    # Every OTHER hole in the mouth face must stay clear of the tool too. A
    # through-pocket shows up only here — its vertical walls are planes, which
    # the sweep above deliberately does not police — and a pocket overlapping
    # the tool means the notch counts already-removed void as material.
    # Conservative by construction: the loop's xy bounding box (a full circle
    # or arc contributes centre ± r), so this can refuse a clearance that a
    # sharper test would admit, but it can never admit an overlap.
    for lp in top.loops[1:]:
        if (len(lp.edges) == 1 and isinstance(lp.edges[0].curve, B.Circle)
                and lp.edges[0].curve.r == r and lp.edges[0].curve.c[0] == cx
                and lp.edges[0].curve.c[1] == cy):
            continue                    # the straddled bore's own rim
        bx, by = [], []
        for e in lp.edges:
            if isinstance(e.curve, B.Circle):
                bx += [e.curve.c[0] - e.curve.r, e.curve.c[0] + e.curve.r]
                by += [e.curve.c[1] - e.curve.r, e.curve.c[1] + e.curve.r]
            else:
                bx += [e.v0[0], e.v1[0]]
                by += [e.v0[1], e.v1[1]]
        if not (max(bx) <= x0 or min(bx) >= x1
                or max(by) <= y0 or min(by) >= y1):
            raise NotchRefused(
                "another opening in the mouth face overlaps the tool "
                "footprint — the notch would double-count its void")

    # -- 5. build ------------------------------------------------------------
    pa2, pb2 = _twelfth_point(cx, cy, 0, r, ka), _twelfth_point(cx, cy, 0, r, kb)
    if not (_inside_rect(pa2, rect) and _inside_rect(pb2, rect)):
        raise NotchRefused("the computed crossings are not on the tool footprint")
    chain = _rect_chain(rect, cx, cy, r, pa2, pb2)

    def P(k, z):
        return _twelfth_point(cx, cy, z, r, k)

    def at(p, z):
        return (p[0], p[1], z)

    c_lo_up = B.Circle((cx, cy, za), UP, XR, r)
    c_lo_dn = B.Circle((cx, cy, za), DN, XR, r)
    c_tp_up = B.Circle((cx, cy, ztop), UP, XR, r)
    c_tp_dn = B.Circle((cx, cy, ztop), DN, XR, r)

    # the z = za rim is split at the crossings AND at the quarters, so that
    # every arc pairs with exactly one arc on the face opposite it, and so the
    # display mesh (which subdivides quarters dyadically) has a grid to share
    cuts = sorted({ka, kb, 0, 3, 6, 9})

    def steps(k_from, total, ccw):
        """The twelfth indices from k_from, sweeping `total` twelfths, broken at
        every cut. Indices are always measured CCW about +z; `ccw` says which
        way the sweep runs, so a rim traversed clockwise counts DOWN."""
        out, k, left = [k_from], k_from, total
        while left > 0:
            nxt = min([d for d in (((c - k) % 12 if ccw else (k - c) % 12)
                                   for c in cuts) if d > 0] + [12])
            nxt = min(nxt, left)
            k = (k + nxt) % 12 if ccw else (k - nxt) % 12
            out.append(k)
            left -= nxt
        return out

    def arcs(circle, ks, z):
        return tuple(B.Edge(circle, P(ks[i], z), P(ks[i + 1], z))
                     for i in range(len(ks) - 1))

    # bore wall BELOW the notch: a FULL turn whose top rim is split, so that
    # every arc pairs with exactly one arc above it
    bottom_rim = [e for lp in bore.loops for e in lp.edges
                  if isinstance(e.curve, B.Circle) and e.curve.c[2] == zlo]
    if not bottom_rim:
        raise NotchRefused("the bore has no rim at its far end")
    band_lo = B.Face(cyl, (B.Loop(
        tuple(bottom_rim) + arcs(c_lo_up, steps(ka, 12, True), za)),), False)

    # bore wall ACROSS the notch: an ordinary trimmed band, its two vertical
    # lines standing at the crossing angles
    band_hi = B.Face(cyl, (B.Loop(
        (B.Edge(c_tp_up, P(kb, ztop), P(ka, ztop)),
         _line(P(ka, ztop), P(ka, za)))
        + arcs(c_lo_dn, steps(ka, kept, False), za)
        + (_line(P(kb, za), P(kb, ztop)),)),), False)

    # the notch floor at za: the removed arc, then the surviving footprint
    floor = B.Face(B.Plane(UP, za), (B.Loop(
        arcs(c_lo_dn, steps(kb, span, False), za)
        + tuple(_line(at(p, za), at(q, za)) for p, q in chain)),), True)

    # the mouth face, with the notch spliced into its inner loop
    inner = B.Loop((B.Edge(c_tp_dn, P(ka, ztop), P(kb, ztop)),)
                   + tuple(_line(at(q, ztop), at(p, ztop))
                           for p, q in reversed(chain)))
    keep_loops = tuple(lp for lp in top.loops
                       if not (len(lp.edges) == 1
                               and isinstance(lp.edges[0].curve, B.Circle)
                               and lp.edges[0].curve.r == r
                               and lp.edges[0].curve.c[0] == cx
                               and lp.edges[0].curve.c[1] == cy))
    mouth = B.Face(top.surface, keep_loops + (inner,), top.sense)

    # one wall per SURVIVING footprint segment — none for a dropped one
    walls = []
    for p, q in chain:
        d = (q[0] - p[0], q[1] - p[1])
        n = (-d[1], d[0], F(0))
        walls.append(B.Face(
            B.Plane(n, n[0] * p[0] + n[1] * p[1]),
            (B.Loop((_line(at(p, za), at(p, ztop)),
                     _line(at(p, ztop), at(q, ztop)),
                     _line(at(q, ztop), at(q, za)),
                     _line(at(q, za), at(p, za)))),), True))

    rest = tuple(f for f in body.faces if f is not bore and f is not top)
    return B.Body(rest + (band_lo, band_hi, floor, mouth) + tuple(walls))
