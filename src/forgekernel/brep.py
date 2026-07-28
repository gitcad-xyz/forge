"""Polyhedral B-rep — convex-faceted solids with native lineage (K1).

A Solid is a closed set of convex polygons, each carrying the id of the
ORIGINAL face it descends from — lineage is data in the model, not a
service bolted on afterward (the ADR-0018 identity requirement). Mass
properties are exact rational integrals (signed tetrahedra / divergence
theorem); validation checks watertightness by exact edge pairing.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.exact import (F, Plane, Vec, add, centroid, cross, dot,
                               is_zero, neg, smul, sub, vec)


class Polygon:
    """Convex planar polygon, CCW around its outward plane normal."""

    __slots__ = ("verts", "plane", "source")

    def __init__(self, verts: list[Vec], source: str,
                 plane: Plane | None = None) -> None:
        if len(verts) < 3:
            raise ValueError("polygon needs >= 3 vertices")
        self.verts = verts
        self.plane = plane or Plane.from_points(verts[0], verts[1], verts[2])
        self.source = source

    def flipped(self) -> "Polygon":
        return Polygon(list(reversed(self.verts)), self.source,
                       self.plane.flipped())

    def area2(self) -> Fraction:
        """Twice the area times |n| — zero iff degenerate (exact test)."""
        acc = (Fraction(0), Fraction(0), Fraction(0))
        v0 = self.verts[0]
        for a, b in zip(self.verts[1:], self.verts[2:]):
            acc = add(acc, cross(sub(a, v0), sub(b, v0)))
        return dot(acc, acc)


class Solid:
    """A (intended-closed) collection of convex polygons."""

    __slots__ = ("polys",)

    def __init__(self, polys: list[Polygon]) -> None:
        self.polys = [p for p in polys if p.area2() != 0]

    # -- constructors ---------------------------------------------------------

    @classmethod
    def box(cls, dx, dy, dz, source_prefix: str = "box") -> "Solid":
        x, y, z = F(dx), F(dy), F(dz)
        if x <= 0 or y <= 0 or z <= 0:
            raise ValueError("box wants positive dimensions")
        o = Fraction(0)
        v = [vec(o, o, o), vec(x, o, o), vec(x, y, o), vec(o, y, o),
             vec(o, o, z), vec(x, o, z), vec(x, y, z), vec(o, y, z)]
        faces = [([0, 3, 2, 1], "bottom"), ([4, 5, 6, 7], "top"),
                 ([0, 1, 5, 4], "front"), ([2, 3, 7, 6], "back"),
                 ([1, 2, 6, 5], "right"), ([3, 0, 4, 7], "left")]
        return cls([Polygon([v[i] for i in idx], f"{source_prefix}.{name}")
                    for idx, name in faces])

    @classmethod
    def prism(cls, loop_xy: list[tuple], height,
              source_prefix: str = "prism") -> "Solid":
        """Extrude a simple CCW polygon (2D loop, no repeated last point)
        along +z. Caps are ear-clipped into triangles — exact orientation
        and containment tests, so non-convex profiles are fine."""
        h = F(height)
        if h <= 0:
            raise ValueError("prism wants positive height")
        loop = [(F(px), F(py)) for px, py in loop_xy]
        if _loop_area2(loop) < 0:
            loop = list(reversed(loop))
        tris = _ear_clip(loop)
        polys: list[Polygon] = []
        for a, b, c in tris:
            polys.append(Polygon([vec(*a, 0), vec(*c, 0), vec(*b, 0)],
                                 f"{source_prefix}.bottom"))
            polys.append(Polygon([vec(*a, h), vec(*b, h), vec(*c, h)],
                                 f"{source_prefix}.top"))
        n = len(loop)
        for i in range(n):
            (x1, y1), (x2, y2) = loop[i], loop[(i + 1) % n]
            polys.append(Polygon(
                [vec(x1, y1, 0), vec(x2, y2, 0), vec(x2, y2, h),
                 vec(x1, y1, h)], f"{source_prefix}.side{i}"))
        return cls(polys)

    # -- rigid/affine ---------------------------------------------------------

    def mapped(self, fn) -> "Solid":
        out = []
        for p in self.polys:
            out.append(Polygon([fn(v) for v in p.verts], p.source))
        return Solid(out)

    def translated(self, t: Vec) -> "Solid":
        return self.mapped(lambda v: add(v, t))

    def scaled(self, fx, fy, fz) -> "Solid":
        sx, sy, sz = F(fx), F(fy), F(fz)
        if sx == 0 or sy == 0 or sz == 0:
            raise ValueError("zero scale factor")
        s = self.mapped(lambda v: (v[0] * sx, v[1] * sy, v[2] * sz))
        if sx * sy * sz < 0:                      # orientation flip
            s = Solid([p.flipped() for p in s.polys])
        return s

    def mirrored(self, axis: str) -> "Solid":
        i = "xyz".index(axis)

        def fn(v: Vec) -> Vec:
            w = list(v)
            w[i] = -w[i]
            return (w[0], w[1], w[2])

        return Solid([p.flipped() for p in self.mapped(fn).polys])

    def rotated_quarter(self, axis: str, quarters: int) -> "Solid":
        """Exact rotation by multiples of 90° about a principal axis."""
        q = quarters % 4
        ax = "xyz".index(axis)

        def rot(v: Vec) -> Vec:
            a, b = (ax + 1) % 3, (ax + 2) % 3
            w = list(v)
            for _ in range(q):
                w[a], w[b] = -w[b], w[a]
            return (w[0], w[1], w[2])

        return self.mapped(rot)

    # -- exact metrics --------------------------------------------------------

    def volume6(self) -> Fraction:
        """Six times the signed volume — exact (sum of origin tetrahedra)."""
        acc = Fraction(0)
        for p in self.polys:
            v0 = p.verts[0]
            for a, b in zip(p.verts[1:], p.verts[2:]):
                acc += dot(v0, cross(a, b))
        return acc

    def volume(self) -> Fraction:
        return self.volume6() / 6

    def centroid(self) -> Vec:
        """Exact volume centroid (tetrahedron decomposition)."""
        v6 = self.volume6()
        if v6 == 0:
            raise ValueError("centroid of zero-volume solid")
        acc = (Fraction(0), Fraction(0), Fraction(0))
        for p in self.polys:
            v0 = p.verts[0]
            for a, b in zip(p.verts[1:], p.verts[2:]):
                w = dot(v0, cross(a, b))
                acc = add(acc, smul(w, add(add(v0, a), b)))
        return smul(Fraction(1, 4) / v6, acc)

    def bbox(self) -> tuple[Vec, Vec]:
        xs = [v for p in self.polys for v in p.verts]
        lo = (min(v[0] for v in xs), min(v[1] for v in xs),
              min(v[2] for v in xs))
        hi = (max(v[0] for v in xs), max(v[1] for v in xs),
              max(v[2] for v in xs))
        return lo, hi

    # -- topology projections -------------------------------------------------

    def logical_faces(self) -> dict[tuple, list[Polygon]]:
        """Fragments grouped by (plane canonical, lineage source) — the
        face an engineer means, reassembled from BSP shards."""
        out: dict[tuple, list[Polygon]] = {}
        for p in self.polys:
            out.setdefault((p.plane.canonical(), p.source), []).append(p)
        return out

    def watertight_violations(self) -> list[str]:
        """Exact closure test, T-junction tolerant: BSP output is
        geometrically closed but combinatorially fragmented, so edges are
        grouped by their carrier LINE (canonical direction + Plücker
        moment) and closure requires the SIGNED interval coverage on every
        line to cancel exactly. Zero everywhere == closed surface."""
        from collections import defaultdict

        def canon_dir(d: Vec) -> Vec | None:
            # Canonical (scale- and sign-invariant) representative of a
            # direction: divide through by the first nonzero component so any
            # scalar multiple ±λ·d maps to the same tuple. Works whether the
            # components are rational (ℚ) or quadratic surds (ℚ[√d]) — an
            # exactly-rotated solid carries √d edge directions.
            lead = None
            for v in d:
                if v != 0:
                    lead = v
                    break
            if lead is None:
                return None
            return tuple(v / lead for v in d)

        lines: dict = defaultdict(list)
        for p in self.polys:
            n = len(p.verts)
            for i in range(n):
                a, b = p.verts[i], p.verts[(i + 1) % n]
                d = sub(b, a)
                cd = canon_dir(d)
                if cd is None:
                    continue
                key = (cd, cross(a, cd))          # moment: line-invariant
                ta, tb = dot(a, cd), dot(b, cd)
                sign = 1 if ta < tb else -1
                lines[key].append((min(ta, tb), max(ta, tb), sign))
        bad: list[str] = []
        for key, segs in lines.items():
            cuts = sorted({t for lo, hi, _ in segs for t in (lo, hi)})
            for lo, hi in zip(cuts, cuts[1:]):
                cov = sum(s for slo, shi, s in segs if slo <= lo and hi <= shi)
                if cov != 0:
                    bad.append(f"open-boundary:line-dir={tuple(float(v) for v in key[0])}"
                               f":t=[{float(lo):g},{float(hi):g}]:coverage={cov}")
                    if len(bad) >= 8:
                        return bad + ["..."]
                    break
        return bad

    def tessellate(self) -> dict[str, list]:
        verts: list[list[float]] = []
        tris: list[list[int]] = []
        index: dict[Vec, int] = {}
        for p in self.polys:
            ids = []
            for v in p.verts:
                if v not in index:
                    index[v] = len(verts)
                    verts.append([float(v[0]), float(v[1]), float(v[2])])
                ids.append(index[v])
            for a, b in zip(ids[1:], ids[2:]):
                tris.append([ids[0], a, b])
        return {"vertices": verts, "triangles": tris}


# -- K3.6b: geometry that does not close (#135) -------------------------------
#
# The anti-Parasolid bet at the border: an open shell is never silently a
# solid. Volume and centroid are ORIGIN-DEPENDENT on an open shell (the
# divergence sum keeps the flux through the missing wall), so they are not
# approximations there — they are not numbers at all. The machinery below
# projects the exact interval-coverage audit back into user millimetres, and
# repairs — only ever by exact vertex merge, never by inventing a face.

class NonClosedShellError(ValueError):
    """A polygon set does not close into the boundary of a solid.

    ``segments`` are the uncovered boundary intervals with EXACT 3D endpoints;
    ``report`` is the same story in display millimetres (see
    :func:`boundary_gap_report`); ``healed`` carries the heal record when a
    vertex merge ran first and could not finish the job."""

    def __init__(self, message: str, *, segments: list, report: dict,
                 healed: dict | None = None) -> None:
        super().__init__(message)
        self.segments = segments
        self.report = report
        self.healed = healed


class SnapClusterError(ValueError):
    """A heal tolerance chained vertices into a cluster wider than itself.

    Merging 0 / 0.008 / 0.016 at tol 0.01 would move a vertex 1.6x the
    promised tolerance — transitive closure quietly breaks the contract, so
    the cluster refuses by name instead."""

    def __init__(self, message: str, *, cluster: list, diameter_sq: Fraction,
                 tolerance: Fraction) -> None:
        super().__init__(message)
        self.cluster = cluster
        self.diameter_sq = diameter_sq
        self.tolerance = tolerance


def _dist2(a: Vec, b: Vec) -> Fraction:
    d = sub(a, b)
    return dot(d, d)


def open_boundary_segments(polys: list[Polygon], cap: int = 256) -> list[dict]:
    """Every uncovered (or overcovered) boundary interval, as exact 3D
    endpoints — the same signed interval-coverage audit as
    ``Solid.watertight_violations``, projected back out of carrier-line
    parameters into points a person can find on the part. Empty == closed."""
    from collections import defaultdict

    lines: dict = defaultdict(list)
    for p in polys:
        n = len(p.verts)
        for i in range(n):
            a, b = p.verts[i], p.verts[(i + 1) % n]
            cd = _canon_dir(sub(b, a))
            if cd is None:
                continue
            key = (cd, cross(a, cd))              # (direction, Plücker moment)
            ta, tb = dot(a, cd), dot(b, cd)
            sign = 1 if ta < tb else -1
            lines[key].append((min(ta, tb), max(ta, tb), sign))
    segs: list[dict] = []
    for (cd, moment), intervals in sorted(lines.items(),
                                          key=lambda kv: repr(kv[0])):
        cuts = sorted({t for lo, hi, _ in intervals for t in (lo, hi)})
        dd = dot(cd, cd)
        # point on the carrier line at parameter t (t = dot(p, cd)):
        # base is the line's closest point to the origin, exact.
        base = smul(1 / dd, cross(cd, moment))
        runs: list[list] = []                     # [t_lo, t_hi, coverage]
        for lo, hi in zip(cuts, cuts[1:]):
            cov = sum(s for slo, shi, s in intervals if slo <= lo and hi <= shi)
            if cov != 0 and runs and runs[-1][1] == lo and runs[-1][2] == cov:
                runs[-1][1] = hi                  # extend the maximal run
            elif cov != 0:
                runs.append([lo, hi, cov])
        segs.extend({"a": add(base, smul(t0 / dd, cd)),
                     "b": add(base, smul(t1 / dd, cd)),
                     "coverage": cov}
                    for t0, t1, cov in runs)
        if len(segs) >= cap:
            break
    return segs[:cap]


def boundary_gap_report(segments: list[dict]) -> dict:
    """The gap report in user millimetres — what an agent (or a person)
    needs to decide between repairing at source and healing: how many open
    edges, how long the open boundary runs, how many connected chains it
    forms, and — when opposing chains exist — how wide the crack between
    them is. Never the raw carrier-line parameters (a 3 um tear renders as
    ``t=[-133323,10.003]`` in those — unreadable and not millimetres)."""
    import math

    def fpt(v: Vec) -> tuple:
        return tuple(float(c) for c in v)

    def length(s: dict) -> float:
        return math.sqrt(float(_dist2(s["a"], s["b"])))

    # connected chains by EXACT shared endpoints
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    index: dict = {}
    for i, s in enumerate(segments):
        for v in (s["a"], s["b"]):
            if v in index:
                ra, rb = find(index[v]), find(i)
                parent[ra] = rb
            else:
                index[v] = i
    chains: dict = {}
    for i in range(len(segments)):
        chains.setdefault(find(i), []).append(i)
    chain_pts = [[v for i in members for v in
                  (segments[i]["a"], segments[i]["b"])]
                 for members in chains.values()]
    max_gap = None
    if len(chain_pts) >= 2:
        gaps = []
        for i, pts in enumerate(chain_pts):
            best = None
            for j, other in enumerate(chain_pts):
                if i == j:
                    continue
                for p in pts:
                    for q in other:
                        d2 = _dist2(p, q)
                        if best is None or d2 < best:
                            best = d2
            gaps.append(best)
        max_gap = math.sqrt(float(max(gaps)))
    return {"open_edges": len(segments),
            "open_perimeter_mm": sum(length(s) for s in segments),
            "chains": len(chain_pts),
            "max_gap_mm": max_gap,
            "segments_mm": [{"a": fpt(s["a"]), "b": fpt(s["b"]),
                             "length_mm": length(s),
                             "coverage": s["coverage"]}
                            for s in segments[:16]],
            "segments_truncated": max(0, len(segments) - 16)}


def _area_vec(verts: list[Vec]) -> Vec:
    acc = (Fraction(0), Fraction(0), Fraction(0))
    v0 = verts[0]
    for a, b in zip(verts[1:], verts[2:]):
        acc = add(acc, cross(sub(a, v0), sub(b, v0)))
    return acc


def snap_vertices(polys: list[Polygon], tolerance) -> tuple[list[Polygon], dict]:
    """Certified heal: merge vertices coincident within ``tolerance`` — and
    nothing else. Exact predicates throughout (dist² <= tol², no sqrt in any
    decision); the representative is the lexicographic minimum of its cluster
    (deterministic, so rebuilds stay byte-canonical); a cluster whose diameter
    exceeds the tolerance REFUSES (transitive chains would move a vertex
    farther than promised); faces that collapse are dropped INTO the record,
    never silently. Returns ``(new_polys, record)`` where the record is the
    certificate: how many vertices moved, the exact max move, and the bound
    |ΔV| <= Σ(affected face area)·max_move — deliberately NOT a delta against
    the pre-heal volume, which is origin-dependent on an open shell and
    therefore meaningless."""
    import math

    tol = tolerance if isinstance(tolerance, Fraction) else Fraction(str(tolerance))
    if tol <= 0:
        raise ValueError("snap_vertices wants a positive tolerance")
    tol2 = tol * tol
    verts = sorted({v for p in polys for v in p.verts})
    n = len(verts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            # cheap exact reject before the full distance: sorted order makes
            # the x-gap monotone, so once it alone exceeds tol, stop the row
            dx = verts[j][0] - verts[i][0]
            if dx * dx > tol2:
                break
            if _dist2(verts[i], verts[j]) <= tol2:
                parent[find(i)] = find(j)
    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    mapping: dict = {}
    moved: list[tuple] = []
    cluster_members: set = set()
    for members in clusters.values():
        if len(members) == 1:
            continue
        pts = [verts[i] for i in members]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                d2 = _dist2(pts[a], pts[b])
                if d2 > tol2:
                    raise SnapClusterError(
                        f"heal tolerance {float(tol):g} chains "
                        f"{len(pts)} vertices into a cluster of diameter "
                        f"{math.sqrt(float(d2)):g} mm — merging would move a "
                        f"vertex farther than the promised tolerance",
                        cluster=[tuple(float(c) for c in p) for p in pts],
                        diameter_sq=d2, tolerance=tol)
        rep = min(pts)
        for p in pts:
            cluster_members.add(p)
            mapping[p] = rep
            if p != rep:
                moved.append((p, rep, _dist2(p, rep)))
    new_polys: list[Polygon] = []
    dropped: list[str] = []
    affected_area = 0.0
    affected_sources: list[str] = []
    for p in polys:
        nv = [mapping.get(v, v) for v in p.verts]
        ded: list[Vec] = []
        for v in nv:
            if not ded or v != ded[-1]:
                ded.append(v)
        if len(ded) > 1 and ded[0] == ded[-1]:
            ded.pop()
        if any(v in cluster_members for v in p.verts):
            affected_sources.append(p.source)
            affected_area += math.sqrt(float(p.area2())) / 2
        if len(ded) < 3 or is_zero(_area_vec(ded)):
            dropped.append(p.source)
            continue
        new_polys.append(Polygon(ded, p.source))
    max_move_sq = max((d2 for _, _, d2 in moved), default=Fraction(0))
    max_move = math.sqrt(float(max_move_sq))
    record = {"tolerance": str(tol),
              "moved": len(moved),
              "vertices_moved": [(tuple(float(c) for c in a),
                                  tuple(float(c) for c in b),
                                  math.sqrt(float(d2)))
                                 for a, b, d2 in moved[:16]],
              "max_move_sq": max_move_sq,
              "max_move_mm": max_move,
              "affected_faces": affected_sources,
              "affected_area_mm2": affected_area,
              "volume_change_bound_mm3": affected_area * max_move,
              "dropped_faces": dropped}
    return new_polys, record


def _pt(v: Vec) -> str:
    return f"({float(v[0]):g},{float(v[1]):g},{float(v[2]):g})"


def _loop_area2(loop: list[tuple]) -> Fraction:
    acc = Fraction(0)
    n = len(loop)
    for i in range(n):
        (x1, y1), (x2, y2) = loop[i], loop[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return acc


def _ear_clip(loop: list[tuple]) -> list[tuple]:
    """Exact ear clipping of a simple CCW polygon -> triangles."""

    def orient(a, b, c) -> Fraction:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(p, a, b, c) -> bool:
        # A blocker is a point in the CLOSED triangle. With strict `> 0` a
        # vertex lying exactly ON the candidate ear's diagonal did not block
        # it, the ear was clipped straight through that vertex, and what was
        # left collapsed to collinear points — so `prism` could not build a T
        # or an I-beam, refusing a perfectly valid profile as "degenerate".
        return (orient(a, b, p) >= 0 and orient(b, c, p) >= 0
                and orient(c, a, p) >= 0)

    pts = list(loop)
    tris: list[tuple] = []
    guard = 0
    while len(pts) > 3:
        guard += 1
        if guard > 10000:
            raise ValueError("ear clipping did not converge (self-intersecting loop?)")
        n = len(pts)
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            if orient(a, b, c) <= 0:
                continue                          # reflex or degenerate
            if any(inside(p, a, b, c) for j, p in enumerate(pts)
                   if p not in (a, b, c)):
                continue
            tris.append((a, b, c))
            pts.pop(i)
            break
        else:
            raise ValueError("no ear found (degenerate loop)")
    tris.append((pts[0], pts[1], pts[2]))
    return tris


def _canon_dir(d: Vec):
    # Scale- and sign-invariant direction key: divide by the first nonzero
    # component. Rational (ℚ) or quadratic-surd (ℚ[√d], from exact rotation).
    lead = None
    for v in d:
        if v != 0:
            lead = v
            break
    if lead is None:
        return None
    return tuple(v / lead for v in d)


def logical_edges(solid: Solid) -> list[dict]:
    """Solid edges as carrier lines with their two adjacent face planes —
    derived by exact grouping of polygon boundary segments. An edge is a
    line where exactly two distinct face planes meet."""
    from collections import defaultdict

    lines: dict = defaultdict(lambda: {"planes": {}, "tmin": None,
                                       "tmax": None, "point": None,
                                       "dir": None})
    for p in solid.polys:
        n = len(p.verts)
        for i in range(n):
            a, b = p.verts[i], p.verts[(i + 1) % n]
            cd = _canon_dir(sub(b, a))
            if cd is None:
                continue
            key = (cd, cross(a, cd))
            e = lines[key]
            e["planes"][p.plane.canonical()] = p.plane
            e["dir"] = cd
            ta, tb = dot(a, cd), dot(b, cd)
            lo, hi = min(ta, tb), max(ta, tb)
            e["tmin"] = lo if e["tmin"] is None else min(e["tmin"], lo)
            e["tmax"] = hi if e["tmax"] is None else max(e["tmax"], hi)
            if e["point"] is None:
                e["point"] = a
    out = []
    for e in lines.values():
        if len(e["planes"]) == 2:
            pa, pb = list(e["planes"].values())
            out.append({"point": e["point"], "dir": e["dir"],
                        "tmin": e["tmin"], "tmax": e["tmax"],
                        "plane_a": pa, "plane_b": pb})
    return out


def _unit_normal(plane: Plane) -> Vec | None:
    """Exact unit normal when it exists in the rationals — i.e. when |n|² is
    a perfect square IN ℚ — else None.

    The perfect-square test must be done on the RATIONAL |n|². It used to be
    ``root = isqrt(int(nn))``, and ``int()`` TRUNCATES a Fraction: a canonical
    normal of (1, 0, 1/6) — a lofted solid's slanted face — has |n|² = 37/36,
    ``int(37/36)`` is 1, ``isqrt(1)`` is 1, and 1·1 == 1, so the guard PASSED
    and handed back (1, 0, 1/6), whose length is √37/6 ≈ 1.0138. The same
    truncation let every non-axis-aligned normal through: canonical (3,4,0) is
    (3/4, 1, 0) with |n|² = 25/16, which also truncates to 1.

    An exactness guard made vacuous by an integer cast. Everything downstream
    trusted the result to be unit — ``chamfer_planar`` offsets ``distance``
    along ``cross(edge_dir, normal)`` — so a non-unit normal scaled every
    chamfer wedge. On a frustum, chamfer(d=1) returned 100.55 of an original
    784 with the answer audit reporting valid=True, and d=2 returned a
    NEGATIVE volume.

    p/q is a perfect rational square iff p and q are each perfect squares, and
    then √(p/q) = √p/√q exactly — the same test the polygon inset already uses
    for Pythagorean edges. No float, no tolerance (ADR-0019).
    """
    import math as _m

    from forgekernel.exact import as_fraction

    c = plane.canonical()[:3]
    nn = as_fraction(F(c[0]) * F(c[0]) + F(c[1]) * F(c[1]) + F(c[2]) * F(c[2]))
    if nn is None:
        # |n|² is not even rational, so there is certainly no RATIONAL unit
        # normal. as_fraction, not `.numerator`: a ROTATED body's plane normal
        # has SurdVal components, so |n|² arrives wearing a ℚ[√d] tag even when
        # its value is an integer — a 45°-turned box canonicalises to
        # n = (1,−1,0) and nn = 2. Reading .numerator off that raised
        # AttributeError straight through the seam, and chamfer's whole refusal
        # message became "'SurdVal' object has no attribute 'numerator'".
        return None
    p, q = nn.numerator, nn.denominator
    rp, rq = _m.isqrt(p), _m.isqrt(q)
    if rp * rp != p or rq * rq != q:
        return None
    root = F(rp) / F(rq)                      # F() is a 1-arg coercion here
    return (F(c[0]) / root, F(c[1]) / root, F(c[2]) / root)


def _unit_vec(v):
    """``v`` scaled to unit length, exactly — rational when it can be, else in
    ℚ[√d]. Used wherever a direction is assumed unit; assuming it without
    dividing is the defect this exists to prevent."""
    import math as _m

    from forgekernel.exact import as_fraction
    from forgekernel.surd import sqrt_rational

    raw = F(v[0]) * F(v[0]) + F(v[1]) * F(v[1]) + F(v[2]) * F(v[2])
    if raw == 0:
        raise ValueError("cannot normalise a zero vector")
    nn = as_fraction(raw)
    if nn is None:
        # |v|² itself left ℚ — a direction whose own length squared is a surd
        # needs a nested radical, which is a field above this one
        raise ValueError(
            f"cannot normalise exactly: |v|² = {raw!r} is not rational, so "
            "|v| needs a nested radical (arrives with K3.2)")
    p, q = nn.numerator, nn.denominator
    rp, rq = _m.isqrt(p), _m.isqrt(q)
    root = (F(rp) / F(rq) if rp * rp == p and rq * rq == q
            else sqrt_rational(nn))
    return (F(v[0]) / root, F(v[1]) / root, F(v[2]) / root)


def _unit_normal_exact(plane: Plane):
    """The unit normal over ℚ[√d] — exact even when |n| is irrational.

    ``_unit_normal`` deliberately returns None there, because its callers
    wanted a RATIONAL vector and silently handing back a near-unit one is what
    made chamfer 87% wrong on a taper. But "not rational" is not "not exact":
    |n| = √37 lives in ℚ[√37], which ``SurdVal`` holds, and dividing a rational
    normal by it gives components in that field with exact equality intact.

    Faces whose |n| lie in DIFFERENT quadratic fields cannot be combined; the
    surd arithmetic raises ``MixedRadicals`` (K3.1) on its own when a caller
    tries, so no check is needed here.
    """
    from forgekernel.exact import as_fraction
    from forgekernel.surd import sqrt_rational

    c = plane.canonical()[:3]
    exact = _unit_normal(plane)
    if exact is not None:
        return exact
    nn = as_fraction(F(c[0]) * F(c[0]) + F(c[1]) * F(c[1]) + F(c[2]) * F(c[2]))
    if nn is None:
        raise ValueError(
            "cannot take an exact unit normal: |n|² is not rational, so |n| "
            "needs a nested radical (arrives with K3.2)")
    root = sqrt_rational(nn)
    return (F(c[0]) / root, F(c[1]) / root, F(c[2]) / root)


def offset_solid_inward(solid: Solid, distance) -> Solid:
    """Erode a planar solid by ``distance`` along each face's OWN unit normal.

    This is the offset a closed SHELL needs. Insetting a cap outline by t in
    xy is the same thing only when the walls are perpendicular to that cap;
    for a tapered wall it is not, and gitcad's prism path returned 206.04 for
    a frustum whose true eroded complement gives 418.04 — 50.7% wrong, walls
    far thinner than asked for.

    Exact, and in a field the kernel already has. A face plane ``n·x = d``
    with rational ``n`` offsets inward to::

        n·x = d − t·|n|

    so the NORMAL stays rational and only the OFFSET leaves ℚ: for a frustum
    |n| = √37, i.e. ℚ[√37], which ``SurdVal`` holds. Faces whose |n| lie in
    DIFFERENT quadratic fields cannot share one field, and the surd arithmetic
    says so itself by raising ``MixedRadicals`` (K3.1) rather than rounding.

    Topology is preserved, not recomputed: each vertex is the exact
    intersection of the offset copies of the planes it lies on. That is valid
    while the combinatorics hold, which is checked on the RESULT rather than
    guessed at from the input — the eroded solid must be closed, strictly
    smaller, positive, and wholly inside the original. Anything else refuses,
    so a ``distance`` that would collapse or invert the solid cannot return a
    plausible wrong number (the failure chamfer had at d=2: −10345.5).
    """
    from forgekernel.surd import sqrt_rational

    t = F(distance)
    if t <= 0:
        raise ValueError("offset wants a positive distance")

    # distinct face planes, keyed by their canonical form so coplanar
    # fragments (a boolean splits walls freely) offset as ONE face
    planes: dict = {}
    for p in solid.polys:
        planes.setdefault(p.plane.canonical(), p.plane)

    def _len(n):
        """|n| exactly: rational when it is, else a SurdVal in ℚ[√k]."""
        nn = F(n[0]) * F(n[0]) + F(n[1]) * F(n[1]) + F(n[2]) * F(n[2])
        num, den = nn.numerator, nn.denominator
        import math as _m
        rn, rd = _m.isqrt(num), _m.isqrt(den)
        if rn * rn == num and rd * rd == den:
            return F(rn) / F(rd)
        return sqrt_rational(nn)

    shifted = {}
    for key, pl in planes.items():
        shifted[key] = (pl.n, pl.d - t * _len(pl.n))

    def _at(vertex):
        """The offset copies of every plane this vertex lies on, solved."""
        rows, rhs = [], []
        for key, pl in planes.items():
            if pl.side(vertex) == 0:
                n, d = shifted[key]
                rows.append((F(n[0]), F(n[1]), F(n[2])))
                rhs.append(d)
        if len(rows) < 3:
            raise ValueError(
                "offset: a vertex lies on fewer than three faces — the solid "
                "is not a closed polyhedron here")
        sol = _solve3(rows, rhs)
        if sol is None:
            raise ValueError(
                "offset: three faces at a vertex are not independent")
        # every OTHER incident plane must pass through the same point, or the
        # offset does not have this vertex's combinatorics and the answer
        # would be invented (a >3-face vertex whose offsets are not concurrent)
        for (a, b, c), d in zip(rows, rhs):
            if a * sol[0] + b * sol[1] + c * sol[2] != d:
                raise ValueError(
                    "offset: the offset faces at a vertex are not concurrent "
                    "— the erosion changes topology here (K4.2)")
        return sol

    cache: dict = {}
    out = []
    for p in solid.polys:
        verts = []
        for v in p.verts:
            k = tuple(v)
            if k not in cache:
                cache[k] = _at(v)
            verts.append(cache[k])
        out.append(Polygon(verts, p.source))
    eroded = Solid(out)

    # validity on the RESULT (never a guess about the input)
    vol, orig = eroded.volume(), solid.volume()
    if vol <= 0:
        raise ValueError(
            f"offset: distance {float(t)} leaves no solid (volume {float(vol)})")
    if vol >= orig:
        raise ValueError("offset: the eroded solid is not smaller")
    return eroded


def _solve3(rows, rhs):
    """Exact 3x3 solve by Cramer's rule — works over ℚ and ℚ[√d] alike,
    since both are fields with exact equality."""
    def det(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    d0 = det(rows)
    if d0 == 0:
        return None
    out = []
    for k in range(3):
        m = [list(r) for r in rows]
        for i in range(3):
            m[i][k] = rhs[i]
        out.append(det(m) / d0)
    return tuple(out)


def chamfer_planar(solid: Solid, distance, edges: list[dict] | None = None) -> Solid:
    """Exact chamfer on convex edges whose face normals admit rational
    unit vectors (axis-aligned and Pythagorean orientations). Each edge
    is cut by the plane through the two lines offset ``distance`` along
    each adjacent face — a parallelepiped tool per edge, subtracted with
    the exact boolean engine. Non-rational orientations refuse (K2
    brings bounded-error constructions)."""
    from forgekernel import csg

    d = F(distance)
    if d <= 0:
        raise ValueError("chamfer wants positive distance")
    todo = edges if edges is not None else logical_edges(solid)
    lo, hi = solid.bbox()
    extent = (hi[0] - lo[0]) + (hi[1] - lo[1]) + (hi[2] - lo[2]) + 1
    out = solid
    for e in todo:
        pa, pb = e["plane_a"], e["plane_b"]
        # ℚ[√d] is enough: a tapered face's |n| = √37 is irrational but EXACT,
        # and dividing a rational normal by it keeps exact equality. Refusing
        # here was right only while the alternative was a non-unit vector —
        # see _unit_normal's note. Faces in different quadratic fields raise
        # MixedRadicals from the arithmetic below (K3.1) on their own.
        na, nb = _unit_normal_exact(pa), _unit_normal_exact(pb)
        if na is None or nb is None:
            raise ValueError(
                "chamfer: face normal is not rational-unit (arrives at K2)")
        # The EDGE DIRECTION must be unit too, and `_canon_dir` does not make
        # it one: a frustum's slanted edges come back with |dir|² = 38. The
        # walk below is `distance × cross(u, n)`, so a non-unit u scales every
        # chamfer wedge by |u| exactly as a non-unit normal did — the same bug
        # one field over, and the reason fixing `_unit_normal` alone still left
        # a frustum's chamfer at 107.8 against a validated truth of 733.7.
        u = _unit_vec(e["dir"])
        p0 = e["point"]
        # direction from the edge into each face: perpendicular to both the
        # edge and the face normal, signed to point into the OTHER face's
        # negative half-space (exact convexity-aware sign choice)
        ca = cross(u, na)
        if pb.side(add(p0, ca)) > 0:
            ca = neg(ca)
        cb = cross(u, nb)
        if pa.side(add(p0, cb)) > 0:
            cb = neg(cb)
        if pb.side(add(p0, ca)) >= 0 or pa.side(add(p0, cb)) >= 0:
            continue                          # reflex edge: skip in K1.1
        qa = add(p0, smul(d, ca))
        qb = add(p0, smul(d, cb))
        span = sub(qb, qa)
        if is_zero(span):
            continue
        # parallelepiped tool: rectangle spanning the cut plane, extruded
        # toward the edge (the material side)
        mid = smul(Fraction(1, 2), add(qa, qb))
        toward = sub(p0, mid)                 # cut plane -> edge direction
        e1 = smul(extent / _norm1(u), u)
        e2 = smul(extent / _norm1(span), span)
        e3 = smul(Fraction(2), toward)
        base = sub(sub(mid, smul(Fraction(1, 2), e1)), smul(Fraction(1, 2), e2))
        tool = _parallelepiped(base, e1, e2, e3, "chamfer")
        out = csg.cut(out, tool)
    return out


def _norm1(v: Vec) -> Fraction:
    return abs(v[0]) + abs(v[1]) + abs(v[2])


def _parallelepiped(base: Vec, e1: Vec, e2: Vec, e3: Vec,
                    source: str) -> Solid:
    v = [base, add(base, e1), add(add(base, e1), e2), add(base, e2)]
    v += [add(p, e3) for p in v]
    faces = [([0, 3, 2, 1], "b"), ([4, 5, 6, 7], "t"), ([0, 1, 5, 4], "f"),
             ([2, 3, 7, 6], "k"), ([1, 2, 6, 5], "r"), ([3, 0, 4, 7], "l")]
    s = Solid([Polygon([v[i] for i in idx], f"{source}.{n}")
               for idx, n in faces])
    return s if s.volume() > 0 else Solid([p.flipped() for p in s.polys])


def _unit_dir(cd: Vec) -> Vec | None:
    import math as _m

    nn = int(cd[0] * cd[0] + cd[1] * cd[1] + cd[2] * cd[2])
    root = _m.isqrt(nn)
    if root * root != nn:
        return None
    return (cd[0] / root, cd[1] / root, cd[2] / root)


def _solve3(rows: list[Vec], rhs: list[Fraction]) -> Vec | None:
    """Exact 3x3 linear solve (Cramer). None when singular."""
    a, b, c = rows
    det = dot(a, cross(b, c))
    if det == 0:
        return None

    def rep(i: int, col: Vec) -> Fraction:
        m = [list(a), list(b), list(c)]
        for r, v in zip(m, rhs):
            r[i] = v
        return dot((m[0][0], m[0][1], m[0][2]),
                   cross((m[1][0], m[1][1], m[1][2]),
                         (m[2][0], m[2][1], m[2][2]))) / det

    # column replacement via transpose trick: solve A x = rhs
    ax = dot((rhs[0], a[1], a[2]), cross((rhs[1], b[1], b[2]), (rhs[2], c[1], c[2]))) / det
    ay = dot((a[0], rhs[0], a[2]), cross((b[0], rhs[1], b[2]), (c[0], rhs[2], c[2]))) / det
    az = dot((a[0], a[1], rhs[0]), cross((b[0], b[1], rhs[1]), (c[0], c[1], rhs[2]))) / det
    return (ax, ay, az)


def _tetra(p0: Vec, p1: Vec, p2: Vec, p3: Vec, source: str) -> Solid:
    s = Solid([Polygon([p0, p1, p2], f"{source}.a"),
               Polygon([p0, p2, p3], f"{source}.b"),
               Polygon([p0, p3, p1], f"{source}.c"),
               Polygon([p1, p3, p2], f"{source}.d")])
    return s if s.volume() > 0 else Solid([q.flipped() for q in s.polys])


def chamfer_corners(solid: Solid, distance,
                    edges: list[dict]) -> Solid:
    """Vertex truncation matching industrial chamfer semantics (the
    OCCT/SolidWorks corner facet). Geometry, derived exactly from the
    first real ref-vs-OCCT disagreement (5568 pure plane-cuts vs 16688/3
    oracle; delta d^3/12 per corner, hand-verified both ways):

    at a corner where three chamfered edges meet, the remaining apex
    pyramid is bounded by the three chamfer planes; the corner facet
    passes through the three points where PAIRWISE chamfer-plane
    intersection lines pierce the original faces. The removed piece is
    the exact rational tetrahedron (facet triangle + chamfer triple
    point), cut with the exact boolean engine."""
    from collections import defaultdict

    from forgekernel import csg

    d = F(distance)
    at_vertex: dict = defaultdict(list)
    for e in edges:
        cd = e["dir"]
        nn = dot(cd, cd)
        p0 = e["point"]
        t0 = dot(p0, cd)
        for t_end, sign in ((e["tmin"], 1), (e["tmax"], -1)):
            v = add(p0, smul((t_end - t0) / nn, cd))
            at_vertex[v].append((smul(F(sign), cd),
                                 e["plane_a"], e["plane_b"]))
    out = solid
    for v, incident in at_vertex.items():
        if len(incident) != 3:
            continue
        units = [_unit_dir(cd) for cd, _, _ in incident]
        if any(u is None for u in units):
            continue                              # K2: non-rational dirs
        # chamfer plane of edge k: normal = u_i + u_j, through v + d*u_i
        m = [add(units[(k + 1) % 3], units[(k + 2) % 3]) for k in range(3)]
        rhs = [dot(m[k], add(v, smul(d, units[(k + 1) % 3])))
               for k in range(3)]
        apex = _solve3(m, rhs)
        if apex is None:
            continue
        # face_k = the original face shared by edges i and j
        pts = []
        ok = True
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            keys_i = {incident[i][1].coplanar_key(),
                      incident[i][2].coplanar_key()}
            shared = None
            for pl in (incident[j][1], incident[j][2]):
                if pl.coplanar_key() in keys_i:
                    shared = pl
                    break
            if shared is None:
                ok = False
                break
            p = _solve3([m[i], m[j], shared.n], [rhs[i], rhs[j], shared.d])
            if p is None:
                ok = False
                break
            pts.append(p)
        if not ok:
            continue
        tool = _tetra(pts[0], pts[1], pts[2], apex, "corner")
        if tool.volume() == 0:
            continue
        out = csg.cut(out, tool)
    return out


def prismatoid(bottom: list[tuple], z0, top: list[tuple], z1,
               source: str = "prismatoid") -> "Solid":
    """Exact solid between two same-count CCW xy loops at heights z0<z1:
    bottom cap, top cap, and side quads (each split into 2 triangles so a
    twisted/tapered side stays exactly planar-triangulated and closed)."""
    z0, z1 = F(z0), F(z1)
    b = [(F(x), F(y)) for x, y in bottom]
    tp = [(F(x), F(y)) for x, y in top]
    if len(b) != len(tp) or len(b) < 3:
        raise ValueError("prismatoid needs two equal-length loops (>=3)")
    if _loop_area2(b) < 0:
        b, tp = list(reversed(b)), list(reversed(tp))
    polys: list[Polygon] = []
    for a, bb, c in _ear_clip(b):
        polys.append(Polygon([vec(*a, z0), vec(*c, z0), vec(*bb, z0)],
                             f"{source}.bottom"))
    for a, bb, c in _ear_clip(tp):
        polys.append(Polygon([vec(*a, z1), vec(*bb, z1), vec(*c, z1)],
                             f"{source}.top"))
    n = len(b)
    for i in range(n):
        j = (i + 1) % n
        b0, b1 = b[i], b[j]
        t0, t1 = tp[i], tp[j]
        # side quad b0-b1-t1-t0 -> two triangles (consistent winding)
        polys.append(Polygon([vec(*b0, z0), vec(*b1, z0), vec(*t1, z1)],
                             f"{source}.side{i}"))
        polys.append(Polygon([vec(*b0, z0), vec(*t1, z1), vec(*t0, z1)],
                             f"{source}.side{i}"))
    return Solid(polys)


def ruled_stack(sections: list[tuple[list[tuple], object]],
                source: str = "loft") -> "Solid":
    """Exact ruled loft through ≥3 same-count xy loops at strictly
    increasing heights, built DIRECTLY as one closed shell: every wall
    band plus the two outer caps.

    This is the same point set as the BSP-union fold of the pairwise
    prismatoids — the interface caps cancel exactly, because piece i's top
    cap and piece i+1's bottom cap are the same ear-clipped section polygon
    with opposite orientation — at O(sections) cost instead of the fold's
    O(n²) polygon splits (a 13×34 crankbait hull: ~26 s → milliseconds).

    Raises ``ValueError`` when the preconditions for the cancellation
    argument do not hold (fewer than 3 sections, unequal loop lengths,
    non-monotonic z, a zero-area section, or mixed loop orientation);
    callers fall back to the fold, whose behaviour is unchanged.
    """
    if len(sections) < 3:
        raise ValueError("ruled_stack wants >=3 sections (2 is prismatoid)")
    zs = [F(z) for _, z in sections]
    loops = [[(F(x), F(y)) for x, y in lp] for lp, _ in sections]
    n = len(loops[0])
    if n < 3 or any(len(lp) != n for lp in loops):
        raise ValueError("ruled_stack needs equal-length loops (>=3)")
    if any(z1 <= z0 for z0, z1 in zip(zs, zs[1:])):
        raise ValueError("ruled_stack needs strictly increasing z")
    areas = [_loop_area2(lp) for lp in loops]
    if any(a == 0 for a in areas):
        raise ValueError("ruled_stack: zero-area section")
    if any((a < 0) != (areas[0] < 0) for a in areas):
        raise ValueError("ruled_stack: mixed section orientation")
    if areas[0] < 0:                     # normalise CCW, as prismatoid does
        loops = [list(reversed(lp)) for lp in loops]
    polys: list[Polygon] = []
    for a, bb, c in _ear_clip(loops[0]):
        polys.append(Polygon([vec(*a, zs[0]), vec(*c, zs[0]), vec(*bb, zs[0])],
                             f"{source}.bottom"))
    for a, bb, c in _ear_clip(loops[-1]):
        polys.append(Polygon([vec(*a, zs[-1]), vec(*bb, zs[-1]),
                              vec(*c, zs[-1])], f"{source}.top"))
    for k, (b, tp) in enumerate(zip(loops, loops[1:])):
        z0, z1 = zs[k], zs[k + 1]
        for i in range(n):
            j = (i + 1) % n
            b0, b1 = b[i], b[j]
            t0, t1 = tp[i], tp[j]
            # side quad b0-b1-t1-t0 -> two triangles (consistent winding,
            # exactly as prismatoid emits them)
            polys.append(Polygon([vec(*b0, z0), vec(*b1, z0), vec(*t1, z1)],
                                 f"{source}.s{k}.side{i}"))
            polys.append(Polygon([vec(*b0, z0), vec(*t1, z1), vec(*t0, z1)],
                                 f"{source}.s{k}.side{i}"))
    return Solid(polys)


def _rational_sqrt(q: Fraction) -> Fraction | None:
    """√q as an exact Fraction when q is a square in ℚ, else None."""
    import math as _m

    if q < 0:
        return None
    n, d = q.numerator, q.denominator
    rn, rd = _m.isqrt(n), _m.isqrt(d)
    return Fraction(rn, rd) if rn * rn == n and rd * rd == d else None


def _prism_footprint(solid: Solid) -> tuple[list[tuple], Fraction, Fraction]:
    """Recover ``(ccw_loop, z0, z1)`` from a Solid that is EXACTLY the +z
    extrusion of one simple polygon — verified by REBUILDING the claimed
    prism and requiring the exact symmetric difference to be empty.

    The predecessor gate here only checked that every vertex sat on one of
    the 4 bbox xy-corners, which a triangular prism over 3 bbox corners
    satisfies — it was then silently drafted as the FULL-BOX frustum (the
    §10 ``_box_check`` defect class: a subset test standing in for an
    equality test). The footprint itself is now the gate.
    """
    from forgekernel import csg

    zs = sorted({v[2] for p in solid.polys for v in p.verts})
    if len(zs) != 2:
        raise ValueError(
            "draft: solid is not a prism (vertices at more than two z "
            "levels) — general drafted solids arrive at K2.3")
    z0, z1 = zs
    # bottom-cap fragments: polys entirely at z0 (walls span both levels)
    bottom = [p for p in solid.polys if all(v[2] == z0 for v in p.verts)]
    if not bottom:
        raise ValueError("draft: no bottom cap found — arrives at K2.3")
    # union boundary of the cap = directed xy edges that do not cancel
    net: dict[tuple, int] = {}
    for p in bottom:
        n = len(p.verts)
        for i in range(n):
            a, b = p.verts[i], p.verts[(i + 1) % n]
            e = ((a[0], a[1]), (b[0], b[1]))
            r = (e[1], e[0])
            if net.get(r, 0) > 0:
                net[r] -= 1
            else:
                net[e] = net.get(e, 0) + 1
    boundary = [e for e, c in net.items() if c > 0]
    if any(c > 1 for c in net.values()):
        raise ValueError(
            "draft: non-manifold footprint boundary — arrives at K2.3")
    nxt = {}
    for a, b in boundary:
        if a in nxt:
            raise ValueError(
                "draft: footprint with holes or multiple loops arrives "
                "at K2.3")
        nxt[a] = b
    start = boundary[0][0]
    loop, cur = [start], nxt[start]
    while cur != start:
        loop.append(cur)
        cur = nxt.get(cur)
        if cur is None or len(loop) > len(boundary):
            raise ValueError(
                "draft: footprint boundary does not close into one loop — "
                "arrives at K2.3")
    if len(loop) != len(boundary):
        raise ValueError(
            "draft: footprint with holes or multiple loops arrives at K2.3")
    if _loop_area2(loop) < 0:
        loop.reverse()
    # merge consecutive collinear edges (same carrier, same sense) so a
    # wall split by an authoring midpoint is one logical footprint edge
    merged: list[tuple] = []
    m = len(loop)
    for i in range(m):
        p0, p1, p2 = loop[i - 1], loop[i], loop[(i + 1) % m]
        d1 = (p1[0] - p0[0], p1[1] - p0[1])
        d2 = (p2[0] - p1[0], p2[1] - p1[1])
        crossz = d1[0] * d2[1] - d1[1] * d2[0]
        if crossz == 0:
            if d1[0] * d2[0] + d1[1] * d2[1] <= 0:
                raise ValueError(
                    "draft: zero-width spike in the footprint — arrives "
                    "at K2.3")
            continue                      # collinear same-sense: drop p1
        merged.append(p1)
    if len(merged) < 3:
        raise ValueError("draft: degenerate footprint — arrives at K2.3")
    # THE GATE: the solid must equal the extrusion of its own footprint,
    # exactly (empty symmetric difference), not merely resemble it
    rebuilt = Solid.prism(merged, z1 - z0).translated((F(0), F(0), z0))
    if csg.cut(solid, rebuilt).volume() != 0 or \
            csg.cut(rebuilt, solid).volume() != 0:
        raise ValueError(
            "draft: solid is not the extrusion of its bottom footprint — "
            "general drafted solids arrive at K2.3")
    return merged, z0, z1


def draft_prism(solid: Solid, t, neutral_z=None, parting_z=None,
                drafted_walls=None) -> Solid:
    """Draft the vertical walls of an extruded prism about pull +z — exact.

    ``t`` is the RATIONAL TANGENT of the draft angle (the spec; degree sugar
    lives in ``kernel.draft``). Each drafted wall's carrier line moves inward
    by ``d(z) = (z − neutral_z)·t`` for a single pull, or by
    ``d(z) = |z − parting_z|·t`` for a parting-line draft — the wall splits
    at the parting plane into two half-drafts and the section is widest AT
    that plane. Note a neutral plane above the base gives d < 0 below it:
    the taper is a transform, not a cut, so material is ADDED outside the
    original footprint there (flare) — intentional, mold-maker semantics.

    ``drafted_walls``: ``None`` drafts every wall; otherwise an iterable of
    frozensets of the two xy endpoints naming each wall to draft.

    Exact-field membership (why the guards are shaped this way): with every
    footprint edge axis-aligned the drafted planes have normals like
    (1, 0, ±t) and every inset vertex re-solves rationally — pure ℚ. A
    non-axis-aligned edge direction (dx, dy) needs the unit normal
    √(dx²+dy²): rational only for Pythagorean directions, one radical
    (SurdVal) otherwise — both arrive at K2.3.
    """
    from forgekernel import csg

    t = F(t)
    if not isinstance(t, Fraction):
        raise ValueError(
            "draft tangent must be rational (ℚ) — surd tangents arrive "
            "with the general tower (K2.3)")
    loop, z0, z1 = _prism_footprint(solid)
    n = len(loop)
    dirs = []
    for i in range(n):
        (x1_, y1_), (x2_, y2_) = loop[i], loop[(i + 1) % n]
        dx, dy = x2_ - x1_, y2_ - y1_
        if dx != 0 and dy != 0:
            if _rational_sqrt(dx * dx + dy * dy) is not None:
                raise ValueError(
                    f"draft: Pythagorean-direction footprint edge "
                    f"({float(dx):g},{float(dy):g}) stays in ℚ but the "
                    "rational-miter inset arrives at K2.3")
            raise ValueError(
                f"draft: footprint edge direction ({float(dx):g},"
                f"{float(dy):g}) needs √(dx²+dy²) — leaves ℚ for ℚ[√d] "
                "(SurdVal/BiSurd tower), arrives at K2.3")
        dirs.append((dx, dy))
    # which walls draft?
    if drafted_walls is None:
        flags = [True] * n
    else:
        wanted = list(dict.fromkeys(frozenset(w) for w in drafted_walls))
        if not wanted:
            raise ValueError(
                "draft of zero walls is the identity — name at least one "
                "wall, or pass None to draft them all")
        edge_sets = [frozenset({loop[i], loop[(i + 1) % n]})
                     for i in range(n)]
        flags = [False] * n
        for w in wanted:
            try:
                flags[edge_sets.index(w)] = True
            except ValueError:
                raise ValueError(
                    "draft: selected wall does not match a footprint edge "
                    "of this prism (collinear-split walls arrive at K2.3)")
    # neutral / parting bands: list of (z_lo, z_hi, d(z_lo), d(z_hi))
    if parting_z is not None:
        zp = F(parting_z)
        if not z0 < zp < z1:
            raise ValueError(
                f"draft: parting plane z={float(zp):g} must lie strictly "
                f"inside the prism z∈({float(z0):g},{float(z1):g})")
        bands = [(z0, zp, t * (zp - z0), F(0)),
                 (zp, z1, F(0), t * (z1 - zp))]
    else:
        nz = F(0) if neutral_z is None else F(neutral_z)
        bands = [(z0, z1, t * (z0 - nz), t * (z1 - nz))]
    d_extremes = [d for b in bands for d in (b[2], b[3])]
    d_max_abs = max(abs(d) for d in d_extremes)

    def inset(d: Fraction) -> list[tuple]:
        # each edge's carrier moves +d along its inward normal (CCW left)
        carriers = []                     # ('x'|'y', coordinate)
        for i in range(n):
            dx, dy = dirs[i]
            off = d if flags[i] else F(0)
            if dy == 0:                   # horizontal: inward is (0, ±1)
                carriers.append(("y", loop[i][1] + (off if dx > 0 else -off)))
            else:                         # vertical: inward is (∓1, 0)
                carriers.append(("x", loop[i][0] + (-off if dy > 0 else off)))
        out = []
        for i in range(n):
            a, b = carriers[i - 1], carriers[i]   # vertex between edges
            if a[0] == b[0]:              # cannot happen post-merge
                raise ValueError(
                    "draft: consecutive parallel footprint edges — arrives "
                    "at K2.3")
            x = a[1] if a[0] == "x" else b[1]
            y = a[1] if a[0] == "y" else b[1]
            out.append((x, y))
        return out

    # GUARDS — exact, and refusals name the measured distance-to-event.
    # (1) no edge may collapse or flip anywhere in the band: each inset
    # vertex coordinate is affine in d, so signed length along the original
    # direction is affine in d and positivity at the band extremes is
    # positivity throughout.
    for d in d_extremes:
        if d == 0:
            continue
        ring = inset(d)
        for i in range(n):
            dx, dy = dirs[i]
            v = (ring[(i + 1) % n][0] - ring[i][0],
                 ring[(i + 1) % n][1] - ring[i][1])
            signed = v[0] * dx + v[1] * dy
            if signed <= 0:
                orig = abs(dx) + abs(dy)          # axis-aligned length
                raise ValueError(
                    f"draft consumes footprint edge {i} (length "
                    f"{float(orig):g} shrinks past zero at draft depth "
                    f"d={float(d):g}) — reduce the tangent or the band "
                    "height, or exclude the edge")
        if _loop_area2(ring) <= 0:
            raise ValueError(
                "draft consumes the whole footprint at depth "
                f"d={float(d):g} — reduce the tangent or the band height")
    # (2) simplicity, conservatively: every inset vertex moves ≤ d_max in
    # L∞, so if every pair of NON-ADJACENT original edges is farther apart
    # than 2·d_max in L∞ no new crossing can appear at any depth. Exact:
    # axis-aligned segments have rational L∞ gaps.
    if d_max_abs > 0:
        segs = []
        for i in range(n):
            (ax, ay), (bx, by) = loop[i], loop[(i + 1) % n]
            segs.append((min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)))
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue              # adjacent around the loop
                sa, sb = segs[i], segs[j]
                gx = max(sa[0] - sb[1], sb[0] - sa[1], F(0))
                gy = max(sa[2] - sb[3], sb[2] - sa[3], F(0))
                gap = max(gx, gy)
                if gap <= 2 * d_max_abs:
                    raise ValueError(
                        f"draft moves non-adjacent walls (edges {i},{j}) "
                        f"within touching range: gap {float(gap):g} ≤ "
                        f"2·d_max {float(2 * d_max_abs):g} — reduce the "
                        "tangent or the band height")
    pieces = [prismatoid(inset(d_lo), lo_z, inset(d_hi), hi_z, "draft")
              for lo_z, hi_z, d_lo, d_hi in bands]
    out = pieces[0]
    for piece in pieces[1:]:
        out = csg.union(out, piece)
    return out


def shell_box(solid: Solid, thickness) -> Solid:
    """Hollow an axis-aligned rectangular prism to wall thickness t (all
    faces closed — a shell with no openings). Result = outer minus the
    inner box inset by t on every face. Exact. Non-box or t too large
    refuse (K2.3 / invalid)."""
    from forgekernel import csg

    t = F(thickness)
    lo, hi = solid.bbox()
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = {(x0, y0), (x1, y0), (x1, y1), (x0, y1)}
    for p in solid.polys:
        for vx, vy, vz in p.verts:
            if (vx, vy) not in corners or vz not in (z0, z1):
                raise ValueError("shell of a non-box solid arrives at K2.3")
    if 2 * t >= min(x1 - x0, y1 - y0, z1 - z0):
        raise ValueError("shell thickness exceeds half the smallest dimension")
    inner = Solid.box(x1 - x0 - 2 * t, y1 - y0 - 2 * t, z1 - z0 - 2 * t,
                      "shell.void").translated(
                          (x0 + t, y0 + t, z0 + t))
    return csg.cut(solid, inner)
