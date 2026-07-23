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
        from math import gcd as _gcd

        def canon_dir(d: Vec) -> Vec | None:
            den = 1
            for v in d:
                den = den * v.denominator // _gcd(den, v.denominator)
            ints = [int(v * den) for v in d]
            g = 0
            for v in ints:
                g = _gcd(g, abs(v))
            if g == 0:
                return None
            ints = [v // g for v in ints]
            for v in ints:
                if v != 0:
                    if v < 0:
                        ints = [-w for w in ints]
                    break
            return (F(ints[0]), F(ints[1]), F(ints[2]))

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
        return (orient(a, b, p) > 0 and orient(b, c, p) > 0
                and orient(c, a, p) > 0)

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
    from math import gcd as _g

    den = 1
    for v in d:
        den = den * v.denominator // _g(den, v.denominator)
    ints = [int(v * den) for v in d]
    g = 0
    for v in ints:
        g = _g(g, abs(v))
    if g == 0:
        return None
    ints = [v // g for v in ints]
    for v in ints:
        if v != 0:
            if v < 0:
                ints = [-w for w in ints]
            break
    return (F(ints[0]), F(ints[1]), F(ints[2]))


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
    """Exact unit normal when it exists in the rationals (axis-aligned
    faces, and any face whose |n| is a perfect rational square)."""
    c = plane.canonical()[:3]
    nn = c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
    import math as _m

    root = _m.isqrt(int(nn))
    if root * root != int(nn):
        return None
    return (F(c[0]) / root, F(c[1]) / root, F(c[2]) / root)


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
        na, nb = _unit_normal(pa), _unit_normal(pb)
        if na is None or nb is None:
            raise ValueError(
                "chamfer: face normal is not rational-unit (arrives at K2)")
        u = e["dir"]
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


def draft_box(solid: Solid, t: Fraction, neutral_z: Fraction) -> Solid:
    """Draft ALL four vertical faces of an axis-aligned rectangular prism
    into a frustum, exact. Inset at height z is (z-neutral_z)*t on each
    side. General (non-rectangular) prism draft arrives at K2.3."""
    lo, hi = solid.bbox()
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    # verify rectangular footprint: all vertices at the 4 xy corners
    corners = {(x0, y0), (x1, y0), (x1, y1), (x0, y1)}
    for p in solid.polys:
        for vx, vy, _vz in p.verts:
            if (vx, vy) not in corners:
                raise ValueError("draft of a non-rectangular prism arrives at K2.3")

    def rect(z):
        d = (z - neutral_z) * t
        return [(x0 + d, y0 + d), (x1 - d, y0 + d),
                (x1 - d, y1 - d), (x0 + d, y1 - d)]

    return prismatoid(rect(z0), z0, rect(z1), z1, "draft")


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
