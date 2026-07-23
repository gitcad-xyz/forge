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
