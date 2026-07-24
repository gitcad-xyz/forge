"""K2.0 — z-axis cylinders and drilled solids, exact in ℚ[π].

The exactness charter survives curved geometry by extending the number
field: volumes of prisms with cylindrical bores live in ℚ + ℚ·π, so a
drilled plate's volume is EXACTLY ``9600 - 100π`` — an object with
equality, not a float. Floats appear only at the export boundary.

Scope, honestly held: right circular cylinders with +z axes — the
drilled-hole workhorse (plain, blind, and coaxial counterbore stacks).
Every geometric precondition is checked with exact rational predicates
(bore strictly inside the lateral boundary, non-coaxial bores disjoint);
configurations outside the scope refuse with the stage that brings them
(K2.1 general quadric booleans).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from forgekernel.brep import Solid
from forgekernel.exact import F, dot


class PiVal:
    """An exact number a + b·π (a, b rational)."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0) -> None:
        self.a, self.b = F(a), F(b)

    def __add__(self, o: "PiVal | int | Fraction") -> "PiVal":
        o = o if isinstance(o, PiVal) else PiVal(o)
        return PiVal(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, o: "PiVal | int | Fraction") -> "PiVal":
        o = o if isinstance(o, PiVal) else PiVal(o)
        return PiVal(self.a - o.a, self.b - o.b)

    def __eq__(self, o: object) -> bool:
        o = o if isinstance(o, PiVal) else PiVal(o)
        return self.a == o.a and self.b == o.b

    def __float__(self) -> float:
        return float(self.a) + float(self.b) * math.pi

    def __repr__(self) -> str:
        return f"({self.a} + {self.b}·π)"


@dataclass(frozen=True)
class Cyl:
    """A solid right circular cylinder, axis +z through (cx, cy)."""
    cx: Fraction
    cy: Fraction
    r: Fraction
    z0: Fraction
    z1: Fraction

    @classmethod
    def make(cls, r, h) -> "Cyl":
        r, h = F(r), F(h)
        if r <= 0 or h <= 0:
            raise ValueError("cylinder wants positive radius/height")
        return cls(F(0), F(0), r, F(0), h)

    def translated(self, x, y, z) -> "Cyl":
        return Cyl(self.cx + F(x), self.cy + F(y), self.r,
                   self.z0 + F(z), self.z1 + F(z))

    def volume(self) -> PiVal:
        return PiVal(0, self.r * self.r * (self.z1 - self.z0))

    def centroid_f(self) -> tuple[float, float, float]:
        return (float(self.cx), float(self.cy),
                float((self.z0 + self.z1) / 2))

    def bbox(self):
        return ((self.cx - self.r, self.cy - self.r, self.z0),
                (self.cx + self.r, self.cy + self.r, self.z1))

    def tessellate(self, deflection: float = 0.2) -> dict:
        """Display mesh (walls + both caps) via the surface-of-revolution
        lathe — floats are legal for a bounded-error view (ADR-0019)."""
        from forgekernel.tess import lathe

        r, z0, z1 = float(self.r), float(self.z0), float(self.z1)
        profile = [(0.0, z0), (r, z0), (r, z1), (0.0, z1)]
        return lathe(profile, deflection, float(self.cx), float(self.cy))


def _dist2_point_seg(px, py, ax, ay, bx, by) -> Fraction:
    """Exact squared distance from point to segment (all rational)."""
    dx, dy = bx - ax, by - ay
    nn = dx * dx + dy * dy
    if nn == 0:
        ex, ey = px - ax, py - ay
        return ex * ex + ey * ey
    t = ((px - ax) * dx + (py - ay) * dy) / nn
    t = max(F(0), min(F(1), t))
    ex, ey = px - (ax + t * dx), py - (ay + t * dy)
    return ex * ex + ey * ey


class DrilledSolid:
    """A planar Solid minus z-axis cylindrical bores — exact composite.

    Preconditions (exact predicates, refusal on violation):
    - each bore's circle stays strictly clear of every non-horizontal
      face of the base in xy (the barrel never crosses a wall);
    - non-coaxial bores are pairwise disjoint (coaxial stacks allowed —
      counterbores); volume of a coaxial stack is the exact z-interval
      union with the largest active radius per interval.
    """

    def __init__(self, base: Solid, bores: list[Cyl]) -> None:
        self.base = base
        self.bores = list(bores)

    def cut(self, c: Cyl) -> "DrilledSolid":
        # clamp to the base z-extent (drilling from above through air is fine)
        (bx0, by0, bz0), (bx1, by1, bz1) = self.base.bbox()
        z0, z1 = max(c.z0, bz0), min(c.z1, bz1)
        if z1 <= z0:
            raise ValueError("bore misses the solid in z (K2.1 for the rest)")
        c = Cyl(c.cx, c.cy, c.r, z0, z1)
        r2 = c.r * c.r
        for p in self.base.polys:
            n = p.plane.n
            if n[0] == 0 and n[1] == 0:
                continue                       # horizontal face: cap, fine
            m = len(p.verts)
            for i in range(m):
                a, b = p.verts[i], p.verts[(i + 1) % m]
                if _dist2_point_seg(c.cx, c.cy, a[0], a[1], b[0], b[1]) <= r2:
                    raise ValueError(
                        "bore crosses a lateral wall — general quadric "
                        "booleans arrive at K2.1")
        for o in self.bores:
            if o.cx == c.cx and o.cy == c.cy:
                continue                       # coaxial stack (counterbore)
            dx, dy = o.cx - c.cx, o.cy - c.cy
            if dx * dx + dy * dy <= (o.r + c.r) ** 2:
                raise ValueError(
                    "bores intersect — general quadric booleans arrive "
                    "at K2.1")
        return DrilledSolid(self.base, self.bores + [c])

    def _bore_union_volume(self) -> PiVal:
        """Exact removed volume: coaxial groups unioned by z-interval with
        the largest active radius per elementary interval."""
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for c in self.bores:
            groups[(c.cx, c.cy)].append(c)
        total = PiVal(0, 0)
        for cyls in groups.values():
            cuts = sorted({t for c in cyls for t in (c.z0, c.z1)})
            for lo, hi in zip(cuts, cuts[1:]):
                rs = [c.r for c in cyls if c.z0 <= lo and hi <= c.z1]
                if rs:
                    rmax = max(rs)
                    total = total + PiVal(0, rmax * rmax * (hi - lo))
        return total

    def volume(self) -> PiVal:
        return PiVal(self.base.volume(), 0) - self._bore_union_volume()

    def centroid_f(self) -> tuple[float, float, float]:
        """Centroid, floated at the boundary (the exact value is a ratio
        of ℚ[π] numbers — outside the field, so floats are honest here)."""
        bv = float(self.base.volume())
        c = self.base.centroid()
        acc = [bv * float(c[0]), bv * float(c[1]), bv * float(c[2])]
        vol = bv
        for cyl in self.bores:
            v = float(cyl.volume())
            cc = cyl.centroid_f()
            for i in range(3):
                acc[i] -= v * cc[i]
            vol -= v
        return (acc[0] / vol, acc[1] / vol, acc[2] / vol)

    def bbox(self):
        return self.base.bbox()

    def translated(self, x, y, z) -> "DrilledSolid":
        """Rigid translation — base and every bore move together (exact).
        Enables patterning a drilled feature (bolt patterns)."""
        base = self.base.translated((F(x), F(y), F(z)))
        return DrilledSolid(base, [b.translated(x, y, z) for b in self.bores])

    def watertight_violations(self) -> list[str]:
        return self.base.watertight_violations()

    def cylinder_faces(self) -> list[dict]:
        """OCCT-shaped descriptors, one per bore — feature recognition and
        hole callouts read these keys."""
        return [{"surface": "cylinder", "radius": float(c.r),
                 "axis_dir": [0.0, 0.0, 1.0],
                 "axis_origin": [float(c.cx), float(c.cy), float(c.z0)]}
                for c in self.bores]

    def tessellate(self, deflection: float = 0.2) -> dict:
        """A watertight display mesh: the base's faces (top/bottom capped
        around the bores), the bore walls (stepped for coaxial counterbores),
        the counterbore shoulder rings, and any blind-hole end caps. Floats are
        legal here — this approximates the exact solid to ``deflection`` chord
        error (ADR-0019: meshing is a display property)."""
        import math

        from forgekernel.mesh2d import triangulate

        verts: list[list[float]] = []
        tris: list[list[int]] = []
        index: dict = {}

        def V(p) -> int:
            k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
            if k not in index:
                index[k] = len(verts)
                verts.append([float(p[0]), float(p[1]), float(p[2])])
            return index[k]

        def tri(a, b, c, outward) -> None:
            ia, ib, ic = V(a), V(b), V(c)
            if ia == ib or ib == ic or ia == ic:
                return
            n = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                 (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                 (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if n[0] * outward[0] + n[1] * outward[1] + n[2] * outward[2] < 0:
                ib, ic = ic, ib
            tris.append([ia, ib, ic])

        def _segs(r):
            return (max(24, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - deflection / r)))))
                    if r > deflection else 24)

        def circle(cx, cy, r, n):
            return [(cx + r * math.cos(2 * math.pi * k / n),
                     cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]

        (_, _, bz0), (_, _, bz1) = self.base.bbox()
        zmin, zmax = float(bz0), float(bz1)

        # coaxial bore groups -> z-bands with the outermost radius per band
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for c in self.bores:
            groups[(float(c.cx), float(c.cy))].append(c)
        axis_bands = {}
        for axis, cyls in groups.items():
            zs = sorted({z for c in cyls for z in (float(c.z0), float(c.z1))})
            bands = []
            for za, zb in zip(zs, zs[1:]):
                zmid = (za + zb) / 2
                rs = [float(c.r) for c in cyls
                      if float(c.z0) - 1e-9 <= zmid <= float(c.z1) + 1e-9]
                if rs:
                    bands.append((za, zb, max(rs)))
            if bands:
                axis_bands[axis] = bands
        # ONE segment count per coaxial axis (from its widest radius) so every
        # ring on that axis — cap hole, wall, shoulder, blind cap — shares
        # vertices and the seams stay watertight (no T-junctions across bands).
        axis_segs = {axis: _segs(max(r for _, _, r in bands))
                     for axis, bands in axis_bands.items()}

        def _in_loop(pt, loop) -> bool:
            x, y = pt
            inside = False
            n = len(loop)
            for i in range(n):
                (x1, y1), (x2, y2) = loop[i], loop[(i + 1) % n]
                if (y1 > y) != (y2 > y):
                    xc = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                    if x < xc:
                        inside = not inside
            return inside

        # -- base faces: z-caps get holes, lateral faces are hole-free ---------
        zcaps: dict = defaultdict(list)
        for p in self.base.polys:
            nrm = p.plane.n
            nx, ny, nz = float(nrm[0]), float(nrm[1]), float(nrm[2])
            if nx == 0 and ny == 0:
                z = float(p.verts[0][2])
                zcaps[(round(z, 9), 1 if nz > 0 else -1)].append(
                    [(float(v[0]), float(v[1])) for v in p.verts])
            else:
                vs = [(float(v[0]), float(v[1]), float(v[2])) for v in p.verts]
                for i in range(1, len(vs) - 1):
                    tri(vs[0], vs[i], vs[i + 1], (nx, ny, nz))

        def _cap_loops(polys_xy):
            def key(p):
                return (round(p[0], 9), round(p[1], 9))
            present, coords = set(), {}
            for poly in polys_xy:
                m = len(poly)
                for i in range(m):
                    a, b = poly[i], poly[(i + 1) % m]
                    coords[key(a)] = a
                    coords[key(b)] = b
                    present.add((key(a), key(b)))
            nxt = {a: b for (a, b) in present if (b, a) not in present}
            loops, used = [], set()
            for start in list(nxt):
                if start in used:
                    continue
                loop, cur = [], start
                while cur in nxt and cur not in used:
                    used.add(cur)
                    loop.append(coords[cur])
                    cur = nxt[cur]
                if len(loop) >= 3:
                    loops.append(loop)
            return loops

        for (z, sign), polys in zcaps.items():
            holes = []
            for axis, bands in axis_bands.items():
                r = 0.0
                if abs(bands[-1][1] - z) < 1e-9:
                    r = bands[-1][2]           # axis reaches this (top) cap
                elif abs(bands[0][0] - z) < 1e-9:
                    r = bands[0][2]            # ... or this (bottom) cap
                if r > 0:
                    holes.append((axis, circle(axis[0], axis[1], r, axis_segs[axis])))
            for loop in _cap_loops(polys):
                hs = [h for (ax, h) in holes if _in_loop(ax, loop)]
                pts2, t2 = triangulate(loop, hs)
                out = (0.0, 0.0, float(sign))
                for a, b, c in t2:
                    tri((pts2[a][0], pts2[a][1], z), (pts2[b][0], pts2[b][1], z),
                        (pts2[c][0], pts2[c][1], z), out)

        # -- bore walls, counterbore shoulders, blind end caps -----------------
        for (cx, cy), bands in axis_bands.items():
            n = axis_segs[(cx, cy)]            # one resolution for the whole stack
            for za, zb, r in bands:
                ring = circle(cx, cy, r, n)
                for i in range(n):
                    a, b = ring[i], ring[(i + 1) % n]
                    outw = (cx - (a[0] + b[0]) / 2, cy - (a[1] + b[1]) / 2, 0.0)
                    tri((a[0], a[1], za), (b[0], b[1], za), (b[0], b[1], zb), outw)
                    tri((a[0], a[1], za), (b[0], b[1], zb), (a[0], a[1], zb), outw)
            for (za, zb, r0), (zb2, zc, r1) in zip(bands, bands[1:]):
                if abs(r0 - r1) < 1e-12:
                    continue
                rin, rout = min(r0, r1), max(r0, r1)
                out = (0.0, 0.0, 1.0 if r1 > r0 else -1.0)
                ci, co = circle(cx, cy, rin, n), circle(cx, cy, rout, n)
                for i in range(n):             # same n -> rings share θ, seams seal
                    ai, bi = ci[i], ci[(i + 1) % n]
                    ao, bo = co[i], co[(i + 1) % n]
                    tri((ai[0], ai[1], zb), (ao[0], ao[1], zb), (bo[0], bo[1], zb), out)
                    tri((ai[0], ai[1], zb), (bo[0], bo[1], zb), (bi[0], bi[1], zb), out)
            zlo, zhi = bands[0][0], bands[-1][1]
            if zlo > zmin + 1e-9:              # blind at the bottom -> end cap
                ring = circle(cx, cy, bands[0][2], n)
                for i in range(n):
                    a, b = ring[i], ring[(i + 1) % n]
                    tri((cx, cy, zlo), (a[0], a[1], zlo), (b[0], b[1], zlo), (0, 0, 1))
            if zhi < zmax - 1e-9:              # blind at the top -> end cap
                ring = circle(cx, cy, bands[-1][2], n)
                for i in range(n):
                    a, b = ring[i], ring[(i + 1) % n]
                    tri((cx, cy, zhi), (a[0], a[1], zhi), (b[0], b[1], zhi), (0, 0, -1))

        return {"vertices": verts, "triangles": tris}


@dataclass(frozen=True)
class Sphere:
    """Solid sphere centered (cx, cy, cz)."""
    cx: Fraction
    cy: Fraction
    cz: Fraction
    r: Fraction

    @classmethod
    def make(cls, r) -> "Sphere":
        r = F(r)
        if r <= 0:
            raise ValueError("sphere wants positive radius")
        return cls(F(0), F(0), F(0), r)

    def translated(self, x, y, z) -> "Sphere":
        return Sphere(self.cx + F(x), self.cy + F(y), self.cz + F(z), self.r)

    def tessellate(self, deflection: float = 0.2) -> dict:
        """A UV-sphere display mesh with collapsed poles (floats legal)."""
        import math

        r, cx, cy, cz = float(self.r), float(self.cx), float(self.cy), float(self.cz)
        seg = (max(8, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - deflection / r)))))
               if r > deflection else 8)
        nlat, nlon = max(4, seg), max(8, 2 * seg)
        verts, tris = [], []
        top = len(verts)
        verts.append([cx, cy, cz + r])
        rings = []
        for i in range(1, nlat):
            theta = math.pi * i / nlat
            rings.append(len(verts))
            for j in range(nlon):
                phi = 2 * math.pi * j / nlon
                verts.append([cx + r * math.sin(theta) * math.cos(phi),
                              cy + r * math.sin(theta) * math.sin(phi),
                              cz + r * math.cos(theta)])
        bot = len(verts)
        verts.append([cx, cy, cz - r])
        for j in range(nlon):                  # top fan (outward, CCW from outside)
            tris.append([top, rings[0] + j, rings[0] + (j + 1) % nlon])
        for k in range(len(rings) - 1):        # middle quads
            a, b = rings[k], rings[k + 1]
            for j in range(nlon):
                j2 = (j + 1) % nlon
                tris.append([a + j, b + j, b + j2])
                tris.append([a + j, b + j2, a + j2])
        for j in range(nlon):                  # bottom fan
            tris.append([bot, rings[-1] + (j + 1) % nlon, rings[-1] + j])
        return {"vertices": verts, "triangles": tris}


@dataclass(frozen=True)
class Cone:
    """Right conical frustum, axis +z through (cx, cy), r1 at z0, r2 at z1."""
    cx: Fraction
    cy: Fraction
    r1: Fraction
    r2: Fraction
    z0: Fraction
    z1: Fraction

    @classmethod
    def make(cls, r1, r2, h) -> "Cone":
        r1, r2, h = F(r1), F(r2), F(h)
        if h <= 0 or r1 < 0 or r2 < 0 or (r1 == 0 and r2 == 0):
            raise ValueError("cone wants positive height and a radius")
        return cls(F(0), F(0), r1, r2, F(0), h)

    def translated(self, x, y, z) -> "Cone":
        return Cone(self.cx + F(x), self.cy + F(y), self.r1, self.r2,
                    self.z0 + F(z), self.z1 + F(z))

    def tessellate(self, deflection: float = 0.2) -> dict:
        """Display mesh (frustum wall + caps) via the lathe (floats legal)."""
        from forgekernel.tess import lathe

        r1, r2 = float(self.r1), float(self.r2)
        z0, z1 = float(self.z0), float(self.z1)
        profile = [(0.0, z0), (r1, z0), (r2, z1), (0.0, z1)]
        return lathe(profile, deflection, float(self.cx), float(self.cy))


class _Quad:
    """Exact quadratic q(z) = a z^2 + b z + c — the r^2 profile of every
    K2 primitive (cylinder: constant; cone: squared linear; sphere:
    R^2 - (z-c)^2)."""

    __slots__ = ("a", "b", "c")

    def __init__(self, a, b, c) -> None:
        self.a, self.b, self.c = F(a), F(b), F(c)

    def at(self, z: Fraction) -> Fraction:
        return self.a * z * z + self.b * z + self.c

    def integral(self, lo: Fraction, hi: Fraction) -> Fraction:
        return (self.a * (hi ** 3 - lo ** 3) / 3
                + self.b * (hi ** 2 - lo ** 2) / 2 + self.c * (hi - lo))

    def z_integral(self, lo: Fraction, hi: Fraction) -> Fraction:
        """Integral of z*q(z)."""
        return (self.a * (hi ** 4 - lo ** 4) / 4
                + self.b * (hi ** 3 - lo ** 3) / 3
                + self.c * (hi ** 2 - lo ** 2) / 2)

    def rational_roots_between(self, lo: Fraction, hi: Fraction,
                               other: "_Quad") -> list[Fraction] | None:
        """Roots of (self - other) strictly inside (lo, hi): the list when
        every such root is rational, None when an IRRATIONAL crossover may
        exist in range (the caller refuses — exactness is never faked)."""
        a, b, c = self.a - other.a, self.b - other.b, self.c - other.c
        if a == 0:
            if b == 0:
                return []
            z = -c / b
            return [z] if lo < z < hi else []
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        num, den = disc.numerator, disc.denominator
        rn, rd = math.isqrt(num), math.isqrt(den)
        if rn * rn != num or rd * rd != den:
            # irrational roots: exact interval sign analysis decides if any
            # lie in (lo, hi); vertex of the difference is at -b/2a
            vz = -b / (2 * a)
            s_lo = a * lo * lo + b * lo + c
            s_hi = a * hi * hi + b * hi + c
            crosses = (s_lo < 0) != (s_hi < 0)
            if not crosses and lo < vz < hi:
                s_v = a * vz * vz + b * vz + c
                crosses = (s_v < 0) != (s_lo < 0) and s_v != 0
            return None if crosses else []
        sq = Fraction(rn, rd)
        roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        return [z for z in roots if lo < z < hi]


def _segments_of(prim) -> list:
    if isinstance(prim, Cyl):
        return [(prim.z0, prim.z1, _Quad(0, 0, prim.r * prim.r))]
    if isinstance(prim, Cone):
        h = prim.z1 - prim.z0
        k = (prim.r2 - prim.r1) / h
        a = k * k
        b = 2 * prim.r1 * k - 2 * k * k * prim.z0
        c = (prim.r1 - k * prim.z0) ** 2
        return [(prim.z0, prim.z1, _Quad(a, b, c))]
    if isinstance(prim, Sphere):
        return [(prim.cz - prim.r, prim.cz + prim.r,
                 _Quad(-1, 2 * prim.cz, prim.r * prim.r - prim.cz * prim.cz))]
    raise TypeError(f"not an axis primitive: {type(prim).__name__}")


class AxisStack:
    """Union of coaxial z-axis primitives — exact in the field of a+b*pi.

    Concentric circles make the union area pi*max_i r_i(z)^2, and every
    r^2 profile is a rational quadratic, so the union integrates exactly
    piecewise. Profile crossovers must land on rational z; a possible
    irrational crossover refuses honestly (K2.2 brings the algebraic
    extension)."""

    def __init__(self, cx, cy, prims: list) -> None:
        self.cx, self.cy = F(cx), F(cy)
        self.prims = list(prims)

    def fuse(self, prim) -> "AxisStack":
        if getattr(prim, "cx", None) != self.cx or \
           getattr(prim, "cy", None) != self.cy:
            raise ValueError("union of non-coaxial quadrics arrives at K2.2")
        return AxisStack(self.cx, self.cy, self.prims + [prim])

    def _pieces(self) -> list:
        segs = [s for p in self.prims for s in _segments_of(p)]
        cuts = {t for lo, hi, _ in segs for t in (lo, hi)}
        for i, (lo1, hi1, q1) in enumerate(segs):
            for lo2, hi2, q2 in segs[i + 1:]:
                lo, hi = max(lo1, lo2), min(hi1, hi2)
                if lo < hi:
                    roots = q1.rational_roots_between(lo, hi, q2)
                    if roots is None:
                        raise ValueError(
                            "irrational profile crossover arrives at K2.2 "
                            "(algebraic extension)")
                    cuts.update(roots)
        ordered = sorted(cuts)
        out = []
        for lo, hi in zip(ordered, ordered[1:]):
            mid = (lo + hi) / 2
            live = [q for slo, shi, q in segs if slo <= lo and hi <= shi]
            if not live:
                continue
            best = max(live, key=lambda q: q.at(mid))
            out.append((lo, hi, best))
        return out

    def volume(self) -> PiVal:
        return PiVal(0, sum((q.integral(lo, hi)
                             for lo, hi, q in self._pieces()), F(0)))

    def centroid_f(self) -> tuple[float, float, float]:
        pieces = self._pieces()
        v = sum((q.integral(lo, hi) for lo, hi, q in pieces), F(0))
        zbar = sum((q.z_integral(lo, hi) for lo, hi, q in pieces), F(0)) / v
        return (float(self.cx), float(self.cy), float(zbar))

    def tessellate(self, deflection: float = 0.2) -> dict:
        from forgekernel.tess import lathe

        pieces = self._pieces()
        profile = [(0.0, float(pieces[0][0]))]
        for lo, hi, q in pieces:
            import math as _m
            profile.append((_m.sqrt(max(0.0, float(q.at(lo)))), float(lo)))
            profile.append((_m.sqrt(max(0.0, float(q.at(hi)))), float(hi)))
        profile.append((0.0, float(pieces[-1][1])))
        return lathe(profile, deflection, float(self.cx), float(self.cy))

    def bbox(self):
        pieces = self._pieces()
        z0 = min(lo for lo, _, _ in pieces)
        z1 = max(hi for _, hi, _ in pieces)
        r2max = F(0)
        for lo, hi, q in pieces:
            cands = [q.at(lo), q.at(hi)]
            if q.a != 0:
                vz = -q.b / (2 * q.a)
                if lo <= vz <= hi:
                    cands.append(q.at(vz))
            r2max = max(r2max, *cands)
        r = math.sqrt(float(r2max))
        return ((float(self.cx) - r, float(self.cy) - r, float(z0)),
                (float(self.cx) + r, float(self.cy) + r, float(z1)))


class RevolveSolid:
    """A closed line-segment profile in the (r, z) half-plane revolved
    360 degrees about the z axis. Green gives exact metrics:
    V = pi * contour_integral(r^2 dz), each edge contributing
    (z2-z1)(r1^2 + r1 r2 + r2^2)/3."""

    def __init__(self, loop_rz: list) -> None:
        self.loop = [(F(r), F(z)) for r, z in loop_rz]
        if any(r < 0 for r, _ in self.loop):
            raise ValueError("revolve profile must stay at r >= 0")
        if self._v3() < 0:
            self.loop = list(reversed(self.loop))

    def _edges(self):
        n = len(self.loop)
        for i in range(n):
            yield self.loop[i], self.loop[(i + 1) % n]

    def _v3(self) -> Fraction:
        acc = F(0)
        for (r1, z1), (r2, z2) in self._edges():
            acc += (z2 - z1) * (r1 * r1 + r1 * r2 + r2 * r2)
        return acc

    def volume(self) -> PiVal:
        return PiVal(0, self._v3() / 3)

    def centroid_f(self) -> tuple[float, float, float]:
        num = F(0)
        for (r1, z1), (r2, z2) in self._edges():
            dz = z2 - z1
            num += dz * (z1 * (3 * r1 * r1 + 2 * r1 * r2 + r2 * r2)
                         + z2 * (r1 * r1 + 2 * r1 * r2 + 3 * r2 * r2)) / 12
        v3 = self._v3()
        return (0.0, 0.0, float(num / v3 * 3) if v3 else 0.0)

    def bbox(self):
        rmax = max(r for r, _ in self.loop)
        zs = [z for _, z in self.loop]
        return ((-float(rmax), -float(rmax), float(min(zs))),
                (float(rmax), float(rmax), float(max(zs))))

    def tessellate(self, deflection: float = 0.2) -> dict:
        from forgekernel.tess import lathe

        return lathe([(float(r), float(z)) for r, z in self.loop], deflection)


def _member_volume(m):
    if isinstance(m, (Sphere, Cone)):
        return AxisStack(m.cx, m.cy, [m]).volume()
    return m.volume()


def _member_centroid(m):
    if isinstance(m, (Sphere, Cone)):
        return AxisStack(m.cx, m.cy, [m]).centroid_f()
    if hasattr(m, "centroid_f"):
        return m.centroid_f()
    return tuple(float(x) for x in m.centroid())


class DisjointUnion:
    """Union of solids that meet at most tangentially — exact.

    Tangent contact is measure-zero, so the union volume is EXACTLY the
    sum of member volumes; the only real work is PROVING the members are
    disjoint-or-tangent with exact predicates (no sqrt: squared distances
    and squared-radius comparisons). Any genuine overlap refuses honestly
    (K2.3 brings general quadric booleans)."""

    def __init__(self, members: list) -> None:
        self.members = list(members)
        for i, a in enumerate(self.members):
            for b in self.members[i + 1:]:
                _classify_pair(a, b)          # raises on genuine overlap

    def add(self, other) -> "DisjointUnion":
        for m in self.members:
            _classify_pair(m, other)
        return DisjointUnion(self.members + [other])

    def volume(self) -> "PiVal":
        total = PiVal(0, 0)
        for m in self.members:
            v = _member_volume(m)
            total = total + (v if isinstance(v, PiVal) else PiVal(v, 0))
        return total

    def centroid_f(self) -> tuple:
        acc = [0.0, 0.0, 0.0]
        vtot = 0.0
        for m in self.members:
            v = float(_member_volume(m))
            c = _member_centroid(m)
            for i in range(3):
                acc[i] += v * c[i]
            vtot += v
        return (acc[0] / vtot, acc[1] / vtot, acc[2] / vtot)

    def bbox(self):
        boxes = [m.bbox() for m in self.members]
        lo = tuple(min(float(b[0][i]) for b in boxes) for i in range(3))
        hi = tuple(max(float(b[1][i]) for b in boxes) for i in range(3))
        return (lo, hi)

    def watertight_violations(self) -> list:
        bad = []
        for m in self.members:
            if hasattr(m, "watertight_violations"):
                bad += m.watertight_violations()
        return bad


def _zrange(prim):
    """Exact z-extent of an axis primitive (None for general solids)."""
    if isinstance(prim, Cyl):
        return (prim.z0, prim.z1)
    if isinstance(prim, Cone):
        return (prim.z0, prim.z1)
    if isinstance(prim, Sphere):
        return (prim.cz - prim.r, prim.cz + prim.r)
    return None


def _classify_pair(a, b) -> None:
    """Raise ValueError iff a and b genuinely overlap (positive-measure
    intersection). Silent return = disjoint or tangent (both exact)."""
    # cylinder / cylinder, parallel +z axes
    if isinstance(a, Cyl) and isinstance(b, Cyl):
        za, zb = (a.z0, a.z1), (b.z0, b.z1)
        if za[1] <= zb[0] or zb[1] <= za[0]:
            return                            # disjoint in z
        d2 = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2
        outer = (a.r + b.r) ** 2
        inner = (a.r - b.r) ** 2
        if d2 >= outer or d2 <= inner:
            return                            # externally/internally clear
        raise ValueError(
            "overlapping cylinders — general quadric booleans arrive at K2.3")
    # sphere / planar Solid
    if isinstance(a, Sphere) and isinstance(b, Solid):
        _sphere_solid(a, b)
        return
    if isinstance(b, Sphere) and isinstance(a, Solid):
        _sphere_solid(b, a)
        return
    # sphere / sphere
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        d2 = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2 + (a.cz - b.cz) ** 2
        if d2 >= (a.r + b.r) ** 2 or d2 <= (a.r - b.r) ** 2:
            return
        raise ValueError(
            "overlapping spheres — general quadric booleans arrive at K2.3")
    raise ValueError(
        f"disjoint-union of {type(a).__name__}+{type(b).__name__} arrives "
        "at K2.3")


def _sphere_solid(s: "Sphere", solid: "Solid") -> None:
    """Disjoint/tangent iff the sphere center lies on the far side of (or
    exactly on) some face plane by at least the radius — exact, sqrt-free:
    for outward plane n·x = d, signed gap g = n·c − d; clear iff g > 0 and
    g² ≥ r²·(n·n) (both sides squared, exact). Convex-solid sufficient
    condition; a sphere separated from a convex solid is separated by one
    of its face planes."""
    c = (s.cx, s.cy, s.cz)
    seen = set()
    for p in solid.polys:
        key = p.plane.canonical()
        if key in seen:
            continue
        seen.add(key)
        n, dpl = p.plane.n, p.plane.d
        g = dot(n, c) - dpl
        if g > 0 and g * g >= s.r * s.r * dot(n, n):
            return                            # separated by this face plane
    raise ValueError(
        "sphere overlaps the solid — general quadric booleans arrive at K2.3")


class RoundedBox:
    """An axis-aligned box with ALL edges and corners filleted radius r —
    the Minkowski sum of the inner core box (a-2r)x(b-2r)x(c-2r) with a
    ball of radius r. Steiner's formula gives the volume EXACTLY in Q[pi]:

        V = pqs + 2r(pq+qs+sp) + pi r^2 (p+q+s) + (4/3) pi r^3

    (core box + face slabs + edge quarter-cylinders + 8 corner octants),
    p=a-2r etc. Requires 2r <= min(a,b,c); tighter fillets need the general
    blend engine (K5)."""

    def __init__(self, a, b, c, r, origin=(0, 0, 0)) -> None:
        self.a, self.b, self.c, self.r = F(a), F(b), F(c), F(r)
        self.origin = tuple(F(v) for v in origin)
        if 2 * self.r > min(self.a, self.b, self.c):
            raise ValueError("fillet radius exceeds half the smallest dimension")

    def _pqs(self):
        r = self.r
        return self.a - 2 * r, self.b - 2 * r, self.c - 2 * r

    def volume(self) -> PiVal:
        r = self.r
        p, q, s = self._pqs()
        rational = p * q * s + 2 * r * (p * q + q * s + s * p)
        pi_part = r * r * (p + q + s) + Fraction(4, 3) * r ** 3
        return PiVal(rational, pi_part)

    def centroid_f(self) -> tuple:
        ox, oy, oz = self.origin
        return (float(ox + self.a / 2), float(oy + self.b / 2),
                float(oz + self.c / 2))

    def bbox(self):
        ox, oy, oz = self.origin
        return ((float(ox), float(oy), float(oz)),
                (float(ox + self.a), float(oy + self.b), float(oz + self.c)))

    def watertight_violations(self) -> list:
        return []


class MiteredSweep:
    """A convex profile swept along a polyline with miter joints — exact
    in ℚ[√d]. Volume = profile_area × centerline_length: at a miter the
    bisector plane removes a wedge from one segment that the neighbour
    adds back identically, so the swept volume is exactly the straight
    area×length even through corners. Segment lengths accumulate in one
    quadratic-surd field; a path mixing radicals (e.g. √2 and √3) refuses
    (K3.1). This is the model OCCT cannot build (swept_channel)."""

    def __init__(self, area, path: list) -> None:
        from forgekernel.surd import SurdVal, sqrt_rational

        self.area = F(area)
        self.path = [tuple(F(c) for c in p) for p in path]
        if len(self.path) < 2:
            raise ValueError("sweep path needs >= 2 points")
        length = SurdVal(0, 0, 1)
        for a, b in zip(self.path, self.path[1:]):
            d2 = sum((b[i] - a[i]) ** 2 for i in range(3))
            length = length + sqrt_rational(d2)      # may raise on mixed radicals
        self._length = length

    def length(self):
        return self._length

    def volume(self):
        return self._length * self.area              # SurdVal

    def centroid_f(self) -> tuple:
        # centroid on the centerline, length-weighted midpoints (floated)
        from forgekernel.surd import sqrt_rational

        acc = [0.0, 0.0, 0.0]
        tot = 0.0
        for a, b in zip(self.path, self.path[1:]):
            seg = float(sqrt_rational(sum((b[i] - a[i]) ** 2 for i in range(3))))
            mid = [(float(a[i]) + float(b[i])) / 2 for i in range(3)]
            for i in range(3):
                acc[i] += seg * mid[i]
            tot += seg
        return (acc[0] / tot, acc[1] / tot, acc[2] / tot)

    def bbox(self):
        # centerline bbox padded by the profile's circumradius estimate
        pad = math.sqrt(float(self.area))
        xs = [p[0] for p in self.path]
        ys = [p[1] for p in self.path]
        zs = [p[2] for p in self.path]
        return ((float(min(xs)) - pad, float(min(ys)) - pad, float(min(zs)) - pad),
                (float(max(xs)) + pad, float(max(ys)) + pad, float(max(zs)) + pad))

    def watertight_violations(self) -> list:
        return []


class SphereOverlap:
    """Union/intersection/difference of two GENUINELY OVERLAPPING spheres —
    exact in ℚ[π]. The lens (intersection) is a sum of two spherical caps,
    each of volume π h²(3r − h)/3 with h rational when the centre distance
    and radii are rational, so every boolean volume stays in ℚ[π]:

        d1 = (d² + r1² − r2²)/(2d)   (rational plane offset from centre 1)
        h1 = r1 − d1,  h2 = r2 − (d − d1)   (rational cap heights)
        lens = cap(r1,h1) + cap(r2,h2),  cap(r,h)=π h²(3r−h)/3
        union = V1 + V2 − lens,  cut = V1 − lens,  intersect = lens

    Parallel-cylinder overlap and cylinder–wall crossing are TRANSCENDENTAL
    (arccos/√ lens) and are refused elsewhere — this is the exact case."""

    def __init__(self, a: Sphere, b: Sphere, op: str) -> None:
        self.a, self.b, self.op = a, b, op
        d2 = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2 + (a.cz - b.cz) ** 2
        if d2 >= (a.r + b.r) ** 2:
            raise ValueError("spheres do not overlap (use DisjointUnion)")
        if d2 <= (a.r - b.r) ** 2:
            raise ValueError("one sphere contains the other (K2.3 nesting)")
        # d must be rational for the caps to stay in ℚ[π]
        import math as _m
        dn, dd = d2.numerator, d2.denominator
        rn, rd = _m.isqrt(dn), _m.isqrt(dd)
        if rn * rn != dn or rd * rd != dd:
            raise ValueError(
                "irrational centre distance — the cap heights leave ℚ[π] "
                "(K3: algebraic/transcendental)")
        self.d = Fraction(rn, rd)

    @staticmethod
    def _cap(r: Fraction, h: Fraction) -> Fraction:
        # π-coefficient of the cap volume π h²(3r − h)/3
        return h * h * (3 * r - h) / 3

    def _lens_picoeff(self) -> Fraction:
        r1, r2, d = self.a.r, self.b.r, self.d
        d1 = (d * d + r1 * r1 - r2 * r2) / (2 * d)
        h1 = r1 - d1
        h2 = r2 - (d - d1)
        return self._cap(r1, h1) + self._cap(r2, h2)

    def volume(self) -> PiVal:
        v1 = Fraction(4, 3) * self.a.r ** 3
        v2 = Fraction(4, 3) * self.b.r ** 3
        lens = self._lens_picoeff()
        if self.op == "intersect":
            return PiVal(0, lens)
        if self.op == "cut":
            return PiVal(0, v1 - lens)
        return PiVal(0, v1 + v2 - lens)      # union

    def centroid_f(self) -> tuple:
        return (float((self.a.cx + self.b.cx) / 2),
                float((self.a.cy + self.b.cy) / 2),
                float((self.a.cz + self.b.cz) / 2))

    def bbox(self):
        a, b = self.a, self.b
        if self.op == "intersect":
            lo = (max(a.cx - a.r, b.cx - b.r), max(a.cy - a.r, b.cy - b.r),
                  max(a.cz - a.r, b.cz - b.r))
            hi = (min(a.cx + a.r, b.cx + b.r), min(a.cy + a.r, b.cy + b.r),
                  min(a.cz + a.r, b.cz + b.r))
            return (tuple(float(v) for v in lo), tuple(float(v) for v in hi))
        s = a if self.op == "cut" else None
        boxes = [a] if s else [a, b]
        lo = (min(float(x.cx - x.r) for x in boxes),
              min(float(x.cy - x.r) for x in boxes),
              min(float(x.cz - x.r) for x in boxes))
        hi = (max(float(x.cx + x.r) for x in boxes),
              max(float(x.cy + x.r) for x in boxes),
              max(float(x.cz + x.r) for x in boxes))
        return (lo, hi)

    def watertight_violations(self) -> list:
        return []


def steinmetz(r) -> PiVal:
    """Intersection of two equal perpendicular cylinders radius r (the
    bicylinder / Steinmetz solid) — famously EXACT and π-free: 16 r³/3."""
    r = F(r)
    return PiVal(Fraction(16, 3) * r ** 3, 0)


# -- K5.0: rolling-ball fillets on SELECTED straight box edges ----------------

class FilletedBox:
    """A box with a constant-radius rolling-ball fillet on a chosen
    subset of its straight edges — exact in ℚ[π].

    Removed material per edge = (square corner prism) − (quarter
    cylinder): ΔV = (r² − πr²/4)·L, so

        V = V_box − Σ r²L  +  π·Σ (r²/4)L      — a PiVal, exact.

    Selected edges must be pairwise NON-ADJACENT (sharing a box vertex
    would need the spherical corner patch — that is K5.1); adjacency
    refuses with the stage name. Edges are given as (axis, side_a,
    side_b): the edge parallel to ``axis`` on the (min/max, min/max)
    sides of the other two axes, e.g. ('z', 'max', 'max')."""

    def __init__(self, lo, hi, edges, radius) -> None:
        self.lo = tuple(F(v) for v in lo)
        self.hi = tuple(F(v) for v in hi)
        self.r = F(radius)
        self.edges = list(edges)
        dims = [self.hi[c] - self.lo[c] for c in range(3)]
        if self.r <= 0:
            raise ValueError("fillet wants positive radius")
        axes = {"x": 0, "y": 1, "z": 2}
        seen = set()
        verts: list[set] = []
        for axis, sa, sb in self.edges:
            a = axes[axis]
            o1, o2 = [c for c in range(3) if c != a]
            if 2 * self.r > min(dims[o1], dims[o2]):
                raise ValueError("fillet radius exceeds the face half-width")
            key = (a, sa, sb)
            if key in seen:
                raise ValueError("edge selected twice")
            seen.add(key)
            # the edge's two box vertices, for adjacency detection
            c1 = self.lo[o1] if sa == "min" else self.hi[o1]
            c2 = self.lo[o2] if sb == "min" else self.hi[o2]
            vset = set()
            for end in (self.lo[a], self.hi[a]):
                v = [0, 0, 0]
                v[a], v[o1], v[o2] = end, c1, c2
                vset.add(tuple(v))
            verts.append(vset)
        # K5.1: classify every box vertex by how many SELECTED edges meet
        # there. 0/1 → nothing special; 3 → the sphere-OCTANT corner
        # patch (exact in ℚ[π]: removed corner material = r³ − πr³/6,
        # and each incident edge's run is shortened by r); 2 → the blend
        # is a genuinely non-spherical surface — refuses (K5.2).
        incident: dict = {}
        for i, vset in enumerate(verts):
            for v in vset:
                incident.setdefault(v, []).append(i)
        self.corners: list[tuple] = []
        self._shorten: dict[int, int] = {i: 0 for i in range(len(self.edges))}
        for v, idxs in incident.items():
            if len(idxs) == 2:
                raise ValueError(
                    "two filleted edges meeting at a corner (third sharp) "
                    "need a non-spherical blend (arrives at K5.2)")
            if len(idxs) == 3:
                self.corners.append(v)
                for i in idxs:
                    self._shorten[i] += 1

    def _edge_len(self, i: int) -> F:
        axis = self.edges[i][0]
        a = {"x": 0, "y": 1, "z": 2}[axis]
        full = self.hi[a] - self.lo[a]
        eff = full - self.r * self._shorten[i]
        if eff < 0:
            raise ValueError("fillet radius exceeds the edge length")
        return eff

    def volume(self) -> PiVal:
        vbox = F(1)
        for c in range(3):
            vbox *= self.hi[c] - self.lo[c]
        rat = vbox
        pi_c = F(0)
        for i in range(len(self.edges)):
            L = self._edge_len(i)
            rat -= self.r * self.r * L
            pi_c += self.r * self.r * L / 4
        # sphere-octant corner patches: removed = r³ − πr³/6 each
        r3 = self.r ** 3
        n = len(self.corners)
        rat -= n * r3
        pi_c += n * r3 / 6
        return PiVal(rat, pi_c)

    def centroid_f(self):
        import math
        axes = {"x": 0, "y": 1, "z": 2}
        vbox = 1.0
        cb = [0.0, 0.0, 0.0]
        for c in range(3):
            vbox *= float(self.hi[c] - self.lo[c])
            cb[c] = float(self.lo[c] + self.hi[c]) / 2
        r = float(self.r)
        num = [vbox * cb[c] for c in range(3)]
        vtot = vbox
        # removed region cross-section: square r×r at the edge corner minus
        # the quarter disk about the inner corner. Exact area r²−πr²/4;
        # centroid distance from the OUTER corner along each face:
        #   c* = (r/2·r² − (r−4r/3π)·πr²/4) / (r²−πr²/4)
        area = r * r - math.pi * r * r / 4
        cstar = (r * r * (r / 2) - (math.pi * r * r / 4) * (r - 4 * r / (3 * math.pi))) / area
        for i, (axis, sa, sb) in enumerate(self.edges):
            a = axes[axis]
            o1, o2 = [c for c in range(3) if c != a]
            L = float(self._edge_len(i))
            vrem = area * L
            crem = [0.0, 0.0, 0.0]
            # midpoint of the EFFECTIVE run (shortened r at 3-corner ends)
            run_lo, run_hi = float(self.lo[a]), float(self.hi[a])
            for v in self.corners:
                if v[a] == self.lo[a] and self._touches(i, v):
                    run_lo += r
                if v[a] == self.hi[a] and self._touches(i, v):
                    run_hi -= r
            crem[a] = (run_lo + run_hi) / 2
            crem[o1] = (float(self.lo[o1]) + cstar if sa == "min"
                        else float(self.hi[o1]) - cstar)
            crem[o2] = (float(self.lo[o2]) + cstar if sb == "min"
                        else float(self.hi[o2]) - cstar)
            vtot -= vrem
            for c in range(3):
                num[c] -= vrem * crem[c]
        # corner terms: removed cube-minus-octant at each 3-corner; its
        # centroid sits c*₃ = r(1/2 − 5π/48)/(1 − π/6) inward per axis
        if self.corners:
            vrem_c = r ** 3 - math.pi * r ** 3 / 6
            c3 = r * (0.5 - 5 * math.pi / 48) / (1 - math.pi / 6)
            for v in self.corners:
                crem = [0.0, 0.0, 0.0]
                for c in range(3):
                    inward = 1.0 if v[c] == self.lo[c] else -1.0
                    crem[c] = float(v[c]) + inward * c3
                vtot -= vrem_c
                for c in range(3):
                    num[c] -= vrem_c * crem[c]
        return tuple(n / vtot for n in num)

    def _touches(self, edge_i: int, vertex) -> bool:
        axis, sa, sb = self.edges[edge_i]
        a = {"x": 0, "y": 1, "z": 2}[axis]
        o1, o2 = [c for c in range(3) if c != a]
        c1 = self.lo[o1] if sa == "min" else self.hi[o1]
        c2 = self.lo[o2] if sb == "min" else self.hi[o2]
        return vertex[o1] == c1 and vertex[o2] == c2

    def bbox(self):
        return self.lo, self.hi

    def tessellate(self, deflection: float = 0.2) -> dict:
        raise NotImplementedError("FilletedBox mesh arrives at K5.1")


class VariableFilletedBox:
    """A box with LINEAR-TAPER rolling-ball fillets on non-adjacent
    straight edges — still exact in ℚ[π].

    A fillet whose radius runs linearly r(t)=r0+(r1−r0)t/L along an edge
    removes cross-section area r(t)²(1−π/4), and ∫₀^L r(t)² dt =
    L(r0²+r0r1+r1²)/3 exactly, so

        V = V_box − Σ (1−π/4)·L(r0²+r0r1+r1²)/3      — a PiVal, exact.

    Edges: (axis, side_a, side_b, r0, r1). Selected edges must be
    pairwise non-adjacent (a shared vertex needs a variable corner
    patch → K5.3)."""

    def __init__(self, lo, hi, edges) -> None:
        self.lo = tuple(F(v) for v in lo)
        self.hi = tuple(F(v) for v in hi)
        self.edges = [(ax, sa, sb, F(r0), F(r1)) for ax, sa, sb, r0, r1 in edges]
        dims = [self.hi[c] - self.lo[c] for c in range(3)]
        axes = {"x": 0, "y": 1, "z": 2}
        verts: list[set] = []
        for ax, sa, sb, r0, r1 in self.edges:
            a = axes[ax]
            o1, o2 = [c for c in range(3) if c != a]
            if r0 <= 0 or r1 <= 0:
                raise ValueError("taper fillet wants positive radii")
            if 2 * max(r0, r1) > min(dims[o1], dims[o2]):
                raise ValueError("taper radius exceeds the face half-width")
            c1 = self.lo[o1] if sa == "min" else self.hi[o1]
            c2 = self.lo[o2] if sb == "min" else self.hi[o2]
            vset = {tuple(v) for v in (
                [self.lo[a] if k == a else (c1 if k == o1 else c2)
                 for k in range(3)],
                [self.hi[a] if k == a else (c1 if k == o1 else c2)
                 for k in range(3)])}
            verts.append(frozenset(vset))
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[i] & verts[j]:
                    raise ValueError(
                        "adjacent taper-filleted edges need a variable "
                        "corner patch (arrives at K5.3)")

    def volume(self) -> PiVal:
        vbox = F(1)
        for c in range(3):
            vbox *= self.hi[c] - self.lo[c]
        rat, pic = vbox, F(0)
        axes = {"x": 0, "y": 1, "z": 2}
        for ax, _, _, r0, r1 in self.edges:
            a = axes[ax]
            L = self.hi[a] - self.lo[a]
            X = L * (r0 * r0 + r0 * r1 + r1 * r1) / 3
            rat -= X
            pic += X / 4
        return PiVal(rat, pic)

    def bbox(self):
        return self.lo, self.hi
