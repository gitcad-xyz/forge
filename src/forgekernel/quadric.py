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
    is_exact_scalar = True   # exact beyond Q; F() must not coerce it away
    """An exact number a + b·π (a, b rational)."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0) -> None:
        self.a, self.b = F(a), F(b)

    def _co(self, o):
        """Coerce, or DEFER. Constructing PiVal(anything) here meant that
        meeting a wider exact type — a ℚ[π] polynomial, which names the same
        numbers and more — raised TypeError instead of letting Python reflect
        to the other operand. volume() returns whichever type fits, so the two
        meet constantly."""
        if isinstance(o, PiVal):
            return o
        if isinstance(o, (int, Fraction)):
            return PiVal(o)
        return NotImplemented

    def __add__(self, o: "PiVal | int | Fraction") -> "PiVal":
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        return PiVal(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, o: "PiVal | int | Fraction") -> "PiVal":
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        return PiVal(self.a - o.a, self.b - o.b)

    def __eq__(self, o: object) -> bool:
        o = self._co(o)
        if o is NotImplemented:
            return NotImplemented
        return self.a == o.a and self.b == o.b

    # -- order, EXACTLY --------------------------------------------------------
    # PiVal could be added, subtracted and compared for equality but not
    # ORDERED, so the one question anyone actually asks of a volume — "is it
    # positive?" — could only be answered by float(v) > 0. That is a float
    # deciding whether a solid is valid, which ADR-0019 forbids, and it is
    # wrong at the boundary: a true volume a hair above zero rounds to 0.0 and
    # the solid reads as inside-out. PiPoly already decides this exactly, by
    # narrowing a rational enclosure of π until the sign is certain, so defer
    # to it rather than grow a second implementation.

    def sign(self) -> int:
        """-1, 0 or +1 — exact, no float anywhere."""
        from forgekernel.polypi import PiPoly

        return PiPoly.from_pival(self).sign()

    def _cmp(self, o):
        from forgekernel.polypi import PiPoly

        try:
            return (PiPoly.from_pival(self) - PiPoly(o)
                    if isinstance(o, (int, Fraction))
                    else PiPoly.from_pival(self) - PiPoly.from_pival(self._co(o))
                    ).sign()
        except (TypeError, ValueError):
            return None

    def __lt__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c < 0

    def __le__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c <= 0

    def __gt__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c > 0

    def __ge__(self, o):
        c = self._cmp(o)
        return NotImplemented if c is None else c >= 0

    def __float__(self) -> float:
        return float(self.a) + float(self.b) * math.pi

    def __repr__(self) -> str:
        return f"({self.a} + {self.b}·π)"


def _mirror_axis(axis: str) -> int:
    """The coordinate index an axis-plane reflection negates.

    Every family in this module is CLOSED under such a reflection — a mirrored
    cylinder is a cylinder, a mirrored frustum is a frustum with its radii
    swapped — so ``mirrored`` is exact and structural: negate a coordinate,
    swap an interval's ends. Nothing costs arithmetic.

    It matters because the feature paths (chamfer, fillet, shell) dispatch on
    representation. A reflection that dropped to the generic canonical ``Body``
    made every one of them refuse a shape they handle perfectly well upright,
    which is 30 of gitcad's 34 mirror-asymmetry pairs. For an isometry σ the
    contract is op(σS) ≡ σ op(S); a kernel that answers one and refuses the
    other is not wrong about geometry, it is wrong about itself.
    """
    i = {"x": 0, "y": 1, "z": 2}.get(axis)
    if i is None:
        raise ValueError(f"mirror axis must be x|y|z, got {axis!r}")
    return i


def _axis_rotation(axis, deg):
    """How a +z-axis family absorbs a rotation, or None if it does not.

    Returns ``(fn, flipped)`` where ``fn(cx, cy) -> (cx', cy')`` moves the axis
    POSITION and ``flipped`` says the +z axis became −z (so z-intervals and any
    end-labelled data travel with their ends). None means the rotation is a
    genuine TILT — the family cannot express the result and the caller should
    fall through to the canonical B-rep, which is an answer rather than a
    refusal.

    Absorbed:

    * any exact rotation about **z** — the axis direction is unchanged and only
      (cx, cy) turn. This is the common case in real modelling and it was going
      through the generic path, which cost the representation and with it every
      feature and boolean that dispatches on one;
    * a half turn about **x** or **y** — +z becomes −z, which Cyl and Cone can
      express by swapping and negating their ends, exactly as ``mirrored``
      does. (x, y) also reflect, one of them.

    A quarter turn about x or y lays the axis into the xy-plane and is NOT
    absorbed; nor is any angle outside the exact ℚ[√d] table.
    """
    from forgekernel.kernel import _cos_sin_deg

    if deg != int(deg):
        return None
    d = int(deg) % 360
    ax, ay, az = (F(axis[0]), F(axis[1]), F(axis[2]))
    if ax == 0 and ay == 0 and az != 0:
        cs = _cos_sin_deg(d if az > 0 else -d)
        if cs is None:
            return None                       # angle outside the exact table
        c, s = cs
        return (lambda x, y: (c * x - s * y, s * x + c * y)), False
    if d != 180:
        return None                           # only a half turn flips the axis
    if ay == 0 and az == 0 and ax != 0:       # 180° about x: y→−y, z→−z
        return (lambda x, y: (x, -y)), True
    if ax == 0 and az == 0 and ay != 0:       # 180° about y: x→−x, z→−z
        return (lambda x, y: (-x, y)), True
    return None


def _scale_factor(f):
    """A uniform scale ratio as an exact positive value, or raise.

    Every family in this module is closed under a SIMILARITY: scale a cylinder
    and it is a cylinder, radius and all. Non-uniform scale is not closed —
    squash a cylinder in x and it is an elliptic cylinder, which nothing here
    holds — so callers that support it hand this a single ratio and get None
    from their own `scaled` for the anisotropic case.
    """
    v = F(f)
    if v <= 0:
        raise ValueError(f"scale factor must be positive, got {f!r}")
    return v


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

    def mirrored(self, axis: str) -> "Cyl":
        i = _mirror_axis(axis)
        if i == 0:
            return Cyl(-self.cx, self.cy, self.r, self.z0, self.z1)
        if i == 1:
            return Cyl(self.cx, -self.cy, self.r, self.z0, self.z1)
        return Cyl(self.cx, self.cy, self.r, -self.z1, -self.z0)

    def rotated(self, axis, deg):
        r = _axis_rotation(axis, deg)
        if r is None:
            return None
        fn, flipped = r
        cx, cy = fn(self.cx, self.cy)
        z0, z1 = (-self.z1, -self.z0) if flipped else (self.z0, self.z1)
        return Cyl(cx, cy, self.r, z0, z1)

    def scaled(self, f) -> "Cyl":
        s = _scale_factor(f)
        return Cyl(self.cx * s, self.cy * s, self.r * s,
                   self.z0 * s, self.z1 * s)

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


def _xy_inside_footprint(solid: Solid, px, py) -> bool:
    """Exact: is (px, py) inside the solid's xy footprint?

    Needed because "the bore does not CROSS a lateral wall" does not
    distinguish a bore strictly inside from one entirely outside: neither
    crosses anything. Ray parity along +x, all rational, no tolerance.

    The footprint is the union of the xy projections of the faces: if a face
    covers (px, py) then the solid occupies that column at some z, and if the
    solid occupies the column then a vertical ray meets a face there. Parity
    over the LATERAL edges instead is only equivalent for a PRISM — a chamfer
    or a draft makes the slanted faces project to bands whose extra crossings
    flip the parity back, and a bore at the dead centre of a chamfered plate
    reads as missing the solid entirely.
    """
    for p in solid.polys:
        verts = [(v[0], v[1]) for v in p.verts]
        inside = False
        m = len(verts)
        for i in range(m):
            (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % m]
            if (y1 > py) != (y2 > py):
                xc = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if px < xc:
                    inside = not inside
        if inside:
            return True
    return False


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
        # The wall check above proves the bore straddles no wall; it does NOT
        # say which side it is on. Without this, a bore placed entirely outside
        # the footprint is recorded as a phantom hole and its volume silently
        # subtracted from a solid it never touches.
        if not _xy_inside_footprint(self.base, c.cx, c.cy):
            raise ValueError("bore misses the solid in xy (nothing to drill)")
        # The column may be a STACK of slabs, not just one: a bore through a
        # shelled plate meets the outer top and bottom plus the void's ceiling
        # and floor. That used to refuse ("meets 4 horizontal levels, not 2"),
        # because a single full-barrel removal would have taken the cavity's
        # air as if it were material. One bore PER SLAB is the honest answer —
        # coaxial bores are already what a counterbore is, and their volumes
        # union by z-interval — and it is exact: the cross-section is constant
        # over the disc, so each slab really is a full barrel.
        pieces = [Cyl(c.cx, c.cy, c.r, max(z0, lo), min(z1, hi))
                  for lo, hi in _column_slabs(self.base, c)
                  if max(z0, lo) < min(z1, hi)]
        if not pieces:
            raise ValueError("bore misses the solid in z (K2.1 for the rest)")
        for o in self.bores:
            if o.cx == c.cx and o.cy == c.cy:
                continue                       # coaxial stack (counterbore)
            dx, dy = o.cx - c.cx, o.cy - c.cy
            if dx * dx + dy * dy <= (o.r + c.r) ** 2:
                raise ValueError(
                    "bores intersect — general quadric booleans arrive "
                    "at K2.1")
        return DrilledSolid(self.base, self.bores + pieces)

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
        of ℚ[π] numbers — outside the field, so floats are honest here).

        Removes the SAME z-band decomposition ``_bore_union_volume`` uses.
        Looping over raw ``self.bores`` double-subtracted wherever coaxial
        bores overlap — a counterbore's pilot hole lies entirely inside the
        wider bore's z-range, so their shared length came off twice. Volume
        was already right, which made the reported mass-properties dict
        internally inconsistent: the wrong centre of mass for an exact mass.
        """
        bv = float(self.base.volume())
        c = self.base.centroid()
        acc = [bv * float(c[0]), bv * float(c[1]), bv * float(c[2])]
        vol = bv
        groups: dict = {}
        for cyl in self.bores:
            groups.setdefault((cyl.cx, cyl.cy), []).append(cyl)
        for (cx, cy), cyls in groups.items():
            cuts = sorted({t for b in cyls for t in (b.z0, b.z1)})
            for lo, hi in zip(cuts, cuts[1:]):
                rs = [b.r for b in cyls if b.z0 <= lo and hi <= b.z1]
                if not rs:
                    continue
                rmax = max(rs)
                v = math.pi * float(rmax) ** 2 * float(hi - lo)
                mid = (float(cx), float(cy), float(lo + hi) / 2)
                for i in range(3):
                    acc[i] -= v * mid[i]
                vol -= v
        return (acc[0] / vol, acc[1] / vol, acc[2] / vol)

    def bbox(self):
        return self.base.bbox()

    def mirrored(self, axis: str) -> "DrilledSolid":
        return DrilledSolid(self.base.mirrored(axis),
                            [b.mirrored(axis) for b in self.bores])

    def rotated(self, axis, deg):
        from forgekernel.kernel import rotate as _rotate_solid

        bores = [b.rotated(axis, deg) for b in self.bores]
        if any(b is None for b in bores):
            return None                       # a bore would leave the family
        try:
            base = _rotate_solid(self.base, axis, deg)
        except ValueError:
            return None                       # angle outside the exact table
        return DrilledSolid(base, bores)

    def scaled(self, f) -> "DrilledSolid":
        s = _scale_factor(f)
        # Solid.scaled takes THREE factors — a uniform scale is the diagonal
        return DrilledSolid(self.base.scaled(s, s, s),
                            [b.scaled(s) for b in self.bores])

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

    def mirrored(self, axis: str) -> "Sphere":
        c = [self.cx, self.cy, self.cz]
        i = _mirror_axis(axis)
        c[i] = -c[i]
        return Sphere(c[0], c[1], c[2], self.r)

    def rotated(self, axis, deg):
        """A sphere absorbs EVERY rotation, being round: only the centre
        moves, and it moves by the same exact matrix everything else uses."""
        from forgekernel.kernel import _rotation_matrix

        try:
            m = _rotation_matrix(axis, deg)
        except ValueError:
            return None                       # angle outside the exact table
        from forgekernel.exact import as_fraction

        c = (self.cx, self.cy, self.cz)
        p = []
        for i in range(3):
            v = sum(m[i][j] * c[j] for j in range(3))
            # demote a coordinate whose VALUE is rational, exactly as
            # kernel.rotate does. Rodrigues builds s/|axis| as a SurdVal even
            # for a quarter turn, so a 90-degree-rotated sphere's centre came
            # back Q[sqrt 1]-typed and the first thing downstream to want an
            # integer — sqrt_rational, computing a rim radius — hit
            # AttributeError. Fourth appearance of this pattern today.
            f = as_fraction(v)
            p.append(v if f is None else f)
        return Sphere(p[0], p[1], p[2], self.r)

    def scaled(self, f) -> "Sphere":
        s = _scale_factor(f)
        return Sphere(self.cx * s, self.cy * s, self.cz * s, self.r * s)

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

    def mirrored(self, axis: str) -> "Cone":
        i = _mirror_axis(axis)
        if i == 0:
            return Cone(-self.cx, self.cy, self.r1, self.r2, self.z0, self.z1)
        if i == 1:
            return Cone(self.cx, -self.cy, self.r1, self.r2, self.z0, self.z1)
        # r1 lives at z0, so reflecting z has to carry it to the other end —
        # the one case where negating a coordinate is not the whole story
        return Cone(self.cx, self.cy, self.r2, self.r1, -self.z1, -self.z0)

    def rotated(self, axis, deg):
        r = _axis_rotation(axis, deg)
        if r is None:
            return None
        fn, flipped = r
        cx, cy = fn(self.cx, self.cy)
        if flipped:                           # the radii travel with their ends
            return Cone(cx, cy, self.r2, self.r1, -self.z1, -self.z0)
        return Cone(cx, cy, self.r1, self.r2, self.z0, self.z1)

    def scaled(self, f) -> "Cone":
        s = _scale_factor(f)
        return Cone(self.cx * s, self.cy * s, self.r1 * s, self.r2 * s,
                    self.z0 * s, self.z1 * s)

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

    # x*x, never x**2. The bounds are only Fractions when nothing upstream has
    # been rotated: a body turned about an oblique axis puts its z-extent in
    # ℚ[√d] or ℚ(√p,√q), and neither SurdVal nor BiSurd implements __pow__ —
    # so `hi ** 3` raised TypeError out of a VOLUME. Repeated multiplication is
    # the house convention here for exactly this reason and costs nothing.
    def integral(self, lo, hi):
        h2, l2 = hi * hi, lo * lo
        return (self.a * (h2 * hi - l2 * lo) / 3
                + self.b * (h2 - l2) / 2 + self.c * (hi - lo))

    def z_integral(self, lo, hi):
        """Integral of z*q(z)."""
        h2, l2 = hi * hi, lo * lo
        h3, l3 = h2 * hi, l2 * lo
        return (self.a * (h3 * hi - l3 * lo) / 4
                + self.b * (h3 - l3) / 3
                + self.c * (h2 - l2) / 2)

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

    def mirrored(self, axis: str) -> "AxisStack":
        # the members are coaxial by construction and a reflection keeps them
        # so; _pieces sorts its cuts, so the z-order need not be restored
        i = _mirror_axis(axis)
        return AxisStack(-self.cx if i == 0 else self.cx,
                         -self.cy if i == 1 else self.cy,
                         [p.mirrored(axis) for p in self.prims])

    def rotated(self, axis, deg):
        r = _axis_rotation(axis, deg)
        if r is None:
            return None
        prims = [p.rotated(axis, deg) for p in self.prims]
        if any(p is None for p in prims):
            return None
        fn, _flipped = r
        cx, cy = fn(self.cx, self.cy)
        return AxisStack(cx, cy, prims)

    def scaled(self, f) -> "AxisStack":
        s = _scale_factor(f)
        return AxisStack(self.cx * s, self.cy * s,
                         [p.scaled(s) for p in self.prims])

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
    360 degrees about a z-parallel axis through ``(cx, cy)``. Green gives
    exact metrics: V = pi * contour_integral(r^2 dz), each edge contributing
    (z2-z1)(r1^2 + r1 r2 + r2^2)/3. The profile need not touch r = 0 — an
    annular loop is a tube (a bored cylinder), still exact."""

    def __init__(self, loop_rz: list, cx=0, cy=0) -> None:
        self.loop = [(F(r), F(z)) for r, z in loop_rz]
        self.cx, self.cy = F(cx), F(cy)
        if any(r < 0 for r, _ in self.loop):
            raise ValueError("revolve profile must stay at r >= 0")
        if self._v3() < 0:
            self.loop = list(reversed(self.loop))

    def translated(self, x, y, z) -> "RevolveSolid":
        return RevolveSolid([(r, zz + F(z)) for r, zz in self.loop],
                            self.cx + F(x), self.cy + F(y))

    def mirrored(self, axis: str) -> "RevolveSolid":
        # the profile is a (r, z) loop swept about a VERTICAL axis, so an x or
        # y reflection only moves the axis — the solid is already symmetric
        # about every plane through it. Reflecting z negates the profile's z,
        # which reverses the loop's winding; __init__ re-orients it.
        i = _mirror_axis(axis)
        if i == 2:
            return RevolveSolid([(r, -z) for r, z in self.loop],
                                self.cx, self.cy)
        return RevolveSolid(list(self.loop),
                            -self.cx if i == 0 else self.cx,
                            -self.cy if i == 1 else self.cy)

    def rotated(self, axis, deg):
        # the solid is already symmetric about its own axis, so a z-rotation
        # only carries the axis position; a half turn about x or y stands the
        # profile on its head (__init__ re-orients the loop's winding)
        r = _axis_rotation(axis, deg)
        if r is None:
            return None
        fn, flipped = r
        cx, cy = fn(self.cx, self.cy)
        loop = [(rr, -z) for rr, z in self.loop] if flipped else list(self.loop)
        return RevolveSolid(loop, cx, cy)

    def scaled(self, f) -> "RevolveSolid":
        s = _scale_factor(f)
        return RevolveSolid([(r * s, z * s) for r, z in self.loop],
                            self.cx * s, self.cy * s)

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
        return (float(self.cx), float(self.cy),
                float(num / v3 * 3) if v3 else 0.0)

    def bbox(self):
        rmax = max(r for r, _ in self.loop)
        zs = [z for _, z in self.loop]
        cx, cy = float(self.cx), float(self.cy)
        return ((cx - float(rmax), cy - float(rmax), float(min(zs))),
                (cx + float(rmax), cy + float(rmax), float(max(zs))))

    def tessellate(self, deflection: float = 0.2) -> dict:
        from forgekernel.tess import lathe

        return lathe([(float(r), float(z)) for r, z in self.loop], deflection,
                     float(self.cx), float(self.cy))


def bore_cyl(cyl: "Cyl", bore: "Cyl") -> object:
    """Cut a COAXIAL bore out of a z-cylinder, exactly.

    A cylinder minus a coaxial cylindrical bore is still a solid of
    revolution, so the result is a ``RevolveSolid`` whose (r, z) profile is
    the cylinder's rectangle minus the bore's — exact in ℚ, with the volume
    following in ℚ[π] from Green's theorem. Returns ``cyl`` unchanged when
    the bore misses it in z. Raises ValueError for the cases that are not a
    single revolved region (a bore floating strictly inside — a closed
    cavity — or one that consumes the whole cylinder)."""
    if bore.cx != cyl.cx or bore.cy != cyl.cy:
        raise ValueError("bore is not coaxial with the cylinder — K2.3")
    if bore.r >= cyl.r:
        raise ValueError("bore is not narrower than the cylinder — K2.3")
    z0, z1 = cyl.z0, cyl.z1
    b0, b1 = max(bore.z0, z0), min(bore.z1, z1)
    if b1 <= b0:
        return cyl                              # bore misses in z: unchanged
    R, rb = cyl.r, bore.r
    if b0 <= z0 and b1 >= z1:                   # through bore -> annulus
        loop = [(rb, z0), (R, z0), (R, z1), (rb, z1)]
    elif b0 <= z0:                              # blind from the bottom
        loop = [(rb, z0), (R, z0), (R, z1), (0, z1), (0, b1), (rb, b1)]
    elif b1 >= z1:                              # blind from the top
        loop = [(0, z0), (R, z0), (R, z1), (rb, z1), (rb, b0), (0, b0)]
    else:
        raise ValueError(
            "bore floats strictly inside the cylinder (a closed cavity) — "
            "general quadric booleans arrive at K2.3")
    return RevolveSolid(loop, cyl.cx, cyl.cy)


def _member_volume(m):
    if isinstance(m, (Sphere, Cone)):
        return AxisStack(m.cx, m.cy, [m]).volume()
    if not hasattr(m, "volume"):
        # a member that already fell through to the canonical B-rep — an
        # operation on a union member returns a Body, and a union whose
        # members are a mix of representations is exactly what ADR-0021's
        # canonical form is for. Ask body.volume rather than AttributeError
        # out through the seam.
        from forgekernel import body as B

        return B.volume(m)
    return m.volume()


def _member_centroid(m):
    if isinstance(m, (Sphere, Cone)):
        return AxisStack(m.cx, m.cy, [m]).centroid_f()
    if hasattr(m, "centroid_f"):
        return m.centroid_f()
    if not hasattr(m, "centroid"):
        from forgekernel import body as B

        return tuple(float(x) for x in B.centroid(m))
    return tuple(float(x) for x in m.centroid())


def _face_z_and_up(p):
    """(z, up) for a horizontal planar face, else None — exact."""
    n = p.plane.n
    if n[0] != 0 or n[1] != 0:
        return None
    zs = {v[2] for v in p.verts}
    if len(zs) != 1:
        return None
    return (next(iter(zs)), n[2] > 0)


def _disc_strictly_inside_face(p, c) -> bool:
    """Exact: the disc (c.cx, c.cy, radius c.r) lies STRICTLY inside the
    face polygon's xy projection — centre inside by ray parity, every
    boundary edge clear of the disc by squared distance. Strict, so a
    tangent-to-the-rim disc keeps the merged-shells fallback rather than
    risking a degenerate band triangle."""
    verts = [(v[0], v[1]) for v in p.verts]
    m = len(verts)
    inside = False
    for i in range(m):
        (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % m]
        if (y1 > c.cy) != (y2 > c.cy):
            if c.cx < x1 + (c.cy - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    if not inside:
        return False
    r2 = c.r * c.r
    for i in range(m):
        (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % m]
        if _dist2_point_seg(c.cx, c.cy, x1, y1, x2, y2) <= r2:
            return False
    return True


def _face_convex_xy(p) -> bool:
    """Exact: the face polygon's xy projection is convex (collinear runs
    allowed). The angular band triangulation is only valid for a convex
    outer loop."""
    verts = [(v[0], v[1]) for v in p.verts]
    m = len(verts)
    pos = neg = False
    for i in range(m):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % m]
        cx, cy = verts[(i + 2) % m]
        cr = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cr > 0:
            pos = True
        elif cr < 0:
            neg = True
    return not (pos and neg)


def _stitch_plan(members) -> list:
    """Full-disc cap-on-face contacts between a Cyl and a planar Solid,
    detected with exact predicates: the cylinder's bottom cap on an
    up-facing horizontal face (a boss standing on a plate) or its top cap
    under a down-facing one, with the whole disc strictly inside a CONVEX
    face polygon. Returns [(solid_idx, poly_idx, cyl_idx, z)].

    Anything more entangled — several discs on one face, one cap meeting
    several faces, non-Cyl members, surd coordinates the predicates cannot
    divide through — keeps the merged-shells fallback for every party."""
    cands = []
    for ci, c in enumerate(members):
        if not isinstance(c, Cyl):
            continue
        for si, s in enumerate(members):
            if not isinstance(s, Solid):
                continue
            for pi, p in enumerate(s.polys):
                try:
                    fz = _face_z_and_up(p)
                    if fz is None:
                        continue
                    z, up = fz
                    if up and c.z0 == z:
                        end = "lo"
                    elif (not up) and c.z1 == z:
                        end = "hi"
                    else:
                        continue
                    if not _face_convex_xy(p):
                        continue
                    if not _disc_strictly_inside_face(p, c):
                        continue
                except (TypeError, ZeroDivisionError, AttributeError):
                    continue
                cands.append((si, pi, ci, end, z))
    from collections import Counter

    per_face = Counter((si, pi) for si, pi, _, _, _ in cands)
    per_cap = Counter((ci, end) for _, _, ci, end, _ in cands)
    return [(si, pi, ci, z) for si, pi, ci, end, z in cands
            if per_face[(si, pi)] == 1 and per_cap[(ci, end)] == 1]


def _ear_clip_f(loop, eps):
    """Float ear clipping of a weakly-simple CCW loop (the keyhole bridge
    duplicates two vertices, whose twins lie exactly ON the bridge edges —
    hence the STRICT-interior blocker test). Returns triangles or None;
    the caller re-checks area coverage, so a wrong clip cannot ship."""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    pts = list(loop)
    tris = []
    guard = 0
    while len(pts) > 3:
        guard += 1
        if guard > 20000:
            return None
        nn = len(pts)
        clipped = False
        for i in range(nn):
            a, b, c = pts[(i - 1) % nn], pts[i], pts[(i + 1) % nn]
            if orient(a, b, c) <= eps:
                continue                      # reflex or degenerate
            blocked = False
            for p in pts:
                if p == a or p == b or p == c:
                    continue
                if (orient(a, b, p) > eps and orient(b, c, p) > eps
                        and orient(c, a, p) > eps):
                    blocked = True
                    break
            if blocked:
                continue
            tris.append((a, b, c))
            pts.pop(i)
            clipped = True
            break
        if not clipped:
            return None
    if orient(*pts) <= eps:
        return None
    tris.append((pts[0], pts[1], pts[2]))
    return tris


def _annulus_band(outer_xy, cx, cy, r, n):
    """Triangulate (convex polygon) minus (inscribed circle n-gon) with NO
    new vertices: bridge the hole to the outer loop through a keyhole and
    ear-clip the merged weakly-simple polygon. No Steiner points means no
    T-vertices on the outer edges the side faces share. Returns CCW
    triangles as ((x,y),)*3, or None if the defensive coverage check
    fails (the caller then keeps the un-stitched mesh).

    The ring points are computed with EXACTLY the sub-expressions
    tess.lathe uses, so the hole rim re-uses the cylinder wall's vertex
    floats verbatim and the stitch is seamless by construction.

    Floats are right here (ADR-0019): a mesh is a bounded-error VIEW, and
    every quantity in this function is already display-precision float.
    The 1e-6 coverage check never decides the exact model's topology — it
    only chooses between two valid renderings of the same solid (the
    stitched band or the merged-shells fallback), and it fails CLOSED to
    the fallback."""
    ring = []
    for k in range(n):
        a = 2 * math.pi * k / n
        ring.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    m = len(outer_xy)
    area2 = sum(outer_xy[i][0] * outer_xy[(i + 1) % m][1]
                - outer_xy[(i + 1) % m][0] * outer_xy[i][1] for i in range(m))
    outer = list(outer_xy) if area2 > 0 else list(reversed(outer_xy))

    # Bridge from the ring's rightmost vertex M = ring[0] (angle 0, so
    # x = cx + r is its strict maximum) to the outer's max-x vertex P:
    # P.x > M.x because (cx + r, cy) is strictly inside the face, so the
    # segment M->P moves monotonically into x > M.x and can never re-enter
    # the disc (which lies wholly in x <= M.x); with a convex outer and
    # both ends inside, the bridge stays inside the outer loop.
    p_i = max(range(m), key=lambda i: outer[i][0])
    o = outer[p_i:] + outer[:p_i]
    hole_cw = [ring[0]] + ring[:0:-1]
    merged = [o[0], ring[0]] + hole_cw[1:] + [ring[0], o[0]] + o[1:]

    ext = max(max(abs(x - cx), abs(y - cy)) for x, y in outer)
    eps = 1e-12 * ext * ext
    tris = _ear_clip_f(merged, eps)
    if tris is None:
        return None

    # defensive: every triangle CCW-positive (the clipper guarantees it)
    # and the band tiles exactly polygon minus ring — any fold or overlap
    # breaks the sum (a float check on a float display mesh)
    def a2(t):
        (ax, ay), (bx, by), (tx, ty) = t
        return (bx - ax) * (ty - ay) - (by - ay) * (tx - ax)

    ring_a2 = sum(ring[j][0] * ring[(j + 1) % n][1]
                  - ring[(j + 1) % n][0] * ring[j][1] for j in range(n))
    target = abs(area2) - ring_a2
    band = sum(a2(t) for t in tris)
    if abs(band - target) > 1e-6 * max(abs(float(target)), 1.0):
        return None
    return tris


def _solid_mesh_with_holes(solid, holemap, deflection):
    """Like Solid.tessellate, but the faces in holemap {poly_idx: Cyl} get
    the contact disc punched out and band-triangulated. Preserves each
    face's stored winding. Returns None if any band fails (the caller
    then keeps the plain merged mesh AND the cylinder caps)."""
    from forgekernel.tess import _nseg

    verts: list = []
    tris: list = []
    index: dict = {}

    def vid(x, y, z) -> int:
        key = (x, y, z)
        j = index.get(key)
        if j is None:
            j = len(verts)
            index[key] = j
            verts.append([x, y, z])
        return j

    for pi, p in enumerate(solid.polys):
        pts = [(float(v[0]), float(v[1]), float(v[2])) for v in p.verts]
        c = holemap.get(pi)
        if c is None:
            ids = [vid(*q) for q in pts]
            for a, b in zip(ids[1:], ids[2:]):
                tris.append([ids[0], a, b])
            continue
        zc = pts[0][2]
        band = _annulus_band([(q[0], q[1]) for q in pts],
                             float(c.cx), float(c.cy), float(c.r),
                             _nseg(float(c.r), deflection))
        if band is None:
            return None
        m = len(pts)
        stored_ccw = sum(pts[i][0] * pts[(i + 1) % m][1]
                         - pts[(i + 1) % m][0] * pts[i][1]
                         for i in range(m)) > 0
        for t in band:
            ids = [vid(x, y, zc) for x, y in t]
            tris.append(ids if stored_ccw else [ids[0], ids[2], ids[1]])
    return {"vertices": verts, "triangles": tris}


def _strip_cap_triangles(mesh, zs) -> dict:
    """Drop triangles lying wholly in a horizontal cap plane z in zs."""
    zf = {float(z) for z in zs}
    v = mesh["vertices"]

    def is_cap(t) -> bool:
        z0 = v[t[0]][2]
        return z0 in zf and v[t[1]][2] == z0 and v[t[2]][2] == z0

    return {"vertices": v,
            "triangles": [t for t in mesh["triangles"] if not is_cap(t)]}


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

    def mirrored(self, axis: str) -> "DisjointUnion":
        # _unchecked, not the validating constructor: a reflection is a
        # bijection, so members that were disjoint still are. Re-running the
        # pairwise classifier would only spend exact arithmetic re-deriving a
        # fact the isometry already guarantees.
        return DisjointUnion._unchecked(
            [m.mirrored(axis) for m in self.members])

    def rotated(self, axis, deg):
        # _unchecked for the same reason as mirrored: a rotation is a
        # bijection, so members that were disjoint still are
        from forgekernel.brep import Solid
        from forgekernel.kernel import rotate as _rotate_solid

        out = []
        for m in self.members:
            if isinstance(m, Solid):
                # a plate under a boss is an ordinary Solid and rotates
                # exactly; requiring `rotated` on every member sent the whole
                # union to the canonical Body for want of one plain plate
                try:
                    out.append(_rotate_solid(m, axis, deg))
                except ValueError:
                    return None               # angle outside the exact table
                continue
            fn = getattr(m, "rotated", None)
            r = fn(axis, deg) if fn is not None else None
            if r is None:
                return None                   # one member cannot absorb it
            out.append(r)
        return DisjointUnion._unchecked(out)

    def scaled(self, f) -> "DisjointUnion":
        # a similarity is a bijection, so disjointness survives it
        from forgekernel.brep import Solid

        s = _scale_factor(f)
        # Solid.scaled takes three factors; the quadrics take one ratio
        return DisjointUnion._unchecked(
            [m.scaled(s, s, s) if isinstance(m, Solid) else m.scaled(s)
             for m in self.members])

    def tessellate(self, deflection: float = 0.2) -> dict:
        """Display mesh = the members' meshes merged — and where a Cyl cap
        stands on (or hangs under) a member face with its full disc inside,
        the contact is STITCHED: the face gets the disc punched out (band
        triangulation over the very ring floats the lathe emits) and the
        buried cap is dropped, so no triangle in the contact patch has
        material on both sides. Concatenating the closed shells verbatim
        kept volume and per-shell pairing right while lying topologically:
        any slicer or half-edge consumer saw internal walls (W17).

        Contacts outside that shape — several discs on one face, a boss
        overhanging the face edge, non-Cyl members — keep the merged-shells
        fallback: each shell stays individually closed, so volume and edge
        pairing remain right, but the coplanar contact faces remain buried
        (the honest residual until the canonical-Body converter arrives)."""
        stitches = _stitch_plan(self.members)
        solid_holes: dict = {}
        for si, pi, ci, z in stitches:
            solid_holes.setdefault(si, {})[pi] = (ci, z)
        premeshed: dict = {}
        cap_strip: dict = {}
        stitched: set = set()
        for si, holemap in solid_holes.items():
            sub = _solid_mesh_with_holes(
                self.members[si],
                {pi: self.members[ci] for pi, (ci, _) in holemap.items()},
                deflection)
            if sub is None:
                continue        # band failed: cancel — caps stay in place
            premeshed[si] = sub
            stitched.add(si)
            for _, (ci, z) in holemap.items():
                cap_strip.setdefault(ci, set()).add(z)
                stitched.add(ci)

        verts: list = []
        tris: list = []
        shared: dict = {}       # float coords -> index, stitched members only

        def add(sub: dict, dedup: bool) -> None:
            if dedup:
                # the hole rim and the wall's bottom ring carry identical
                # floats by construction; keying on them fuses the seam so
                # the stitched result pairs edge-for-edge BY INDEX
                remap = []
                for vv in sub["vertices"]:
                    key = (vv[0], vv[1], vv[2])
                    j = shared.get(key)
                    if j is None:
                        j = len(verts)
                        shared[key] = j
                        verts.append([vv[0], vv[1], vv[2]])
                    remap.append(j)
                tris.extend([remap[a], remap[b], remap[c]]
                            for a, b, c in sub["triangles"])
            else:
                off = len(verts)
                verts.extend([list(vv) for vv in sub["vertices"]])
                tris.extend([a + off, b + off, c + off]
                            for a, b, c in sub["triangles"])

        for mi, m in enumerate(self.members):
            if mi in premeshed:
                add(premeshed[mi], True)
                continue
            if type(m).__name__ == "Body":
                # a cut can now leave a canonical Body as a member; asking it
                # for .cx (via AxisStack) is a bare AttributeError, and the
                # union becomes unrenderable while still measuring fine
                from forgekernel.body import tessellate as _btess

                add(_btess(m, deflection), False)
                continue
            if not hasattr(m, "tessellate"):    # bare Sphere/Cone: via a stack
                m = AxisStack(m.cx, m.cy, [m])
            try:
                sub = m.tessellate(deflection)
            except TypeError:                   # planar Solid: meshes exactly
                sub = m.tessellate()
            if mi in cap_strip:
                sub = _strip_cap_triangles(sub, cap_strip[mi])
            add(sub, mi in stitched)
        return {"vertices": verts, "triangles": tris}

    @classmethod
    def _unchecked(cls, members: list) -> "DisjointUnion":
        """Build without re-running the pairwise predicates — for members
        derived from an already-validated union by an operation that can only
        SHRINK them (see ``cut``)."""
        u = cls.__new__(cls)
        u.members = list(members)
        return u

    def cut(self, tool) -> "DisjointUnion":
        """Subtract ``tool`` from every member: (A ∪ B) ∖ C = (A∖C) ∪ (B∖C).
        Members that the tool misses come back unchanged. Disjointness is
        preserved because cutting only shrinks a member — (A∖C) ∩ (B∖C) ⊆
        A ∩ B, already proven measure-zero — so the result needs no
        re-validation (and its members may now be other exact types)."""
        from forgekernel.brep import Solid

        out = []
        for m in self.members:
            if isinstance(m, Cyl) and isinstance(tool, Cyl):
                out.append(bore_cyl(m, tool))
            elif isinstance(m, (Solid, DrilledSolid)) and isinstance(tool, Cyl):
                base = DrilledSolid(m, []) if isinstance(m, Solid) else m
                try:
                    out.append(base.cut(tool))
                except ValueError as exc:
                    if "misses the solid in" not in str(exc):
                        raise                   # a real precondition violation
                    out.append(m)               # tool misses this member (z or xy)
            else:
                raise ValueError(
                    f"cut of {type(m).__name__} by {type(tool).__name__} in a "
                    "disjoint union arrives at K2.3")
        return DisjointUnion._unchecked(out)

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
        # a member may be a canonical Body — every member already may be, for
        # volume and centroid (see `_member_volume`), and this was the one
        # accessor that still assumed a feature representation and
        # AttributeError'd straight through the seam. `body.bbox` is the right
        # answer here BECAUSE this method is documented as float: nothing
        # topological is decided on it (`_body_outer_bbox` is what separation
        # uses), so the float bound is a bound and not a rounded predicate.
        from forgekernel import body as B

        boxes = [m.bbox() if hasattr(m, "bbox") else B.bbox(m)
                 for m in self.members]
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
        if d2 >= (a.r + b.r) ** 2:
            return                            # externally clear (or tangent)
        # NOTE: d2 <= (ra−rb)² is CONTAINMENT, not separation — these are solid
        # cylinders, not tubes, so a nested one overlaps with positive measure
        # (identical cylinders land here too, at d2 = 0). Treating it as clear
        # silently doubled the volume; it must refuse.
        raise ValueError(
            "overlapping cylinders — general quadric booleans arrive at K2.3")
    # sphere / planar Solid
    if isinstance(a, Sphere) and isinstance(b, Solid):
        _sphere_solid(a, b)
        return
    if isinstance(b, Sphere) and isinstance(a, Solid):
        _sphere_solid(b, a)
        return
    # z-cylinder / planar Solid (a mounting boss standing on a face)
    if isinstance(a, Cyl) and isinstance(b, Solid):
        _cyl_solid(a, b)
        return
    if isinstance(b, Cyl) and isinstance(a, Solid):
        _cyl_solid(b, a)
        return
    # sphere / sphere
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        d2 = (a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2 + (a.cz - b.cz) ** 2
        if d2 >= (a.r + b.r) ** 2:
            return                            # externally clear (or tangent)
        # containment (d2 ≤ (ra−rb)²) is an overlap, not a separation — see the
        # cylinder note above; SphereOverlap already rejects it as "nesting".
        raise ValueError(
            "overlapping spheres — general quadric booleans arrive at K2.3")
    # General fallback: EXACT axis-aligned separation. If the two bounding
    # boxes do not overlap on some axis then neither do the solids, whatever
    # they are — so the pair needs no type-specific predicate at all. This is
    # the common case in practice (a boss standing on a plate, two features on
    # opposite ends of a bracket) and it was refusing purely for want of a
    # rule for that particular pair of types. Touching counts as separated:
    # a tangent contact has measure zero.
    #
    # OUTER, not exact-tight: separation is the one question an over-estimate
    # cannot get wrong. See `_outer_bbox`.
    ba, bb = _outer_bbox(a), _outer_bbox(b)
    if ba is not None and bb is not None:
        if any(ba[1][k] <= bb[0][k] or bb[1][k] <= ba[0][k] for k in range(3)):
            return
    raise ValueError(
        f"disjoint-union of {type(a).__name__}+{type(b).__name__} arrives "
        "at K2.3")


def _seg_dist2(px, py, ax, ay, bx, by):
    """Exact squared distance from a point to a segment (rational throughout)."""
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * dx + (py - ay) * dy) / den
    t = F(0) if t < 0 else (F(1) if t > 1 else t)
    qx, qy = ax + t * dx, ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _column_slabs(base: Solid, c: "Cyl") -> list:
    """The SOLID z-intervals the bore's disc passes through, in order.

    The bore's whole DISC must sit over a constant cross-section.

    ``_bore_union_volume`` removes π r² (z1−z0) — the volume of a full-height
    barrel — so it is only right when the material really is full height
    across the entire disc, not merely under its centre. Where it is not, the
    barrel passes through a chamfer's taper or a shell's cavity without
    crossing any lateral wall, so both existing checks pass and the removed
    volume is overstated: a Ø0.5 bore under a 2 mm chamfer removed π/4 where
    the truth is π/8, and a bore through a hollowed plate removed the full
    10 mm where only the 2 mm walls carry material.

    A non-horizontal face reaching the disc is exactly the signal that the
    span varies, and it is an exact rational test.
    """
    r2 = c.r * c.r
    levels = []
    for p in base.polys:
        n = p.plane.n
        verts = [(v[0], v[1]) for v in p.verts]
        m = len(verts)
        if n[0] == 0 and n[1] == 0:
            # horizontal: it caps the column. TWO levels is a simple slab;
            # more means the column passes through a cavity — a shelled plate
            # has an outer top and bottom plus the void's ceiling and floor,
            # and a full-barrel removal would take material that is not there.
            inside = False
            for i in range(m):
                (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % m]
                if (y1 > c.cy) != (y2 > c.cy):
                    if c.cx < x1 + (c.cy - y1) * (x2 - x1) / (y2 - y1):
                        inside = not inside
            if inside:
                # ORIENTATION, not just height. A down-facing cap (n·ẑ < 0) is
                # where material STARTS going up; an up-facing one is where it
                # ends. Recording z alone in a set collapsed the two caps of a
                # shared interface — three stacked plates drilled through
                # reported two bores instead of three and billed 5853.39 for a
                # true 5811.50, while `union` of the same three plates gave the
                # right answer. Multiplicity matters too: keep a list.
                levels.append((p.plane.d / n[2], -1 if n[2] < 0 else 1))
            continue
        hit = False
        for i in range(m):
            (ax, ay), (bx, by) = verts[i], verts[(i + 1) % m]
            if _seg_dist2(c.cx, c.cy, ax, ay, bx, by) < r2:
                hit = True
                break
        if not hit:                             # or the face covers the disc
            inside = False
            for i in range(m):
                (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % m]
                if (y1 > c.cy) != (y2 > c.cy):
                    if c.cx < x1 + (c.cy - y1) * (x2 - x1) / (y2 - y1):
                        inside = not inside
            hit = inside
        if hit:
            raise ValueError(
                "bore reaches a non-vertical wall, so the solid is not full "
                "height across the hole — a full-barrel removal would "
                "overstate it (K2.1)")
    # No lateral face reaches the disc (checked above), so the cross-section is
    # CONSTANT over the whole disc, and the column is decided by SWEEPING the
    # caps in z with a depth counter: a down-facing cap opens material, an
    # up-facing one closes it, and a shared interface contributes both and
    # cancels. Alternating a sorted SET of heights assumed every level flips
    # the state, which is only true when no two bodies touch.
    if not levels:
        raise ValueError(
            "bore column meets no horizontal cap — the material over the disc "
            "is not decidable here (K2.1)")
    slabs, depth, start = [], 0, None
    for z, delta in sorted(levels):
        was = depth
        depth += -delta          # -1 opens (down-facing), +1 closes
        if depth < 0:
            raise ValueError(
                "bore column's caps do not nest — more ceilings than floors "
                "over the disc, so the material is not decidable (K2.1)")
        if was == 0 and depth > 0:
            start = z
        elif was > 0 and depth == 0:
            slabs.append((start, z))
    if depth != 0 or not slabs:
        raise ValueError(
            "bore column is not closed over the disc — a floor without its "
            "ceiling means the material is not decidable here (K2.1)")
    return slabs


def _exact_bbox(shape):
    """Exact axis-aligned bounds, or None when the representation cannot give
    them without leaving ℚ. Never a float — this decides topology."""
    from forgekernel.brep import Solid

    if isinstance(shape, Solid):
        vs = [v for p in shape.polys for v in p.verts]
        if not vs:
            return None
        return (tuple(min(v[k] for v in vs) for k in range(3)),
                tuple(max(v[k] for v in vs) for k in range(3)))
    if isinstance(shape, Cyl):
        return ((shape.cx - shape.r, shape.cy - shape.r, min(shape.z0, shape.z1)),
                (shape.cx + shape.r, shape.cy + shape.r, max(shape.z0, shape.z1)))
    if isinstance(shape, Sphere):
        return ((shape.cx - shape.r, shape.cy - shape.r, shape.cz - shape.r),
                (shape.cx + shape.r, shape.cy + shape.r, shape.cz + shape.r))
    if isinstance(shape, Cone):
        r = max(shape.r1, shape.r2)
        return ((shape.cx - r, shape.cy - r, min(shape.z0, shape.z1)),
                (shape.cx + r, shape.cy + r, max(shape.z0, shape.z1)))
    if isinstance(shape, RevolveSolid):
        r = max(rr for rr, _ in shape.loop)
        zs = [zz for _, zz in shape.loop]
        return ((shape.cx - r, shape.cy - r, min(zs)),
                (shape.cx + r, shape.cy + r, max(zs)))
    if isinstance(shape, DrilledSolid):
        return _exact_bbox(shape.base)         # bores only remove material
    if isinstance(shape, RoundedBox):
        o = shape.origin
        return (tuple(o), (o[0] + shape.a, o[1] + shape.b, o[2] + shape.c))
    if isinstance(shape, AxisStack):
        parts = [_exact_bbox(m) for m in shape.prims]
    elif isinstance(shape, DisjointUnion):
        parts = [_exact_bbox(m) for m in shape.members]
    else:
        return None
    if not parts or any(p is None for p in parts):
        return None
    return (tuple(min(p[0][k] for p in parts) for k in range(3)),
            tuple(max(p[1][k] for p in parts) for k in range(3)))


def _body_outer_bbox(shape):
    """A sound OUTER bound for a canonical ``Body`` — exact, or None.

    Deliberately allowed to be loose, which `_exact_bbox` is not. The two
    contracts must not be merged: `_flat_on_bar` asks `_exact_bbox` whether a
    tool COVERS a bar, and a box that is too big would approve a cut that
    actually misses. The only caller here proves the opposite direction —
    DISJOINTNESS — where an over-estimate can only fail to prove separation
    and can never claim it wrongly. Loose in the safe direction is the whole
    reason this is a separate function.

    A ``Body`` is the representation every transformed or already-cut solid
    falls into (ADR-0021), so without this the general fallback above could
    never fire for the commonest operand in the composed grid: two solids at
    opposite ends of a bracket refused as an unbuilt curved-surface boolean,
    naming an intersection that is empty.

    Exact throughout — `body.bbox` exists but is explicitly a FLOAT bound, and
    a rounded coordinate must not decide a topological question (ADR-0019).
    Anything whose reach is not rational in the face's own field — an obliquely
    axed cylinder, a cone, a torus — returns None so the caller refuses by name
    instead of guessing.
    """
    from forgekernel import body as B

    lo: list = [None] * 3
    hi: list = [None] * 3

    def bump(i, v):
        if lo[i] is None or v < lo[i]:
            lo[i] = v
        if hi[i] is None or v > hi[i]:
            hi[i] = v

    try:
        for f in shape.faces:
            s = f.surface
            if isinstance(s, B.Plane):
                pass                          # bounded entirely by its loops
            elif isinstance(s, B.SphereS):
                # sound whether or not the face carries loops — a whole sphere
                # carries none, and c +- r still contains it
                for i in range(3):
                    bump(i, s.c[i] - s.r)
                    bump(i, s.c[i] + s.r)
            elif isinstance(s, B.Cylinder):
                ax = [i for i in range(3) if s.d[i] != 0]
                if len(ax) != 1:
                    return None               # oblique: the reach needs a sqrt
                if not f.loops:
                    return None               # nothing bounds it along the axis
                for i in range(3):
                    if i == ax[0]:
                        continue              # the loops below bound this one
                    bump(i, s.p[i] - s.r)
                    bump(i, s.p[i] + s.r)
            else:
                return None                   # Cone, Torus, anything new
            for lp in f.loops:
                for e in lp.edges:
                    for v in (e.v0, e.v1):
                        for i in range(3):
                            bump(i, v[i])
    except (TypeError, ValueError, ArithmeticError):
        # comparing across two different quadratic fields (a 45-degree cut
        # meeting a 30-degree one lands in Q(sqrt2,sqrt3)) can refuse to order
        # rather than lie. That is the exact arithmetic working; treat it as
        # "cannot prove it" and let the caller say so by name.
        return None
    if any(v is None for v in lo) or any(v is None for v in hi):
        return None
    return (tuple(lo), tuple(hi))


def _outer_bbox(shape):
    """`_exact_bbox` where it applies, widened to cover the canonical Body.

    Sound OUTER bound only — never use where tightness is load-bearing.
    """
    from forgekernel import body as B

    if isinstance(shape, B.Body):
        return _body_outer_bbox(shape)
    return _exact_bbox(shape)


def _require_convex(solid: "Solid", what: str) -> None:
    """Raise unless every vertex is on the inward side of every face plane.

    The separating-plane predicates below are sound ONLY for a convex solid: a
    convex body is the intersection of its face half-spaces, so "outside one
    face plane" implies "outside the solid". For a non-convex solid that
    implication fails — a bore sitting inside an L-bracket's arm is outside the
    plane of the other arm's face — and accepting it would silently double-count
    the overlap. Exact (rational comparisons); refuses rather than guesses."""
    planes, verts = {}, set()
    for p in solid.polys:
        planes.setdefault(p.plane.canonical(), (p.plane.n, p.plane.d))
        verts.update(p.verts)
    for n, dpl in planes.values():
        for v in verts:
            if dot(n, v) - dpl > 0:
                raise ValueError(
                    f"{what} against a non-convex solid — the separating-plane "
                    "test is not sound there; general quadric booleans arrive "
                    "at K2.3")


def _sphere_solid(s: "Sphere", solid: "Solid") -> None:
    """Disjoint/tangent iff the sphere center lies on the far side of (or
    exactly on) some face plane by at least the radius — exact, sqrt-free:
    for outward plane n·x = d, signed gap g = n·c − d; clear iff g > 0 and
    g² ≥ r²·(n·n) (both sides squared, exact). Convex-solid sufficient
    condition; a sphere separated from a convex solid is separated by one
    of its face planes."""
    _require_convex(solid, "sphere")
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


def _cyl_solid(c: "Cyl", solid: "Solid") -> None:
    """Disjoint/tangent iff some face plane of the solid separates the
    z-cylinder — exact and sqrt-free. For outward plane n·x = d, the
    cylinder's minimum of n·x is

        n_x·cx + n_y·cy + min(n_z·z0, n_z·z1) − r·√(n_x² + n_y²),

    so the cylinder is clear of that plane iff A ≥ r·√(n_x²+n_y²) where
    A = n_x·cx + n_y·cy + min(n_z·z0, n_z·z1) − d. Both sides are squared
    (A ≥ 0 and A² ≥ r²·(n_x²+n_y²)) to stay in ℚ. A mounting boss standing
    ON a face gives A = 0 exactly — tangent, measure-zero, admitted.
    Sufficient for a convex solid (separating-plane argument); for a
    non-convex one it is conservative, so it refuses rather than guesses."""
    _require_convex(solid, "cylinder")
    seen = set()
    for p in solid.polys:
        key = p.plane.canonical()
        if key in seen:
            continue
        seen.add(key)
        n, dpl = p.plane.n, p.plane.d
        a = (n[0] * c.cx + n[1] * c.cy
             + min(n[2] * c.z0, n[2] * c.z1) - dpl)
        if a >= 0 and a * a >= c.r * c.r * (n[0] * n[0] + n[1] * n[1]):
            return                            # separated by this face plane
    raise ValueError(
        "cylinder overlaps the solid — general quadric booleans arrive at K2.3")


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

    def mirrored(self, axis: str) -> "RoundedBox":
        # the box spans origin .. origin + (a, b, c), so reflecting an axis
        # sends that interval to −(origin + size) .. −origin: the FAR corner
        # becomes the new origin. Extents and radius are unchanged.
        i = _mirror_axis(axis)
        o = list(self.origin)
        o[i] = -(o[i] + (self.a, self.b, self.c)[i])
        return RoundedBox(self.a, self.b, self.c, self.r, tuple(o))

    def rotated(self, axis, deg):
        """QUARTER TURNS ONLY. The extents are axis-aligned by construction, so
        90° permutes them exactly and anything else genuinely leaves the
        family — a rounded box turned 45° is not a rounded box."""
        if deg != int(deg) or int(deg) % 90 != 0:
            return None
        r = _axis_rotation(axis, deg)
        if r is None and int(deg) % 180 != 0:
            return None
        from forgekernel.kernel import _rotation_matrix

        try:
            m = _rotation_matrix(axis, deg)
        except ValueError:
            return None
        size = (self.a, self.b, self.c)
        # a quarter turn sends each axis to ±another axis: read the permutation
        # off the matrix exactly, and refuse if it is not one (an oblique axis)
        perm, sign = [], []
        for i in range(3):
            nz = [j for j in range(3) if m[i][j] != 0]
            if len(nz) != 1 or abs(m[i][nz[0]]) != 1:
                return None
            perm.append(nz[0])
            sign.append(1 if m[i][nz[0]] > 0 else -1)
        lo = self.origin
        hi = tuple(lo[j] + size[j] for j in range(3))
        new_lo = tuple(min(sign[i] * lo[perm[i]], sign[i] * hi[perm[i]])
                       for i in range(3))
        new_size = tuple(size[perm[i]] for i in range(3))
        return RoundedBox(new_size[0], new_size[1], new_size[2], self.r,
                          new_lo)

    def scaled(self, f) -> "RoundedBox":
        # FOUR lengths, not three: forgetting the fillet radius yields a box
        # of the right size with the wrong corners
        s = _scale_factor(f)
        return RoundedBox(self.a * s, self.b * s, self.c * s, self.r * s,
                          tuple(v * s for v in self.origin))

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


def _sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _axis_seg_dist2(a, b):
    """Exact squared distance between two AXIS-ALIGNED 2D segments (ℚ)."""
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    alox, ahix = min(ax0, ax1), max(ax0, ax1)
    aloy, ahiy = min(ay0, ay1), max(ay0, ay1)
    blox, bhix = min(bx0, bx1), max(bx0, bx1)
    bloy, bhiy = min(by0, by1), max(by0, by1)
    gx = max(F(0), blox - ahix, alox - bhix)
    gy = max(F(0), bloy - ahiy, aloy - bhiy)
    return gx * gx + gy * gy


class FilletedPrism:
    """A right prism over a RECTILINEAR simple polygon with EVERY edge
    filleted at one radius r — exact in ℚ[π], with a π² term wherever the
    profile has a REENTRANT corner.

    Every edge is axis-aligned and right-angled, so each feature is the one
    the rounded box already has — with one addition. A convex vertical edge
    blends as a quarter-cylinder and its cap ends as sphere octants; a
    REENTRANT vertical edge takes an INSIDE fillet: a quarter-cylinder of
    ADDED material whose axis sits r along the void diagonal, and where that
    concave blend meets each cap band the rolling ball's centre traces an arc
    of radius 2r about the same axis — a quarter-swept TORUS patch (major 2r,
    minor r). Its swept angle is the quarter the reentrant corner turns, so
    the volume stays in ℚ[π] and picks up the π² term.

    With δ(t) = r − √(r² − t²) the cap-band cross-section at depth t is the
    polygon inset by δ, convex corners rounded at r−δ, reentrant corners
    relieved at r+δ (the torus section), giving

        V = (h−2r)·A_mid + 2·V_cap
        A_mid = A₀ − (n_cv − n_rf)(1−π/4)r²
        V_cap = A₀r − P·r²(1−π/4) + 4r³(5/3 − π/2)
                − n_cv(1−π/4)·2r³/3 + n_rf(1−π/4)·r³(14/3 − π)

    (∫δ = r²(1−π/4), ∫δ² = r³(5/3−π/2), ∫(r−δ)² = 2r³/3,
    ∫(r+δ)² = r³(14/3−π), each over t ∈ [0, r]).

    Verified before shipping, independently of this code: the L-profile
    (30×10 + 10×20, h=8, r=1) gives 3752 + 178π/3 + π²/2 both by these slabs
    and by the boolean decomposition (two rounded boxes + reentrant wedge),
    whose pieces were pinned by Monte-Carlo membership written straight from
    the rolling-ball definition — A∩B = 2134/3 + 43π/2 to ±0.009, wedge
    (1−π/4)(46/3 − 2π) to ±0.001, whole solid to ±0.57 — and the T/H profiles
    (two and four reentrant corners) to ±0.36 / ±0.16.

    GUARDS (all exact, all refusals): the profile must be rectilinear, simple
    and non-degenerate; every edge ≥ 2r (corner features consume r at each
    end); h ≥ 2r; and every NON-ADJACENT edge pair ≥ 2r apart. That last is
    the local-feature-size bound: every blend, corner square and reentrant
    wedge lies within r of the boundary element that generates it, so
    elements 2r apart cannot produce colliding features — an H whose crossbar
    is thinner than 2r refuses here even though every edge is long enough.
    """

    def __init__(self, poly, z0, h, r) -> None:
        self.r, self.z0, self.h = F(r), F(z0), F(h)
        if self.r <= 0:
            raise ValueError("fillet wants a positive radius")
        if self.h < 2 * self.r:
            raise ValueError("fillet radius exceeds half the prism height")
        pts = [(F(x), F(y)) for x, y in poly]
        if len(pts) < 4:
            raise ValueError("a rectilinear profile has at least 4 vertices")
        if len(set(pts)) != len(pts):
            raise ValueError("repeated vertex in the profile")
        area2 = sum(a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(pts, pts[1:] + pts[:1]))
        if area2 == 0:
            raise ValueError("degenerate (zero-area) profile")
        if area2 < 0:                       # normalise to CCW
            pts.reverse()
            area2 = -area2
        self.poly = tuple(pts)
        self.area = area2 / 2
        n = len(pts)
        edges = []
        self.perimeter = F(0)
        for i in range(n):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            if (dx == 0) == (dy == 0):
                raise ValueError(
                    "profile edge is not axis-aligned — a diagonal edge's "
                    "blend needs the general engine (K5.2)")
            length = abs(dx) + abs(dy)
            if length < 2 * self.r:
                raise ValueError(
                    "profile edge shorter than 2r: its two corner blends "
                    "would overlap (K5.2)")
            self.perimeter += length
            edges.append((pts[i], pts[(i + 1) % n]))
        # vertex classification: diag is the INWARD diagonal at a convex
        # corner and the VOID diagonal at a reentrant one; shift is how the
        # vertex moves under a unit inward inset (sum of edge inward normals)
        self.verts = []
        for i in range(n):
            pv, v, nx = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            e0 = (v[0] - pv[0], v[1] - pv[1])
            e1 = (nx[0] - v[0], nx[1] - v[1])
            cr = e0[0] * e1[1] - e0[1] * e1[0]
            if cr == 0:
                raise ValueError("consecutive collinear profile edges")
            n0 = (-_sgn(e0[1]), _sgn(e0[0]))          # inward (left) normals
            n1 = (-_sgn(e1[1]), _sgn(e1[0]))
            diag = (n0[0] + n1[0], n0[1] + n1[1])
            if cr > 0:
                self.verts.append((v, True, diag))
            else:
                self.verts.append((v, False, (-diag[0], -diag[1])))
        self.ncv = sum(1 for _, cv, _ in self.verts if cv)
        self.nrf = n - self.ncv
        # local feature size ≥ 2r between all non-adjacent edges. Exact, and
        # it doubles as the simplicity guard (distance > 0 forbids crossings).
        rr4 = 4 * self.r * self.r
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if _axis_seg_dist2(edges[i], edges[j]) < rr4:
                    raise ValueError(
                        "two profile edges closer than 2r: their blends "
                        "would collide mid-wall (K5.2)")

    def volume(self):
        """Exact: PiVal when the profile is convex (a rectangle — Steiner),
        PiPoly with the reentrant π² term otherwise."""
        from forgekernel.polypi import PiPoly

        r, h = self.r, self.h
        k1 = PiPoly([Fraction(1), Fraction(-1, 4)])                       # 1 − π/4
        r2, r3 = r * r, r * r * r
        a_mid = k1 * (-(self.ncv - self.nrf) * r2) + self.area
        v_cap = (PiPoly([self.area * r])
                 + k1 * (-self.perimeter * r2)
                 + PiPoly([Fraction(5, 3) * 4 * r3, Fraction(-1, 2) * 4 * r3])
                 + k1 * (-self.ncv * Fraction(2, 3) * r3)
                 + k1 * PiPoly([Fraction(14, 3), Fraction(-1)]) * (self.nrf * r3))
        v = a_mid * (h - 2 * r) + v_cap * 2
        if v.degree <= 1:
            return PiVal(v[0], v[1])
        return v

    # -- float paths (display/meshing only, ADR-0019) -------------------------

    def _slice_f(self, delta: float):
        """(area, ∫x dA, ∫y dA) of the cross-section at inset δ — floats."""
        shifts = [(d[0], d[1]) if cv else (-d[0], -d[1])
                  for _, cv, d in self.verts]
        pts = [(float(v[0]) + delta * s[0], float(v[1]) + delta * s[1])
               for (v, _, _), s in zip(self.verts, shifts)]
        a = mx = my = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            cr = x0 * y1 - x1 * y0
            a += cr
            mx += (x0 + x1) * cr
            my += (y0 + y1) * cr
        a, mx, my = a / 2, mx / 6, my / 6
        r = float(self.r)
        k = 1 - math.pi / 4
        xi = (5 / 6 - math.pi / 4) / k        # corner-region centroid ratio
        for v, cv, d in self.verts:
            vx, vy = float(v[0]), float(v[1])
            if cv:
                rho = r - delta
                ar = k * rho * rho
                cx = vx + (delta + xi * rho) * d[0]
                cy = vy + (delta + xi * rho) * d[1]
                a -= ar
                mx -= ar * cx
                my -= ar * cy
            else:
                rho = r + delta
                ar = k * rho * rho
                cx = vx + (xi * rho - delta) * d[0]
                cy = vy + (xi * rho - delta) * d[1]
                a += ar
                mx += ar * cx
                my += ar * cy
        return a, mx, my

    def centroid_f(self) -> tuple:
        r, h = float(self.r), float(self.h)
        a0, mx0, my0 = self._slice_f(0.0)
        acc = [(h - 2 * r) * a0, (h - 2 * r) * mx0, (h - 2 * r) * my0]
        # cap bands via t = r·sin(u): the integrand becomes smooth, so
        # Simpson converges to machine precision (display only, ADR-0019)
        n = 512
        step = (math.pi / 2) / n
        for i in range(n + 1):
            u = step * i
            w = 1.0 if i in (0, n) else (4.0 if i % 2 else 2.0)
            aa, mx, my = self._slice_f(r * (1 - math.cos(u)))
            jac = w * r * math.cos(u)
            acc[0] += 2 * aa * jac * step / 3
            acc[1] += 2 * mx * jac * step / 3
            acc[2] += 2 * my * jac * step / 3
        return (acc[1] / acc[0], acc[2] / acc[0], float(self.z0) + h / 2)

    def bbox(self):
        xs = [float(x) for x, _ in self.poly]
        ys = [float(y) for _, y in self.poly]
        return ((min(xs), min(ys), float(self.z0)),
                (max(xs), max(ys), float(self.z0 + self.h)))

    def watertight_violations(self) -> list:
        return []


class FilletedChamferedBox:
    """``chamfer(box, d)`` with EVERY edge blended at one radius r — exact in
    ℚ(√2,√3)[π], the biquadratic field (BiSurd's first consumer).

    The solid is convex, so ``fillet(all, r)`` is the OPENING ``P₋ᵣ ⊕ B_r``:
    the union of every r-ball inside the chamfered box. Eroding offsets each
    face plane inward by r; the √2-normal chamfer planes move their constant
    by r√2, so ``P₋ᵣ`` is the chamfered box of dims (A−2r, B−2r, C−2r) with
    setback ``t' = d − (2−√2)r`` — combinatorics preserved exactly when
    t' > 0 and every axis facet stays nonempty. Steiner then gives

        V = V(P₋ᵣ) + S(P₋ᵣ)·r + (r²/2)·Σ L_e·α_e + (4π/3)r³

    with α = π/4 on the 24 axis–chamfer edges (dihedral 3π/4) and α = π/3 on
    the 24 chamfer–chamfer edges (cos(dihedral) = −1/2 EXACTLY, so 2π/3 by
    Niven — verified on the solid, not assumed). Where each surd comes from:
    √2 rides in on t'; √3 enters as the ch-ch edge length (√3/2)t' times its
    π/3 sweep, surfacing as 2√6·π for the probe cell. Probe (20³, d=2, r=1):

        V = 8040 − 456√2 + (166/3 − 6√2 + 2√6)π = 7557.6867093895…

    qhull-checked pieces (≤1e-12) and MC membership at three seeds — the
    record lives in tests/golden/test_fillet_chamfered_box.py (gitcad).

    THE TRAP, banked from the derivation: the 32 corner patches are
    individually TRANSCENDENTAL over ℚ[√d][π] — type A carries arccos(1/3),
    type B arccos(1/√3), and neither is a rational multiple of π. They cancel
    only in aggregate (8Ω_A + 24Ω_B = 4π, a convex body's patches sum to one
    ball). So the volume MUST be assembled by this Steiner/opening
    decomposition; a per-corner-patch accumulation cannot stay in the field.

    GUARDS (all exact, all refusals):

    * ``d > 0``, ``r > 0``, ``2d < min(A,B,C)`` — the base chamfered box must
      itself be valid;
    * ``t' = d − (2−√2)r > 0`` — otherwise the ball BRIDGES the chamfer facet
      (the eroded polytope loses it) and adjacent edge blends collide; the
      kernel refuses collision regimes rather than switching semantics (the
      FilletedPrism precedent). With rational d, r the boundary t' = 0 is
      unreachable (it would force r = 0), so `>` vs `≥` costs nothing;
    * ``X − 2d − (2√2−2)r > 0`` per axis — the eroded axis facet stays
      nonempty; equally unreachable at equality for rational inputs.
    """

    def __init__(self, lo, dims, d, r) -> None:
        from forgekernel.surd import SurdVal

        self.lo = tuple(F(v) for v in lo)
        self.dims = tuple(F(v) for v in dims)
        self.d, self.r = F(d), F(r)
        if any(v <= 0 for v in self.dims):
            raise ValueError("chamfered box wants positive dimensions")
        if self.d <= 0:
            raise ValueError("chamfered box wants a positive setback")
        if self.r <= 0:
            raise ValueError("fillet wants a positive radius")
        if 2 * self.d >= min(self.dims):
            raise ValueError(
                "chamfer setback consumes a whole face (2d ≥ min dimension)")
        tp = SurdVal(self.d - 2 * self.r, self.r, 2)     # t' = d − (2−√2)r
        if tp._sign() <= 0:
            raise ValueError(
                "fillet(chamfered box): the ball bridges the chamfer facet — "
                "d ≤ (2−√2)r erodes it away and the three edge blends at each "
                "corner collide (K5.2)")
        for x in self.dims:
            # eroded axis facet nonempty: X' − 2t' = X − 2d + (2 − 2√2)r > 0
            if SurdVal(x - 2 * self.d + 2 * self.r, -2 * self.r, 2)._sign() <= 0:
                raise ValueError(
                    "fillet(chamfered box): blends from opposite edges meet "
                    "across the face — needs X − 2d − (2√2−2)r > 0 on every "
                    "axis (K5.2)")

    def volume(self):
        """Exact ℚ(√2,√3)[π], by the Steiner/opening decomposition ONLY (see
        the class docstring for why per-corner patches are forbidden)."""
        from forgekernel.bisurd import BiSurd
        from forgekernel.polypi import PiPoly

        r = self.r
        ea, eb, ec = (x - 2 * r for x in self.dims)      # eroded dims, rational
        s = ea + eb + ec
        t = BiSurd(self.d - 2 * r, r, 0, 0, 2, 3)        # t' = (d−2r) + r√2
        sqrt2 = BiSurd(0, 1, 0, 0, 2, 3)
        sqrt3 = BiSurd(0, 0, 1, 0, 2, 3)
        # V(P₋ᵣ): chamfered box of dims (ea, eb, ec), setback t'
        v0 = ea * eb * ec - 2 * t * t * s + 6 * t * t * t
        # S(P₋ᵣ): 6 axis rectangles + 12 chamfer hexagons √2(ℓt' − (3/2)t'²)
        s0 = (2 * ((ea - 2 * t) * (eb - 2 * t) + (eb - 2 * t) * (ec - 2 * t)
                   + (ea - 2 * t) * (ec - 2 * t))
              + 4 * sqrt2 * t * s - 18 * sqrt2 * t * t)
        # edge term (r²/2)ΣLα: 8 axis–chamfer edges per axis, length X'−2t',
        # α = π/4; 24 chamfer–chamfer edges, length (√3/2)t', α = π/3
        pi1 = (r * r * (s - 6 * t) + 2 * sqrt3 * r * r * t
               + Fraction(4, 3) * r * r * r)             # + one full ball
        return PiPoly([v0 + s0 * r, pi1])

    def centroid_f(self) -> tuple:
        # the solid is centrally symmetric about the box centre — exact
        return tuple(float(self.lo[i] + self.dims[i] / 2) for i in range(3))

    def bbox(self):
        # the blend spheres touch the original box faces: erode r, dilate r
        return (self.lo, tuple(self.lo[i] + self.dims[i] for i in range(3)))

    def watertight_violations(self) -> list:
        return []


class MiteredSweep:
    """A convex profile swept along a polyline with miter joints — exact
    in ℚ[√d]. Volume = profile_area × centerline_length: at a miter the
    bisector plane removes a wedge from one segment that the neighbour
    adds back identically, so the swept volume is exactly the straight
    area×length even through corners. Segment lengths accumulate in one
    quadratic-surd field; a path mixing radicals (e.g. √2 and √3) refuses
    (K3.1). This is the model OCCT cannot build (swept_channel).

    Metrics (bbox / centroid_f) need the actual profile POLYGON, passed as
    ``profile`` — vertices (u, v) in profile coordinates with (0, 0) riding
    the path. Orientation convention: a +z first leg maps profile (u, v) to
    world (x, y); in general u₀ ∝ ŷ × t̂₀ (falling back to ẑ × t̂₀ when the
    first leg runs along ±y), v₀ = t̂₀ × u₀, and the frame is transported
    across each corner by the affine reflection in the miter plane (normal
    ∝ t̂_in + t̂_out — the picture-frame fold mapping each leg's prism onto
    the next; it fixes the joint face pointwise, so cross-sections agree).
    Each leg is then the profile prism clipped by its two planes — a convex
    polytope whose extreme points are profile vertices on those planes, so
    the box is exact, and whose centroid is a closed form in the profile's
    area/first/second moments. Built from an area alone the solid still
    knows its exact volume, but bbox/centroid REFUSE rather than guess
    (W8/W11: the old √area pad and wire centroid were silent wrong numbers
    through the seam — the pad even understated any profile wider than 4:1,
    excluding real material from the box)."""

    def __init__(self, area, path: list, profile: list | None = None) -> None:
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
        self.profile = None
        if profile is not None:
            pts = [(F(p[0]), F(p[1])) for p in profile]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                raise ValueError("sweep profile needs >= 3 vertices")
            n = len(pts)
            area2 = sum(pts[i][0] * pts[(i + 1) % n][1]
                        - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
            if area2 == 0:
                raise ValueError("sweep profile is degenerate (zero area)")
            if area2 < 0:                            # normalise to CCW
                pts = pts[::-1]
                area2 = -area2
            if self.area != area2 / 2:
                raise ValueError(
                    "sweep profile polygon area disagrees with the stated "
                    "area — refuse rather than pick one silently")
            self.profile = pts

    def length(self):
        return self._length

    def volume(self):
        return self._length * self.area              # SurdVal

    # -- metric machinery (W8/W11) --------------------------------------------

    def _leg_frames(self):
        """Float frames per leg + the clipping plane per leg end.

        Returns (origins, dirs, frames, planes): per-leg path origin P_k and
        unit direction t̂_k, per-leg (u, v) profile axes, and n+1 planes as
        (point, normal) with UNNORMALISED miter normals (every use divides
        two dot products with the same normal, so scale cancels).
        Degeneracy is decided on the EXACT rational path deltas — no float
        epsilon: a zero-length segment has no direction, and a straight
        reversal (t̂₁ + t̂₂ = 0) has no miter plane.
        """
        if self.profile is None:
            raise ValueError(
                "MiteredSweep built from an area alone knows its exact "
                "volume but not its shape: bbox/centroid need the profile "
                "polygon (pass profile=). Refusing rather than padding by "
                "sqrt(area) — the old pad was a silent wrong box (W8)")
        deltas = [tuple(b[i] - a[i] for i in range(3))
                  for a, b in zip(self.path, self.path[1:])]
        for d in deltas:
            if d == (0, 0, 0):
                raise ValueError(
                    "zero-length sweep segment has no direction — the miter "
                    "frame is undefined")
        for d1, d2 in zip(deltas, deltas[1:]):
            cross = (d1[1] * d2[2] - d1[2] * d2[1],
                     d1[2] * d2[0] - d1[0] * d2[2],
                     d1[0] * d2[1] - d1[1] * d2[0])
            if cross == (0, 0, 0) and sum(a * b for a, b in zip(d1, d2)) < 0:
                raise ValueError(
                    "sweep path reverses on itself — the miter plane "
                    "(normal t1+t2) is undefined for a straight reversal")
        origins = [tuple(float(c) for c in p) for p in self.path[:-1]]
        dirs = []
        for d in deltas:
            fd = tuple(float(c) for c in d)
            ln = math.sqrt(fd[0] * fd[0] + fd[1] * fd[1] + fd[2] * fd[2])
            dirs.append((fd[0] / ln, fd[1] / ln, fd[2] / ln))

        def _cross(a, b):
            return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0])

        def _dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        t0 = dirs[0]
        # documented first-leg convention (exact test: is the first delta ±y?)
        d0 = deltas[0]
        ref = (0.0, 0.0, 1.0) if (d0[0] == 0 and d0[2] == 0) else (0.0, 1.0, 0.0)
        u0 = _cross(ref, t0)
        un = math.sqrt(_dot(u0, u0))
        u0 = (u0[0] / un, u0[1] / un, u0[2] / un)
        v0 = _cross(t0, u0)
        frames = [(u0, v0)]
        for k in range(1, len(dirs)):
            n = tuple(dirs[k - 1][i] + dirs[k][i] for i in range(3))
            nn = _dot(n, n)
            u_p, v_p = frames[-1]
            frames.append((
                tuple(u_p[i] - 2 * _dot(u_p, n) / nn * n[i] for i in range(3)),
                tuple(v_p[i] - 2 * _dot(v_p, n) / nn * n[i] for i in range(3))))
        planes = [(origins[0], dirs[0])]
        for k in range(1, len(dirs)):
            planes.append((tuple(float(c) for c in self.path[k]),
                           tuple(dirs[k - 1][i] + dirs[k][i] for i in range(3))))
        planes.append((tuple(float(c) for c in self.path[-1]), dirs[-1]))
        return origins, dirs, frames, planes

    def _plane_coeffs(self, P, u, v, t, plane):
        """s(a, b) = α + β·a + γ·b where the leg line P + a·u + b·v + s·t̂
        pierces the plane (Q, m). Normal scale cancels in the ratios."""
        Q, m = plane
        tm = t[0] * m[0] + t[1] * m[1] + t[2] * m[2]
        qm = sum((Q[i] - P[i]) * m[i] for i in range(3))
        um = u[0] * m[0] + u[1] * m[1] + u[2] * m[2]
        vm = v[0] * m[0] + v[1] * m[1] + v[2] * m[2]
        return qm / tm, -um / tm, -vm / tm

    def _profile_integrals(self):
        """Exact polygon integrals about the profile origin (CCW): area A,
        first moments ∫a, ∫b, second moments ∫a², ∫b², ∫ab — Green's
        theorem closed forms, all Fractions."""
        pts = self.profile
        n = len(pts)
        A = Ma = Mb = Iaa = Ibb = Iab = F(0)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            cr = x0 * y1 - x1 * y0
            A += cr
            Ma += (x0 + x1) * cr
            Mb += (y0 + y1) * cr
            Iaa += (x0 * x0 + x0 * x1 + x1 * x1) * cr
            Ibb += (y0 * y0 + y0 * y1 + y1 * y1) * cr
            Iab += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * cr
        return A / 2, Ma / 6, Mb / 6, Iaa / 12, Ibb / 12, Iab / 24

    def centroid_f(self) -> tuple:
        # W11: the old code returned length-weighted CENTRELINE midpoints —
        # the wire's centroid. The solid's centroid weights each leg's
        # clipped-prism centroid (miter wedge included): with the leg's two
        # plane bounds affine in profile coords, s ∈ [α₀+β₀a+γ₀b, α₁+β₁a+γ₁b],
        # every moment is a closed form in the profile's area/1st/2nd moments.
        origins, dirs, frames, planes = self._leg_frames()
        A, Ma, Mb, Iaa, Ibb, Iab = (float(x) for x in self._profile_integrals())

        def E(al, be, ga):                    # ∫ s(a,b)² dA over the profile
            return (al * al * A + 2 * al * be * Ma + 2 * al * ga * Mb
                    + be * be * Iaa + 2 * be * ga * Iab + ga * ga * Ibb)

        vol = 0.0
        M = [0.0, 0.0, 0.0]
        for k, t in enumerate(dirs):
            P, (u, v) = origins[k], frames[k]
            a0, b0, g0 = self._plane_coeffs(P, u, v, t, planes[k])
            a1, b1, g1 = self._plane_coeffs(P, u, v, t, planes[k + 1])
            da, db, dg = a1 - a0, b1 - b0, g1 - g0
            Vk = da * A + db * Ma + dg * Mb
            Mu = da * Ma + db * Iaa + dg * Iab
            Mv = da * Mb + db * Iab + dg * Ibb
            Ms = (E(a1, b1, g1) - E(a0, b0, g0)) / 2
            vol += Vk
            for i in range(3):
                M[i] += P[i] * Vk + u[i] * Mu + v[i] * Mv + t[i] * Ms
        return (M[0] / vol, M[1] / vol, M[2] / vol)

    def bbox(self):
        # W8: exact box of the union of clipped prisms — every extreme point
        # of a leg polytope is a profile vertex on one of its two planes
        # (never the √area pad, which understated wide profiles).
        origins, dirs, frames, planes = self._leg_frames()
        prof = [(float(a), float(b)) for a, b in self.profile]
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for k, t in enumerate(dirs):
            P, (u, v) = origins[k], frames[k]
            for plane in (planes[k], planes[k + 1]):
                al, be, ga = self._plane_coeffs(P, u, v, t, plane)
                for a, b in prof:
                    s = al + be * a + ga * b
                    for i in range(3):
                        c = P[i] + a * u[i] + b * v[i] + s * t[i]
                        if c < lo[i]:
                            lo[i] = c
                        if c > hi[i]:
                            hi[i] = c
        return (tuple(lo), tuple(hi))

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

    def _lens_moment_picoeff(self) -> Fraction:
        """π-coefficient of the lens's first moment along the axis of
        centres, measured from sphere A's centre. A cap {ξ ≥ r−h} of a
        sphere radius r has moment about its own centre ∫ξ·π(r²−ξ²)dξ
        = π h²(2r−h)²/4; B's cap rides at offset d and points backward."""
        r1, r2, d = self.a.r, self.b.r, self.d
        d1 = (d * d + r1 * r1 - r2 * r2) / (2 * d)
        h1 = r1 - d1
        h2 = r2 - (d - d1)
        return (h1 * h1 * (2 * r1 - h1) ** 2 / 4
                + d * self._cap(r2, h2)
                - h2 * h2 * (2 * r2 - h2) ** 2 / 4)

    def centroid_f(self) -> tuple:
        # W7: the old midpoint-of-centres was right only for the two
        # symmetric ops (union/intersect of EQUAL spheres) — a cut centroid
        # came back on the wrong side of the origin. The true centroid sits
        # on the axis of centres at a RATIONAL offset ξ from A's centre
        # (every term is rational·π, so π cancels in the ratio); the floats
        # returned are the correctly-rounded exact values.
        v1 = Fraction(4, 3) * self.a.r ** 3
        v2 = Fraction(4, 3) * self.b.r ** 3
        lens = self._lens_picoeff()
        m = self._lens_moment_picoeff()
        if self.op == "intersect":
            xi = m / lens
        elif self.op == "cut":
            xi = -m / (v1 - lens)         # A's own moment about A is zero
        else:                             # union = A + B − lens
            xi = (self.d * v2 - m) / (v1 + v2 - lens)
        a, b = self.a, self.b
        w = xi / self.d                   # rational barycentric weight
        return (float(a.cx + w * (b.cx - a.cx)),
                float(a.cy + w * (b.cy - a.cy)),
                float(a.cz + w * (b.cz - a.cz)))

    def _support(self, e) -> float:
        """Exact support (farthest reach) of the solid along the axis
        direction ``e`` (a ±1 one-hot tuple). Every DECISION is a rational
        comparison; only the final square root floats. Geometry: the cut
        solid ends at the radical plane ξ = d1 = (d²+r1²−r2²)/(2d), the
        lens's transverse extreme sits on the waist circle (centre
        A + d1·û, radius w = √(r1²−d1²)), and a far pole survives iff it
        is not strictly past the plane / inside the other sphere — with
        rim and pole formulas agreeing exactly at the boundary. (The
        pole-inside-A ∧ pole-inside-B corner case is impossible for a cut:
        it would need d < |r1−r2|, which __init__ already refused.)"""
        a, b = self.a, self.b
        ac = (a.cx, a.cy, a.cz)
        bc = (b.cx, b.cy, b.cz)
        sa = sum(c * x for c, x in zip(ac, e)) + a.r
        sb = sum(c * x for c, x in zip(bc, e)) + b.r
        if self.op == "union":
            return float(max(sa, sb))
        d, r1, r2 = self.d, a.r, b.r
        d1 = (d * d + r1 * r1 - r2 * r2) / (2 * d)
        eu_d = sum((bc[i] - ac[i]) * e[i] for i in range(3))   # d·(e·û)
        if self.op == "cut":
            if r1 * eu_d <= d1 * d:       # A's far pole is not past the plane
                return float(sa)
        else:                             # intersect (the lens)
            if sum((ac[i] + r1 * e[i] - bc[i]) ** 2 for i in range(3)) <= r2 * r2:
                return float(sa)          # A's far pole is inside B
            if sum((bc[i] + r2 * e[i] - ac[i]) ** 2 for i in range(3)) <= r1 * r1:
                return float(sb)          # B's far pole is inside A
        # extreme on the waist circle
        w2 = r1 * r1 - d1 * d1
        ce = sum(ac[i] * e[i] for i in range(3)) + d1 * eu_d / d
        return float(ce) + math.sqrt(float(w2 * (1 - eu_d * eu_d / (d * d))))

    def bbox(self):
        # W12: the cut box was sphere A's whole box (+25% on the cut axis)
        # and the intersect box ignored the lens waist. Exact per-axis
        # support of the actual solid; union keeps its two-sphere box.
        axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        lo = tuple(-self._support((-e[0], -e[1], -e[2])) for e in axes)
        hi = tuple(self._support(e) for e in axes)
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
    """A box with LINEAR-TAPER **disc-sweep** fillets on non-adjacent
    straight edges — exact in ℚ[π].

    THE SEMANTIC IS LOAD-BEARING. Two G1-valid surfaces answer to "linear
    variable-radius fillet", they share their tangency lines on both faces,
    and they sit on opposite sides of the exact-field boundary:

    * DISC-SWEEP (this class): each slice perpendicular to the edge is the
      2D fillet arc of radius r(t) = r0 + (r1−r0)·t/L. The surface is an
      OBLIQUE circular cone with apex on the edge line where r extrapolates
      to 0 (scaling identity verified to 4e-14), G1-tangent to both faces.
      A slice removes area r(t)²(1−π/4) and ∫₀^L r(t)² dt =
      L(r0²+r0r1+r1²)/3 exactly, so

        V = V_box − Σ (1−π/4)·L(r0²+r0r1+r1²)/3      — a PiVal, exact.

    * ROLLING BALL (the Parasolid/SolidWorks semantic — envelope of spheres
      of radius r(t)): a RIGHT circular cone whose perpendicular slice is an
      ELLIPSE. The slice-segment area carries the eccentric-anomaly span Δu
      linearly, with cos Δu = b² for taper slope b = (r1−r0)/L — and by
      Niven's theorem Δu/π is rational only at b² ∈ {0, ½, 1}: constant
      radius, the irrational slope b = 1/√2, or the degenerate b = 1. For
      EVERY rational nonzero taper the volume is transcendental over every
      ℚ[√d][π] (the Lindemann class) — so the rolling ball is a wall, not
      backlog, and this kernel does not pretend to it.

    Divergence of the two (removed volume, computed): 0.33% at b = 0.1,
    4.9% at b = 0.4, 16.1% at b = 0.8. Consumers comparing against a
    Parasolid-derived kernel must expect exactly that gap, by definition,
    not by defect.

    Edges: (axis, side_a, side_b, r0, r1). Selected edges must be
    pairwise non-adjacent (a shared vertex needs a variable corner
    patch → K5.3; a sphere octant at matched vertex radii would be
    C0-watertight but NOT G1 against the taper cones)."""

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

    def centroid_f(self) -> tuple:
        """Float centroid from the same closed forms as the volume
        (FilletedBox precedent — floats are display here, the exactness
        claim lives in ``volume()``).

        Per edge, with r(t) = r0 + (r1−r0)t/L and the slice's removed area
        a(t) = r(t)²(1−π/4):

          ∫a dt      = (1−π/4)·L(r0²+r0r1+r1²)/3
          ∫a·t dt    = (1−π/4)·L²(r0²+2r0r1+3r1²)/12   (axial moment)
          ∫a·c* dt   = (5/6−π/4)·L(r0³+r0²r1+r0r1²+r1³)/4

        where c* is the slice centroid's offset from the box corner along
        each transverse axis (first moment of the corner-minus-quarter-disc
        region is r³(5/6−π/4))."""
        import math
        axes = {"x": 0, "y": 1, "z": 2}
        vbox = 1.0
        cb = [0.0, 0.0, 0.0]
        for c in range(3):
            vbox *= float(self.hi[c] - self.lo[c])
            cb[c] = float(self.lo[c] + self.hi[c]) / 2
        num = [vbox * cb[c] for c in range(3)]
        vtot = vbox
        ka = 1 - math.pi / 4
        kt = 5 / 6 - math.pi / 4
        for ax, sa, sb, r0f, r1f in self.edges:
            a = axes[ax]
            o1, o2 = [c for c in range(3) if c != a]
            L = float(self.hi[a] - self.lo[a])
            r0, r1 = float(r0f), float(r1f)
            i2 = L * (r0 * r0 + r0 * r1 + r1 * r1) / 3
            i2t = L * L * (r0 * r0 + 2 * r0 * r1 + 3 * r1 * r1) / 12
            i3 = L * (r0 ** 3 + r0 ** 2 * r1 + r0 * r1 ** 2 + r1 ** 3) / 4
            vrem = ka * i2
            vtot -= vrem
            num[a] -= ka * (i2 * float(self.lo[a]) + i2t)
            for o, side in ((o1, sa), (o2, sb)):
                corner = float(self.lo[o] if side == "min" else self.hi[o])
                inward = 1.0 if side == "min" else -1.0
                num[o] -= vrem * corner + inward * kt * i3
        return tuple(n / vtot for n in num)

    def bbox(self):
        return self.lo, self.hi

    def tessellate(self, deflection: float = 0.2) -> dict:
        raise NotImplementedError(
            "variable-taper fillet mesh needs the oblique-cone band (K5.3)")
