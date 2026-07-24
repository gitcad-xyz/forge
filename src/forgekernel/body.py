"""The canonical B-rep (ADR-0021) — one topology every representation maps to.

forge grew a separate solid class per feature, so every operation had to be
written once per representation: O(ops × representations), a matrix that could
never be filled. ``Body`` is the single form each of those classes converts
into, so an operation is written ONCE.

    Body  = [Face]
    Face  = (Surface, [Loop], sense)   # sense: outward normal == surface normal
    Loop  = [Edge]                     # loops[0] is the outer bound, rest holes
    Edge  = (Curve, v0, v1)            # v0 == v1 for a full circle

Surfaces and curves are ANALYTIC and EXACT — a hole is a ``Cylinder`` with a
rational radius, never a polygon fan. Nothing here approximates: mass properties
come from the divergence theorem applied per face in closed form (ℚ or ℚ[π]),
and an operation that would leave the number field refuses with its stage
(ADR-0019). Tessellation is the one deliberate exception, and only because
meshing is a display property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from forgekernel.exact import F, cross, dot, sub

Vec = tuple


# -- surfaces ----------------------------------------------------------------

@dataclass(frozen=True)
class Plane:
    """``n · x = d``; ``n`` need not be normalised (exactness over unit length)."""
    n: Vec
    d: Fraction

    def transformed(self, m):
        # A plane's normal transforms by the INVERSE-TRANSPOSE, not by the map
        # itself. Rigid maps make the two agree and a uniform scale makes them
        # parallel (all a plane needs), but under a NON-UNIFORM scale they
        # diverge: a slanted face would keep a normal no longer perpendicular to
        # it. m.normal() applies the cofactor matrix = det.M^-T — exact in Q, and
        # a parallel normal suffices because d rescales with it.
        p = m.point(_plane_point(self))
        n = m.normal(self.n)
        return Plane(n, dot(n, p))


@dataclass(frozen=True)
class Cylinder:
    """Radius ``r`` about the axis through ``p`` along ``d``."""
    p: Vec
    d: Vec
    r: Fraction

    def transformed(self, m):
        return Cylinder(m.point(self.p), m.direction(self.d), m.scale_len(self.r))


@dataclass(frozen=True)
class Cone:
    """Half-angle given exactly as ``tan = rise/run``; apex at ``p``, axis ``d``."""
    p: Vec
    d: Vec
    tan_half: Fraction

    def transformed(self, m):
        return Cone(m.point(self.p), m.direction(self.d), self.tan_half)


@dataclass(frozen=True)
class SphereS:
    c: Vec
    r: Fraction

    def transformed(self, m):
        return SphereS(m.point(self.c), m.scale_len(self.r))


# -- curves ------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    p: Vec
    d: Vec

    def transformed(self, m):
        return Line(m.point(self.p), m.direction(self.d))


@dataclass(frozen=True)
class Circle:
    c: Vec
    n: Vec          # plane normal
    ref: Vec        # zero-angle direction, in the plane
    r: Fraction

    def transformed(self, m):
        return Circle(m.point(self.c), m.direction(self.n),
                      m.direction(self.ref), m.scale_len(self.r))


# -- topology ----------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    curve: object
    v0: Vec
    v1: Vec

    def transformed(self, m):
        return Edge(self.curve.transformed(m), m.point(self.v0), m.point(self.v1))


@dataclass(frozen=True)
class Loop:
    edges: tuple

    def transformed(self, m):
        out = tuple(e.transformed(m) for e in self.edges)
        if m.flips:
            # An orientation-reversing map (mirror, negative scale) reverses the
            # winding of the vertex list while the surface normal is mirrored
            # too — so the loop must be re-reversed to stay consistent with the
            # outward normal. Reversing the WINDING is the truthful fix; flipping
            # the face sense instead would claim the mirrored solid is inside-out.
            out = tuple(Edge(e.curve, e.v1, e.v0) for e in reversed(out))
        return Loop(out)


@dataclass(frozen=True)
class Face:
    surface: object
    loops: tuple
    sense: bool = True

    def transformed(self, m):
        return Face(self.surface.transformed(m),
                    tuple(lp.transformed(m) for lp in self.loops), self.sense)


@dataclass(frozen=True)
class Body:
    faces: tuple

    def transformed(self, m):
        return Body(tuple(f.transformed(m) for f in self.faces))


# -- exact affine maps -------------------------------------------------------

class Affine:
    """An exact affine map: 3x3 matrix + translation, over ℚ or ℚ[√d].

    ``flips`` records whether the map reverses orientation (a mirror or a
    negative scale), so faces can flip their sense and stay outward-oriented.
    ``scale_len`` is how a radius transforms — defined only for maps that are
    conformal (rigid or uniform scale); a non-uniform scale would turn a circle
    into an ellipse, which is not in the canonical surface set, so it refuses.
    """

    def __init__(self, rows, t=(0, 0, 0), *, uniform=None, flips=False):
        self.rows = tuple(tuple(r) for r in rows)
        self.t = tuple(t)
        self.uniform = uniform          # the scalar radii scale by, or None
        self.flips = flips

    @classmethod
    def identity(cls):
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)), uniform=F(1))

    @classmethod
    def translation(cls, x, y, z):
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (F(x), F(y), F(z)),
                   uniform=F(1))

    @classmethod
    def scaling(cls, fx, fy, fz):
        fx, fy, fz = F(fx), F(fy), F(fz)
        if fx == 0 or fy == 0 or fz == 0:
            raise ValueError(
                "a zero scale factor collapses the solid — not an invertible "
                "transform")
        uniform = abs(fx) if (abs(fx) == abs(fy) == abs(fz)) else None
        neg = (fx < 0) + (fy < 0) + (fz < 0)
        return cls(((fx, 0, 0), (0, fy, 0), (0, 0, fz)),
                   uniform=uniform, flips=bool(neg % 2))

    @classmethod
    def mirror(cls, axis: str):
        i = "xyz".index(axis)
        rows = [[1 if a == b else 0 for b in range(3)] for a in range(3)]
        rows[i][i] = -1
        return cls(rows, uniform=F(1), flips=True)

    @classmethod
    def rotation(cls, axis, deg):
        from forgekernel.kernel import _rotation_matrix

        return cls(_rotation_matrix(axis, deg), uniform=F(1), flips=False)

    def point(self, v):
        r = self.rows
        return tuple(r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] + self.t[i]
                     for i in range(3))

    def direction(self, v):
        r = self.rows
        return tuple(r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2]
                     for i in range(3))

    def normal(self, v):
        """Map a SURFACE NORMAL — by the cofactor matrix (det.M^-T), so it stays
        perpendicular to the transformed surface even under a non-uniform scale.
        Exact: cofactors are products of matrix entries, no division."""
        r = self.rows
        c = [[r[(i + 1) % 3][(j + 1) % 3] * r[(i + 2) % 3][(j + 2) % 3]
              - r[(i + 1) % 3][(j + 2) % 3] * r[(i + 2) % 3][(j + 1) % 3]
              for j in range(3)] for i in range(3)]
        out = tuple(c[i][0] * v[0] + c[i][1] * v[1] + c[i][2] * v[2]
                    for i in range(3))
        # The cofactor matrix is det·M⁻ᵀ, so an ORIENTATION-REVERSING map (a
        # mirror, det < 0) would hand back an inward-pointing normal. Divide the
        # sign of det back out: |det|·M⁻ᵀ keeps the normal outward, and any
        # positive multiple of M⁻ᵀ is an equally valid plane normal.
        det = sum(self.rows[0][j] * c[0][j] for j in range(3))
        return tuple(-x for x in out) if det < 0 else out

    def scale_len(self, r):
        if self.uniform is None:
            raise ValueError(
                "non-uniform scale turns a circle into an ellipse — not in the "
                "canonical surface set (elliptical surfaces arrive with K3.7)")
        return self.uniform * r


def _plane_point(pl: Plane) -> Vec:
    """Any exact point on the plane: the foot of the normal from the origin."""
    nn = dot(pl.n, pl.n)
    return tuple(pl.n[i] * pl.d / nn for i in range(3))


# -- exact metrics -----------------------------------------------------------
#
# Volume by the divergence theorem, V = (1/3)∮ x·n dA, evaluated per face in
# CLOSED FORM. Each surface type contributes a rational or ℚ[π] term, so the
# total is exact — no sampling, no facets.

def _loop_is_circle(loop: Loop):
    if len(loop.edges) == 1 and isinstance(loop.edges[0].curve, Circle):
        return loop.edges[0].curve
    if (len(loop.edges) == 2
            and all(isinstance(e.curve, Circle) for e in loop.edges)
            and loop.edges[0].curve == loop.edges[1].curve):
        return loop.edges[0].curve          # split into two half-circles
    return None


def _planar_loop_area2(loop: Loop, n: Vec):
    """Twice the signed area of a polygonal loop, projected along n (exact)."""
    pts = [e.v0 for e in loop.edges]
    acc = (F(0), F(0), F(0))
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        c = cross(a, b)
        acc = (acc[0] + c[0], acc[1] + c[1], acc[2] + c[2])
    return dot(acc, n)


def _as_fraction(x):
    """x as a Fraction if it is rational — including a ``SurdVal`` whose surd
    part is zero, which is what an exact rotation leaves behind (a 45° turn
    types every coordinate as ℚ[√2] even where the value is rational)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    b = getattr(x, "b", None)
    if b is not None and b == 0:
        return x.a
    return None


def _exact(x, what: str) -> Fraction:
    """Coerce an exact value to ℚ, refusing if it genuinely carries a surd —
    a volume outside ℚ[π] is not representable and must not be floated."""
    f = _as_fraction(x)
    if f is None:
        raise ValueError(
            f"{what} volume term left ℚ[π] ({x!r}) — a bigger number field "
            "arrives with K3.7")
    return f


def _rational_sqrt(x):
    """√x as a Fraction when it is one, else None (the caller then refuses)."""
    from math import isqrt

    x = _as_fraction(x)
    if x is None or x < 0:
        return None
    n, d = x.numerator, x.denominator
    rn, rd = isqrt(n), isqrt(d)
    return Fraction(rn, rd) if rn * rn == n and rd * rd == d else None


def volume(body: Body):
    """Exact signed volume by the divergence theorem, V = (1/3)∮ x·n̂ dA,
    evaluated per face in CLOSED FORM. Returns a ``PiVal`` (ℚ + ℚ·π)."""
    from forgekernel.quadric import PiVal

    total = PiVal(0, 0)
    for face in body.faces:
        total = total + _face_volume_term(face)
    return total


def _face_volume_term(face: Face):
    """(1/3)∮ x·n̂ dA over one face — exact, per surface type."""
    from forgekernel.quadric import PiVal

    s = face.surface
    sgn = 1 if face.sense else -1
    if isinstance(s, Plane):
        # x·n̂ = d/|n| is CONSTANT on the plane, so the term is (1/3)(d/|n|)·Area.
        # For a polygon loop, Area = ((Σ vᵢ × vᵢ₊₁)·n)/(2|n|), and the two |n|
        # combine into n·n — entirely rational, no square root needed.
        nn = dot(s.n, s.n)
        rat, pi = F(0), F(0)
        area_rat, area_pi = F(0), F(0)     # for the sanity check below
        for i, lp in enumerate(face.loops):
            circ = _loop_is_circle(lp)
            if circ is None:
                a2 = _planar_loop_area2(lp, s.n)
                rat += s.d * (a2 if i == 0 else -abs(a2)) / (6 * nn)
                ln2 = _rational_sqrt(nn)
                if ln2 is not None:
                    area_rat += (a2 if i == 0 else -abs(a2)) / (2 * ln2)
            else:
                # a circular loop has area π r², which needs |n| on its own
                ln = _rational_sqrt(nn)
                if ln is None:
                    raise ValueError(
                        "circular loop on a plane whose normal has irrational "
                        "length — outside ℚ[π] (arrives with K3.7)")
                pi += (1 if i == 0 else -1) * s.d * circ.r * circ.r / (3 * ln)
                area_pi += (1 if i == 0 else -1) * circ.r * circ.r
        # A face cannot have negative area. If it does, an inner loop was
        # attached to a face that does not contain it — refuse rather than
        # return a confidently wrong exact volume.
        if float(_as_fraction(area_rat) or 0) + math.pi * float(
                _as_fraction(area_pi) or 0) < -1e-12:
            raise ValueError(
                "face has negative area — an inner loop is larger than the "
                "boundary carrying it (a hole attached to the wrong face)")
        return PiVal(_exact(sgn * rat, "planar area"),
                     _exact(sgn * pi, "circular area"))
    if isinstance(s, Cylinder):
        # Over a full band of radius r and axial height h, ∮ x·û dθ drops the
        # axis-offset term (∫û dθ = 0), leaving (1/3)(2π r² h).
        h = _band_height(face, s)
        return PiVal(0, _exact(sgn * 2 * s.r * s.r * h / 3, "cylinder band"))
    if isinstance(s, SphereS):
        # x·n̂ = r + c·n̂, and ∮ c·n̂ dA = 0 over the whole sphere: (1/3)r·4πr².
        return PiVal(0, _exact(sgn * 4 * s.r ** 3 / 3, "sphere"))
    raise ValueError(
        f"no exact volume term for {type(s).__name__} yet (K3.7)")


def _band_height(face: Face, cyl: Cylinder) -> Fraction:
    """Axial extent of a cylindrical face, from its two circular rim loops."""
    ts = []
    for lp in face.loops:
        for e in lp.edges:
            if isinstance(e.curve, Circle):
                ts.append(dot(sub(e.curve.c, cyl.p), cyl.d))
    if len(ts) < 2:
        raise ValueError("cylindrical face without two circular rims (K3.7)")
    # t = (c−p)·d is the axial offset scaled by |d| (NOT |d|²), so the true
    # height divides by |d| once. A non-unit axis arises after scaling.
    ln = _rational_sqrt(dot(cyl.d, cyl.d))
    if ln is None:
        raise ValueError(
            "cylinder axis with irrational length — outside ℚ[π] (K3.7)")
    return (max(ts) - min(ts)) / ln


def bbox(body: Body):
    """Float bbox of the body (a bound, not a topological decision)."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for f in body.faces:
        if isinstance(f.surface, SphereS):      # a whole sphere carries no loops
            c, r = f.surface.c, float(f.surface.r)
            for i in range(3):
                lo[i] = min(lo[i], float(c[i]) - r)
                hi[i] = max(hi[i], float(c[i]) + r)
            continue
        for lp in f.loops:
            circ = _loop_is_circle(lp)
            if circ is not None:
                c, r = circ.c, float(circ.r)
                # a circle's extent per axis: r·sqrt(1 − (n_i/|n|)²)
                n = circ.n
                nn = math.sqrt(float(dot(n, n))) or 1.0
                for i in range(3):
                    ext = r * math.sqrt(max(0.0, 1 - (float(n[i]) / nn) ** 2))
                    lo[i] = min(lo[i], float(c[i]) - ext)
                    hi[i] = max(hi[i], float(c[i]) + ext)
                continue
            for e in lp.edges:
                for v in (e.v0, e.v1):
                    for i in range(3):
                        lo[i] = min(lo[i], float(v[i]))
                        hi[i] = max(hi[i], float(v[i]))
    return (tuple(lo), tuple(hi))


def tessellate(body: Body, deflection: float = 0.2) -> dict:
    """One display mesh for the canonical form — planar faces are triangulated
    (with circular holes cut out), cylindrical faces become bands, spheres a UV
    mesh. Floats are legal here: meshing is a display property (ADR-0019)."""
    from forgekernel.mesh2d import triangulate

    verts: list = []
    tris: list = []
    index: dict = {}

    def V(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in index:
            index[k] = len(verts)
            verts.append([float(p[0]), float(p[1]), float(p[2])])
        return index[k]

    def tri(a, b, c, outward):
        ia, ib, ic = V(a), V(b), V(c)
        if len({ia, ib, ic}) < 3:
            return
        n = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
             (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
             (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if sum(n[i] * outward[i] for i in range(3)) < 0:
            ib, ic = ic, ib
        tris.append([ia, ib, ic])

    def circle_pts(c: Circle, n=None):
        r = float(c.r)
        segs = n or max(24, int(math.ceil(
            math.pi / math.acos(max(-1.0, 1 - deflection / r))))) if r > deflection else 24
        u = _unit(tuple(float(x) for x in c.ref))
        w = _unit(_cross_f(tuple(float(x) for x in c.n), u))
        cc = tuple(float(x) for x in c.c)
        return [tuple(cc[i] + r * (math.cos(2 * math.pi * k / segs) * u[i]
                                   + math.sin(2 * math.pi * k / segs) * w[i])
                      for i in range(3)) for k in range(segs)], segs

    for face in body.faces:
        s = face.surface
        if isinstance(s, Plane):
            nf = _unit(tuple(float(x) for x in s.n))
            out = nf if face.sense else tuple(-x for x in nf)
            u = _unit(_perp_f(nf))
            w = _cross_f(nf, u)
            org = tuple(float(x) for x in _plane_point(s))

            def to2(p):
                q = tuple(p[i] - org[i] for i in range(3))
                return (sum(q[i] * u[i] for i in range(3)),
                        sum(q[i] * w[i] for i in range(3)))

            def to3(a):
                return tuple(org[i] + a[0] * u[i] + a[1] * w[i] for i in range(3))

            rings = []
            for lp in face.loops:
                circ = _loop_is_circle(lp)
                if circ is not None:
                    pts, _ = circle_pts(circ)
                    rings.append([to2(p) for p in pts])
                else:
                    rings.append([to2(tuple(float(x) for x in e.v0))
                                  for e in lp.edges])
            if not rings:
                continue
            pts2, t2 = triangulate(rings[0], rings[1:])
            for a, b, c in t2:
                tri(to3(pts2[a]), to3(pts2[b]), to3(pts2[c]), out)
        elif isinstance(s, Cylinder):
            circs = [e.curve for lp in face.loops for e in lp.edges
                     if isinstance(e.curve, Circle)]
            if len(circs) < 2:
                continue
            lo, hi = circs[0], circs[-1]
            pa, segs = circle_pts(lo)
            pb, _ = circle_pts(hi, segs)
            ax = tuple(float(x) for x in s.p)
            for i in range(segs):
                j = (i + 1) % segs
                mid = tuple((pa[i][t] + pa[j][t]) / 2 for t in range(3))
                radial = _unit(tuple(mid[t] - ax[t] for t in range(3)))
                out = radial if face.sense else tuple(-x for x in radial)
                tri(pa[i], pa[j], pb[j], out)
                tri(pa[i], pb[j], pb[i], out)
        elif isinstance(s, SphereS):
            c = tuple(float(x) for x in s.c)
            r = float(s.r)
            seg = max(8, int(math.ceil(math.pi / math.acos(
                max(-1.0, 1 - deflection / r))))) if r > deflection else 8
            nlat, nlon = max(4, seg), max(8, 2 * seg)
            for i in range(nlat):
                t0, t1 = math.pi * i / nlat, math.pi * (i + 1) / nlat
                for j in range(nlon):
                    p0, p1 = 2 * math.pi * j / nlon, 2 * math.pi * (j + 1) / nlon
                    q = [(c[0] + r * math.sin(t) * math.cos(p),
                          c[1] + r * math.sin(t) * math.sin(p),
                          c[2] + r * math.cos(t))
                         for t, p in ((t0, p0), (t0, p1), (t1, p1), (t1, p0))]
                    mid = tuple(sum(x[k] for x in q) / 4 - c[k] for k in range(3))
                    out = _unit(mid) if face.sense else tuple(-x for x in _unit(mid))
                    tri(q[0], q[1], q[2], out)
                    tri(q[0], q[2], q[3], out)
    return {"vertices": verts, "triangles": tris}


def _unit(v):
    m = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / m for x in v)


def _cross_f(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _perp_f(n):
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    d = sum(a[i] * n[i] for i in range(3))
    return _unit(tuple(a[i] - d * n[i] for i in range(3)))


def edges_info(body: Body) -> list[dict]:
    """Descriptors for every distinct edge — the selection surface that
    fillet/chamfer targeting and dimension anchoring read. ONE implementation
    covering every representation, because they all become a Body."""
    out, seen = [], set()
    for f in body.faces:
        for lp in f.loops:
            for e in lp.edges:
                if isinstance(e.curve, Circle):
                    c = e.curve
                    key = ("circle", tuple(float(x) for x in c.c),
                           tuple(float(x) for x in c.n), float(c.r))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"curve": "circle", "radius": float(c.r),
                                "centroid": [float(x) for x in c.c],
                                "length": 2 * math.pi * float(c.r),
                                "axis": list(_unit(tuple(float(y) for y in c.n)))})
                else:
                    a = tuple(float(x) for x in e.v0)
                    b = tuple(float(x) for x in e.v1)
                    key = ("line",) + tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key)
                    d = tuple(b[i] - a[i] for i in range(3))
                    out.append({"curve": "line", "point": list(a), "dir": list(d),
                                "centroid": [(a[i] + b[i]) / 2 for i in range(3)],
                                "length": math.sqrt(sum(x * x for x in d))})
    return out


def _poly_area_centroid_f(vs, n):
    """Signed area (about n) and area centroid of a 3D planar polygon."""
    acc, area = [0.0, 0.0, 0.0], 0.0
    v0 = vs[0]
    for i in range(1, len(vs) - 1):
        a = [vs[i][k] - v0[k] for k in range(3)]
        b = [vs[i + 1][k] - v0[k] for k in range(3)]
        cr = _cross_f(a, b)
        ta = 0.5 * sum(cr[k] * n[k] for k in range(3))
        tc = [(v0[k] + vs[i][k] + vs[i + 1][k]) / 3 for k in range(3)]
        for k in range(3):
            acc[k] += tc[k] * ta
        area += ta
    if area == 0:
        m = len(vs)
        return 0.0, tuple(sum(v[k] for v in vs) / m for k in range(3))
    return area, tuple(acc[k] / area for k in range(3))


def faces_info(body: Body) -> list[dict]:
    """Descriptors for every face — surface kind plus the ANALYTIC parameters a
    caller needs (a bore reports its radius and axis, never a facet count)."""
    out = []
    for f in body.faces:
        s = f.surface
        if isinstance(s, Plane):
            nf = _unit(tuple(float(x) for x in s.n))
            pts, area = [], 0.0
            for i, lp in enumerate(f.loops):
                circ = _loop_is_circle(lp)
                if circ is not None:
                    a = math.pi * float(circ.r) ** 2
                    area += a if i == 0 else -a
                    if i == 0:
                        pts.append((tuple(float(x) for x in circ.c), a))
                else:
                    vs = [tuple(float(x) for x in e.v0) for e in lp.edges]
                    a, c = _poly_area_centroid_f(vs, nf)
                    area += a if i == 0 else -abs(a)
                    if i == 0:
                        pts.append((c, abs(a)))
            tot = sum(w for _, w in pts) or 1.0
            cen = [sum(p[i] * w for p, w in pts) / tot for i in range(3)]
            out.append({"surface": "plane", "plane": list(nf),
                        "centroid": cen, "area": abs(area)})
        elif isinstance(s, Cylinder):
            try:
                h = float(_band_height(f, s))
            except ValueError:
                h = 0.0
            axis = _unit(tuple(float(x) for x in s.d))
            base = tuple(float(x) for x in s.p)
            out.append({"surface": "cylinder", "radius": float(s.r),
                        "axis_dir": list(axis), "axis_origin": list(base),
                        "area": 2 * math.pi * float(s.r) * h,
                        "centroid": [base[i] + axis[i] * h / 2 for i in range(3)]})
        elif isinstance(s, SphereS):
            out.append({"surface": "sphere", "radius": float(s.r),
                        "centroid": [float(x) for x in s.c],
                        "area": 4 * math.pi * float(s.r) ** 2})
        else:
            out.append({"surface": type(s).__name__.lower()})
    return out


# -- converters: every representation becomes a Body -------------------------

def from_solid(solid) -> Body:
    """A planar forge ``Solid`` — each polygon becomes a planar face."""
    faces = []
    for p in solid.polys:
        vs = [tuple(v) for v in p.verts]
        edges = tuple(Edge(Line(vs[i], sub(vs[(i + 1) % len(vs)], vs[i])),
                           vs[i], vs[(i + 1) % len(vs)])
                      for i in range(len(vs)))
        faces.append(Face(Plane(p.plane.n, p.plane.d), (Loop(edges),), True))
    return Body(tuple(faces))


def _circle_at(cx, cy, z, r) -> Circle:
    return Circle((F(cx), F(cy), F(z)), (F(0), F(0), F(1)),
                  (F(1), F(0), F(0)), F(r))


def _disk_face(cx, cy, z, r, up: bool) -> Face:
    c = _circle_at(cx, cy, z, r)
    v = (F(cx) + F(r), F(cy), F(z))
    return Face(Plane((F(0), F(0), F(1)), F(z)), (Loop((Edge(c, v, v),)),), up)


def from_cyl(c) -> Body:
    """A z-axis cylinder: two disk caps and one cylindrical wall."""
    from forgekernel.quadric import Cyl as _C

    c = _C(F(c.cx), F(c.cy), F(c.r), F(c.z0), F(c.z1))
    wall_lo = _circle_at(c.cx, c.cy, c.z0, c.r)
    wall_hi = _circle_at(c.cx, c.cy, c.z1, c.r)
    v0 = (c.cx + c.r, c.cy, c.z0)
    v1 = (c.cx + c.r, c.cy, c.z1)
    wall = Face(Cylinder((c.cx, c.cy, c.z0), (F(0), F(0), F(1)), c.r),
                (Loop((Edge(wall_lo, v0, v0), Edge(wall_hi, v1, v1))),), True)
    return Body((_disk_face(c.cx, c.cy, c.z1, c.r, True),
                 _disk_face(c.cx, c.cy, c.z0, c.r, False),
                 wall))


def _face_contains_xy(face: "Face", px, py) -> bool:
    """Exact: does (px, py) lie inside this face's OUTER loop, in xy?

    A cap is often split into several coplanar fragments — ear-clipped prism
    caps, boolean-split tops — so matching a bore to a cap by z alone attaches
    its hole to every fragment at that height, including ones the bore never
    touches. (quadric.DrilledSolid.cut already carries the whole-solid version
    of this predicate, _xy_inside_footprint; this is the per-face analogue.)"""
    pts = [e.v0 for e in face.loops[0].edges]
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        if (y1 > py) != (y2 > py):
            if px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def from_drilled(d) -> Body:
    """A planar base minus z-cylindrical bores: cap faces gain circular inner
    loops, each bore contributes a cylindrical wall (and a disk if blind)."""
    body = from_solid(d.base)
    (_, _, bz0), (_, _, bz1) = d.base.bbox()
    faces = list(body.faces)
    from collections import defaultdict

    from forgekernel.quadric import Cyl as _C

    # Coaxial bores (a counterbore stack) are ONE stepped void, not several
    # independent ones: emitting a hole per bore double-subtracts their overlap
    # on the shared cap and leaves the inner wall running through the wider
    # bore's empty space. Merge them into z-bands carrying the OUTERMOST radius,
    # exactly as the section and tessellation paths do.
    groups: dict = defaultdict(list)
    for c in d.bores:
        c = _C(F(c.cx), F(c.cy), F(c.r), F(c.z0), F(c.z1))
        z0, z1 = max(c.z0, F(bz0)), min(c.z1, F(bz1))
        if z1 > z0:
            groups[(c.cx, c.cy)].append(_C(c.cx, c.cy, c.r, z0, z1))

    cap_faces = [i for i, f in enumerate(faces)
                 if isinstance(f.surface, Plane)
                 and f.surface.n[0] == 0 and f.surface.n[1] == 0]
    for (cx, cy), cyls in groups.items():
        zs = sorted({z for c in cyls for z in (c.z0, c.z1)})
        bands = []
        for za, zb in zip(zs, zs[1:]):
            zmid = (za + zb) / 2
            rs = [c.r for c in cyls if c.z0 <= zmid <= c.z1]
            if rs:
                bands.append((za, zb, max(rs)))
        if not bands:
            continue
        zlo, zhi = bands[0][0], bands[-1][1]
        # cap holes: only on a fragment this stack actually passes through
        for i in cap_faces:
            f = faces[i]
            zc = f.loops[0].edges[0].v0[2]
            r = (bands[-1][2] if zc == zhi else
                 bands[0][2] if zc == zlo else None)
            if r is None or not _face_contains_xy(f, cx, cy):
                continue
            circ = _circle_at(cx, cy, zc, r)
            v = (cx + r, cy, zc)
            faces[i] = Face(f.surface, f.loops + (Loop((Edge(circ, v, v),)),),
                            f.sense)
        for za, zb, r in bands:                       # one wall per band
            lo_c, hi_c = _circle_at(cx, cy, za, r), _circle_at(cx, cy, zb, r)
            a, b = (cx + r, cy, za), (cx + r, cy, zb)
            faces.append(Face(Cylinder((cx, cy, za), (F(0), F(0), F(1)), r),
                              (Loop((Edge(lo_c, a, a), Edge(hi_c, b, b))),),
                              False))
        for (za, zb, r0), (zb2, zc, r1) in zip(bands, bands[1:]):
            if r0 == r1:
                continue                              # no step: no shoulder
            rin, rout = min(r0, r1), max(r0, r1)
            # the exposed ring faces INTO the wider bore
            up = r1 > r0
            outer = _circle_at(cx, cy, zb, rout)
            inner = _circle_at(cx, cy, zb, rin)
            vo, vi = (cx + rout, cy, zb), (cx + rin, cy, zb)
            faces.append(Face(Plane((F(0), F(0), F(1)), zb),
                              (Loop((Edge(outer, vo, vo),)),
                               Loop((Edge(inner, vi, vi),))), up))
        if zlo > F(bz0):                              # blind at the bottom
            faces.append(_disk_face(cx, cy, zlo, bands[0][2], True))
        if zhi < F(bz1):                              # blind at the top
            faces.append(_disk_face(cx, cy, zhi, bands[-1][2], False))
    return Body(tuple(faces))


def from_sphere(s) -> Body:
    # coerce through F(): the quadric dataclasses annotate Fraction but do not
    # enforce it, so a directly-constructed Sphere(0,0,0,6) holds ints and
    # 4·r³/3 would silently become a FLOAT — the one thing the charter forbids.
    return Body((Face(SphereS((F(s.cx), F(s.cy), F(s.cz)), F(s.r)), (), True),))


def to_body(shape) -> Body:
    """Convert any forge representation to the canonical B-rep, or raise."""
    from forgekernel.brep import Solid
    from forgekernel.quadric import Cyl, DisjointUnion, DrilledSolid, Sphere

    if isinstance(shape, Body):
        return shape
    if isinstance(shape, Solid):
        return from_solid(shape)
    if isinstance(shape, Cyl):
        return from_cyl(shape)
    if isinstance(shape, DrilledSolid):
        return from_drilled(shape)
    if isinstance(shape, Sphere):
        return from_sphere(shape)
    if isinstance(shape, DisjointUnion):
        faces = []
        for m in shape.members:
            faces.extend(to_body(m).faces)
        return Body(tuple(faces))
    raise ValueError(
        f"no canonical-B-rep converter for {type(shape).__name__} yet "
        "(ADR-0021 converters land per representation)")
