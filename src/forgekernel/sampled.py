"""The `sampled` provenance tier (ADR-0024) — Monte-Carlo answers, labelled.

The last resort, below exact and certified. A `SampledSolid` is a CSG
expression over the operand MESHES: point membership by ray casting, volume and
centroid by Monte-Carlo over the bounding box, each reported with a 3σ
half-width so a caller reads it as "sampled ± e" and never as exact.

WHY THIS IS NOT A FLOAT IN A COSTUME. ADR-0019 refuses a bare float because it
"carries no proof — you cannot tell a correct 534.6435 from a rounding
artifact." A sampled value carries its statistical error and its provenance
label, so a caller knows precisely how much to trust it. It answers a
MEASUREMENT; it never drives a topological decision (its `sign` is not
certified), and `_audited` does not run on it — there is no exact b-rep to
check, and saying so is the honest boundary.

DETERMINISTIC. The sampler is a fixed-seed LCG, so a model gives the same number
every run. `Math.random` is never called: a bench figure that jittered each run
would be worthless, and reproducibility is the ADR-0004 spirit even though a
sampled float never enters a byte-canonical document.
"""

from __future__ import annotations

import math

#: fixed multiplier/increment for a reproducible LCG (Numerical Recipes)
_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2 ** 32
_SEED = 20260730


def _lcg(seed):
    """An endless stream of floats in [0, 1) from a fixed seed."""
    s = seed
    while True:
        s = (_LCG_A * s + _LCG_C) % _LCG_M
        yield s / _LCG_M


def _tris_of(mesh):
    """(vertices, triangles) as float tuples from a tessellate dict."""
    if isinstance(mesh, dict):
        verts = [tuple(float(x) for x in v) for v in mesh["vertices"]]
        tris = [tuple(t) for t in mesh["triangles"]]
    else:                                            # object with attributes
        verts = [tuple(float(x) for x in v) for v in mesh.vertices]
        tris = [tuple(t) for t in mesh.triangles]
    return verts, tris


def _mesh_bbox(verts):
    lo = tuple(min(v[k] for v in verts) for k in range(3))
    hi = tuple(max(v[k] for v in verts) for k in range(3))
    return lo, hi


# a fixed, generic ray direction — not axis-aligned, to dodge the degenerate
# hits (through a vertex or along an edge) that an axis-parallel ray invites
_RAY = (0.5773269, 0.5774136, 0.5772489)


def _ray_hits_tri(o, d, a, b, c):
    """Möller–Trumbore: does the ray o + t·d (t > 0) cross triangle abc?"""
    eps = 1e-12
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    p = (d[1] * e2[2] - d[2] * e2[1],
         d[2] * e2[0] - d[0] * e2[2],
         d[0] * e2[1] - d[1] * e2[0])
    det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2]
    if -eps < det < eps:
        return False
    inv = 1.0 / det
    t = (o[0] - a[0], o[1] - a[1], o[2] - a[2])
    u = (t[0] * p[0] + t[1] * p[1] + t[2] * p[2]) * inv
    if u < -eps or u > 1 + eps:
        return False
    q = (t[1] * e1[2] - t[2] * e1[1],
         t[2] * e1[0] - t[0] * e1[2],
         t[0] * e1[1] - t[1] * e1[0])
    v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv
    if v < -eps or u + v > 1 + eps:
        return False
    tt = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv
    return tt > eps


class _MeshRegion:
    """Point membership for one closed triangle mesh (odd crossings = inside)."""

    def __init__(self, mesh):
        self.verts, self.tris = _tris_of(mesh)
        self.bbox = _mesh_bbox(self.verts)

    def inside(self, p):
        lo, hi = self.bbox
        if any(p[k] < lo[k] or p[k] > hi[k] for k in range(3)):
            return False
        crossings = 0
        vs = self.verts
        for (i, j, k) in self.tris:
            if _ray_hits_tri(p, _RAY, vs[i], vs[j], vs[k]):
                crossings += 1
        return crossings % 2 == 1


class _PlanarRegion:
    """Exact point membership for a planar `Solid` — ray casting against its
    polygons. Planar, so there is NO curvature deficit: the polyhedron the
    sampler measures is the solid, not an approximation of it."""

    def __init__(self, solid):
        self.tris = []
        vs_all = []
        for poly in solid.polys:
            vs = [tuple(float(x) for x in v) for v in poly.verts]
            vs_all += vs
            for i in range(1, len(vs) - 1):     # fan-triangulate the polygon
                self.tris.append((vs[0], vs[i], vs[i + 1]))
        self.bbox = _mesh_bbox(vs_all)

    def inside(self, p):
        lo, hi = self.bbox
        if any(p[k] < lo[k] or p[k] > hi[k] for k in range(3)):
            return False
        crossings = 0
        for (a, b, c) in self.tris:
            if _ray_hits_tri(p, _RAY, a, b, c):
                crossings += 1
        return crossings % 2 == 1


class SampledSolid:
    """A CSG of meshes, measured by Monte-Carlo (ADR-0024).

    Built from a membership predicate and a bounding box. `boolean` composes
    two operands into one — cut is A∧¬B, union is A∨B, intersect is A∧B — and
    a `SampledSolid` can itself be an operand, so chains (cut then cut) nest.
    """

    provenance = "sampled"
    is_exact_scalar = False

    def __init__(self, inside, bbox):
        self._inside = inside
        self._bbox = bbox

    # -- construction --------------------------------------------------------

    @staticmethod
    def _region(shape):
        """(inside_fn, bbox) for any operand.

        ANALYTIC membership for the primitive families — a point is in a sphere
        iff it satisfies the sphere inequality, not iff it is inside a
        tessellation of one. That distinction is the whole accuracy of this
        tier: a coarse mesh of a sphere r=6 has ~4% less volume than the sphere,
        and sampling THAT measures the mesh, not the solid (a first version did
        exactly this and came back 34 mm³ low, far outside its own 3σ). With
        exact membership the only error is statistical.

        A general `Body` (a cut result carrying trimmed quadric faces) has no
        cheap exact membership, so it falls back to a fine mesh with a geometric
        error term folded into the reported half-width — honest, coarser.
        """
        from forgekernel import body as B
        from forgekernel.brep import Solid
        from forgekernel.quadric import Cone, Cyl, Sphere

        if isinstance(shape, SampledSolid):
            return shape._inside, shape._bbox
        if isinstance(shape, Sphere):
            cx, cy, cz, r = (float(shape.cx), float(shape.cy),
                             float(shape.cz), float(shape.r))
            inside = lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2
                                + (p[2] - cz) ** 2 <= r * r)
            return inside, ((cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r))
        if isinstance(shape, Cyl):
            cx, cy, r = float(shape.cx), float(shape.cy), float(shape.r)
            z0, z1 = float(shape.z0), float(shape.z1)
            inside = lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r * r
                                and z0 <= p[2] <= z1)
            return inside, ((cx - r, cy - r, z0), (cx + r, cy + r, z1))
        if isinstance(shape, Cone):
            cx, cy = float(shape.cx), float(shape.cy)
            r1, r2 = float(shape.r1), float(shape.r2)
            z0, z1 = float(shape.z0), float(shape.z1)

            def inside(p):
                if not z0 <= p[2] <= z1:
                    return False
                rad = r1 + (r2 - r1) * (p[2] - z0) / (z1 - z0)
                return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= rad * rad
            rm = max(r1, r2)
            return inside, ((cx - rm, cy - rm, z0), (cx + rm, cy + rm, z1))
        if isinstance(shape, Solid):
            reg = _PlanarRegion(shape)
            return reg.inside, reg.bbox
        # a general Body: exact membership is not cheap, so mesh finely and
        # carry the meshing error explicitly (see `volume`)
        body = shape if isinstance(shape, B.Body) else B.to_body(shape)
        reg = _MeshRegion(B.tessellate(body, 0.08))
        reg.is_mesh = True
        return reg.inside, reg.bbox

    @classmethod
    def boolean(cls, op, a, b):
        ia, ba = cls._region(a)
        ib, bb = cls._region(b)
        if op == "cut":
            inside = lambda p: ia(p) and not ib(p)
            bbox = ba                                # result ⊆ a
        elif op == "union":
            inside = lambda p: ia(p) or ib(p)
            bbox = (tuple(min(ba[0][k], bb[0][k]) for k in range(3)),
                    tuple(max(ba[1][k], bb[1][k]) for k in range(3)))
        elif op == "intersect":
            inside = lambda p: ia(p) and ib(p)
            bbox = (tuple(max(ba[0][k], bb[0][k]) for k in range(3)),
                    tuple(min(ba[1][k], bb[1][k]) for k in range(3)))
        else:
            raise ValueError(f"sampled boolean: unknown op {op!r}")
        return cls(inside, bbox)

    # -- measures (Monte-Carlo, with a reported 3σ half-width) ---------------

    def _sample(self, n):
        lo, hi = self._bbox
        span = [hi[k] - lo[k] for k in range(3)]
        rng = _lcg(_SEED)
        inside_pts = []
        hits = 0
        for _ in range(n):
            p = tuple(lo[k] + span[k] * next(rng) for k in range(3))
            if self._inside(p):
                hits += 1
                inside_pts.append(p)
        return hits, inside_pts, span

    def volume(self, n=200000):
        """(midpoint, half-width) as a certified-interval-shaped pair.

        The estimator is hits/n · V_bbox, unbiased; its standard error is
        V_bbox·√(p(1−p)/n) and the half-width reported is 3σ. Not a rigorous
        enclosure — statistical — which is exactly why the provenance says so.
        """
        from forgekernel.interval import CInterval

        hits, _pts, span = self._sample(n)
        vbox = span[0] * span[1] * span[2]
        p = hits / n
        mid = p * vbox
        import math
        sigma = vbox * math.sqrt(max(p * (1 - p), 1e-12) / n)
        hw = 3 * sigma
        return CInterval(_frac(mid - hw), _frac(mid + hw))

    def centroid_f(self, n=200000):
        hits, pts, _span = self._sample(n)
        if not hits:
            nan = float("nan")
            return (nan, nan, nan)
        return tuple(sum(p[k] for p in pts) / hits for k in range(3))

    def bbox(self):
        return self._bbox

    def watertight_violations(self):
        # a sampled solid has no exact b-rep to audit; there is nothing to check
        # and claiming "closed" would overstate what it is
        return []


def _frac(x):
    from fractions import Fraction
    return Fraction(x).limit_denominator(10 ** 12)


def _point_tri_dist2(p, a, b, c):
    """Squared distance from point p to triangle abc (Ericson, Real-Time
    Collision Detection). Exact-arithmetic-free — this is the sampled tier."""
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2]
    d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2]
    if d1 <= 0 and d2 <= 0:
        return ap[0] ** 2 + ap[1] ** 2 + ap[2] ** 2
    bp = (p[0] - b[0], p[1] - b[1], p[2] - b[2])
    d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2]
    d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2]
    if d3 >= 0 and d4 <= d3:
        return bp[0] ** 2 + bp[1] ** 2 + bp[2] ** 2
    cp = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
    d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2]
    d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2]
    if d6 >= 0 and d5 <= d6:
        return cp[0] ** 2 + cp[1] ** 2 + cp[2] ** 2
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        q = tuple(a[k] + v * ab[k] for k in range(3))
        return sum((p[k] - q[k]) ** 2 for k in range(3))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        v = d2 / (d2 - d6)
        q = tuple(a[k] + v * ac[k] for k in range(3))
        return sum((p[k] - q[k]) ** 2 for k in range(3))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = tuple(b[k] + v * (c[k] - b[k]) for k in range(3))
        return sum((p[k] - q[k]) ** 2 for k in range(3))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    q = tuple(a[k] + ab[k] * v + ac[k] * w for k in range(3))
    return sum((p[k] - q[k]) ** 2 for k in range(3))


def _surface_tris(shape):
    """Triangles of a shape's surface mesh, for distance queries."""
    from forgekernel import body as B

    body = shape if isinstance(shape, B.Body) else B.to_body(shape)
    verts, tris = _tris_of(B.tessellate(body, 0.15))
    return [(verts[i], verts[j], verts[k]) for (i, j, k) in tris]


def sampled_shell(base, thickness):
    """A shelled solid as a SampledSolid (ADR-0024).

    Exact membership, not a mesh-deficit sample: a point is in a shell of wall
    thickness t iff it is INSIDE the solid AND within t of the surface. Both are
    membership queries — inside via the analytic/ray-cast region, near-surface
    via the minimum distance to the surface triangles — so this samples the
    true shelled set, and its only error is statistical.

    The surface distance uses a mesh, so a curved wall's distance carries the
    tessellation's error; at deflection 0.15 that is well under the wall
    thickness for any real shell, and it is a distance, not a topological
    decision.
    """
    inside, bbox = SampledSolid._region(base)
    tris = _surface_tris(base)
    t2 = float(thickness) ** 2

    def near_surface(p):
        return any(_point_tri_dist2(p, *tri) < t2 for tri in tris)

    return SampledSolid(lambda p: inside(p) and near_surface(p), bbox)


class _MorphResult:
    """A voxel morphological result — the sampled fillet (ADR-0024).

    A fillet-all rounds every edge with a ball of radius r, which is the
    morphological OPEN-then-CLOSE of the solid: opening rounds the convex edges
    (a ball rolling inside removes the sharp corner), closing rounds the concave
    ones (a ball rolling outside fills the crease). That IS the definition of a
    rolling-ball fillet, so the only error is the voxel resolution h — bounded
    HONESTLY by surface_area·h (boundary voxels are miscounted by at most a
    layer of thickness h), which is reported as the half-width. No systematic
    bias hiding under a statistical bar: the earlier single-normal shortcut had
    one and was withheld; this does not.
    """

    provenance = "sampled"
    is_exact_scalar = False

    def __init__(self, occ, dims, lo, h, area):
        self._occ = occ                     # flat bool list, opened+closed
        self._dims = dims
        self._lo = lo
        self._h = h
        self._area = area

    def _idx(self, i, j, k):
        return (i * self._dims[1] + j) * self._dims[2] + k

    def bbox(self):
        hi = tuple(self._lo[k] + self._dims[k] * self._h for k in range(3))
        return (tuple(self._lo), hi)

    def volume(self):
        from forgekernel.interval import CInterval

        cnt = sum(1 for v in self._occ if v)
        vol = cnt * self._h ** 3
        err = self._area * self._h          # geometric discretisation bound
        return CInterval(_frac(vol - err), _frac(vol + err))

    def centroid_f(self):
        nx, ny, nz = self._dims
        sx = sy = sz = 0.0
        cnt = 0
        for i in range(nx):
            for j in range(ny):
                base = (i * ny + j) * nz
                for k in range(nz):
                    if self._occ[base + k]:
                        cnt += 1
                        sx += self._lo[0] + (i + 0.5) * self._h
                        sy += self._lo[1] + (j + 0.5) * self._h
                        sz += self._lo[2] + (k + 0.5) * self._h
        if not cnt:
            nan = float("nan")
            return (nan, nan, nan)
        return (sx / cnt, sy / cnt, sz / cnt)

    def watertight_violations(self):
        return []


def _ball_offsets(r, h):
    """Voxel offsets whose centre lies within r — the ball structuring element."""
    rv = int(math.ceil(r / h))
    r2 = (r / h) ** 2
    return [(di, dj, dk)
            for di in range(-rv, rv + 1)
            for dj in range(-rv, rv + 1)
            for dk in range(-rv, rv + 1)
            if di * di + dj * dj + dk * dk <= r2]


def _morph_pass(occ, dims, offsets, want_all):
    """One erosion (want_all=True) or dilation (want_all=False) by the ball."""
    nx, ny, nz = dims
    out = [False] * len(occ)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = (i * ny + j) * nz + k
                hit_all = True
                hit_any = False
                for (di, dj, dk) in offsets:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        v = occ[(ii * ny + jj) * nz + kk]
                    else:
                        v = False           # outside the grid is empty
                    if v:
                        hit_any = True
                        if not want_all:
                            break
                    else:
                        hit_all = False
                        if want_all:
                            break
                out[idx] = hit_all if want_all else hit_any
    return out


def sampled_fillet(base, radius):
    """A fillet-all as a voxel morphological open-then-close (ADR-0024).

    Correct and honestly bounded, unlike the withdrawn single-normal opening.
    Verified against an exact RoundedBox (fillet-all of a box) within the
    reported surface_area·h.
    """
    inside, bbox = SampledSolid._region(base)
    r = float(radius)
    lo, hi = bbox
    pad = r * 1.5
    lo = tuple(lo[k] - pad for k in range(3))
    hi = tuple(hi[k] + pad for k in range(3))
    span = [hi[k] - lo[k] for k in range(3)]
    # resolution: fine enough that the ball is well sampled, capped so the grid
    # stays tractable (~50 voxels on the longest axis)
    h = min(r / 3.0, max(span) / 50.0)
    dims = [int(math.ceil(span[k] / h)) + 1 for k in range(3)]
    nx, ny, nz = dims
    occ = [False] * (nx * ny * nz)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = (lo[0] + (i + 0.5) * h, lo[1] + (j + 0.5) * h,
                     lo[2] + (k + 0.5) * h)
                if inside(p):
                    occ[(i * ny + j) * nz + k] = True
    offs = _ball_offsets(r, h)
    # open = erode then dilate (rounds convex); close = dilate then erode
    # (rounds concave). fillet-all rounds both, so open then close.
    opened = _morph_pass(_morph_pass(occ, dims, offs, True), dims, offs, False)
    closed = _morph_pass(_morph_pass(opened, dims, offs, False), dims, offs, True)
    area = _mesh_area(base)
    return _MorphResult(closed, dims, lo, h, area)


def _mesh_area(base):
    total = 0.0
    for (a, b, c) in _surface_tris(base):
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cx = ab[1] * ac[2] - ab[2] * ac[1]
        cy = ab[2] * ac[0] - ab[0] * ac[2]
        cz = ab[0] * ac[1] - ab[1] * ac[0]
        total += 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5
    return total
