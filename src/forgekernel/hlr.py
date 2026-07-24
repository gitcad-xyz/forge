"""Native hidden-line removal for 2D orthographic drawings (ADR-0020).

Visibility is a *display* property, not a topological decision, so — under the
exactness charter (ADR-0019) — this layer may use floats; the geometry it draws
is the exact solid. For a view direction it returns two lists of 2D polylines,
``visible`` and ``hidden``, in the sheet frame the drawing engine consumes:
the x-axis is ``xdir``, the y-axis is ``direction × xdir``, the viewer looks
along +``direction`` (so smaller ``dot(P, direction)`` is closer to the viewer).

Approach: extract the solid's real edges (exact straight edges for planar
solids; sampled circles + silhouettes for cylinders; sharp + silhouette edges
from the tessellation for everything else), project them, then split each into
visible/hidden spans by asking, at sample midpoints, whether a hair toward the
viewer lands *inside* the solid (an inside classifier that respects holes).
"""

from __future__ import annotations

import math

_EPS = 1e-7
# a fixed, non-axis-aligned ray for inside/parity tests — dodges the coplanar
# and edge-grazing degeneracies that axis-aligned rays hit on boxy models.
_RAY = (0.41931, 0.77913, 0.46671)


def _f3(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(v):
    m = math.sqrt(_dot(v, v)) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


class _View:
    def __init__(self, direction, xdir):
        self.look = _norm(_f3(direction))
        self.right = _norm(_f3(xdir))
        # sheet-up = direction × xdir (the drawing engine's convention)
        self.up = _norm(_cross(self.look, self.right))

    def xy(self, p):
        return (_dot(p, self.right), _dot(p, self.up))


# -- inside/occlusion classifiers ---------------------------------------------

def _ray_hits_convex(orig, dirn, poly, n):
    """t>0 where the ray orig+t·dirn crosses the convex polygon (with plane
    normal n), or None. Point-in-polygon via consistent edge cross-signs."""
    denom = _dot(dirn, n)
    if abs(denom) < 1e-12:
        return None
    t = _dot(_sub(poly[0], orig), n) / denom
    if t <= _EPS:
        return None
    h = _add(orig, _mul(dirn, t))
    m = len(poly)
    sign = 0
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        c = _dot(_cross(_sub(b, a), _sub(h, a)), n)
        if c > 1e-12:
            s = 1
        elif c < -1e-12:
            s = -1
        else:
            continue
        if sign == 0:
            sign = s
        elif s != sign:
            return None
    return t


def _parity_inside(polys_n, q):
    """Odd crossings of the fixed ray ⇒ q is inside a closed polygon shell."""
    c = 0
    for poly, n in polys_n:
        if _ray_hits_convex(q, _RAY, poly, n) is not None:
            c += 1
    return c % 2 == 1


def _solid_polys_n(solid):
    out = []
    for p in solid.polys:
        v = [_f3(x) for x in p.verts]
        out.append((v, _cross(_sub(v[1], v[0]), _sub(v[2], v[0]))))
    return out


def _tri_polys_n(mesh):
    verts = [tuple(float(c) for c in v) for v in mesh["vertices"]]
    out = []
    for i, j, k in mesh["triangles"]:
        tri = [verts[i], verts[j], verts[k]]
        out.append((tri, _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))))
    return out


def _cyl_tuple(c):
    return (float(c.cx), float(c.cy), float(c.r), float(c.z0), float(c.z1))


def _inside_fn(solid):
    """A callable q(float3) -> bool: is q strictly inside the solid? Composed so
    holes are respected without triangulating them away."""
    name = type(solid).__name__
    if name == "Solid":
        pn = _solid_polys_n(solid)
        return lambda q: _parity_inside(pn, q)
    if name == "DrilledSolid":
        base_in = _inside_fn(solid.base)
        bores = [_cyl_tuple(c) for c in solid.bores]

        def inside(q):
            if not base_in(q):
                return False
            x, y, z = q
            for cx, cy, r, z0, z1 in bores:
                if z0 - _EPS <= z <= z1 + _EPS and \
                        (x - cx) ** 2 + (y - cy) ** 2 <= r * r + _EPS:
                    return False        # in a bore = removed
            return True
        return inside
    if name == "Cyl":
        cx, cy, r, z0, z1 = _cyl_tuple(solid)
        return lambda q: (z0 - _EPS <= q[2] <= z1 + _EPS
                          and (q[0] - cx) ** 2 + (q[1] - cy) ** 2 <= r * r + _EPS)
    # general: parity against the tessellation
    try:
        pn = _tri_polys_n(solid.tessellate())
        return lambda q: _parity_inside(pn, q)
    except Exception:                   # noqa: BLE001 - no mesh ⇒ never occlude
        return lambda q: False


# -- edges to draw -------------------------------------------------------------

def _solid_edges(solid):
    """Exact straight edges of a planar solid, as 3D float segments."""
    from forgekernel.brep import logical_edges

    out = []
    for e in logical_edges(solid):
        d = _f3(e["dir"])
        pt = _f3(e["point"])
        dd = _dot(d, d) or 1.0
        tp = _dot(pt, d)
        tmin, tmax = float(e["tmin"]), float(e["tmax"])
        p0 = _add(pt, _mul(d, (tmin - tp) / dd))
        p1 = _add(pt, _mul(d, (tmax - tp) / dd))
        out.append([p0, p1])
    return out


def _circle(cx, cy, r, z, segs):
    return [(cx + r * math.cos(2 * math.pi * k / segs),
             cy + r * math.sin(2 * math.pi * k / segs), z)
            for k in range(segs + 1)]


def _cyl_edges(c, view, deflection):
    """A z-axis cylinder's drawing edges: the two rim circles plus, when the
    view is not down the axis, the two silhouette generators."""
    cx, cy, r, z0, z1 = _cyl_tuple(c)
    segs = max(24, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - deflection / r))))
               if r > deflection else 24)
    out = [_circle(cx, cy, r, z0, segs), _circle(cx, cy, r, z1, segs)]
    lx, ly = view.look[0], view.look[1]           # axis is +z; silhouette needs
    m = math.hypot(lx, ly)                         # the in-plane look component
    if m > 1e-6:
        px, py = -ly / m, lx / m                   # perpendicular to look, in xy
        for s in (1.0, -1.0):
            x, y = cx + s * r * px, cy + s * r * py
            out.append([(x, y, z0), (x, y, z1)])
    return out


def _mesh_edges(solid, view):
    """Sharp feature edges + view-dependent silhouette edges from a mesh."""
    try:
        mesh = solid.tessellate()
    except Exception:                   # noqa: BLE001
        return []
    verts = [tuple(float(c) for c in v) for v in mesh["vertices"]]
    from collections import defaultdict
    faces = defaultdict(list)           # undirected edge -> [triangle normals]
    tri_of = defaultdict(list)          # undirected edge -> [(a,b) as given]
    for i, j, k in mesh["triangles"]:
        n = _norm(_cross(_sub(verts[j], verts[i]), _sub(verts[k], verts[i])))
        for a, b in ((i, j), (j, k), (k, i)):
            e = (min(a, b), max(a, b))
            faces[e].append(n)
            tri_of[e].append((verts[a], verts[b]))
    out = []
    for e, normals in faces.items():
        seg = [verts[e[0]], verts[e[1]]]
        if len(normals) == 1:
            out.append(seg)             # boundary edge
            continue
        n0, n1 = normals[0], normals[1]
        if _dot(n0, n1) < 0.985:        # sharp feature edge (~10°)
            out.append(seg)
            continue
        # silhouette: the two faces face opposite ways relative to the viewer
        f0 = _dot(n0, view.look) < 0
        f1 = _dot(n1, view.look) < 0
        if f0 != f1:
            out.append(seg)
    return out


def _draw_edges(solid, view, deflection):
    name = type(solid).__name__
    if name == "Solid":
        return _solid_edges(solid)
    if name == "DrilledSolid":
        segs = _solid_edges(solid.base)
        for c in solid.bores:
            segs += _cyl_edges(c, view, deflection)
        return segs
    if name == "Cyl":
        return _cyl_edges(solid, view, deflection)
    if name == "DisjointUnion":
        segs = []
        for m in solid.members:
            segs += _draw_edges(m, view, deflection)
        return segs
    return _mesh_edges(solid, view)


# -- main ----------------------------------------------------------------------

def _split_polyline(pts3, view, inside, deflection):
    """Project a 3D polyline and split it into visible/hidden runs by testing,
    at each segment's samples, whether a hair toward the viewer is occluded."""
    toward = _mul(view.look, -1.0)              # from a surface point to viewer
    vis, hid = [], []
    for a, b in zip(pts3[:-1], pts3[1:]):
        seg = _sub(b, a)
        length = math.sqrt(_dot(seg, seg))
        nseg = max(1, int(math.ceil(length / max(deflection, 1e-6))))
        prev_hidden = None
        run = []
        for s in range(nseg + 1):
            t = s / nseg
            p = _add(a, _mul(seg, t))
            xy = view.xy(p)
            if s < nseg:                        # visibility of the sub-segment
                mid = _add(a, _mul(seg, (s + 0.5) / nseg))
                h = inside(_add(mid, _mul(toward, 1e-4 * max(1.0, length))))
            else:
                h = prev_hidden
            if prev_hidden is None or h == prev_hidden:
                run.append(xy)
            else:
                (hid if prev_hidden else vis).append(run)
                run = [run[-1], xy]
            prev_hidden = h
        if len(run) >= 2:
            (hid if prev_hidden else vis).append(run)
    return vis, hid


def hidden_line(solid, direction, xdir, *, deflection=0.05):
    """Return ``{"visible": [polyline…], "hidden": [polyline…]}`` for the view,
    each polyline a list of ``(x, y)`` floats in the sheet frame."""
    view = _View(direction, xdir)
    inside = _inside_fn(solid)
    visible, hidden = [], []
    for edge in _draw_edges(solid, view, deflection):
        v, h = _split_polyline(edge, view, inside, deflection)
        visible += [p for p in v if len(p) >= 2]
        hidden += [p for p in h if len(p) >= 2]
    return {"visible": visible, "hidden": hidden}
