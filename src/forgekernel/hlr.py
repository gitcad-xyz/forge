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


def _circle_segs(r, deflection):
    """Facet count for a circle of radius ``r`` at chord error ``deflection``."""
    if r <= deflection:
        return 24
    return max(24, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - deflection / r)))))


def _cyl_edges(c, view, deflection):
    """A z-axis cylinder's drawing edges: the two rim circles plus, when the
    view is not down the axis, the two silhouette generators."""
    cx, cy, r, z0, z1 = _cyl_tuple(c)
    segs = _circle_segs(r, deflection)
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


# -- section curves (plane ∩ solid boundary) ----------------------------------
#
# A section view cuts the solid with a plane and draws the intersection of that
# plane with the solid's *surface* — the closed loops the section engine chains
# and hatches. Like HLR this is a display computation (floats legal); the plane
# and the solid it cuts are exact. The plane is ``{P : dot(P, direction)=offset}``
# (``direction`` is the view/look axis, so the section lies flat in the sheet).

def _plane_from(direction, offset):
    """Cut plane as a unit normal + signed distance ``dot(P, n) = d``."""
    raw = _f3(direction)
    scale = math.sqrt(_dot(raw, raw)) or 1.0
    return _norm(raw), float(offset) / scale


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _two_plane_point(n1, d1, n2, d2, u):
    """A point on the line where planes (n1,d1) and (n2,d2) meet; ``u=n1×n2``."""
    m = [list(n1), list(n2), list(u)]
    det = _det3(m)
    if abs(det) < 1e-16:
        return (0.0, 0.0, 0.0)
    rhs = (d1, d2, 0.0)
    out = []
    for j in range(3):
        mj = [row[:] for row in m]
        for i in range(3):
            mj[i][j] = rhs[i]
        out.append(_det3(mj) / det)
    return tuple(out)


def _line_in_convex(p0, u, poly, fn):
    """λ-interval of ``p0+λu`` inside a convex planar polygon (normal ``fn``),
    found by the two boundary crossings — winding-agnostic."""
    lambdas = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        e = _sub(b, a)
        en = _cross(fn, e)
        denom = _dot(u, en)
        if abs(denom) < 1e-15:
            continue
        lam = _dot(_sub(a, p0), en) / denom
        pt = _add(p0, _mul(u, lam))
        ee = _dot(e, e) or 1.0
        s = _dot(_sub(pt, a), e) / ee
        if -1e-9 <= s <= 1 + 1e-9:
            lambdas.append(lam)
    if len(lambdas) < 2:
        return None
    return (min(lambdas), max(lambdas))


def _circle_lambdas(p0, u, c3, r):
    """λ-interval where ``p0+λu`` lies inside the circle (centre ``c3``, radius
    ``r``, both in the line's plane), or None."""
    w = _sub(p0, c3)
    a = _dot(u, u)
    b = 2.0 * _dot(w, u)
    c = _dot(w, w) - r * r
    disc = b * b - 4 * a * c
    if disc <= 1e-12 or a < 1e-16:
        return None
    sq = math.sqrt(disc)
    return ((-b - sq) / (2 * a), (-b + sq) / (2 * a))


def _subtract(intervals, lo, hi):
    out = []
    for a, b in intervals:
        if hi <= a or lo >= b:
            out.append((a, b))
            continue
        if a < lo:
            out.append((a, lo))
        if hi < b:
            out.append((hi, b))
    return out


def _planar_face_section(poly, fn, nc, dc, bores):
    """Segments of the cut plane across one planar face, with any bore disks
    that open onto the face subtracted (so through-holes stay clear)."""
    u = _cross(nc, fn)
    if _dot(u, u) < 1e-16:
        return []                        # face parallel to the cut plane
    df = _dot(fn, poly[0])
    p0 = _two_plane_point(nc, dc, fn, df, u)
    span = _line_in_convex(p0, u, poly, fn)
    if span is None:
        return []
    intervals = [span]
    nfu = _norm(fn)
    if abs(nfu[2]) > 0.99:               # a z-cap face — bores open onto it
        zf = float(poly[0][2])
        for cx, cy, r, z0, z1 in bores:
            if abs(zf - z0) < 1e-6 or abs(zf - z1) < 1e-6:
                hit = _circle_lambdas(p0, u, (cx, cy, zf), r)
                if hit:
                    intervals = _subtract(intervals, hit[0], hit[1])
    segs = []
    for a, b in intervals:
        if b - a > 1e-9:
            segs.append((_add(p0, _mul(u, a)), _add(p0, _mul(u, b))))
    return segs


def _wall_xy(cx, cy, r, nc, dc):
    """The up-to-two (x, y) where a z-cylinder wall (radius r) meets a cut plane
    parallel to the axis (nz≈0), or [] if the plane misses it."""
    nx, ny, _ = nc
    mag = r * math.hypot(nx, ny)
    if mag < 1e-12:
        return []
    cth = (dc - (nx * cx + ny * cy)) / mag
    if cth < -1 - 1e-9 or cth > 1 + 1e-9:
        return []
    phi = math.atan2(ny, nx)
    ac = math.acos(max(-1.0, min(1.0, cth)))
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in {phi + ac, phi - ac}]


def _cyl_wall_seg(cx, cy, r, z0, z1, nc, dc, deflection):
    """Segments where the cut plane crosses a z-cylinder wall over [z0, z1].
    Two vertical generators when the plane is parallel to the axis; a sampled
    ellipse/circle otherwise."""
    nx, ny, nz = nc
    if abs(nz) < 1e-9:                    # plane parallel to the axis
        return [((x, y, z0), (x, y, z1)) for (x, y) in _wall_xy(cx, cy, r, nc, dc)]
    segs = _circle_segs(r, deflection)   # plane crosses the axis: sample θ
    prev = None
    out = []
    for k in range(segs + 1):
        th = 2 * math.pi * k / segs
        x, y = cx + r * math.cos(th), cy + r * math.sin(th)
        z = (dc - nx * x - ny * y) / nz
        cur = (x, y, z) if z0 - 1e-9 <= z <= z1 + 1e-9 else None
        if prev is not None and cur is not None:
            out.append((prev, cur))
        prev = cur
    return out


def _cyl_wall_section(c, nc, dc, deflection):
    cx, cy, r, z0, z1 = _cyl_tuple(c)
    return _cyl_wall_seg(cx, cy, r, z0, z1, nc, dc, deflection)


def _bore_bands(group):
    """Partition a coaxial bore group into z-bands, each tagged with the
    OUTERMOST radius covering it — the profile of the bores' union. ``group`` is
    a list of (cx, cy, r, z0, z1) tuples sharing (cx, cy)."""
    zs = sorted({z for (_, _, _, z0, z1) in group for z in (z0, z1)})
    bands = []
    for za, zb in zip(zs, zs[1:]):
        zmid = (za + zb) / 2
        rs = [r for (_, _, r, z0, z1) in group if z0 - 1e-9 <= zmid <= z1 + 1e-9]
        if rs:
            bands.append((za, zb, max(rs)))
    return bands


def _section_bore_group(group, nc, dc, deflection):
    """Section a coaxial bore stack (e.g. a counterbore) as ONE stepped-cylinder
    profile: walls at the outermost radius per z-band, plus shoulder rings where
    the radius steps (only visible when the plane is parallel to the axis).
    Sectioning each bore independently would draw the inner wall straight through
    the empty counterbore cavity and omit the shoulder."""
    cx, cy = float(group[0][0]), float(group[0][1])
    bands = _bore_bands(group)
    out = []
    for za, zb, r in bands:
        out += _cyl_wall_seg(cx, cy, r, za, zb, nc, dc, deflection)
    if abs(nc[2]) < 1e-9:                 # shoulders only cut when plane ∥ axis
        for (za, zb, r0), (zb2, zc, r1) in zip(bands, bands[1:]):
            if abs(r0 - r1) < 1e-12:
                continue
            lo = _wall_xy(cx, cy, min(r0, r1), nc, dc)
            hi = _wall_xy(cx, cy, max(r0, r1), nc, dc)
            for h in hi:                 # connect each outer point to its side's inner
                if not lo:
                    continue
                l = min(lo, key=lambda p: (p[0] - h[0]) ** 2 + (p[1] - h[1]) ** 2)
                out.append(((l[0], l[1], zb), (h[0], h[1], zb)))
    return out


def _cyl_cap_sections(c, nc, dc):
    cx, cy, r, z0, z1 = _cyl_tuple(c)
    out = []
    for zc in (z0, z1):
        fn = (0.0, 0.0, 1.0)
        u = _cross(nc, fn)
        if _dot(u, u) < 1e-16:
            continue
        p0 = _two_plane_point(nc, dc, fn, zc, u)
        hit = _circle_lambdas(p0, u, (cx, cy, zc), r)
        if hit:
            out.append((_add(p0, _mul(u, hit[0])), _add(p0, _mul(u, hit[1]))))
    return out


def _mesh_section(solid, nc, dc):
    try:
        mesh = solid.tessellate()
    except Exception:                    # noqa: BLE001
        return []
    verts = [tuple(float(x) for x in v) for v in mesh["vertices"]]
    out = []
    for tri in mesh["triangles"]:
        pts = [verts[i] for i in tri]
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = pts[a], pts[b]
            da, db = _dot(pa, nc) - dc, _dot(pb, nc) - dc
            if (da <= 0 < db) or (db <= 0 < da):
                t = da / (da - db)
                hits.append(_add(pa, _mul(_sub(pb, pa), t)))
        if len(hits) == 2:
            out.append((hits[0], hits[1]))
    return out


def _section_segments(solid, nc, dc, deflection):
    name = type(solid).__name__
    if name == "Solid":
        return [s for poly, fn in _solid_polys_n(solid)
                for s in _planar_face_section(poly, fn, nc, dc, [])]
    if name == "DrilledSolid":
        bores = [_cyl_tuple(x) for x in solid.bores]
        out = [s for poly, fn in _solid_polys_n(solid.base)
               for s in _planar_face_section(poly, fn, nc, dc, bores)]
        groups = {}                      # coaxial bores section as one profile
        for b in bores:
            groups.setdefault((round(b[0], 9), round(b[1], 9)), []).append(b)
        for g in groups.values():
            out += _section_bore_group(g, nc, dc, deflection)
        return out
    if name == "Cyl":
        return (_cyl_wall_section(solid, nc, dc, deflection)
                + _cyl_cap_sections(solid, nc, dc))
    if name == "DisjointUnion":
        out = []
        for m in solid.members:
            out += _section_segments(m, nc, dc, deflection)
        return out
    return _mesh_section(solid, nc, dc)


def section_polys(solid, direction, xdir, offset, *, deflection=0.05):
    """Cut ``solid`` with the plane ``dot(P, direction)=offset`` and return the
    section curves as 2D segments ``[[(x,y),(x,y)], …]`` in the same sheet frame
    ``hidden_line`` uses, so a section view overlays its projection exactly."""
    view = _View(direction, xdir)
    nc, dc = _plane_from(direction, offset)
    return [[view.xy(a), view.xy(b)]
            for a, b in _section_segments(solid, nc, dc, deflection)]
