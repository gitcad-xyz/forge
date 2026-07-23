"""K3.8 — spline sketch profiles with EXACT area (Green's theorem).

A closed 2D profile of line and polynomial-Bézier segments encloses an
area that is *exactly rational*:

    A = ½ ∮ (x dy − y dx)

Over a line segment this is ½(x0 y1 − x1 y0); over a Bézier segment
x(t), y(t) are polynomials, so ½(x y' − y x') is a polynomial and its
integral over [0,1] is exact. Extruding such a profile therefore has an
exactly rational volume A·h — a curved-boundary solid OCCT can only
Gauss-quadrature.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.bsolid import _lagrange_weights, _nodes

F = Fraction


def _bezier2(pts, t):
    """De Casteljau on 2D control points at parameter t (exact)."""
    p = [(F(a), F(b)) for a, b in pts]
    n = len(p)
    for r in range(1, n):
        p = [((1 - t) * p[i][0] + t * p[i + 1][0],
              (1 - t) * p[i][1] + t * p[i + 1][1]) for i in range(n - r)]
    return p[0]


def _bezier2_d(pts):
    """Control points of the derivative curve (degree p-1), exact."""
    p = len(pts) - 1
    return [(F(p) * (F(pts[i + 1][0]) - F(pts[i][0])),
             F(p) * (F(pts[i + 1][1]) - F(pts[i][1]))) for i in range(p)]


def segments_to_beziers(start, segments):
    """Normalize a profile into a list of Bézier control-point lists.
    Lines → degree-1 Béziers; arcs are rejected here (they belong to the
    ℚ[π] path). ``spline`` segments carry explicit control points."""
    beziers = []
    cur = (F(start[0]), F(start[1]))
    for seg in segments:
        kind = seg["kind"]
        to = (F(seg["to"][0]), F(seg["to"][1]))
        if kind == "line":
            beziers.append([cur, to])
        elif kind == "spline":
            # Bézier control points between cur and to (exclusive endpoints
            # given in "ctrl"); a cubic if two ctrl points, etc.
            ctrl = [(F(a), F(b)) for a, b in seg.get("ctrl", [])]
            beziers.append([cur, *ctrl, to])
        else:
            raise ValueError(f"profile2d: segment kind {kind!r} not exact "
                             f"(arcs → ℚ[π] path)")
        cur = to
    return beziers


def exact_area(start, segments) -> Fraction:
    """Signed area of the closed profile via Green's theorem — exact ℚ.
    Returns the absolute area (a positive Fraction)."""
    beziers = segments_to_beziers(start, segments)
    total = F(0)
    for bez in beziers:
        if len(bez) == 2:                      # line: ½(x0 y1 − x1 y0)
            (x0, y0), (x1, y1) = bez
            total += (x0 * y1 - x1 * y0) / 2
        else:                                   # Bézier: ∫ ½(x y' − y x') dt
            dctrl = _bezier2_d(bez)
            p = len(bez) - 1
            # integrand degree ≤ 2p-1 → 2p nodes exact
            nn = _nodes(2 * p)
            ww = _lagrange_weights(nn)
            seg_int = F(0)
            for w, t in zip(ww, nn):
                x, y = _bezier2(bez, t)
                dx, dy = _bezier2(dctrl, t)
                seg_int += w * (x * dy - y * dx) / 2
            total += seg_int
    return abs(total)


class SplinePrism:
    """Extrusion of a closed line/Bézier profile — exact rational volume
    A·h (Green's-theorem area). Curved boundary; planar top/bottom caps."""

    provenance = "exact"

    def __init__(self, start, segments, height, base_z=0) -> None:
        self.start = (F(start[0]), F(start[1]))
        self.segments = segments
        self.h = F(height)
        self.z0 = F(base_z)
        self._area = exact_area(start, segments)
        if self._area == 0 or self.h == 0:
            raise ValueError("spline prism has zero area or height")

    def volume(self) -> Fraction:
        return self._area * abs(self.h)

    def area(self) -> Fraction:
        return self._area

    def bbox_f(self):
        beziers = segments_to_beziers(self.start, self.segments)
        xs = [float(c[0]) for b in beziers for c in b]
        ys = [float(c[1]) for b in beziers for c in b]
        # control-net hull bounds the curve; sample for a tight-ish box
        for b in beziers:
            if len(b) > 2:
                for k in range(9):
                    x, y = _bezier2(b, F(k, 8))
                    xs.append(float(x)); ys.append(float(y))
        z0, z1 = float(self.z0), float(self.z0 + self.h)
        return ((min(xs), min(ys), min(z0, z1)),
                (max(xs), max(ys), max(z0, z1)))

    def centroid_f(self):
        (x0, y0, z0), (x1, y1, z1) = self.bbox_f()
        return ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)

    def tessellate(self, deflection: float = 0.2) -> dict:
        beziers = segments_to_beziers(self.start, self.segments)
        ring = []
        for b in beziers:
            steps = 1 if len(b) == 2 else 12
            for k in range(steps):
                x, y = _bezier2(b, F(k, steps))
                ring.append((float(x), float(y)))
        z0, z1 = float(self.z0), float(self.z0 + self.h)
        n = len(ring)
        verts = [[x, y, z0] for x, y in ring] + [[x, y, z1] for x, y in ring]
        tris = []
        for i in range(n):
            j = (i + 1) % n
            tris += [[i, j, n + i], [j, n + j, n + i]]
        # fan caps (approximate; render only)
        for i in range(1, n - 1):
            tris.append([0, i, i + 1])
            tris.append([n, n + i + 1, n + i])
        return {"vertices": verts, "triangles": tris}
