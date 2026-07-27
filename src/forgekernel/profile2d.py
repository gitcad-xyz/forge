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

import math
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


def _bezier_axis_extrema(bez, dim):
    """Parameters t in (0,1) where one coordinate of a Bézier is stationary.

    The derivative of a degree-p Bézier is degree p−1, so for p ≤ 3 the
    stationary parameters are roots of a linear or quadratic with RATIONAL
    coefficients — closed form. A rational root is returned exactly; an
    irrational root is returned as the exact rational value of its float
    (within 1 ulp of the true parameter). Either way the caller evaluates
    the curve AT that parameter, so the point used for the box lies exactly
    on the curve — near a stationary point the coordinate error is second
    order in the 1-ulp parameter error, i.e. far below float resolution.
    Callers must handle p ≥ 4 themselves (no closed form here)."""
    coeffs = [c[dim] for c in _bezier2_d(bez)]
    ts = []
    if len(coeffs) == 2:                          # quadratic curve: linear B'
        d0, d1 = coeffs
        if d0 != d1:
            t = d0 / (d0 - d1)
            if 0 < t < 1:
                ts.append(t)
        return ts
    d0, d1, d2 = coeffs                           # cubic curve: quadratic B'
    a = d0 - 2 * d1 + d2
    b = 2 * (d1 - d0)
    c = d0
    if a == 0:
        if b != 0:
            t = -c / b
            if 0 < t < 1:
                ts.append(t)
        return ts
    disc = b * b - 4 * a * c
    if disc < 0:
        return ts
    root = F(math.sqrt(disc))
    for sgn in (1, -1):
        t = (-b + sgn * root) / (2 * a)
        if 0 < t < 1:
            ts.append(t)
    return ts


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
    # close the loop back to the start if the last point doesn't already —
    # Green's theorem needs a CLOSED contour or the area is meaningless
    # (the line-extrude path auto-closes; the spline path must too).
    s0 = (F(start[0]), F(start[1]))
    if cur != s0:
        beziers.append([cur, s0])
    return beziers


def exact_signed_area(start, segments) -> Fraction:
    """Signed area of the closed profile via Green's theorem — exact ℚ.
    CCW positive. Keep the sign when pairing with the area moments so the
    orientation cancels in the centroid ratio."""
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
    return total


def exact_area(start, segments) -> Fraction:
    """Absolute enclosed area (a positive Fraction), exact ℚ."""
    return abs(exact_signed_area(start, segments))


def exact_moments(start, segments):
    """First area moments (Qx = ∫∫ x dA, Qy = ∫∫ y dA) of the closed profile
    via Green's theorem — exact ℚ. Qx = ∮ ½x² dy and Qy = −∮ ½y² dx; the
    integrand ``x²·y'`` has degree ≤ 3p−1 over a degree-p Bézier, so 3p
    interpolatory nodes integrate it exactly. Signs follow the loop
    orientation (pair with :func:`exact_signed_area`)."""
    beziers = segments_to_beziers(start, segments)
    qx = qy = F(0)
    for bez in beziers:
        dctrl = _bezier2_d(bez)
        p = len(bez) - 1
        nn = _nodes(3 * p)
        ww = _lagrange_weights(nn)
        for w, t in zip(ww, nn):
            x, y = _bezier2(bez, t)
            dx, dy = _bezier2(dctrl, t)
            qx += w * (x * x) / 2 * dy
            qy += w * (-(y * y)) / 2 * dx
    return qx, qy


def _boundary_is_simple(start, segments, samples: int = 16) -> bool:
    """Flatten the line/Bézier boundary to a polyline and test for a proper
    self-intersection of non-adjacent edges (a simple-loop guard)."""
    ring = []
    for bez in segments_to_beziers(start, segments):
        steps = 1 if len(bez) == 2 else samples
        for k in range(steps):
            t = F(k, steps)
            ring.append(_bezier2(bez, t))
    n = len(ring)
    if n < 3:
        return False

    def seg_cross(a, b, c, d):
        def orient(p, q, r):
            return ((q[0] - p[0]) * (r[1] - p[1])
                    - (q[1] - p[1]) * (r[0] - p[0]))
        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue                       # skip shared-vertex neighbours
            c, d = ring[j], ring[(j + 1) % n]
            if seg_cross(a, b, c, d):
                return False
    return True


class SplinePrism:
    """Extrusion of a closed line/Bézier profile — exact rational volume
    A·h (Green's-theorem area). Curved boundary; planar top/bottom caps."""

    provenance = "exact"

    def __init__(self, start, segments, height, base_z=0) -> None:
        self.start = (F(start[0]), F(start[1]))
        self.segments = segments
        self.h = F(height)
        self.z0 = F(base_z)
        if self.h == 0:
            raise ValueError("spline prism has zero height")
        # a self-intersecting boundary makes the signed Green's area
        # meaningless (opposite lobes cancel) — catch it explicitly rather
        # than mis-diagnosing it as "zero area".
        if not _boundary_is_simple(start, segments):
            raise ValueError("profile boundary self-intersects (not a simple loop)")
        self._area = exact_area(start, segments)
        if self._area == 0:
            raise ValueError("spline prism has zero enclosed area")

    def volume(self) -> Fraction:
        return self._area * abs(self.h)

    def area(self) -> Fraction:
        return self._area

    def bbox_f(self):
        """Tight bound of the CURVE, not of the Bézier control net.

        A segment's extreme coordinates occur at its endpoints or where the
        coordinate's derivative vanishes; both are on-curve points. The old
        control-net hull was initialised from the interior control points —
        which can sit far off the curve they steer — so the sampling loop
        below it never tightened anything and dy overstated by 37% on an
        ordinary cubic blade profile."""
        beziers = segments_to_beziers(self.start, self.segments)
        xs: list = []
        ys: list = []
        for b in beziers:
            for c in (b[0], b[-1]):               # endpoints are on-curve
                xs.append(float(c[0])); ys.append(float(c[1]))
            p = len(b) - 1
            if p <= 1:
                continue
            if p <= 3:
                for dim, acc in ((0, xs), (1, ys)):
                    for t in _bezier_axis_extrema(b, dim):
                        acc.append(float(_bezier2(b, t)[dim]))
            else:
                # degree ≥ 4: the derivative roots have no closed form here;
                # keep the conservative control-net hull for this segment
                # (an outward bound — never cuts inside the solid).
                for c in b[1:-1]:
                    xs.append(float(c[0])); ys.append(float(c[1]))
        z0, z1 = float(self.z0), float(self.z0 + self.h)
        return ((min(xs), min(ys), min(z0, z1)),
                (max(xs), max(ys), max(z0, z1)))

    def centroid(self):
        """Exact centroid in ℚ³. The profile's area centroid (Green's-theorem
        first moments ÷ signed area) in x, y; the extrusion mid-height in z.
        Orientation cancels because moments and area share the signed factor."""
        a = exact_signed_area(self.start, self.segments)
        qx, qy = exact_moments(self.start, self.segments)
        return (qx / a, qy / a, self.z0 + self.h / 2)

    def centroid_f(self):
        """Float centroid derived from the exact :meth:`centroid` — the
        profile's true area centroid, not the bbox centre (which is wrong
        for an asymmetric profile)."""
        return tuple(float(c) for c in self.centroid())

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
