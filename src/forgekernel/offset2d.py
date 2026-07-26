"""Exact inward offset of a polygon, with the topology repair that needs.

Offsetting every edge inward by t and reconnecting the corners is right only
while nothing runs out. When a region is thinner than 2t its two walls cross,
and the naive answer is a polygon turned inside out — which is how a lathe
with a 1 mm base, inset 1 mm, produced a void whose floor sat BELOW its
ceiling. The kernel caught that as "the void self-intersects" and refused,
correctly but unhelpfully: the true erosion is simply the part that survives.

The fix is to treat offsetting as a PROCESS rather than a formula. Push every
edge inward at unit speed; each vertex slides along the intersection of its two
moving edges; and an edge whose two ends meet has been consumed and leaves the
polygon. Advancing event by event and removing consumed edges gives the exact
offset at any distance — this is the straight skeleton, restricted to the part
of it this kernel needs.

EXACTNESS (ADR-0019). An edge's offset line is ``n·x = d + t`` with n the
INWARD UNIT normal, so t enters linearly and every event time is the solution
of a linear system — rational, no roots. That holds only while n is rational,
i.e. for axis-aligned edges and Pythagorean ones (3-4-5 and friends); anything
else has an irrational unit normal and is refused by name rather than floated.
Every lathe profile and every rectilinear prism is in the exact set.

WHAT IS NOT HERE. A SPLIT event — a reflex vertex reaching a far edge and
cutting the polygon in two — is detected and refused, not handled. Supporting
it means returning several loops and letting callers deal with a void that
became two voids, which is a larger change than the one this file makes.
"""

from __future__ import annotations

from fractions import Fraction as F

__all__ = ["inset_polygon", "OffsetError"]


class OffsetError(ValueError):
    """The offset cannot be computed exactly, or does not exist."""


def _q(x) -> F:
    return x if isinstance(x, F) else F(x)


def _inward_normal(a, b, ccw: bool):
    """The INWARD unit normal of edge a->b, exactly, or None if irrational.

    For a counter-clockwise loop the interior is to the left, so the inward
    normal is the left perpendicular of the edge direction. Its length is
    √(dx²+dy²), which is rational only for axis-aligned and Pythagorean edges —
    the exact set this file works over.
    """
    dx, dy = _q(b[0]) - _q(a[0]), _q(b[1]) - _q(a[1])
    if dx == 0 and dy == 0:
        return None
    sq = dx * dx + dy * dy
    from math import isqrt

    n, d = sq.numerator, sq.denominator
    rn, rd = isqrt(n), isqrt(d)
    if rn * rn != n or rd * rd != d:
        return None                      # irrational length: not in ℚ
    ln = F(rn, rd)
    nx, ny = (-dy / ln, dx / ln) if ccw else (dy / ln, -dx / ln)
    return nx, ny


def _signed_area2(poly) -> F:
    n = len(poly)
    return sum(_q(poly[i][0]) * _q(poly[(i + 1) % n][1])
               - _q(poly[(i + 1) % n][0]) * _q(poly[i][1]) for i in range(n))


def _corner(l0, l1):
    """Where two offset lines meet, as (point0, velocity) affine in t.

    Each line is n·x = d + t, so the intersection is the solution of a 2x2
    system whose right-hand side is affine in t — hence so is the vertex, and
    every event time below comes out rational.
    """
    (a0, b0, d0), (a1, b1, d1) = l0, l1
    det = a0 * b1 - a1 * b0
    if det == 0:
        return None                      # parallel: the corner never forms
    # x(t) = X0 + Xv·t  by Cramer, with rhs (d0 + t, d1 + t)
    x0 = (d0 * b1 - d1 * b0) / det
    xv = (b1 - b0) / det
    y0 = (a0 * d1 - a1 * d0) / det
    yv = (a0 - a1) / det
    return (x0, y0), (xv, yv)


def inset_polygon(poly, t):
    """The polygon eroded inward by ``t``, exactly. May be empty.

    Returns a list of (x, y) pairs, or [] when nothing survives. Raises
    ``OffsetError`` for an edge whose unit normal is irrational, for a
    self-intersecting input, or for a split event.
    """
    t = _q(t)
    if t < 0:
        raise OffsetError("a negative inset is a dilation, not this")
    pts = [(_q(p[0]), _q(p[1])) for p in poly]
    # drop repeated points, which make a zero-length edge and no normal
    pts = [p for i, p in enumerate(pts) if p != pts[i - 1]]
    if len(pts) < 3:
        raise OffsetError("a polygon needs three distinct vertices")
    ccw = _signed_area2(pts) > 0
    if _signed_area2(pts) == 0:
        raise OffsetError("a degenerate (zero-area) profile has no inside")
    # Convexity decides whether "consumed" can be concluded GLOBALLY. The
    # erosion of a convex set is convex, so when its walls meet there is
    # nothing anywhere. A non-convex region can have one part close while
    # another survives — a tube's 1 mm base vanishes while its 3 mm wall does
    # not — so concluding empty there would silently throw the wall away. That
    # is a wrong answer, and the whole point of this file is not to give one.
    convex = _is_convex(pts, ccw)

    lines = []
    for i, p in enumerate(pts):
        nrm = _inward_normal(p, pts[(i + 1) % len(pts)], ccw)
        if nrm is None:
            raise OffsetError(
                f"edge {i} has an irrational unit normal — an exact offset "
                "needs axis-aligned or Pythagorean edges (K4.2)")
        nx, ny = nrm
        lines.append((nx, ny, nx * p[0] + ny * p[1]))

    now = F(0)
    guard = 0
    while True:
        guard += 1
        if guard > 4 * len(poly) + 8:
            raise OffsetError("offset did not converge (self-intersecting?)")
        if len(lines) < 3:
            return []                    # everything was consumed
        corners = []
        for i in range(len(lines)):
            c = _corner(lines[i - 1], lines[i])
            if c is None:
                # Parallel neighbours are an OUTCOME, not an error. Opposed
                # normals mean the two walls now face each other with nothing
                # between: the region is used up. Equal normals mean the edges
                # went collinear, which is harmless — drop one and carry on.
                n0, n1 = lines[i - 1][:2], lines[i][:2]
                if (n0[0] + n1[0], n0[1] + n1[1]) == (0, 0):
                    return _consumed(convex)
                lines.pop(i)
                break
            corners.append(c)
        else:
            corners = corners            # no restart needed
        if len(corners) != len(lines):
            continue                     # a collinear pair was merged; redo
        # when does edge i's two ends meet? Its length is affine in t.
        best, best_i = None, -1
        for i in range(len(lines)):
            (p0, v0) = corners[i]
            (p1, v1) = corners[(i + 1) % len(lines)]
            dx0, dy0 = p1[0] - p0[0], p1[1] - p0[1]
            dxv, dyv = v1[0] - v0[0], v1[1] - v0[1]
            # project onto the edge direction so the zero is where it collapses
            ex, ey = lines[i][1], -lines[i][0]        # along the edge
            l0 = dx0 * ex + dy0 * ey
            lv = dxv * ex + dyv * ey
            if lv == 0:
                continue                 # this edge keeps its length
            tau = now - l0 / lv
            if tau > now and (best is None or tau < best):
                best, best_i = tau, i
        if best is None or best > t:
            out = [(p[0] + v[0] * t, p[1] + v[1] * t) for p, v in corners]
            if _signed_area2(out) <= 0:
                return _consumed(convex)
            _reject_split(out)
            _verify_clearance(pts, out, t)
            return out
        now = best
        lines.pop(best_i)                # that edge has been used up


def _reject_split(poly) -> None:
    """Refuse a result that crosses itself: that is a SPLIT event, where a
    reflex vertex has reached a far edge and cut the region in two. Handling it
    means returning several loops, which the callers do not model yet — so say
    that rather than hand back a figure-of-eight (K4.2)."""
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if _crosses(a, b, c, d):
                raise OffsetError(
                    "the offset splits the region in two (a reflex corner "
                    "reached a far wall) — several loops are not modelled "
                    "yet (K4.2)")


def _is_convex(pts, ccw: bool) -> bool:
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cr = ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if (cr < 0) if ccw else (cr > 0):
            return False
    return True


def _consumed(convex: bool):
    """Two walls met. For a CONVEX profile that settles it globally; for a
    non-convex one it settles only this neighbourhood, and another part may
    well survive — so say so instead of returning an empty region that would
    read as a definite answer."""
    if convex:
        return []
    raise OffsetError(
        "two opposite walls met before any corner did — that is a split "
        "event, and this offset models only corner events (K4.2). The region "
        "is thinner than 2t somewhere, and some other part of it may survive.")


def _seg_dist2(a, b, p) -> F:
    """Squared distance from p to segment ab, exactly (a clamped ratio)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    if den == 0:
        qx, qy = a
    else:
        u = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den
        u = min(max(u, F(0)), F(1))
        qx, qy = a[0] + u * dx, a[1] + u * dy
    return (p[0] - qx) ** 2 + (p[1] - qy) ** 2


def _verify_clearance(orig, out, t) -> None:
    """Every vertex of the result must be at least t from the ORIGINAL wall.

    This validates the ANSWER instead of trusting the event bookkeeping, which
    is the whole lesson of this kernel's bug history — and it is what catches
    the events this file does not model. When two OPPOSITE walls close on each
    other (a 1 mm base under a 1 mm inset), no adjacent edge ever collapses, so
    the simulation sails past it and produces a region that is nowhere near t
    from the boundary. Exact: a clamped ratio compared against t².
    """
    t2 = t * t
    for p in out:
        for i in range(len(orig)):
            if _seg_dist2(orig[i], orig[(i + 1) % len(orig)], p) < t2:
                raise OffsetError(
                    "two opposite walls met before any corner did — that is a "
                    "split event, and this offset models only corner events "
                    "(K4.2). The region is thinner than 2t somewhere.")


def _side(a, b, p) -> int:
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return (v > 0) - (v < 0)


def _crosses(a, b, c, d) -> bool:
    """Proper segment crossing, exactly. Touching at an endpoint is not a
    crossing — an offset legitimately brings corners together."""
    d1, d2 = _side(a, b, c), _side(a, b, d)
    d3, d4 = _side(c, d, a), _side(c, d, b)
    return d1 * d2 < 0 and d3 * d4 < 0
