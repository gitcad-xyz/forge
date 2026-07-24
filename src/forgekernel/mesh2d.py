"""2D polygon triangulation with holes, for display meshes.

Tessellation is a bounded-error *view* of the exact solid (ADR-0019: meshing is
a display property, floats are legal). ``triangulate(outer, holes)`` returns a
shared vertex list plus triangle indices, used to cap a drilled solid's top and
bottom faces around its bores.

This is a faithful port of the mapbox/earcut ear-clipping algorithm (ISC) — a
robust linked-list ear clipper with hole elimination. A hand-rolled version
tore on multi-hole inputs (bolt patterns) because two holes bridging to the same
outer vertex need distinct linked-list nodes and angular ordering; earcut gets
that right. The O(n²) ear scan (no z-order hash) is ample for a face's facet
count.
"""

from __future__ import annotations

Pt = tuple[float, float]


class _Node:
    __slots__ = ("i", "x", "y", "prev", "next", "steiner")

    def __init__(self, i: int, x: float, y: float) -> None:
        self.i = i
        self.x = x
        self.y = y
        self.prev = None
        self.next = None
        self.steiner = False


def triangulate(outer: list[Pt], holes: list[list[Pt]] = ()) -> tuple[list[Pt], list[tuple[int, int, int]]]:
    """Triangulate ``outer`` minus ``holes``; returns ``(points, triangles)``
    with triangle indices into points."""
    pts: list[Pt] = list(outer)
    hole_indices = []
    for h in holes:
        hole_indices.append(len(pts))
        pts.extend(h)

    flat = [c for p in pts for c in p]
    tris: list[int] = []
    outer_node = _linked_list(flat, 0, len(outer) * 2, 2, True)
    if outer_node is None or outer_node.next is outer_node.prev:
        return pts, []
    if hole_indices:
        outer_node = _eliminate_holes(flat, hole_indices, outer_node, 2)
    _earcut_linked(outer_node, tris, 2)
    return pts, [(tris[i], tris[i + 1], tris[i + 2]) for i in range(0, len(tris), 3)]


def _linked_list(data, start, end, dim, clockwise):
    last = None
    if clockwise == (_signed_area(data, start, end, dim) > 0):
        for i in range(start, end, dim):
            last = _insert_node(i // dim, data[i], data[i + 1], last)
    else:
        for i in range(end - dim, start - 1, -dim):
            last = _insert_node(i // dim, data[i], data[i + 1], last)
    if last is not None and _equals(last, last.next):
        _remove_node(last)
        last = last.next
    return last


def _filter_points(start, end=None):
    if start is None:
        return start
    if end is None:
        end = start
    p = start
    again = True
    while again or p is not end:
        again = False
        if not p.steiner and (_equals(p, p.next) or _area(p.prev, p, p.next) == 0):
            _remove_node(p)
            p = end = p.prev
            if p is p.next:
                break
            again = True
        else:
            p = p.next
    return end


def _earcut_linked(ear, triangles, dim, pass_=0):
    if ear is None:
        return
    ear = _filter_points(ear)
    stop = ear
    while ear.prev is not ear.next:
        prev, nxt = ear.prev, ear.next
        if _is_ear(ear):
            triangles.append(prev.i)
            triangles.append(ear.i)
            triangles.append(nxt.i)
            _remove_node(ear)
            ear = nxt.next
            stop = nxt.next
            continue
        ear = nxt
        if ear is stop:
            if pass_ == 0:
                _earcut_linked(_filter_points(ear), triangles, dim, 1)
            elif pass_ == 1:
                ear = _cure_local_intersections(_filter_points(ear), triangles)
                _earcut_linked(ear, triangles, dim, 2)
            elif pass_ == 2:
                _split_earcut(ear, triangles, dim)
            return


def _is_ear(ear):
    a, b, c = ear.prev, ear, ear.next
    if _area(a, b, c) >= 0:
        return False                        # reflex
    p = c.next
    while p is not a:
        if (_point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y)
                and _area(p.prev, p, p.next) >= 0):
            return False
        p = p.next
    return True


def _cure_local_intersections(start, triangles):
    p = start
    while True:
        a, b = p.prev, p.next.next
        if (not _equals(a, b) and _intersects(a, p, p.next, b)
                and _locally_inside(a, b) and _locally_inside(b, a)):
            triangles.append(a.i)
            triangles.append(p.i)
            triangles.append(b.i)
            _remove_node(p)
            _remove_node(p.next)
            p = start = b
        p = p.next
        if p is start:
            break
    return _filter_points(p)


def _split_earcut(start, triangles, dim):
    a = start
    while True:
        b = a.next.next
        while b is not a.prev:
            if a.i != b.i and _is_valid_diagonal(a, b):
                c = _split_polygon(a, b)
                a = _filter_points(a, a.next)
                c = _filter_points(c, c.next)
                _earcut_linked(a, triangles, dim, 0)
                _earcut_linked(c, triangles, dim, 0)
                return
            b = b.next
        a = a.next
        if a is start:
            break


def _eliminate_holes(data, hole_indices, outer_node, dim):
    queue = []
    for i, start in enumerate(hole_indices):
        end = hole_indices[i + 1] * dim if i + 1 < len(hole_indices) else len(data)
        lst = _linked_list(data, start * dim, end, dim, False)
        if lst is lst.next:
            lst.steiner = True
        queue.append(_left_most(lst))
    queue.sort(key=lambda n: n.x)
    for hole in queue:
        outer_node = _eliminate_hole(hole, outer_node)
    return outer_node


def _eliminate_hole(hole, outer_node):
    bridge = _find_hole_bridge(hole, outer_node)
    if bridge is None:
        return outer_node
    bridge_reverse = _split_polygon(bridge, hole)
    _filter_points(bridge_reverse, bridge_reverse.next)
    return _filter_points(bridge, bridge.next)


def _find_hole_bridge(hole, outer_node):
    p = outer_node
    hx, hy = hole.x, hole.y
    qx = -float("inf")
    m = None
    # find the outer edge to the left of the hole point that the +x ray hits
    while True:
        if hy <= p.y and hy >= p.next.y and p.next.y != p.y:
            x = p.x + (hy - p.y) / (p.next.y - p.y) * (p.next.x - p.x)
            if hx >= x > qx:
                qx = x
                m = p if p.x < p.next.x else p.next
                if x == hx:
                    return m
        p = p.next
        if p is outer_node:
            break
    if m is None:
        return None
    # the bridge must not cross the polygon: pick the reflex vertex of minimum
    # angle (or minimum x) inside the triangle (hole, (qx,hy), m)
    stop = m
    mx, my = m.x, m.y
    tan_min = float("inf")
    p = m
    while True:
        cond = hx > mx if hy < my else hx < mx
        if (min(hx, qx) <= p.x <= max(hx, qx) and hx != p.x
                and _point_in_triangle(hx if hy < my else qx, hy,
                                       mx, my,
                                       qx if hy < my else hx, hy,
                                       p.x, p.y)):
            tan = abs(hy - p.y) / (hx - p.x) if hx != p.x else float("inf")
            if _locally_inside(p, hole) and (tan < tan_min or (
                    tan == tan_min and (p.x > m.x or (p.x == m.x and _sector_contains(m, p))))):
                m = p
                tan_min = tan
        p = p.next
        if p is stop:
            break
    return m


def _sector_contains(m, p):
    return _area(m.prev, m, p.prev) < 0 and _area(p.next, m, m.next) < 0


def _split_polygon(a, b):
    a2 = _Node(a.i, a.x, a.y)
    b2 = _Node(b.i, b.x, b.y)
    an, bp = a.next, b.prev
    a.next = b
    b.prev = a
    a2.next = an
    an.prev = a2
    b2.next = a2
    a2.prev = b2
    bp.next = b2
    b2.prev = bp
    return b2


def _insert_node(i, x, y, last):
    p = _Node(i, x, y)
    if last is None:
        p.prev = p
        p.next = p
    else:
        p.next = last.next
        p.prev = last
        last.next.prev = p
        last.next = p
    return p


def _remove_node(p):
    p.next.prev = p.prev
    p.prev.next = p.next


def _left_most(start):
    p = start
    leftmost = start
    p = p.next
    while p is not start:
        if p.x < leftmost.x or (p.x == leftmost.x and p.y < leftmost.y):
            leftmost = p
        p = p.next
    return leftmost


def _signed_area(data, start, end, dim):
    s = 0.0
    j = end - dim
    for i in range(start, end, dim):
        s += (data[j] - data[i]) * (data[i + 1] + data[j + 1])
        j = i
    return s


def _area(p, q, r):
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _equals(p1, p2):
    return p1.x == p2.x and p1.y == p2.y


def _sign(n):
    return (n > 0) - (n < 0)


def _on_segment(p, q, r):
    return (min(p.x, r.x) <= q.x <= max(p.x, r.x)
            and min(p.y, r.y) <= q.y <= max(p.y, r.y))


def _intersects(p1, q1, p2, q2):
    o1 = _sign(_area(p1, q1, p2))
    o2 = _sign(_area(p1, q1, q2))
    o3 = _sign(_area(p2, q2, p1))
    o4 = _sign(_area(p2, q2, q1))
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _intersects_polygon(a, b):
    p = a
    while True:
        if (p.i != a.i and p.next.i != a.i and p.i != b.i and p.next.i != b.i
                and _intersects(p, p.next, a, b)):
            return True
        p = p.next
        if p is a:
            break
    return False


def _locally_inside(a, b):
    if _area(a.prev, a, a.next) < 0:
        return _area(a, b, a.next) >= 0 and _area(a, a.prev, b) >= 0
    return _area(a, b, a.prev) < 0 or _area(a, a.next, b) < 0


def _middle_inside(a, b):
    p = a
    inside = False
    px, py = (a.x + b.x) / 2, (a.y + b.y) / 2
    while True:
        if ((p.y > py) != (p.next.y > py) and p.next.y != p.y
                and px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x):
            inside = not inside
        p = p.next
        if p is a:
            break
    return inside


def _is_valid_diagonal(a, b):
    return (a.next.i != b.i and a.prev.i != b.i
            and not _intersects_polygon(a, b)
            and ((_locally_inside(a, b) and _locally_inside(b, a) and _middle_inside(a, b)
                  and (_area(a.prev, a, b.prev) != 0 or _area(a, b.prev, b) != 0))
                 or (_equals(a, b) and _area(a.prev, a, a.next) > 0
                     and _area(b.prev, b, b.next) > 0)))


def _point_in_triangle(ax, ay, bx, by, cx, cy, px, py):
    return ((cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0
            and (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0
            and (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0)
