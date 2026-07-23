"""Exact CSG booleans on convex-faceted solids (K1).

BSP-based clipping in the classic csg.js structure, but with EXACT
rational classification: a vertex is FRONT/BACK/ON by the sign of an
exact expression, so coplanar faces, shared edges, and slivers take a
deterministic branch instead of an epsilon gamble. Splitting a convex
polygon by a plane yields convex polygons with exactly computed
intersection points; degenerate fragments vanish by exact area test.
Lineage rides through every split (Polygon.source is preserved).
"""

from __future__ import annotations

from forgekernel.brep import Polygon, Solid
from forgekernel.exact import Plane, add, dot, smul, sub

_COPLANAR, _FRONT, _BACK, _SPANNING = 0, 1, 2, 3


def _split(plane: Plane, poly: Polygon, cof: list, cob: list,
           front: list, back: list) -> None:
    sides = [plane.side(v) for v in poly.verts]
    kind = 0
    for s in sides:
        kind |= _FRONT if s > 0 else (_BACK if s < 0 else 0)
    if kind == _COPLANAR:
        (cof if dot(plane.n, poly.plane.n) > 0 else cob).append(poly)
    elif kind == _FRONT:
        front.append(poly)
    elif kind == _BACK:
        back.append(poly)
    else:
        f: list = []
        b: list = []
        n = len(poly.verts)
        for i in range(n):
            j = (i + 1) % n
            vi, vj = poly.verts[i], poly.verts[j]
            si, sj = sides[i], sides[j]
            if si >= 0:
                f.append(vi)
            if si <= 0:
                b.append(vi)
            if si * sj < 0:                      # strict crossing: exact t
                t = (plane.d - dot(plane.n, vi)) / dot(plane.n, sub(vj, vi))
                x = add(vi, smul(t, sub(vj, vi)))
                f.append(x)
                b.append(x)
        if len(f) >= 3:
            p = Polygon(f, poly.source, poly.plane)
            if p.area2() != 0:
                front.append(p)
        if len(b) >= 3:
            p = Polygon(b, poly.source, poly.plane)
            if p.area2() != 0:
                back.append(p)


class _Node:
    __slots__ = ("plane", "front", "back", "polys")

    def __init__(self, polys: list[Polygon] | None = None) -> None:
        self.plane: Plane | None = None
        self.front: _Node | None = None
        self.back: _Node | None = None
        self.polys: list[Polygon] = []
        if polys:
            self.build(polys)

    def build(self, polys: list[Polygon]) -> None:
        if not polys:
            return
        if self.plane is None:
            self.plane = polys[0].plane
        front: list[Polygon] = []
        back: list[Polygon] = []
        for p in polys:
            _split(self.plane, p, self.polys, self.polys, front, back)
        if front:
            if self.front is None:
                self.front = _Node()
            self.front.build(front)
        if back:
            if self.back is None:
                self.back = _Node()
            self.back.build(back)

    def invert(self) -> None:
        self.polys = [p.flipped() for p in self.polys]
        if self.plane is not None:
            self.plane = self.plane.flipped()
        if self.front:
            self.front.invert()
        if self.back:
            self.back.invert()
        self.front, self.back = self.back, self.front

    def clip_polygons(self, polys: list[Polygon]) -> list[Polygon]:
        if self.plane is None:
            return list(polys)
        front: list[Polygon] = []
        back: list[Polygon] = []
        for p in polys:
            _split(self.plane, p, front, back, front, back)
        front = self.front.clip_polygons(front) if self.front else front
        back = self.back.clip_polygons(back) if self.back else []
        return front + back

    def clip_to(self, other: "_Node") -> None:
        self.polys = other.clip_polygons(self.polys)
        if self.front:
            self.front.clip_to(other)
        if self.back:
            self.back.clip_to(other)

    def all_polygons(self) -> list[Polygon]:
        out = list(self.polys)
        if self.front:
            out += self.front.all_polygons()
        if self.back:
            out += self.back.all_polygons()
        return out


def union(a: Solid, b: Solid) -> Solid:
    na, nb = _Node(list(a.polys)), _Node(list(b.polys))
    na.clip_to(nb)
    nb.clip_to(na)
    nb.invert()
    nb.clip_to(na)
    nb.invert()
    na.build(nb.all_polygons())
    return Solid(na.all_polygons())


def cut(a: Solid, b: Solid) -> Solid:
    na, nb = _Node(list(a.polys)), _Node(list(b.polys))
    na.invert()
    na.clip_to(nb)
    nb.clip_to(na)
    nb.invert()
    nb.clip_to(na)
    nb.invert()
    na.build(nb.all_polygons())
    na.invert()
    return Solid(na.all_polygons())


def intersect(a: Solid, b: Solid) -> Solid:
    na, nb = _Node(list(a.polys)), _Node(list(b.polys))
    na.invert()
    nb.clip_to(na)
    nb.invert()
    na.clip_to(nb)
    nb.clip_to(na)
    na.build(nb.all_polygons())
    na.invert()
    return Solid(na.all_polygons())
