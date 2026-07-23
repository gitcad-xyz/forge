"""K3.4 — STEP AP214 (ISO 10303-21) geometry reader → nurbs objects.

Reads B-spline curves and surfaces out of a STEP file into
``BSplineCurve``/``BSplineSurface`` with **exact** control data: STEP
writes reals as decimal text, and decimal text converts to ``Fraction``
without any loss — so a STEP import into forge is *exact by
construction*, byte-for-byte faithful to what the file says. (A float
kernel rounds the same text to 53 bits on read.)

Handles the simple entities::

    B_SPLINE_CURVE_WITH_KNOTS(...)
    B_SPLINE_SURFACE_WITH_KNOTS(...)

and the complex (multi-leaf) rational forms::

    ( BOUNDED_SURFACE() B_SPLINE_SURFACE(...) B_SPLINE_SURFACE_WITH_KNOTS
      (...) ... RATIONAL_B_SPLINE_SURFACE((weights...)) ... )

Anything else in the file is ignored — this is a geometry reader, not a
topology reader (that arrives with K3.5 shells).
"""

from __future__ import annotations

import re
from fractions import Fraction

from forgekernel.nurbs import BSplineCurve, BSplineSurface

F = Fraction


def _num(tok: str) -> Fraction:
    """Exact decimal→rational (STEP reals are decimal text)."""
    t = tok.strip()
    if t.endswith("."):
        t += "0"
    return F(t)


def _split_args(s: str) -> list[str]:
    """Split a STEP argument list at top-level commas."""
    out, depth, cur, in_str = [], 0, [], False
    for ch in s:
        if in_str:
            cur.append(ch)
            if ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def _parse_list(s: str) -> list[str]:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"expected a STEP list, got {s[:40]!r}")
    return _split_args(s[1:-1])


def _expand_knots(mults: list[int], knots: list[Fraction]) -> list[Fraction]:
    out: list[Fraction] = []
    for m, k in zip(mults, knots):
        out.extend([k] * m)
    return out


class StepFile:
    """A parsed Part 21 DATA section: entity id → (type(s), argument text)."""

    def __init__(self, text: str) -> None:
        self.entities: dict[int, tuple[list[str], str]] = {}
        data = text
        m = re.search(r"DATA;(.*?)ENDSEC;", text, re.S)
        if m:
            data = m.group(1)
        # one entity per statement: #id = <body> ;
        for em in re.finditer(r"#(\d+)\s*=\s*(.*?);", data, re.S):
            eid = int(em.group(1))
            body = em.group(2).strip().replace("\n", " ")
            if body.startswith("("):
                # complex entity: ( LEAF1(args) LEAF2(args) ... )
                leaves = re.findall(r"([A-Z_0-9]+)\s*\(", body)
                self.entities[eid] = (leaves, body)
            else:
                tm = re.match(r"([A-Z_0-9]+)\s*\((.*)\)\s*$", body, re.S)
                if tm:
                    self.entities[eid] = ([tm.group(1)], tm.group(2))

    # -- points ---------------------------------------------------------------

    def point(self, ref: str):
        eid = int(ref.strip().lstrip("#"))
        types, args = self.entities[eid]
        if "CARTESIAN_POINT" not in types:
            raise ValueError(f"#{eid} is not a CARTESIAN_POINT")
        if len(types) == 1:
            parts = _split_args(args)
            coords = _parse_list(parts[1])
        else:  # complex form: find the CARTESIAN_POINT leaf's list
            m = re.search(r"CARTESIAN_POINT\s*\(([^)]*\([^)]*\))", args)
            coords = _parse_list(_split_args(m.group(1))[-1])
        vals = [_num(c) for c in coords]
        while len(vals) < 3:
            vals.append(F(0))
        return tuple(vals[:3])

    # -- curves ---------------------------------------------------------------

    def _leaf_args(self, body: str, leaf: str) -> str:
        """Argument text of one leaf inside a complex entity body."""
        i = body.index(leaf) + len(leaf)
        while body[i] != "(":
            i += 1
        depth, j = 0, i
        while True:
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        return body[i + 1:j]

    def curve(self, eid: int) -> BSplineCurve:
        types, body = self.entities[eid]
        if types == ["B_SPLINE_CURVE_WITH_KNOTS"]:
            a = _split_args(body)
            degree = int(a[1])
            cps = [self.point(r) for r in _parse_list(a[2])]
            mults = [int(x) for x in _parse_list(a[6])]
            knots = [_num(x) for x in _parse_list(a[7])]
            return BSplineCurve(degree, cps, _expand_knots(mults, knots))
        if "B_SPLINE_CURVE_WITH_KNOTS" in types:      # complex → rational
            ka = _split_args(self._leaf_args(body, "B_SPLINE_CURVE_WITH_KNOTS"))
            ba = _split_args(self._leaf_args(body, "B_SPLINE_CURVE"))
            degree = int(ba[0])
            cps = [self.point(r) for r in _parse_list(ba[1])]
            # WITH_KNOTS leaf: (mults, knots, spec)
            mults = [int(x) for x in _parse_list(ka[0])]
            knots = [_num(x) for x in _parse_list(ka[1])]
            w = [_num(x) for x in _parse_list(
                self._leaf_args(body, "RATIONAL_B_SPLINE_CURVE"))]
            return BSplineCurve(degree, cps, _expand_knots(mults, knots), w)
        raise ValueError(f"#{eid} is not a B-spline curve")

    # -- surfaces -------------------------------------------------------------

    def surface(self, eid: int) -> BSplineSurface:
        types, body = self.entities[eid]
        if types == ["B_SPLINE_SURFACE_WITH_KNOTS"]:
            a = _split_args(body)
            du, dv = int(a[1]), int(a[2])
            net = [[self.point(r) for r in _parse_list(row)]
                   for row in _parse_list(a[3])]
            umult = [int(x) for x in _parse_list(a[8])]
            vmult = [int(x) for x in _parse_list(a[9])]
            uk = [_num(x) for x in _parse_list(a[10])]
            vk = [_num(x) for x in _parse_list(a[11])]
            return BSplineSurface(du, dv, net, _expand_knots(umult, uk),
                                  _expand_knots(vmult, vk))
        if "B_SPLINE_SURFACE_WITH_KNOTS" in types:    # complex → rational
            ba = _split_args(self._leaf_args(body, "B_SPLINE_SURFACE"))
            du, dv = int(ba[0]), int(ba[1])
            net = [[self.point(r) for r in _parse_list(row)]
                   for row in _parse_list(ba[2])]
            ka = _split_args(self._leaf_args(body, "B_SPLINE_SURFACE_WITH_KNOTS"))
            umult = [int(x) for x in _parse_list(ka[0])]
            vmult = [int(x) for x in _parse_list(ka[1])]
            uk = [_num(x) for x in _parse_list(ka[2])]
            vk = [_num(x) for x in _parse_list(ka[3])]
            wrows = [[_num(x) for x in _parse_list(row)] for row in _parse_list(
                self._leaf_args(body, "RATIONAL_B_SPLINE_SURFACE"))]
            return BSplineSurface(du, dv, net, _expand_knots(umult, uk),
                                  _expand_knots(vmult, vk), wrows)
        raise ValueError(f"#{eid} is not a B-spline surface")

    # -- discovery ------------------------------------------------------------

    def bspline_curves(self) -> list[int]:
        return [e for e, (t, _) in sorted(self.entities.items())
                if "B_SPLINE_CURVE_WITH_KNOTS" in t]

    def bspline_surfaces(self) -> list[int]:
        return [e for e, (t, _) in sorted(self.entities.items())
                if "B_SPLINE_SURFACE_WITH_KNOTS" in t]


def read_step_geometry(text: str) -> dict:
    """All B-spline geometry in a STEP file, exactly."""
    sf = StepFile(text)
    return {"curves": [sf.curve(e) for e in sf.bspline_curves()],
            "surfaces": [sf.surface(e) for e in sf.bspline_surfaces()]}


# -- K3.6: planar-solid topology import (MANIFOLD_SOLID_BREP → Solid) ---------

def _refs(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"#(\d+)", text)]


def _newell(loop) -> tuple:
    """Exact Newell normal of a 3D polygon (rational)."""
    nx = ny = nz = F(0)
    n = len(loop)
    for i in range(n):
        (x1, y1, z1), (x2, y2, z2) = loop[i], loop[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return (nx, ny, nz)


class _Topo(StepFile):
    """Topology walk for planar-faced solids: solid → shell → faces →
    bounds → edge loops → vertices. Straight edges only; freeform faces
    or inner bounds (holes) refuse with the stage name."""

    def planar_solid_faces(self, solid_eid: int):
        types, args = self.entities[solid_eid]
        if "MANIFOLD_SOLID_BREP" not in types:
            raise ValueError(f"#{solid_eid} is not a MANIFOLD_SOLID_BREP")
        shell = _refs(_split_args(args)[1])[0]
        _, sargs = self.entities[shell]
        faces = []
        for feid in _refs(_split_args(sargs)[1]):
            ftypes, fargs = self.entities[feid]
            if "ADVANCED_FACE" not in ftypes and "FACE_SURFACE" not in ftypes:
                continue
            fa = _split_args(fargs)
            bounds = _refs(fa[1])
            surf_eid = _refs(fa[2])[0]
            same_sense = fa[3].strip() == ".T."
            stypes, _ = self.entities[surf_eid]
            if "PLANE" not in stypes:
                raise ValueError(
                    "import_step: freeform face topology (arrives at K3.7)")
            if len(bounds) != 1:
                raise ValueError(
                    "import_step: face with inner bounds/holes (K3.7)")
            btypes, bargs = self.entities[bounds[0]]
            ba = _split_args(bargs)
            loop_eid = _refs(ba[1])[0]
            loop = self._vertex_loop(loop_eid)
            # orient by GEOMETRY, not flag interpretation: the face's
            # outward normal is the PLANE's axis direction (negated when
            # same_sense = .F.); flip the loop until its exact Newell
            # normal agrees. Robust to writer flag conventions.
            n_srf = self._plane_normal(surf_eid)
            if not same_sense:
                n_srf = tuple(-c for c in n_srf)
            nw = _newell(loop)
            if sum(nw[c] * n_srf[c] for c in range(3)) < 0:
                loop = list(reversed(loop))
            faces.append(loop)
        return faces

    def _plane_normal(self, surf_eid: int):
        """Exact axis direction of a PLANE's AXIS2_PLACEMENT_3D."""
        _, sargs = self.entities[surf_eid]
        place_eid = _refs(_split_args(sargs)[1])[0]
        _, pargs = self.entities[place_eid]
        pa = _split_args(pargs)          # (name, #origin, #axis, #ref_dir)
        axis_eid = _refs(pa[2])[0]
        _, dargs = self.entities[axis_eid]
        return tuple(_num(c) for c in _parse_list(_split_args(dargs)[1]))

    def _vertex_loop(self, loop_eid: int):
        _, largs = self.entities[loop_eid]
        pts = []
        for oe in _refs(_parse_list(_split_args(largs)[1].strip()
                                    if False else _split_args(largs)[1])[0]) \
                if False else _refs(_split_args(largs)[1]):
            otypes, oargs = self.entities[oe]
            oa = _split_args(oargs)
            edge_eid = _refs(oa[3])[0]
            forward = oa[4].strip() == ".T."
            _, eargs = self.entities[edge_eid]
            ea = _split_args(eargs)
            v1, v2 = _refs(ea[1])[0], _refs(ea[2])[0]
            start = v1 if forward else v2
            _, vargs = self.entities[start]
            pts.append(self.point(_split_args(vargs)[1]))
        return pts

    def solids(self) -> list[int]:
        return [e for e, (t, _) in sorted(self.entities.items())
                if "MANIFOLD_SOLID_BREP" in t]


def read_step_planar_solid(text: str):
    """Import the first planar-faced solid in a STEP file as an exact
    forge Solid (coordinates via lossless decimal→rational). Refuses
    freeform faces and holes with their stage names."""
    from forgekernel.brep import Polygon, Solid

    topo = _Topo(text)
    sids = topo.solids()
    if not sids:
        raise ValueError("import_step: no MANIFOLD_SOLID_BREP in file")
    loops = topo.planar_solid_faces(sids[0])
    polys = [Polygon([tuple(p) for p in loop], f"step.face{i}")
             for i, loop in enumerate(loops)]
    s = Solid(polys)
    if s.volume() < 0:
        s = Solid([p.flipped() for p in polys])
    if s.volume() <= 0:
        raise ValueError("import_step: could not orient the shell")
    return s
