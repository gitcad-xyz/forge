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
