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
        # One entity per statement: #id = <body> ;  — the terminating ';'
        # must be found OUTSIDE quoted strings: a lazy .*?; used to split
        # at a ';' inside a name (PRODUCT('a;b')), resync mid-string and
        # silently DROP the entity. A body is a run of quoted strings
        # ('...' — STEP escapes a quote by doubling it, so 'a''b' is two
        # adjacent string tokens and stays matched) and non-quote,
        # non-semicolon characters.
        for em in re.finditer(r"#(\d+)\s*=\s*((?:'[^']*'|[^;'])*);", data):
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
            # basis-curve guard: the planar walk reads only vertex points,
            # so a curved EDGE_CURVE (arc, B-spline) would be silently
            # replaced by its chord — a wrong volume wearing an exact
            # costume. Refuse the curve family by name instead.
            basis_eid = _refs(ea[3])[0]
            ctypes, _ = self.entities[basis_eid]
            if "LINE" not in ctypes:
                raise ValueError(
                    "import_step: curved edge on the planar path — "
                    f"EDGE_CURVE #{edge_eid} basis #{basis_eid} is "
                    f"{'/'.join(ctypes)}, not LINE; chording it would "
                    "silently change the volume (curved-edge topology "
                    "arrives at K3.7)")
            v1, v2 = _refs(ea[1])[0], _refs(ea[2])[0]
            start = v1 if forward else v2
            _, vargs = self.entities[start]
            pts.append(self.point(_split_args(vargs)[1]))
        return pts

    def solids(self) -> list[int]:
        return [e for e, (t, _) in sorted(self.entities.items())
                if "MANIFOLD_SOLID_BREP" in t]


def read_step_planar_solid(text: str, *, heal_tolerance=None,
                           report: dict | None = None):
    """Import the first planar-faced solid in a STEP file as an exact
    forge Solid (coordinates via lossless decimal→rational). Refuses
    freeform faces and holes with their stage names.

    The border is audited (#135): closure is established BEFORE the
    orientation flip — on an open shell the volume sign is origin-dependent,
    so flipping by it first used to invert every face of a far-from-origin
    open shell. A shell that does not close refuses with
    :class:`~forgekernel.brep.NonClosedShellError` carrying a gap report in
    user millimetres — unless ``heal_tolerance`` (opt-in, the caller's
    recorded intent) authorizes an exact vertex merge, which repairs tears
    but can never invent a missing face. ``report`` (a dict, filled in
    place) receives ``dropped`` entries (degenerate faces, extra solids in a
    multi-solid file) and the ``healed`` certificate."""
    from forgekernel.brep import (NonClosedShellError, Polygon, Solid,
                                  _area_vec, boundary_gap_report,
                                  open_boundary_segments, snap_vertices)
    from forgekernel.exact import dot as _dot
    from forgekernel.exact import is_zero as _is_zero
    from forgekernel.exact import sub as _sub

    rep = report if report is not None else {}
    topo = _Topo(text)
    sids = topo.solids()
    if not sids:
        raise ValueError("import_step: no MANIFOLD_SOLID_BREP in file")
    if len(sids) > 1:
        # a multi-solid file used to import only its first solid with the
        # rest silently absent — the drop now lands in the report
        rep.setdefault("dropped", []).extend(
            f"MANIFOLD_SOLID_BREP #{e}: multi-solid file — only the first "
            f"solid imports (compounds arrive at K3.7)" for e in sids[1:])
    loops = topo.planar_solid_faces(sids[0])
    polys: list = []
    for i, loop in enumerate(loops):
        pts = [tuple(p) for p in loop]
        ded = [v for j, v in enumerate(pts) if v != pts[j - 1]]
        area = _area_vec(ded) if len(ded) >= 3 else None
        if area is None or _is_zero(area):
            # a degenerate loop is a DROP, reported — not Polygon's raw
            # "collinear points" crash and not a silent Solid.__init__ filter
            rep.setdefault("dropped", []).append(
                f"step.face{i}: degenerate face loop (zero area) dropped")
            continue
        # exact planarity: the loop must lie in ONE plane. The reader never
        # checks the declared PLANE's origin, so a bent quad — 3 um of sag —
        # used to import clean and silently mean whatever the first three
        # vertices said. Refuse by name instead (exact test, no epsilon).
        v0 = ded[0]
        sag = [j for j, v in enumerate(ded) if _dot(_sub(v, v0), area) != 0]
        if sag:
            raise ValueError(
                f"import_step: non-planar face loop (step.face{i}: vertex "
                f"{tuple(float(c) for c in ded[sag[0]])} is exactly off the "
                f"loop's plane) — forge planar import cannot represent a "
                f"bent loop; repair or re-export from the source system "
                f"(freeform faces arrive at K3.7)")
        polys.append(Polygon(ded, f"step.face{i}"))
    # -- closure audit at the border, BEFORE any orientation decision ---------
    segs = open_boundary_segments(polys)
    healed = None
    if segs and heal_tolerance is not None:
        polys2, healed = snap_vertices(polys, heal_tolerance)
        segs2 = open_boundary_segments(polys2)
        if segs2:
            gr = boundary_gap_report(segs2)
            raise NonClosedShellError(
                f"import_step: shell still does not close after healing "
                f"(moved {healed['moved']} vertices; a vertex merge cannot "
                f"invent a missing face): {gr['open_edges']} open boundary "
                f"edges, {gr['open_perimeter_mm']:.6g} mm open perimeter",
                segments=segs2, report=gr, healed=healed)
        for p2 in polys2:
            # a merge can bend a quad out of plane; shipping it would trade
            # an open shell for a silently-wrong closed one
            a2 = _area_vec(p2.verts)
            if any(_dot(_sub(v, p2.verts[0]), a2) != 0 for v in p2.verts):
                raise ValueError(
                    f"import_step: non-planar face loop ({p2.source}: the "
                    f"heal bent this face out of its exact plane) — lower "
                    f"heal_tolerance or repair in the source system")
        polys = polys2
        rep["healed"] = healed
        rep.setdefault("dropped", []).extend(
            f"{src}: face collapsed by heal and was dropped"
            for src in healed["dropped_faces"])
    elif segs:
        gr = boundary_gap_report(segs)
        raise NonClosedShellError(
            f"import_step: the shell does not close — {gr['open_edges']} "
            f"open boundary edges, {gr['open_perimeter_mm']:.6g} mm open "
            f"perimeter"
            + (f", crack width {gr['max_gap_mm']:.6g} mm"
               if gr["max_gap_mm"] is not None else ""),
            segments=segs, report=gr)
    s = Solid(polys)
    # orientation by volume sign is SOUND here: the shell is closed, so the
    # sign is origin-independent
    if s.volume() < 0:
        s = Solid([p.flipped() for p in polys])
    if s.volume() <= 0:
        raise ValueError("import_step: could not orient the shell")
    return s


# -- K7.0c: native STEP AP214 export (planar solids) --------------------------

def _dec(x: Fraction, digits: int = 15) -> str:
    """Rational → STEP decimal real. Exact for terminating rationals;
    high-precision rounded otherwise (STEP is a decimal interchange
    format — this is the honest boundary of an exact→float export)."""
    from decimal import Decimal, getcontext
    getcontext().prec = digits + 6
    d = (Decimal(x.numerator) / Decimal(x.denominator))
    s = format(d.normalize(), "f")
    if "." not in s:
        s += "."
    return s


def _real(x) -> str:
    """A STEP REAL literal.

    Part 21 types ``4`` as an INTEGER, so a direction ratio or a radius
    written without a decimal point is a type error even though the number is
    right — strict readers reject the file. Almost every direction in a
    mechanical part is a whole number, so this is the common case, not an
    edge case.
    """
    s = f"{float(x):.15g}"
    if "e" in s or "E" in s:
        mant, _, exp = s.replace("E", "e").partition("e")
        return f"{mant if '.' in mant else mant + '.'}E{exp}"
    return s if "." in s else s + "."


def _unit3(v):
    """Float unit vector of an exact rational direction."""
    import math
    f = [float(c) for c in v]
    n = math.sqrt(sum(c * c for c in f)) or 1.0
    return (f[0] / n, f[1] / n, f[2] / n), n


def _perp(n):
    """A float unit vector orthogonal to n (for the plane's ref dir)."""
    import math
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    d = a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
    p = (a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2])
    m = math.sqrt(sum(c * c for c in p)) or 1.0
    return (p[0] / m, p[1] / m, p[2] / m)


def _poly_contains_xy(verts, px, py) -> bool:
    """Ray parity in xy for a face polygon (used to decide which cap fragment
    a bore breaks through)."""
    inside = False
    n = len(verts)
    for i in range(n):
        (x1, y1) = verts[i][0], verts[i][1]
        (x2, y2) = verts[(i + 1) % n][0], verts[(i + 1) % n][1]
        if (y1 > py) != (y2 > py):
            if px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def write_step_planar_solid(solid, *, name: str = "gitcad_part",
                            bores=()) -> str:
    """Emit a forge solid as a valid AP214 STEP file (full product structure +
    MANIFOLD_SOLID_BREP with shared edges).

    ``solid`` supplies the planar faces; ``bores`` is an optional list of z-axis
    ``Cyl`` bores drilled through it. A bore contributes a CYLINDRICAL_SURFACE
    wall (as two half-faces, so no periodic seam is needed), a circular inner
    bound on each cap it breaks through, and a flat disk face where it ends
    blind. Circles and cylinders are emitted as EXACT analytic surfaces, not
    faceted — a drilled plate exports as a real cylindrical hole that any CAD
    system reads back as a hole."""
    from forgekernel.brep import (NonClosedShellError, boundary_gap_report,
                                  open_boundary_segments)

    # #135: this writer used to emit CLOSED_SHELL unconditionally — an open
    # shell went out wearing a closed label and the lie propagated to the
    # next reader. Audit first; a refusal beats a well-formed falsehood.
    open_segs = open_boundary_segments(solid.polys)
    if open_segs:
        gr = boundary_gap_report(open_segs)
        raise NonClosedShellError(
            f"export refuses to emit CLOSED_SHELL over an open shell: "
            f"{gr['open_edges']} open boundary edges, "
            f"{gr['open_perimeter_mm']:.6g} mm open perimeter",
            segments=open_segs, report=gr)
    lines: list[str] = []
    nid = [0]

    def emit(body: str) -> int:
        nid[0] += 1
        lines.append(f"#{nid[0]} = {body};")
        return nid[0]

    fnum = _real

    # --- product + context chain (required for OCCT to transfer a solid)
    appctx = emit("APPLICATION_CONTEXT('automotive_design')")
    emit(f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
         f"'automotive_design',2000,#{appctx})")
    pctx = emit(f"PRODUCT_CONTEXT('',#{appctx},'mechanical')")
    pdctx = emit(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{appctx},"
                 f"'design')")
    prod = emit(f"PRODUCT('{name}','{name}','',(#{pctx}))")
    emit(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part','',(#{prod}))")
    pdf = emit(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    pdef = emit(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})")
    pds = emit(f"PRODUCT_DEFINITION_SHAPE('','',#{pdef})")
    # units + geometric context
    lu = emit("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    au = emit("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = emit("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = emit(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{lu},"
               f"'distance_accuracy_value','')")
    ctx = emit(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
               f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
               f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))"
               f"REPRESENTATION_CONTEXT('',''))")

    vids: dict = {}

    def vertex(v):
        key = (v[0], v[1], v[2])
        if key not in vids:
            cp = emit(f"CARTESIAN_POINT('',({_dec(v[0])},{_dec(v[1])},"
                      f"{_dec(v[2])}))")
            vids[key] = emit(f"VERTEX_POINT('',#{cp})")
        return vids[key]

    edges: dict = {}

    def edge_curve(a, b):
        ka, kb = (a[0], a[1], a[2]), (b[0], b[1], b[2])
        key = frozenset((ka, kb))
        if key not in edges:
            va, vb = vertex(a), vertex(b)
            p0 = emit(f"CARTESIAN_POINT('',({_dec(a[0])},{_dec(a[1])},"
                      f"{_dec(a[2])}))")
            (dx, dy, dz), ln = _unit3((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
            dr = emit(f"DIRECTION('',({fnum(dx)},{fnum(dy)},{fnum(dz)}))")
            vec = emit(f"VECTOR('',#{dr},{fnum(ln)})")
            crv = emit(f"LINE('',#{p0},#{vec})")
            edges[key] = (emit(f"EDGE_CURVE('',#{va},#{vb},#{crv},.T.)"),
                          ka, kb)
        return edges[key]

    def zplacement(cx, cy, z, up=1):
        """AXIS2_PLACEMENT_3D on the +z (or −z) axis through (cx, cy, z),
        ref direction +x — the frame every circle and cylinder here uses."""
        o = emit(f"CARTESIAN_POINT('',({_dec(cx)},{_dec(cy)},{_dec(z)}))")
        ax = emit(f"DIRECTION('',(0.,0.,{'1.' if up > 0 else '-1.'}))")
        rd = emit(f"DIRECTION('',(1.,0.,0.))")
        return emit(f"AXIS2_PLACEMENT_3D('',#{o},#{ax},#{rd})")

    def rim_edges(c, z):
        """The two half-circle EDGE_CURVEs of a bore's rim at height z, split at
        angles 0 and π so the cylindrical face needs no periodic seam. Cached,
        so the cap's inner bound and the wall share the very same edges."""
        key = ("rim", c.cx, c.cy, c.r, z)
        if key in edges:
            return edges[key]
        p0 = (c.cx + c.r, c.cy, z)
        p1 = (c.cx - c.r, c.cy, z)
        circ = emit(f"CIRCLE('',#{zplacement(c.cx, c.cy, z)},{_dec(c.r)})")
        e_a = emit(f"EDGE_CURVE('',#{vertex(p0)},#{vertex(p1)},#{circ},.T.)")
        e_b = emit(f"EDGE_CURVE('',#{vertex(p1)},#{vertex(p0)},#{circ},.T.)")
        edges[key] = (e_a, e_b, p0, p1)
        return edges[key]

    def oriented(eid, fwd):
        return emit(f"ORIENTED_EDGE('',*,*,#{eid},{'.T.' if fwd else '.F.'})")

    def loop_of(oes):
        return emit(f"EDGE_LOOP('',({','.join('#' + str(x) for x in oes)}))")

    bores = list(bores)
    face_ids = []
    for poly in solid.polys:
        vs = [tuple(v) for v in poly.verts]
        oe = []
        for i in range(len(vs)):
            a, b = vs[i], vs[(i + 1) % len(vs)]
            ec, ka, kb = edge_curve(a, b)
            fwd = (a[0], a[1], a[2]) == ka
            oe.append(oriented(ec, fwd))
        bound = emit(f"FACE_OUTER_BOUND('',#{loop_of(oe)},.T.)")
        bounds = [bound]
        # a z-cap gets a circular INNER bound for every bore that breaks it
        n = poly.plane.n
        if n[0] == 0 and n[1] == 0:
            zc = vs[0][2]
            for c in bores:
                if (c.z0 < zc < c.z1 or c.z0 == zc or c.z1 == zc) and \
                        _poly_contains_xy(vs, c.cx, c.cy):
                    a_e, b_e, _, _ = rim_edges(c, zc)
                    # each rim half-circle is shared with the bore wall, which
                    # traverses the LOWER rim forward and the UPPER rim
                    # backward; the cap must take the opposite sense so every
                    # edge is used once .T. and once .F. (manifold closure).
                    fwd = n[2] > 0                       # top cap vs bottom cap
                    hole = loop_of([oriented(a_e, fwd), oriented(b_e, fwd)])
                    bounds.append(emit(f"FACE_BOUND('',#{hole},.T.)"))
        (nx, ny, nz), _ = _unit3(poly.plane.n)
        rx, ry, rz = _perp((nx, ny, nz))
        origin = emit(f"CARTESIAN_POINT('',({_dec(vs[0][0])},{_dec(vs[0][1])},"
                      f"{_dec(vs[0][2])}))")
        axis = emit(f"DIRECTION('',({fnum(nx)},{fnum(ny)},{fnum(nz)}))")
        rdir = emit(f"DIRECTION('',({fnum(rx)},{fnum(ry)},{fnum(rz)}))")
        place = emit(f"AXIS2_PLACEMENT_3D('',#{origin},#{axis},#{rdir})")
        plane = emit(f"PLANE('',#{place})")
        blist = ",".join("#" + str(b) for b in bounds)
        face_ids.append(emit(f"ADVANCED_FACE('',({blist}),#{plane},.T.)"))

    # -- bore walls (two half-cylinder faces) + blind end caps ----------------
    (_, _, sz0), (_, _, sz1) = solid.bbox()
    for c in bores:
        z0, z1 = max(c.z0, sz0), min(c.z1, sz1)
        if z1 <= z0:
            continue
        lo_a, lo_b, lp0, lp1 = rim_edges(c, z0)
        hi_a, hi_b, hp0, hp1 = rim_edges(c, z1)
        seam0 = edge_curve(lp0, hp0)[0]         # the two vertical seam edges
        seam1 = edge_curve(lp1, hp1)[0]
        cyl = emit(f"CYLINDRICAL_SURFACE('',#{zplacement(c.cx, c.cy, z0)},"
                   f"{_dec(c.r)})")
        for lo_e, hi_e, s_from, s_to in ((lo_a, hi_a, seam1, seam0),
                                         (lo_b, hi_b, seam0, seam1)):
            oes = [oriented(lo_e, True), oriented(s_from, True),
                   oriented(hi_e, False), oriented(s_to, False)]
            b = emit(f"FACE_OUTER_BOUND('',#{loop_of(oes)},.T.)")
            # material lies OUTSIDE a bore, so the solid's outward normal points
            # toward the axis — opposite the cylinder's own normal: .F.
            face_ids.append(emit(f"ADVANCED_FACE('',(#{b}),#{cyl},.F.)"))
        # A blind end gets a flat disk. Its outward normal points INTO the bore
        # (material is on the far side): +z at the floor, −z at the ceiling.
        # The loop sense is independent of that — it mirrors the wall, which
        # takes the lower rim forward and the upper rim backward.
        for zb, axis_up, fwd in ((z0, 1, False), (z1, -1, True)):
            if (zb == sz0 and c.z0 <= sz0) or (zb == sz1 and c.z1 >= sz1):
                continue                        # breaks through: no disk here
            a_e, b_e, _, _ = rim_edges(c, zb)
            disk = loop_of([oriented(a_e, fwd), oriented(b_e, fwd)])
            b = emit(f"FACE_OUTER_BOUND('',#{disk},.T.)")
            pl = emit(f"PLANE('',#{zplacement(c.cx, c.cy, zb, axis_up)})")
            face_ids.append(emit(f"ADVANCED_FACE('',(#{b}),#{pl},.T.)"))

    shell = emit(f"CLOSED_SHELL('',({','.join('#' + str(x) for x in face_ids)}))")
    brep = emit(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
    absr = emit(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{brep}),#{ctx})")
    emit(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{absr})")

    body = "\n".join(lines)
    header = (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_DESCRIPTION(('gitcad forge STEP export'),'2;1');\n"
        f"FILE_NAME('{name}.step','',(''),(''),'forgekernel','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n")
    return header + body + "\nENDSEC;\nEND-ISO-10303-21;\n"
