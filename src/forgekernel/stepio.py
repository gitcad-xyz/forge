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


def _emit_product_chain(emit, name: str) -> tuple[int, int]:
    """The AP214 product + context chain (required for OCCT to transfer a
    solid). Shared verbatim by both writers so their preambles cannot
    drift; returns (product_definition_shape, geometric_context)."""
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
    return pds, ctx


def _step_header(name: str, body: str) -> str:
    header = (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_DESCRIPTION(('gitcad forge STEP export'),'2;1');\n"
        f"FILE_NAME('{name}.step','',(''),(''),'forgekernel','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n")
    return header + body + "\nENDSEC;\nEND-ISO-10303-21;\n"


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

    pds, ctx = _emit_product_chain(emit, name)

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

    return _step_header(name, "\n".join(lines))


# -- K3.7: freeform-solid topology import (trimmed B-spline faces) ------------
#
# The planar walk's "arrives at K3.7" refusal retires here: a
# MANIFOLD_SOLID_BREP whose faces are B-spline surfaces (planar faces
# LIFT to bilinear patches) imports as a
# :class:`forgekernel.trimshell.TrimmedShell`. The mapping:
#
#   ADVANCED_FACE -> ShellFace(surface  = StepFile.surface(eid) / lift,
#                              sense    = +1 iff same_sense (audit-backed:
#                                         global orientation is re-decided
#                                         by certified volume sign),
#                              strip    = de Casteljau control-hull boxes
#                                         of curved pcurve pieces + chord
#                                         boxes off the domain border,
#                              inside   = exact even-odd parity vs the
#                                         face's chord loops)
#   FACE_OUTER_BOUND -> add_loop(..., outer=True); FACE_BOUND -> holes
#   one TrimVertex per STEP VERTEX_POINT entity (identity is the entity);
#   curved edges subdivide PER EDGE ENTITY at shared breakpoints (a
#   per-face walk creates T-junctions and the pairing audit refuses).
#
# Closure is established BEFORE any orientation decision (the #135
# order): chain breaks, pcurve-endpoint/vertex disagreements and
# unpaired trim edges collect into a NonClosedShellError carrying a gap
# report in user millimetres; heal_tolerance (recorded intent) merges
# VERTEX_POINT entities whose exact positions agree within tolerance —
# a vertex-identity merge that can repair a tear but never invent a face.
#
# Tiers (ADR-0019, never a bare float): polynomial surfaces whose trim
# pcurves are all degree 1 in (u, v) get ShellFace.tight = the EXACT
# span-safe trimmed_patch_flux -> width-0 volume brackets; curved pcurves
# and one-signed rational surfaces stay certified (hull enclosure /
# reciprocal rule); analytic surfaces, sign-varying weights, missing
# pcurves, VERTEX_LOOPs and periodic seams refuse by name.

from forgekernel.nurbs import _deboor4, _insert_knot_once
from forgekernel.nurbs import bezier_segments as _bezier_segments


def _bez_split(cps, s):
    """De Casteljau split of a Bezier control polygon at local s — exact;
    returns (left, right) control polygons whose hulls each CONTAIN their
    curve piece (the enclosure property every strip box rests on)."""
    left = [tuple(cps[0])]
    rights = [tuple(cps[-1])]
    work = [tuple(p) for p in cps]
    while len(work) > 1:
        work = [tuple((1 - s) * a[c] + s * b[c] for c in range(len(a)))
                for a, b in zip(work, work[1:])]
        left.append(work[0])
        rights.append(work[-1])
    return left, list(reversed(rights))


def _box2(cps):
    """(u0, u1, v0, v1) bounding box of 2D-embedded control points."""
    us = [p[0] for p in cps]
    vs = [p[1] for p in cps]
    return (min(us), max(us), min(vs), max(vs))


def _on_border(a, b, dom):
    """Does the segment a-b lie ON the domain border? (Border trims never
    enter the open domain, so they add nothing to the strip.)"""
    (u0, u1), (v0, v1) = dom
    if a[0] == b[0] and a[0] in (u0, u1):
        return True
    if a[1] == b[1] and a[1] in (v0, v1):
        return True
    return False


class _TrimTopo(StepFile):
    """K3.7 topology walk: freeform faces, pcurve chains, shared edge
    subdivision. Every refusal is by name; every number is exact."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._edges: dict[int, dict] = {}
        self._surf_shapes: dict[int, tuple] = {}

    def solids(self) -> list[int]:
        return [e for e, (t, _) in sorted(self.entities.items())
                if "MANIFOLD_SOLID_BREP" in t]

    # -- geometry helpers -----------------------------------------------------

    def point2(self, ref: str):
        """Exact 2D parameter-space CARTESIAN_POINT."""
        eid = int(str(ref).strip().lstrip("#"))
        types, args = self.entities[eid]
        if "CARTESIAN_POINT" not in types:
            raise ValueError(f"import_step: #{eid} is not a CARTESIAN_POINT")
        vals = [_num(c) for c in _parse_list(_split_args(args)[1])]
        if len(vals) < 2:
            raise ValueError(
                f"import_step: parameter-space point #{eid} has fewer than "
                f"two coordinates")
        if any(v != 0 for v in vals[2:]):
            raise ValueError(
                f"import_step: parameter-space point #{eid} carries a "
                f"nonzero third coordinate — not a (u, v) point")
        return vals[0], vals[1]

    def _direction(self, ref: str):
        eid = int(str(ref).strip().lstrip("#"))
        _, dargs = self.entities[eid]
        vals = [_num(c) for c in _parse_list(_split_args(dargs)[1])]
        while len(vals) < 3:
            vals.append(F(0))
        return tuple(vals[:3])

    def vertex_pos(self, veid: int):
        _, vargs = self.entities[veid]
        return self.point(_split_args(vargs)[1])

    def plane_frame(self, surf_eid: int):
        """(origin, axis, ref_dir) of a PLANE, exact as written."""
        _, sargs = self.entities[surf_eid]
        place = _refs(_split_args(sargs)[1])[0]
        _, pargs = self.entities[place]
        pa = _split_args(pargs)
        return (self.point(pa[1]), self._direction(pa[2]),
                self._direction(pa[3]))

    def pcurve2(self, eid: int) -> BSplineCurve:
        """A parameter-space trim curve as a 2D-embedded BSplineCurve.
        Rational pcurves refuse by name (a rational trim in (u, v) has no
        exact chord/hull story yet)."""
        types, body = self.entities[eid]
        if types == ["B_SPLINE_CURVE_WITH_KNOTS"]:
            a = _split_args(body)
            degree = int(a[1])
            cps = [self.point2(r) for r in _parse_list(a[2])]
            mults = [int(x) for x in _parse_list(a[6])]
            knots = [_num(x) for x in _parse_list(a[7])]
            return BSplineCurve(degree, [(u, v, F(0)) for (u, v) in cps],
                                _expand_knots(mults, knots))
        if "B_SPLINE_CURVE_WITH_KNOTS" in types:
            raise ValueError(
                f"import_step: rational pcurve #{eid} — rational "
                f"parameter-space trim curves are not supported yet")
        raise ValueError(
            f"import_step: pcurve geometry {'/'.join(types)} (#{eid}) — "
            f"only B-spline pcurves are supported")

    # -- edges ----------------------------------------------------------------

    def edge_rec(self, eid: int) -> dict:
        rec = self._edges.get(eid)
        if rec is not None:
            return rec
        types, args = self.entities[eid]
        if "EDGE_CURVE" not in types:
            raise ValueError(f"import_step: #{eid} is not an EDGE_CURVE")
        ea = _split_args(args)
        v1, v2 = _refs(ea[1])[0], _refs(ea[2])[0]
        basis = _refs(ea[3])[0]
        ss = ea[4].strip() == ".T."
        btypes, bargs = self.entities[basis]
        if "SEAM_CURVE" in btypes:
            raise ValueError(
                f"import_step: edge #{eid} rides a SEAM_CURVE — the seam "
                f"edge of a closed periodic surface refuses by name (split "
                f"the surface at the seam before export)")
        pcs: dict[int, BSplineCurve] = {}
        if "SURFACE_CURVE" in btypes or "BOUNDED_SURFACE_CURVE" in btypes:
            ba = _split_args(bargs)
            for ref in _refs(ba[2]):
                ptypes, pargs = self.entities[ref]
                if "PCURVE" not in ptypes:
                    continue
                pa = _split_args(pargs)
                surf = _refs(pa[1])[0]
                defrep = _refs(pa[2])[0]
                _, da = self.entities[defrep]
                c2 = _refs(_split_args(da)[1])[0]
                pcs[surf] = self.pcurve2(c2)
        curve3 = basis
        if "SURFACE_CURVE" in btypes or "BOUNDED_SURFACE_CURVE" in btypes:
            curve3 = _refs(_split_args(bargs)[1])[0]
        rec = {"v1": v1, "v2": v2, "ss": ss, "basis": basis,
               "curve3": curve3, "pcs": pcs, "breaks": None, "pts3": {}}
        self._edges[eid] = rec
        return rec

    def _surf_shape(self, surf_eid: int):
        """(p, q, interior_u_knots, interior_v_knots, rational) of the
        face geometry under a pcurve — a PLANE lifts as an affine
        (degree-(1, 1), knot-free, polynomial) map."""
        shape = self._surf_shapes.get(surf_eid)
        if shape is not None:
            return shape
        if "PLANE" in self.entities[surf_eid][0]:
            shape = (1, 1, (), (), False)
        else:
            s = self.surface(surf_eid)
            u0, u1 = s.U[s.p], s.U[s.nu]
            v0, v1 = s.V[s.q], s.V[s.nv]
            shape = (s.p, s.q,
                     tuple(sorted({k for k in s.U if u0 < k < u1})),
                     tuple(sorted({k for k in s.V if v0 < k < v1})),
                     any(w != 1 for row in s.w for w in row))
        self._surf_shapes[surf_eid] = shape
        return shape

    def edge_breaks(self, eid: int, subdiv: int):
        """The edge's SHARED breakpoint parameters — dense enough that
        exact cross-face agreement at every breakpoint CERTIFIES the two
        faces' composed border curves identical, never merely samples
        them.

        Per adjacent face, the border image t ↦ S(u(t), v(t)) is (on
        each span) a polynomial — or one-signed-rational — function of
        degree at most d_u·p + d_v·q, where d_u/d_v are the pcurve's
        coordinate degrees and (p, q) the surface degrees. Two such
        functions that agree at more exact rational parameters than the
        cross-multiplied degree bound are IDENTICAL on the span
        (polynomial identity in ℚ), so the breakpoint set carries, per
        span, bound+1 uniform rational samples. Span delimiters are the
        pcurve knots plus the exact parameters where a degree-1 pcurve
        crosses an interior surface knot line; a CURVED pcurve whose
        control hull spans an interior knot line refuses by name (its
        crossing parameter is algebraic — no exact certificate).

        Dyadic per-span refinement when any pcurve is curved is kept
        (unioned in) for strip-box tightness. Shared per EDGE ENTITY:
        both adjacent faces evaluate their own pcurve at the same
        parameters, so the loop vertices pair exactly (a per-face
        refinement mints T-junctions and the pairing audit refuses).
        ``None`` for a pcurve-less edge."""
        rec = self.edge_rec(eid)
        if rec["breaks"] is not None or not rec["pcs"]:
            return rec["breaks"]
        pcs = list(rec["pcs"].values())
        doms = {(pc.U[pc.p], pc.U[len(pc.cp)]) for pc in pcs}
        if len(doms) != 1:
            raise ValueError(
                f"import_step: the pcurves of edge #{eid} disagree about "
                f"the edge's parameter range — the file's representations "
                f"are inconsistent")
        lo, hi = doms.pop()
        knots = sorted({k for pc in pcs for k in pc.U if lo <= k <= hi})

        # certification: span delimiters + per-piece composed-degree bounds
        delims = set(knots)
        face_pieces: list[list] = []      # per face: [(a, b, Dnum, Dden)]
        for surf_eid, pc in rec["pcs"].items():
            p, q, int_u, int_v, rational = self._surf_shape(surf_eid)
            pieces = []
            for (a, b, cps) in _bezier_segments(pc):
                us = [c[0] for c in cps]
                vs = [c[1] for c in cps]
                d_u = 0 if min(us) == max(us) else pc.p
                d_v = 0 if min(vs) == max(vs) else pc.p
                bound = d_u * p + d_v * q
                for interior, ws in ((int_u, us), (int_v, vs)):
                    w0, w1 = min(ws), max(ws)
                    for k in interior:
                        if not (w0 < k < w1):
                            continue
                        if pc.p == 1:
                            # affine piece: exact rational crossing
                            t = a + (b - a) * (k - ws[0]) / (ws[-1] - ws[0])
                            if a < t < b:
                                delims.add(t)
                        else:
                            raise ValueError(
                                f"import_step: curved pcurve of edge "
                                f"#{eid} spans an interior knot of "
                                f"surface #{surf_eid} — the composed "
                                f"border curve breaks at an algebraic "
                                f"parameter with no exact certificate; "
                                f"split the pcurve at the surface knot "
                                f"lines before export")
                pieces.append((a, b, bound, bound if rational else 0))
            face_pieces.append(pieces)

        breaks_set = set(delims)
        dl = sorted(delims)
        for x, y in zip(dl, dl[1:]):
            bounds = []
            for pieces in face_pieces:
                for (a, b, dn, dd) in pieces:
                    if a <= x and y <= b:
                        bounds.append((dn, dd))
                        break
            if len(bounds) == 2:
                (n1, d1), (n2, d2) = bounds
                need = max(n1 + d2, n2 + d1)
            elif bounds:
                need = sum(bounds[0])
            else:
                need = 1
            for j in range(1, need):      # need+1 samples incl. endpoints
                breaks_set.add(x + (y - x) * j / F(need))

        if any(pc.p >= 2 for pc in pcs):
            n = 2 ** subdiv
            for a, b in zip(knots, knots[1:]):
                step = (b - a) / n
                breaks_set.update(a + step * i for i in range(1, n))
        rec["breaks"] = sorted(breaks_set)
        return rec["breaks"]

    def straight_border_certificate(self, eid: int, A, B) -> bool:
        """CERTIFIED image coincidence for a pointwise-mismatched shared
        edge. Two faces may parameterize the SAME border differently —
        e.g. a one-signed rational reparameterization of a straight
        border — and still close the shell: closure is about image SETS.
        This proves, exactly in ℚ, that every adjacent face's composed
        border image is EXACTLY the segment A–B: each composed border's
        (homogeneous) control points are computed exactly — degree-1
        axis-parallel pcurves compose to an iso-curve extracted by de
        Boor + Boehm restriction; a plane composes affinely — and must
        be collinear with A–B with segment coordinate s ∈ [0, 1]. With
        one-signed weights the hull property then pins the image inside
        the segment, and continuity from A to B covers it, so the image
        IS the segment for every face. Anything this cannot prove
        (curved or diagonal pcurves, off-line control points) returns
        ``False`` and the gap stands — a proof or a refusal, never a
        sample."""
        d = tuple(B[c] - A[c] for c in range(3))
        dd = sum(c * c for c in d)
        if dd == 0:
            return False

        def on_segment(P) -> bool:
            r = tuple(P[c] - A[c] for c in range(3))
            cx = (r[1] * d[2] - r[2] * d[1], r[2] * d[0] - r[0] * d[2],
                  r[0] * d[1] - r[1] * d[0])
            if any(c != 0 for c in cx):
                return False
            s = sum(r[c] * d[c] for c in range(3)) / dd
            return 0 <= s <= 1

        rec = self.edge_rec(eid)
        for surf_eid, pc in rec["pcs"].items():
            if pc.p != 1:
                return False
            if "PLANE" in self.entities[surf_eid][0]:
                # affine lift: piece images are segments between the
                # exact images of the piece's endpoints
                O, ax, rd = self.plane_frame(surf_eid)
                d2v = (ax[1] * rd[2] - ax[2] * rd[1],
                       ax[2] * rd[0] - ax[0] * rd[2],
                       ax[0] * rd[1] - ax[1] * rd[0])
                for (_a, _b, cps) in _bezier_segments(pc):
                    for (u, v, _z) in cps:
                        if not on_segment(tuple(
                                O[c] + u * rd[c] + v * d2v[c]
                                for c in range(3))):
                            return False
                continue
            s = self.surface(surf_eid)
            for (_a, _b, cps) in _bezier_segments(pc):
                (u0, v0), (u1, v1) = cps[0][:2], cps[1][:2]
                if u0 == u1:            # iso-curve in v at fixed u
                    deg, knots = s.q, list(s.V)
                    pts = [_deboor4(s.p, s.U,
                                    [s.H[i][j] for i in range(s.nu)], u0)
                           for j in range(s.nv)]
                    w0, w1 = min(v0, v1), max(v0, v1)
                elif v0 == v1:          # iso-curve in u at fixed v
                    deg, knots = s.p, list(s.U)
                    pts = [_deboor4(s.q, s.V, s.H[i], v0)
                           for i in range(s.nu)]
                    w0, w1 = min(u0, u1), max(u0, u1)
                else:
                    return False        # diagonal: no exact composition here
                pts = [tuple(p) for p in pts]
                for t in (w0, w1):      # Boehm restriction to the trim range
                    while knots.count(t) < deg:
                        knots, pts = _insert_knot_once(deg, knots, pts, t)
                for i, h in enumerate(pts):
                    # control points ACTIVE on [w0, w1] (a superset only
                    # loosens the hull — sound: it can refuse, not accept)
                    if not (knots[i] < w1 and knots[i + deg + 1] > w0):
                        continue
                    if h[3] == 0:
                        return False
                    if not on_segment((h[0] / h[3], h[1] / h[3],
                                       h[2] / h[3])):
                        return False
        return True

    def edge_on_face(self, eid: int, surf_eid: int):
        """(uv points at the shared breakpoints, hull boxes of curved
        pieces) for this edge's pcurve on this surface — points exact ON
        the trim curve, boxes provably ENCLOSING it (de Casteljau)."""
        rec = self.edge_rec(eid)
        pc = rec["pcs"][surf_eid]
        breaks = rec["breaks"]
        pts = [pc.eval(t)[:2] for t in breaks]
        boxes: list = []
        if pc.p >= 2:
            for (a, b, cps) in _bezier_segments(pc):
                inner = [t for t in breaks if a < t < b]
                cur, last = cps, a
                for t in inner:
                    s = (t - last) / (b - last)
                    left, cur = _bez_split(cur, s)
                    boxes.append(_box2(left))
                    last = t
                boxes.append(_box2(cur))
        return pts, boxes


def read_step_freeform_solid(text: str, *, heal_tolerance=None,
                             report: dict | None = None, depth: int = 4,
                             trim_subdiv: int = 3):
    """Import the first freeform (B-spline-faced) solid in a STEP file as
    an audited :class:`~forgekernel.trimshell.TrimmedShell`.

    Closure is established BEFORE any orientation decision (#135), and
    it is CERTIFIED, never sampled: adjacent faces' shared-edge images
    are proven coincident (exact agreement in ℚ at more breakpoints per
    span than the composed border curves' degree bound is a polynomial
    identity), or the import refuses with a gap report in millimetres —
    healable only by the recorded-intent vertex merge, which can repair
    a vertex tear but never a mid-edge crack. Global orientation is
    decided by the certified volume sign — audit-backed, never
    flag-trusted — and the full three-oracle audit runs at the door.
    Volume brackets are width-0 (exact tier) for polynomial faces with
    degree-1 pcurves, certified otherwise; refusals are by name."""
    import math

    from forgekernel.brep import (NonClosedShellError, SnapClusterError,
                                  boundary_gap_report)
    from forgekernel.bsolid import _parity_in, trimmed_patch_flux
    from forgekernel.interval import CInterval
    from forgekernel.trimshell import (ShellAuditUncertified, ShellFace,
                                       TrimmedShell, TrimVertex)

    rep = report if report is not None else {}
    topo = _TrimTopo(text)
    sids = topo.solids()
    if not sids:
        raise ValueError("import_step: no MANIFOLD_SOLID_BREP in file")
    if len(sids) > 1:
        rep.setdefault("dropped", []).extend(
            f"MANIFOLD_SOLID_BREP #{e}: multi-solid file — only the first "
            f"solid imports (compounds are a later stage)"
            for e in sids[1:])

    # -- pre-pass: face specs, surface-kind refusals, seam/loop refusals ------
    _, sargs = topo.entities[sids[0]]
    shell_eid = _refs(_split_args(sargs)[1])[0]
    _, shargs = topo.entities[shell_eid]
    face_specs: list[dict] = []
    for feid in _refs(_split_args(shargs)[1]):
        ftypes, fargs = topo.entities[feid]
        if "ADVANCED_FACE" not in ftypes and "FACE_SURFACE" not in ftypes:
            continue
        fa = _split_args(fargs)
        surf_eid = _refs(fa[2])[0]
        same_sense = fa[3].strip() == ".T."
        stypes = topo.entities[surf_eid][0]
        if ("B_SPLINE_SURFACE_WITH_KNOTS" not in stypes
                and "PLANE" not in stypes):
            raise ValueError(
                f"import_step: analytic/procedural surface "
                f"{'/'.join(stypes)} (#{surf_eid}) on a freeform shell — "
                f"an exact quadric needs irrational weights the rational "
                f"B-spline importer cannot hold; only B-spline and planar "
                f"faces import on this path")
        loops = []
        for beid in _refs(fa[1]):
            btypes, bargs = topo.entities[beid]
            outer = "FACE_OUTER_BOUND" in btypes
            if not outer and "FACE_BOUND" not in btypes:
                raise ValueError(
                    f"import_step: face bound #{beid} of type "
                    f"{'/'.join(btypes)} — only FACE_OUTER_BOUND/"
                    f"FACE_BOUND are supported")
            loop_eid = _refs(_split_args(bargs)[1])[0]
            ltypes, largs = topo.entities[loop_eid]
            if "VERTEX_LOOP" in ltypes:
                raise ValueError(
                    f"import_step: VERTEX_LOOP bound on face #{feid} — a "
                    f"degenerate (zero-length) trim bound refuses by name")
            if "EDGE_LOOP" not in ltypes:
                raise ValueError(
                    f"import_step: loop #{loop_eid} of type "
                    f"{'/'.join(ltypes)} — only EDGE_LOOP trims import")
            oedges = []
            for oe in _refs(_split_args(largs)[1]):
                otypes, oargs = topo.entities[oe]
                oa = _split_args(oargs)
                oedges.append((_refs(oa[3])[0], oa[4].strip() == ".T."))
            loops.append((outer, oedges))
        if not loops:
            raise ValueError(
                f"import_step: face #{feid} carries no trim bounds — an "
                f"unbounded face cannot close a shell")
        counts: dict[int, int] = {}
        for _outer, oedges in loops:
            for eid, _fwd in oedges:
                counts[eid] = counts.get(eid, 0) + 1
        for eid, n in counts.items():
            if n > 1:
                raise ValueError(
                    f"import_step: edge #{eid} is used twice by face "
                    f"#{feid} — the seam edge of a closed periodic "
                    f"surface refuses by name (split the surface at the "
                    f"seam before export)")
        face_specs.append({"feid": feid, "surf": surf_eid,
                           "same_sense": same_sense, "loops": loops})
    if not face_specs:
        raise ValueError("import_step: the CLOSED_SHELL carries no faces")

    # -- vertex identity: one TrimVertex per VERTEX_POINT entity; heal is
    #    the recorded-intent merge of entities within tolerance ---------------
    vset: set[int] = set()
    for spec in face_specs:
        for _outer, oedges in spec["loops"]:
            for eid, _fwd in oedges:
                r = topo.edge_rec(eid)
                vset.add(r["v1"])
                vset.add(r["v2"])
    vpos = {v: topo.vertex_pos(v) for v in vset}
    parent = {v: v for v in vset}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    healed = None
    tol2 = None
    if heal_tolerance is not None:
        tol = heal_tolerance if isinstance(heal_tolerance, Fraction) \
            else Fraction(str(heal_tolerance))
        tol2 = tol * tol
        vl = sorted(vset)
        for i, a in enumerate(vl):
            for b in vl[i + 1:]:
                d2 = sum((vpos[a][c] - vpos[b][c]) ** 2 for c in range(3))
                if d2 <= tol2:
                    parent[find(a)] = find(b)
        clusters: dict[int, list[int]] = {}
        for v in vset:
            clusters.setdefault(find(v), []).append(v)
        moved, max_move2 = 0, F(0)
        for _root, members in sorted(clusters.items()):
            if len(members) < 2:
                continue
            worst = max(
                sum((vpos[a][c] - vpos[b][c]) ** 2 for c in range(3))
                for i, a in enumerate(members) for b in members[i + 1:])
            if worst > tol2:
                raise SnapClusterError(
                    f"import_step: heal tolerance chains {len(members)} "
                    f"vertices into a cluster wider than itself",
                    cluster=[tuple(float(c) for c in vpos[m])
                             for m in members],
                    diameter_sq=worst, tolerance=tol)
            rep_pos = min(vpos[m] for m in members)
            for m in members:
                d2 = sum((vpos[m][c] - rep_pos[c]) ** 2 for c in range(3))
                if d2 > 0:
                    moved += 1
                    max_move2 = max(max_move2, d2)
        if moved:
            healed = {"tolerance": float(tol), "moved": moved,
                      "max_move_mm": math.sqrt(float(max_move2)),
                      # the freeform heal merges vertex IDENTITIES; every
                      # face's parameter-space trim is untouched, so the
                      # flux — hence the reported volume — cannot change
                      "volume_change_bound_mm3": 0.0,
                      "dropped_faces": []}

    # -- main pass: build one ShellFace per ADVANCED_FACE ---------------------
    verts_obj: dict = {}
    gap_segs: list[dict] = []
    faces = []

    def vert(key) -> TrimVertex:
        vx = verts_obj.get(key)
        if vx is None:
            vx = verts_obj[key] = TrimVertex()
        return vx

    for spec in face_specs:
        surf_eid = spec["surf"]
        stypes = topo.entities[surf_eid][0]
        planar = "PLANE" in stypes
        project = None
        if not planar:
            surface = topo.surface(surf_eid)
            ws = [w for row in surface.w for w in row]
            rational = any(w != 1 for w in ws)
            if rational and not (all(w > 0 for w in ws)
                                 or all(w < 0 for w in ws)):
                raise ValueError(
                    f"import_step: sign-varying weights on surface "
                    f"#{surf_eid} — a pole crosses the patch; refusing by "
                    f"name rather than bracketing through a singularity")
            dom = ((surface.U[surface.p], surface.U[surface.nu]),
                   (surface.V[surface.q], surface.V[surface.nv]))

            def eval3(u, v, _s=surface):
                return _s.eval(u, v)
        else:
            surface = None
            rational = False
            O, ax, rd = topo.plane_frame(surf_eid)
            has_pc = any(surf_eid in topo.edge_rec(eid)["pcs"]
                         for _o, oedges in spec["loops"]
                         for eid, _f in oedges)
            aa = sum(c * c for c in ax)
            rr = sum(c * c for c in rd)
            ar = sum(ax[c] * rd[c] for c in range(3))
            if has_pc and (aa != 1 or rr != 1 or ar != 0):
                raise ValueError(
                    f"import_step: PLANE #{surf_eid} carries pcurves but "
                    f"its placement is not an exact orthonormal frame — "
                    f"parameter-space trims cannot be mapped exactly")
            if has_pc:
                d1 = rd
            else:
                d1 = tuple(rd[c] * aa - ax[c] * ar for c in range(3))
                if all(c == 0 for c in d1):
                    raise ValueError(
                        f"import_step: PLANE #{surf_eid} ref direction is "
                        f"parallel to its axis — no frame")
            d2v_ = (ax[1] * d1[2] - ax[2] * d1[1],
                    ax[2] * d1[0] - ax[0] * d1[2],
                    ax[0] * d1[1] - ax[1] * d1[0])
            d11 = sum(c * c for c in d1)
            d22 = sum(c * c for c in d2v_)
            dom = None                       # bbox of the trim, found below

            def eval3(u, v, _O=O, _d1=d1, _d2=d2v_):
                return tuple(_O[c] + u * _d1[c] + v * _d2[c]
                             for c in range(3))

            def project(p, _O=O, _d1=d1, _d2=d2v_, _d11=d11, _d22=d22):
                r = tuple(p[c] - _O[c] for c in range(3))
                return (sum(r[c] * _d1[c] for c in range(3)) / _d11,
                        sum(r[c] * _d2[c] for c in range(3)) / _d22)

        floops = []                          # (outer, [(key, veid, uv)])
        chords: list = []                    # straight boundary pieces
        hull_boxes: list = []                # curved-piece enclosures
        curved_face = False
        for outer, oedges in spec["loops"]:
            entries: list = []
            first_info = None
            last_info = None
            for eid, fwd in oedges:
                rec = topo.edge_rec(eid)
                breaks = topo.edge_breaks(eid, trim_subdiv)
                v_lo = rec["v1"] if rec["ss"] else rec["v2"]
                v_hi = rec["v2"] if rec["ss"] else rec["v1"]
                if surf_eid in rec["pcs"]:
                    pts, boxes = topo.edge_on_face(eid, surf_eid)
                    if rec["pcs"][surf_eid].p >= 2:
                        curved_face = True
                        hull_boxes.extend(boxes)
                    else:
                        chords.extend(zip(pts, pts[1:]))
                    keys = ([("v", find(v_lo))]
                            + [("e", eid, t) for t in breaks[1:-1]]
                            + [("v", find(v_hi))])
                    vids_ = [v_lo] + [None] * (len(breaks) - 2) + [v_hi]
                    # cross-face agreement: both faces' pcurves must place
                    # every shared breakpoint at the SAME exact 3D point.
                    # edge_breaks makes this a CERTIFICATE, not a sample:
                    # per span it carries more exact rational parameters
                    # than the composed border curves' degree bound, so
                    # agreement everywhere proves the two images are one
                    # curve — and any disagreement is a REAL open slit,
                    # collected as a gap segment for the mm report (a
                    # vertex merge can never heal a mid-edge crack)
                    for t, uv in zip(breaks, pts):
                        p3 = eval3(uv[0], uv[1])
                        prev = rec["pts3"].get(t)
                        if prev is None:
                            rec["pts3"][t] = p3
                        elif prev != p3:
                            gap_segs.append({"a": prev, "b": p3,
                                             "coverage": 1, "edge": eid})
                else:
                    if not planar:
                        raise ValueError(
                            f"import_step: edge #{eid} has no pcurve on "
                            f"B-spline surface #{surf_eid} — curve "
                            f"inversion is not built; re-export with "
                            f"parameter-space pcurves")
                    ctypes = topo.entities[rec["curve3"]][0]
                    straight = "LINE" in ctypes
                    if not straight and "B_SPLINE_CURVE_WITH_KNOTS" in ctypes:
                        ca = _split_args(topo.entities[rec["curve3"]][1])
                        straight = int(ca[1]) <= 1
                    if not straight:
                        raise ValueError(
                            f"import_step: curved edge #{eid} "
                            f"({'/'.join(ctypes)}) without a pcurve on "
                            f"planar face #{spec['feid']} — projecting "
                            f"its chord would silently change the volume")
                    if breaks is not None and len(breaks) != 2:
                        raise ValueError(
                            f"import_step: edge #{eid} subdivides on an "
                            f"adjacent face but has no pcurve on planar "
                            f"face #{spec['feid']}")
                    pts = [project(vpos[v_lo]), project(vpos[v_hi])]
                    chords.append((pts[0], pts[1]))
                    keys = [("v", find(v_lo)), ("v", find(v_hi))]
                    vids_ = [v_lo, v_hi]
                # endpoint fidelity: the trim's corner must BE the vertex
                for uv, veid in ((pts[0], v_lo), (pts[-1], v_hi)):
                    p3 = eval3(uv[0], uv[1])
                    d2p = sum((p3[c] - vpos[veid][c]) ** 2
                              for c in range(3))
                    if d2p != 0 and not (tol2 is not None and d2p <= tol2):
                        gap_segs.append({"a": p3, "b": vpos[veid],
                                         "coverage": 1})
                seq = list(zip(keys, vids_, pts))
                if fwd != rec["ss"]:
                    seq.reverse()
                if last_info is not None and last_info[0] != seq[0][0]:
                    gap_segs.append({"a": vpos[last_info[1]],
                                     "b": vpos[seq[0][1]], "coverage": 1})
                if first_info is None:
                    first_info = (seq[0][0], seq[0][1])
                last_info = (seq[-1][0], seq[-1][1])
                entries.extend(seq[:-1])
            if last_info is not None and first_info is not None \
                    and last_info[0] != first_info[0]:
                gap_segs.append({"a": vpos[last_info[1]],
                                 "b": vpos[first_info[1]], "coverage": 1})
            dedup: list = []
            for item in entries:
                if not dedup or dedup[-1][0] != item[0]:
                    dedup.append(item)
            if len(dedup) > 1 and dedup[0][0] == dedup[-1][0]:
                dedup.pop()
            floops.append((outer, dedup))

        # domain: fixed for a B-spline surface (trim must stay inside);
        # a lifted plane takes the exact bbox of everything it must hold
        if planar:
            us = [uv[0] for _o, es in floops for _k, _v, uv in es]
            vs = [uv[1] for _o, es in floops for _k, _v, uv in es]
            for (bu0, bu1, bv0, bv1) in hull_boxes:
                us.extend((bu0, bu1))
                vs.extend((bv0, bv1))
            u0d, u1d = min(us), max(us)
            v0d, v1d = min(vs), max(vs)
            if u0d == u1d or v0d == v1d:
                raise ValueError(
                    f"import_step: planar face #{spec['feid']} has a "
                    f"degenerate (zero-extent) trim")
            dom = ((u0d, u1d), (v0d, v1d))
            surface = BSplineSurface(
                1, 1,
                [[eval3(u0d, v0d), eval3(u0d, v1d)],
                 [eval3(u1d, v0d), eval3(u1d, v1d)]],
                [u0d, u0d, u1d, u1d], [v0d, v0d, v1d, v1d])
        else:
            (u0d, u1d), (v0d, v1d) = dom
            for _o, es in floops:
                for _k, _v, uv in es:
                    if not (u0d <= uv[0] <= u1d and v0d <= uv[1] <= v1d):
                        raise ValueError(
                            f"import_step: trim of face #{spec['feid']} "
                            f"leaves the surface's parameter domain")

        strip = list(hull_boxes)
        for a, b in chords:
            if not _on_border(a, b, dom):
                strip.append((min(a[0], b[0]), max(a[0], b[0]),
                              min(a[1], b[1]), max(a[1], b[1])))

        loops_uv = [[uv for _k, _v, uv in es] for _o, es in floops]

        def inside(u, v, _L=loops_uv):
            return _parity_in(_L, u, v)

        face = ShellFace(surface, 1 if spec["same_sense"] else -1,
                         strip, inside)
        for (outer, es) in floops:
            face.add_loop([(vert(k), uv) for k, _v, uv in es], outer=outer)
        if not curved_face and not rational:
            # EXACT tier: every trim edge is its own chord, so the exact
            # span-safe Green's-theorem flux applies; the audit's strip
            # bracket intersects with it to width 0
            oriented = []
            for (outer, _es), uv_loop in zip(floops, loops_uv):
                area = F(0)
                m = len(uv_loop)
                for i in range(m):
                    (au, av) = uv_loop[i]
                    (bu, bv) = uv_loop[(i + 1) % m]
                    area += au * bv - bu * av
                if (area > 0) != outer:
                    uv_loop = list(reversed(uv_loop))
                oriented.append(uv_loop)
            face.tight = CInterval.exact(
                trimmed_patch_flux(surface, oriented))
        faces.append(face)

    # -- closure BEFORE orientation (the #135 order) --------------------------
    directed: dict = {}
    for fi, face in enumerate(faces):
        for a, b in face.outward_edges():
            directed.setdefault((id(a), id(b)), []).append((fi, a, b))
    seen = set()
    for (ia, ib), uses in sorted(directed.items()):
        key = (min(ia, ib), max(ia, ib))
        if key in seen:
            continue
        seen.add(key)
        total = len(uses) + len(directed.get((ib, ia), []))
        if total != 2:
            fi, a, b = uses[0]
            f = faces[fi]
            gap_segs.append({"a": f.surface.eval(*a.uv(f)),
                             "b": f.surface.eval(*b.uv(f)),
                             "coverage": total - 2 if total > 2 else 1})
    # a pointwise parameter mismatch is not yet a crack: two faces may
    # parameterize the SAME border differently and still close the shell.
    # Only a PROOF may drop the gap — the exact straight-segment
    # certificate on the edge's composed borders; everything unproven
    # stays in the report and refuses in millimetres
    if gap_segs:
        cert_cache: dict[int, bool] = {}
        kept = []
        for g in gap_segs:
            geid = g.pop("edge", None)
            if geid is not None:
                ok = cert_cache.get(geid)
                if ok is None:
                    r = topo.edge_rec(geid)
                    ok = cert_cache[geid] = topo.straight_border_certificate(
                        geid, vpos[r["v1"]], vpos[r["v2"]])
                if ok:
                    continue
            kept.append(g)
        gap_segs = kept
    if gap_segs:
        gr = boundary_gap_report(gap_segs)
        raise NonClosedShellError(
            "import_step: the freeform shell does not close"
            + (" (after healing — a vertex merge cannot invent a missing "
               "face)" if healed else "")
            + f" — {gr['open_edges']} open trim edges/vertex gaps, "
              f"{gr['open_perimeter_mm']:.6g} mm open perimeter",
            segments=gap_segs, report=gr, healed=healed)

    shell = TrimmedShell(faces)      # pairing-direction audit at the door

    # -- global orientation by certified volume sign, audit-backed ------------
    d_used = None
    for d in (depth, depth + 1, depth + 2):
        vol = shell.volume(depth=d)
        if vol.hi < 0:
            for f in faces:
                f.sense = -f.sense
            d_used = d
            break
        if vol.lo > 0:
            d_used = d
            break
    if d_used is None:
        raise ShellAuditUncertified(
            f"import_step: the volume bracket straddles zero up to depth "
            f"{depth + 2} — orientation is uncertified; raise depth "
            f"(tighten-or-refuse, never a guess)")

    last_exc = None
    audit = None
    for d in (d_used, d_used + 1):
        try:
            audit = shell.audit(depth=d)
            break
        except ShellAuditUncertified as exc:
            last_exc = exc
    if audit is None:
        raise last_exc

    if healed is not None:
        rep["healed"] = healed
    rep["faces"] = len(faces)
    rep["edges"] = audit["edges"]
    rep["volume_bracket_mm3"] = (float(audit["volume"].lo),
                                 float(audit["volume"].hi))
    return shell


def read_step_solid(text: str, *, heal_tolerance=None,
                    report: dict | None = None, depth: int = 4,
                    trim_subdiv: int = 3):
    """Route a STEP solid to its importer: all-PLANE faces take the exact
    planar path (a :class:`~forgekernel.brep.Solid`); any other surface
    routes the whole solid to :func:`read_step_freeform_solid` (a
    :class:`~forgekernel.trimshell.TrimmedShell`, or a refusal by
    name)."""
    sf = StepFile(text)
    sids = [e for e, (t, _) in sorted(sf.entities.items())
            if "MANIFOLD_SOLID_BREP" in t]
    if not sids:
        raise ValueError("import_step: no MANIFOLD_SOLID_BREP in file")
    _, args = sf.entities[sids[0]]
    shell_eid = _refs(_split_args(args)[1])[0]
    _, shargs = sf.entities[shell_eid]
    freeform = False
    for feid in _refs(_split_args(shargs)[1]):
        ftypes, fargs = sf.entities[feid]
        if "ADVANCED_FACE" not in ftypes and "FACE_SURFACE" not in ftypes:
            continue
        surf = _refs(_split_args(fargs)[2])[0]
        if "PLANE" not in sf.entities[surf][0]:
            freeform = True
            break
    if freeform:
        return read_step_freeform_solid(text, heal_tolerance=heal_tolerance,
                                        report=report, depth=depth,
                                        trim_subdiv=trim_subdiv)
    return read_step_planar_solid(text, heal_tolerance=heal_tolerance,
                                  report=report)


# -- K3.7: native STEP export of a PatchSolid (B-spline faces) ----------------


def _side_pts(patch, side):
    if side == "u0":
        return tuple(patch.cp[0][j] for j in range(patch.nv))
    if side == "u1":
        return tuple(patch.cp[-1][j] for j in range(patch.nv))
    if side == "v0":
        return tuple(patch.cp[i][0] for i in range(patch.nu))
    return tuple(patch.cp[i][-1] for i in range(patch.nu))


def _knot_lists(U):
    """(multiplicities, distinct knots) STEP lists of a knot vector."""
    mults: list[int] = []
    knots: list = []
    for k in U:
        if knots and knots[-1] == k:
            mults[-1] += 1
        else:
            knots.append(k)
            mults.append(1)
    return mults, knots


def write_step_patch_solid(psolid, *, name: str = "gitcad_part") -> str:
    """Emit a :class:`~forgekernel.bsolid.PatchSolid` as an AP214
    MANIFOLD_SOLID_BREP of B_SPLINE_SURFACE_WITH_KNOTS faces with
    full-domain trims: one EDGE_CURVE per proven seam, each carrying a
    SURFACE_CURVE with a 3D B-spline basis and one degree-1
    parameter-space pcurve per adjacent face, vertices deduplicated by
    exact 3D coordinates.

    Requires the solid's ``seams`` — the side-gluing topology proven by
    exact control-point equality. Without it the writer cannot know
    which borders are one edge, and emitting CLOSED_SHELL over unproven
    topology would be a well-formed falsehood; it refuses by name.
    Polynomial patches only (rational export is a later stage)."""
    patches = list(psolid.patches)
    seams = getattr(psolid, "seams", None)
    if seams is None:
        raise ValueError(
            "write_step_patch_solid: the solid carries no proven seam "
            "topology (seams=None) — cannot emit CLOSED_SHELL over "
            "unproven shared edges; derive seams by exact control-point "
            "equality first")
    for i, p in enumerate(patches):
        if any(w != F(1) for row in p.w for w in row):
            raise ValueError(
                f"write_step_patch_solid: patch {i} is rational — "
                f"polynomial patches only")
    side_map: dict = {}
    for idx, (ka, kb, _flip) in enumerate(seams):
        side_map[ka] = idx
        side_map[kb] = idx
    for fi in range(len(patches)):
        for s in ("u0", "u1", "v0", "v1"):
            if (fi, s) not in side_map:
                raise ValueError(
                    f"write_step_patch_solid: side {s} of patch {fi} is "
                    f"not on any proven seam — the shell's topology is "
                    f"incomplete")

    lines: list[str] = []
    nid = [0]

    def emit(body: str) -> int:
        nid[0] += 1
        lines.append(f"#{nid[0]} = {body};")
        return nid[0]

    pds, ctx = _emit_product_chain(emit, name)
    ctx2 = emit("( GEOMETRIC_REPRESENTATION_CONTEXT(2) "
                "PARAMETRIC_REPRESENTATION_CONTEXT() "
                "REPRESENTATION_CONTEXT('2d parameter space','') )")

    def cp3(p) -> int:
        return emit(f"CARTESIAN_POINT('',({_dec(p[0])},{_dec(p[1])},"
                    f"{_dec(p[2])}))")

    def cp2(u, v) -> int:
        return emit(f"CARTESIAN_POINT('',({_dec(u)},{_dec(v)}))")

    surf_ids = []
    for p in patches:
        rows = ",".join(
            "(" + ",".join(f"#{cp3(p.cp[i][j])}" for j in range(p.nv)) + ")"
            for i in range(p.nu))
        mu, ku = _knot_lists(p.U)
        mv, kv = _knot_lists(p.V)
        surf_ids.append(emit(
            f"B_SPLINE_SURFACE_WITH_KNOTS('',{p.p},{p.q},({rows}),"
            f".UNSPECIFIED.,.F.,.F.,.F.,"
            f"({','.join(str(m) for m in mu)}),"
            f"({','.join(str(m) for m in mv)}),"
            f"({','.join(_dec(k) for k in ku)}),"
            f"({','.join(_dec(k) for k in kv)}),.UNSPECIFIED.)"))

    vids: dict = {}

    def vertex(pt) -> int:
        key = (pt[0], pt[1], pt[2])
        if key not in vids:
            vids[key] = emit(f"VERTEX_POINT('',#{cp3(pt)})")
        return vids[key]

    def side_dom(patch, side):
        if side in ("u0", "u1"):
            return patch.V[patch.q], patch.V[patch.nv]
        return patch.U[patch.p], patch.U[patch.nu]

    def side_uv(patch, side):
        """(start, end) uv corners of a side, in its own t-increasing
        order."""
        a0, a1 = patch.U[patch.p], patch.U[patch.nu]
        b0, b1 = patch.V[patch.q], patch.V[patch.nv]
        return {"u0": ((a0, b0), (a0, b1)), "u1": ((a1, b0), (a1, b1)),
                "v0": ((a0, b0), (a1, b0)), "v1": ((a0, b1), (a1, b1))}[side]

    edge_ids = []
    for (ka, kb, flip) in seams:
        (fi, si), (fj, sj) = ka, kb
        cps = _side_pts(patches[fi], si)
        t0, t1 = side_dom(patches[fi], si)
        deg = len(cps) - 1
        refs = ",".join(f"#{cp3(c)}" for c in cps)
        m = deg + 1
        c3 = emit(f"B_SPLINE_CURVE_WITH_KNOTS('',{deg},({refs}),"
                  f".UNSPECIFIED.,.F.,.F.,({m},{m}),"
                  f"({_dec(t0)},{_dec(t1)}),.UNSPECIFIED.)")
        pcs = []
        for (face_idx, side, rev) in ((fi, si, False), (fj, sj, flip)):
            a, b = side_uv(patches[face_idx], side)
            if rev:
                a, b = b, a
            c2 = emit(f"B_SPLINE_CURVE_WITH_KNOTS('',1,"
                      f"(#{cp2(*a)},#{cp2(*b)}),.UNSPECIFIED.,.F.,.F.,"
                      f"(2,2),({_dec(t0)},{_dec(t1)}),.UNSPECIFIED.)")
            dr = emit(f"DEFINITIONAL_REPRESENTATION('',(#{c2}),#{ctx2})")
            pcs.append(emit(f"PCURVE('',#{surf_ids[face_idx]},#{dr})"))
        sc = emit(f"SURFACE_CURVE('',#{c3},"
                  f"({','.join('#' + str(pc) for pc in pcs)}),.PCURVE_S1.)")
        edge_ids.append(emit(f"EDGE_CURVE('',#{vertex(cps[0])},"
                             f"#{vertex(cps[-1])},#{sc},.T.)"))

    face_ids = []
    for fi, p in enumerate(patches):
        oes = []
        for (s, fwd) in (("v0", True), ("u1", True),
                         ("v1", False), ("u0", False)):
            idx = side_map[(fi, s)]
            ka, _kb, flip = seams[idx]
            canon_here = (ka == (fi, s)) or (not flip)
            flag = fwd if canon_here else (not fwd)
            oes.append(emit(f"ORIENTED_EDGE('',*,*,#{edge_ids[idx]},"
                            f"{'.T.' if flag else '.F.'})"))
        loop = emit(f"EDGE_LOOP('',({','.join('#' + str(o) for o in oes)}))")
        bound = emit(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        face_ids.append(emit(f"ADVANCED_FACE('',(#{bound}),"
                             f"#{surf_ids[fi]},.T.)"))

    shell = emit(f"CLOSED_SHELL('',"
                 f"({','.join('#' + str(x) for x in face_ids)}))")
    brep = emit(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
    absr = emit(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{brep}),"
                f"#{ctx})")
    emit(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{absr})")
    return _step_header(name, "\n".join(lines))
