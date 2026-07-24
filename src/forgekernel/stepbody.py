"""AP214 export written ONCE against the canonical B-rep (ADR-0021).

``stepio.write_step_planar_solid`` is written against one representation: a
planar ``Solid`` plus a list of z-axis ``Cyl`` bores. Everything else — a boss,
a bored boss, a bare cylinder — had no planar base to hang topology on and was
refused. That is the O(ops x reps) shape the canonical B-rep exists to remove:
a ``Body`` already knows its faces are planes and cylinders with exact
parameters, so the writer needs to know nothing about how the solid was built.

Surfaces are emitted ANALYTICALLY. A bore is a CYLINDRICAL_SURFACE that any CAD
system reads back as a hole, never a fan of facets.

Exactness (ADR-0019): STEP is a decimal interchange format, so coordinates are
rounded on the way out — that is the honest boundary of an exact export and it
decides nothing. What must NOT be approximated is topology, so the seam points
that split a periodic face are taken from the circle's own exact frame
(``c +- r*ref``), never from a float sample: both the cap and the wall then
derive bit-identical vertices and the shared edge really is shared.
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.body import (Body, Circle, Cylinder, Line, Plane, SphereS,
                              dot, sub)
from forgekernel.stepio import _dec, _perp, _real, _unit3


def _f(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def _key(v):
    """Hashable exact vertex key. Coordinates may be Fraction or SurdVal, so
    hash the value itself — never a rounded float, which would fuse two
    genuinely distinct vertices a micron apart."""
    return tuple(v)


def _seam_points(c: Circle):
    """The two exact points where a full circle is cut into half-arcs.

    A closed periodic face has no valid single EDGE_LOOP, so a cylinder wall
    ships as two half-faces. The cut must land on the SAME two points from
    every face that touches the circle, or the shared edge silently becomes
    two edges and the shell is no longer closed — hence the circle's own ref
    direction rather than any computed frame.
    """
    if dot(c.ref, c.ref) != 1:
        raise ValueError(
            "circle reference direction is not exactly unit length — the seam "
            "points would leave the exact field (K3.7)")
    p = tuple(c.c[i] + c.r * c.ref[i] for i in range(3))
    q = tuple(c.c[i] - c.r * c.ref[i] for i in range(3))
    return p, q


def _split_full_circles(face):
    """Rewrite every whole-circle edge as two half-arcs, so every edge has two
    distinct ends. Yields (curve, v0, v1, half) where ``half`` names WHICH arc
    this is — 0 is the ref-to-anti-ref side, 1 the other, None for an edge that
    was already partial.

    The tag exists because the two faces meeting at an arc traverse it in
    OPPOSITE directions, so neither the endpoint order nor a point measured
    from the start identifies the arc the same way from both sides. Without a
    direction-free name the cap and the bore wall each mint their own
    EDGE_CURVE for the one shared rim and the shell is no longer closed.
    """
    out = []
    for lp in face.loops:
        seq = []
        for e in lp.edges:
            if isinstance(e.curve, Circle) and _key(e.v0) == _key(e.v1):
                p, q = _seam_points(e.curve)
                seq.append((e.curve, p, q, 0))
                seq.append((e.curve, q, p, 1))
            else:
                seq.append((e.curve, e.v0, e.v1, None))
        out.append(seq)
    return out


def _reverse(seq):
    return [(c, b, a, h) for c, a, b, h in reversed(seq)]


def _oriented_bounds(n, loops):
    """Wind the outer bound WITH the surface normal and every hole AGAINST it.

    A ``Body`` does not promise this: the exact volume forces each inner
    loop's sign (``-abs(area)``), so a hole whose loop happens to wind the
    same way as its outer boundary costs nothing there. STEP has no such
    escape — a FACE_BOUND that runs the wrong way makes the shared rim edge
    traversed identically by both its faces, and the shell reads as open.
    A through-hole trips exactly one of its two caps, so a plate exported
    with two of its edges non-manifold and the other sixteen fine.
    """
    out = []
    for i, lp in enumerate(loops):
        circ = (lp[0][0] if len(lp) == 2 and isinstance(lp[0][0], Circle)
                and lp[0][0] == lp[1][0] else None)
        if circ is not None:
            area = dot(circ.n, n)
        else:
            acc = (Fraction(0),) * 3
            for _curve, a, b, _h in lp:          # the loop chains, so b is next
                cr = _cross(a, b)
                acc = tuple(acc[k] + cr[k] for k in range(3))
            area = dot(acc, n)
        want = 1 if i == 0 else -1
        out.append(lp if (area > 0) == (want > 0) else _reverse(lp))
    return out


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def write_step_body(body: Body, *, name: str = "gitcad_part") -> str:
    """Emit a canonical ``Body`` as a valid AP214 file with a full product
    structure and a MANIFOLD_SOLID_BREP whose edges are shared."""
    lines: list[str] = []
    nid = [0]

    def emit(text: str) -> int:
        nid[0] += 1
        lines.append(f"#{nid[0]} = {text};")
        return nid[0]

    fnum = _real

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
    lu = emit("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    au = emit("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    su = emit("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = emit(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{lu},"
               f"'distance_accuracy_value','')")
    ctx = emit(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
               f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
               f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{lu},#{au},#{su}))"
               f"REPRESENTATION_CONTEXT('',''))")

    def point(v) -> int:
        return emit(f"CARTESIAN_POINT('',({_dec(_f(v[0]))},{_dec(_f(v[1]))},"
                    f"{_dec(_f(v[2]))}))")

    def direction(d) -> int:
        (x, y, z), _ = _unit3(d)
        return emit(f"DIRECTION('',({fnum(x)},{fnum(y)},{fnum(z)}))")

    def placement(origin, axis, ref) -> int:
        return emit(f"AXIS2_PLACEMENT_3D('',#{point(origin)},"
                    f"#{direction(axis)},#{direction(ref)})")

    vids: dict = {}

    def vertex(v) -> int:
        k = _key(v)
        if k not in vids:
            vids[k] = emit(f"VERTEX_POINT('',#{point(v)})")
        return vids[k]

    curves: dict = {}

    def circle_geom(c: Circle) -> int:
        k = (_key(c.c), _key(c.n), _key(c.ref), c.r)
        if k not in curves:
            curves[k] = emit(
                f"CIRCLE('',#{placement(c.c, c.n, c.ref)},{fnum(c.r)})")
        return curves[k]

    edges: dict = {}

    def edge_curve(curve, a, b, half):
        """One EDGE_CURVE per geometric edge, reused by both adjacent faces.

        The key is DIRECTION-FREE — the unordered endpoints plus, for a split
        circle, which half. Both faces at an edge therefore land on the same
        entity, and the one that runs against the stored direction gets a .F.
        ORIENTED_EDGE.
        """
        ends = tuple(sorted((_key(a), _key(b)), key=repr))
        if isinstance(curve, Circle):
            k = ("arc", _key(curve.c), curve.r, _key(curve.n), half, ends)
        else:
            k = ("line", ends)
        if k not in edges:
            va, vb = vertex(a), vertex(b)
            if isinstance(curve, Circle):
                geom = circle_geom(curve)
            else:
                (dx, dy, dz), ln = _unit3(sub(b, a))
                dr = emit(f"DIRECTION('',({fnum(dx)},{fnum(dy)},{fnum(dz)}))")
                vec = emit(f"VECTOR('',#{dr},{fnum(ln)})")
                geom = emit(f"LINE('',#{point(a)},#{vec})")
            edges[k] = (emit(f"EDGE_CURVE('',#{va},#{vb},#{geom},.T.)"),
                        _key(a), _key(b))
        return edges[k]

    def edge_loop(seq) -> int:
        oes = []
        for curve, a, b, half in seq:
            eid, ka, _ = edge_curve(curve, a, b, half)
            sense = ".T." if ka == _key(a) else ".F."
            oes.append(emit(f"ORIENTED_EDGE('',*,*,#{eid},{sense})"))
        return emit(f"EDGE_LOOP('',({','.join('#%d' % o for o in oes)}))")

    def surface_of(s) -> int:
        if isinstance(s, Plane):
            org = tuple(s.n[i] * s.d / dot(s.n, s.n) for i in range(3))
            (nx, ny, nz), _ = _unit3(s.n)
            return emit(f"PLANE('',#{placement(org, s.n, _perp((nx, ny, nz)))})")
        if isinstance(s, Cylinder):
            (dx, dy, dz), _ = _unit3(s.d)
            return emit(f"CYLINDRICAL_SURFACE('',"
                        f"#{placement(s.p, s.d, _perp((dx, dy, dz)))},"
                        f"{fnum(s.r)})")
        if isinstance(s, SphereS):
            return emit(f"SPHERICAL_SURFACE('',"
                        f"#{placement(s.c, (0, 0, 1), (1, 0, 0))},{fnum(s.r)})")
        raise ValueError(
            f"STEP export of a {type(s).__name__} face is not implemented yet "
            "— spherical and conical surfaces arrive with K3.7")

    def advanced_face(surf_id, bounds, sense) -> int:
        parts = []
        for i, lp in enumerate(bounds):
            kind = "FACE_OUTER_BOUND" if i == 0 else "FACE_BOUND"
            parts.append(emit(f"{kind}('',#{edge_loop(lp)},.T.)"))
        return emit(f"ADVANCED_FACE('',({','.join('#%d' % p for p in parts)}),"
                    f"#{surf_id},{'.T.' if sense else '.F.'})")

    faces: list[int] = []
    for face in body.faces:
        s = face.surface
        if isinstance(s, SphereS):
            # A whole sphere carries no loops and its parametrisation is
            # periodic in longitude AND degenerate at the poles. Split it into
            # two lunes along the meridian great circle through the poles: the
            # seam then runs pole to pole, where the surface is degenerate
            # anyway, and each face is a clean 180° longitude range. Putting
            # the split anywhere else (an equator, say) leaves a pole stranded
            # in a face's interior, which readers handle far less reliably.
            c, r = s.c, s.r
            north = (c[0], c[1], c[2] + r)
            south = (c[0], c[1], c[2] - r)
            meridian = Circle(c, (_f(0), _f(-1), _f(0)),
                              (_f(0), _f(0), _f(1)), r)
            surf = surface_of(s)
            lune = [(meridian, north, south, 0), (meridian, south, north, 1)]
            faces.append(advanced_face(surf, [lune], face.sense))
            faces.append(advanced_face(surf, [_reverse(lune)], face.sense))
            continue
        loops = _split_full_circles(face)
        if isinstance(s, Cylinder):
            # a tube is periodic: split it into two half-faces bounded by the
            # rim half-arcs plus the two seam lines, so no bound wraps the
            # surface's period
            # both rims live in ONE loop, so regroup by circle rather than by
            # loop, then order them along the axis
            byrim: dict = {}
            for item in [x for lp in loops for x in lp]:
                if not isinstance(item[0], Circle):
                    raise ValueError(
                        "cylindrical face with a straight edge — a trimmed "
                        "cylinder needs general seam handling (K3.7)")
                byrim.setdefault((_key(item[0].c), item[0].r), []).append(item)
            rims = [v for _, v in sorted(
                byrim.items(),
                key=lambda kv: float(dot(sub(kv[1][0][0].c, s.p), s.d)))]
            if len(rims) != 2 or any(len(r) != 2 for r in rims):
                raise ValueError(
                    "cylindrical face without two whole circular rims — a "
                    "trimmed cylinder needs general seam handling (K3.7)")
            rims = [sorted(r, key=lambda x: x[3]) for r in rims]
            surf = surface_of(s)
            for bottom, top in zip(rims[0], rims[1]):
                bc, b0, b1, bh = bottom
                tc, t0, t1, th = top
                faces.append(advanced_face(surf, [[
                    (bc, b0, b1, bh),
                    (Line(b1, sub(t1, b1)), b1, t1, None),
                    (tc, t1, t0, th),
                    (Line(t0, sub(b0, t0)), t0, b0, None),
                ]], face.sense))
            continue
        faces.append(advanced_face(surface_of(s),
                                   _oriented_bounds(s.n, loops), face.sense))

    shell = emit(f"CLOSED_SHELL('',({','.join('#%d' % f for f in faces)}))")
    brep = emit(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
    absr = emit(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{brep}),#{ctx})")
    emit(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{absr})")

    head = ("ISO-10303-21;\nHEADER;\n"
            "FILE_DESCRIPTION((''),'2;1');\n"
            f"FILE_NAME('{name}','',(''),(''),'forge','',''); \n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n")
    return head + "\n".join(lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"
