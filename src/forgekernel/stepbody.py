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

from forgekernel.body import (Body, Circle, Cone, Cylinder, Line, Plane,
                              SphereS, dot, sub)
from forgekernel.brep import _canon_dir
from forgekernel.stepio import _dec, _perp, _real, _unit3, export_rational


def _f(x) -> Fraction:
    """A coordinate as a rational, ready for ``_dec`` to render.

    Reached at exactly one place — writing a CARTESIAN_POINT — so this is a
    NUMERAL conversion, never a predicate. It delegates to
    ``stepio.export_rational``, which converts values beyond ℚ (a rotated
    part's ℚ[√2] coordinates, a lathe's ℚ[π] ones) by exact comparison only.

    It used to be a bare ``Fraction(x)``, which raises on any exact scalar by
    design (``is_exact_scalar`` exists to stop silent coercion). That
    TypeError surfaced at the seam as "export_step on Solid yet — K3.7",
    blaming the canonical B-rep for a capability it had; what was missing was
    the numeral.
    """
    return x if isinstance(x, Fraction) else export_rational(x)


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


def _arc_samples(curve, a, b):
    """Interior points of an arc, spaced so no step exceeds a quarter turn.

    ``_param_area`` unwraps its longitudes by the ±π rule, which can only be
    right when consecutive SAMPLES are less than π apart. Sampling only the
    loop's vertices, a single edge sweeping 270° read as −90° — sign inverted
    and magnitude wrong, so the writer reversed a bound that was already
    correct and opened the shell. Floats are fine here: this feeds a sign, and
    the arc's own endpoints are still the exact ones.
    """
    import math

    if not isinstance(curve, Circle):
        return []
    cf = [float(x) for x in curve.c]
    n = _unit3(curve.n)[0]
    u = _unit3(sub(a, curve.c))[0]
    w = (n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2],
         n[0] * u[1] - n[1] * u[0])
    v = [float(b[i]) - cf[i] for i in range(3)]
    theta = math.atan2(sum(v[i] * w[i] for i in range(3)),
                       sum(v[i] * u[i] for i in range(3)))
    sweep = theta % (2 * math.pi)                 # CCW about the circle's axis
    if sweep < 1e-12:
        sweep = 2 * math.pi                       # a closed edge is the whole turn
    m = max(1, math.ceil(sweep / (math.pi / 2)))
    r = float(curve.r)
    out = []
    for j in range(1, m):
        t = sweep * j / m
        out.append(tuple(cf[i] + r * (math.cos(t) * u[i] + math.sin(t) * w[i])
                         for i in range(3)))
    return out


def _param_area(surf, loop):
    """Signed area of a loop in the SURFACE's parameter space (float).

    Only the SIGN is used, and it decides how a bound is written, not any
    geometry — so evaluating in floats is the same boundary STEP's decimal
    output already sits on.
    """
    import math

    # Sample each edge's START vertex AND the interior of any arc it carries:
    # the unwrap below is only valid when consecutive samples are under π
    # apart, and one 270° edge is not (see _arc_samples).
    samples = []
    for crv, a, b, _h in loop:
        samples.append(a)
        samples.extend(_arc_samples(crv, a, b))
    pts = []
    if isinstance(surf, SphereS):
        c = tuple(float(x) for x in surf.c)
        for a in samples:
            v = [float(a[i]) - c[i] for i in range(3)]
            pts.append((math.atan2(v[1], v[0]),
                        math.atan2(v[2], math.hypot(v[0], v[1]))))
    else:
        (dx, dy, dz), _ = _unit3(surf.d)
        d = (dx, dy, dz)
        u = _perp(d)
        w = (d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2],
             d[0] * u[1] - d[1] * u[0])
        o = tuple(float(x) for x in surf.p)
        for a in samples:
            v = [float(a[i]) - o[i] for i in range(3)]
            pts.append((math.atan2(sum(v[i] * w[i] for i in range(3)),
                                   sum(v[i] * u[i] for i in range(3))),
                        sum(v[i] * d[i] for i in range(3))))
    # unwrap so a sector that straddles the seam still reads as one sweep
    for i in range(1, len(pts)):
        while pts[i][0] - pts[i - 1][0] > math.pi:
            pts[i] = (pts[i][0] - 2 * math.pi, pts[i][1])
        while pts[i][0] - pts[i - 1][0] < -math.pi:
            pts[i] = (pts[i][0] + 2 * math.pi, pts[i][1])
    n = len(pts)
    return sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
               for i in range(n))


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
        if isinstance(curve, Circle) and half is not None:
            # CANONICALISE the stored direction by the half tag.
            #
            # A STEP file has no `half` field. A reader sees only (v0, v1,
            # CIRCLE), and for a HALF circle both halves share those two
            # endpoints — so the stored ORDER, read as increasing parameter
            # from the circle's own ref, is the only thing that names which arc
            # is meant. The writer's tag has to be pushed into that order here
            # or it is lost on the way out.
            #
            # It was being lost. `_reverse` swaps endpoints and keeps the tag,
            # so after a bound was rewound the tag no longer matched the
            # geometry, and each bore-wall face ended up pairing the BOTTOM
            # rim's [0,π] half with the TOP rim's [π,2π] half: a face that
            # wraps instead of a clean half-cylinder, spanning 360° of the
            # surface's period and therefore delimiting no region at all. The
            # shell still passed its own edge audit — every edge used twice,
            # once each way — because the pairing was right and only the
            # GEOMETRY was wrong. Every exported bore, lathe and cone carried
            # it.
            p, q = _seam_points(curve)
            a, b = (p, q) if half == 0 else (q, p)
        ends = tuple(sorted((_key(a), _key(b)), key=repr))
        if isinstance(curve, Circle):
            # Identify the arc by its GEOMETRIC circle. Two faces meeting at
            # one arc can carry different vectors for the same normal — a
            # band's rim is the unit axis while an octant's comes from a cross
            # product of length r^2, and it points the other way besides. Using
            # the raw vector minted a second EDGE_CURVE for every shared arc
            # and left a rounded box with 48 open edges.
            #
            # The direction key is now EXACT: ``_canon_dir`` divides by the
            # leading component, so it is invariant under both scale and sign
            # just as the old key was — but it was a unit float triple rounded
            # to 12 places, i.e. a FLOAT deciding whether two faces share an
            # edge, which ADR-0019 forbids outright.
            #
            # ``ends`` alone does not NAME an arc — the 90° and the 270° arc
            # between the same two points of one circle have identical
            # endpoints — and no cheap key fixes that, because the arc a face
            # means is CCW about ITS OWN normal and the two faces at a shared
            # arc disagree about both. The ``half`` tag settles the case the
            # writer itself creates; anything else that collided would show up
            # as an edge used more than twice, which is what the closing audit
            # below refuses on. A wrong file that opens is the bad outcome.
            k = ("arc", _key(curve.c), curve.r, _canon_dir(curve.n),
                 half, ends)
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

    uses: dict = {}

    def edge_loop(seq, sense_flag) -> int:
        oes = []
        for curve, a, b, half in seq:
            eid, ka, _ = edge_curve(curve, a, b, half)
            fwd = ka == _key(a)
            # record the EFFECTIVE traversal: the ORIENTED_EDGE sense composed
            # with the face's same_sense flag, which is what a reader sees
            uses.setdefault(eid, []).append(fwd == sense_flag)
            oes.append(emit(f"ORIENTED_EDGE('',*,*,#{eid},"
                            f"{'.T.' if fwd else '.F.'})"))
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
        if isinstance(s, Cone):
            # AP214 places a cone by a REFERENCE circle, not by its apex: the
            # placement origin sits where the radius equals `radius`, and
            # semi_angle opens from there along the axis. The apex itself is
            # the radius-0 reference, which is always exact and always on the
            # surface, so use it.
            import math as _m
            (dx, dy, dz), _ = _unit3(s.d)
            return emit(
                f"CONICAL_SURFACE('',"
                f"#{placement(s.p, s.d, _perp((dx, dy, dz)))},"
                f"{fnum(0)},{fnum(_m.atan(float(s.tan_half)))})")
        raise ValueError(
            f"STEP export of a {type(s).__name__} face is not implemented yet "
            "— spherical and conical surfaces arrive with K3.7")

    def advanced_face(surf_id, bounds, sense) -> int:
        parts = []
        for i, lp in enumerate(bounds):
            kind = "FACE_OUTER_BOUND" if i == 0 else "FACE_BOUND"
            parts.append(emit(f"{kind}('',#{edge_loop(lp, sense)},.T.)"))
        return emit(f"ADVANCED_FACE('',({','.join('#%d' % p for p in parts)}),"
                    f"#{surf_id},{'.T.' if sense else '.F.'})")

    # A DEGENERATE face has no place in a closed shell: a rounded box whose
    # fillet radius reaches half its smallest dimension has flats that are
    # points and bands of zero length. Its volume and mesh are still exact
    # (the solid is a sphere), but the AP214 topology cannot express it, so
    # say that rather than emit 18 dangling edges.
    for face in body.faces:
        for lp in face.loops:
            for e in lp.edges:
                # a whole circle legitimately has v0 == v1; a zero-length LINE
                # does not, and it is exactly what a collapsed flat leaves
                if isinstance(e.curve, Line) and _key(e.v0) == _key(e.v1):
                    raise ValueError(
                        "a face has collapsed to a point or a line — this "
                        "shape's topology is degenerate and AP214 cannot "
                        "express it (K3.7)")

    faces: list[int] = []
    for face in body.faces:
        s = face.surface
        # A face that is ALREADY trimmed carries a proper single loop --
        # a rounded box's quarter bands are arc/line/arc/line and its corners
        # three arcs. Forcing the periodic split on those is both unnecessary
        # and wrong: there is no period left to straddle.
        if (isinstance(s, (Cylinder, SphereS, Cone)) and len(face.loops) == 1
                and any(isinstance(e.curve, Circle)
                        and _key(e.v0) != _key(e.v1)
                        for e in face.loops[0].edges)):
            lp = _split_full_circles(face)[0]
            # STEP defines a bound's orientation in the surface's PARAMETER
            # space, and a curved face's loop is not planar, so no 3D area
            # test settles it. Half the bands of a rounded box wind one way
            # and half the other, which left 12 of its 48 edges traversed
            # identically by both their faces.
            if _param_area(s, lp) < 0:
                lp = _reverse(lp)
            faces.append(advanced_face(surface_of(s), [lp], face.sense))
            continue
        if isinstance(s, SphereS):
            if face.loops or s.pole is not None:
                # A TRIMMED spherical face (a napkin ring's zone, a blind
                # bore's one-rim cap complement). This branch would silently
                # emit the WHOLE sphere as two lunes and drop the rims — a
                # well-formed file describing a different solid, which is
                # worse than no file. Until the writer can trim a spherical
                # surface by its rim loops, refuse by name. (In practice a
                # zone's irrational rim height already refused downstream;
                # a rational-rim zone would NOT have, hence this guard.)
                raise ValueError(
                    "STEP export of a trimmed spherical face (zone or "
                    "pole-trimmed cap complement) is not implemented yet "
                    "(K3.7)")
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
        if isinstance(s, Cone):
            # A taper is periodic in the same way a tube is, so it also ships
            # as two half-faces. A TRUE cone closes at its apex: that half-face
            # is bounded by an arc and two generators meeting at one vertex,
            # not by two rims — the degenerate rim is a point, not a mistake.
            rims = []
            for item in [x for lp in loops for x in lp]:
                if not isinstance(item[0], Circle):
                    raise ValueError(
                        "conical face with a straight edge — a trimmed cone "
                        "needs general seam handling (K3.7)")
                rims.append(item)
            byrim: dict = {}
            for item in rims:
                byrim.setdefault((_key(item[0].c), item[0].r), []).append(item)
            groups = [v for _, v in sorted(
                byrim.items(),
                key=lambda kv: float(dot(sub(kv[1][0][0].c, s.p), s.d)))]
            if not groups or any(len(g) != 2 for g in groups) or len(groups) > 2:
                raise ValueError(
                    "conical face without one or two whole circular rims "
                    "(K3.7)")
            groups = [sorted(g, key=lambda x: x[3]) for g in groups]
            surf = surface_of(s)
            if len(groups) == 1:                      # cone: rim + apex point
                apex = tuple(s.p)
                # The apex is the degenerate OTHER rim, so the real rim plays
                # whichever role the apex does not. Traversing it forward when
                # it is the upper rim makes it run the same way as the cap that
                # shares it, and the two faces stop closing the shell.
                upper = dot(sub(groups[0][0][0].c, s.p), s.d) > 0
                for bc, b0, b1, bh in groups[0]:
                    arc = (bc, b1, b0, bh) if upper else (bc, b0, b1, bh)
                    x, y = (b0, b1) if upper else (b1, b0)
                    faces.append(advanced_face(surf, [[
                        arc,
                        (Line(x, sub(apex, x)), x, apex, None),
                        (Line(apex, sub(y, apex)), apex, y, None),
                    ]], face.sense))
            else:
                for bottom, top in zip(groups[0], groups[1]):
                    bc, b0, b1, bh = bottom
                    tc, t0, t1, th = top
                    faces.append(advanced_face(surf, [[
                        (bc, b0, b1, bh),
                        (Line(b1, sub(t1, b1)), b1, t1, None),
                        (tc, t1, t0, th),
                        (Line(t0, sub(b0, t0)), t0, b0, None),
                    ]], face.sense))
            continue
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

    # THE CLOSING AUDIT. In a closed shell every EDGE_CURVE is used by exactly
    # two faces, traversed once each way after the ADVANCED_FACE same_sense
    # flag is folded in. This is the oracle the tests apply to the emitted
    # file; running it in the writer is what makes the difference between an
    # honest refusal and a MANIFOLD_SOLID_BREP over an open shell — a file that
    # opens, and then will not boolean, will not mesh for CAM, and imports as a
    # surface soup. Every topology defect found in this writer so far (rims
    # minted twice, bounds wound the wrong way, a pre-flipped mouth circle)
    # would have surfaced here at the moment it was introduced.
    open_edges = sorted(e for e, us in uses.items() if sorted(us) != [False, True])
    if open_edges:
        raise ValueError(
            f"the shell is not closed: {len(open_edges)} edge(s) of "
            f"{len(uses)} are not traversed once each way (first #"
            f"{open_edges[0]}) — refusing to write a MANIFOLD_SOLID_BREP over "
            "an open shell (K3.7)")

    shell = emit(f"CLOSED_SHELL('',({','.join('#%d' % f for f in faces)}))")
    brep = emit(f"MANIFOLD_SOLID_BREP('{name}',#{shell})")
    absr = emit(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{brep}),#{ctx})")
    emit(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{absr})")

    head = ("ISO-10303-21;\nHEADER;\n"
            "FILE_DESCRIPTION((''),'2;1');\n"
            f"FILE_NAME('{name}','',(''),(''),'forge','',''); \n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n")
    return head + "\n".join(lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"
