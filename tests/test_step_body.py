"""AP214 export written once against the canonical B-rep (ADR-0021).

The acceptance test is a MANIFOLD ORACLE on the emitted file itself, not a
smoke test that it parses: in a closed shell every EDGE_CURVE is used by
exactly two faces, traversed in opposite directions once the ADVANCED_FACE
same_sense flag is folded in. An exporter that gets this wrong still produces
a file that opens — and a solid that will not boolean, will not mesh for CAM,
and imports as a surface soup.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from forgekernel import body as B
from forgekernel.brep import Solid
from forgekernel.kernel import boolean, prism, translate
from forgekernel.quadric import Cyl, DisjointUnion, DrilledSolid, Sphere
from forgekernel.stepbody import write_step_body

L_PRISM = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


def _entities(text):
    """id -> entity name. A COMPLEX entity opens with '(' (the unit and
    geometric-context instances do), so it needs its own pattern or its id
    reads as undefined and every reference to it looks dangling."""
    out = dict(re.findall(r"#(\d+) = (\w+)", text))
    out.update({i: "COMPLEX" for i in re.findall(r"#(\d+) = \(", text)})
    return out


def _edge_uses(text):
    """Effective traversal direction of every EDGE_CURVE, per using face."""
    oriented = {i: (r, s) for i, r, s in re.findall(
        r"#(\d+) = ORIENTED_EDGE\(.*?#(\d+),(\.[TF]\.)\)", text)}
    loops = {i: re.findall(r"#(\d+)", a) for i, a in re.findall(
        r"#(\d+) = EDGE_LOOP\(.*?\((.*?)\)\)", text)}
    bounds = {i: m for i, m in re.findall(
        r"#(\d+) = FACE_(?:OUTER_)?BOUND\(.*?#(\d+),", text)}
    uses = defaultdict(list)
    for bl, flag in re.findall(
            r"#\d+ = ADVANCED_FACE\(.*?\((.*?)\),#\d+,(\.[TF]\.)\)", text):
        for b in re.findall(r"#(\d+)", bl):
            for o in loops[bounds[b]]:
                ref, sense = oriented[o]
                uses[ref].append((sense == ".T.") == (flag == ".T."))
    return uses


SHAPES = [
    ("box", lambda: Solid.box(20, 20, 10), 6, 12),
    ("L-prism", lambda: prism(L_PRISM, 5), 8, 18),
    ("notched plate",
     lambda: boolean("cut", Solid.box(40, 20, 10),
                     translate(Solid.box(10, 20, 10), 15, 0, 6)), 10, 28),
    # a through hole: two caps, a split cylindrical wall, shared rim arcs
    ("drilled plate",
     lambda: DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)]), 8, 18),
    ("blind hole",
     lambda: DrilledSolid(Solid.box(30, 30, 10), [Cyl(15, 15, 3, 4, 10)]), 9, 18),
    ("counterbore",
     lambda: DrilledSolid(Solid.box(30, 30, 10), [])
     .cut(Cyl(15, 15, 2, 0, 10)).cut(Cyl(15, 15, 4, 7, 10)), 11, 24),
    ("four mounting holes",
     lambda: DrilledSolid(Solid.box(100, 60, 10), [])
     .cut(Cyl(10, 10, 3, 5, 10)).cut(Cyl(90, 10, 3, 5, 10))
     .cut(Cyl(10, 50, 3, 5, 10)).cut(Cyl(90, 50, 3, 5, 10)), 18, 36),
    # neither of these has a planar base to hang topology on: both used to
    # refuse outright, which is the O(ops x reps) gap ADR-0021 removes
    ("bare cylinder", lambda: Cyl(0, 0, 5, 0, 12), 4, 6),
    ("boss", lambda: DisjointUnion([Solid.box(30, 30, 3), Cyl(15, 15, 4, 3, 9)]),
     10, 18),
]


@pytest.mark.parametrize("label,build,nfaces,nedges", SHAPES,
                         ids=[s[0] for s in SHAPES])
def test_the_exported_shell_is_closed(label, build, nfaces, nedges) -> None:
    text = write_step_body(B.to_body(build()))
    ents = _entities(text)
    uses = _edge_uses(text)
    curves = [i for i, t in ents.items() if t == "EDGE_CURVE"]
    assert len(curves) == nedges
    assert sum(1 for t in ents.values() if t == "ADVANCED_FACE") == nfaces
    open_edges = [e for e in curves if sorted(uses.get(e, [])) != [False, True]]
    assert open_edges == [], (
        f"{len(open_edges)} edge(s) not traversed once each way — the shell is "
        "open, which no manifold check downstream will forgive")


@pytest.mark.parametrize("label,build,nfaces,nedges", SHAPES,
                         ids=[s[0] for s in SHAPES])
def test_the_file_has_the_product_structure_a_reader_needs(
        label, build, nfaces, nedges) -> None:
    text = write_step_body(B.to_body(build()), name="part_x")
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    kinds = set(_entities(text).values())
    for required in ("APPLICATION_CONTEXT", "PRODUCT", "CLOSED_SHELL",
                     "MANIFOLD_SOLID_BREP",
                     "ADVANCED_BREP_SHAPE_REPRESENTATION",
                     "SHAPE_DEFINITION_REPRESENTATION"):
        assert required in kinds, f"missing {required}"
    assert "PRODUCT('part_x'" in text
    # every referenced entity must exist — a dangling #id is a corrupt file
    defined = {int(i) for i in _entities(text)}
    body = text.split("DATA;", 1)[1]
    assert not [r for r in re.findall(r"#(\d+)", body)
                if int(r) not in defined]


def test_a_hole_is_an_exact_cylinder_not_a_fan_of_facets() -> None:
    text = write_step_body(B.to_body(
        DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])))
    ents = _entities(text)
    assert sum(1 for t in ents.values() if t == "CYLINDRICAL_SURFACE") == 1
    assert sum(1 for t in ents.values() if t == "CIRCLE") == 2
    assert "CYLINDRICAL_SURFACE('',#" in text and ",4.)" in text


def test_the_seam_points_come_from_the_circles_exact_frame() -> None:
    """Both the cap and the bore wall must derive the SAME two split points,
    or the shared rim silently becomes two edges. Taking them from a float
    sample would be the easy way to break this."""
    text = write_step_body(B.to_body(
        DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])))
    # bore at (20,10) r=4 -> seam points at x = 24 and x = 16, on both caps
    for want in ("(24.,10.,0.)", "(16.,10.,0.)", "(24.,10.,5.)", "(16.,10.,5.)"):
        assert f"CARTESIAN_POINT('',{want})" in text


NUMERIC = ("DIRECTION", "VECTOR", "CIRCLE", "CYLINDRICAL_SURFACE",
           "CARTESIAN_POINT")


@pytest.mark.parametrize("writer", ["body", "planar"])
def test_every_real_literal_carries_a_decimal_point(writer) -> None:
    """Part 21 types ``0`` as an INTEGER, so ``DIRECTION('',(0,0,1))`` is a
    type error even though the numbers are right — and almost every direction
    in a mechanical part is whole, so this is the common case. Both writers
    shipped it."""
    plate = DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])
    if writer == "body":
        text = write_step_body(B.to_body(plate))
    else:
        from forgekernel.stepio import write_step_planar_solid
        text = write_step_planar_solid(plate.base, bores=plate.bores)
    offenders = [
        ln.strip() for ln in text.splitlines()
        if any(f"= {k}(" in ln for k in NUMERIC)
        and re.search(r"(?<![\w.#])-?\d+(?![\w.])", ln.split("=", 1)[1])
    ]
    assert offenders == [], f"integer literals where STEP wants REAL: {offenders[:3]}"


def test_a_sphere_ships_as_two_lunes_seamed_at_the_poles() -> None:
    """A whole sphere has no loops at all, and its parametrisation is periodic
    in longitude AND degenerate at the poles. Splitting along a meridian puts
    the seam where the surface is already degenerate; splitting at the equator
    would strand a pole inside a face."""
    text = write_step_body(B.to_body(Sphere(1, 2, 3, 6)))
    ents = _entities(text)
    assert sum(1 for t in ents.values() if t == "SPHERICAL_SURFACE") == 1
    assert sum(1 for t in ents.values() if t == "ADVANCED_FACE") == 2
    # exactly two vertices: the poles, at z = 3 +- 6
    assert sum(1 for t in ents.values() if t == "VERTEX_POINT") == 2
    assert "CARTESIAN_POINT('',(1.,2.,9.))" in text
    assert "CARTESIAN_POINT('',(1.,2.,-3.))" in text
    curves = [i for i, t in ents.items() if t == "EDGE_CURVE"]
    uses = _edge_uses(text)
    assert [e for e in curves if sorted(uses.get(e, [])) != [False, True]] == []


def test_a_surface_with_no_writer_refuses_with_its_stage() -> None:
    from fractions import Fraction as Q

    cone = B.Face(B.Cone((Q(0), Q(0), Q(0)), (Q(0), Q(0), Q(1)), Q(1)), (), True)
    with pytest.raises(ValueError, match="K3.7"):
        write_step_body(B.Body((cone,)))


TRIMMED = [
    ("rounded box", lambda: __import__(
        "forgekernel.quadric", fromlist=["RoundedBox"]).RoundedBox(20, 20, 20, 3),
     26, 48),
    ("rounded slab", lambda: __import__(
        "forgekernel.quadric", fromlist=["RoundedBox"]).RoundedBox(30, 20, 10, 2),
     26, 48),
    ("half-integer radius", lambda: __import__(
        "forgekernel.quadric", fromlist=["RoundedBox"]).RoundedBox(21, 13, 9, 2.5),
     26, 48),
]


@pytest.mark.parametrize("label,build,nfaces,nedges", TRIMMED,
                         ids=[t[0] for t in TRIMMED])
def test_a_trimmed_quadric_body_exports_a_closed_shell(label, build, nfaces,
                                                       nedges) -> None:
    """A rounded box's quarter bands and corner octants already carry proper
    single loops, so forcing the periodic two-half split on them is both
    unnecessary and wrong — there is no period left to straddle.

    Two things had to be settled to close it. An arc is identified by its
    GEOMETRIC circle: a band's rim carries the unit axis while the octant
    beside it carries a cross product of length r^2 pointing the other way,
    and using the raw vector minted a second EDGE_CURVE for every shared arc
    (48 open edges). And a bound's orientation is defined in the SURFACE's
    parameter space — a curved loop is not planar, so no 3D area test settles
    it, and half the bands wound the other way (12 more)."""
    text = write_step_body(B.to_body(build()))
    ents = _entities(text)
    uses = _edge_uses(text)
    curves = [i for i, t in ents.items() if t == "EDGE_CURVE"]
    assert sum(1 for t in ents.values() if t == "ADVANCED_FACE") == nfaces
    assert len(curves) == nedges
    assert sum(1 for t in ents.values() if t == "CYLINDRICAL_SURFACE") == 12
    assert sum(1 for t in ents.values() if t == "SPHERICAL_SURFACE") == 8
    assert [e for e in curves if sorted(uses.get(e, [])) != [False, True]] == []


def test_a_degenerate_shape_refuses_rather_than_dangling() -> None:
    """When the fillet radius reaches half the smallest dimension the core box
    collapses: the flats become points and the bands zero length. The volume
    and mesh are still exact (the solid IS a sphere) but AP214 cannot express
    that topology, and emitting it left 18 dangling edges."""
    from forgekernel.quadric import RoundedBox

    with pytest.raises(ValueError, match="degenerate"):
        write_step_body(B.to_body(RoundedBox(12, 12, 12, 6)))


def _sector(deg: int):
    """A trimmed cylinder face sweeping `deg` degrees, built by hand.

    Nothing in the corpus reaches a face wider than a quarter turn yet, so this
    is the only way to exercise the >180-degree branch.
    """
    from fractions import Fraction as Q

    from forgekernel.body import Circle, Cylinder, Line

    up, dn, rf = (Q(0), Q(0), Q(1)), (Q(0), Q(0), Q(-1)), (Q(1), Q(0), Q(0))
    cyl = Cylinder((Q(0), Q(0), Q(0)), up, Q(1))
    c0 = Circle((Q(0), Q(0), Q(0)), up, rf, Q(1))         # bottom rim, CCW +z
    c1 = Circle((Q(0), Q(0), Q(2)), dn, rf, Q(1))         # top rim, CCW -z
    pts = {0: (Q(1), Q(0)), 90: (Q(0), Q(1)),
           180: (Q(-1), Q(0)), 270: (Q(0), Q(-1))}
    a2 = pts[deg % 360]                                   # b is `deg` CCW of a
    a, b = (Q(1), Q(0), Q(0)), (a2[0], a2[1], Q(0))
    at, bt = (a[0], a[1], Q(2)), (b[0], b[1], Q(2))
    ln = lambda p, q: (Line(p, tuple(q[i] - p[i] for i in range(3))), p, q, None)
    return cyl, [(c0, a, b, None), ln(b, bt), (c1, bt, at, None), ln(at, a)]


@pytest.mark.parametrize("deg", [90, 180, 270])
def test_a_sector_wider_than_half_a_turn_still_reads_as_positive(deg) -> None:
    """``_param_area`` unwraps longitudes by the +-pi rule, which is only valid
    when consecutive SAMPLES are under pi apart. Sampling just the loop's
    vertices, a single edge sweeping 270 degrees read as -90: sign INVERTED and
    magnitude wrong, so the writer would reverse a bound that was already right
    and open the shell. Every one of these winds positively."""
    import math

    from forgekernel.stepbody import _param_area

    surf, loop = _sector(deg)
    got = _param_area(surf, loop)
    assert got == pytest.approx(2 * math.radians(deg) * 2)   # 2 x area, h = 2
    assert got > 0


def test_the_writer_refuses_an_open_shell_rather_than_emitting_one() -> None:
    """The manifold oracle now runs INSIDE the writer. A MANIFOLD_SOLID_BREP
    over an open shell is a file that opens and then will not boolean, will not
    mesh for CAM, and imports as a surface soup — the worst outcome available,
    because nothing downstream reports it."""
    from forgekernel.brep import Solid

    body = B.to_body(Solid.box(10, 10, 10))
    maimed = B.Body(body.faces[:-1])                # drop one face
    with pytest.raises(ValueError, match="shell is not closed"):
        write_step_body(maimed)


def test_the_arc_key_is_exact_not_a_rounded_float_axis() -> None:
    """Whether two faces SHARE an edge was decided by a unit float triple
    rounded to 12 places — a float deciding topology, which ADR-0019 forbids
    outright. The exact key is scale- and sign-invariant just as that one was
    meant to be, and the rounded box (whose bands and octants carry the same
    axis at wildly different lengths) still shares all 48."""
    from fractions import Fraction as Q

    from forgekernel.body import Circle
    from forgekernel.brep import _canon_dir

    unit = Circle((Q(0),) * 3, (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0)), Q(1))
    scaled = Circle((Q(0),) * 3, (Q(0), Q(0), Q(-49)), (Q(1), Q(0), Q(0)), Q(1))
    assert _canon_dir(unit.n) == _canon_dir(scaled.n)
