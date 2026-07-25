"""The canonical B-rep (ADR-0021) — one topology every representation maps to.

The acceptance test for the architecture is the SHADOW RUN the ADR requires: a
representation's own exact metrics must reproduce after converting it to
``Body``. Exact equality in ℚ[π], not a float tolerance — that is the whole
point of the canonical form preserving exactness.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction as F

import pytest

from forgekernel import body as B
from forgekernel.brep import Solid
from forgekernel.kernel import boolean, prism, translate
from forgekernel.quadric import (Cone, Cyl, DisjointUnion, DrilledSolid, PiVal,
                                 RevolveSolid, RoundedBox, Sphere)
from forgekernel.tess import mesh_volume

L_PRISM = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]

SHADOW = [
    ("box", Solid.box(20, 20, 10), PiVal(4000, 0)),
    ("L-prism", prism(L_PRISM, 5), PiVal(320, 0)),
    ("cylinder", Cyl(0, 0, 5, 0, 12), PiVal(0, 300)),
    ("sphere", Sphere(0, 0, 0, 6), PiVal(0, 288)),
    ("drilled through", DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)]),
     PiVal(4000, -80)),
    ("drilled blind", DrilledSolid(Solid.box(30, 30, 10), [Cyl(15, 15, 3, 4, 10)]),
     PiVal(9000, -54)),
    ("two holes", DrilledSolid(Solid.box(60, 30, 5),
                               [Cyl(15, 15, 3, 0, 5), Cyl(45, 15, 3, 0, 5)]),
     PiVal(9000, -90)),
    ("boss", DisjointUnion([Solid.box(30, 30, 3), Cyl(15, 15, 4, 3, 9)]),
     PiVal(2700, 96)),
]


@pytest.mark.parametrize("label,shape,exact", SHADOW, ids=[s[0] for s in SHADOW])
def test_canonical_form_reproduces_the_exact_volume(label, shape, exact) -> None:
    """The ADR-0021 shadow run: converting to the canonical B-rep and
    re-integrating by the divergence theorem must land on the SAME exact
    number the representation computes for itself."""
    assert B.volume(B.to_body(shape)) == exact


@pytest.mark.parametrize("label,shape,exact", SHADOW, ids=[s[0] for s in SHADOW])
def test_canonical_form_meshes_watertight(label, shape, exact) -> None:
    mesh = B.tessellate(B.to_body(shape), 0.05)
    ec = defaultdict(int)
    for a, b, c in mesh["triangles"]:
        for e in ((a, b), (b, c), (c, a)):
            ec[tuple(sorted(e))] += 1
    assert all(n == 2 for n in ec.values()), "mesh is not closed"
    assert mesh_volume(mesh) == pytest.approx(float(exact), rel=0.03)


def test_rigid_transforms_preserve_volume_exactly() -> None:
    body = B.to_body(DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)]))
    exact = B.volume(body)
    for m in (B.Affine.translation(3, -7, 11),
              B.Affine.mirror("x"), B.Affine.mirror("z"),
              B.Affine.rotation((0, 0, 1), 90),
              B.Affine.rotation((0, 0, 1), 45)):
        moved = body.transformed(m)
        # a mirror reverses orientation; face senses flip so the body stays
        # outward-oriented and the volume keeps its sign
        assert B.volume(moved) == exact


def test_uniform_scale_is_exact_and_cubes_the_volume() -> None:
    body = B.to_body(Cyl(0, 0, 5, 0, 12))
    scaled = body.transformed(B.Affine.scaling(2, 2, 2))
    assert B.volume(scaled) == PiVal(0, 300 * 8)


def test_non_uniform_scale_of_a_cylinder_refuses() -> None:
    """An ellipse is not in the canonical surface set — refuse, never
    approximate (ADR-0019)."""
    body = B.to_body(Cyl(0, 0, 5, 0, 12))
    with pytest.raises(ValueError, match="ellipse"):
        body.transformed(B.Affine.scaling(2, 3, 1))


def test_bbox_of_a_curved_body_bounds_the_curve_not_its_vertices() -> None:
    lo, hi = B.bbox(B.to_body(Cyl(0, 0, 5, 0, 12)))
    assert (lo[0], hi[0]) == pytest.approx((-5.0, 5.0))
    assert (lo[1], hi[1]) == pytest.approx((-5.0, 5.0))
    assert (lo[2], hi[2]) == pytest.approx((0.0, 12.0))


def test_unknown_representation_refuses_with_its_stage() -> None:
    with pytest.raises(ValueError, match="no canonical-B-rep converter"):
        B.to_body(object())


def test_entity_descriptors_are_analytic_for_every_representation() -> None:
    """Edge/face selection works on curved solids once they are in canonical
    form: a bore reports an exact radius and axis, not a facet count."""
    plate = DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])
    body = B.to_body(plate)

    edges = B.edges_info(body)
    circles = [e for e in edges if e["curve"] == "circle"]
    assert len(edges) == 14                       # 12 box edges + 2 bore rims
    assert len(circles) == 2
    assert all(c["radius"] == pytest.approx(4.0) for c in circles)
    assert all(c["length"] == pytest.approx(2 * math.pi * 4) for c in circles)
    assert all(abs(c["axis"][2]) == pytest.approx(1.0) for c in circles)

    faces = B.faces_info(body)
    bore = [f for f in faces if f["surface"] == "cylinder"]
    assert len(bore) == 1
    assert bore[0]["radius"] == pytest.approx(4.0)
    assert bore[0]["area"] == pytest.approx(2 * math.pi * 4 * 5)
    caps = [f for f in faces if f["surface"] == "plane"
            and abs(f["plane"][2]) == pytest.approx(1.0)]
    # each cap is the plate minus the hole
    assert all(f["area"] == pytest.approx(40 * 20 - math.pi * 16) for f in caps)


def test_a_sphere_reports_its_analytic_surface() -> None:
    faces = B.faces_info(B.to_body(Sphere(0, 0, 0, 6)))
    assert len(faces) == 1 and faces[0]["surface"] == "sphere"
    assert faces[0]["area"] == pytest.approx(4 * math.pi * 36)


def test_plane_normals_survive_a_non_uniform_scale() -> None:
    """A plane's normal transforms by the INVERSE-TRANSPOSE. Applying the map
    directly leaves a SLANTED face with a normal that is no longer
    perpendicular to it — the volume can still come out right (the errors
    cancel over a closed body) while faces_info and mesh orientation are wrong,
    so the volume alone is not a sufficient check."""
    tri = prism([(0, 0), (10, 0), (0, 6)], 4)          # slanted lateral face
    for factors, want in (((2, 3, 1), 720), ((1, 3, 5), 1800), ((2, 2, 2), 960)):
        body = B.to_body(tri).transformed(B.Affine.scaling(*factors))
        assert float(B.volume(body)) == pytest.approx(want)
        for face in body.faces:
            vs = [tuple(float(x) for x in e.v0) for e in face.loops[0].edges]
            a = [vs[1][i] - vs[0][i] for i in range(3)]
            c = [vs[2][i] - vs[0][i] for i in range(3)]
            true = (a[1] * c[2] - a[2] * c[1], a[2] * c[0] - a[0] * c[2],
                    a[0] * c[1] - a[1] * c[0])
            stored = tuple(float(x) for x in face.surface.n)
            cr = (true[1] * stored[2] - true[2] * stored[1],
                  true[2] * stored[0] - true[0] * stored[2],
                  true[0] * stored[1] - true[1] * stored[0])
            assert math.sqrt(sum(x * x for x in cr)) < 1e-9 * max(
                1.0, math.sqrt(sum(x * x for x in true))), "normal not perpendicular"


FRAGMENTED = [
    # a prism cap is EAR-CLIPPED into triangles, so any profile with >3 points
    # has coplanar cap fragments — matching a bore to a cap by z alone attached
    # its hole to every fragment, including ones the bore never reaches
    ("ear-clipped prism cap",
     lambda: DrilledSolid(prism([(0, 0), (20, 0), (20, 20), (0, 20)], 10), [])
     .cut(Cyl(10, 10, 2, -1, 11))),
    # a pocket splits the top face into two coplanar pieces
    ("slotted plate",
     lambda: DrilledSolid(
         boolean("cut", Solid.box(40, 20, 10),
                 translate(Solid.box(10, 20, 10), 15, 0, 6)), [])
     .cut(Cyl(5, 10, 3, 0, 10))),
    # two blind holes at a COMMON depth: each bore's floor disk sits at the
    # other's clamped z, so each punched a phantom hole through the other
    ("two blind holes, same depth",
     lambda: DrilledSolid(Solid.box(40, 20, 10), [])
     .cut(Cyl(10, 10, 3, 4, 10)).cut(Cyl(30, 10, 3, 4, 10))),
    ("blind holes, unequal radii",
     lambda: DrilledSolid(Solid.box(40, 20, 10), [])
     .cut(Cyl(10, 10, 2, 4, 10)).cut(Cyl(30, 10, 5, 4, 10))),
    ("four mounting holes",
     lambda: DrilledSolid(Solid.box(100, 60, 10), [])
     .cut(Cyl(10, 10, 3, 5, 10)).cut(Cyl(90, 10, 3, 5, 10))
     .cut(Cyl(10, 50, 3, 5, 10)).cut(Cyl(90, 50, 3, 5, 10))),
]


@pytest.mark.parametrize("label,build", FRAGMENTED, ids=[f[0] for f in FRAGMENTED])
def test_a_bore_only_holes_the_cap_fragment_it_reaches(label, build) -> None:
    """A cap is often split into coplanar fragments. Attaching a bore's hole to
    every fragment at that height silently overstates the removed material —
    on a four-hole mounting plate by 50%. The canonical volume must equal the
    representation's own exact volume."""
    d = build()
    assert B.volume(B.to_body(d)) == d.volume()


STACKS = [
    ("counterbore", lambda: DrilledSolid(Solid.box(30, 30, 10), [])
     .cut(Cyl(15, 15, 2, 0, 10)).cut(Cyl(15, 15, 4, 7, 10))),
    ("widening from below", lambda: DrilledSolid(Solid.box(30, 30, 10), [])
     .cut(Cyl(15, 15, 5, 0, 4)).cut(Cyl(15, 15, 2, 0, 10))),
    ("three-step stack", lambda: DrilledSolid(Solid.box(40, 40, 12), [])
     .cut(Cyl(20, 20, 2, 0, 12)).cut(Cyl(20, 20, 4, 6, 12))
     .cut(Cyl(20, 20, 6, 10, 12))),
]


@pytest.mark.parametrize("label,build", STACKS, ids=[s[0] for s in STACKS])
def test_coaxial_bores_are_one_stepped_void(label, build) -> None:
    """A counterbore stack is ONE void, not several. Emitting a hole per bore
    double-subtracted their overlap on the shared cap and left the inner wall
    running through the wider bore's empty space."""
    d = build()
    assert B.volume(B.to_body(d)) == d.volume()


def test_a_whole_sphere_still_bounds_its_bbox() -> None:
    """A SphereS face carries no loops, and bbox walked loops only — so a
    sphere contributed nothing and the bounds came back inf/-inf."""
    lo, hi = B.bbox(B.to_body(Sphere(0, 0, 0, 6)))
    assert lo == pytest.approx((-6.0, -6.0, -6.0))
    assert hi == pytest.approx((6.0, 6.0, 6.0))


def test_a_singular_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="zero scale factor"):
        B.Affine.scaling(0, 1, 1)


_A, _a = 40 * 20, math.pi * 16
_BOSS = 2700 + 96 * math.pi

CENTROIDS = [
    ("box", lambda: Solid.box(20, 20, 10), (10, 10, 5)),
    # the trap: a planar cone's centroid is ¾ of the face's, but a cylindrical
    # band's is NOT — decomposing the wall the same way puts this at z=3
    ("cylinder", lambda: Cyl(0, 0, 5, 0, 12), (0, 0, 6)),
    ("sphere off origin", lambda: Sphere(1, 2, 3, 6), (1, 2, 3)),
    ("L-prism", lambda: prism(L_PRISM, 5), (3.875, 3.875, 2.5)),
    # a hole pulls the centre of mass toward the remaining material
    ("plate, off-centre bore",
     lambda: DrilledSolid(Solid.box(40, 20, 5), [Cyl(30, 10, 4, 0, 5)]),
     ((_A * 20 - _a * 30) / (_A - _a), 10, 2.5)),
    ("boss", lambda: DisjointUnion([Solid.box(30, 30, 3), Cyl(15, 15, 4, 3, 9)]),
     (15, 15, (2700 * 1.5 + 96 * math.pi * 6) / _BOSS)),
]


@pytest.mark.parametrize("label,build,want", CENTROIDS,
                         ids=[c[0] for c in CENTROIDS])
def test_centre_of_mass_matches_the_analytic_value(label, build, want) -> None:
    got = B.centroid(B.to_body(build()))
    assert got == pytest.approx(want, abs=1e-9)


COAXIAL = [
    ("counterbore", lambda: DrilledSolid(Solid.box(30, 30, 10), [])
     .cut(Cyl(15, 15, 2, 0, 10)).cut(Cyl(15, 15, 4, 7, 10))),
    ("widening from below", lambda: DrilledSolid(Solid.box(30, 30, 10), [])
     .cut(Cyl(15, 15, 5, 0, 4)).cut(Cyl(15, 15, 2, 0, 10))),
    ("three-step stack", lambda: DrilledSolid(Solid.box(40, 40, 12), [])
     .cut(Cyl(20, 20, 2, 0, 12)).cut(Cyl(20, 20, 4, 6, 12))
     .cut(Cyl(20, 20, 6, 10, 12))),
    ("independent bores", lambda: DrilledSolid(
        Solid.box(60, 30, 5), [Cyl(15, 15, 3, 0, 5), Cyl(45, 15, 3, 0, 5)])),
]


@pytest.mark.parametrize("label,build", COAXIAL, ids=[c[0] for c in COAXIAL])
def test_the_two_centroid_paths_agree_on_a_coaxial_stack(label, build) -> None:
    """DrilledSolid.centroid_f looped over raw bores while volume() unioned
    them by z-band, so a counterbore's pilot hole came off twice where the two
    overlap. The reported mass-properties dict was internally inconsistent:
    a wrong centre of mass attached to an exact mass."""
    d = build()
    assert d.centroid_f() == pytest.approx(B.centroid(B.to_body(d)), abs=1e-9)


def test_a_centroid_is_not_the_bbox_centre() -> None:
    """The regression this replaces: mass_props reported the bbox centre and
    flagged it with a key no caller read. On an L-bracket that is wrong by a
    fifth of the part, presented as fact."""
    body = B.to_body(prism(L_PRISM, 5))
    lo, hi = B.bbox(body)
    mid = tuple((lo[i] + hi[i]) / 2 for i in range(3))
    got = B.centroid(body)
    assert abs(got[0] - mid[0]) > 1.0 and abs(got[1] - mid[1]) > 1.0


def test_a_centroid_rides_along_with_a_rigid_transform() -> None:
    body = B.to_body(DrilledSolid(Solid.box(40, 20, 5), [Cyl(30, 10, 4, 0, 5)]))
    c = B.centroid(body)
    moved = B.Affine.translation(3, -7, 11)
    assert B.centroid(body.transformed(moved)) == pytest.approx(
        (c[0] + 3, c[1] - 7, c[2] + 11), abs=1e-9)
    assert B.centroid(body.transformed(B.Affine.mirror("x"))) == pytest.approx(
        (-c[0], c[1], c[2]), abs=1e-9)
    assert B.centroid(body.transformed(B.Affine.scaling(2, 2, 2))) == \
        pytest.approx(tuple(2 * x for x in c), abs=1e-9)


def test_a_capped_faces_centroid_accounts_for_its_holes() -> None:
    """faces_info weighted only the outer loop, so a drilled plate's cap
    reported the plate centre no matter where the bore was."""
    body = B.to_body(DrilledSolid(Solid.box(40, 20, 5), [Cyl(30, 10, 4, 0, 5)]))
    cap = [f for f in B.faces_info(body)
           if f["surface"] == "plane" and abs(f["plane"][2]) > 0.9][0]
    assert cap["area"] == pytest.approx(_A - _a)
    assert cap["centroid"][0] == pytest.approx((_A * 20 - _a * 30) / (_A - _a))


def _slot_face():
    """A slot outline in z=0: two straight flanks and two 180° arc ends. The
    loop MIXES arcs and lines, so it is neither a polygon nor a whole circle.
    """
    from fractions import Fraction as Q

    def P(x, y):
        return (Q(x), Q(y), Q(0))

    up, dn = (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0))
    right = B.Circle(P(10, 0), up, dn, Q(3))
    left = B.Circle(P(0, 0), up, dn, Q(3))
    edges = (
        B.Edge(B.Line(P(0, -3), (Q(10), Q(0), Q(0))), P(0, -3), P(10, -3)),
        B.Edge(right, P(10, -3), P(10, 3)),
        B.Edge(B.Line(P(10, 3), (Q(-10), Q(0), Q(0))), P(10, 3), P(0, 3)),
        B.Edge(left, P(0, 3), P(0, -3)),
    )
    return B.Face(B.Plane(up, Q(0)), (B.Loop(edges),), True)


def test_a_loop_mixing_arcs_and_lines_refuses_rather_than_chord_them() -> None:
    """Walking a loop's v0 vertices turns every arc into a chord. For this
    slot that silently drops both semicircular ends — 9π mm² of 60+9π, a 32%
    under-report presented as an exact value."""
    with pytest.raises(ValueError, match="mixing arcs and lines"):
        B.volume(B.Body((_slot_face(),)))


def test_the_display_mesh_still_follows_the_arcs() -> None:
    """Refusing the EXACT area does not mean refusing to draw it — meshing is
    a display property where floats are legal (ADR-0019)."""
    mesh = B.tessellate(B.Body((_slot_face(),)), 0.002)
    area = 0.0
    for a, b, c in mesh["triangles"]:
        va, vb, vc = (mesh["vertices"][i] for i in (a, b, c))
        e1 = [vb[k] - va[k] for k in range(3)]
        e2 = [vc[k] - va[k] for k in range(3)]
        cr = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
              e1[0] * e2[1] - e1[1] * e2[0])
        area += 0.5 * math.sqrt(sum(x * x for x in cr))
    # chording both ends would give exactly the 10x6 rectangle, 60
    assert area > 88.0
    assert area == pytest.approx(60 + 9 * math.pi, rel=1e-3)


def test_bbox_covers_an_arcs_bulge_past_its_endpoints() -> None:
    lo, hi = B.bbox(B.Body((_slot_face(),)))
    assert (lo[0], hi[0]) == pytest.approx((-3.0, 13.0))
    assert (lo[1], hi[1]) == pytest.approx((-3.0, 3.0))


def test_a_lone_arc_is_not_mistaken_for_a_whole_circle() -> None:
    """The single-edge shortcut never checked that the edge CLOSES. A quarter
    arc therefore reported the full πr², so a quarter disc of area 0.785
    measured 3.14 — four times too big, as an exact value."""
    from fractions import Fraction as Q

    c = B.Circle((Q(0), Q(0), Q(0)), (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0)), Q(1))
    quarter = B.Loop((B.Edge(c, (Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0))),))
    assert B._loop_is_circle(quarter) is None
    whole = B.Loop((B.Edge(c, (Q(1), Q(0), Q(0)), (Q(1), Q(0), Q(0))),))
    assert B._loop_is_circle(whole) == c


def test_faces_info_declines_the_same_loops_the_exact_path_declines() -> None:
    """Two measurement paths must not give two answers for one face: the exact
    volume refused a mixed arc/line loop while faces_info walked its vertices
    as a polygon and reported 0.5 for a quarter disc of true 0.785."""
    body = B.Body((_slot_face(),))
    with pytest.raises(ValueError, match="mixing arcs and lines"):
        B.faces_info(body)
    with pytest.raises(ValueError, match="mixing arcs and lines"):
        B.centroid(body)


def test_a_circle_whose_ref_is_not_perpendicular_is_refused() -> None:
    from fractions import Fraction as Q

    bad = B.Circle((Q(0), Q(0), Q(0)), (Q(0), Q(0), Q(1)),
                   (Q(1), Q(0), Q(1)), Q(1))
    with pytest.raises(ValueError, match="perpendicular"):
        B._arc_pts(bad, (Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)), 0.1)


def test_a_circle_split_into_three_arcs_is_still_one_circle() -> None:
    """Only two-way splits were recognised; a third intersection turned the
    bore into a 3-point 'polygon' of zero area."""
    from fractions import Fraction as Q

    c = B.Circle((Q(0), Q(0), Q(0)), (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0)), Q(5))
    pts = [(Q(5), Q(0), Q(0)), (Q(3), Q(4), Q(0)), (Q(-5), Q(0), Q(0))]
    loop = B.Loop(tuple(B.Edge(c, pts[i], pts[(i + 1) % 3]) for i in range(3)))
    assert B._loop_is_circle(loop) == c
    assert not B._loop_has_arcs(loop)


def _non_manifold(mesh) -> int:
    ec: dict = defaultdict(int)
    for a, b, c in mesh["triangles"]:
        for e in ((a, b), (b, c), (c, a)):
            ec[tuple(sorted(e))] += 1
    return sum(1 for n in ec.values() if n != 2)


WHOLE_FACES = [
    # a Solid's polys are BSP working units, not faces: a box arrives as 12
    # triangles, an L-prism's caps ear-clipped into 4 each
    ("box", lambda: Solid.box(20, 20, 10), 6, 12),
    ("L-prism", lambda: prism(L_PRISM, 5), 8, 18),
    ("drilled plate", lambda: DrilledSolid(Solid.box(40, 20, 5),
                                           [Cyl(20, 10, 4, 0, 5)]), 7, 14),
    # the ear-clip diagonal runs through the bore centre, so the hole belongs
    # to NEITHER triangle — merging is what makes the question answerable
    ("bore on the ear-clip seam",
     lambda: DrilledSolid(prism([(0, 0), (20, 0), (20, 20), (0, 20)], 10), [])
     .cut(Cyl(10, 10, 2, -1, 11)), 7, 14),
    # a notch across the top: 8-gon section extruded, 8 walls + 2 caps
    ("notched plate",
     lambda: boolean("cut", Solid.box(40, 20, 10),
                     translate(Solid.box(10, 20, 10), 15, 0, 6)), 10, None),
]


@pytest.mark.parametrize("label,build,nfaces,nedges", WHOLE_FACES,
                         ids=[w[0] for w in WHOLE_FACES])
def test_coplanar_fragments_merge_into_whole_faces(label, build, nfaces,
                                                   nedges) -> None:
    """One canonical Face per PLANAR REGION, not per BSP fragment. Fragments
    are not merely verbose — they make "which face carries this feature?"
    unanswerable, and a bore centred on an ear-clip diagonal lies inside
    neither triangle."""
    body = B.to_body(build())
    assert len(body.faces) == nfaces
    if nedges is not None:
        assert len(B.edges_info(body)) == nedges


@pytest.mark.parametrize("label,build,nfaces,nedges", WHOLE_FACES,
                         ids=[w[0] for w in WHOLE_FACES])
def test_merged_faces_still_mesh_watertight(label, build, nfaces,
                                            nedges) -> None:
    assert _non_manifold(B.tessellate(B.to_body(build()), 0.05)) == 0


def test_a_boolean_t_junction_is_split_not_left_to_tear_the_mesh() -> None:
    """A cut leaves one face's long edge facing two short ones. Volume never
    notices; the mesh does — the seam is used once from the long side and
    twice from the short, so an STL of a notched plate leaks."""
    notched = boolean("cut", Solid.box(40, 20, 10),
                      translate(Solid.box(10, 20, 10), 15, 0, 6))
    body = B.to_body(notched)
    assert _non_manifold(B.tessellate(body, 0.05)) == 0
    assert B.volume(body) == PiVal(40 * 20 * 10 - 10 * 20 * 4, 0)


ROTATIONS = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]


@pytest.mark.parametrize("axis", ROTATIONS, ids=["z", "x", "y"])
@pytest.mark.parametrize("deg", [30, 45, 60, 90, 180, 270])
def test_a_rotated_solid_converts_at_all(axis, deg) -> None:
    """``abs()`` on a plane normal killed to_body for EVERY rotated solid.
    Rotation is the one thing 0.9.x shipped exact-angle support for, and a
    rotated coordinate is a SurdVal — which had no ``__abs__``, so this died
    with a bare TypeError rather than any honest refusal. Three separate call
    sites shared the single missing method."""
    from forgekernel.kernel import rotate

    body = B.to_body(rotate(Solid.box(10, 6, 4), axis, deg))
    assert len(body.faces) == 6
    assert B.volume(body) == PiVal(240, 0)


def test_abs_is_defined_over_the_surd_field() -> None:
    from forgekernel.surd import SurdVal

    s = SurdVal(-1, -1, 2)                      # −1 − √2 < 0
    assert abs(s) == -s and abs(-s) == -s
    assert abs(SurdVal(3, 0, 2)) == SurdVal(3, 0, 2)


def _perforated(k):
    from fractions import Fraction as Q
    s = Solid.box(Q(10 * k), Q(10 * k), Q(4))
    for i in range(k):
        for j in range(k):
            s = boolean("cut", s, translate(Solid.box(Q(4), Q(4), Q(20)),
                                            Q(10 * i + 3), Q(10 * j + 3), Q(-2)))
    return s


def test_conversion_stays_near_linear_in_the_polygon_count() -> None:
    """Scanning every vertex for every edge made the T-junction split
    O(polys x edges x verts) in Fractions — 7.1 s on a 966-polygon part
    against 0.010 s before the split existed, growing as the square. Bucketing
    by the edge's carrier line makes the candidate set the bucket."""
    import time

    small, big = _perforated(3), _perforated(8)
    t0 = time.perf_counter()
    B.to_body(small)
    dt_small = time.perf_counter() - t0
    t0 = time.perf_counter()
    B.to_body(big)
    dt_big = time.perf_counter() - t0
    growth = len(big.polys) / len(small.polys)
    # quadratic would be growth**2 (~29x); allow generous slack for noise
    assert dt_big < max(0.05, dt_small * growth * 3)


WATERTIGHT = [
    ("square through-hole",
     lambda: boolean("cut", Solid.box(10, 10, 2),
                     translate(Solid.box(4, 4, 6), 3, 3, -2))),
    ("blind square pocket",
     lambda: boolean("cut", Solid.box(20, 10, 4),
                     translate(Solid.box(4, 4, 20), 8, 3, -5))),
    ("5x5 perforated plate", lambda: _perforated(5)),
    ("notched plate",
     lambda: boolean("cut", Solid.box(40, 20, 10),
                     translate(Solid.box(10, 20, 10), 15, 0, 6))),
]


@pytest.mark.parametrize("label,build", WATERTIGHT, ids=[w[0] for w in WATERTIGHT])
@pytest.mark.parametrize("turn", [None, 45], ids=["upright", "rot45"])
def test_the_display_mesh_is_watertight_and_orientation_independent(
        label, build, turn) -> None:
    """Splitting T-junctions exactly at the FACE level is necessary but not
    sufficient: adjacent faces are triangulated independently in their own 2D
    frames, so the mesher can route the seam differently on each side. The
    volume stays right, which is why it went unnoticed — a hairline crack that
    only shows up when the STL will not print. It was also orientation
    DEPENDENT: 0 upright, 6 after a 45 degree turn, because a float
    collinearity test broke the tie differently per face."""
    body = B.to_body(build())
    if turn:
        body = body.transformed(B.Affine.rotation((1, 0, 0), turn))
    assert _non_manifold(B.tessellate(body, 0.2)) == 0


def test_an_unmergeable_group_falls_back_to_its_fragments() -> None:
    """Two coplanar squares touching at a single corner leave that vertex with
    two ways out — the boundary is ambiguous, so keep honest fragments rather
    than guess a loop."""
    from fractions import Fraction as Q

    def sq(x, y):
        return [(Q(x), Q(y), Q(0)), (Q(x + 1), Q(y), Q(0)),
                (Q(x + 1), Q(y + 1), Q(0)), (Q(x), Q(y + 1), Q(0))]

    pl = B.Plane((Q(0), Q(0), Q(1)), Q(0))
    assert B._merge_coplanar(pl, [sq(0, 0), sq(1, 1)]) is None


# --- cones (ADR-0021 converter) --------------------------------------------

def _frustum_volume(r1, r2, h):
    from fractions import Fraction as Q
    return Q(h) * (Q(r1) ** 2 + Q(r1) * Q(r2) + Q(r2) ** 2) / 3


def _frustum_centroid_z(r1, r2, z0, z1):
    h = z1 - z0
    return z0 + h * (r1 * r1 + 2 * r1 * r2 + 3 * r2 * r2) / (
        4 * (r1 * r1 + r1 * r2 + r2 * r2))


CONES = [
    ("frustum", (0, 0, 2, 5, 0, 10)),
    ("true cone, apex below", (0, 0, 0, 6, 0, 9)),
    ("true cone, apex above", (0, 0, 4, 0, 0, 8)),
    ("narrowing", (0, 0, 5, 2, 0, 10)),
    ("off-axis, offset in z", (7, -3, 3, 6, 2, 8)),
    ("half-integer radii", (0, 0, 1.5, 4.5, 0, 7)),
]


@pytest.mark.parametrize("label,args", CONES, ids=[c[0] for c in CONES])
def test_a_cone_has_an_exact_volume_in_the_pi_field(label, args) -> None:
    """A cone looks like it needs a square root — the slant length carries
    sqrt(1+t^2). It cancels: every point satisfies (x - apex).n = 0, so the
    divergence term is (p.d)*n_ax*Area, where n_ax contributes 1/sqrt(1+t^2)
    and Area contributes the slant's sqrt(1+t^2). A taper is exact."""
    from forgekernel.quadric import Cone as QCone

    cx, cy, r1, r2, z0, z1 = args
    body = B.to_body(QCone(*args))
    assert B.volume(body) == PiVal(0, _frustum_volume(r1, r2, z1 - z0))


@pytest.mark.parametrize("label,args", CONES, ids=[c[0] for c in CONES])
def test_a_cones_centre_of_mass_matches_the_analytic_frustum(label, args) -> None:
    from forgekernel.quadric import Cone as QCone

    cx, cy, r1, r2, z0, z1 = args
    got = B.centroid(B.to_body(QCone(*args)))
    assert got == pytest.approx(
        (cx, cy, _frustum_centroid_z(r1, r2, z0, z1)), abs=1e-9)


@pytest.mark.parametrize("label,args", CONES, ids=[c[0] for c in CONES])
def test_a_cone_meshes_watertight(label, args) -> None:
    """Both rims of a taper have DIFFERENT radii, so a per-circle segment
    count gives the wall and the caps different vertex counts and the mesh
    tears along every rim. The count belongs to the AXIS, not the circle."""
    from forgekernel.quadric import Cone as QCone

    assert _non_manifold(B.tessellate(B.to_body(QCone(*args)), 0.02)) == 0


def test_a_cone_with_equal_radii_is_routed_to_the_cylinder() -> None:
    """Not a degenerate cone — the apex runs off to infinity and every axial
    measurement would divide by a zero slope."""
    from forgekernel.quadric import Cone as QCone

    body = B.to_body(QCone(0, 0, 4, 4, 0, 10))
    assert {type(f.surface).__name__ for f in body.faces} == {"Plane", "Cylinder"}
    assert B.volume(body) == PiVal(0, 16 * 10)


REVOLVES = [
    ("cylinder profile", [(0, 0), (5, 0), (5, 12), (0, 12)], 3),
    ("cone profile", [(0, 0), (6, 0), (0, 9)], 2),
    # an annular profile never touches r = 0: a tube, with a real bore whose
    # wall must face INWARD
    ("tube", [(3, 0), (6, 0), (6, 10), (3, 10)], 4),
    ("stepped shaft", [(0, 0), (4, 0), (4, 5), (7, 5), (7, 9), (0, 9)], 5),
    ("tapered tube", [(2, 0), (5, 0), (4, 8), (3, 8)], 4),
    ("vase", [(0, 0), (6, 0), (6, 3), (3, 6), (5, 10), (0, 10)], 5),
]


@pytest.mark.parametrize("label,profile,nfaces", REVOLVES,
                         ids=[r[0] for r in REVOLVES])
def test_a_lathed_profile_converts_exactly(label, profile, nfaces) -> None:
    """One canonical face per profile segment, all analytic: constant z is an
    annular PLANE, constant r a CYLINDER band, anything else a cone frustum.
    Nothing is faceted, and the volume must equal the representation's own."""
    from forgekernel.quadric import RevolveSolid

    rev = RevolveSolid(profile, 0, 0)
    body = B.to_body(rev)
    assert len(body.faces) == nfaces
    assert B.volume(body) == rev.volume()


@pytest.mark.parametrize("label,profile,nfaces", REVOLVES,
                         ids=[r[0] for r in REVOLVES])
def test_a_lathed_profile_meshes_watertight(label, profile, nfaces) -> None:
    from forgekernel.quadric import RevolveSolid

    assert _non_manifold(
        B.tessellate(B.to_body(RevolveSolid(profile, 0, 0)), 0.02)) == 0


def test_a_tubes_bore_faces_inward() -> None:
    """Orientation comes from the profile's own winding: the outward normal is
    the right-hand perpendicular (dz, -dr), so a segment travelling UP faces
    out and one travelling DOWN faces in. Get it wrong and a tube's bore reads
    as solid — the volume comes back as the outer cylinder."""
    from forgekernel.quadric import RevolveSolid

    tube = RevolveSolid([(3, 0), (6, 0), (6, 10), (3, 10)], 0, 0)
    body = B.to_body(tube)
    walls = {float(f.surface.r): f.sense for f in body.faces
             if isinstance(f.surface, B.Cylinder)}
    assert walls == {6.0: True, 3.0: False}
    assert B.volume(body) == PiVal(0, (36 - 9) * 10)


# --- native text format for the canonical B-rep (ADR-0004/0021) ------------

SERIALIZE = [
    ("drilled plate",
     lambda: DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])),
    ("counterbore", lambda: DrilledSolid(Solid.box(30, 30, 10), [])
     .cut(Cyl(15, 15, 2, 0, 10)).cut(Cyl(15, 15, 4, 7, 10))),
    ("boss", lambda: DisjointUnion([Solid.box(30, 30, 3), Cyl(15, 15, 4, 3, 9)])),
    ("sphere", lambda: Sphere(1, 2, 3, 6)),
    # a ROTATED solid carries Q[sqrt d] coordinates, which a num/den-only
    # format cannot describe at all
    ("45-degree box",
     lambda: __import__("forgekernel.kernel", fromlist=["rotate"]).rotate(
         Solid.box(10, 6, 4), (0, 0, 1), 45)),
]


@pytest.mark.parametrize("label,build", SERIALIZE, ids=[s[0] for s in SERIALIZE])
def test_the_canonical_form_round_trips_through_text_exactly(label, build) -> None:
    """Text is source, geometry is a build artifact (ADR-0004): the round trip
    must be BIT-exact and the bytes canonical, or two equal solids stop
    hashing equal and a git diff stops meaning anything."""
    from forgekernel import io

    body = B.to_body(build())
    text = io.dumps_body(body)
    back = io.loads_body(text)
    assert B.volume(back) == B.volume(body)
    assert len(back.faces) == len(body.faces)
    assert io.dumps_body(back) == text          # byte-canonical


def test_the_text_says_what_the_shape_IS_not_how_it_was_faceted() -> None:
    """A bore serialises as a cylinder with an exact radius. Facets would make
    every revision diff look like a change even when the geometry is
    identical."""
    from forgekernel import io

    text = io.dumps_body(B.to_body(
        DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])))
    assert '"kind":"cylinder"' in text and '"r":"4/1"' in text
    assert '"kind":"circle"' in text


def test_an_unknown_schema_is_refused_rather_than_guessed() -> None:
    from forgekernel import io

    with pytest.raises(ValueError, match="unsupported body schema"):
        io.loads_body('{"schema":"forge/body@99","faces":[]}')


def test_a_zero_height_cone_refuses_instead_of_dividing_by_it() -> None:
    from forgekernel.quadric import Cone as QCone

    with pytest.raises(ValueError, match="zero height"):
        B.to_body(QCone(0, 0, 2, 5, 3, 3))


def test_a_cone_face_spanning_both_nappes_refuses() -> None:
    """Unreachable through the converters (with r >= 0 the apex is outside the
    rim span) but a loaded document can carry one, and no single sign of the
    axial term is right for both halves: it reported -15pi where the truth is
    -5pi."""
    from fractions import Fraction as Q

    apex = (Q(0), Q(0), Q(5))
    axis = (Q(0), Q(0), Q(1))
    rims = tuple(B.Edge(B.Circle((Q(0), Q(0), Q(z)), axis, (Q(1), Q(0), Q(0)),
                                 Q(abs(z - 5))),
                        (Q(abs(z - 5)), Q(0), Q(z)), (Q(abs(z - 5)), Q(0), Q(z)))
                 for z in (4, 7))
    face = B.Face(B.Cone(apex, axis, Q(1)), (B.Loop(rims),), True)
    with pytest.raises(ValueError, match="BOTH nappes"):
        B.volume(B.Body((face,)))


HOSTILE = [
    ('{"schema":"forge/body@1"}', "missing 'faces'"),
    ('{"schema":"forge/body@1","faces":3}', "wrong type"),
    ('{"schema":"forge/body@1","faces":[{"sense":true,"loops":[]}]}',
     "missing 'surface'"),
    ('{"schema":"forge/body@1","faces":[{"surface":{"kind":"sphere","c":'
     '["0/1","0/1","0/1"],"r":"1/0"},"sense":true,"loops":[]}]}', "zero denominator"),
    ('{"schema":"forge/body@1","faces":[{"surface":{"kind":"sphere","c":'
     '["0/1","0/1","0/1"],"r":"S:1/1:1/1:-2/1"},"sense":true,"loops":[]}]}',
     "negative radicand"),
    ('{"schema":"forge/body@1","faces":[{"surface":{"kind":"plane","n":["1/1"],'
     '"d":"0/1"},"sense":true,"loops":[]}]}', "one-component normal"),
    ('{"schema":"forge/body@1","faces":[{"surface":{"kind":"sphere","c":'
     '["0/1","0/1","0/1"],"r":"1/1"},"sense":"yes","loops":[]}]}',
     "sense that is merely truthy"),
    ('[1,2,3]', "not an object"),
]


@pytest.mark.parametrize("doc,why", HOSTILE, ids=[h[1] for h in HOSTILE])
def test_a_hostile_body_document_refuses_rather_than_crashing(doc, why) -> None:
    """A document is UNTRUSTED input (ADR-0006/0007). These escaped as bare
    KeyError/TypeError/ZeroDivisionError, and some loaded SILENTLY — a negative
    surd radicand, a one-component normal, a truthy string for `sense` that
    then reported a volume as fact."""
    from forgekernel import io

    with pytest.raises(ValueError):
        io.loads_body(doc)


def test_face_order_does_not_change_the_bytes() -> None:
    from forgekernel import io

    body = B.to_body(DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)]))
    shuffled = B.Body(tuple(reversed(body.faces)))
    assert io.dumps_body(shuffled) == io.dumps_body(body)


UNIFORM = [
    # a bore under a chamfer: the barrel crosses no wall and its centre is
    # inside the footprint, so BOTH existing checks pass while the material is
    # only half as thick as the removal assumes
    ("under a chamfer", lambda: __import__(
        "forgekernel.kernel", fromlist=["chamfer"]).chamfer(
            Solid.box(20, 20, 4), 2.0), (1.5, 1.0, 0.25), (0, 4), False),
    ("dead centre of a chamfered plate", lambda: __import__(
        "forgekernel.kernel", fromlist=["chamfer"]).chamfer(
            Solid.box(20, 20, 4), 2.0), (10, 10, 2), (0, 4), True),
    # the bore passes through the CAVITY: two slabs, not one, and each is a
    # full barrel in its own right — see the volume test below
    ("through a shelled plate", lambda: __import__(
        "forgekernel.kernel", fromlist=["shell"]).shell(
            Solid.box(20, 20, 10), 2.0), (10, 10, 2), (0, 10), True),
    ("a plain slab", lambda: Solid.box(40, 20, 5), (20, 10, 4), (0, 5), True),
    # non-convex but still one slab through the column
    ("an L-prism", lambda: prism(
        [(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)], 6),
     (4, 4, 2), (0, 6), True),
]


@pytest.mark.parametrize("label,build,bore,zs,ok", UNIFORM,
                         ids=[u[0] for u in UNIFORM])
def test_a_bore_needs_a_full_height_column_under_its_whole_disc(
        label, build, bore, zs, ok) -> None:
    """_bore_union_volume removes pi r^2 (z1-z0) — a FULL barrel — so it is
    only right where the material really is full height across the entire
    disc, not merely under its centre."""
    cx, cy, r = bore
    cut = DrilledSolid(build(), []).cut
    if ok:
        assert cut(Cyl(cx, cy, r, zs[0], zs[1])) is not None
    else:
        with pytest.raises(ValueError, match="K2.1"):
            cut(Cyl(cx, cy, r, zs[0], zs[1]))


# --- trimmed quadrics: the rounded box (ADR-0021, K5.1) ---------------------

ROUNDED = [
    ("cube r3", (20, 20, 20, 3)),
    ("slab r2", (30, 20, 10, 2)),
    ("plate r5", (40, 25, 15, 5)),
    # 2r == the smallest dimension: the core box degenerates and the whole
    # solid IS a sphere — the limiting case the formula must still hit
    ("degenerate: a sphere", (12, 12, 12, 6)),
    ("half-integer radius", (21, 13, 9, 2.5)),
]


@pytest.mark.parametrize("label,args", ROUNDED, ids=[r[0] for r in ROUNDED])
def test_a_rounded_box_matches_steiner_exactly(label, args) -> None:
    """The FIRST trimmed-quadric body: its bands sweep a right angle rather
    than a full turn and its corners are sphere octants. Both stay in Q[pi]
    precisely because the trims are at right angles, where sin and cos are 0
    and +-1 — so the volume must equal Steiner's formula EXACTLY, not nearly.

        V = pqs + 2r(pq+qs+sp) + pi r^2 (p+q+s) + (4/3) pi r^3
    """
    from forgekernel.quadric import RoundedBox

    rb = RoundedBox(*args)
    body = B.to_body(rb)
    assert len(body.faces) == 6 + 12 + 8
    assert B.volume(body) == rb.volume()


@pytest.mark.parametrize("label,args", ROUNDED, ids=[r[0] for r in ROUNDED])
def test_a_rounded_box_meshes_watertight(label, args) -> None:
    """A trimmed band and the corner octants beside it SHARE quarter arcs.
    Projecting a chord midpoint onto a circle lands on the ANGULAR midpoint,
    so a depth-d octant gives 2^d uniform steps per quarter and the band has
    to match it — sampling the band independently tore every seam (912
    unpaired edges)."""
    from forgekernel.quadric import RoundedBox

    assert _non_manifold(B.tessellate(B.to_body(RoundedBox(*args)), 0.1)) == 0


def test_an_arc_off_a_quarter_turn_refuses() -> None:
    """A trim is exact only at right angles: anywhere else the band's
    integral of n_hat carries a transcendental and leaves the field."""
    from fractions import Fraction as Q

    c = B.Circle((Q(0), Q(0), Q(0)), (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0)), Q(5))
    with pytest.raises(ValueError, match="quarter turn"):
        B._arc_quarters(c, (Q(5), Q(0), Q(0)), (Q(3), Q(4), Q(0)))


def test_a_rounded_box_survives_a_rigid_transform_exactly() -> None:
    from forgekernel.quadric import RoundedBox

    rb = RoundedBox(30, 20, 10, 2)
    body = B.to_body(rb)
    exact = B.volume(body)
    for m in (B.Affine.translation(3, -7, 11), B.Affine.mirror("x"),
              B.Affine.rotation((0, 0, 1), 90)):
        assert B.volume(body.transformed(m)) == exact


@pytest.mark.parametrize("label,args", ROUNDED, ids=[r[0] for r in ROUNDED])
def test_a_rounded_boxs_centre_of_mass_is_its_centre(label, args) -> None:
    """centroid used the FULL-band and WHOLE-sphere moments for trimmed faces,
    so a quarter band contributed four times its share. It did not refuse: a
    12-cube's centre of mass came back at (48,48,48), four radii outside its
    own bounding box, with an exact volume underneath it."""
    from forgekernel.quadric import RoundedBox

    rb = RoundedBox(*args)
    want = (float(rb.a) / 2, float(rb.b) / 2, float(rb.c) / 2)
    assert B.centroid(B.to_body(rb)) == pytest.approx(want, abs=1e-9)


def test_a_rounded_boxs_centroid_rides_along_with_a_rotation() -> None:
    """Symmetry alone is a weak oracle for a symmetric solid — check the
    formula in a GENERAL frame too."""
    from forgekernel.quadric import RoundedBox

    body = B.to_body(RoundedBox(30, 20, 10, 2, origin=(5, -3, 7)))
    base = B.centroid(body)
    for axis, deg in (((0, 0, 1), 90), ((1, 0, 0), 90), ((0, 0, 1), 45),
                      ((0, 1, 0), 30)):
        m = B.Affine.rotation(axis, deg)
        want = m.point(tuple(F(x).limit_denominator(10 ** 9) for x in base))
        got = B.centroid(body.transformed(m))
        assert got == pytest.approx([float(x) for x in want], abs=1e-6)


def test_trimmed_faces_report_their_own_area_not_a_whole_turns() -> None:
    """faces_info reported 2pi r h for a QUARTER band and 4 pi r^2 for an
    octant: a rounded box claimed 5247.50 mm2 of surface against a true
    2080.78, and edges_info returned 48 arcs where the solid has 24 (a band
    rim and the octant arc beside it are the SAME arc, but the dedup key used
    a raw normal whose length was r^2)."""
    from forgekernel.quadric import RoundedBox

    body = B.to_body(RoundedBox(20, 20, 20, 3))
    p = q = s = 14
    steiner = (2 * (p * q + q * s + s * p)
               + 2 * math.pi * 3 * (p + q + s) + 4 * math.pi * 9)
    assert sum(f["area"] for f in B.faces_info(body)) == pytest.approx(steiner)

    arcs = [e for e in B.edges_info(body) if e["curve"] == "circle"]
    assert len(arcs) == 24
    assert sum(e["length"] for e in arcs) == pytest.approx(24 * math.pi * 3 / 2)
    # an arc's centroid is ON the arc, not at the circle centre (which for a
    # rounded box is a point strictly inside the solid)
    for e in arcs:
        d = math.dist(e["centroid"], [0, 0, 0])
        assert e["sweep_quarters"] == 1


def test_a_rim_split_into_quarter_arcs_is_still_a_whole_cylinder() -> None:
    """_band_arc reads any non-whole-circle loop as a TRIM, and a rim split
    into four arcs that each carry their own ref is exactly that. Taking the
    sweep from one arc read a d=10 x 12 cylinder as 471.24 against 942.48 —
    half, silently — and the split survives a text round trip."""
    from fractions import Fraction as Q

    r, z0, z1 = Q(5), Q(0), Q(12)
    axis = (Q(0), Q(0), Q(1))
    dirs = [(Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)),
            (Q(-1), Q(0), Q(0)), (Q(0), Q(-1), Q(0))]

    def rim(z, n):
        pts = [tuple(r * d[i] + (z if i == 2 else Q(0)) for i in range(3))
               for d in dirs]
        order = range(4) if n == axis else range(3, -1, -1)
        idx = list(order)
        return tuple(B.Edge(B.Circle((Q(0), Q(0), z), n, dirs[idx[k]], r),
                            pts[idx[k]], pts[idx[(k + 1) % 4]])
                     for k in range(4))

    wall = B.Face(B.Cylinder((Q(0), Q(0), Q(0)), axis, r),
                  (B.Loop(rim(z0, axis) + rim(z1, tuple(-x for x in axis))),),
                  True)
    caps = [f for f in B.to_body(Cyl(0, 0, 5, 0, 12)).faces
            if isinstance(f.surface, B.Plane)]
    assert B.volume(B.Body(tuple(caps) + (wall,))) == PiVal(0, 300)


def test_a_spherical_patch_that_is_not_an_octant_refuses() -> None:
    """Summing three corners and reading off signs accepted 143 trios of
    rational unit vectors that are not the signed axes; one measured 2.68x its
    true term, and reversing the loop named the complementary 7/8 patch and
    read identically."""
    from fractions import Fraction as Q

    c = (Q(3), Q(-5), Q(7))
    rels = [(Q(0), Q(0), Q(1)), (Q(1, 3), Q(2, 3), Q(2, 3)),
            (Q(2, 3), Q(1, 3), Q(-2, 3))]
    pts = [tuple(c[i] + v[i] for i in range(3)) for v in rels]
    edges = tuple(B.Edge(B.Circle(c, (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0)), Q(1)),
                         pts[k], pts[(k + 1) % 3]) for k in range(3))
    face = B.Face(B.SphereS(c, Q(1)), (B.Loop(edges),), True)
    with pytest.raises(ValueError, match="perpendicular radii"):
        B.volume(B.Body((face,)))


def test_a_uniform_scale_of_a_rounded_box_no_longer_refuses() -> None:
    """Affine.direction does not normalise, so a scaled body carries
    |n| = |ref| = k. Demanding EXACTLY unit length lost the volume and — worse
    — the MESH, gating a display property on an exactness predicate."""
    from forgekernel.quadric import RoundedBox

    body = B.to_body(RoundedBox(20, 20, 20, 3))
    exact = B.volume(body)
    for k in (2, 3):
        scaled = body.transformed(B.Affine.scaling(k, k, k))
        assert B.volume(scaled) == PiVal(exact.a * k ** 3,
                                         exact.b * k ** 3)
        assert _non_manifold(B.tessellate(scaled, 0.2)) == 0


# --- torus: the surface a blend sweeps (needs pi^2) ------------------------

def test_a_whole_torus_is_pappus_exactly() -> None:
    """V = 2 pi R * pi a^2. This is the first volume that does NOT fit in
    PiVal's a + b*pi, which is why Q[pi] became a polynomial ring."""
    from forgekernel.polypi import PiPoly

    t = B.Face(B.Torus((F(0), F(0), F(0)), (F(0), F(0), F(1)), F(10), F(2)),
               (), True)
    assert B.volume(B.Body((t,))) == PiPoly.term(2 * 4 * 10, 2)


def test_a_whole_torus_is_centred_on_its_own_centre() -> None:
    for c in ((0, 0, 0), (3, -4, 7)):
        t = B.Face(B.Torus(tuple(F(x) for x in c), (F(0), F(0), F(1)),
                           F(10), F(2)), (), True)
        assert B.centroid(B.Body((t,))) == pytest.approx(c, abs=1e-9)


def test_a_filleted_cylinder_matches_green_by_hand() -> None:
    """A fillet lives on the (r, z) PROFILE, so the whole solid's volume is
    Green's pi * contour integral of r^2 dz. For a d=10 x 12 cylinder rounded
    1 mm the wall contributes 250 and each arc 2pi + 50/3, giving
    pi(850/3) + 4 pi^2 — the pi^2 term being the torus.

    Built here from the segments directly, so forge stays self-contained.
    """
    from forgekernel.polypi import PiPoly

    segs = [("line", (F(0), F(0)), (F(4), F(0))),
            ("arc", (F(4), F(1)), (F(4), F(0)), (F(5), F(1))),
            ("line", (F(5), F(1)), (F(5), F(11))),
            ("arc", (F(4), F(11)), (F(5), F(11)), (F(4), F(12))),
            ("line", (F(4), F(12)), (F(0), F(12)))]
    body = B.lathe_body(segs, F(0), F(0))
    assert B.volume(body) == PiPoly([0, F(850, 3), 4])
    assert sum(1 for f in body.faces if isinstance(f.surface, B.Torus)) == 2


def test_a_torus_sweeping_past_a_full_turn_refuses() -> None:
    t = B.Face(B.Torus((F(0), F(0), F(0)), (F(0), F(0), F(1)), F(10), F(2),
                       0, 5), (), True)
    with pytest.raises(ValueError, match="full turn"):
        B.volume(B.Body((t,)))


def _filleted_cylinder():
    segs = [("line", (F(0), F(0)), (F(4), F(0))),
            ("arc", (F(4), F(1)), (F(4), F(0)), (F(5), F(1))),
            ("line", (F(5), F(1)), (F(5), F(11))),
            ("arc", (F(4), F(11)), (F(5), F(11)), (F(4), F(12))),
            ("line", (F(4), F(12)), (F(0), F(12)))]
    return B.lathe_body(segs, F(0), F(0))


TRANSFORMS = [
    ("translate", lambda: B.Affine.translation(1, 2, 3)),
    ("rotate z90", lambda: B.Affine.rotation((0, 0, 1), 90)),
    ("rotate z45", lambda: B.Affine.rotation((0, 0, 1), 45)),
    ("rotate x90", lambda: B.Affine.rotation((1, 0, 0), 90)),
    ("mirror x", lambda: B.Affine.mirror("x")),
]


@pytest.mark.parametrize("label,make", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_a_filleted_lathe_survives_a_rigid_transform(label, make) -> None:
    """The Torus term was the only one not pushing its coordinate-derived
    scalar through _exact(). After any rigid map those coordinates are SurdVal
    (with a zero surd part), and handing one to Fraction() is a bare TypeError
    — so EVERY transform of a filleted part crashed."""
    body = _filleted_cylinder()
    assert B.volume(body.transformed(make())) == B.volume(body)


def test_a_torus_contributes_to_the_bounding_box() -> None:
    """A torus carries no loops, so it contributed NOTHING to the bound. When
    a fillet consumes the whole adjacent face the extreme lives only on the
    torus — and part.interference uses the AABB as a PRE-FILTER, so two
    filleted barrels that physically interlock came back clean."""
    body = _filleted_cylinder()
    lo, hi = B.bbox(body)
    assert (lo[2], hi[2]) == pytest.approx((0.0, 12.0))
    assert (lo[0], hi[0]) == pytest.approx((-5.0, 5.0))


@pytest.mark.parametrize("deflection", [0.2, 0.05, 0.02, 0.01, 0.005, 0.002])
def test_a_rotated_fillet_still_meets_the_wall_it_blends(deflection) -> None:
    """The torus built its own theta reference while every coaxial ring uses
    circle.ref. Identical unrotated — both (1,0,0) — and 90 degrees apart
    after a z-rotation, which interleaved the two rings and tore the seam at
    some deflections but not others."""
    body = _filleted_cylinder().transformed(B.Affine.rotation((0, 0, 1), 90))
    assert _non_manifold(B.tessellate(body, deflection)) == 0


def test_a_torus_face_reports_its_own_area() -> None:
    """faces_info emitted a bare {"surface": "torus"} — an 18% under-report to
    anyone summing areas, and nothing for a dimension to anchor to."""
    info = B.faces_info(_filleted_cylinder())
    tori = [f for f in info if f["surface"] == "torus"]
    assert len(tori) == 2
    # a quarter tube: 2*pi*a*(R*phi + a*dsin) with R=4, a=1, phi=pi/2
    assert all(t["area"] == pytest.approx(2 * math.pi * (4 * math.pi / 2 + 1))
               for t in tori)
    assert sum(f["area"] for f in info) == pytest.approx(506.2134, rel=1e-4)


def test_a_clockwise_lathe_arc_refuses_rather_than_reading_three_quarters() -> None:
    """span = (k1 - k0) % 4 assumes counter-clockwise, so a CW quarter reads
    as three. fillet never produces one, but lathe_body is public API."""
    segs = [("line", (F(0), F(0)), (F(4), F(0))),
            ("arc", (F(4), F(1)), (F(5), F(1)), (F(4), F(0))),
            ("line", (F(4), F(0)), (F(0), F(0)))]
    with pytest.raises(ValueError, match="clockwise"):
        B.lathe_body(segs, F(0), F(0))


def test_a_lathe_arc_off_a_quarter_turn_refuses_by_name() -> None:
    segs = [("arc", (F(0), F(0)), (F(5), F(0)), (F(3), F(4)))]
    with pytest.raises(ValueError, match="quarter turn"):
        B.lathe_body(segs, F(0), F(0))


T_PROFILE = [(0, 0), (9, 0), (9, 3), (6, 3), (6, 9), (3, 9), (3, 3), (0, 3)]
I_BEAM = [(0, 0), (9, 0), (9, 2), (6, 2), (6, 7), (9, 7), (9, 9), (0, 9),
          (0, 7), (3, 7), (3, 2), (0, 2)]


@pytest.mark.parametrize("name,prof,area", [("T", T_PROFILE, 45),
                                            ("I-beam", I_BEAM, 51)])
def test_a_rectilinear_profile_with_a_vertex_on_a_diagonal_extrudes(
        name, prof, area) -> None:
    """``_ear_clip`` counted a blocker only in the OPEN triangle, so a vertex
    lying exactly ON a candidate ear's diagonal did not block it. The ear was
    clipped straight through that vertex and the remainder collapsed to
    collinear points — so a T and an I-beam, the two most ordinary non-convex
    profiles there are, came back as ``degenerate profile``. A blocker is a
    point in the CLOSED triangle."""
    from fractions import Fraction as Q

    from forgekernel.brep import _ear_clip
    from forgekernel.kernel import prism

    tris = _ear_clip([(Q(x), Q(y)) for x, y in prof])
    assert len(tris) == len(prof) - 2
    got = sum(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
              for a, b, c in tris) / 2
    assert got == area, "the triangles must tile the profile exactly"
    assert prism(prof, 5).volume() == area * 5


MANIFOLD_SHAPES = [
    ("box", lambda: Solid.box(20, 20, 10)),
    ("L-prism", lambda: prism([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10),
                               (0, 10)], 5)),
    ("T-prism", lambda: prism(T_PROFILE, 5)),
    ("drilled plate",
     lambda: DrilledSolid(Solid.box(40, 20, 5), [Cyl(20, 10, 4, 0, 5)])),
    ("blind hole",
     lambda: DrilledSolid(Solid.box(30, 30, 10), [Cyl(15, 15, 3, 4, 10)])),
    ("bare cylinder", lambda: Cyl(0, 0, 5, 0, 12)),
    ("cone", lambda: Cone(0, 0, 2, 5, 0, 10)),
    ("sphere", lambda: Sphere(1, 2, 3, 6)),
    ("rounded box", lambda: RoundedBox(20, 20, 20, 3)),
    ("revolve", lambda: RevolveSolid(
        [(0, 0), (4, 0), (4, 5), (7, 5), (7, 9), (0, 9)], 0, 0)),
]


@pytest.mark.parametrize("label,build", MANIFOLD_SHAPES,
                         ids=[s[0] for s in MANIFOLD_SHAPES])
def test_every_corpus_body_pairs_all_of_its_edges(label, build) -> None:
    """In a closed shell every edge is shared by exactly two faces. This is the
    check ``validate`` could not run on a Body at all — it reached for
    ``Solid.watertight_violations``, which a Body does not have, so every
    pocketed or curved-shelled solid came back "unsupported representation".
    Unvalidatable reads far too much like fine."""
    assert B.manifold_violations(B.to_body(build())) == []


def test_a_body_missing_a_face_is_reported_unpaired() -> None:
    body = B.to_body(Solid.box(10, 10, 10))
    bad = B.manifold_violations(B.Body(body.faces[:-1]))
    assert len(bad) == 4 and all("used by 1 face" in m for m in bad)


def test_the_edge_key_is_exact_and_direction_free() -> None:
    """Both faces at an edge must land on the same key however each traverses
    it, and the key must not round: two edges a micron apart are two edges."""
    from fractions import Fraction as Q

    p, q = (Q(0), Q(0), Q(0)), (Q(1), Q(2), Q(3))
    fwd = B.Edge(B.Line(p, tuple(q[i] - p[i] for i in range(3))), p, q)
    rev = B.Edge(B.Line(q, tuple(p[i] - q[i] for i in range(3))), q, p)
    assert B._edge_key(fwd) == B._edge_key(rev)
    near = (Q(1), Q(2), Q(3) + Q(1, 10 ** 6))
    off = B.Edge(B.Line(p, tuple(near[i] - p[i] for i in range(3))), p, near)
    assert B._edge_key(fwd) != B._edge_key(off)


SLABS = [
    # (label, base, bore, exact removed volume as a multiple of pi)
    ("through a shelled plate",
     lambda: __import__("forgekernel.kernel", fromlist=["shell"]).shell(
         Solid.box(20, 20, 10), 2.0), Cyl(10, 10, 2, 0, 10), 4 * 2 + 4 * 2),
    ("through a shelled cube",
     lambda: __import__("forgekernel.kernel", fromlist=["shell"]).shell(
         Solid.box(20, 20, 20), 2.0), Cyl(10, 10, 1, 0, 20), 1 * 2 + 1 * 2),
    # only the top wall: the tool stops inside the cavity, so the lower slab
    # contributes nothing and clipping must drop it rather than remove air
    ("stopping inside the cavity",
     lambda: __import__("forgekernel.kernel", fromlist=["shell"]).shell(
         Solid.box(20, 20, 20), 2.0), Cyl(10, 10, 1, 10, 20), 1 * 2),
    ("a plain slab is still one barrel", lambda: Solid.box(40, 20, 5),
     Cyl(20, 10, 4, 0, 5), 16 * 5),
]


@pytest.mark.parametrize("label,build,bore,removed", SLABS,
                         ids=[s[0] for s in SLABS])
def test_a_bore_through_a_cavity_removes_each_slab_and_no_air(
        label, build, bore, removed) -> None:
    """A bore's column may be a STACK of slabs — through a shelled plate it is
    the top wall and the bottom wall with the void between. One full barrel
    over the whole span would have taken the cavity's air as if it were metal,
    so the kernel refused outright ("meets 4 horizontal levels, not 2"). One
    bore per slab is exact: no lateral face reaches the disc, so the
    cross-section is constant and each slab really is a full barrel."""
    base = build()
    out = DrilledSolid(base, []).cut(bore)
    got = PiVal(base.volume(), F(0)) - out.volume()
    assert got == PiVal(F(0), F(removed)), f"removed {got}, want {removed} pi"
    body = B.to_body(out)
    assert B.manifold_violations(body) == []
    from forgekernel.stepbody import write_step_body
    write_step_body(body)                 # its own closing audit runs here


def test_an_odd_level_count_still_refuses() -> None:
    """Two slabs is a stack; an odd count cannot be one, so the material over
    the disc is not decidable and the barrel arithmetic must not guess."""
    from forgekernel.quadric import _column_slabs

    sh = __import__("forgekernel.kernel", fromlist=["shell"]).shell(
        Solid.box(20, 20, 10), 2.0)
    assert _column_slabs(sh, Cyl(10, 10, 2, 0, 10)) == [
        (F(0), F(2)), (F(8), F(10))]
    # a chamfer's taper is a non-horizontal face over the disc: still refused,
    # because there the cross-section is not constant at all
    ch = __import__("forgekernel.kernel", fromlist=["chamfer"]).chamfer(
        Solid.box(20, 20, 4), 2.0)
    with pytest.raises(ValueError, match="K2.1"):
        _column_slabs(ch, Cyl(1.5, 1.0, 0.25, 0, 4))


ROTATIONS = [0, 30, 45, 60, 90]


@pytest.mark.parametrize("deg", ROTATIONS)
@pytest.mark.parametrize("deflection", [0.3, 0.05])
def test_a_rounded_box_meshes_the_same_however_it_is_turned(deg, deflection):
    """A sphere octant's three corners are the loop's OWN vertices. They used
    to be rebuilt as centre + r along each signed GLOBAL axis, reading
    ``_sphere_octant``'s return as if it were a sign triple — which it only is
    when the patch happens to be axis-aligned. That function returns the SUM of
    the three radii (frame-free, which is what the exact volume needs), so for
    a rotated corner it is a diagonal of length r*sqrt(3) and the subdivision
    meshed the wrong spherical triangle entirely.

    A 20-cube filleted r=3 and turned 45 degrees about z came out with 88 of
    its 170 mesh edges unpaired — an STL full of holes — while 0 and 90 degrees
    were clean and the exact volume was right the whole time. Meshing is a
    display property, so nothing exact caught it.

    Rotation is an isometry: the triangle count, the edge pairing and the mesh
    volume must all be invariant."""
    from forgekernel.kernel import fillet_box, rotate

    base = fillet_box(20, 20, 20, 3)
    shape = base if deg == 0 else B.to_body(base).transformed(
        B.Affine.rotation((0, 0, 1), deg))
    body = shape if isinstance(shape, B.Body) else B.to_body(shape)
    mesh = B.tessellate(body, deflection)

    ec = defaultdict(int)
    for a, b, c in mesh["triangles"]:
        for e in ((a, b), (b, c), (c, a)):
            ec[tuple(sorted(e))] += 1
    assert [n for n in ec.values() if n != 2] == [], "the mesh is not closed"

    ref = B.tessellate(B.to_body(base), deflection)
    assert len(mesh["triangles"]) == len(ref["triangles"])
    assert mesh_volume(mesh) == pytest.approx(mesh_volume(ref), rel=1e-12)
    # and it converges to the exact volume from inside
    exact = float(B.volume(body))
    assert 0 < mesh_volume(mesh) <= exact * (1 + 1e-12)
