"""The canonical B-rep (ADR-0021) — one topology every representation maps to.

The acceptance test for the architecture is the SHADOW RUN the ADR requires: a
representation's own exact metrics must reproduce after converting it to
``Body``. Exact equality in ℚ[π], not a float tolerance — that is the whole
point of the canonical form preserving exactness.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from forgekernel import body as B
from forgekernel.brep import Solid
from forgekernel.kernel import boolean, prism, translate
from forgekernel.quadric import Cyl, DisjointUnion, DrilledSolid, PiVal, Sphere
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
