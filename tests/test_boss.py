"""Bosses and bored cylinders — a z-cylinder standing on a planar face
(tangent, measure-zero) and coaxial bores through solids of revolution.

A cylinder minus a coaxial bore is still a solid of revolution, so the result
stays an exact ℚ[π] object rather than degrading to a mesh.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from forgekernel.brep import Solid
from forgekernel.quadric import Cyl, DisjointUnion, RevolveSolid, bore_cyl


def _nonmanifold(mesh):
    ec = defaultdict(int)
    for a, b, c in mesh["triangles"]:
        for e in ((a, b), (b, c), (c, a)):
            ec[tuple(sorted(e))] += 1
    return sum(1 for n in ec.values() if n != 2)


def test_boss_standing_on_a_plate_is_a_tangent_union() -> None:
    # the cylinder's base sits EXACTLY on the plate's top face: contact is
    # measure-zero, so the union volume is exactly the sum.
    plate = Solid.box(30, 30, 3)
    boss = Cyl(15, 15, 4, 3, 9)
    u = DisjointUnion([plate, boss])
    assert float(u.volume()) == pytest.approx(30 * 30 * 3 + math.pi * 16 * 6)


def test_boss_sunk_into_the_plate_is_refused() -> None:
    plate = Solid.box(30, 30, 3)
    with pytest.raises(ValueError, match="overlaps the solid"):
        DisjointUnion([plate, Cyl(15, 15, 4, 2, 9)])     # base below the top face


def test_cylinder_beside_the_plate_is_separated_by_a_side_face() -> None:
    plate = Solid.box(10, 10, 10)
    DisjointUnion([plate, Cyl(20, 5, 3, 0, 10)])         # clear in x: no raise


def test_non_convex_solids_refuse_rather_than_double_count() -> None:
    """SOUNDNESS: the separating-plane test only holds for a convex solid.

    In an L-prism, a bore buried in one arm lies outside the plane of the
    OTHER arm's face — so a plane-only test would call it disjoint and silently
    add its volume twice. Both predicates must refuse instead."""
    from forgekernel import kernel as fk
    from forgekernel.quadric import Sphere

    ell = fk.prism([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)], 5)
    for buried in (Cyl(2, 7, 1, 1, 4), Sphere(2, 7, 2, 1)):
        with pytest.raises(ValueError, match="non-convex"):
            DisjointUnion([ell, buried])


def test_a_bore_outside_the_footprint_is_not_a_phantom_hole() -> None:
    """SOUNDNESS: "the bore crosses no lateral wall" does not mean "the bore is
    inside" — a bore entirely OUTSIDE the footprint crosses nothing either. It
    used to be recorded as a hole, silently deleting volume from a solid the
    tool never touched (a Ø4 tool 40 mm away removed 125.66 mm³ from a box)."""
    from forgekernel.quadric import DrilledSolid

    box = Solid.box(10, 10, 10)
    with pytest.raises(ValueError, match="misses the solid in xy"):
        DrilledSolid(box, []).cut(Cyl(50, 50, 2, 0, 10))
    inside = DrilledSolid(box, []).cut(Cyl(5, 5, 2, 0, 10))     # a real hole
    assert float(inside.volume()) == pytest.approx(1000 - math.pi * 4 * 10)


def test_cut_leaves_members_the_tool_misses_untouched() -> None:
    # a post standing 50 mm clear of a plate; boring the POST must not shrink
    # the plate (the tool never enters it in xy).
    plate, post = Solid.box(100, 100, 10), Cyl(150, 50, 8, 0, 20)
    out = DisjointUnion([plate, post]).cut(Cyl(150, 50, 4, 0, 20))
    assert float(out.volume()) == pytest.approx(100000 + (64 - 16) * 20 * math.pi)


def test_coaxial_bore_of_a_cylinder_is_an_exact_revolve() -> None:
    cyl = Cyl(0, 0, 10, 0, 2)
    washer = bore_cyl(cyl, Cyl(0, 0, 4, 0, 2))           # through bore: annulus
    assert isinstance(washer, RevolveSolid)
    assert float(washer.volume()) == pytest.approx(math.pi * (100 - 16) * 2)
    mesh = washer.tessellate(0.05)
    assert _nonmanifold(mesh) == 0                       # watertight tube

    blind = bore_cyl(Cyl(0, 0, 4, 3, 9), Cyl(0, 0, 1, 4, 9))   # blind from top
    assert float(blind.volume()) == pytest.approx(math.pi * 16 * 6 - math.pi * 1 * 5)

    up = bore_cyl(Cyl(0, 0, 4, 0, 6), Cyl(0, 0, 1, -5, 2))     # blind from below
    assert float(up.volume()) == pytest.approx(math.pi * 16 * 6 - math.pi * 1 * 2)


def test_bore_that_misses_in_z_leaves_the_cylinder_unchanged() -> None:
    cyl = Cyl(0, 0, 4, 0, 6)
    assert bore_cyl(cyl, Cyl(0, 0, 1, 10, 12)) is cyl


def test_non_representable_bores_refuse() -> None:
    cyl = Cyl(0, 0, 4, 0, 6)
    with pytest.raises(ValueError, match="not coaxial"):
        bore_cyl(cyl, Cyl(1, 0, 1, 0, 6))
    with pytest.raises(ValueError, match="not narrower"):
        bore_cyl(cyl, Cyl(0, 0, 4, 0, 6))
    with pytest.raises(ValueError, match="closed cavity"):
        bore_cyl(cyl, Cyl(0, 0, 1, 2, 4))                # floats strictly inside


def test_cut_distributes_over_a_disjoint_union() -> None:
    # the boss + pilot: (plate ∪ boss) ∖ pilot, pilot blind in the boss only
    plate = Solid.box(30, 30, 3)
    u = DisjointUnion([plate, Cyl(15, 15, 4, 3, 9)])
    out = u.cut(Cyl(15, 15, 1, 4, 9))
    expect = 30 * 30 * 3 + math.pi * 16 * 6 - math.pi * 1 * 5
    assert float(out.volume()) == pytest.approx(expect)
    # the plate member is untouched (the tool misses it in z)
    assert any(isinstance(m, Solid) for m in out.members)
    assert _nonmanifold(out.tessellate(0.05)) == 0


def test_boss_on_plate_mesh_buries_no_faces_in_the_contact_disc() -> None:
    """The flagship boss-on-plate must mesh with NO material-buried faces.

    Concatenating the member shells emitted the plate's top face as a full
    rectangle and laid the boss's bottom cap on top of it — every triangle
    in the contact disc had solid on both sides. Volume and per-shell
    closure stayed right (both shells are individually closed), which is
    exactly why nothing caught it: the lie was topological, and any
    slicer/half-edge consumer saw internal walls.

    The stitched mesh must have zero z=5 triangles covering the contact
    region, still enclose the same volume, and pair every edge BY INDEX —
    one closed surface, not two shells hiding a wall."""
    plate = Solid.box(30, 30, 5)
    boss = Cyl(15, 15, 6, 5, 15)
    mesh = DisjointUnion([plate, boss]).tessellate(0.05)
    verts, tris = mesh["vertices"], mesh["triangles"]

    # probe a grid of points well inside the contact disc: NO triangle lying
    # in the plane z=5 may cover any of them (in the old mesh each is covered
    # twice: plate top + boss cap)
    flat = [[verts[i] for i in t] for t in tris
            if all(verts[i][2] == 5.0 for i in t)]

    def covered(px, py) -> int:
        hits = 0
        for a, b, c in flat:
            d1 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
            d2 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
            d3 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
            if (d1 >= 0 and d2 >= 0 and d3 >= 0) or \
               (d1 <= 0 and d2 <= 0 and d3 <= 0):
                hits += 1
        return hits
    probes = [(15 + dx, 15 + dy) for dx in (-3.7, -1.3, 0.0, 1.9, 3.1)
              for dy in (-3.9, -1.1, 0.7, 2.3, 3.3)]
    # all probes sit well inside even the INSCRIBED ring polygon of the
    # boss wall (r·cos(pi/n) > 5.9 at this deflection)
    assert all((px - 15) ** 2 + (py - 15) ** 2 < 34 for px, py in probes)
    buried = sum(covered(px, py) for px, py in probes)
    assert buried == 0

    # the enclosed volume is untouched by the stitch: still the sum of the
    # plate and the boss's inscribed-polygon prism
    from forgekernel.tess import mesh_volume
    plate_v = mesh_volume(plate.tessellate())
    boss_v = mesh_volume(boss.tessellate(0.05))
    assert mesh_volume(mesh) == pytest.approx(plate_v + boss_v, rel=1e-9)

    # and the stitched result is ONE closed surface: every edge pairs by
    # vertex index (the hole rim reuses the boss wall's ring vertices)
    assert _nonmanifold(mesh) == 0


def test_overhanging_boss_keeps_the_merged_shell_fallback() -> None:
    # the boss overhangs the plate edge: the contact patch is not a full
    # disc, the single-hole stitch does not apply, and the union must keep
    # meshing as two individually-closed shells (volume still the sum)
    plate = Solid.box(30, 30, 5)
    boss = Cyl(30, 15, 6, 5, 15)                  # centre ON the plate edge
    u = DisjointUnion([plate, boss])
    mesh = u.tessellate(0.05)
    from forgekernel.tess import mesh_volume
    plate_v = mesh_volume(plate.tessellate())
    boss_v = mesh_volume(boss.tessellate(0.05))
    assert mesh_volume(mesh) == pytest.approx(plate_v + boss_v, rel=1e-9)
    assert _nonmanifold(mesh) == 0


def test_revolve_solid_can_sit_off_axis() -> None:
    loop = [(0, 0), (2, 0), (2, 5), (0, 5)]              # a plain cylinder r=2
    at_origin, moved = RevolveSolid(loop), RevolveSolid(loop, 7, -3)
    assert float(at_origin.volume()) == float(moved.volume())
    assert moved.centroid_f()[0] == pytest.approx(7.0)
    assert moved.centroid_f()[1] == pytest.approx(-3.0)
    lo, hi = moved.bbox()
    assert (lo[0], hi[0]) == pytest.approx((5.0, 9.0))
    assert (lo[1], hi[1]) == pytest.approx((-5.0, -1.0))
