import math
"""K1 acceptance: exactness asserted with EXACT EQUALITY — the point of
a rational kernel is that == is the right operator."""

from fractions import Fraction

from forgekernel.brep import Solid
from forgekernel.exact import F
from forgekernel.kernel import boolean, box, mirror, prism, rotate_quarter, scale, translate


def test_box_metrics_exact() -> None:
    b = box(30, 20, 10)
    assert b.volume() == 6000
    assert b.centroid() == (15, 10, 5)
    assert b.watertight_violations() == []
    assert len(b.logical_faces()) == 6


def test_prism_L_volume_exact() -> None:
    L = [(0, 0), (30, 0), (30, 8), (8, 8), (8, 25), (0, 25)]
    p = prism(L, 12)
    area = Fraction(30 * 8 + 8 * 17)
    assert p.volume() == area * 12
    assert p.watertight_violations() == []


def test_boolean_volumes_exact() -> None:
    a = box(10, 10, 10)
    b = translate(box(10, 10, 10), 5, 5, 5)
    assert boolean("union", a, b).volume() == 2000 - 125
    assert boolean("cut", a, b).volume() == 1000 - 125
    assert boolean("intersect", a, b).volume() == 125


def test_coincident_face_union_exact() -> None:
    a = box(20, 20, 10)
    b = translate(box(20, 20, 10), 20, 0, 0)
    u = boolean("union", a, b)
    assert u.volume() == 8000
    assert u.watertight_violations() == []


def test_sliver_cut_exact_even_at_float_precision() -> None:
    a = box(30, 30, 10)
    b = translate(box(30, 30, 10), 29.999, 0, 0)
    out = boolean("cut", a, b)
    expected = 9000 - (30 - F(29.999)) * 30 * 10   # same-float arithmetic
    assert out.volume() == expected
    assert out.watertight_violations() == []


def test_union_idempotent_and_disjoint_additive() -> None:
    a = box(10, 10, 10)
    assert boolean("union", a, a).volume() == 1000
    far = translate(box(5, 5, 5), 100, 0, 0)
    assert boolean("union", a, far).volume() == 1125


def test_transforms_exact() -> None:
    a = translate(box(10, 20, 30), 1, 2, 3)
    r = rotate_quarter(a, "z", 1)
    assert r.volume() == 6000
    assert r.watertight_violations() == []
    m = mirror(a, "x")
    assert m.volume() == 6000
    s = scale(a, 2)
    assert s.volume() == 48000
    s2 = scale(a, 2, 1, 1)
    assert s2.volume() == 12000


def test_lineage_survives_booleans() -> None:
    a = box(10, 10, 10, ) if False else Solid.box(10, 10, 10, "A")
    b = translate(Solid.box(10, 10, 10, "B"), 5, 5, 5)
    u = boolean("union", a, b)
    sources = {src for _, src in u.logical_faces()}
    assert any(s.startswith("A.") for s in sources)
    assert any(s.startswith("B.") for s in sources)


def test_chamfered_cube_topology_and_exactness() -> None:
    from forgekernel.kernel import chamfer

    c = chamfer(box(10, 10, 10), 2)
    assert c.watertight_violations() == []
    # hand-derived, twice over: edge cuts give 1000 - 480/wedges + 64/pairs
    # - 16/triples = 808; industrial corner facets remove d^3/12 per corner
    # (8 x 2/3) more. ref returns the EXACT rational OCCT approximates.
    assert c.volume() == Fraction(2408, 3)
    # the classic chamfered-cube topology: 6 octagons + 12 chamfer faces
    # + 8 corner triangles = 26 planes
    planes = {key[0] for key in c.logical_faces()}
    assert len(planes) == 26
    assert chamfer(box(10, 10, 10), 2).volume() == c.volume()


def test_chamfer_block_matches_oracle_exactly() -> None:
    from forgekernel.kernel import chamfer

    # the first real ref-vs-OCCT disagreement, resolved: OCCT reports
    # 5562.666666666667 for box(30,20,10) chamfer 2 — ref returns the
    # exact rational behind that float.
    assert chamfer(box(30, 20, 10), 2).volume() == Fraction(16688, 3)


def test_serialization_round_trip_bit_exact() -> None:
    from forgekernel import io
    from forgekernel.kernel import boolean

    s = boolean("cut", box(30, 30, 10),
                translate(box(30, 30, 10), 29.999, 0, 0))
    text = io.dumps(s)
    s2 = io.loads(text)
    assert io.dumps(s2) == text                  # bit-exact round trip
    assert s2.volume() == s.volume()
    stl = io.to_stl(s)
    assert stl.startswith("solid") and "endsolid" in stl


# -- K2.0: exact ℚ[π] drilled solids ------------------------------------------

def test_pival_field_arithmetic() -> None:
    from forgekernel.quadric import PiVal

    v = PiVal(9600) - PiVal(0, 100)
    assert v == PiVal(9600, -100)
    assert abs(float(v) - (9600 - 100 * 3.141592653589793)) < 1e-12


def test_drilled_plate_volume_is_exact_in_pi() -> None:
    from forgekernel.quadric import Cyl, DrilledSolid, PiVal

    plate = DrilledSolid(box(60, 40, 4), [])
    for i in range(4):
        plate = plate.cut(Cyl.make(Fraction(5, 2), 4).translated(10 + 13 * i, 20, 0))
    # EXACTLY 9600 - 4·π·(5/2)²·4 = 9600 - 25π... per hole: 6.25·4=25
    assert plate.volume() == PiVal(9600, -100)
    assert len(plate.cylinder_faces()) == 4
    assert plate.cylinder_faces()[0]["surface"] == "cylinder"


def test_counterbore_stack_unions_by_z() -> None:
    from forgekernel.quadric import Cyl, DrilledSolid, PiVal

    base = DrilledSolid(box(20, 20, 10), [])
    base = base.cut(Cyl.make(2, 10).translated(10, 10, 0))          # thru r2
    base = base.cut(Cyl.make(4, 3).translated(10, 10, 7))           # cbore r4
    # removed = π(4·7 + 16·3) = π·76
    assert base.volume() == PiVal(400 * 10, -76)


def test_drill_preconditions_refuse_exactly() -> None:
    import pytest as _pytest

    from forgekernel.quadric import Cyl, DrilledSolid

    base = DrilledSolid(box(20, 20, 10), [])
    with _pytest.raises(ValueError, match="lateral wall"):
        base.cut(Cyl.make(3, 10).translated(1, 10, 0))     # crosses x=0 wall
    ok = base.cut(Cyl.make(3, 10).translated(6, 10, 0))
    with _pytest.raises(ValueError, match="intersect"):
        ok.cut(Cyl.make(3, 10).translated(11, 10, 0))      # touches first bore
    with _pytest.raises(ValueError, match="misses"):
        base.cut(Cyl.make(2, 5).translated(10, 10, 20))    # above the solid


def test_blind_hole_clamps_to_material() -> None:
    from forgekernel.quadric import Cyl, DrilledSolid, PiVal

    base = DrilledSolid(box(20, 20, 10), [])
    # drilled from above, tool extends past the top: clamped to material
    out = base.cut(Cyl.make(2, 8).translated(10, 10, 6))
    assert out.volume() == PiVal(4000, -16)                # π·4·(10-6)


# -- K2.1: coaxial quadric stacks + revolve, exact in Q[pi] --------------------

def test_axis_stack_boss_exact() -> None:
    from forgekernel.quadric import AxisStack, Cone, Cyl, PiVal, Sphere

    s = AxisStack(0, 0, [Cyl.make(20, 8)])
    s = s.fuse(Cone.make(10, 6, 12).translated(0, 0, 8))
    s = s.fuse(Sphere.make(5).translated(0, 0, 22))
    # cylinder 3200pi + frustum 784pi + spherical cap (sphere dominated by
    # the cone on [17,20], proven by exact interval analysis) 392pi/3
    assert s.volume() == PiVal(0, Fraction(12344, 3))


def test_revolve_green_theorem_exact() -> None:
    from forgekernel.quadric import PiVal, RevolveSolid

    loop = [(5, 0), (15, 0), (15, 4), (8, 8), (8, 20), (5, 20)]
    assert RevolveSolid(loop).volume() == PiVal(0, Fraction(5140, 3))
    # washer: rectangle (2..4) x (0..5) -> pi (16-4) * 5
    washer = RevolveSolid([(2, 0), (4, 0), (4, 5), (2, 5)])
    assert washer.volume() == PiVal(0, 60)


def test_sphere_and_cone_alone_exact() -> None:
    from forgekernel.quadric import AxisStack, Cone, PiVal, Sphere

    assert AxisStack(0, 0, [Sphere.make(3)]).volume() == PiVal(0, 36)
    # full cone r=3 h=6: pi r^2 h / 3 = 18 pi
    assert AxisStack(0, 0, [Cone.make(3, 0, 6)]).volume() == PiVal(0, 18)


def test_irrational_crossover_refuses_honestly() -> None:
    import pytest as _pytest

    from forgekernel.quadric import AxisStack, Cyl, Sphere

    # cyl r=2 vs sphere r=3 overlapping: 4 = 9 - z^2 -> z = sqrt(5),
    # an irrational crossover strictly inside the overlap
    s = AxisStack(0, 0, [Cyl.make(2, 6).translated(0, 0, -3)])
    s = s.fuse(Sphere.make(3))
    with _pytest.raises(ValueError, match="K2.2"):
        s.volume()


def test_coaxial_requirement_refuses() -> None:
    import pytest as _pytest

    from forgekernel.quadric import AxisStack, Cyl

    s = AxisStack(0, 0, [Cyl.make(5, 5)])
    with _pytest.raises(ValueError, match="non-coaxial"):
        s.fuse(Cyl.make(5, 5).translated(20, 0, 0))


# -- W-A: tangent-contact unions (measure-zero, exact) ------------------------

def test_tangent_cylinders_sum_exactly() -> None:
    from forgekernel.quadric import Cyl, DisjointUnion, PiVal

    # r=10 each, centers 20 apart: d^2 = 400 = (r1+r2)^2 -> tangent
    u = DisjointUnion([Cyl.make(10, 10), Cyl.make(10, 10).translated(20, 0, 0)])
    assert u.volume() == PiVal(0, 2000)         # 2 * 100 * 10, no overlap term


def test_sphere_tangent_to_box_face_exact() -> None:
    from forgekernel.brep import Solid
    from forgekernel.quadric import DisjointUnion, PiVal, Sphere

    # box top at z=10, sphere r=5 centered z=15: gap 15-10 == 5 == r, tangent
    u = DisjointUnion([Solid.box(30, 30, 10), Sphere.make(5).translated(15, 15, 15)])
    assert u.volume() == PiVal(9000, Fraction(500, 3))
    v = u.volume()
    assert abs(float(v) - (9000 + 500 / 3 * math.pi)) < 1e-9


def test_genuine_overlap_refuses() -> None:
    import pytest as _pytest

    from forgekernel.brep import Solid
    from forgekernel.quadric import Cyl, DisjointUnion, Sphere

    with _pytest.raises(ValueError, match="K2.3"):
        DisjointUnion([Cyl.make(10, 10), Cyl.make(10, 10).translated(15, 0, 0)])
    with _pytest.raises(ValueError, match="K2.3"):
        DisjointUnion([Solid.box(30, 30, 10),
                       Sphere.make(5).translated(15, 15, 8)])   # center inside


def test_disjoint_in_z_passes_despite_close_axes() -> None:
    from forgekernel.quadric import Cyl, DisjointUnion, PiVal

    # axes 2 apart (would overlap) but z-ranges don't touch -> disjoint, exact
    u = DisjointUnion([Cyl.make(10, 5), Cyl.make(10, 5).translated(2, 0, 10)])
    assert u.volume() == PiVal(0, 1000)


def test_internally_tangent_cylinders_exact() -> None:
    from forgekernel.quadric import Cyl, DisjointUnion, PiVal

    # r=10 and r=4, centers 6 apart: d^2 = 36 = (10-4)^2 internally tangent
    u = DisjointUnion([Cyl.make(10, 8), Cyl.make(4, 8).translated(6, 0, 0)])
    assert u.volume() == PiVal(0, (100 + 16) * 8)


# -- W-B: draft (frustum via exact prismatoid) --------------------------------

def test_draft_frustum_matches_exact_integral() -> None:
    import math as _m

    from forgekernel.exact import F
    from forgekernel.kernel import box, draft

    d = draft(box(30, 30, 15), 3.0)
    tf = F(_m.tan(_m.radians(3.0)))
    Vp = lambda z: -(30 - 2 * tf * z) ** 3 / (6 * tf)   # noqa: E731
    assert d.volume() == Vp(F(15)) - Vp(F(0))
    assert d.watertight_violations() == []


def test_prismatoid_exact_volume() -> None:
    from forgekernel.brep import prismatoid
    from forgekernel.exact import F

    # frustum: 10x10 base at z=0, 4x4 top at z=6; prismatoid formula
    # V = h/6 (A0 + 4Am + A1), Am = 7x7 = 49 -> 6/6(100+196+16)=312
    p = prismatoid([(0, 0), (10, 0), (10, 10), (0, 10)], 0,
                   [(3, 3), (7, 3), (7, 7), (3, 7)], 6)
    assert p.volume() == F(312)
    assert p.watertight_violations() == []


def test_draft_nonrect_refuses() -> None:
    import pytest as _pytest

    from forgekernel.brep import Solid
    from forgekernel.kernel import draft

    tri = Solid.prism([(0, 0), (10, 0), (5, 8)], 6)
    with _pytest.raises(ValueError, match="K2.3"):
        draft(tri, 3.0)


# -- W-C: shell (hollow box, exact) -------------------------------------------

def test_shell_box_exact() -> None:
    from forgekernel.kernel import box, shell

    s = shell(box(40, 30, 20), 2)
    assert s.volume() == 40 * 30 * 20 - 36 * 26 * 16   # == 9024
    assert s.watertight_violations() == []


def test_shell_too_thick_refuses() -> None:
    import pytest as _pytest

    from forgekernel.kernel import box, shell

    with _pytest.raises(ValueError, match="exceeds"):
        shell(box(10, 10, 10), 5)                       # 2t == smallest dim


# -- W-D: fillet (rounded box, Steiner formula, exact Q[pi]) ------------------

def test_rounded_box_steiner_volume_exact() -> None:
    from forgekernel.quadric import PiVal, RoundedBox

    # box 30x20x10 r=5/2: V = pqs + 2r(pq+qs+sp) + pi r^2(p+q+s) + 4/3 pi r^3
    # p,q,s = 25,15,5 -> 1875 + 2875 + (1125/4 + 125/6)pi = 4750 + 3625/12 pi
    rb = RoundedBox(30, 20, 10, Fraction(5, 2))
    assert rb.volume() == PiVal(4750, Fraction(3625, 12))
    # this exact rational is OCCT's float to the last bit (locked in bench)
    assert abs(float(rb.volume()) - 5699.022780771917) < 1e-9


def test_fillet_too_large_refuses() -> None:
    import pytest as _pytest

    from forgekernel.quadric import RoundedBox

    with _pytest.raises(ValueError, match="exceeds"):
        RoundedBox(10, 10, 10, 6)                       # 2r=12 > smallest dim
    # 2r == dim is the valid degenerate: a fully-rounded cube is a sphere
    from forgekernel.quadric import PiVal
    assert RoundedBox(10, 10, 10, 5).volume() == PiVal(0, Fraction(500, 3))


def test_loft_square_to_square_prismatoid() -> None:
    from forgekernel.brep import prismatoid
    from forgekernel.exact import F

    # corpus loft: 20x20 at z=0 -> 8x8 at z=25; prismatoid 25/6(400+4*196+64)
    p = prismatoid([(-10, -10), (10, -10), (10, 10), (-10, 10)], 0,
                   [(-4, -4), (4, -4), (4, 4), (-4, 4)], 25)
    assert p.volume() == F(5200)
    assert p.watertight_violations() == []
