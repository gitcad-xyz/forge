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
