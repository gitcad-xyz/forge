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
