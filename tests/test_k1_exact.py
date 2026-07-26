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
    # Hand-derived: edge cuts give 1000 - 480/wedges + 64/pairs - 16/triples
    # = 808. THAT DERIVATION WAS ALWAYS RIGHT. This test used to subtract a
    # further 8 x d^3/12 for "industrial corner facets" so the number would
    # match an OCCT float reading, and that subtraction made the kernel remove
    # material no edge chamfered away. See gitcad's
    # tests/invariants/test_chamfer_is_local_to_its_edges.py, where a point
    # clearing all twelve edge half-spaces is shown to have been cut anyway.
    # The three chamfer planes at a corner already meet at the single point
    # (d/2, d/2, d/2), so there is no corner left to truncate.
    assert c.volume() == 808
    # 6 octagons + 12 chamfer faces = 18 planes. NOT 26: the 8 corner
    # triangles are gone, because they were never geometrically there.
    planes = {key[0] for key in c.logical_faces()}
    assert len(planes) == 18
    assert chamfer(box(10, 10, 10), 2).volume() == c.volume()


def test_chamfer_block_matches_oracle_exactly() -> None:
    from forgekernel.kernel import chamfer

    # the first real ref-vs-OCCT disagreement, resolved: OCCT reports
    # 5562.666666666667 for box(30,20,10) chamfer 2 — ref returns the
    # exact rational behind that float.
    # The closed form, derived here and independently confirmed by exact
    # rational vertex-enumeration of the twelve edge half-spaces:
    #     V = abc - 2(a+b+c)d^2 + 6d^3
    # For 30x20x10 at d=2:  6000 - 480 + 48 = 5568.
    #
    # This assertion used to read Fraction(16688, 3) = 5562.666..., matched to
    # an OCCT reading. The gap is exactly 2d^3/3 - the eight corner tetrahedra
    # the kernel was removing on top of the chamfers it was asked for. OCCT is
    # no longer in this project (ADR-0020) so it cannot be re-measured; the
    # justification therefore rests on the DEFINITION of a chamfer (the union
    # of its per-edge wedges), which is the sounder basis regardless.
    assert chamfer(box(30, 20, 10), 2).volume() == 5568
    for a, b, c in ((30, 20, 10), (20, 20, 20), (40, 30, 20)):
        for d in (1, 2, 3):
            assert chamfer(box(a, b, c), d).volume() == (
                a * b * c - 2 * (a + b + c) * d ** 2 + 6 * d ** 3)


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


def test_internally_tangent_cylinders_are_an_overlap_not_a_union() -> None:
    """INTERNAL tangency of two SOLID cylinders is containment, not disjointness.

    r=10 at the origin and r=4 at (6,0): d² = 36 = (10−4)², so they touch at
    (10,0) — but the small cylinder lies wholly INSIDE the big one (its farthest
    point is 6+4 = 10 = the big radius) and their z-ranges coincide. The union is
    therefore just the big cylinder, 800π — NOT 928π. An earlier version of this
    test asserted 928π, and that is what let ``_classify_pair`` admit nested
    cylinders into a DisjointUnion and silently double-count the overlap. Only
    EXTERNAL separation (d² ≥ (ra+rb)²) certifies disjointness.
    """
    import pytest

    from forgekernel.quadric import Cyl, DisjointUnion, PiVal

    with pytest.raises(ValueError, match="overlapping cylinders"):
        DisjointUnion([Cyl.make(10, 8), Cyl.make(4, 8).translated(6, 0, 0)])

    # externally tangent (d² = (ra+rb)²) IS measure-zero contact: exact sum
    u = DisjointUnion([Cyl.make(10, 8), Cyl.make(4, 8).translated(14, 0, 0)])
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


# -- W-E: mitered sweep in Q[sqrt d] — the model OCCT cannot build -------------

def test_surd_field_arithmetic() -> None:
    from forgekernel.surd import SurdVal, sqrt_rational

    assert sqrt_rational(450) == SurdVal(0, 15, 2)      # 15√2
    assert sqrt_rational(16) == SurdVal(4, 0, 1)        # exactly 4
    assert (SurdVal(1, 1, 2) * SurdVal(1, 1, 2)) == SurdVal(3, 2, 2)   # (1+√2)^2
    assert sqrt_rational(Fraction(9, 4)) == SurdVal(Fraction(3, 2), 0, 1)


def test_mixed_radicals_refuse() -> None:
    import pytest as _pytest

    from forgekernel.surd import sqrt_rational

    with _pytest.raises(ValueError, match="bigger field"):
        sqrt_rational(2) + sqrt_rational(3)


def test_mitered_sweep_exact_in_root2() -> None:
    from forgekernel.kernel import sweep
    from forgekernel.surd import SurdVal

    # corpus swept_channel: 4x4 profile, path with a 45-degree corner
    ms = sweep(16, [[0, 0, 0], [0, 0, 20], [15, 0, 35], [40, 0, 35]])
    assert ms.length() == SurdVal(45, 15, 2)            # 20 + 15√2 + 25
    assert ms.volume() == SurdVal(720, 240, 2)          # 16 × length, EXACT


def test_mitered_sweep_bbox_is_the_prism_box_not_a_sqrt_area_pad() -> None:
    """W8: bbox used to pad the centreline by sqrt(area) — for any profile
    wider than 4:1 that UNDERSTATES, so a 20x1 plate swept straight up got a
    half-width-4.47 box that cut 5.5 mm inside real material on both x sides
    (and padded 4.47 of air on y and z). A rectangle swept along a straight
    line is a prism; its box is the profile's extents times the length."""
    import pytest as _pytest

    from forgekernel.kernel import sweep

    prof = [(-10, Fraction(-1, 2)), (10, Fraction(-1, 2)),
            (10, Fraction(1, 2)), (-10, Fraction(1, 2))]
    ms = sweep(20, [[0, 0, 0], [0, 0, 30]], profile=prof)
    assert ms.volume() == 600                        # 20*1*30, unchanged
    lo, hi = ms.bbox()
    assert lo == _pytest.approx((-10.0, -0.5, 0.0), rel=0, abs=1e-12)
    assert hi == _pytest.approx((10.0, 0.5, 30.0), rel=0, abs=1e-12)


def test_mitered_sweep_bbox_around_a_right_angle_corner() -> None:
    """The L: a 10x10 square swept z-up then x-along. Leg 1 is the prism
    |x|,|y| <= 5 clipped by the 45-degree miter plane x + z = 10, leg 2 its
    mirror image — so the union's exact box is (-5,-5,0)..(10,5,15), the
    outer miter corner reaching z = 15 at x = -5."""
    import pytest as _pytest

    from forgekernel.kernel import sweep

    prof = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    ms = sweep(100, [[0, 0, 0], [0, 0, 10], [10, 0, 10]], profile=prof)
    lo, hi = ms.bbox()
    assert lo == _pytest.approx((-5.0, -5.0, 0.0), rel=0, abs=1e-12)
    assert hi == _pytest.approx((10.0, 5.0, 15.0), rel=0, abs=1e-12)


def test_mitered_sweep_centroid_is_the_solid_not_the_wire() -> None:
    """W11: centroid_f returned length-weighted midpoints of the CENTRELINE
    segments — the wire's centroid, not the solid's. For the 10x10 L-sweep
    the closed form is (1.875, 0, 8.125): leg 1 = {|x|,|y|<=5, 0<=z,
    x+z<=10} has m_x = -2500/3, leg 2 (its miter mirror) m_x = 5000-1250/3,
    V = 2000, so cx = 3750/2000; cz = 10 - cx by the miter symmetry. The
    wire answer was (2.5, 0, 7.5) — biased a**2/(2(L1+L2)) per corner."""
    import pytest as _pytest

    from forgekernel.kernel import sweep

    prof = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    ms = sweep(100, [[0, 0, 0], [0, 0, 10], [10, 0, 10]], profile=prof)
    assert ms.centroid_f() == _pytest.approx((1.875, 0.0, 8.125),
                                             rel=0, abs=1e-12)


def test_mitered_sweep_straight_centroid_rides_the_profile_centroid() -> None:
    """A straight sweep of an OFF-CENTRE profile: the solid's centroid sits
    over the profile's own centroid, not the path (a prism, no oracle
    needed). Also pins the frame convention: +z first leg maps profile
    (u, v) to world (x, y)."""
    import pytest as _pytest

    from forgekernel.kernel import sweep

    prof = [(0, 0), (6, 0), (6, 2), (0, 2)]          # centroid (3, 1)
    ms = sweep(12, [[0, 0, 0], [0, 0, 10]], profile=prof)
    assert ms.centroid_f() == _pytest.approx((3.0, 1.0, 5.0),
                                             rel=0, abs=1e-12)
    lo, hi = ms.bbox()
    assert lo == _pytest.approx((0.0, 0.0, 0.0), rel=0, abs=1e-12)
    assert hi == _pytest.approx((6.0, 2.0, 10.0), rel=0, abs=1e-12)


def test_mitered_sweep_metrics_without_profile_refuse() -> None:
    """Built from an area alone the solid knows its exact volume but NOT its
    shape — bbox/centroid must refuse rather than guess (the sqrt-area pad
    was a silent wrong number through the seam)."""
    import pytest as _pytest

    from forgekernel.kernel import sweep

    ms = sweep(16, [[0, 0, 0], [0, 0, 20]])
    with _pytest.raises(ValueError, match="profile"):
        ms.bbox()
    with _pytest.raises(ValueError, match="profile"):
        ms.centroid_f()


def test_mitered_sweep_reversing_path_refuses_metrics() -> None:
    """A path that doubles straight back has no miter plane (t1 + t2 = 0) —
    the sweep model is undefined there, so metrics refuse by an EXACT
    rational test on the path deltas, not a float epsilon."""
    import pytest as _pytest

    from forgekernel.kernel import sweep

    prof = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ms = sweep(4, [[0, 0, 0], [0, 0, 10], [0, 0, 3]], profile=prof)
    with _pytest.raises(ValueError, match="reverses"):
        ms.bbox()
    with _pytest.raises(ValueError, match="reverses"):
        ms.centroid_f()


# -- W-K: composite tessellation (bounded-error view) -------------------------

def test_revolve_mesh_approximates_exact_volume() -> None:
    from forgekernel.quadric import RevolveSolid
    from forgekernel.tess import mesh_volume

    rs = RevolveSolid([(2, 0), (4, 0), (4, 5), (2, 5)])   # washer, 60pi
    exact = float(rs.volume())
    for defl in (0.5, 0.1, 0.01):
        v = mesh_volume(rs.tessellate(defl))
        assert v <= exact                                 # inscribed
        assert (exact - v) / exact < 0.5 * defl + 0.05    # converges


def test_axisstack_mesh_converges() -> None:
    from forgekernel.quadric import AxisStack, Cone, Cyl
    from forgekernel.tess import mesh_volume

    s = AxisStack(0, 0, [Cyl.make(10, 8)]).fuse(Cone.make(10, 6, 12).translated(0, 0, 8))
    exact = float(s.volume())
    coarse = mesh_volume(s.tessellate(0.5))
    fine = mesh_volume(s.tessellate(0.02))
    assert abs(fine - exact) < abs(coarse - exact)        # finer is closer
    assert abs(fine - exact) / exact < 0.02


# -- K2.2: non-coaxial quadric booleans that stay exact in Q[pi] --------------

def test_sphere_overlap_booleans_exact() -> None:
    from forgekernel.quadric import PiVal, Sphere, SphereOverlap

    a = Sphere.make(5)
    b = Sphere.make(5).translated(6, 0, 0)   # d=6 < 10 overlap, > 0 not nested
    # d1=3, h1=h2=2, cap=52/3 each, lens=104/3
    assert SphereOverlap(a, b, "intersect").volume() == PiVal(0, Fraction(104, 3))
    assert SphereOverlap(a, b, "union").volume() == PiVal(0, Fraction(896, 3))
    assert SphereOverlap(a, b, "cut").volume() == PiVal(0, 132)   # 500/3 - 104/3


def test_sphere_overlap_centroid_is_the_cap_moment_not_the_midpoint() -> None:
    """W7: centroid_f returned the midpoint of the two centres for EVERY op —
    right only for the two symmetric cases (union/intersect of equal
    spheres), which is why nothing caught it. For r=5 spheres 6 apart the
    lens spans x in [1,5]; direct integration gives the lens x-moment 104π
    and lens volume 104π/3, so the cut centroid is (0 - 104π)/(500π/3 -
    104π/3) = -26/33 — NEGATIVE, 3.79 mm from the reported +3.0 and on the
    other side of the origin. Every ratio is rational (π cancels), so the
    floats below are the correctly-rounded exact values."""
    from forgekernel.quadric import Sphere, SphereOverlap

    a = Sphere.make(5)
    b = Sphere.make(5).translated(6, 0, 0)
    assert SphereOverlap(a, b, "cut").centroid_f() == (
        float(Fraction(-26, 33)), 0.0, 0.0)
    # the two symmetric cases the old code got right must stay right
    assert SphereOverlap(a, b, "intersect").centroid_f() == (3.0, 0.0, 0.0)
    assert SphereOverlap(a, b, "union").centroid_f() == (3.0, 0.0, 0.0)


def test_sphere_overlap_centroid_unequal_radii() -> None:
    """Unequal spheres r1=5, r2=3, d=6: d1 = 13/3, h1 = 2/3, h2 = 4/3, lens
    volume 20π/3, lens moment about A 736π/27 (cap moment about its own
    sphere centre is π h²(2r−h)²/4). Closed forms:
        cut       -23/135,  intersect  184/45,  union  26/27
    (cross-checked by 4M-sample Monte Carlo this session)."""
    from forgekernel.quadric import Sphere, SphereOverlap

    a = Sphere.make(5)
    b = Sphere.make(3).translated(6, 0, 0)
    assert SphereOverlap(a, b, "cut").centroid_f() == (
        float(Fraction(-23, 135)), 0.0, 0.0)
    assert SphereOverlap(a, b, "intersect").centroid_f() == (
        float(Fraction(184, 45)), 0.0, 0.0)
    assert SphereOverlap(a, b, "union").centroid_f() == (
        float(Fraction(26, 27)), 0.0, 0.0)


def test_sphere_overlap_bbox_stops_at_plane_and_waist() -> None:
    """W12: the cut bbox was sphere A's whole box (+25% on the cut axis) and
    the intersect box ignored the lens waist. The cut solid ends at the
    radical plane x = d1 = (d²+r1²−r2²)/(2d) = 3; the lens's transverse
    extent is the waist radius √(r1²−d1²) = 4. Both were already computed
    for the volume — the box just never asked."""
    from forgekernel.quadric import Sphere, SphereOverlap

    a = Sphere.make(5)
    b = Sphere.make(5).translated(6, 0, 0)
    assert SphereOverlap(a, b, "cut").bbox() == ((-5.0, -5.0, -5.0),
                                                 (3.0, 5.0, 5.0))
    assert SphereOverlap(a, b, "intersect").bbox() == ((1.0, -4.0, -4.0),
                                                       (5.0, 4.0, 4.0))
    # union keeps the exact two-sphere box it always had
    assert SphereOverlap(a, b, "union").bbox() == ((-5.0, -5.0, -5.0),
                                                   (11.0, 5.0, 5.0))


def test_sphere_overlap_bbox_off_axis_pair() -> None:
    """Centres 7 apart along the rational direction (2,3,6)/7 — the support
    construction must hold per WORLD axis, not just when the pair rides x.
    Equal r=5: d1 = 7/2, waist² = 51/4. Per axis e: the far pole of A
    survives the cut iff r1·(e·û) <= d1, else the extreme sits on the waist
    circle at C·e + w·√(1−(e·û)²). Verified by 2M-sample rejection sampling
    this session: observed extremes approach every bound from INSIDE
    (pole axes to <5e-3; the waist axis to 0.023, slow because the true
    extreme sits on a curve of measure zero) and never exceed it."""
    import pytest as _pytest

    from forgekernel.quadric import Sphere, SphereOverlap

    a = Sphere.make(5)
    b = Sphere.make(5).translated(2, 3, 6)
    lo, hi = SphereOverlap(a, b, "cut").bbox()
    w = math.sqrt(51 / 4)
    assert lo == _pytest.approx((-5.0, -5.0, -5.0), rel=0, abs=1e-12)
    # x: r1·(2/7) = 10/7 < 7/2 -> pole survives; y: 15/7 < 7/2 -> pole;
    # z: 30/7 > 7/2 -> waist: 3 + w·√(1-36/49)
    assert hi == _pytest.approx(
        (5.0, 5.0, 3.0 + w * math.sqrt(13.0 / 49.0)), rel=0, abs=1e-12)


def test_sphere_overlap_refuses_nonoverlap_and_irrational() -> None:
    import pytest as _pytest

    from forgekernel.quadric import Sphere, SphereOverlap

    with _pytest.raises(ValueError, match="do not overlap"):
        SphereOverlap(Sphere.make(2), Sphere.make(2).translated(10, 0, 0), "union")
    with _pytest.raises(ValueError, match="contains"):
        SphereOverlap(Sphere.make(5), Sphere.make(1).translated(1, 0, 0), "union")
    with _pytest.raises(ValueError, match="irrational"):
        SphereOverlap(Sphere.make(5), Sphere.make(5).translated(1, 1, 0), "union")  # d=√2


def test_steinmetz_is_exact_and_pi_free() -> None:
    from forgekernel.quadric import PiVal, steinmetz

    assert steinmetz(3) == PiVal(144, 0)      # 16*27/3, no pi at all
    assert steinmetz(Fraction(1, 2)) == PiVal(Fraction(2, 3), 0)
