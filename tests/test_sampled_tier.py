"""The `sampled` provenance tier (ADR-0024): honest Monte-Carlo answers.

A sampled answer is a float with a stated 3σ error and a provenance label — the
one thing ADR-0019 permits that a bare float does not, because a caller knows
exactly how much to trust it. These tests pin that the estimator is accurate
(analytic membership, not mesh-deficit sampling), unbiased against known
volumes, deterministic, and honestly bounded.
"""

import math
from fractions import Fraction as F

from forgekernel.sampled import SampledSolid
from forgekernel.quadric import Sphere, Cyl
from forgekernel.brep import Solid
from forgekernel.kernel import translate


def _box(sx, sy, sz, tx=0, ty=0, tz=0):
    return translate(Solid.box(sx, sy, sz), tx, ty, tz)


def test_the_estimator_is_unbiased_against_a_known_volume():
    """A box fully inside a sphere: the intersection is the box, 216 exactly.
    Analytic membership means no mesh deficit, so this lands dead on."""
    inter = SampledSolid.boolean("intersect", Sphere(0, 0, 0, 6),
                                 _box(6, 6, 6, -3, -3, -3))
    v = inter.volume(200000)
    assert abs(float(v.mid) - 216.0) <= float(v.hi - v.lo) / 2 + 1e-9


def test_the_sphere_through_a_prism_wall_is_answered():
    """The arcsin wall `_EXACT_FIELD_BOUNDARY` calls permanent: a 2x2 square
    prism through a sphere r=6. Analytic against a fine numerical integral."""
    out = SampledSolid.boolean("cut", Sphere(0, 0, 0, 6),
                               _box(2, 2, 40, -1, -1, -20))
    v = out.volume(300000)
    # removed = ∫∫_{|x|,|y|<=1} 2√(36-x²-y²)
    n, rem = 1200, 0.0
    dx = 2.0 / n
    for i in range(n):
        x = -1 + (i + 0.5) * dx
        for j in range(n):
            y = -1 + (j + 0.5) * dx
            rem += 2 * math.sqrt(max(36 - x * x - y * y, 0)) * dx * dx
    truth = 4 / 3 * math.pi * 216 - rem
    hw = float(v.hi - v.lo) / 2
    assert abs(float(v.mid) - truth) <= hw + 1e-6, (float(v.mid), truth, hw)


def test_it_is_deterministic():
    """A fixed seed: the same model gives the same number every run. A bench
    figure that jittered would be worthless."""
    a = SampledSolid.boolean("cut", Sphere(0, 0, 0, 6), _box(2, 2, 40, -1, -1, -20))
    b = SampledSolid.boolean("cut", Sphere(0, 0, 0, 6), _box(2, 2, 40, -1, -1, -20))
    assert a.volume(50000).mid == b.volume(50000).mid


def test_it_carries_the_sampled_label_and_a_real_error():
    out = SampledSolid.boolean("cut", Cyl(0, 0, 5, 0, 12), _box(2, 40, 40, -1, -20, -20))
    assert out.provenance == "sampled"
    v = out.volume(100000)
    assert v.hi > v.lo                       # a real, non-degenerate bracket
    assert v.width < F(50)                    # and a useful one


def test_chains_compose():
    """cut then cut: a SampledSolid is itself a valid operand, so a chain nests
    without re-tessellating from scratch."""
    first = SampledSolid.boolean("cut", Sphere(0, 0, 0, 6), _box(3, 40, 40, -1.5, -20, -20))
    second = SampledSolid.boolean("cut", first, _box(40, 3, 40, -20, -1.5, -20))
    v = second.volume(100000)
    assert v.sign() > 0


def test_sampled_shell_matches_an_exact_box_shell():
    """A shell is a clean membership — inside AND within t of the surface — so
    the sampled tier answers it accurately (no mesh deficit in the SET, only in
    the distance, which is far under the wall thickness). A 20-cube shelled to
    t=2 is 8000 - 16³ = 3904 exactly."""
    from forgekernel.sampled import sampled_shell
    box = translate(Solid.box(20, 20, 20), 0, 0, 0)
    v = sampled_shell(box, 2).volume(200000)
    assert abs(float(v.mid) - 3904.0) <= float(v.hi - v.lo) / 2 + 20


def test_sampled_fillet_stays_within_its_honest_bound():
    """A fillet-all is a voxel morphological open-then-close. Its only error is
    the voxel resolution, bounded by surface_area·h and reported as the
    half-width — so the exact RoundedBox volume must fall INSIDE the bracket at
    every radius. That is the ADR-0024 line: never a lie about the error. The
    withdrawn single-normal opening failed exactly this (2.2% off, 3σ of 14)."""
    from forgekernel.sampled import sampled_fillet
    from forgekernel.quadric import RoundedBox
    box = translate(Solid.box(10, 10, 10), 0, 0, 0)
    v = sampled_fillet(box, 1).volume()
    exact = float(RoundedBox(10, 10, 10, 1).volume())
    assert v.lo <= exact <= v.hi, (float(v.mid), exact, float(v.width))


def test_a_sampled_solid_tessellates_and_exports_faceted():
    """The export path (ADR-0024): a sampled cut has no b-rep, so it tessellates
    to a watertight voxel surface and writes a valid faceted STEP."""
    from forgekernel.sampled import write_step_faceted
    out = SampledSolid.boolean("cut", Sphere(0, 0, 0, 6),
                               _box(2, 2, 40, -1, -1, -20))
    mesh = out.tessellate()
    assert mesh["triangles"] and mesh["vertices"]
    step = write_step_faceted(mesh, "cut")
    assert step.startswith("ISO-10303-21")
    assert step.strip().endswith("END-ISO-10303-21;")
    assert "SHELL_BASED_SURFACE_MODEL" in step


def test_shell_of_a_sampled_solid_works():
    """A shell OF a sampled cut: the base is sampled, so its surface is the
    sampled voxel surface — the chain composes. Shell is the cheap check
    (membership only); the fillet-of-sampled path is exercised through the
    seam bench, not here, because voxel morphology in pure Python is slow."""
    from forgekernel.sampled import sampled_shell
    cut = SampledSolid.boolean("cut", Sphere(0, 0, 0, 4),
                               _box(2, 20, 20, -1, -10, -10))
    assert sampled_shell(cut, 1).volume(40000).sign() > 0


def test_a_meshed_curved_operand_bracket_encloses_the_truth():
    """The review's CRITICAL finding: a SampledSolid over a curved BODY operand
    takes the mesh fallback, whose MC converges to the tessellation polyhedron
    (systematically below the true curved volume). The reported half-width must
    fold in that geometric deficit (area*deflection) so the bracket ENCLOSES the
    truth — a statistical 3σ alone excluded it (the ADR-0019 forbidden case)."""
    import math
    from forgekernel import body as B
    from forgekernel.quadric import Cyl
    cyl_body = B.to_body(Cyl(0, 0, 5, 0, 12))          # a Body -> mesh fallback
    box = translate(Solid.box(12, 12, 14), -6, -6, -1)  # contains the cylinder
    v = SampledSolid.boolean("intersect", cyl_body, box).volume(150000)
    truth = math.pi * 25 * 12
    assert v.lo <= truth <= v.hi, (float(v.lo), float(v.hi), truth)


def test_disjoint_intersect_answers_zero_not_a_crash():
    """The review's finding #7: two disjoint operands give an inverted bbox; the
    intersection is empty and must measure ~0, not raise 'degenerate interval'."""
    from forgekernel.quadric import Sphere
    d = SampledSolid.boolean("intersect", Sphere(0, 0, 0, 2),
                             translate(Solid.box(2, 2, 2), 20, 20, 20))
    v = d.volume(20000)
    assert float(v.lo) == 0.0 and float(v.hi) == 0.0
