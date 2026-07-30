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
    box = translate(Solid.box(20, 20, 20), 0, 0, 0)
    for r in (1, 2):
        v = sampled_fillet(box, r).volume()
        exact = float(RoundedBox(20, 20, 20, r).volume())
        assert v.lo <= exact <= v.hi, (r, float(v.mid), exact, float(v.width))
