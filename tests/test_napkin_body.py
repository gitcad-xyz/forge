"""The canonical B-rep of a napkin ring — two faces, and the right volume.

`NapkinRing` already knew its own volume. This is about the CONVERTER: the
canonical `Body` that the rest of the kernel (meshing, STEP, the answer audit)
actually consumes. A converter is exactly the place a plausible wrong number is
born, because the representation's own `volume()` keeps agreeing with the closed
form no matter what the converter builds.

So nothing here is checked against `NapkinRing.volume()`. The oracle is the
napkin-ring theorem, re-derived by hand:

    V = (4/3)·π·(R² − r²)^(3/2) = (4/3)·π·d·√d   with d = R² − r²

which in ℚ[√d][π] is the surd ``0 + (4d/3)√d`` times π — written out below as a
literal `SurdVal`, never computed by the code under test.

The topology claim is just as load-bearing: TWO faces. The bore removes the
sphere's polar caps entirely, so no planar annulus survives. A converter that
adds caps produces a closed, manifold, positively-oriented body whose volume is
wrong by exactly the caps' contribution — and every structural check would pass.
"""

import math
from fractions import Fraction as Q

import pytest

from forgekernel import body as B
from forgekernel.surd import SurdVal
from forgekernel.surdrev import NapkinRing

RINGS = [(6, 1), (10, 6), (5, 3), (13, 5)]
IDS = [f"R{R}r{r}" for R, r in RINGS]


@pytest.mark.parametrize("R,r0", RINGS, ids=IDS)
def test_the_converted_body_has_two_faces_and_no_caps(R, r0) -> None:
    body = B.to_body(NapkinRing(R, r0))
    assert len(body.faces) == 2, "a napkin ring has a zone and a wall, nothing else"
    kinds = sorted(type(f.surface).__name__ for f in body.faces)
    assert kinds == ["Cylinder", "SphereS"], (
        f"got {kinds} — a Plane here means the converter invented polar caps "
        "that the bore already removed")


@pytest.mark.parametrize("R,r0", RINGS, ids=IDS)
def test_the_converted_body_is_a_closed_shell(R, r0) -> None:
    assert B.manifold_violations(B.to_body(NapkinRing(R, r0))) == []


@pytest.mark.parametrize("R,r0", RINGS, ids=IDS)
def test_the_converted_bodys_volume_is_the_closed_form_EXACTLY(R, r0) -> None:
    """Not approx. The whole point of ℚ[√d][π] is that this is an equality."""
    d = R * R - r0 * r0
    expected = SurdVal(Q(0), Q(4 * d, 3), d)        # (4d/3)·√d, by hand
    got = B.volume(B.to_body(NapkinRing(R, r0)))
    assert got.a == 0, f"a napkin ring's volume has no rational part: {got!r}"
    assert got.b == expected, f"{got.b!r} != {expected!r}"


def test_two_different_spheres_with_one_band_height_agree_EXACTLY() -> None:
    """The property no wrong converter satisfies by accident.

    (10,6) and (17,15) both have d = 64, so their volumes must be the SAME
    exact object even though one sphere is nearly twice the other. A converter
    that leaked R or r into the term on its own still lands on a positive,
    plausible number — and only this catches it.
    """
    a = B.volume(B.to_body(NapkinRing(10, 6)))
    b = B.volume(B.to_body(NapkinRing(17, 15)))
    assert a.b == b.b, f"same band d=64 gave {a!r} and {b!r}"


@pytest.mark.parametrize("R,r0", RINGS, ids=IDS)
def test_the_sign_of_the_volume_is_decidable_without_a_float(R, r0) -> None:
    """ADR-0019. The volume carries a SURD coefficient on π, and the ONE
    question every construction asks of a volume — is it positive? — must be
    answerable exactly. `PiPoly.from_pival` used to force both coefficients
    through the stdlib `Fraction`, which raised TypeError on the surd and made
    the answer audit unable to check a napkin ring at all.
    """
    v = B.volume(B.to_body(NapkinRing(R, r0)))
    assert v.sign() == 1
    assert not v <= 0
    assert v > 0


@pytest.mark.parametrize("R,r0", RINGS, ids=IDS)
def test_the_converted_body_meshes_watertight_and_refines(R, r0) -> None:
    """Floats are legal here — this is meshing (ADR-0019).

    Every directed edge of a closed, correctly-wound mesh is used as often as
    its reverse. And a finer deflection must actually produce more triangles:
    a converter whose mesher ignores the caller's deflection would satisfy the
    watertightness check while being useless to anyone printing the part.
    """
    from collections import Counter

    body = B.to_body(NapkinRing(R, r0))
    seen: Counter = Counter()
    coarse = B.tessellate(body, 0.5)
    for tri in coarse["triangles"]:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            seen[(u, v)] += 1
    assert [k for k, n in seen.items() if seen.get((k[1], k[0]), 0) != n] == []
    assert len(coarse["triangles"]) > 0
    assert len(B.tessellate(body, 0.01)["triangles"]) > len(coarse["triangles"])


@pytest.mark.parametrize("R,r0", RINGS, ids=IDS)
def test_the_bounding_box_is_sound_even_where_it_is_loose(R, r0) -> None:
    """Floats are legal here — this is a bound, not a topology decision
    (ADR-0019). `bbox` must CONTAIN the mesh. It is allowed to be loose in z
    (the untrimmed sphere reaches ±R while the ring stops at ±√d); it is not
    allowed to be tight enough to clip anything, because callers use it as an
    interference skip-filter.
    """
    body = B.to_body(NapkinRing(R, r0))
    lo, hi = B.bbox(body)
    for v in B.tessellate(body, 0.1)["vertices"]:
        for i in range(3):
            assert float(lo[i]) - 1e-9 <= v[i] <= float(hi[i]) + 1e-9


def test_a_translated_ring_converts_about_its_own_centre() -> None:
    """Volume is translation-invariant; the box moves with it. A converter that
    hard-coded the origin would still pass every test above."""
    d = 35
    ring = NapkinRing(6, 1).translated(2, -3, 7)
    got = B.volume(B.to_body(ring))
    assert got.b == SurdVal(Q(0), Q(4 * d, 3), d)
    lo, hi = B.bbox(B.to_body(ring))
    assert float(lo[0]) == pytest.approx(-4.0)
    assert float(hi[1]) == pytest.approx(3.0)
    assert float(hi[2]) == pytest.approx(7 + 6.0)


def test_the_zone_term_is_not_applied_to_a_whole_sphere() -> None:
    """Guard against the seductive over-application of Archimedes' 2πrΔz.

    `_sphere_zone` must return None for anything that is not two full coaxial
    rims lying on the sphere, so an untrimmed sphere keeps its own (4/3)πr³
    term. If the zone path ever swallowed it, a plain sphere's volume would
    silently change — checked here because it is the cheapest tripwire.
    """
    from forgekernel.quadric import Sphere

    v = B.volume(B.to_body(Sphere(Q(0), Q(0), Q(0), Q(3))))
    assert float(v) == pytest.approx((4 / 3) * math.pi * 27, rel=1e-15)
