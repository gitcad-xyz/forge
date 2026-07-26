"""A revolved profile whose VERTICES are surds, as a canonical B-rep body.

``surdrev.contour_r2_dz`` proved the Green term over ℚ[√d]. This is the next
thing along: the same profile turned into faces, and the divergence sum run
over those faces. It is a different computation with a different failure mode —
``contour_r2_dz`` integrates the profile directly, while ``body.volume``
integrates ``(1/3)∮x·n̂ dA`` per SURFACE, and a cone band's term is built from
the apex position and the half-angle rather than from the rim radii. The two
must agree, and neither is allowed to leave the exact field.

This is the arithmetic that shelling a cone stands on: the offset of a slanted
profile edge lands at an irrational radius, so the void's annular caps have
surd radii and the void's conical band has a surd apex.
"""

import math
from fractions import Fraction as F

import pytest

from forgekernel import body as B
from forgekernel.polypi import PiPoly
from forgekernel.quadric import PiVal, RevolveSolid
from forgekernel.surd import SurdVal, exact_sqrt
from forgekernel.surdrev import contour_r2_dz


def _pi_coeff(v):
    if isinstance(v, PiPoly):
        assert v.degree <= 1 and v[0] == 0, repr(v)
        return v[1]
    assert v.a == 0, repr(v)
    return v.b


# the void of a shelled cone(r1=6, r2=2, h=10) at t=1
SURD_FRUSTUM = [(F(0), F(1)),
                (SurdVal(F(28, 5), F(-1, 5), 29), F(1)),
                (SurdVal(F(12, 5), F(-1, 5), 29), F(9)),
                (F(0), F(9))]


def test_exact_sqrt_stays_in_the_smallest_field_that_holds_it() -> None:
    """A perfect square must come back as a ℚ value, not a ℚ[√1] wrapper —
    otherwise every rectilinear profile in the kernel silently changes TYPE
    the moment the offset code is shared with the slanted one."""
    assert exact_sqrt(16) == F(4) and isinstance(exact_sqrt(16), F)
    assert exact_sqrt(F(9, 4)) == F(3, 2) and isinstance(exact_sqrt(F(9, 4)), F)
    assert exact_sqrt(116) == SurdVal(0, 2, 29)
    assert isinstance(exact_sqrt(116), SurdVal)
    assert float(exact_sqrt(116)) == pytest.approx(math.sqrt(116))
    with pytest.raises(ValueError):
        exact_sqrt(-1)


def test_a_surd_frustums_body_volume_matches_its_green_term() -> None:
    rev = RevolveSolid(SURD_FRUSTUM, 0, 0)
    edges = [("line", SURD_FRUSTUM[i], SURD_FRUSTUM[(i + 1) % 4])
             for i in range(4)]
    green = contour_r2_dz(edges)
    assert _pi_coeff(B.volume(B.to_body(rev))) == green
    # and against the frustum formula, computed from the surd radii directly
    a, b = SURD_FRUSTUM[1][0], SURD_FRUSTUM[2][0]
    assert green == 8 * (a * a + a * b + b * b) / 3


def test_the_surd_frustums_volume_never_becomes_a_float() -> None:
    v = B.volume(B.to_body(RevolveSolid(SURD_FRUSTUM, 0, 0)))
    c = _pi_coeff(v)
    assert isinstance(c, SurdVal), f"left ℚ[√d]: {type(c).__name__}"
    assert c == SurdVal(F(10808, 75), F(-64, 5), 29)


def test_a_rational_frustum_still_lands_in_the_narrow_ring() -> None:
    """The widening must not promote every existing answer to PiPoly: a
    rational lathe still returns the legacy ``PiVal``."""
    rev = RevolveSolid([(F(0), F(0)), (F(6), F(0)), (F(2), F(10)),
                        (F(0), F(10))], 0, 0)
    v = B.volume(B.to_body(rev))
    assert isinstance(v, PiVal)
    assert v == PiVal(0, F(520, 3))


def test_the_surd_frustum_meshes_watertight() -> None:
    from collections import defaultdict

    mesh = B.tessellate(B.to_body(RevolveSolid(SURD_FRUSTUM, 0, 0)), 0.05)
    ec: dict = defaultdict(int)
    for x, y, z in mesh["triangles"]:
        for e in ((x, y), (y, z), (z, x)):
            ec[tuple(sorted(e))] += 1
    assert all(n == 2 for n in ec.values())


def test_the_surd_frustum_is_a_closed_manifold() -> None:
    assert B.manifold_violations(B.to_body(RevolveSolid(SURD_FRUSTUM, 0, 0))) == []


@pytest.mark.parametrize("loop,zhi", [
    ([(F(0), F(0)), (F(6), F(0)), (F(2), F(10)), (F(0), F(10))], 10.0),
    (SURD_FRUSTUM, 9.0),
])
def test_a_frustums_box_stops_at_its_rim_not_at_its_apex(loop, zhi) -> None:
    """``bbox`` skipped the apex only when the face had two LOOPS, but every
    frustum this kernel builds puts both rims in one loop — so cone(6,2,10)
    reported a 15 mm z-extent for a 10 mm solid. Sound (a bbox promises a
    bound), which is why no invariant caught it, and 50% loose on exactly the
    shape whose point is the taper.

    Floats are legal here: a bbox is a bound for display and AABB pre-filtering,
    never a topological decision (ADR-0019).
    """
    lo, hi = B.bbox(B.to_body(RevolveSolid(loop, 0, 0)))
    assert hi[2] == pytest.approx(zhi)
    assert lo[2] == pytest.approx(float(loop[0][1]))


def test_a_pointed_cone_still_reaches_its_apex() -> None:
    """The other half of the same rule: with one rim the apex IS on the face
    and dropping it collapsed the box along the axis."""
    lo, hi = B.bbox(B.to_body(RevolveSolid(
        [(F(0), F(0)), (F(6), F(0)), (F(0), F(10))], 0, 0)))
    assert hi[2] == pytest.approx(10.0) and lo[2] == pytest.approx(0.0)
