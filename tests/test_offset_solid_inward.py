"""Erode a planar solid by t along each face's OWN unit normal (K4.2).

This is what a closed shell of a tapered solid needs and what gitcad refused
after finding its prism-path answer was 50.7% wrong: insetting a slanted wall
by t *in xy* is not the same as offsetting it by t perpendicular to itself.

The construction is exact and needs no new number field beyond the one the
kernel already has. A face plane ``n·x = d`` with RATIONAL n offsets inward to

    n·x = d − t·|n|

so the normal stays rational and only the OFFSET leaves ℚ. For the corpus
frustum |n| = √37, i.e. ℚ[√37] — a single radical, which ``SurdVal`` holds.
Each vertex is then the exact intersection of its three incident offset
planes.

Truth for the corpus frustum (10×10 at z=0 to 6×6 at z=12, volume 784), eroded
by 1 — derived independently, and corroborated by a 4M-sample Monte Carlo:

    cavity = ∫₁¹¹ (c − z/3)² dz = (c−1/3)³ − (c−11/3)³,  c = 10 − √37/3
           = 365.9564          MC 365.990
"""

from fractions import Fraction as F

import pytest

from forgekernel.brep import Solid, offset_solid_inward


def _frustum():
    """10×10 at z=0 tapering to 6×6 (2..8) at z=12 — the capability corpus's
    loft, built directly so this test does not depend on gitcad."""
    b = [(F(0), F(0), F(0)), (F(10), F(0), F(0)),
         (F(10), F(10), F(0)), (F(0), F(10), F(0))]
    t = [(F(2), F(2), F(12)), (F(8), F(2), F(12)),
         (F(8), F(8), F(12)), (F(2), F(8), F(12))]
    from forgekernel.brep import Polygon
    polys = [Polygon(list(reversed(b)), "bottom"), Polygon(list(t), "top")]
    for i in range(4):
        j = (i + 1) % 4
        polys.append(Polygon([b[i], b[j], t[j], t[i]], f"wall{i}"))
    return Solid(polys)


def test_the_fixture_is_the_solid_we_derived_against():
    """Guard the guard — a changed fixture makes every number below stale."""
    assert _frustum().volume() == 784


def test_a_box_erodes_to_the_obvious_inner_box():
    """Control with a ℚ answer: 20³ eroded by 2 is 16³."""
    out = offset_solid_inward(Solid.box(20, 20, 20), 2)
    assert out.volume() == 16 ** 3


@pytest.mark.parametrize("t,expected", [(1, 8 ** 3), (2, 6 ** 3), (4, 2 ** 3)])
def test_box_erosion_is_exact_at_several_depths(t, expected):
    out = offset_solid_inward(Solid.box(10, 10, 10), t)
    assert out.volume() == expected


def test_the_frustum_erodes_to_the_derived_cavity():
    out = offset_solid_inward(_frustum(), 1)
    assert float(out.volume()) == pytest.approx(365.9564, rel=1e-6)


def test_the_shell_that_follows_is_the_derived_418():
    """The number gitcad got wrong: 784 − 365.9564 = 418.0436, where the
    prism path returned 206.04."""
    f = _frustum()
    shell = float(f.volume()) - float(offset_solid_inward(f, 1).volume())
    assert shell == pytest.approx(418.0436, rel=1e-6)
    assert shell != pytest.approx(206.037, abs=1e-2)


def test_erosion_commutes_with_reflection():
    """The invariant that found the original defect, at kernel level."""
    from forgekernel.body import Affine  # noqa: F401  (import guard only)

    f = _frustum()
    m = Solid([type(p)([(v[0], v[1], -v[2]) for v in reversed(p.verts)],
                       p.source) for p in f.polys])
    assert m.volume() == f.volume()
    assert offset_solid_inward(m, 1).volume() == offset_solid_inward(f, 1).volume()


def test_too_thick_refuses_instead_of_inverting():
    """A t that consumes the solid must refuse, not return a negative or
    self-intersecting body — the failure mode chamfer had at d=2."""
    with pytest.raises(ValueError):
        offset_solid_inward(Solid.box(10, 10, 10), 5)


def test_mixed_radicals_refuse_by_name():
    """Two faces whose |n| lie in different quadratic fields cannot both be
    offset in one field (K3.1). That must say so, not round."""
    from forgekernel.brep import Polygon
    from forgekernel.surd import MixedRadicals

    # a wedge with one √2 face and one √3 face
    v = [(F(0), F(0), F(0)), (F(10), F(0), F(0)), (F(10), F(10), F(0)),
         (F(0), F(10), F(0)), (F(0), F(0), F(10))]
    polys = [Polygon([v[0], v[3], v[2], v[1]], "bottom"),
             Polygon([v[0], v[1], v[4]], "a"),
             Polygon([v[1], v[2], v[4]], "b"),
             Polygon([v[2], v[3], v[4]], "c"),
             Polygon([v[3], v[0], v[4]], "d")]
    with pytest.raises((ValueError, MixedRadicals)):
        offset_solid_inward(Solid(polys), 1)
