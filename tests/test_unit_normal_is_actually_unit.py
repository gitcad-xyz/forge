"""``_unit_normal`` must return a UNIT vector, or None.

Found 2026-07-28 while tracing a chamfer that removed 87% of a tapered solid.
The guard was:

    nn = c0² + c1² + c2²                 # a Fraction
    root = math.isqrt(int(nn))           # <-- int() TRUNCATES
    if root * root != int(nn): return None

For a canonical normal of (1, 0, 1/6) — a lofted solid's slanted face —
``nn`` is 37/36. ``int(37/36)`` is **1**, ``isqrt(1)`` is 1, and 1*1 == 1, so
the perfect-square test PASSED and the function returned (1, 0, 1/6), whose
length is √37/6 ≈ 1.0138.

An exactness guard made vacuous by an integer truncation. Everything
downstream that trusted the result to be unit then walked the wrong distance:
``chamfer_planar`` offsets ``d`` along ``cross(edge_dir, normal)``, so a
non-unit normal scales every chamfer wedge, and on the capability corpus's
frustum chamfer(d=1) returned 100.55 of an original 784 — with the answer
audit reporting valid=True. At d=2 it returned **negative** volume.

The correct test is the one the shell code already uses: |n|² = p/q is a
perfect rational square iff p and q are both perfect squares, and then
|n| = √p/√q exactly.
"""

from fractions import Fraction as F

import pytest

from forgekernel.brep import Plane, _unit_normal


def _n(*coeffs):
    return _unit_normal(Plane(tuple(F(c) for c in coeffs), F(0)))


def _len2(v):
    return sum(F(c) * F(c) for c in v)


@pytest.mark.parametrize("normal", [
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0),
    (3, 4, 0), (0, 5, 12), (2, 3, 6), (120, 0, 0),
])
def test_a_returned_normal_is_exactly_unit(normal):
    """Whatever comes back must satisfy |n|² == 1 EXACTLY, in ℚ."""
    u = _n(*normal)
    if u is None:
        return                               # refusing is always allowed
    assert _len2(u) == 1, f"{normal} -> {u} has |n|² = {_len2(u)}"


@pytest.mark.parametrize("normal", [
    (120, 0, 20),        # the loft's slanted face: |n| = √37/6 after canon
    (72, 0, 12),         # the same face at another scale
    (0, -120, 20),
    (1, 1, 0),           # √2
    (1, 1, 1),           # √3
    (6, 0, 1),
])
def test_a_non_pythagorean_normal_refuses(normal):
    """These have no rational unit vector, so the only correct answer is
    None. Returning a near-unit vector is what caused the chamfer defect."""
    assert _n(*normal) is None, f"{normal} must refuse, got {_n(*normal)}"


def test_the_exact_case_that_broke_chamfer():
    """The verbatim canonical normal from the corpus loft."""
    u = _unit_normal(Plane((F(120), F(0), F(20)), F(1200)))
    assert u is None, f"expected refusal, got {u} with |n|² = {_len2(u)}"


@pytest.mark.parametrize("normal", [(3, 4, 0), (0, 3, 4), (4, 0, 3),
                                    (2, 3, 6), (1, 2, 2), (6, 6, 7)])
def test_pythagorean_normals_still_resolve(normal):
    """The fix must not cost the cases that were always right: any normal
    whose length IS rational must still return its exact unit vector."""
    u = _n(*normal)
    assert u is not None, f"{normal} has a rational length and must resolve"
    assert _len2(u) == 1


def test_a_rational_but_non_integer_length_resolves():
    """|n|² = 25/4 -> |n| = 5/2, a perfect rational square. The old int()
    truncation would have tested isqrt(6) == 6, refusing a case that is
    exactly representable."""
    u = _n(F(3, 2), F(2), F(0))               # |n|² = 9/4 + 4 = 25/4
    assert u is not None
    assert _len2(u) == 1
