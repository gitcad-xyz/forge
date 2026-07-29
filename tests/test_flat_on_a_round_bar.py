"""A flat milled on a round bar — the first rung of the quadric boolean (K2.x).

``boolean.cut(cylinder, box)`` where the box passes clean through one side
refused at EVERY depth, including the ones that are exactly representable. The
guard said "the prism reaches the lathe's wall", which is the right refusal for
a prism cutting INTO a lathe and the wrong one here: reaching the outer wall is
the whole point of a flat.

The plane is parallel to the axis, so the intersection is a pair of straight
lines up the wall — no ellipse, no transcendence — and the kept cross-section
is a disc minus a circular segment:

    keep x <= h,  cos(theta) = h/r
    A = r^2 (pi - theta + sin theta cos theta),   V = A * height

That closed form is the oracle, and it is checked first on the case anyone can
verify by inspection: h = 0 halves the bar, so V must be exactly pi r^2 H / 2.

EXACT ONLY AT TWELFTHS. theta must be a multiple of 30 degrees, because those
are the angles whose sine and cosine the trimmed-arc machinery holds. sqrt2/2
is NOT among them — 45 degrees is an eighth, and although the rotation table
has it (a rotation needs only the matrix, never an arc endpoint) the arc
machinery does not. The first draft of this file claimed otherwise and the
sweep across all depths caught it; spot-checking two values would not have.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

from forgekernel import body as B
from forgekernel.flat import FlatRefused, flat_cut
from forgekernel.quadric import Cyl
from forgekernel.surd import sqrt_rational

R, H = F(5), F(12)


def _bar():
    return Cyl(F(0), F(0), R, F(0), H)


def _closed_form(deg: int) -> float:
    th = math.radians(deg)
    return (float(R) ** 2 * (math.pi - th + math.sin(th) * math.cos(th))
            * float(H))


#: (name, offset h, the chord's half-angle in degrees)
DEPTHS = [
    ("h = r cos30", R * sqrt_rational(3) / 2, 30),
    ("h = r/2", R / 2, 60),
    ("h = 0 (halved)", F(0), 90),
    ("h = -r/2", -R / 2, 120),
    ("h = -r cos30", -R * sqrt_rational(3) / 2, 150),
]


def test_the_closed_form_is_right_about_the_case_anyone_can_check():
    """Validate the oracle before using it: halving a bar leaves half its
    volume. If this is wrong, nothing below means anything."""
    assert _closed_form(90) == pytest.approx(
        math.pi * float(R) ** 2 * float(H) / 2, abs=1e-9)


@pytest.mark.parametrize("name,h,deg", DEPTHS, ids=[d[0] for d in DEPTHS])
def test_the_volume_matches_the_closed_form(name, h, deg):
    v = float(B.volume(flat_cut(_bar(), h)))
    assert v == pytest.approx(_closed_form(deg), abs=1e-9), name


@pytest.mark.parametrize("name,h,deg", DEPTHS, ids=[d[0] for d in DEPTHS])
def test_the_shell_is_closed(name, h, deg):
    """Four faces, every edge shared by exactly two of them. A body with the
    right volume and an open shell is the classic silent defect."""
    out = flat_cut(_bar(), h)
    assert len(out.faces) == 4
    assert B.manifold_violations(out) == []


@pytest.mark.parametrize("name,h,deg", DEPTHS, ids=[d[0] for d in DEPTHS])
def test_a_deeper_flat_leaves_less(name, h, deg):
    """Monotonicity — the cheapest check that catches a sign error, and the
    one that would have caught a plane pointing the wrong way."""
    v = float(B.volume(flat_cut(_bar(), h)))
    assert 0 < v < math.pi * float(R) ** 2 * float(H)


def test_the_volume_decreases_as_the_flat_goes_deeper():
    vs = [float(B.volume(flat_cut(_bar(), h))) for _n, h, _d in DEPTHS]
    assert vs == sorted(vs, reverse=True), vs


def test_it_matches_an_independent_monte_carlo():
    """The closed form is analytic, so this is a cross-check on the FACES the
    kernel actually built rather than on the arithmetic: sample the bar and
    count what the returned body's own half-spaces keep."""
    import random

    h = R / 2
    out = flat_cut(_bar(), h)
    r, hh, H_ = float(R), float(h), float(H)
    random.seed(5)
    n = inside = 0
    for _ in range(200_000):
        x = random.uniform(-r, r)
        y = random.uniform(-r, r)
        z = random.uniform(0, H_)
        n += 1
        if x * x + y * y <= r * r and x <= hh:
            inside += 1
    mc = (2 * r) * (2 * r) * H_ * inside / n
    assert float(B.volume(out)) == pytest.approx(mc, rel=0.01)


def test_an_eighth_is_not_a_twelfth():
    """sqrt2/2 is cos 45, which the rotation table holds and the ARC machinery
    does not. Refusing is correct; quietly rounding to 30 or 60 would not be."""
    with pytest.raises(FlatRefused, match="twelfth"):
        flat_cut(_bar(), R * sqrt_rational(2) / 2)


@pytest.mark.parametrize("h", [F(2), F(1), F(7, 3)])
def test_an_arbitrary_depth_refuses_by_name(h):
    with pytest.raises(FlatRefused, match="twelfth"):
        flat_cut(_bar(), h)


@pytest.mark.parametrize("h", [R, -R, R * 2, -R * 3])
def test_a_flat_that_misses_or_annihilates_refuses(h):
    with pytest.raises(FlatRefused, match="does not meet"):
        flat_cut(_bar(), h)


def test_it_refuses_a_shape_that_is_not_a_bar():
    from forgekernel.brep import Solid

    with pytest.raises(FlatRefused, match="wants a Cyl"):
        flat_cut(Solid.box(10, 10, 10), F(0))


def test_the_flat_face_is_where_the_flat_is():
    """The bug this file's construction actually had: the flat's plane normal
    was set to the twelfth direction that locates the chord's ENDPOINTS rather
    than to the direction the material was cut away in. Those coincide only
    for a tangent chord, which a flat never is — and the caps still measured
    correctly, so the volume was right for h >= 0 and only fell over at
    h = -r/2. A wrong plane that agrees on the easy half is what ships."""
    out = flat_cut(_bar(), R / 2)
    planes = [f.surface for f in out.faces
              if type(f.surface).__name__ == "Plane"]
    verticals = [p for p in planes if p.n[2] == 0]
    assert len(verticals) == 1, "exactly one face is the flat"
    n, d = verticals[0].n, verticals[0].d
    assert (n[0], n[1]) == (F(1), F(0)), f"normal should be +x, got {n}"
    assert d == R / 2, f"offset should be h, got {d}"
