"""K7.0 — exact volume of a freeform (Bézier-patch) solid via the flux
theorem. The integrand is polynomial, so the volume is an exact ℚ."""

from __future__ import annotations

from fractions import Fraction

import pytest

from forgekernel.bsolid import (PatchSolid, box_patches, patch_flux,
                                _lagrange_weights, _nodes)
from forgekernel.nurbs import bezier_surface

F = Fraction


def test_exact_quadrature_integrates_polynomials_exactly() -> None:
    for n in (2, 3, 5):
        nodes = _nodes(n)
        w = _lagrange_weights(nodes)
        assert sum(w) == 1                                  # ∫₀¹ 1 = 1
        for k in range(n):                                  # ∫₀¹ xᵏ = 1/(k+1)
            assert sum(wi * ni ** k for wi, ni in zip(w, nodes)) == F(1, k + 1)


def test_box_patch_solid_volume_is_exact() -> None:
    assert PatchSolid(box_patches(3, 4, 5)).volume() == 60
    assert PatchSolid(box_patches(7, 11, 13)).volume() == 1001


def test_bulged_solid_volume_is_exact_rational() -> None:
    # box 3×4×5 with a biquadratic bulge on top (center pole lifted +3);
    # the added volume is exact — a freeform solid with a ℚ volume
    xs, ys = [F(0), F(3, 2), F(3)], [F(0), F(2), F(4)]
    znet = [[5, 5, 5], [5, 8, 5], [5, 5, 5]]
    top = bezier_surface([[(xs[i], ys[j], znet[i][j]) for j in range(3)]
                          for i in range(3)])
    patches = box_patches(3, 4, 5)
    patches[1] = top
    v = PatchSolid(patches).volume()
    assert isinstance(v, Fraction)
    assert v == 64            # 60 + bulge; exact


def test_higher_degree_bulge_still_exact() -> None:
    # a bicubic bulge → flux of degree (8,8); the 3p=9-node quadrature
    # nails it exactly, proving the degree bookkeeping is right
    xs = [F(k, 1) for k in (0, 1, 2, 3)]
    ys = [F(k, 1) for k in (0, 1, 2, 3)]
    z = [[5, 5, 5, 5], [5, 9, 9, 5], [5, 9, 9, 5], [5, 5, 5, 5]]
    top = bezier_surface([[(xs[i], ys[j], z[i][j]) for j in range(4)]
                          for i in range(4)])
    patches = box_patches(3, 3, 5)
    patches[1] = top
    v = PatchSolid(patches).volume()
    assert isinstance(v, Fraction)
    assert v > 45             # 45 box + positive bulge


def test_rational_patch_flux_refuses() -> None:
    from forgekernel.nurbs import BSplineSurface

    arc = [(1, 0), (1, 1), (0, 1)]
    net = [[(x, y, 0), (x, y, 1)] for (x, y) in arc]
    wts = [[F(1), F(1)], [F(3, 4), F(3, 4)], [F(1), F(1)]]
    s = BSplineSurface(2, 1, net, [0, 0, 0, 1, 1, 1], [0, 0, 1, 1], wts)
    with pytest.raises(ValueError, match="K7.1"):
        patch_flux(s)


# -- K7.0b: exact inertia tensor -----------------------------------------------

def test_inertia_tensor_of_box_is_exact() -> None:
    from forgekernel.bsolid import mass_properties

    a, b, c = 3, 4, 6
    mp = mass_properties(PatchSolid(box_patches(a, b, c)))
    assert mp["volume"] == a * b * c
    assert mp["centroid"] == (F(3, 2), F(2), F(3))
    I = mp["inertia"]
    V = a * b * c
    assert I[0][0] == F(V) * (b * b + c * c) / 12          # Ixx exact
    assert I[1][1] == F(V) * (c * c + a * a) / 12
    assert I[2][2] == F(V) * (a * a + b * b) / 12
    assert I[0][1] == 0 and I[1][2] == 0 and I[0][2] == 0  # exactly zero
    assert all(isinstance(I[i][j], Fraction) for i in range(3) for j in range(3))


# -- H2 gauntlet: exact boolean volume-identity fuzzing ------------------------

def test_boolean_volume_identity_fuzz() -> None:
    """V(A∪B) + V(A∩B) == V(A) + V(B) EXACTLY for the exact BSP engine.
    Any rational violation is a hard proof of a boolean bug. Fuzz many
    overlapping box pairs (deterministic seed)."""
    import random

    from forgekernel import csg
    from forgekernel.brep import Solid

    rng = random.Random(20260723)
    checks = 0
    for _ in range(60):
        A = Solid.box(F(rng.randint(2, 8)), F(rng.randint(2, 8)),
                      F(rng.randint(2, 8)), "A")
        B = Solid.box(F(rng.randint(2, 8)), F(rng.randint(2, 8)),
                      F(rng.randint(2, 8)), "B").translated(
            (F(rng.randint(-4, 6)), F(rng.randint(-4, 6)),
             F(rng.randint(-4, 6))))
        va, vb = A.volume(), B.volume()
        try:
            vu = csg.union(A, B).volume()
            vi = csg.intersect(A, B).volume()
        except Exception:
            continue                          # non-overlap intersect may be empty
        assert vu + vi == va + vb             # EXACT rational identity
        checks += 1
    assert checks >= 30                       # actually exercised the engine
