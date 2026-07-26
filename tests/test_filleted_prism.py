"""FilletedPrism — every edge of a rectilinear right prism blended at one r.

The closed form was verified before being written down (never from the
implementation): slab integration and an independent boolean decomposition
(union of two rounded boxes + the reentrant wedge) agree symbolically, and
Monte-Carlo membership written straight from the rolling-ball definition
measures 3942.57 ± 0.57 (L), 3152.58 ± 0.36 (T), against 3943.336 / 3152.289.
The decomposed pieces were pinned tighter: A∩B = 2134/3 + 43π/2 to ±0.009,
reentrant blend (1−π/4)(46/3 − 2π) to ±0.001.

    V = (h−2r)·A_mid + 2·V_cap
    A_mid = A₀ − (n_cv − n_rf)(1−π/4)r²
    V_cap = A₀r − P·r²(1−π/4) + 4r³(5/3−π/2)
            − n_cv(1−π/4)(2r³/3) + n_rf(1−π/4)r³(14/3−π)

The n_rf term is the reentrant corner: a quarter-swept torus (major 2r,
minor r) where the concave vertical blend meets each cap band — the source
of the π² coefficient n_rf·r³/2.
"""

from fractions import Fraction as F

import pytest

from forgekernel.polypi import PiPoly
from forgekernel.quadric import FilletedPrism, PiVal, RoundedBox

L = [(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)]
T = [(0, 0), (30, 0), (30, 10), (20, 10), (20, 20), (10, 20),
     (10, 10), (0, 10)]


def test_l_prism_volume_exact():
    v = FilletedPrism(L, 0, 8, 1).volume()
    assert v == PiPoly([F(3752), F(178, 3), F(1, 2)])


def test_t_prism_volume_exact():
    v = FilletedPrism(T, 0, 8, 1).volume()
    assert v == PiPoly([F(3000), F(136, 3), F(1)])


def test_rectangle_collapses_to_steiner():
    # a convex rectilinear polygon is a rectangle; the formula must reproduce
    # the RoundedBox Steiner value exactly, and the type stays PiVal (no π²)
    v = FilletedPrism([(0, 0), (20, 0), (20, 20), (0, 20)], 0, 20, 3).volume()
    assert v == RoundedBox(20, 20, 20, 3).volume()
    assert isinstance(v, PiVal)


def test_orientation_and_translation_invariance():
    v1 = FilletedPrism(L, 0, 8, 1).volume()
    v2 = FilletedPrism(list(reversed(L)), 0, 8, 1).volume()      # CW input
    v3 = FilletedPrism([(x + 5, y - 7) for x, y in L], -3, 8, 1).volume()
    assert v1 == v2 == v3


def test_centroid_symmetry_and_bbox():
    fp = FilletedPrism(L, 0, 8, 1)
    cx, cy, cz = fp.centroid_f()
    assert cx == pytest.approx(cy, abs=1e-9)         # x=y mirror symmetry
    assert cx == pytest.approx(10.987, abs=0.008)    # MC: 10.987 ± 0.002
    assert cz == pytest.approx(4.0)
    assert fp.bbox() == ((0.0, 0.0, 0.0), (30.0, 30.0, 8.0))
    assert fp.watertight_violations() == []
    t = FilletedPrism(T, 0, 8, 1)
    cx, cy, cz = t.centroid_f()
    assert cx == pytest.approx(15.0, abs=1e-9)       # mirror symmetry
    assert cy == pytest.approx(7.490, abs=0.008)     # MC: 7.490 ± 0.001


def test_guards():
    with pytest.raises(ValueError):                  # diagonal edge
        FilletedPrism([(0, 0), (20, 0), (0, 20)], 0, 8, 1)
    with pytest.raises(ValueError):                  # edge shorter than 2r
        FilletedPrism(L, 0, 8, 6)
    with pytest.raises(ValueError):                  # height shorter than 2r
        FilletedPrism(L, 0, 1, 1)
    with pytest.raises(ValueError):                  # zero radius
        FilletedPrism(L, 0, 8, 0)
    with pytest.raises(ValueError):                  # collinear doubled point
        FilletedPrism([(0, 0), (10, 0), (30, 0), (30, 10), (10, 10),
                       (10, 30), (0, 30)], 0, 8, 1)
    # the H: every edge is >= 4 long but the 3mm crossbar is thinner than
    # 2r = 4 — only the local-feature-size (edge-pair distance) guard sees it
    h_pts = [(0, 0), (10, 0), (10, F(17, 2)), (20, F(17, 2)), (20, 0),
             (30, 0), (30, 20), (20, 20), (20, F(23, 2)), (10, F(23, 2)),
             (10, 20), (0, 20)]
    with pytest.raises(ValueError):
        FilletedPrism(h_pts, 0, 8, 2)


def test_h_volume():
    # ... and at r = 1 the same H is fine (crossbar 3 >= 2r = 2).
    # A0 = 430, P = 134, n_cv = 8, n_rf = 4, r = 1, h = 8:
    #   A_mid = 430 - 4(1-pi/4) = 426 + pi
    #   V_cap = 430 - 134(1-pi/4) + 4(5/3-pi/2) - 8(1-pi/4)(2/3)
    #           + 4(1-pi/4)(14/3-pi) = 316 + 145pi/6 + pi^2
    #   V = 6(426+pi) + 2*V_cap = 3188 + 163pi/3 + 2pi^2 = 3378.4324...
    # Monte-Carlo membership, 180M samples: 3378.494 +- 0.163 (z = +0.4),
    # and the structural membership (union of the three core dilations plus
    # four reentrant blends) agreed pointwise with the cross-section one on
    # 20M samples.
    h_pts = [(0, 0), (10, 0), (10, F(17, 2)), (20, F(17, 2)), (20, 0),
             (30, 0), (30, 20), (20, 20), (20, F(23, 2)), (10, F(23, 2)),
             (10, 20), (0, 20)]
    v = FilletedPrism(h_pts, 0, 8, 1).volume()
    assert v == PiPoly([F(3188), F(163, 3), F(2)])
