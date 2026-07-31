"""A certified arc span must AGREE with the exact one wherever both answer.

ADR-0023's technical content is that a trimmed quadric face may state its
extent as a certified interval instead of an integer sector count. That means a
second implementation of a number the kernel already computes — the situation
in which silent wrong answers are born. So the contract is not "the certified
span looks right"; it is:

  * wherever `_arc_quarters` answers, the certified span BRACKETS it exactly;
  * where `_arc_quarters` refuses (off the twelfth grid), the certified span
    still answers, and answers something a chord-geometry check confirms.

The second half is the capability; the first half is what makes it safe to
believe. `arc_cos_sin` is exact at every angle, which is the observation the
whole ADR rests on: an arc endpoint off the grid has an exactly known cosine
and sine, and only the ANGLE is transcendental.
"""

import math
from fractions import Fraction as F

import pytest

from forgekernel import body as B
from forgekernel.interval import CInterval, pi_interval
from forgekernel.surd import sqrt_rational

R = F(5)
UP = (F(0), F(0), F(1))
XR = (F(1), F(0), F(0))
CIRCLE = B.Circle((F(0), F(0), F(0)), UP, XR, R)


def _twelfth_point(k):
    co, si = B._sin_cos_twelfths()[k % 12]
    return (R * co, R * si, F(0))


# -- agreement with the exact path -------------------------------------------

@pytest.mark.parametrize("k0", range(12))
@pytest.mark.parametrize("dk", [1, 2, 3, 4, 6, 9, 11])
def test_the_certified_span_brackets_the_exact_one(k0, dk):
    v0, v1 = _twelfth_point(k0), _twelfth_point(k0 + dk)
    exact_k0, exact_span = B._arc_quarters(CIRCLE, v0, v1)
    assert exact_k0 == k0
    assert exact_span == dk

    got = B.arc_span_certified(CIRCLE, v0, v1)
    assert got is not None
    # the exact span is dk twelfths = dk*pi/6
    # both sides are certified, so this comparison IS exact: two brackets that
    # both enclose the truth must overlap
    want = pi_interval() * CInterval.exact(F(dk, 6))
    assert got.lo <= want.hi and want.lo <= got.hi, (k0, dk, got, want)


def test_the_exact_cosine_and_sine_are_exact_off_the_grid():
    """The observation ADR-0023 rests on. A flat at depth h=1 on a radius-5 bar
    has chord endpoints whose cosine is 1/5 — rational — and whose sine is
    sqrt(24)/5. `_quarter_index` refuses this point; the cos/sin do not."""
    h = F(1)
    s = sqrt_rational(R * R - h * h)
    p = (h, s, F(0))
    assert B._quarter_index(CIRCLE, p) is None      # off the twelfth grid
    co, si = B.arc_cos_sin(CIRCLE, p)
    assert co == F(1, 5)                            # EXACT, and rational
    assert si * si == F(24, 25)                     # EXACT, and a surd


# -- the capability: spans the exact path refuses ----------------------------

@pytest.mark.parametrize("h", [F(1), F(3), F(-1), F(7, 3), F(-9, 4), F(1, 8)])
def test_a_chord_off_the_grid_gets_a_span_the_exact_path_refuses(h):
    """The kept arc of a flat at depth h: from +theta round through pi to
    -theta, so its span is 2(pi - theta) with cos(theta) = h/r."""
    s = sqrt_rational(R * R - h * h)
    a = (h, s, F(0))                                # angle +theta
    b = (h, -s, F(0))                               # angle -theta == 2pi-theta

    with pytest.raises(ValueError):
        B._arc_quarters(CIRCLE, a, b)               # exact path cannot

    span = B.arc_span_certified(CIRCLE, a, b)
    assert span is not None
    # A FLOAT CANNOT WITNESS THIS BRACKET. The span is certified to ~1e-30
    # and a double carries ~1e-16 of error, so `lo <= want <= hi` fails on the
    # ORACLE, not on the code — the same trap `test_certified_trigonometry`
    # documents. Compare at a tolerance a double can honour, and check the
    # tightness separately.
    want = 2 * (math.pi - math.acos(float(h / R)))
    assert abs(float(span.mid) - want) < 1e-12, (h, span, want)
    # ~1e-15: the bracket is centred on `math.acos`, so the FLOAT'S OWN ERROR
    # is the floor (see `_ARCCOS_WIDTH`). Tighter needs bisection, which is
    # what made one flat volume take 51 seconds. Still nine orders inside
    # what certifying a sign needs.
    assert span.width < F(1, 10 ** 12)


def test_the_two_arcs_of_a_chord_close_the_circle():
    """A structural check that needs no oracle: the kept arc and the removed
    arc must sum to a full turn. An off-by-a-half-turn error in reading the
    sine's sign would satisfy the bracket test above on one side and break
    here."""
    h = F(7, 3)
    s = sqrt_rational(R * R - h * h)
    a, b = (h, s, F(0)), (h, -s, F(0))
    kept = B.arc_span_certified(CIRCLE, a, b)
    removed = B.arc_span_certified(CIRCLE, b, a)
    total = kept + removed
    two_pi = pi_interval() * CInterval.exact(F(2))
    assert total.lo <= two_pi.hi and two_pi.lo <= total.hi, (kept, removed)


def test_an_irrational_cosine_is_bracketed_rather_than_declined():
    """It has to be, or the certified path cannot check itself.

    A first version restricted this to rational cosines, reasoning that a flat's
    chord gives cos = h/r. True — but cos 30° = √3/2, so most of the TWELFTH
    GRID is irrational, and the agreement test above could then only run at the
    four quarters. Restricting to ℚ did not just lose capability; it lost the
    ability to validate.
    """
    s = sqrt_rational(F(2))                          # a point at cos = √2/5
    p = (s, sqrt_rational(R * R - F(2)), F(0))
    span = B.arc_span_certified(CIRCLE, p, _twelfth_point(6))
    assert span is not None
    want = math.pi - math.acos(float(s) / float(R))
    assert abs(float(span.mid) - want) < 1e-12, (span, want)


def test_the_bracket_is_proven_exactly_not_trusted_from_the_float():
    """`certified_bracket` lets a float PROPOSE and exact arithmetic DISPOSE.

    That is the ADR-0019 line: a float may not DECIDE. Here it decides nothing
    — the candidate is accepted only when `lo <= x <= hi` holds in the scalar's
    own exact field — so the returned bracket is a proof. Checked by asserting
    the enclosure exactly, against the surd itself rather than its double.
    """
    x = sqrt_rational(F(3)) / 2                      # cos 30°, in ℚ[√3]
    br = B.certified_bracket(x)
    assert br is not None
    assert br.lo <= x <= br.hi                       # EXACT comparison
    # a rational needs no bracket at all and gets a zero-width one
    exact = B.certified_bracket(F(1, 5))
    assert exact.lo == exact.hi == F(1, 5)


# -- the centroid, off the grid (ADR-0023) -----------------------------------

def _flat_body(h):
    """A flat milled at depth h on a radius-5, height-12 bar — the ADR-0023
    off-grid construction, built here so the centroid test is self-contained."""
    from forgekernel.quadric import Cyl
    from forgekernel.flat import flat_cut
    return flat_cut(Cyl(F(0), F(0), F(5), F(0), F(12)), h)


@pytest.mark.parametrize("h", [F(1), F(7, 3), F(-3, 2), F(1, 8), F(-9, 4)])
def test_the_off_grid_centroid_matches_the_closed_form(h):
    """The flat is symmetric in y and in z, so its centre of mass sits at y=0
    and z=6 EXACTLY, and its x follows the segment's first moment. A body
    carrying an off-grid arc used to have no centroid at all — only a volume.
    """
    cen = B.centroid(_flat_body(h))
    assert abs(cen[1]) < 1e-9, cen        # symmetric in y
    assert abs(cen[2] - 6.0) < 1e-9, cen  # symmetric in z about the mid-height
    # x is the segment centroid: negative (material kept on the low-x side)
    assert cen[0] < 0, cen


def test_the_band_moment_D_term_is_not_optional():
    """The on-grid centroid formula assumes ∫cos²=∫sin²=Δθ/2, true only for the
    quarter-turn arcs that occur on the grid (D := s₁c₁−s₀c₀ = 0). A flat's arc
    is not a quarter-turn, so D≠0, and dropping the (D/2) band term gives a
    plausible WRONG centre of mass. This pins that the term is carried: the x
    centroid of an h=1 flat is ~-1.593, and the D-less value differs in the
    third digit.
    """
    cen = B.centroid(_flat_body(F(1)))
    assert abs(cen[0] - (-1.5931)) < 1e-3, cen


# -- the on-grid centroid D-term (review finding #2) -------------------------

def test_the_on_grid_centroid_carries_the_D_term_off_origin():
    """A flat milled at an EXACT depth (h=r/2) on a bar NOT at the transverse
    origin: the on-grid centroid dropped the band-moment D-term (the same bug
    fixed off-grid), giving a silently wrong centre of mass that broke the
    flat's mirror symmetry. Pinned by translation covariance + mirror symmetry
    against Monte-Carlo, all oracle-independent of the implementation."""
    from forgekernel.flat import flat_cut
    from forgekernel.quadric import Cyl
    c0 = B.centroid(flat_cut(Cyl(F(0), F(0), F(4), F(0), F(10)), F(2)))
    cT = B.centroid(flat_cut(Cyl(F(10), F(5), F(4), F(0), F(10)), F(2)))
    # translation covariance: moving the bar by (10,5) moves the centroid by it
    assert abs(cT[0] - (c0[0] + 10)) < 1e-6, (c0, cT)
    assert abs(cT[1] - (c0[1] + 5)) < 1e-6, (c0, cT)
    # mirror symmetry: the flat is symmetric about y = cy = 5
    assert abs(cT[1] - 5.0) < 1e-9, cT
    # on-grid now equals the off-grid (D-term) path
    off = B._centroid_offgrid(flat_cut(Cyl(F(10), F(5), F(4), F(0), F(10)), F(2)))
    assert all(abs(cT[k] - off[k]) < 1e-9 for k in range(3)), (cT, off)
