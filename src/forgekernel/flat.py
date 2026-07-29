"""A FLAT milled on a round bar — the first rung of the general quadric
boolean (K2.x).

The everyday mechanical version of "a plane cuts a cylinder": a D-shaft, a
spanner flat, a keyed hub. It is exact, so it is worth having on its own terms
rather than waiting for the general engine.

This docstring used to claim it was "the smallest honest piece of the boolean
family that ~77 of the composed grid's gaps need". That 77 came from reading
refusal strings and is wrong; instrumenting the operands
(`gitcad.bench.refusals`) puts the plane-cuts-a-cylinder family at ~50 of the
composed gaps, of which the ones needing only LINES and CIRCLES — this family
and the bores, not ellipses — are the large majority. Left here rather than
quietly deleted, because the number was load-bearing for a roadmap decision.

The plane is PARALLEL to the axis, so the intersection curve is a pair of
straight lines up the wall — no ellipse, no transcendence. Keeping the side of
the plane containing the axis leaves a disc minus a circular segment, and the
volume has a closed form:

    keep x <= h on a disc of radius r,  cos(theta) = h/r
    A = r^2 * (pi - theta + sin(theta) cos(theta))
    V = A * height

EXACTNESS is the same rule the rest of the trimmed-quadric work uses: theta
must land on a TWELFTH — a multiple of 30 degrees — because those are the
angles whose sine and cosine this kernel's trimmed-arc machinery holds. The
cosines available are therefore {0, +-1/2, +-sqrt3/2, +-1} and NOT sqrt2/2:
45 degrees is an eighth, not a twelfth, and although the ROTATION table has it
(rotations only need the matrix, not an arc endpoint) ``_sin_cos_twelfths``
does not. An earlier draft of this docstring claimed sqrt2/2 was available;
the sweep across all the depths said otherwise, which is the whole reason to
sweep rather than to spot-check.

Anywhere off a twelfth, theta is an arccos with no algebraic home (Niven), the
arc term leaves Q[sqrt d][pi], and this refuses by name rather than rounding.
So the depth is not free — but the exact set still covers the halving cut
(h = 0) and the two flats either side of it (h = +-r/2, +-r*sqrt3/2).

The result is a Body of FOUR faces: two caps whose loops are a major arc closed
by the chord, the remaining cylindrical band, and the flat itself. Every edge
is shared by exactly two faces by construction, and ``flat_cut`` checks that on
the answer rather than trusting the construction.
"""

from __future__ import annotations

from fractions import Fraction as F

from forgekernel import body as B

#: the twelfth index whose cosine is each exactly-held value, as (index, cos)
#: pairs — cos(k*pi/6) for k = 0..11 is what _sin_cos_twelfths holds
UP = (F(0), F(0), F(1))
DN = (F(0), F(0), F(-1))
XR = (F(1), F(0), F(0))


class FlatRefused(ValueError):
    """The cut is not a representable flat. Named so a caller can tell this
    apart from a bug and from an unwritten algorithm (ADR-0019 / #114)."""


def _twelfth_cos_index(ratio):
    """The twelfth index k with cos(k * 30 deg) == ratio, or None.

    Exact comparison against the twelve held values — no float, no tolerance.
    ``ratio`` is h/r and may live in Q[sqrt d], which is exactly where
    cos(30 deg) and cos(45 deg) live, so equality here is field equality.
    """
    for k in range(12):
        co, _si = B._sin_cos_twelfths()[k]
        if co == ratio:
            return k
    return None


def _pt(cx, cy, z, r, k):
    co, si = B._sin_cos_twelfths()[k % 12]
    return (cx + r * co, cy + r * si, z)


def _arc_edges(circle, cx, cy, z, r, ks):
    """A chain of edges through consecutive twelfth points.

    One Edge between two points on a circle is AMBIGUOUS — there are two arcs
    joining them — so the walk is split at every twelfth it passes, which is
    the same disambiguation notch.py uses.
    """
    return tuple(B.Edge(circle, _pt(cx, cy, z, r, ks[i]),
                        _pt(cx, cy, z, r, ks[i + 1]))
                 for i in range(len(ks) - 1))


def flat_cut(cyl, h, keep_axis_side: bool = True) -> B.Body:
    """A ``Cyl`` with one flat milled at signed offset ``h`` along +x.

    ``h`` is the signed distance from the axis to the flat; the material kept
    is x <= cx + h. h > 0 leaves more than half the bar, h = 0 halves it, and
    h < 0 leaves less than half. |h| < r is required — a flat that misses the
    bar is not a flat, and one that removes everything is not a solid.
    """
    from forgekernel.quadric import Cyl

    if not isinstance(cyl, Cyl):
        raise FlatRefused(
            f"flat_cut wants a Cyl, got {type(cyl).__name__}")
    r, cx, cy, z0, z1 = cyl.r, cyl.cx, cyl.cy, cyl.z0, cyl.z1
    h = F(h) if not hasattr(h, "is_exact_scalar") else h
    if not (-r < h < r):
        raise FlatRefused(
            f"a flat at offset {float(h):.6g} does not meet a bar of radius "
            f"{float(r):.6g} (it must satisfy -r < h < r)")

    k = _twelfth_cos_index(h / r)
    if k is None:
        raise FlatRefused(
            f"h/r = {float(h / r):.6g} is not the cosine of a twelfth, so the "
            "kept area's arc term is an arccos with no algebraic value this "
            "kernel holds (Niven) — the exact flats are h/r in "
            "{0, +-1/2, +-sqrt3/2}, i.e. the chord meeting the circle at a "
            "multiple of 30 degrees")

    # the chord meets the circle at +-k twelfths; keep the arc going the LONG
    # way round through 180 deg, which is the side the axis is on
    ks = [k + i for i in range(12 - 2 * k + 1)]        # k, k+1, ... , 12-k
    if len(ks) < 2:
        raise FlatRefused("the flat removes the whole bar")

    c_lo_dn = B.Circle((cx, cy, z0), DN, XR, r)
    c_lo_up = B.Circle((cx, cy, z0), UP, XR, r)
    c_hi_up = B.Circle((cx, cy, z1), UP, XR, r)
    c_hi_dn = B.Circle((cx, cy, z1), DN, XR, r)

    a_lo, b_lo = _pt(cx, cy, z0, r, ks[0]), _pt(cx, cy, z0, r, ks[-1])
    a_hi, b_hi = _pt(cx, cy, z1, r, ks[0]), _pt(cx, cy, z1, r, ks[-1])

    def line(p, q):
        return B.Edge(B.Line(p, tuple(q[i] - p[i] for i in range(3))), p, q)

    # 1. bottom cap: the kept arc, closed by the chord
    bottom = B.Face(B.Plane(DN, -z0), (B.Loop(
        _arc_edges(c_lo_dn, cx, cy, z0, r, list(reversed(ks)))
        + (line(a_lo, b_lo),)),), True)

    # 2. top cap: the same loop the other way up
    top = B.Face(B.Plane(UP, z1), (B.Loop(
        _arc_edges(c_hi_up, cx, cy, z1, r, ks) + (line(b_hi, a_hi),)),), True)

    # 3. the remaining wall: a trimmed band over the kept arc
    band = B.Face(B.Cylinder((cx, cy, z0), UP, r), (B.Loop(
        _arc_edges(c_lo_up, cx, cy, z0, r, ks)
        + (line(b_lo, b_hi),)
        + _arc_edges(c_hi_dn, cx, cy, z1, r, list(reversed(ks)))
        + (line(a_hi, a_lo),)),), True)

    # 4. the flat itself — a rectangle on the chord.
    #
    # The normal is +x, the direction the material was cut AWAY in, at offset
    # cx + h. Not the twelfth direction (cos k, sin k, 0) that locates the
    # chord's ENDPOINTS — those are two different vectors and they coincide
    # only when the chord is tangent, which is the one case a flat never is.
    # Using the endpoint direction put the plane at y = 5 for a halving cut,
    # and the caps' loops still measured correctly, so the volume came out
    # right for h >= 0 and only fell over at h = -r/2. A wrong plane that
    # agrees on the easy half is exactly the kind of thing that ships.
    flat = B.Face(B.Plane(XR, cx + h),
                  (B.Loop((line(b_lo, a_lo), line(a_lo, a_hi),
                           line(a_hi, b_hi), line(b_hi, b_lo))),), True)

    out = B.Body((bottom, top, band, flat))
    bad = B.manifold_violations(out)
    if bad:
        raise FlatRefused(
            f"the flat produced a shell with {len(bad)} unpaired edges — "
            "refusing rather than returning a body that is not closed")
    return out
