"""A plane cutting a SPHERE — the second rung of the quadric boolean (K2.x).

Easier than the flat on a bar, and exact in a strictly wider set of places. The
intersection of a plane with a sphere is ALWAYS a circle — there is no oblique
case to classify, no ellipse, no line pair — and the cap the plane removes has
an elementary closed form with no trigonometry in it at all:

    cap of height a on a sphere of radius r:  V = pi a^2 (3r - a) / 3

so the kept volume is exact in Q[pi] for EVERY rational cut height. The flat on
a bar needed the chord to meet the circle at a twelfth because its area carries
an arc angle; this one carries none, so there is no twelfth constraint and
nothing to refuse on that account. The rim's RADIUS is sqrt(r^2 - z^2), which
is a surd, but it is only ever squared again by the terms that use it.

The result is two faces: the pole-containing remainder of the sphere, and the
disc that caps it. ``SphereS.pole`` already exists to carry exactly this trim —
a one-rim spherical face is ambiguous between the cap and the remainder, and
`pole` is what resolves it (see `_sphere_pole_span`) — so this rung is mostly a
matter of ASKING for a representation the kernel already had.
"""

from __future__ import annotations

from fractions import Fraction as F

from forgekernel import body as B
from forgekernel.exact import as_fraction
from forgekernel.surd import sqrt_rational

UP = (F(0), F(0), F(1))
DN = (F(0), F(0), F(-1))
XR = (F(1), F(0), F(0))


class SphereCutRefused(ValueError):
    """Not a representable plane-cuts-sphere. Named so a caller can tell it
    from a bug and from an unwritten algorithm (ADR-0019 / #114)."""


def cut_sphere_on_axis(sphere, axis: int, cut, keep_low: bool = True):
    """A ``Sphere`` cut by the plane perpendicular to a PRINCIPAL axis.

    ``axis`` is 0, 1 or 2. The z case is the one written out below; x and y
    permute the coordinates into it and permute the answer back, which is
    exact because a permutation of axes is a signed relabelling and nothing
    here is measured until the face is built.
    """
    from forgekernel.quadric import Sphere

    if axis == 2:
        return cut_sphere_at_z(sphere, cut, keep_low)
    if axis not in (0, 1):
        raise SphereCutRefused(f"axis must be 0, 1 or 2, got {axis!r}")
    c = (sphere.cx, sphere.cy, sphere.cz)
    # send `axis` to z, keeping the other two in order
    order = ([1, 2, 0] if axis == 0 else [2, 0, 1])
    inv = [order.index(j) for j in range(3)]
    turned = Sphere(c[order[0]], c[order[1]], c[order[2]], sphere.r)
    out = cut_sphere_at_z(turned, cut, keep_low)
    return _permute_body(out, inv)


def _permute_body(body: B.Body, perm) -> B.Body:
    """Relabel a body's axes by ``perm`` — exact, and orientation-preserving
    because only cyclic permutations are used above (an odd permutation would
    turn every face inside out)."""
    def pv(v):
        return (v[perm[0]], v[perm[1]], v[perm[2]])

    def pcurve(cv):
        if isinstance(cv, B.Circle):
            return B.Circle(pv(cv.c), pv(cv.n), pv(cv.ref), cv.r)
        return B.Line(pv(cv.p), pv(cv.d))

    def pface(f):
        s = f.surface
        if isinstance(s, B.Plane):
            surf = B.Plane(pv(s.n), s.d)
        elif isinstance(s, B.SphereS):
            surf = B.SphereS(pv(s.c), s.r,
                             None if s.pole is None else pv(s.pole))
        else:                                   # pragma: no cover - not built
            raise SphereCutRefused(
                f"cannot permute a {type(s).__name__} face")
        loops = tuple(B.Loop(tuple(B.Edge(pcurve(e.curve), pv(e.v0), pv(e.v1))
                                   for e in lp.edges)) for lp in f.loops)
        return B.Face(surf, loops, f.sense)

    return B.Body(tuple(pface(f) for f in body.faces))


def cut_sphere_at_z(sphere, zc, keep_below: bool = True) -> B.Body:
    """A ``Sphere`` cut by the horizontal plane z = ``zc``.

    ``keep_below`` keeps the part containing the SOUTH pole (z < zc), which is
    the remainder; otherwise the north cap's side is kept. Either way the
    result is one pole-trimmed spherical face plus one disc.
    """
    from forgekernel.quadric import Sphere

    if not isinstance(sphere, Sphere):
        raise SphereCutRefused(
            f"cut_sphere_at_z wants a Sphere, got {type(sphere).__name__}")
    cx, cy, cz, r = sphere.cx, sphere.cy, sphere.cz, sphere.r
    z = as_fraction(zc) if as_fraction(zc) is not None else zc
    t = z - cz                               # cut height relative to centre
    if not (-r < t < r):
        raise SphereCutRefused(
            f"the plane z = {float(z):.6g} does not cut a sphere of radius "
            f"{float(r):.6g} centred at z = {float(cz):.6g}")

    # rim radius sqrt(r^2 - t^2) — a surd in general, and exact as one
    rim2 = r * r - t * t
    rim = sqrt_rational(rim2)

    pole = DN if keep_below else UP
    # the disc's outward normal points AWAY from the kept material
    disc_n = UP if keep_below else DN
    disc_d = z if keep_below else -z

    circ = B.Circle((cx, cy, z), UP if keep_below else DN, XR, rim)
    p0 = (cx + rim, cy, z)
    rim_edge = B.Edge(circ, p0, p0)           # a full circle: v0 == v1

    ball = B.Face(B.SphereS((cx, cy, cz), r, pole),
                  (B.Loop((rim_edge,)),), True)
    disc = B.Face(B.Plane(disc_n, disc_d),
                  (B.Loop((B.Edge(B.Circle((cx, cy, z),
                                           DN if keep_below else UP, XR, rim),
                                  p0, p0),)),), True)

    out = B.Body((ball, disc))
    bad = B.manifold_violations(out)
    if bad:
        raise SphereCutRefused(
            f"the cut produced a shell with {len(bad)} unpaired edges — "
            "refusing rather than returning a body that is not closed")
    return out
