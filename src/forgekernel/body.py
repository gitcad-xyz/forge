"""The canonical B-rep (ADR-0021) — one topology every representation maps to.

forge grew a separate solid class per feature, so every operation had to be
written once per representation: O(ops × representations), a matrix that could
never be filled. ``Body`` is the single form each of those classes converts
into, so an operation is written ONCE.

    Body  = [Face]
    Face  = (Surface, [Loop], sense)   # sense: outward normal == surface normal
    Loop  = [Edge]                     # loops[0] is the outer bound, rest holes
    Edge  = (Curve, v0, v1)            # v0 == v1 for a full circle

Surfaces and curves are ANALYTIC and EXACT — a hole is a ``Cylinder`` with a
rational radius, never a polygon fan. Nothing here approximates: mass properties
come from the divergence theorem applied per face in closed form (ℚ or ℚ[π]),
and an operation that would leave the number field refuses with its stage
(ADR-0019). Tessellation is the one deliberate exception, and only because
meshing is a display property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from forgekernel.exact import F, cross, dot, sub

Vec = tuple


# -- surfaces ----------------------------------------------------------------

@dataclass(frozen=True)
class Plane:
    """``n · x = d``; ``n`` need not be normalised (exactness over unit length)."""
    n: Vec
    d: Fraction

    def transformed(self, m):
        # A plane's normal transforms by the INVERSE-TRANSPOSE, not by the map
        # itself. Rigid maps make the two agree and a uniform scale makes them
        # parallel (all a plane needs), but under a NON-UNIFORM scale they
        # diverge: a slanted face would keep a normal no longer perpendicular to
        # it. m.normal() applies the cofactor matrix = det.M^-T — exact in Q, and
        # a parallel normal suffices because d rescales with it.
        p = m.point(_plane_point(self))
        n = m.normal(self.n)
        return Plane(n, dot(n, p))


@dataclass(frozen=True)
class Cylinder:
    """Radius ``r`` about the axis through ``p`` along ``d``."""
    p: Vec
    d: Vec
    r: Fraction

    def transformed(self, m):
        return Cylinder(m.point(self.p), m.direction(self.d), m.scale_len(self.r))


@dataclass(frozen=True)
class Cone:
    """Half-angle given exactly as ``tan = rise/run``; apex at ``p``, axis ``d``."""
    p: Vec
    d: Vec
    tan_half: Fraction

    def transformed(self, m):
        return Cone(m.point(self.p), m.direction(self.d), self.tan_half)


@dataclass(frozen=True)
class Torus:
    """Tube of minor radius ``a`` about the circle of major radius ``R`` that
    lies in the plane through ``c`` perpendicular to ``d``.

    A fillet on a lathed rim sweeps one of these, and it is the first surface
    whose volume needs π² — Pappus gives V = 2πR·πa². ``PiVal`` (a + bπ)
    cannot hold that, which is why ``polypi.PiPoly`` exists.
    """
    c: Vec
    d: Vec
    R: Fraction
    a: Fraction
    k0: int = 0
    span: int = 4

    def transformed(self, m):
        return Torus(m.point(self.c), m.direction(self.d),
                     m.scale_len(self.R), m.scale_len(self.a),
                     self.k0, self.span)


@dataclass(frozen=True)
class SphereS:
    """``pole`` is a TRIM carried on the surface, following the Torus
    precedent (its k0/span): a spherical face bounded by ONE rim is ambiguous
    — the cap on one side and the pole-containing remainder on the other are
    both bounded by that rim, and nothing structural could catch the
    substitution (`_sphere_zone`'s docstring). ``pole`` is the direction (need
    not be unit) of the pole the face CONTAINS. ``None`` — the default, and
    what every existing constructor builds — means untrimmed: a whole sphere,
    an octant, or a two-rim zone, exactly as before."""
    c: Vec
    r: Fraction
    pole: Vec | None = None

    def transformed(self, m):
        return SphereS(m.point(self.c), m.scale_len(self.r),
                       None if self.pole is None else m.direction(self.pole))


# -- curves ------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    p: Vec
    d: Vec

    def transformed(self, m):
        return Line(m.point(self.p), m.direction(self.d))


@dataclass(frozen=True)
class Circle:
    c: Vec
    n: Vec          # plane normal
    ref: Vec        # zero-angle direction, in the plane
    r: Fraction

    def transformed(self, m):
        return Circle(m.point(self.c), m.direction(self.n),
                      m.direction(self.ref), m.scale_len(self.r))


# -- topology ----------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    curve: object
    v0: Vec
    v1: Vec

    def transformed(self, m):
        return Edge(self.curve.transformed(m), m.point(self.v0), m.point(self.v1))


@dataclass(frozen=True)
class Loop:
    edges: tuple

    def transformed(self, m):
        out = tuple(e.transformed(m) for e in self.edges)
        if m.flips:
            # An orientation-reversing map (mirror, negative scale) reverses the
            # winding of the vertex list while the surface normal is mirrored
            # too — so the loop must be re-reversed to stay consistent with the
            # outward normal. Reversing the WINDING is the truthful fix; flipping
            # the face sense instead would claim the mirrored solid is inside-out.
            out = tuple(Edge(e.curve, e.v1, e.v0) for e in reversed(out))
        return Loop(out)


@dataclass(frozen=True)
class Face:
    surface: object
    loops: tuple
    sense: bool = True

    def transformed(self, m):
        return Face(self.surface.transformed(m),
                    tuple(lp.transformed(m) for lp in self.loops), self.sense)


@dataclass(frozen=True)
class Body:
    faces: tuple

    def transformed(self, m):
        return Body(tuple(f.transformed(m) for f in self.faces))


# -- exact affine maps -------------------------------------------------------

class Affine:
    """An exact affine map: 3x3 matrix + translation, over ℚ or ℚ[√d].

    ``flips`` records whether the map reverses orientation (a mirror or a
    negative scale), so faces can flip their sense and stay outward-oriented.
    ``scale_len`` is how a radius transforms — defined only for maps that are
    conformal (rigid or uniform scale); a non-uniform scale would turn a circle
    into an ellipse, which is not in the canonical surface set, so it refuses.
    """

    def __init__(self, rows, t=(0, 0, 0), *, uniform=None, flips=False):
        self.rows = tuple(tuple(r) for r in rows)
        self.t = tuple(t)
        self.uniform = uniform          # the scalar radii scale by, or None
        self.flips = flips

    @classmethod
    def identity(cls):
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)), uniform=F(1))

    @classmethod
    def translation(cls, x, y, z):
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (F(x), F(y), F(z)),
                   uniform=F(1))

    @classmethod
    def scaling(cls, fx, fy, fz):
        fx, fy, fz = F(fx), F(fy), F(fz)
        if fx == 0 or fy == 0 or fz == 0:
            raise ValueError(
                "a zero scale factor collapses the solid — not an invertible "
                "transform")
        uniform = abs(fx) if (abs(fx) == abs(fy) == abs(fz)) else None
        neg = (fx < 0) + (fy < 0) + (fz < 0)
        return cls(((fx, 0, 0), (0, fy, 0), (0, 0, fz)),
                   uniform=uniform, flips=bool(neg % 2))

    @classmethod
    def mirror(cls, axis: str):
        i = "xyz".index(axis)
        rows = [[1 if a == b else 0 for b in range(3)] for a in range(3)]
        rows[i][i] = -1
        return cls(rows, uniform=F(1), flips=True)

    @classmethod
    def rotation(cls, axis, deg):
        from forgekernel.kernel import _rotation_matrix

        return cls(_rotation_matrix(axis, deg), uniform=F(1), flips=False)

    def point(self, v):
        r = self.rows
        return tuple(r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] + self.t[i]
                     for i in range(3))

    def direction(self, v):
        r = self.rows
        return tuple(r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2]
                     for i in range(3))

    def normal(self, v):
        """Map a SURFACE NORMAL — by the cofactor matrix (det.M^-T), so it stays
        perpendicular to the transformed surface even under a non-uniform scale.
        Exact: cofactors are products of matrix entries, no division."""
        r = self.rows
        c = [[r[(i + 1) % 3][(j + 1) % 3] * r[(i + 2) % 3][(j + 2) % 3]
              - r[(i + 1) % 3][(j + 2) % 3] * r[(i + 2) % 3][(j + 1) % 3]
              for j in range(3)] for i in range(3)]
        out = tuple(c[i][0] * v[0] + c[i][1] * v[1] + c[i][2] * v[2]
                    for i in range(3))
        # The cofactor matrix is det·M⁻ᵀ, so an ORIENTATION-REVERSING map (a
        # mirror, det < 0) would hand back an inward-pointing normal. Divide the
        # sign of det back out: |det|·M⁻ᵀ keeps the normal outward, and any
        # positive multiple of M⁻ᵀ is an equally valid plane normal.
        det = sum(self.rows[0][j] * c[0][j] for j in range(3))
        return tuple(-x for x in out) if det < 0 else out

    def scale_len(self, r):
        if self.uniform is None:
            raise ValueError(
                "non-uniform scale turns a circle into an ellipse — not in the "
                "canonical surface set (elliptical surfaces arrive with K3.7)")
        return self.uniform * r


def _plane_point(pl: Plane) -> Vec:
    """Any exact point on the plane: the foot of the normal from the origin."""
    nn = dot(pl.n, pl.n)
    return tuple(pl.n[i] * pl.d / nn for i in range(3))


# -- exact metrics -----------------------------------------------------------
#
# Volume by the divergence theorem, V = (1/3)∮ x·n dA, evaluated per face in
# CLOSED FORM. Each surface type contributes a rational or ℚ[π] term, so the
# total is exact — no sampling, no facets.

def _loop_is_circle(loop: Loop):
    """The whole circle a loop traces, or None.

    A loop is closed by construction, so any number of arcs that all lie on
    the SAME circle and chain end to end must cover it exactly once — two
    halves from a boolean split, three from a triple intersection.
    """
    es = loop.edges
    if not es or not all(isinstance(e.curve, Circle) for e in es):
        return None
    if len(es) == 1:
        # a LONE edge is only the whole circle when it closes on itself; a
        # single proper arc reported a quarter-disc's area as the full πr²
        return es[0].curve if es[0].v0 == es[0].v1 else None
    return es[0].curve if all(e.curve == es[0].curve for e in es) else None


def _loop_has_arcs(loop: Loop) -> bool:
    """A loop with curved edges that is NOT one whole circle — a slot end, a
    D-profile. Treating its vertices as a polygon quietly turns every arc into
    a chord, so the exact paths refuse instead."""
    return (any(isinstance(e.curve, Circle) for e in loop.edges)
            and _loop_is_circle(loop) is None)


def _refuse_arc_loop(loop: Loop) -> None:
    """Every measurement path must decline the same loops.

    ``_face_volume_term`` refused a mixed arc/line loop while ``faces_info``
    and ``centroid`` walked its vertices as a polygon — two paths, two answers
    for one face: a quarter disc measured 0.5 against a true πr²/4 = 0.785,
    and reported it as fact.
    """
    if _loop_has_arcs(loop):
        raise ValueError(
            "planar loop mixing arcs and lines (a slot end, a D-profile) — "
            "chording it under-reports the area, so refuse (K3.7)")


def _arc_pts(c: Circle, v0, v1, deflection: float):
    """Display sampling of one arc, from v0 round to v1 about the circle's
    normal. Floats are legal here — this is meshing (ADR-0019).

    CONTRACT: an ``Edge`` on a ``Circle`` runs COUNTER-CLOCKWISE about
    ``curve.n``, from v0 to v1. An edge written the other way round is not an
    error anyone detects — it silently means the COMPLEMENTARY arc, so a
    quarter disc written backwards meshes as the other three quarters. A face
    that needs the reverse sweep must carry a circle with ``-n``.
    """
    if dot(c.ref, c.n) != 0:
        raise ValueError(
            "circle reference direction is not perpendicular to its normal — "
            "the sampled points would not lie on the circle at all")
    r = float(c.r)
    u = _unit(tuple(float(x) for x in c.ref))
    w = _unit(_cross_f(tuple(float(x) for x in c.n), u))
    cc = tuple(float(x) for x in c.c)

    def ang(p):
        q = [float(p[i]) - cc[i] for i in range(3)]
        return math.atan2(sum(q[i] * w[i] for i in range(3)),
                          sum(q[i] * u[i] for i in range(3)))

    a0, a1 = ang(v0), ang(v1)
    while a1 <= a0:
        a1 += 2 * math.pi
    step = (2 * math.acos(max(-1.0, 1 - deflection / r))
            if r > deflection else math.pi / 12)
    n = max(2, int(math.ceil((a1 - a0) / max(step, 1e-6))))
    return [tuple(cc[i] + r * (math.cos(a0 + (a1 - a0) * k / n) * u[i]
                               + math.sin(a0 + (a1 - a0) * k / n) * w[i])
                  for i in range(3)) for k in range(n)]


def _circle_frame(c: Circle):
    """The circle's exact orthonormal frame (u, w). Refuses unless both are
    exactly unit and perpendicular — a quarter position has to be EXACT or
    none of the trimmed-quadric arithmetic below is."""
    ln, lr = _rational_sqrt(dot(c.n, c.n)), _rational_sqrt(dot(c.ref, c.ref))
    if not ln or not lr or dot(c.ref, c.n) != 0:
        raise ValueError(
            "circle frame is not exactly orthonormal — a trimmed quadric "
            "needs exact quarter positions (K3.7)")
    # NORMALISE rather than demand unit length. A uniform scale maps n and ref
    # through Affine.direction, which does not normalise, so a scaled body
    # carries |n| = |ref| = k — and refusing there lost the volume AND the
    # MESH, gating a display property on an exactness predicate.
    n, u = tuple(x / ln for x in c.n), tuple(x / lr for x in c.ref)
    return u, cross(n, u)


def _quarter_depth(r: float, deflection: float) -> int:
    """Subdivision depth for a quarter arc, shared by trimmed bands and corner
    octants so that they agree vertex for vertex along the arcs they share."""
    # No cap: capping at 128 segments made a rounded box and a bare cylinder
    # disagree about what a deflection MEANS (at r=4999, deflection=0.01 the
    # true sagitta was 9.4x the request). The untrimmed path has no cap.
    d = 1
    while r * (1 - math.cos(math.pi / 4 / (2 ** d))) > deflection:
        d += 1
    return d


def _sin_cos_twelfths():
    """(cos, sin) at every θ = k·π/6, exactly.

    Quarters need only 0 and ±1; twelfths additionally need ±1/2 and ±√3/2,
    which live in ℚ[√3]. That is why the unit here is a TWELFTH and not a
    quarter: a hexagonal boss's blends, a 30° trimmed band and a 60° lune all
    land on this grid, and the volume they produce stays exact because ℚ[√3][π]
    holds it (the π coefficient picks up the √3, which is precisely what the
    field extension was for).
    """
    from forgekernel.surd import SurdVal

    h = Fraction(1, 2)
    r3 = SurdVal(0, Fraction(1, 2), 3)          # √3/2
    return ((1, 0), (r3, h), (h, r3), (0, 1), (-h, r3), (-r3, h),
            (-1, 0), (-r3, -h), (-h, -r3), (0, -1), (h, -r3), (r3, -h))


def _twelfth_dir(c: Circle, k: int, frame=None):
    """The unit direction at k·π/6 in the circle's own frame, or None when it
    is not REPRESENTABLE.

    Computed one k at a time, and that matters: an exactly-rotated body carries
    ℚ[√2] coordinates, and multiplying those by the √3/2 of an odd twelfth is a
    mixed radical that ℚ[√d] cannot hold (K3.1). Building all twelve eagerly
    therefore threw on bodies that only ever needed the four rational ones. A
    direction that cannot be expressed is simply not a candidate.
    """
    u, w = frame if frame is not None else _circle_frame(c)
    co, si = _sin_cos_twelfths()[k % 12]
    try:
        return tuple(co * u[i] + si * w[i] for i in range(3))
    except ValueError:
        return None                 # mixed radicals: not representable, so not here


def _quarter_index(c: Circle, p):
    """Which TWELFTH turn from `ref` the point sits at (0..11), or None.

    Trimmed quadrics stay in an exact field only where sin and cos are exact.
    At quarters those are 0 and ±1 (ℚ); at twelfths they are additionally ±1/2
    and ±√3/2 (ℚ[√3]). Anywhere else the band's ∫n̂ dθ term is a transcendental
    that leaves every field the kernel has, so this is still the predicate that
    decides exact-or-refuse — it just decides YES three times as often.

    The name is kept because the unit is the only thing that changed; a quarter
    is now index 3.
    """
    if c.r == 0:
        return None                 # a degenerate circle has no quarters
    rel = sub(p, c.c)
    frame = _circle_frame(c)
    for k in range(12):
        v = _twelfth_dir(c, k, frame)
        if v is not None and rel == tuple(c.r * x for x in v):
            return k
    return None


def _arc_quarters(c: Circle, v0, v1):
    """(start index, span) for an arc in TWELFTHS of a turn, counter-clockwise
    about the circle's normal. A closed edge is the whole twelve."""
    k0, k1 = _quarter_index(c, v0), _quarter_index(c, v1)
    if k0 is None or k1 is None:
        raise ValueError(
            "arc endpoint is not at a twelfth turn — a trimmed quadric is "
            "exact only at multiples of 30° (K3.7)")
    span = (k1 - k0) % 12
    return k0, (12 if span == 0 else span)


def _quarter_antiderivative(c: Circle, k: int):
    """∫n̂ dθ at θ = k·π/6, i.e. sin(θ)u − cos(θ)w. In ℚ, or ℚ[√3] off the
    quarters."""
    u, w = _circle_frame(c)
    cos_, sin_ = _sin_cos_twelfths()[k % 12]
    return tuple(sin_ * u[i] - cos_ * w[i] for i in range(3))


def _band_arc(face: Face):
    """The face's arcs as (circle, start quarter, quarter span), or None when
    every loop is a whole circle (the untrimmed case)."""
    arcs = []
    for lp in face.loops:
        if _loop_is_circle(lp) is not None:
            continue
        for e in lp.edges:
            if isinstance(e.curve, Circle) and _key3(e.v0) != _key3(e.v1):
                arcs.append((e.curve,) + _arc_quarters(e.curve, e.v0, e.v1))
    return arcs or None


def _key3(v):
    return tuple(v)


def _planar_loop_area2(loop: Loop, n: Vec):
    """Twice the signed area of a polygonal loop, projected along n (exact)."""
    pts = [e.v0 for e in loop.edges]
    acc = (F(0), F(0), F(0))
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        c = cross(a, b)
        acc = (acc[0] + c[0], acc[1] + c[1], acc[2] + c[2])
    return dot(acc, n)


def _mixed_loop_area(loop: Loop, n: Vec):
    """(rational, π-coefficient) signed area of a planar loop mixing Lines and
    Circle arcs — exact, and the only new geometry #123 needs.

    The loop's chord polygon plus one circular-segment correction per arc:
    an arc of radius r sweeping θ adds (r²/2)(θ − sin θ) beyond its chord when
    it runs COUNTER-CLOCKWISE about the face normal, and subtracts it when it
    runs the other way (which is how the same 90° arc bounds a notch floor of
    4 − π on one face and a shoulder hole of 4 + 3π on the next).

    ADR-0019: θ is what decides whether this face is measurable at all, so
    ``_arc_quarters`` decides it — an exact equality against ``r·_twelfth_dir``
    that returns a span in TWELFTHS or refuses. ``atan2`` would accept a 44.999°
    trim as a quarter and hand back a confidently exact wrong number; ``sin θ``
    likewise comes from ``_sin_cos_twelfths`` (0, ±1/2, ±√3/2, ±1) and never
    from ``math.sin``. Nothing here is a float.

    Returns the two parts separately so the caller can route them through
    ``_pi_value``: at quarter turns the answer is in ℚ[π], at odd twelfths the
    sine brings in √3 and it lives in ℚ[√3][π], which ``PiPoly`` holds.
    """
    ln = _rational_sqrt(dot(n, n))
    if ln is None:
        raise ValueError(
            "arc/line loop on a plane whose normal has irrational length — "
            "outside ℚ[π] (K3.7)")
    rat = _planar_loop_area2(loop, n) / (2 * ln)
    pi = F(0)
    for e in loop.edges:
        if not isinstance(e.curve, Circle):
            continue
        # the arc must lie IN the face's plane, or the chord polygon it is
        # correcting is not the projection of this loop at all
        axial = dot(e.curve.n, n)
        if any(x != 0 for x in cross(e.curve.n, n)) or axial == 0:
            raise ValueError(
                "an arc on a planar face whose circle is not coplanar with it "
                "— the face is malformed, not merely unsupported")
        _k0, span = _arc_quarters(e.curve, e.v0, e.v1)
        _co, si = _sin_cos_twelfths()[span % 12]
        s = 1 if axial > 0 else -1
        rat += s * (-e.curve.r ** 2 * si / 2)
        pi += s * e.curve.r ** 2 * Fraction(span, 12)
    return rat, pi


def _mixed_loop_area_as_hole(loop: Loop, n: Vec):
    """A mixed loop's area as a NEGATIVE quantity, whichever way it winds.

    The polygon path takes ``-abs(a2)`` on a rational. A mixed loop's area is a
    ℚ[π] value — the counterbore shoulder's inner loop is 4 + 3π — and
    ``abs(float(...))`` there would be a float deciding the sign of a face's
    area, which decides whether the face refuses (ADR-0019). ``sign()`` narrows
    a rational enclosure of π until the answer is certain, and copes with a
    surd coefficient.
    """
    from forgekernel.polypi import PiPoly

    a_rat, a_pi = _mixed_loop_area(loop, n)
    if PiPoly([a_rat, a_pi]).sign() > 0:
        return -a_rat, -a_pi
    return a_rat, a_pi


def _mixed_loop_area_centroid_f(loop: Loop, nf):
    """Signed area and area centroid of a mixed arc/line loop, in FLOATS.

    The float counterpart of :func:`_mixed_loop_area`, for the two reporting
    paths — ``centroid`` and ``faces_info`` — whose answers are ratios that
    leave ℚ[π] anyway. Floats are legal here and only here (ADR-0019): the
    arc's SPAN still comes from ``_arc_quarters``, exactly, so the same loops
    are admitted and refused as on the exact path; only the arithmetic on an
    already-decided span is floating.

    Chord polygon + one circular segment per arc. A segment of half-angle α has
    area r²(α − sin α cos α) and its first moment about the circle centre is
    (2/3)r³sin³α along the chord's outward bisector — so the two combine
    without ever dividing by a vanishing area.
    """
    vs = [tuple(float(x) for x in e.v0) for e in loop.edges]
    area, cen = _poly_area_centroid_f(vs, nf)
    acc = [cen[k] * area for k in range(3)]
    for e in loop.edges:
        if not isinstance(e.curve, Circle):
            continue
        _k0, span = _arc_quarters(e.curve, e.v0, e.v1)
        r = float(e.curve.r)
        alpha = span * math.pi / 12.0
        s = 1.0 if sum(float(e.curve.n[k]) * nf[k] for k in range(3)) > 0 else -1.0
        a_seg = s * r * r * (alpha - math.sin(alpha) * math.cos(alpha))
        c = tuple(float(x) for x in e.curve.c)
        mid = [(float(e.v0[k]) + float(e.v1[k])) / 2 - c[k] for k in range(3)]
        ln = math.sqrt(sum(x * x for x in mid))
        if ln == 0:                     # a half turn: the chord midpoint IS the
            u, w = _circle_frame(e.curve)       # centre, so use the frame
            th = (_k0 + span / 2.0) * math.pi / 6.0
            bis = [math.cos(th) * float(u[k]) + math.sin(th) * float(w[k])
                   for k in range(3)]
        else:
            bis = [x / ln for x in mid]
        d = (2.0 / 3.0) * r ** 3 * math.sin(alpha) ** 3
        area += a_seg
        for k in range(3):
            acc[k] += s * d * bis[k] + a_seg * c[k]
    if area == 0.0:
        return 0.0, cen
    return area, tuple(acc[k] / area for k in range(3))


def _as_fraction(x):
    """x as a Fraction if it is rational — including a ``SurdVal`` whose surd
    part is zero, which is what an exact rotation leaves behind (a 45° turn
    types every coordinate as ℚ[√2] even where the value is rational)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    b = getattr(x, "b", None)
    if b is not None and b == 0:
        return x.a
    return None


def _exact(x, what: str) -> Fraction:
    """Coerce an exact value to ℚ, refusing if it genuinely carries a surd —
    a volume outside ℚ[π] is not representable and must not be floated."""
    f = _as_fraction(x)
    if f is None:
        raise ValueError(
            f"{what} volume term left ℚ[π] ({x!r}) — a bigger number field "
            "arrives with K3.7")
    return f


def _pi_value(rat, pi, what: str):
    """A volume term ``rat + pi·π``, in the smallest field that holds it.

    Returns a ``PiVal`` when both parts are rational — which keeps every
    existing answer byte-identical — and a ``PiPoly`` over ℚ[√d] when a part
    genuinely carries a surd. A trimmed band ending on a twelfth rather than a
    quarter is exactly that case: sin and cos bring in √3/2, so the π
    coefficient leaves ℚ while staying perfectly exact.

    Refusing here (which is what ``_exact`` alone did) would mean the arithmetic
    could represent the answer and the plumbing threw it away.
    """
    from forgekernel.polypi import PiPoly
    from forgekernel.quadric import PiVal

    fr, fp = _as_fraction(rat), _as_fraction(pi)
    if fr is not None and fp is not None:
        return PiVal(fr, fp)
    for part, name in ((rat, "offset"), (pi, "sweep")):
        if _as_fraction(part) is None and not hasattr(part, "d"):
            raise ValueError(
                f"{what} {name} volume term left ℚ[√d][π] ({part!r}) — a "
                "bigger number field arrives with K3.1")
    return PiPoly([rat, pi])


def _rational_sqrt(x):
    """√x as a Fraction when it is one, else None (the caller then refuses)."""
    from math import isqrt

    x = _as_fraction(x)
    if x is None or x < 0:
        return None
    n, d = x.numerator, x.denominator
    rn, rd = isqrt(n), isqrt(d)
    return Fraction(rn, rd) if rn * rn == n and rd * rd == d else None


def volume(body: Body):
    """Exact signed volume by the divergence theorem, V = (1/3)∮ x·n̂ dA,
    evaluated per face in CLOSED FORM. Returns a ``PiVal`` (ℚ + ℚ·π)."""
    from forgekernel.polypi import PiPoly
    from forgekernel.quadric import PiVal

    total = PiPoly.rational(0)
    for face in body.faces:
        term = _face_volume_term(face)
        total = total + (term if isinstance(term, PiPoly)
                         else PiPoly.from_pival(term))
    # Hand back the legacy a + b*pi whenever the answer fits in it: every
    # existing caller and test compares against PiVal, and a torus is the only
    # thing that needs the wider ring.
    if total.degree <= 1:
        return PiVal(total[0], total[1])
    return total


def _face_volume_term(face: Face):
    """(1/3)∮ x·n̂ dA over one face — exact, per surface type."""
    from forgekernel.quadric import PiVal

    s = face.surface
    sgn = 1 if face.sense else -1
    if isinstance(s, Plane):
        # x·n̂ = d/|n| is CONSTANT on the plane, so the term is (1/3)(d/|n|)·Area.
        # For a polygon loop, Area = ((Σ vᵢ × vᵢ₊₁)·n)/(2|n|), and the two |n|
        # combine into n·n — entirely rational, no square root needed.
        nn = dot(s.n, s.n)
        rat, pi = F(0), F(0)
        area_rat, area_pi = F(0), F(0)     # for the sanity check below
        for i, lp in enumerate(face.loops):
            circ = _loop_is_circle(lp)
            if circ is None and _loop_has_arcs(lp):
                # A loop mixing arcs and lines used to refuse outright, because
                # chording it under-reports the area. It has an exact area —
                # chord polygon plus one circular-segment term per arc — and
                # #123's straddled bore is built entirely out of such loops.
                # The refusal survives where it is still right: an arc off the
                # twelfth grid leaves every field the kernel has, and
                # _arc_quarters says so.
                ln = _rational_sqrt(nn)
                if ln is None:
                    raise ValueError(
                        "arc/line loop on a plane whose normal has irrational "
                        "length — outside ℚ[π] (arrives with K3.7)")
                a_rat, a_pi = (_mixed_loop_area(lp, s.n) if i == 0
                               else _mixed_loop_area_as_hole(lp, s.n))
                rat += s.d * a_rat / (3 * ln)
                pi += s.d * a_pi / (3 * ln)
                area_rat += a_rat
                area_pi += a_pi
            elif circ is None:
                a2 = _planar_loop_area2(lp, s.n)
                rat += s.d * (a2 if i == 0 else -abs(a2)) / (6 * nn)
                ln2 = _rational_sqrt(nn)
                if ln2 is not None:
                    area_rat += (a2 if i == 0 else -abs(a2)) / (2 * ln2)
            else:
                # a circular loop has area π r², which needs |n| on its own
                ln = _rational_sqrt(nn)
                if ln is None:
                    raise ValueError(
                        "circular loop on a plane whose normal has irrational "
                        "length — outside ℚ[π] (arrives with K3.7)")
                pi += (1 if i == 0 else -1) * s.d * circ.r * circ.r / (3 * ln)
                area_pi += (1 if i == 0 else -1) * circ.r * circ.r
        # A face cannot have negative area. If it does, an inner loop was
        # attached to a face that does not contain it — refuse rather than
        # return a confidently wrong exact volume.
        #
        # Decided EXACTLY. It used to float both parts (before that, via
        # _as_fraction, which quietly reads a surd as zero — so a face carrying
        # an odd-twelfth arc, or a shelled cone's annular cap at radius
        # (28−√29)/5, would have had a term dropped from the very check that is
        # supposed to catch a sign error). PiPoly.sign() narrows an enclosure
        # of π until the answer is certain and takes surd coefficients, so the
        # sanity bound is now as exact as the value it bounds.
        from forgekernel.polypi import PiPoly

        if PiPoly([area_rat, area_pi]).sign() < 0:
            raise ValueError(
                "face has negative area — an inner loop is larger than the "
                "boundary carrying it (a hole attached to the wrong face)")
        # _pi_value, not _exact: an annular cap whose rim radius is a surd is
        # legitimate (the inward offset of a slanted profile edge), and a mixed
        # loop ending on an ODD twelfth has a √3/2 in its segment term, so the
        # area is in ℚ[√3][π] — exactly the field the trimmed-band branch below
        # already hands back as a PiPoly. Both parts rational still returns a
        # plain PiVal, byte for byte as before.
        return _pi_value(sgn * rat, sgn * pi, "planar face")
    if isinstance(s, Cylinder):
        # x·n̂ = p·N̂ + r on the surface, so ∮x·n̂ dA = r·h·(p·∫N̂dθ + r·Δθ).
        # Over a FULL band ∫N̂dθ vanishes and this is the familiar
        # (1/3)(2πr²h); over a TRIMMED one it does not, and the leftover term
        # is rational exactly when the arc ends on quarter turns — which is
        # what keeps a rounded box's edge fillets inside ℚ[π].
        h = _band_height(face, s)
        arcs = _band_arc(face)
        if arcs is None:
            # _pi_value, not _exact: an untrimmed band whose height is
            # irrational is legitimate now — a napkin ring's bore wall runs
            # between z = ±√(R²−r²). _exact enforces ℚ[π] and would refuse it.
            # The term itself is unchanged; only the field it is allowed to
            # land in is wider, and _pi_value still returns a plain PiVal when
            # the height happens to be rational, so nothing existing moves.
            return _pi_value(0, sgn * 2 * s.r * s.r * h / 3, "cylinder band")
        span, vint = _band_sweep(arcs, s)
        rat = sgn * s.r * h * dot(s.p, vint) / 3
        pi = sgn * s.r * h * s.r * Fraction(span, 6) / 3          # Δθ = span·π/6
        return _pi_value(rat, pi, "trimmed band")
    if isinstance(s, SphereS):
        # x·n̂ = r + c·n̂. Over the whole sphere ∮c·n̂ dA = 0, leaving
        # (1/3)r·4πr². Over one OCTANT neither term vanishes: the area is
        # πr²/2 and ∮n̂ dA is (πr²/4) times the octant's sign vector.
        span = (_sphere_pole_span(face, s) if s.pole is not None
                else _sphere_zone(face, s))
        if span is not None:
            # A ZONE between two z-parallel latitudes — two rims for a
            # napkin ring's band, or one rim plus a surviving pole for a
            # blind bore. On a sphere x·n̂ = r + c·n̂, so the term is
            # (1/3)(r·Area + c·∮n̂dA). Archimedes gives Area = 2πr·Δz
            # exactly — it depends only on the SLAB HEIGHT — and
            # ∮n̂dA = π((z_hi−c_z)² − (z_lo−c_z)²)·ẑ by the same projection.
            # The OFFSET term is identically zero for a zone symmetric about
            # the sphere's centre (every napkin ring), which is why its
            # absence was invisible until the blind bore arrived: a one-rim
            # face has ∮n̂dA = −πr_rim²·ẑ ≠ 0, and dropping it goes wrong by
            # exactly c_z·πr_rim²/3 the moment the solid leaves the origin.
            # Δz and the offset are generally in ℚ[√d] — precisely why this
            # needed F() widened first.
            zlo, zhi = span
            off = s.c[2] * ((zhi - s.c[2]) * (zhi - s.c[2])
                            - (zlo - s.c[2]) * (zlo - s.c[2]))
            return _pi_value(
                0, sgn * (2 * s.r * s.r * (zhi - zlo) + off) / 3,
                "sphere zone")
        oct_ = _sphere_octant(face)
        if oct_ is None:
            return PiVal(0, _exact(sgn * 4 * s.r ** 3 / 3, "sphere"))
        return PiVal(0, _exact(
            sgn * (dot(s.c, oct_) * s.r * s.r / 4 + s.r ** 3 / 2) / 3,
            "sphere octant"))
    if isinstance(s, Cone):
        # Every point of a cone satisfies (x − apex)·n̂ = 0, so x·n̂ = p·n̂ and
        # ∮x·n̂ dA = p·∮n̂ dA. The radial part of ∮n̂ dA cancels round the band,
        # leaving (p·d̂)·n̂_ax·Area — and n̂_ax carries 1/√(1+t²) while Area
        # carries the slant's √(1+t²), so the root CANCELS and the term stays
        # in ℚ[π]. No approximation is needed for a taper.
        (ua, ra), (ub, rb) = _cone_rims(face, s)
        ln = _rational_sqrt(dot(s.d, s.d))
        if ln is None:
            raise ValueError(
                "cone axis with irrational length — outside ℚ[π] (K3.7)")
        if ua < 0 < ub:
            raise ValueError(
                "a conical face spanning BOTH nappes is not one face — its "
                "two halves have opposite normals, so no single sign is "
                "right (K3.7)")
        su = 1 if ua + ub > 0 else -1
        nax_area = -su * s.tan_half * (ra + rb) * abs(ub - ua)
        # _pi_value, not _exact: offsetting a slanted profile edge inward keeps
        # the taper (the offset line is PARALLEL, so tan_half stays rational)
        # but slides the apex to an irrational height and the rims to
        # irrational radii. The term is unchanged; only the field it may land
        # in is wider, and a rational frustum still returns a plain PiVal.
        return _pi_value(0, sgn * dot(s.p, s.d) / ln * nax_area / 3,
                         "cone band")
    if isinstance(s, Torus):
        # x·n̂ = c·n̂ + R cos(phi) + a on the tube, and dA = a(R + a cos phi),
        # so the phi integral carries cos and cos^2 terms. At quarter turns
        # those evaluate to rationals plus rational*phi, and phi is a multiple
        # of pi/2 -- so the term lands in Q[pi] with a pi^2 piece. Over a WHOLE
        # tube it collapses to Pappus: (1/3)(6 pi^2 a^2 R) = 2 pi^2 a^2 R.
        from forgekernel.polypi import PiPoly

        k0, span = _torus_sweep(face, s)
        # integral over phi of (R cos + a)(R + a cos) d phi
        #   = R^2 sin + a R (phi/2 + sin2phi/4) + a R phi + a^2 sin
        sin_ = lambda k: (0, 1, 0, -1)[k % 4]
        cos_ = lambda k: (1, 0, -1, 0)[k % 4]
        d_sin = sin_(k0 + span) - sin_(k0)
        d_cos = cos_(k0 + span) - cos_(k0)
        d_sin2 = sin_(k0 + span) ** 2 - sin_(k0) ** 2
        rat = (s.R * s.R + s.a * s.a) * d_sin
        # phi terms: aR*(phi/2) + aR*phi = (3/2) aR phi, phi = span*pi/2
        phi_coeff = Fraction(3, 2) * s.a * s.R * Fraction(span, 2)
        inner = PiPoly([rat, phi_coeff])            # rational + rational*pi
        total = PiPoly.term(2 * s.a, 1) * inner / 3
        # ...plus the axis-OFFSET term. Over a whole tube it vanishes, which is
        # why Pappus alone looks sufficient; over a swept sector it does not,
        # and a fillet's torus is centred on the core corner rather than the
        # origin, so dropping it would be wrong for every real blend.
        ln = _rational_sqrt(dot(s.d, s.d))
        if ln is None:
            raise ValueError(
                "torus axis with irrational length — outside ℚ[π] (K3.7)")
        # _exact() is what every other surface term uses: after a rigid map
        # these coordinates are SurdVal (with a zero surd part), and handing
        # one to Fraction() is a bare TypeError rather than a refusal
        cd = _exact(dot(s.c, s.d) / ln, "torus axis offset")
        offset = 2 * s.a * cd * (-s.R * d_cos + s.a * Fraction(d_sin2, 2)) / 3
        return (total + PiPoly.term(offset, 1)) * sgn
    raise ValueError(
        f"no exact volume term for {type(s).__name__} yet (K3.7)")


def _torus_sweep(face: Face, tor: Torus):
    """(start quarter, quarter span) of the tube's CROSS-SECTION sweep.

    Unlike a trimmed cylinder, this cannot be read off the face's edges: a
    torus band's rims are full circles of revolution, and the sector that the
    tube's cross-section sweeps is a different angle entirely. It is carried
    on the surface instead — the trim is part of what the surface IS here, not
    a property of its boundary.
    """
    if tor.span < 1 or tor.span > 4:
        raise ValueError("a torus sweeping outside one full turn (K3.7)")
    return tor.k0, tor.span


def _cone_rims(face: Face, cone: Cone):
    """The face's two rims as (axial distance from the APEX, radius), ordered.

    A true cone closes at its apex, where the rim degenerates to a point — so
    a face with a single circular rim is not malformed, it is a cone rather
    than a frustum, and the missing rim is (0, 0).
    """
    ln = _rational_sqrt(dot(cone.d, cone.d))
    if ln is None:
        raise ValueError("cone axis with irrational length — outside ℚ[π] (K3.7)")
    rims = []
    for lp in face.loops:
        for e in lp.edges:
            if isinstance(e.curve, Circle):
                u = dot(sub(e.curve.c, cone.p), cone.d) / ln
                rims.append((u, e.curve.r))
    rims = sorted(set(rims))
    if len(rims) == 1:
        rims = sorted(rims + [(F(0), F(0))])
    if len(rims) != 2:
        raise ValueError(
            "conical face without one or two circular rims — a trimmed cone "
            "needs general seam handling (K3.7)")
    return rims[0], rims[1]


def _quarter_ends(c: Circle, k0: int, span: int):
    """The two endpoints of the arc starting at twelfth k0 and sweeping span."""
    frame = _circle_frame(c)
    a, b = _twelfth_dir(c, k0, frame), _twelfth_dir(c, k0 + span, frame)
    if a is None or b is None:
        raise ValueError(
            "an arc endpoint at an odd twelfth of a rotated circle needs "
            "ℚ[√2, √3] — a bigger field arrives at K3.1")
    return (tuple(c.c[i] + c.r * a[i] for i in range(3)),
            tuple(c.c[i] + c.r * b[i] for i in range(3)))


def _band_sweep(arcs, cyl: Cylinder):
    """Total swept angle (in quarter turns) and ∫n̂ dθ for a trimmed band.

    Reading these off ONE arc was wrong twice over. A whole rim SPLIT into
    four quarter arcs — which survives a text round trip and which
    ``_loop_is_circle`` does not recognise unless every arc carries an
    identical ``Circle`` — looks like a trim, and taking the first arc read a
    Ø10×12 cylinder as 471.24 against a true 942.48. And a band whose rims are
    BOTH written the wrong way round passed the per-arc span check while
    reporting 3× its volume. Summing every arc on the chosen rim fixes both,
    and a sweep beyond a full turn is refused rather than wrapped.
    """
    fwd = [x for x in arcs if dot(x[0].n, cyl.d) > 0]
    if not fwd:
        raise ValueError(
            "no rim of this band runs with the axis — an Edge on a Circle is "
            "counter-clockwise about that circle's own normal, so a band's "
            "two rims must carry opposite normals (K3.7)")
    back = [x for x in arcs if x not in fwd]
    total, vint = 0, (F(0), F(0), F(0))
    for circ, k0, span in fwd:
        total += span
        v = sub(_quarter_antiderivative(circ, k0 + span),
                _quarter_antiderivative(circ, k0))
        vint = tuple(vint[i] + v[i] for i in range(3))
    if total > 12:
        raise ValueError("a rim sweeping more than a full turn (K3.7)")
    if back and sum(sp for _c, _k, sp in back) != total:
        raise ValueError(
            "a trimmed band whose two rims sweep different angles is not one "
            "face (K3.7)")
    return total, vint


def _sphere_zone(face: Face, s: "SphereS"):
    """(z_lo, z_hi) of a spherical ZONE bounded by two z-parallel circular
    rims, or None.

    Deliberately narrow: exactly two full-circle rims, both perpendicular to z,
    both actually on the sphere. Anything else falls through to the octant path
    and its refusal, because a zone term applied to a patch that is not a zone
    would return a plausible wrong number — and Archimedes' 2πrΔz is seductive
    enough to be applied too widely.

    ONE AMBIGUITY IS NOT RESOLVED HERE, and must not be papered over by anyone
    widening this. Two rims bound a sphere in two ways: the BAND between them
    and the pair of polar caps outside them. The b-rep carries no flag
    distinguishing the two ON A TWO-RIM FACE, so this function assumes the
    band — which is what `from_napkin_ring` builds and the only thing in the
    kernel that reaches here. If a caps-shaped face ever needs representing,
    it needs a representational change (a trim domain), not a looser
    predicate: the term for the caps is 4πr³/3 − 2πr²Δz/3 and nothing
    structural would catch the substitution.

    The ONE-RIM edition of that ambiguity (the cap above the rim versus the
    pole-containing remainder below it) IS resolved, by exactly such a
    representational change: `SphereS.pole` carries the trim on the surface
    (the Torus k0/span precedent), and `_sphere_pole_span` reads it. A
    one-rim face WITHOUT the flag still refuses through the octant path.
    """
    if len(face.loops) != 1:
        return None
    edges = list(face.loops[0].edges)
    if len(edges) != 2:
        return None
    zs = []
    for e in edges:
        c = e.curve
        if not isinstance(c, Circle) or e.v0 != e.v1:
            return None                      # not a full circle
        if c.n[0] != 0 or c.n[1] != 0:
            return None                      # rim not perpendicular to z
        if c.c[0] != s.c[0] or c.c[1] != s.c[1]:
            return None                      # not coaxial with the sphere
        dz = c.c[2] - s.c[2]
        if dz * dz + c.r * c.r != s.r * s.r:
            return None                      # rim does not lie ON the sphere
        zs.append(c.c[2])
    hi, lo = (zs[0], zs[1]) if _gt(zs[0], zs[1]) else (zs[1], zs[0])
    return lo, hi


def _sphere_pole_span(face: Face, s: "SphereS"):
    """(z_lo, z_hi) of a spherical face bounded by ONE z-parallel rim, where
    ``s.pole`` says which pole the face contains. Raises on anything else —
    a malformed pole-trimmed face must refuse, never fall back to a guess.

    The pole itself is a singular point on no edge, exactly like a pointed
    cone's apex (`_cone_rim_count`), so the face's single edge is the rim —
    and so `manifold_violations` needs no exemption: the rim pairs with the
    bore wall and nothing dangles.
    """
    if s.pole is None:
        raise ValueError("not a pole-trimmed sphere face")
    if s.pole[0] != 0 or s.pole[1] != 0 or s.pole[2] == 0:
        raise ValueError(
            "a pole-trimmed sphere face whose pole is off the z axis is "
            "outside what this term computes (K3.7)")
    if len(face.loops) != 1 or len(face.loops[0].edges) != 1:
        raise ValueError(
            "a pole-trimmed sphere face must carry exactly one rim (K3.7)")
    e = face.loops[0].edges[0]
    c = e.curve
    if (not isinstance(c, Circle) or e.v0 != e.v1
            or c.n[0] != 0 or c.n[1] != 0
            or c.c[0] != s.c[0] or c.c[1] != s.c[1] or c.r == 0):
        raise ValueError(
            "a pole-trimmed sphere face needs one full circular rim, "
            "coaxial with the sphere and of positive radius (K3.7)")
    dz = c.c[2] - s.c[2]
    if dz * dz + c.r * c.r != s.r * s.r:
        raise ValueError(
            "the rim of a pole-trimmed sphere face does not lie on the "
            "sphere (K3.7)")
    if _gt(0 * s.pole[2], s.pole[2]):                  # pole points down
        return s.c[2] - s.r, c.c[2]
    return c.c[2], s.c[2] + s.r


def _gt(a, b) -> bool:
    """a > b, EXACTLY, whichever exact field the two live in (ADR-0019).

    A rim's z is a SurdVal on a napkin ring, and `SurdVal` decides its own sign
    by an integer-square-root enclosure. Ordering by `float(a) > float(b)`
    would be a float deciding which rim is the top one — the charter forbids
    it, and it is wrong for two rims closer together than a double can resolve.
    """
    d = a - b
    return d.sign() > 0 if hasattr(d, "sign") else d > 0


def _sphere_octant(face: Face):
    """The octant's outward sign vector, or None for a whole sphere.

    A corner patch of a rounded box is bounded by three quarter arcs whose
    shared corners are the sphere centre plus r along each signed axis, so
    summing those corners recovers the octant directly — exactly, with no
    angle arithmetic at all.
    """
    if not face.loops:
        return None
    pts = [e.v0 for lp in face.loops for e in lp.edges]
    if len(pts) != 3:
        raise ValueError(
            "a spherical patch that is not a whole sphere or an octant is "
            "outside ℚ[π] (K3.7)")
    c, r = face.surface.c, face.surface.r
    vs = [sub(q, c) for q in pts]
    # Summing the corners and reading off signs accepted 143 different trios of
    # rational unit vectors that are NOT the signed axes — one of them measured
    # 2.68x its true term. An octant is three mutually PERPENDICULAR radii;
    # test that directly, which is also frame-free, so a rotated corner works.
    for i in range(3):
        if dot(vs[i], vs[i]) != r * r or dot(vs[i], vs[(i + 1) % 3]) != 0:
            raise ValueError(
                "spherical patch corners are not three mutually perpendicular "
                "radii — not an octant (K3.7)")
    return tuple(sum(v[i] for v in vs) / r for i in range(3))


def _band_height(face: Face, cyl: Cylinder) -> Fraction:
    """Axial extent of a cylindrical face, from its two circular rim loops."""
    ts = []
    for lp in face.loops:
        for e in lp.edges:
            if isinstance(e.curve, Circle):
                ts.append(dot(sub(e.curve.c, cyl.p), cyl.d))
    if len(ts) < 2:
        raise ValueError("cylindrical face without two circular rims (K3.7)")
    # t = (c−p)·d is the axial offset scaled by |d| (NOT |d|²), so the true
    # height divides by |d| once. A non-unit axis arises after scaling.
    ln = _rational_sqrt(dot(cyl.d, cyl.d))
    if ln is None:
        raise ValueError(
            "cylinder axis with irrational length — outside ℚ[π] (K3.7)")
    return (max(ts) - min(ts)) / ln


def _stitch_cracks(verts, tris, rounds: int = 40):
    """Split any triangle edge that another vertex lies on. Returns triangles.

    Exact T-junction splitting at the FACE level is necessary but not
    sufficient: two faces that share a seam are triangulated independently, in
    their own 2D frames, and the mesher is free to route the seam differently
    on each side. The result is a hairline crack — the volume is right, so
    nothing notices until an STL will not print.

    This is the mesh-level counterpart, and it is deliberately the LAST word:
    whatever the mesher decided, an edge with a vertex sitting on it gets
    split. Floats and a tolerance are legal here — meshing is a display
    property (ADR-0019) and no topology decision rides on it.
    """
    for _ in range(rounds):
        use: dict = {}
        for t in tris:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                use[tuple(sorted(e))] = use.get(tuple(sorted(e)), 0) + 1
        open_edges = {e for e, n in use.items() if n != 2}
        if not open_edges:
            break
        # only vertices touching an open edge can heal one
        live = sorted({i for e in open_edges for i in e})
        out, healed = [], False
        for t in tris:
            best = None
            for k in range(3):
                ia, ib = t[k], t[(k + 1) % 3]
                if tuple(sorted((ia, ib))) not in open_edges:
                    continue
                a, b = verts[ia], verts[ib]
                ab = [b[i] - a[i] for i in range(3)]
                den = sum(x * x for x in ab)
                if den == 0:
                    continue
                for iv in live:
                    if iv in t:
                        continue
                    p = verts[iv]
                    s = sum((p[i] - a[i]) * ab[i] for i in range(3)) / den
                    if not 1e-9 < s < 1 - 1e-9:
                        continue
                    d2 = sum((p[i] - a[i] - s * ab[i]) ** 2 for i in range(3))
                    if d2 <= 1e-18 * den:
                        best = (k, iv)
                        break
                if best:
                    break
            if best is None:
                out.append(t)
                continue
            k, iv = best
            x, y, z = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            out.extend(([x, iv, z], [iv, y, z]))
            healed = True
        tris = out
        if not healed:
            break
    return tris


def centroid(body: Body):
    """Centre of mass, by the same per-face decomposition the volume uses.

    Every face spans a cone back to the origin. That cone's volume is
    ``(1/3)∮ x·n̂ dA`` — which is exactly what :func:`volume` sums — and its
    first moment is ``(1/4)∮ x (x·n̂) dA``, closed-form per surface type. Both
    decompositions are over the SAME cones, so the terms may be summed
    together; mixing in a per-face form from a different decomposition (the
    tempting "a planar cone's centroid is ¾ of the face's") silently puts a
    cylinder's contribution in the wrong place — a Ø10×12 cylinder comes back
    with its centre of mass at z=3 instead of z=6.

    A centre of mass is a RATIO of ℚ[π] quantities and so leaves the field
    (ADR-0019); floats are the honest boundary here, exactly as forge's other
    ``centroid_f`` do. The volume it divides by stays exact.
    """
    m = [0.0, 0.0, 0.0]
    for f in body.faces:
        s = f.surface
        sgn = 1.0 if f.sense else -1.0
        if isinstance(s, Plane):
            nf = _unit(tuple(float(x) for x in s.n))
            acc, area = [0.0, 0.0, 0.0], 0.0
            for i, lp in enumerate(f.loops):
                circ = _loop_is_circle(lp)
                if circ is not None:
                    a = math.pi * float(circ.r) ** 2
                    c = tuple(float(x) for x in circ.c)
                elif _loop_has_arcs(lp):
                    a, c = _mixed_loop_area_centroid_f(lp, nf)
                    a = abs(a)
                else:
                    vs = [tuple(float(x) for x in e.v0) for e in lp.edges]
                    a, c = _poly_area_centroid_f(vs, nf)
                    a = abs(a)
                w = a if i == 0 else -a
                area += w
                for k in range(3):
                    acc[k] += c[k] * w
            # x·n̂ is the constant plane offset, so ∮x(x·n̂)dA = h·∫x dA
            h = sum(nf[k] * float(x) for k, x in enumerate(_plane_point(s)))
            for k in range(3):
                m[k] += sgn * 0.25 * h * acc[k]
        elif isinstance(s, Cylinder):
            h = float(_band_height(f, s))
            d = _unit(tuple(float(x) for x in s.d))
            q = tuple(float(x) for x in s.p)
            ts = [sum((float(e.curve.c[k]) - q[k]) * d[k] for k in range(3))
                  for lp in f.loops for e in lp.edges
                  if isinstance(e.curve, Circle)]
            t = (min(ts) + max(ts)) / 2
            mid = tuple(q[k] + t * d[k] for k in range(3))       # axis midpoint
            md = sum(mid[k] * d[k] for k in range(3))
            rr = float(s.r)
            arcs = _band_arc(f)
            if arcs is None:
                A, vn = 2 * math.pi, (0.0, 0.0, 0.0)
                mm = tuple(math.pi * (mid[k] - md * d[k]) for k in range(3))
            else:
                # A TRIMMED band sweeps less than a full turn, so neither
                # ∫n̂dθ nor ∫n̂(n̂·p)dθ vanishes. Both are rational at quarter
                # turns: ∫cos² = ∫sin² = Δθ/2 and ∫cos·sin = (sin²θ1−sin²θ0)/2.
                span, vint = _band_sweep(arcs, s)
                A = span * math.pi / 6                  # Δθ = span·π/6
                vn = tuple(float(x) for x in vint)
                circ, k0, _sp = min(
                    (x for x in arcs if dot(x[0].n, s.d) > 0), key=lambda x: x[1])
                u, w = (tuple(float(x) for x in v)
                        for v in _circle_frame(circ))
                # sin²(k·π/6), for ∫cos·sin dθ = (sin²θ₁ − sin²θ₀)/2. At
                # quarters this was (0, 1, 0, 1); the twelfths in between are
                # 1/4 and 3/4, and leaving them out put a rounded box's centre
                # of mass 2.09 mm from its own centre.
                sq = lambda kk: (0.0, .25, .75, 1.0, .75, .25,
                                 0.0, .25, .75, 1.0, .75, .25)[kk % 12]
                S = (sq(k0 + span) - sq(k0)) / 2
                pu = sum(mid[i] * u[i] for i in range(3))
                pw = sum(mid[i] * w[i] for i in range(3))
                mm = tuple((A / 2) * (mid[k] - md * d[k])
                           + S * (u[k] * pw + w[k] * pu) for k in range(3))
            pv = sum(mid[k] * vn[k] for k in range(3))
            for k in range(3):
                val = rr * h * ((mid[k]) * (pv + rr * A)
                                + rr * (mm[k] + rr * vn[k]))
                m[k] += sgn * 0.25 * val
        elif isinstance(s, SphereS):
            rr = float(s.r)
            cc = tuple(float(x) for x in s.c)
            oct_ = _sphere_octant(f)
            if oct_ is None:
                v = 4 / 3 * math.pi * rr ** 3
                for k in range(3):
                    m[k] += sgn * v * cc[k]
            else:
                # one OCTANT, not a whole sphere: area πr²/2, ∮n̂dA = (πr²/4)s,
                # ∫n_i n_j dA = πr²/6 on the diagonal and s_i s_j r²/3 off it
                sv = [float(x) for x in oct_]
                cs = sum(cc[k] * sv[k] for k in range(3))
                for k in range(3):
                    mk = math.pi * rr * rr / 6 * cc[k] + sum(
                        sv[k] * sv[j] * rr * rr / 3 * cc[j]
                        for j in range(3) if j != k)
                    val = (cc[k] * math.pi * rr * rr / 4 * cs
                           + rr * cc[k] * math.pi * rr * rr / 2
                           + rr * mk
                           + rr * rr * math.pi * rr * rr / 4 * sv[k] / rr)
                    m[k] += sgn * 0.25 * val
        elif isinstance(s, Cone):
            # (1/4)∮x(x·n̂)dA with x·n̂ = p·n̂. Integrating round φ kills the
            # lone-N̂ terms and turns N̂N̂ᵀ into π(I − d̂d̂ᵀ), leaving an integral
            # in u (axial distance from the apex) with ρ = t·u.
            (ua, ra), (ub, rb) = _cone_rims(f, s)
            d = _unit(tuple(float(x) for x in s.d))
            p = tuple(float(x) for x in s.p)
            pd = sum(p[k] * d[k] for k in range(3))
            t = float(s.tan_half)
            a, b2 = float(ua), float(ub)
            i2 = (b2 ** 2 - a ** 2) / 2
            i3 = (b2 ** 3 - a ** 3) / 3
            for k in range(3):
                perp = p[k] - pd * d[k]
                val = (-2 * math.pi * t * t * pd * (p[k] * i2 + d[k] * i3)
                       + math.pi * t * t * perp * i3)
                m[k] += sgn * 0.25 * val
        elif isinstance(s, Torus):
            # (1/4)∮x(x·n̂)dA. Integrating round θ first kills every lone N̂ and
            # turns N̂N̂ᵀ into π(I − d̂d̂ᵀ), leaving trig monomials in φ of degree
            # at most three — each with an exact antiderivative, so the FORMULA
            # is closed even though the arithmetic here is float (a centre of
            # mass is a ratio and leaves the field anyway).
            k0, span = _torus_sweep(f, s)
            RR, aa = float(s.R), float(s.a)
            cc = tuple(float(x) for x in s.c)
            ax = _unit(tuple(float(x) for x in s.d))
            S = sum(cc[t] * ax[t] for t in range(3))
            cperp = tuple(cc[t] - S * ax[t] for t in range(3))
            p0 = k0 * math.pi / 2
            p1 = (k0 + span) * math.pi / 2

            def I(f_anti):
                return f_anti(p1) - f_anti(p0)

            i_1 = I(lambda x: x)
            i_c = I(math.sin)
            i_s = I(lambda x: -math.cos(x))
            i_cc = I(lambda x: x / 2 + math.sin(2 * x) / 4)
            i_ss = I(lambda x: x / 2 - math.sin(2 * x) / 4)
            i_sc = I(lambda x: math.sin(x) ** 2 / 2)
            i_scc = I(lambda x: -math.cos(x) ** 3 / 3)
            i_css = I(lambda x: math.sin(x) ** 3 / 3)
            i_ccc = I(lambda x: math.sin(x) - math.sin(x) ** 3 / 3)

            rhoA = RR * RR * i_c + aa * RR * i_1 + aa * RR * i_cc + aa * aa * i_c
            rho_s = RR * i_s + aa * i_sc
            rho_sA = (RR * RR * i_sc + aa * RR * i_s + aa * RR * i_scc
                      + aa * aa * i_sc)
            rho_ss = RR * i_ss + aa * i_css
            rho2c = RR * RR * i_c + 2 * aa * RR * i_cc + aa * aa * i_ccc
            for k in range(3):
                val = (2 * math.pi * aa * (rhoA * cc[k] + S * rho_s * cc[k])
                       + 2 * math.pi * aa * aa * ax[k] * (rho_sA + S * rho_ss)
                       + math.pi * aa * rho2c * cperp[k])
                m[k] += sgn * 0.25 * val
        else:
            raise ValueError(
                f"centroid of a {type(s).__name__} face is not implemented "
                "yet — arrives with K3.7")
    v = float(volume(body))
    if v == 0:
        raise ValueError("centroid of a zero-volume body is undefined")
    return tuple(x / v for x in m)


def vector_area(body: Body):
    """Σ over faces of Areaᵢ·n̂ᵢ — exactly zero for ANY closed shell.

    A third oracle, and it sees a class the other two structurally cannot.
    Edge pairing audits counts, so a face whose loop runs the wrong way round
    still pairs; the volume is a single number, so a sign error that happens to
    keep it positive sails through. The vector area is the divergence theorem
    applied to the constant fields x̂, ŷ, ẑ, so it must vanish on any closed
    surface whatever its shape — and it does not vanish when one face's
    orientation disagrees with its neighbours.

    Measured on the #123 counterbore result: the correct 13-face body gives
    (0,0,0); emitting the tool's fourth wall (the classic mistake, a
    self-intersecting shell) gives (4, 4, π); flipping the sign of one
    circular-segment term gives (0, 0, 2π).

    Returns a triple of ``PiPoly``, or ``None`` when the body carries a surface
    whose ∮n̂ dA this does not compute yet — an unknown answer is not a pass.
    """
    from forgekernel.polypi import PiPoly

    out = [PiPoly.rational(0)] * 3
    for f in body.faces:
        s, sgn = f.surface, (1 if f.sense else -1)
        if isinstance(s, Plane):
            ln = _rational_sqrt(dot(s.n, s.n))
            if ln is None:
                return None
            rat, pi = F(0), F(0)
            for i, lp in enumerate(f.loops):
                circ = _loop_is_circle(lp)
                if circ is not None:
                    pi += (1 if i == 0 else -1) * circ.r * circ.r
                elif _loop_has_arcs(lp):
                    a = (_mixed_loop_area(lp, s.n) if i == 0
                         else _mixed_loop_area_as_hole(lp, s.n))
                    rat, pi = rat + a[0], pi + a[1]
                else:
                    a2 = _planar_loop_area2(lp, s.n)
                    rat += (a2 if i == 0 else -abs(a2)) / (2 * ln)
            for i in range(3):
                out[i] = out[i] + PiPoly([rat, pi]) * sgn * (s.n[i] / ln)
        elif isinstance(s, Cylinder):
            # ∮n̂ dA over a band is r·h·∫N̂dθ, which vanishes over a full turn
            # and does not over a trimmed one
            arcs = _band_arc(f)
            if arcs is None:
                continue
            h = _band_height(f, s)
            vint = _band_sweep(arcs, s)[1]
            for i in range(3):
                out[i] = out[i] + PiPoly([h * s.r * vint[i] * sgn, F(0)])
        else:
            return None
    return tuple(out)


def _edge_key(e: Edge):
    """A geometric, EXACT, direction-free name for an edge.

    Direction-free because the two faces meeting at an edge traverse it
    opposite ways; exact because an edge key that rounded would fuse two
    genuinely distinct edges a micron apart (ADR-0019).

    It also carries the arc's SPAN, and that is not cosmetic. Two arcs on one
    circle with the same endpoints are the same set only if they sweep the same
    angle: the 90° arc E→N and the 270° arc N→E share a centre, a radius, an
    axis and an endpoint pair, so without the span they keyed identically and
    ``manifold_violations`` — the audit that is supposed to stop a hand-built
    body lying about what it is — could not see a complemented rim at all. A
    quarter cylinder (5π) and the 3/4 wedge glued to its flat walls (35π/3) are
    built from the very same faces and both came back clean.

    Direction-freeness survives, because the span does not depend on which side
    walks the edge: a face needing the reverse sweep carries the circle with −n
    (the ``Edge`` contract in ``_arc_pts``), and the same arc measured about −n
    from the other end has the same span. A CLOSED edge (v0 == v1) is the whole
    circle from either side and is left alone.
    """
    from forgekernel.brep import _canon_dir

    ends = tuple(sorted((_key3(e.v0), _key3(e.v1)), key=repr))
    if isinstance(e.curve, Circle):
        # sign-invariant axis: a band's rim carries the unit axis while the
        # octant beside it carries a cross product pointing the other way
        try:
            span = (12 if _key3(e.v0) == _key3(e.v1)
                    else _arc_quarters(e.curve, e.v0, e.v1)[1])
        except ValueError:
            # an arc off the twelfth grid, or a circle whose frame is not
            # exactly orthonormal: no exact span exists, so key it as before.
            # Both faces at the edge compute the same thing, so pairing is
            # unaffected; such a body cannot be measured exactly anyway.
            span = None
        # ends stays LAST: manifold_violations names the offending edge with
        # k[-1][0], so every key must end with its endpoint pair
        return ("arc", _key3(e.curve.c), e.curve.r,
                _canon_dir(e.curve.n), span, ends)
    return ("line", ends)


def manifold_violations(body: Body) -> list[str]:
    """Every edge of a closed shell is shared by exactly two faces.

    This is the check ``validate`` had no way to run on a ``Body``: it called
    ``Solid.watertight_violations``, which a Body does not have, so every
    pocketed or curved-shelled solid came back as "unsupported representation"
    — unvalidatable, which reads far too much like fine.

    It audits COUNTS, not directions. A Body stores no traversal direction for
    a full-circle edge (the writer derives it from the surface normal and the
    loop's role), so a direction audit belongs where that derivation happens —
    ``write_step_body`` does exactly that before it emits. Counts alone still
    catch the whole family this kernel has actually produced: a rim minted
    twice, a T-junction left unsplit, a face dropped, an arc unpaired.
    """
    seen: dict = {}
    for f in body.faces:
        for lp in f.loops:
            for e in lp.edges:
                k = _edge_key(e)
                seen[k] = seen.get(k, 0) + 1
    bad = []
    for k, n in sorted(seen.items(), key=repr):
        if n != 2:
            bad.append(f"edge {k[0]} at {k[-1][0]} used by {n} face(s), not 2")
    return bad


def _cone_rim_count(face: Face) -> int:
    """How many circular rims bound a conical face — 2 for a frustum, 1 for a
    pointed cone (the apex is a singular point on no edge)."""
    return sum(1 for lp in face.loops for e in lp.edges
               if isinstance(e.curve, Circle))


def bbox(body: Body):
    """Float bbox of the body (a bound, not a topological decision)."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for f in body.faces:
        if isinstance(f.surface, Torus):
            # A torus carries no loops either, so it contributed NOTHING. When
            # a fillet eats the whole adjacent face the extreme lives only on
            # the torus and was lost — and part.interference uses the AABB as
            # a pre-filter, so two filleted barrels that physically interlock
            # by 1.5 mm came back as a clean bill of health.
            t = f.surface
            k0, span = _torus_sweep(f, t)
            ks = range(k0, k0 + span + 1)
            cs = [(1, 0, -1, 0)[k % 4] for k in ks]
            sn = [(0, 1, 0, -1)[k % 4] for k in ks]
            ax = _unit(tuple(float(x) for x in t.d))
            cen = tuple(float(x) for x in t.c)
            RR, aa = float(t.R), float(t.a)
            radmax = RR + aa * max(cs)
            for i in range(3):
                perp = radmax * math.sqrt(max(0.0, 1 - ax[i] ** 2))
                axl = [aa * v * ax[i] for v in (min(sn), max(sn))]
                lo[i] = min(lo[i], cen[i] - perp + min(axl))
                hi[i] = max(hi[i], cen[i] + perp + max(axl))
            continue
        if isinstance(f.surface, SphereS):      # a whole sphere carries no loops
            c, r = f.surface.c, float(f.surface.r)
            for i in range(3):
                lo[i] = min(lo[i], float(c[i]) - r)
                hi[i] = max(hi[i], float(c[i]) + r)
            continue
        if isinstance(f.surface, Cone) and _cone_rim_count(f) < 2:
            # A POINTED cone's apex is a singular point lying on no edge, so
            # the walk below never visits it and the box collapsed along the
            # axis: scale(cone(6,0,10), 2) reported a z-extent of 0.0 for a
            # 20 mm solid. Third instance of one pattern, after Torus and
            # SphereS: the extreme is not on any edge.
            #
            # Two RIMS means the cone is truncated at both ends and the apex is
            # not part of the face, so it must NOT be added — a bound may be
            # loose, but needlessly loosening every frustum blunts the AABB
            # pre-filter that part.interference depends on.
            #
            # Counting rims, not LOOPS: every frustum this kernel builds puts
            # both rims in ONE loop (from_cone and from_revolve both do), so
            # `len(f.loops) < 2` was true for all of them and every frustum
            # body reported its box out to the apex — cone(6,2,10) came back
            # 15 mm tall instead of 10. Sound, so no invariant caught it, and
            # 50% loose on a shape whose whole point is the taper.
            ap = f.surface.p
            for i in range(3):
                lo[i] = min(lo[i], float(ap[i]))
                hi[i] = max(hi[i], float(ap[i]))
            # fall through: the base circle still contributes its own extent
        for lp in f.loops:
            circ = _loop_is_circle(lp)
            if circ is not None:
                c, r = circ.c, float(circ.r)
                # a circle's extent per axis: r·sqrt(1 − (n_i/|n|)²)
                n = circ.n
                nn = math.sqrt(float(dot(n, n))) or 1.0
                for i in range(3):
                    ext = r * math.sqrt(max(0.0, 1 - (float(n[i]) / nn) ** 2))
                    lo[i] = min(lo[i], float(c[i]) - ext)
                    hi[i] = max(hi[i], float(c[i]) + ext)
                continue
            for e in lp.edges:
                if isinstance(e.curve, Circle):
                    # an arc bulges past its endpoints; its whole circle is a
                    # sound (if loose) bound, and a bound is all this promises
                    c, r = e.curve.c, float(e.curve.r)
                    nn = math.sqrt(float(dot(e.curve.n, e.curve.n))) or 1.0
                    for i in range(3):
                        ext = r * math.sqrt(
                            max(0.0, 1 - (float(e.curve.n[i]) / nn) ** 2))
                        lo[i] = min(lo[i], float(c[i]) - ext)
                        hi[i] = max(hi[i], float(c[i]) + ext)
                    continue
                for v in (e.v0, e.v1):
                    for i in range(3):
                        lo[i] = min(lo[i], float(v[i]))
                        hi[i] = max(hi[i], float(v[i]))
    return (tuple(lo), tuple(hi))


def tessellate(body: Body, deflection: float = 0.2) -> dict:
    """One display mesh for the canonical form — planar faces are triangulated
    (with circular holes cut out), cylindrical faces become bands, spheres a UV
    mesh. Floats are legal here: meshing is a display property (ADR-0019)."""
    from forgekernel.mesh2d import triangulate

    verts: list = []
    tris: list = []
    index: dict = {}

    def V(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in index:
            index[k] = len(verts)
            verts.append([float(p[0]), float(p[1]), float(p[2])])
        return index[k]

    def tri(a, b, c, outward):
        ia, ib, ic = V(a), V(b), V(c)
        if len({ia, ib, ic}) < 3:
            return
        n = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
             (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
             (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if n == (0.0, 0.0, 0.0):
            return                       # a collinear sliver carries no area
        if sum(n[i] * outward[i] for i in range(3)) < 0:
            ib, ic = ic, ib
        tris.append([ia, ib, ic])

    def _segs_for(r: float, defl: float | None = None) -> int:
        """Segments round a whole circle — always 4·2^d, and at least 32.

        DYADIC, and that is the whole point. Three samplers walk the same
        circles: ``circle_pts`` for a whole rim, the trimmed-band path, and a
        corner octant's midpoint subdivision. The last two are dyadic by
        construction (2^d steps per quarter), so a free-form count from
        ``ceil(π/acos(...))`` gave a full rim 24 points where the band beside it
        wanted 8 — a shell that tears wherever a trimmed band meets a whole
        circle on the same axis. Rounding the count up to 4·2^d puts all three
        on one grid, and rounding UP means no mesh gets coarser than it was.

        ``defl`` lets a caller spend less than the whole budget in this one
        direction. A quad's centroid sits off the true surface by roughly the
        SUM of its two directions' sagittas, so a patch sampled in two curved
        directions that spends the full deflection on each overshoots the
        caller's tolerance by about 2x — see the sphere-zone branch.
        """
        d_eff = deflection if defl is None else defl
        d = _quarter_depth(r, d_eff) if r > 0 else 1
        while 4 * 2 ** d < 24:
            d += 1
        return 4 * 2 ** d

    # ONE segment count per AXIS, taken from the widest circle on it. Deriving
    # it per circle tears every taper: a frustum's r=2 and r=5 rims want
    # different counts, so the wall and the caps stop sharing vertices. Coaxial
    # bores in a counterbore have the same problem.
    #
    # #137: an axis that carries a TWELFTH trim (a 30°/60° crossing from the
    # odd-twelfth straddling-bore family) gets a 12·2^d count instead of the
    # dyadic 4·2^d, so the trimmed band, the rim arcs and every whole circle
    # on that axis still land on one shared grid. The only sampler that
    # cannot follow is a dyadic corner OCTANT (2^d points per quarter — no
    # power of two divides into thirds), so the refusal narrows to the case
    # where an octant genuinely shares the axis, instead of refusing every
    # twelfth-trimmed body outright.
    axis_segs: dict = {}
    twelfth_axes: set = set()
    octant_axes: set = set()
    for f in body.faces:
        if isinstance(f.surface, Torus):
            # a torus shares its axis with the cylinders and disks it blends
            # between, so it has to sit in the SAME bucket or the fillet tears
            # away from the wall it is supposed to be tangent to
            k = _line_key(f.surface.c, f.surface.d)
            axis_segs[k] = max(axis_segs.get(k, 0),
                               _segs_for(float(f.surface.R + f.surface.a)))
        zone_span = None
        if isinstance(f.surface, SphereS):
            zone_span = (_sphere_pole_span(f, f.surface)
                         if f.surface.pole is not None
                         else _sphere_zone(f, f.surface))
        if zone_span is not None:
            # A spherical ZONE is WIDER than either of its rims — a napkin ring
            # bored at r=1 through a sphere of R=6 bulges to r=6 in between.
            # Taking the bucket from the rims alone (the loop below) meshed
            # that bulge as a 24-gon at every deflection: watertight, refining
            # in latitude, and 1.5% short on volume, which is exactly the shape
            # of a silent wrong answer. The count has to come from the WIDEST
            # circle the face actually spans, and it has to land in the same
            # bucket as the bore wall or the shell tears along both rims.
            # A pole-trimmed face (blind bore) spans rim to POLE, so its z
            # extent comes from the span, not from its single rim.
            #
            # Floats are legal here — this is meshing (ADR-0019).
            rims = [e.curve for lp in f.loops for e in lp.edges]
            cz = float(f.surface.c[2])
            zs = [float(z) for z in zone_span]
            rr = float(f.surface.r)
            widest = rr if zs[0] <= cz <= zs[1] else max(
                math.sqrt(max(0.0, rr * rr - (z - cz) ** 2)) for z in zs)
            k = _line_key(rims[0].c, rims[0].n)
            axis_segs[k] = max(axis_segs.get(k, 0),
                               _segs_for(widest, deflection / 2))
        is_octant = (isinstance(f.surface, SphereS) and f.loops
                     and f.surface.pole is None and zone_span is None)
        for lp in f.loops:
            for e in lp.edges:
                if isinstance(e.curve, Circle):
                    k = _line_key(e.curve.c, e.curve.n)
                    axis_segs[k] = max(axis_segs.get(k, 0),
                                       _segs_for(float(e.curve.r)))
                    if is_octant:
                        octant_axes.add(k)
                    if e.v0 != e.v1:
                        try:
                            a_k0, a_span = _arc_quarters(e.curve, e.v0, e.v1)
                        except ValueError:
                            pass                # off the twelfth grid entirely
                        else:
                            if a_k0 % 3 or a_span % 3:
                                twelfth_axes.add(k)
    for k in twelfth_axes:
        n = axis_segs.get(k, 0)
        if n and n % 12:
            # the next 12·2^d at or above the dyadic count — never coarser,
            # and every quarter and twelfth point lies on the same grid
            d = 1
            while 12 * 2 ** d < n:
                d += 1
            axis_segs[k] = 12 * 2 ** d

    def circle_pts(c: Circle, n=None):
        r = float(c.r)
        segs = n or axis_segs.get(_line_key(c.c, c.n)) or _segs_for(r)
        u = _unit(tuple(float(x) for x in c.ref))
        w = _unit(_cross_f(tuple(float(x) for x in c.n), u))
        cc = tuple(float(x) for x in c.c)
        return [tuple(cc[i] + r * (math.cos(2 * math.pi * k / segs) * u[i]
                                   + math.sin(2 * math.pi * k / segs) * w[i])
                      for i in range(3)) for k in range(segs)], segs

    def _per_quarter(c: Circle) -> int:
        """Sample points per QUARTER turn on this circle's axis — the axis
        bucket's count divided by four, so a trimmed band, a whole rim and a
        corner octant on one axis all land on the same points."""
        n = axis_segs.get(_line_key(c.c, c.n))
        if n:
            return n // 4
        return 2 ** _quarter_depth(float(c.r), deflection)

    def _grid_arc_pts(c: Circle, v0, v1):
        """One arc's points on the axis grid, v0 up to but excluding v1.

        Falls back to plain deflection sampling only where no grid exists to
        share — an arc off the twelfth grid, or one whose sweep is not a whole
        number of quarters, has no trimmed band beside it to agree with (that
        band refuses to mesh at all).
        """
        try:
            k0, span = _arc_quarters(c, v0, v1)
            u, w = (tuple(float(x) for x in q) for q in _circle_frame(c))
        except ValueError:
            return _arc_pts(c, v0, v1, deflection)
        if span % 3:
            # an odd-twelfth arc: only meshable on a shared grid when its
            # whole axis runs at 12·2^d (#137) — free sampling here would
            # tear against the trimmed band beside it
            n_axis = axis_segs.get(_line_key(c.c, c.n), 0)
            if not n_axis or n_axis % 12:
                return _arc_pts(c, v0, v1, deflection)
            nseg = (n_axis // 12) * span
        else:
            nseg = _per_quarter(c) * (span // 3)
        cc = tuple(float(x) for x in c.c)
        rr = float(c.r)
        return [tuple(cc[t] + rr * (math.cos((k0 + span * m / nseg) * math.pi / 6)
                                    * u[t]
                                    + math.sin((k0 + span * m / nseg) * math.pi / 6)
                                    * w[t]) for t in range(3))
                for m in range(nseg)]

    for face in body.faces:
        s = face.surface
        if isinstance(s, Plane):
            nf = _unit(tuple(float(x) for x in s.n))
            out = nf if face.sense else tuple(-x for x in nf)
            u = _unit(_perp_f(nf))
            w = _cross_f(nf, u)
            org = tuple(float(x) for x in _plane_point(s))

            def to2(p):
                q = tuple(p[i] - org[i] for i in range(3))
                return (sum(q[i] * u[i] for i in range(3)),
                        sum(q[i] * w[i] for i in range(3)))

            def to3(a):
                return tuple(org[i] + a[0] * u[i] + a[1] * w[i] for i in range(3))

            rings = []
            for lp in face.loops:
                circ = _loop_is_circle(lp)
                if circ is not None:
                    pts, _ = circle_pts(circ)
                    rings.append([to2(p) for p in pts])
                else:
                    ring = []
                    for e in lp.edges:
                        # a slot end is an ARC — walking v0 alone chords it
                        if isinstance(e.curve, Circle):
                            # ...on the axis GRID, not by free deflection. A
                            # notch floor's 90° arc is the same edge as the bore
                            # band's rim beside it; sampled independently the
                            # two disagreed and the shell tore along the seam.
                            ring.extend(_grid_arc_pts(e.curve, e.v0, e.v1))
                        else:
                            ring.append(tuple(float(x) for x in e.v0))
                    rings.append([to2(p) for p in ring])
            if not rings:
                continue
            # keep_collinear: the mid-edge vertices in these rings are
            # T-junction seams shared with a neighbouring face. Splitting them
            # exactly (_split_t_junctions) buys nothing if the mesher then
            # drops them on one side and keeps them on the other.
            pts2, t2 = triangulate(rings[0], rings[1:], keep_collinear=True)
            for a, b, c in t2:
                tri(to3(pts2[a]), to3(pts2[b]), to3(pts2[c]), out)
        elif isinstance(s, Cylinder) and _band_arc(face) is not None:
            # a TRIMMED band: sample only the swept sector, from the rim whose
            # normal runs with the axis (the other carries -n by contract)
            arcs = _band_arc(face)
            fwd = [x for x in arcs if dot(x[0].n, s.d) > 0] or arcs
            span, _v = _band_sweep(arcs, s)
            circ, k0, _sp = min(fwd, key=lambda x: x[1])
            u, w = (tuple(float(x) for x in v) for v in _circle_frame(circ))
            rr = float(s.r)
            # a trimmed band and the corner octants next to it SHARE their
            # quarter arcs, so both must land on the same points. Projecting a
            # chord midpoint onto a circle lands on the ANGULAR midpoint, so a
            # depth-d octant subdivision gives 2^d uniform steps per quarter —
            # match the band to that or the shell tears along every seam.
            per = _per_quarter(circ)
            if span % 3:
                # The exact arithmetic reaches twelfths (30° steps), and so
                # does the mesh now (#137): the axis bucket runs at 12·2^d
                # whenever a twelfth trim is present, so the band, the rim
                # arcs and every whole circle share one grid. The one sampler
                # that cannot follow is a dyadic corner OCTANT (2^d points per
                # quarter — no power of two divides into thirds), so only a
                # genuine octant on this axis still refuses.
                band_key = _line_key(circ.c, circ.n)
                n_axis = axis_segs.get(band_key, 0)
                if band_key in octant_axes or not n_axis or n_axis % 12:
                    raise ValueError(
                        f"a band spanning {span} twelfths of a turn cannot "
                        "share a mesh grid with dyadic corner octants — 30° "
                        "trims are exact but not meshable beside an octant "
                        "(K3.7)")
                nseg = (n_axis // 12) * span
            else:
                nseg = per * (span // 3)
            axis = _unit(tuple(float(x) for x in s.d))
            # every CIRCULAR rim, not only the trimmed arcs — the same source
            # _band_height reads. A band whose far rim is a whole circle while
            # the near one is split into quarters (which is what a bore wall
            # below a notch is) had ends == {one value}, so it meshed as a
            # zero-height strip and emitted nothing at all: 5 mm of bore wall
            # silently missing, and the rings above and below it left open.
            ends = sorted({sum((float(e.curve.c[t]) - float(s.p[t])) * axis[t]
                               for t in range(3))
                           for lp in face.loops for e in lp.edges
                           if isinstance(e.curve, Circle)})
            base = tuple(float(x) for x in s.p)
            def at(tt, ang):
                # k0 and ang are both in TWELFTHS of a turn, so the step is
                # π/6. It was π/2 while the unit was quarters, and leaving it
                # behind put every band a factor of three round the cylinder.
                th = (k0 + ang) * math.pi / 6
                return tuple(base[t] + tt * axis[t]
                             + rr * (math.cos(th) * u[t] + math.sin(th) * w[t])
                             for t in range(3))
            for m in range(nseg):
                a0, a1 = span * m / nseg, span * (m + 1) / nseg
                pa, pb = at(ends[0], a0), at(ends[0], a1)
                qa, qb = at(ends[-1], a0), at(ends[-1], a1)
                mid = tuple((pa[t] + pb[t]) / 2 for t in range(3))
                rel = tuple(mid[t] - base[t] for t in range(3))
                axl = sum(rel[t] * axis[t] for t in range(3))
                radial = _unit(tuple(rel[t] - axl * axis[t] for t in range(3)))
                outn = radial if face.sense else tuple(-x for x in radial)
                tri(pa, pb, qb, outn)
                tri(pa, qb, qa, outn)
        elif isinstance(s, Torus):
            k0, span = _torus_sweep(face, s)
            RR, aa = float(s.R), float(s.a)
            cc = tuple(float(x) for x in s.c)
            ax = _unit(tuple(float(x) for x in s.d))
            # take the reference direction from a COAXIAL circle, the same
            # source circle_pts uses. Building an independent frame agreed
            # unrotated (both (1,0,0)) and was 90 degrees out after a
            # z-rotation, interleaving the two rings and tearing the seam.
            ref = None
            for g in body.faces:
                for lp in g.loops:
                    for e in lp.edges:
                        if (isinstance(e.curve, Circle)
                                and _line_key(e.curve.c, e.curve.n)
                                == _line_key(s.c, s.d)):
                            ref = e.curve.ref
                            break
                    if ref is not None:
                        break
                if ref is not None:
                    break
            u0 = _unit(tuple(float(x) for x in ref)) if ref else _unit(_perp_f(ax))
            w0 = _cross_f(ax, u0)
            nth = axis_segs.get(_line_key(s.c, s.d)) or _segs_for(RR + aa)
            nph = max(4, 2 ** _quarter_depth(aa, deflection)) * span
            def pt(i, j):
                th = 2 * math.pi * i / nth
                ph = (k0 + span * j / nph) * math.pi / 2
                rad = RR + aa * math.cos(ph)
                N = tuple(math.cos(th) * u0[t] + math.sin(th) * w0[t]
                          for t in range(3))
                return tuple(cc[t] + rad * N[t] + aa * math.sin(ph) * ax[t]
                             for t in range(3)), N, ph
            for i in range(nth):
                for j in range(nph):
                    (pa, Na, pha) = pt(i, j)
                    (pb, _n, _p) = pt(i + 1, j)
                    (pc, _n, _p) = pt(i + 1, j + 1)
                    (pd, _n, _p) = pt(i, j + 1)
                    nrm = _unit(tuple(math.cos(pha) * Na[t]
                                      + math.sin(pha) * ax[t] for t in range(3)))
                    outn = nrm if face.sense else tuple(-x for x in nrm)
                    tri(pa, pb, pc, outn)
                    tri(pa, pc, pd, outn)
        elif isinstance(s, SphereS) and (
                s.pole is not None or _sphere_zone(face, s) is not None):
            # A spherical ZONE: the band of a sphere between two coaxial rims
            # (a through bore), or between one rim and a surviving POLE (a
            # blind bore — `SphereS.pole` carries which side). It is neither a
            # whole sphere nor an octant, so the octant path below rightly
            # refuses it — but refusing is a mesh that does not exist, and
            # #124 removed the fallback that quietly meshed the representation
            # instead (it discarded the caller's deflection).
            #
            # The ONLY thing watertightness depends on is that the boundary
            # rings land on exactly the vertices the bore wall already puts
            # there. So they are not recomputed from the sphere: they come from
            # `circle_pts` on the SAME Circle objects the wall meshes, at the
            # same per-axis segment count. The interior rings are free — they
            # are shared with nothing. A pole end is a single point: its
            # "ring" repeats it, the vertex indexer dedupes it, and the quad
            # loop's degenerate triangles collapse into a clean fan — the same
            # way a pointed cone's apex meshes.
            #
            # Floats are legal here — this is meshing (ADR-0019).
            rims = [e.curve for lp in face.loops for e in lp.edges]
            cc = tuple(float(x) for x in s.c)
            rr = float(s.r)
            if s.pole is not None:
                zlo_e, zhi_e = _sphere_pole_span(face, s)
                c_ref = rims[0]
                ring_rim, segs = circle_pts(c_ref)
                south = float(s.pole[2]) < 0
                pole_pt = (cc[0], cc[1], cc[2] + (-rr if south else rr))
                pole_ring = [pole_pt] * segs
                ring_lo, ring_hi = ((pole_ring, ring_rim) if south
                                    else (ring_rim, pole_ring))
                zlo, zhi = float(zlo_e), float(zhi_e)
            else:
                c_lo, c_hi = sorted(rims, key=lambda c: float(c.c[2]))
                c_ref = c_lo
                ring_lo, segs = circle_pts(c_lo)
                ring_hi, _ = circle_pts(c_hi, segs)
                zlo, zhi = float(c_lo.c[2]), float(c_hi.c[2])
            u = _unit(tuple(float(x) for x in c_ref.ref))
            w = _unit(_cross_f(tuple(float(x) for x in c_ref.n), u))
            # The step is in POLAR ANGLE, and it spends only HALF the
            # deflection — matching what the axis bucket was budgeted with
            # above. The zone is curved in both latitude and longitude and the
            # two chord errors add at a quad's centre. Measured on R=6 r=1 at
            # deflection 0.01: the full budget per direction put the worst
            # centroid 0.017 off the sphere, 1.7x what the caller asked for,
            # while the kernel's own plain-sphere mesher stays under it.
            half = deflection / 2
            step = 2 * math.acos(max(-1.0, min(1.0, 1 - half / rr))) \
                if rr > half else math.pi / 4
            f_lo = math.asin(max(-1.0, min(1.0, (zlo - cc[2]) / rr)))
            f_hi = math.asin(max(-1.0, min(1.0, (zhi - cc[2]) / rr)))
            nlat = max(2, int(math.ceil(abs(f_hi - f_lo) / step))
                       if step > 0 else 2)
            rings = [ring_lo]
            for j in range(1, nlat):
                # UNIFORM IN LATITUDE, not in z. Stepping z uniformly is
                # tempting — Archimedes says equal z slabs carry equal area —
                # but area is not the error metric, chord sagitta is, and near
                # a rim the surface is nearly vertical: a napkin ring bored at
                # r=1 through R=6 jumps from radius 1 to radius 2.5 in a single
                # uniform-z step. The mesh stayed watertight and refined on
                # request while running 0.6% under the exact volume, which is
                # precisely the failure mode that survives every structural
                # check. nlat is derived from the polar span, so the sampling
                # has to be in the same variable the count was budgeted in.
                ph = f_lo + (f_hi - f_lo) * j / nlat
                z = cc[2] + rr * math.sin(ph)
                rad = rr * math.cos(ph)
                # the ring's centre is on the sphere's axis at height z, and
                # `_sphere_zone` has already proved the rims are coaxial with
                # the sphere, so cc[0], cc[1] are the axis
                org = (cc[0], cc[1], z)
                rings.append([
                    tuple(org[t] + rad * (math.cos(2 * math.pi * m / segs) * u[t]
                                          + math.sin(2 * math.pi * m / segs) * w[t])
                          for t in range(3))
                    for m in range(segs)])
            rings.append(ring_hi)
            for j in range(nlat):
                a_ring, b_ring = rings[j], rings[j + 1]
                for m in range(segs):
                    m2 = (m + 1) % segs
                    pa, pb = a_ring[m], a_ring[m2]
                    qa, qb = b_ring[m], b_ring[m2]
                    mid = tuple((pa[t] + pb[t] + qa[t] + qb[t]) / 4
                                for t in range(3))
                    nrm = _unit(tuple(mid[t] - cc[t] for t in range(3)))
                    outn = nrm if face.sense else tuple(-x for x in nrm)
                    tri(pa, pb, qb, outn)
                    tri(pa, qb, qa, outn)
        elif isinstance(s, SphereS) and face.loops:
            # The corners are the loop's OWN vertices. They used to be rebuilt
            # as centre + r along each signed GLOBAL axis, reading the octant
            # vector as if it were a sign triple — which it only is when the
            # patch happens to be axis-aligned. _sphere_octant returns the SUM
            # of the three radii (frame-free, which is what the exact volume
            # needs); for a rotated corner that sum is a diagonal of length
            # r*sqrt(3), so the mesh subdivided the wrong spherical triangle
            # entirely. A rounded box rotated 45 degrees about z came out with
            # 88 of its 170 mesh edges unpaired — an STL full of holes — while
            # 0 and 90 degrees were clean, and the exact volume was right all
            # along.
            _sphere_octant(face)         # still refuses anything but an octant
            rr = float(s.r)
            cc = tuple(float(x) for x in s.c)
            corners = [tuple(float(x) for x in e.v0)
                       for lp in face.loops for e in lp.edges]
            # the octant's three quarter arcs are shared with the fillet bands
            # it blends, so its depth is theirs: 2^depth points per quarter must
            # equal the band's per-quarter count on the SAME axis, or the corner
            # peels away from the wall it is tangent to
            depth = max([_quarter_depth(rr, deflection)] + [
                (_per_quarter(e.curve)).bit_length() - 1
                for lp in face.loops for e in lp.edges
                if isinstance(e.curve, Circle)])

            def onsphere(q):
                v = [q[t] - cc[t] for t in range(3)]
                ln = math.sqrt(sum(x * x for x in v)) or 1.0
                return tuple(cc[t] + rr * v[t] / ln for t in range(3))

            def sub(a, b2, c2, lvl):
                if lvl == 0:
                    m = tuple((a[t] + b2[t] + c2[t]) / 3 for t in range(3))
                    outn = _unit(tuple(m[t] - cc[t] for t in range(3)))
                    tri(a, b2, c2, outn if face.sense
                        else tuple(-x for x in outn))
                    return
                ab = onsphere(tuple((a[t] + b2[t]) / 2 for t in range(3)))
                bc = onsphere(tuple((b2[t] + c2[t]) / 2 for t in range(3)))
                ca = onsphere(tuple((c2[t] + a[t]) / 2 for t in range(3)))
                sub(a, ab, ca, lvl - 1)
                sub(ab, b2, bc, lvl - 1)
                sub(ca, bc, c2, lvl - 1)
                sub(ab, bc, ca, lvl - 1)

            sub(corners[0], corners[1], corners[2], depth)
        elif isinstance(s, Cylinder):
            circs = [e.curve for lp in face.loops for e in lp.edges
                     if isinstance(e.curve, Circle)]
            if len(circs) < 2:
                continue
            lo, hi = circs[0], circs[-1]
            pa, segs = circle_pts(lo)
            pb, _ = circle_pts(hi, segs)
            ax = tuple(float(x) for x in s.p)
            for i in range(segs):
                j = (i + 1) % segs
                mid = tuple((pa[i][t] + pa[j][t]) / 2 for t in range(3))
                radial = _unit(tuple(mid[t] - ax[t] for t in range(3)))
                out = radial if face.sense else tuple(-x for x in radial)
                tri(pa[i], pa[j], pb[j], out)
                tri(pa[i], pb[j], pb[i], out)
        elif isinstance(s, Cone):
            circs = [e.curve for lp in face.loops for e in lp.edges
                     if isinstance(e.curve, Circle)]
            if not circs:
                continue
            ax = tuple(float(x) for x in s.p)
            ad = _unit(tuple(float(x) for x in s.d))
            lo = circs[0]
            pa, segs = circle_pts(lo)
            if len(circs) > 1:
                pb, _ = circle_pts(circs[-1], segs)
            else:
                pb = [ax] * segs            # a true cone closes at the apex
            for i in range(segs):
                j = (i + 1) % segs
                mid = tuple((pa[i][t] + pa[j][t]) / 2 for t in range(3))
                rel = tuple(mid[t] - ax[t] for t in range(3))
                axl = sum(rel[t] * ad[t] for t in range(3))
                radial = _unit(tuple(rel[t] - axl * ad[t] for t in range(3)))
                # the surface normal tilts off radial by the half-angle: the
                # generator runs from the apex, so the outward normal is
                # radial minus tan_half along the axis (times the opening sign)
                tw = float(s.tan_half) * (1.0 if axl > 0 else -1.0)
                nrm = _unit(tuple(radial[t] - tw * ad[t] for t in range(3)))
                out = nrm if face.sense else tuple(-x for x in nrm)
                tri(pa[i], pa[j], pb[j], out)
                if len(circs) > 1:
                    tri(pa[i], pb[j], pb[i], out)
        elif isinstance(s, SphereS):
            c = tuple(float(x) for x in s.c)
            r = float(s.r)
            seg = max(8, int(math.ceil(math.pi / math.acos(
                max(-1.0, 1 - deflection / r))))) if r > deflection else 8
            nlat, nlon = max(4, seg), max(8, 2 * seg)
            for i in range(nlat):
                t0, t1 = math.pi * i / nlat, math.pi * (i + 1) / nlat
                for j in range(nlon):
                    p0, p1 = 2 * math.pi * j / nlon, 2 * math.pi * (j + 1) / nlon
                    q = [(c[0] + r * math.sin(t) * math.cos(p),
                          c[1] + r * math.sin(t) * math.sin(p),
                          c[2] + r * math.cos(t))
                         for t, p in ((t0, p0), (t0, p1), (t1, p1), (t1, p0))]
                    mid = tuple(sum(x[k] for x in q) / 4 - c[k] for k in range(3))
                    out = _unit(mid) if face.sense else tuple(-x for x in _unit(mid))
                    tri(q[0], q[1], q[2], out)
                    tri(q[0], q[2], q[3], out)
    return {"vertices": verts, "triangles": _stitch_cracks(verts, tris)}


def _unit(v):
    m = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / m for x in v)


def _cross_f(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _perp_f(n):
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    d = sum(a[i] * n[i] for i in range(3))
    return _unit(tuple(a[i] - d * n[i] for i in range(3)))


def edges_info(body: Body) -> list[dict]:
    """Descriptors for every distinct edge — the selection surface that
    fillet/chamfer targeting and dimension anchoring read. ONE implementation
    covering every representation, because they all become a Body."""
    out, seen = [], set()
    for f in body.faces:
        for lp in f.loops:
            for e in lp.edges:
                if isinstance(e.curve, Circle):
                    c = e.curve
                    # NORMALISE the axis in the key: a band rim and the octant
                    # arc beside it are the SAME arc, but from_rounded_box
                    # builds the octant's circle from a cross product whose
                    # length is r^2, so (0,0,1) and (0,0,9) never collided and
                    # a rounded box reported 48 arcs where it has 24.
                    axis = _unit(tuple(float(y) for y in c.n))
                    whole = _key3(e.v0) == _key3(e.v1)
                    ends = tuple(sorted((tuple(float(x) for x in e.v0),
                                         tuple(float(x) for x in e.v1))))
                    # ...and canonicalise its SIGN: the same geometric circle
                    # is carried with +n by one face and -n by the other,
                    # because an Edge is CCW about its own circle's normal
                    lead = next((v for v in axis if abs(v) > 1e-12), 1.0)
                    if lead < 0:
                        axis_key = tuple(round(-v, 12) for v in axis)
                    else:
                        axis_key = tuple(round(v, 12) for v in axis)
                    key = ("circle", tuple(float(x) for x in c.c),
                           axis_key, float(c.r), ends)
                    if key in seen:
                        continue
                    seen.add(key)
                    # spans are TWELFTHS of a turn now (a quarter is 3)
                    span = 12 if whole else _arc_quarters(c, e.v0, e.v1)[1]
                    frac = span / 12
                    if whole:
                        cen = [float(x) for x in c.c]
                    else:
                        # the centroid of an ARC is on the arc, not at the
                        # circle centre — a dimension anchored to the old value
                        # pointed at a spot strictly inside the solid
                        pts = _arc_pts(c, e.v0, e.v1, float(c.r) / 64) + [
                            tuple(float(x) for x in e.v1)]
                        cen = [sum(q[i] for q in pts) / len(pts)
                               for i in range(3)]
                    out.append({"curve": "circle", "radius": float(c.r),
                                "centroid": cen,
                                "length": 2 * math.pi * float(c.r) * frac,
                                "sweep_twelfths": span,
                                "axis": list(axis)})
                else:
                    a = tuple(float(x) for x in e.v0)
                    b = tuple(float(x) for x in e.v1)
                    key = ("line",) + tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key)
                    d = tuple(b[i] - a[i] for i in range(3))
                    out.append({"curve": "line", "point": list(a), "dir": list(d),
                                "centroid": [(a[i] + b[i]) / 2 for i in range(3)],
                                "length": math.sqrt(sum(x * x for x in d))})
    return out


def _poly_area_centroid_f(vs, n):
    """Signed area (about n) and area centroid of a 3D planar polygon."""
    acc, area = [0.0, 0.0, 0.0], 0.0
    v0 = vs[0]
    for i in range(1, len(vs) - 1):
        a = [vs[i][k] - v0[k] for k in range(3)]
        b = [vs[i + 1][k] - v0[k] for k in range(3)]
        cr = _cross_f(a, b)
        ta = 0.5 * sum(cr[k] * n[k] for k in range(3))
        tc = [(v0[k] + vs[i][k] + vs[i + 1][k]) / 3 for k in range(3)]
        for k in range(3):
            acc[k] += tc[k] * ta
        area += ta
    if area == 0:
        m = len(vs)
        return 0.0, tuple(sum(v[k] for v in vs) / m for k in range(3))
    return area, tuple(acc[k] / area for k in range(3))


def faces_info(body: Body) -> list[dict]:
    """Descriptors for every face — surface kind plus the ANALYTIC parameters a
    caller needs (a bore reports its radius and axis, never a facet count)."""
    out = []
    for f in body.faces:
        s = f.surface
        if isinstance(s, Plane):
            nf = _unit(tuple(float(x) for x in s.n))
            # a hole subtracts from the area AND pulls the centroid: weighting
            # only the outer loop put a drilled plate's cap centroid at the
            # plate centre no matter where the bore was
            acc, area, outer = [0.0, 0.0, 0.0], 0.0, None
            for i, lp in enumerate(f.loops):
                circ = _loop_is_circle(lp)
                if circ is not None:
                    a = math.pi * float(circ.r) ** 2
                    c = tuple(float(x) for x in circ.c)
                elif _loop_has_arcs(lp):
                    a, c = _mixed_loop_area_centroid_f(lp, nf)
                    a = abs(a)
                else:
                    vs = [tuple(float(x) for x in e.v0) for e in lp.edges]
                    a, c = _poly_area_centroid_f(vs, nf)
                    a = abs(a)
                w = a if i == 0 else -a
                area += w
                for k in range(3):
                    acc[k] += c[k] * w
                if i == 0:
                    outer = c
            cen = ([acc[k] / area for k in range(3)] if area
                   else list(outer or (0.0, 0.0, 0.0)))
            out.append({"surface": "plane", "plane": list(nf),
                        "centroid": cen, "area": abs(area)})
        elif isinstance(s, Cylinder):
            try:
                h = float(_band_height(f, s))
            except ValueError:
                h = 0.0
            axis = _unit(tuple(float(x) for x in s.d))
            base = tuple(float(x) for x in s.p)
            arcs = _band_arc(f)
            span = 12 if arcs is None else _band_sweep(arcs, s)[0]
            mid = [base[i] + axis[i] * h / 2 for i in range(3)]
            if arcs is not None:
                # a TRIMMED band's surface centroid is out on the swept sector,
                # not on the axis: reporting the axis point made a rounded box
                # claim 5247.50 mm2 of surface against a true 2080.78
                circ, k0, _sp = min(
                    (x for x in arcs if dot(x[0].n, s.d) > 0), key=lambda x: x[1])
                pts = _arc_pts(circ, *_quarter_ends(circ, k0, span),
                               float(s.r) / 64)
                bar = [sum(q[i] for q in pts) / len(pts) for i in range(3)]
                off = sum((bar[i] - base[i]) * axis[i] for i in range(3))
                mid = [bar[i] - axis[i] * off + axis[i] * (h / 2) for i in range(3)]
            out.append({"surface": "cylinder", "radius": float(s.r),
                        "axis_dir": list(axis), "axis_origin": list(base),
                        "area": 2 * math.pi * float(s.r) * h * span / 12,
                        "sweep_twelfths": span,
                        "centroid": mid})
        elif isinstance(s, SphereS):
            oct_ = _sphere_octant(f)
            rr = float(s.r)
            cc = [float(x) for x in s.c]
            if oct_ is None:
                out.append({"surface": "sphere", "radius": rr,
                            "centroid": cc, "area": 4 * math.pi * rr ** 2})
            else:
                sv = [float(x) for x in oct_]
                # an octant is an EIGHTH of the sphere, and its surface
                # centroid sits out on the patch at 3r/4 along each axis
                out.append({"surface": "sphere", "radius": rr,
                            "octant": sv,
                            "centroid": [cc[i] + sv[i] * 3 * rr / 4
                                         for i in range(3)],
                            "area": 4 * math.pi * rr ** 2 / 8})
        elif isinstance(s, Cone):
            (ua, ra), (ub, rb) = _cone_rims(f, s)
            axis = _unit(tuple(float(x) for x in s.d))
            apex = tuple(float(x) for x in s.p)
            t = float(s.tan_half)
            slant = math.hypot(float(ub - ua), float(rb - ra))
            # the lateral-area centroid sits at 2/3 of the height for a full
            # cone, NOT at the axial midpoint: area grows with the radius, so
            # the wide end carries more of it
            fa, fb = float(ua), float(ub)
            umid = ((2 / 3) * (fb ** 3 - fa ** 3) / (fb ** 2 - fa ** 2)
                    if fb ** 2 != fa ** 2 else (fa + fb) / 2)
            out.append({"surface": "cone", "half_angle_tan": t,
                        "axis_dir": list(axis), "apex": list(apex),
                        "radii": [float(ra), float(rb)],
                        "area": math.pi * (float(ra) + float(rb)) * slant,
                        "centroid": [apex[i] + axis[i] * umid for i in range(3)]})
        elif isinstance(s, Torus):
            k0, span = _torus_sweep(f, s)
            RR, aa = float(s.R), float(s.a)
            cc = [float(x) for x in s.c]
            ax = list(_unit(tuple(float(x) for x in s.d)))
            sin_ = lambda k: (0, 1, 0, -1)[k % 4]
            dphi = span * math.pi / 2
            dsin = sin_(k0 + span) - sin_(k0)
            out.append({"surface": "torus", "major_radius": RR,
                        "minor_radius": aa, "axis_dir": ax, "centre": cc,
                        "sweep_twelfths": span,
                        "area": 2 * math.pi * aa * (RR * dphi + aa * dsin),
                        "centroid": cc})
        else:
            out.append({"surface": type(s).__name__.lower()})
    return out


# -- converters: every representation becomes a Body -------------------------

def _ring_face(pl: Plane, ring, holes=()) -> Face:
    def loop(vs):
        return Loop(tuple(Edge(Line(vs[i], sub(vs[(i + 1) % len(vs)], vs[i])),
                               vs[i], vs[(i + 1) % len(vs)])
                          for i in range(len(vs))))
    return Face(pl, (loop(ring),) + tuple(loop(h) for h in holes), True)


def _line_key(a, d):
    """Exact identity of the LINE through a with direction d.

    Direction canonicalised by its leading component (signed, so d and −d
    agree), position by the Plücker moment a × d̂ — which is the same for
    every point on the line, since (a + t·d̂) × d̂ = a × d̂. Two collinear
    edges therefore land in one bucket no matter where they start or which
    way they run.
    """
    lead = next(x for x in d if x != 0)
    u = tuple(x / lead for x in d)
    return (u, cross(a, u))


def _split_t_junctions(polys):
    """Insert every vertex that lies in the INTERIOR of another polygon's edge.

    A boolean leaves one face's long edge facing two short ones — a T-junction.
    It is invisible to volume (the geometry is identical) but it tears the
    mesh: the long edge is used once and each short edge once, so the shared
    seam is non-manifold and an STL of a slotted plate leaks. Splitting is
    exact — collinear is ``cross == 0``, between is ``0 < t < 1`` — and it is
    also what lets coplanar fragments cancel and merge at all.

    Zero-length edges are dropped: a degenerate edge has no interior, and
    leaving it in would make the same vertex appear twice in a ring.
    """
    # Bucket endpoints by their edge's CARRIER LINE. A vertex can only split
    # an edge it is collinear with, and a T-junction vertex is always an
    # endpoint of some OTHER edge on that same line — the face on the split
    # side carries both halves. So the candidate set is the bucket, not the
    # whole vertex set. Scanning every vertex for every edge cost 7.1 s on a
    # 966-polygon part (0.010 s before the split existed) and grew as the
    # square, which is minutes for a mesh-scale solid.
    on_line: dict = {}
    for vs in polys:
        n = len(vs)
        for i in range(n):
            a, b = vs[i], vs[(i + 1) % n]
            d = sub(b, a)
            if d == (F(0), F(0), F(0)):
                continue
            on_line.setdefault(_line_key(a, d), set()).update((a, b))

    out = []
    for vs in polys:
        ring = []
        n = len(vs)
        for i in range(n):
            a, b = vs[i], vs[(i + 1) % n]
            ab = sub(b, a)
            if ab == (F(0), F(0), F(0)):
                continue        # a degenerate edge has no interior AND no
                # start: appending `a` first would leave the vertex in twice
            ring.append(a)
            den = dot(ab, ab)
            hits = []
            for p in on_line[_line_key(a, ab)]:
                if p == a or p == b:
                    continue
                t = dot(sub(p, a), ab) / den
                if 0 < t < 1:
                    hits.append((t, p))
            ring.extend(p for _, p in sorted(hits))
        out.append(ring)                    # index-aligned with the input
    return out


def _ring_area2(ring, n):
    acc = (F(0), F(0), F(0))
    for i in range(len(ring)):
        c = cross(ring[i], ring[(i + 1) % len(ring)])
        acc = (acc[0] + c[0], acc[1] + c[1], acc[2] + c[2])
    return dot(acc, n)


def _ring_contains(ring, p, k):
    """Even-odd, projected off the plane's dominant axis k. Exact."""
    i, j = (k + 1) % 3, (k + 2) % 3
    inside = False
    for a in range(len(ring)):
        b = (a + 1) % len(ring)
        y1, y2 = ring[a][j], ring[b][j]
        if (y1 > p[j]) != (y2 > p[j]):
            x = ring[a][i] + (p[j] - y1) * (ring[b][i] - ring[a][i]) / (y2 - y1)
            if p[i] < x:
                inside = not inside
    return inside


def _plane_key(pl: Plane):
    """Identify a plane by GEOMETRY, not by the normal's arbitrary length.

    A ``Solid``'s plane normal is the raw cross product, so its magnitude is
    twice the polygon's area — ear-clipping one L-shaped cap into unequal
    triangles gives every fragment a DIFFERENT n for the same plane. Scaling
    n and d together by a positive rational leaves the plane and its facing
    untouched, and stays exact, so divide through by the leading component.
    """
    lead = next((x for x in pl.n if x != 0), None)
    if lead is None:
        return (pl.n, pl.d)
    s = abs(lead)                # exact over ℚ and ℚ[√d] alike (SurdVal.__abs__)
    return (tuple(x / s for x in pl.n), pl.d / s)


def _merge_coplanar(pl: Plane, polys):
    """Fuse coplanar polygons into whole faces by directed-edge cancellation.

    Returns None — meaning "keep the fragments" — whenever the merge is not
    provably area-preserving. A T-junction leaves an open chain, coplanar
    faces touching at a point leave a vertex with two ways out; in both cases
    a guessed loop would be worse than honest fragments.
    """
    if len(polys) == 1:
        return [_ring_face(pl, polys[0])]

    live: dict = {}
    for vs in polys:
        for i in range(len(vs)):
            a, b = vs[i], vs[(i + 1) % len(vs)]
            if live.get((b, a)):
                live[(b, a)] -= 1
                if not live[(b, a)]:
                    del live[(b, a)]
            else:
                live[(a, b)] = live.get((a, b), 0) + 1
    if not live:
        return None

    nxt: dict = {}
    for (a, b), count in live.items():
        if count != 1 or a in nxt:
            return None                     # fan-out: the boundary is ambiguous
        nxt[a] = b

    rings, unused = [], set(nxt)
    while unused:
        start = next(iter(unused))
        chain, v = [], start
        while v in unused:
            unused.discard(v)
            chain.append(v)
            v = nxt[v]
        if v != start or len(chain) < 3:
            return None                     # open chain: T-junction
        rings.append(chain)

    # NOT an area check. Signed doubled area is a sum over DIRECTED edges, and
    # cancellation removes each interior pair exactly (cross(a,b)+cross(b,a)=0)
    # while chaining consumes every survivor once — so comparing the ring areas
    # against the fragment areas is an identity that can never fail. It was
    # instrumented over ~3800 merges and fired zero times. Area preservation
    # rests instead on the PRECONDITION that the fragments tile the plane
    # without overlap, which a BSP solid's polygons satisfy by construction;
    # overlapping coplanar input is not detected here and would merge to a
    # region larger than its union. The structural guards above (fan-out at a
    # shared vertex, an open chain from a T-junction) are what actually fire.
    areas = [_ring_area2(r, pl.n) for r in rings]
    if any(a == 0 for a in areas):
        return None                         # a degenerate ring: bail

    k = max(range(3), key=lambda i: abs(pl.n[i]))
    outers = [(a, r) for a, r in zip(areas, rings) if a > 0]
    holes = [r for a, r in zip(areas, rings) if a < 0]
    if not outers:
        return None
    assigned: dict = {id(r): [] for _, r in outers}
    for h in holes:
        owner = min((o for o in outers if _ring_contains(o[1], h[0], k)),
                    key=lambda o: o[0], default=None)
        if owner is None:
            return None
        assigned[id(owner[1])].append(h)
    return [_ring_face(pl, r, assigned[id(r)]) for _, r in outers]


def from_solid(solid) -> Body:
    """A planar forge ``Solid`` — planar faces, with coplanar polygons MERGED.

    A ``Solid``'s polygons are the BSP's working units, not its faces: a prism
    cap arrives ear-clipped into triangles and a boolean leaves a face split
    along the cut. One canonical ``Face`` per fragment is not merely verbose,
    it is WRONG for anything that asks *which face carries this feature*. A
    Ø4 bore at the centre of a square cap straddles the ear-clip diagonal, so
    it lies inside NEITHER fragment — the caps kept no hole at all and the
    mesh came back with 48 non-manifold edges around an unclosed bore.
    """
    planes = [Plane(tuple(p.plane.n), p.plane.d) for p in solid.polys]
    rings = _split_t_junctions([[tuple(v) for v in p.verts] for p in solid.polys])
    groups: dict = {}
    for pl, ring in zip(planes, rings):
        if len(ring) >= 3:
            groups.setdefault(_plane_key(pl), (pl, []))[1].append(ring)
    faces = []
    for pl, polys in groups.values():
        merged = _merge_coplanar(pl, polys)
        faces.extend(merged if merged is not None
                     else [_ring_face(pl, vs) for vs in polys])
    return Body(tuple(faces))


def _circle_at(cx, cy, z, r) -> Circle:
    return Circle((F(cx), F(cy), F(z)), (F(0), F(0), F(1)),
                  (F(1), F(0), F(0)), F(r))


def _disk_face(cx, cy, z, r, up: bool) -> Face:
    c = _circle_at(cx, cy, z, r)
    v = (F(cx) + F(r), F(cy), F(z))
    return Face(Plane((F(0), F(0), F(1)), F(z)), (Loop((Edge(c, v, v),)),), up)


def from_cyl(c) -> Body:
    """A z-axis cylinder: two disk caps and one cylindrical wall."""
    from forgekernel.quadric import Cyl as _C

    c = _C(F(c.cx), F(c.cy), F(c.r), F(c.z0), F(c.z1))
    wall_lo = _circle_at(c.cx, c.cy, c.z0, c.r)
    wall_hi = _circle_at(c.cx, c.cy, c.z1, c.r)
    v0 = (c.cx + c.r, c.cy, c.z0)
    v1 = (c.cx + c.r, c.cy, c.z1)
    wall = Face(Cylinder((c.cx, c.cy, c.z0), (F(0), F(0), F(1)), c.r),
                (Loop((Edge(wall_lo, v0, v0), Edge(wall_hi, v1, v1))),), True)
    return Body((_disk_face(c.cx, c.cy, c.z1, c.r, True),
                 _disk_face(c.cx, c.cy, c.z0, c.r, False),
                 wall))


def from_napkin_ring(n) -> Body:
    """A sphere with a coaxial bore through it: ONE spherical zone + ONE wall.

    Two faces, no caps. The bore removes the sphere's polar caps entirely, so
    nothing planar survives — a converter that adds annular caps here is
    describing a different solid, and its volume would be plausible and wrong.

    The two faces share both rims, at z = cz ± √(R² − r²). That height is
    IRRATIONAL, which is why this converter could not exist until `F()` learned
    to admit ℚ[√d] — see tests/test_exact_field_coercion.py.

    Face senses are opposite because the wall faces INWARD: the material is
    outside the bore and inside the sphere.
    """
    h = n._half_height()
    zlo, zhi = n.cz - h, n.cz + h
    lo = _circle_at(n.cx, n.cy, zlo, n.r)
    hi = _circle_at(n.cx, n.cy, zhi, n.r)
    vlo = (n.cx + n.r, n.cy, zlo)
    vhi = (n.cx + n.r, n.cy, zhi)
    rim = Loop((Edge(lo, vlo, vlo), Edge(hi, vhi, vhi)))
    zone = Face(SphereS((n.cx, n.cy, n.cz), n.R), (rim,), True)
    wall = Face(Cylinder((n.cx, n.cy, zlo), (F(0), F(0), F(1)), n.r),
                (Loop((Edge(lo, vlo, vlo), Edge(hi, vhi, vhi))),), False)
    return Body((zone, wall))


def from_sphere_blind_bore(b) -> Body:
    """A sphere with a coaxial BLIND bore from +z: three faces.

      * the sphere minus its north cap — ONE rim at z = cz + √(R²−r²), with
        `SphereS.pole` pointing at the SOUTH pole the face contains (the
        one-rim band-versus-caps ambiguity is decided on the surface, never
        assumed);
      * the bore wall, floor to rim, facing INWARD;
      * the FLAT floor disk at the tool's end, facing UP into the hole — a
        plane, not the spherical cap an earlier backlog note guessed.

    The rim is shared by zone and wall; the floor circle by wall and floor
    disk; the south pole is a singular point on no edge (the pointed-cone
    precedent), so edge pairing closes with nothing dangling. The rim height
    is generally IRRATIONAL — the same reason `from_napkin_ring` needed F()
    widened to ℚ[√d].
    """
    h = b._half_height()
    zrim = b.cz + h
    zfl = b.cz + b.f
    rim = _circle_at(b.cx, b.cy, zrim, b.r)
    flo = _circle_at(b.cx, b.cy, zfl, b.r)
    vr = (b.cx + b.r, b.cy, zrim)
    vf = (b.cx + b.r, b.cy, zfl)
    zone = Face(SphereS((b.cx, b.cy, b.cz), b.R, pole=(F(0), F(0), F(-1))),
                (Loop((Edge(rim, vr, vr),)),), True)
    wall = Face(Cylinder((b.cx, b.cy, zfl), (F(0), F(0), F(1)), b.r),
                (Loop((Edge(flo, vf, vf), Edge(rim, vr, vr))),), False)
    return Body((zone, wall, _disk_face(b.cx, b.cy, zfl, b.r, True)))


def _face_contains_xy(face: "Face", px, py) -> bool:
    """Exact: does (px, py) lie inside this face's OUTER loop, in xy?

    A cap is often split into several coplanar fragments — ear-clipped prism
    caps, boolean-split tops — so matching a bore to a cap by z alone attaches
    its hole to every fragment at that height, including ones the bore never
    touches. (quadric.DrilledSolid.cut already carries the whole-solid version
    of this predicate, _xy_inside_footprint; this is the per-face analogue.)"""
    pts = [e.v0 for e in face.loops[0].edges]
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        if (y1 > py) != (y2 > py):
            if px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def from_drilled(d) -> Body:
    """A planar base minus z-cylindrical bores: cap faces gain circular inner
    loops, each bore contributes a cylindrical wall (and a disk if blind)."""
    body = from_solid(d.base)
    (_, _, bz0), (_, _, bz1) = d.base.bbox()
    faces = list(body.faces)
    from collections import defaultdict

    from forgekernel.quadric import Cyl as _C

    # Coaxial bores (a counterbore stack) are ONE stepped void, not several
    # independent ones: emitting a hole per bore double-subtracts their overlap
    # on the shared cap and leaves the inner wall running through the wider
    # bore's empty space. Merge them into z-bands carrying the OUTERMOST radius,
    # exactly as the section and tessellation paths do.
    groups: dict = defaultdict(list)
    for c in d.bores:
        c = _C(F(c.cx), F(c.cy), F(c.r), F(c.z0), F(c.z1))
        z0, z1 = max(c.z0, F(bz0)), min(c.z1, F(bz1))
        if z1 > z0:
            groups[(c.cx, c.cy)].append(_C(c.cx, c.cy, c.r, z0, z1))

    cap_faces = [i for i, f in enumerate(faces)
                 if isinstance(f.surface, Plane)
                 and f.surface.n[0] == 0 and f.surface.n[1] == 0]
    for (cx, cy), cyls in groups.items():
        zs = sorted({z for c in cyls for z in (c.z0, c.z1)})
        bands = []
        for za, zb in zip(zs, zs[1:]):
            zmid = (za + zb) / 2
            rs = [c.r for c in cyls if c.z0 <= zmid <= c.z1]
            if rs:
                bands.append((za, zb, max(rs)))
        if not bands:
            continue
        # A mouth belongs at every band boundary that is not a STEP between two
        # touching bands (a counterbore's shoulder, handled below). The stack
        # may have GAPS: a bore through a shelled plate is two bands with the
        # cavity between them, and taking only the outermost two z values left
        # the cavity's ceiling and floor unpunched — the two rims there paired
        # with nothing and the shell was open.
        ends = []
        for j, (za, zb, r) in enumerate(bands):
            if j == 0 or bands[j - 1][1] != za:
                ends.append((za, r, True))            # a floor: faces UP
            if j == len(bands) - 1 or bands[j + 1][0] != zb:
                ends.append((zb, r, False))           # a ceiling: faces DOWN
        for z, r, up in ends:
            cut_in = False
            for i in cap_faces:
                f = faces[i]
                if f.loops[0].edges[0].v0[2] != z or not _face_contains_xy(
                        f, cx, cy):
                    continue
                circ = _circle_at(cx, cy, z, r)
                v = (cx + r, cy, z)
                faces[i] = Face(f.surface,
                                f.loops + (Loop((Edge(circ, v, v),)),), f.sense)
                cut_in = True
            if not cut_in:
                # nothing to punch: the bore ends inside material, so it needs
                # a floor of its own (the blind case)
                faces.append(_disk_face(cx, cy, z, r, up))
        for za, zb, r in bands:                       # one wall per band
            lo_c, hi_c = _circle_at(cx, cy, za, r), _circle_at(cx, cy, zb, r)
            a, b = (cx + r, cy, za), (cx + r, cy, zb)
            faces.append(Face(Cylinder((cx, cy, za), (F(0), F(0), F(1)), r),
                              (Loop((Edge(lo_c, a, a), Edge(hi_c, b, b))),),
                              False))
        for (za, zb, r0), (zb2, zc, r1) in zip(bands, bands[1:]):
            if zb2 != zb:
                # the bands do not touch: a slab of material sits between two
                # blind bores, and each already got its own floor/ceiling from
                # the `ends` walk above. Emitting a shoulder here hung a
                # phantom annulus in solid material — volume wrong by an exact
                # pi multiple while validate stayed happy (W6), and the
                # annulus' two rims tore edge pairing (used 3x and 1x).
                continue
            if r0 == r1:
                continue                              # no step: no shoulder
            rin, rout = min(r0, r1), max(r0, r1)
            # the exposed ring faces INTO the wider bore
            up = r1 > r0
            outer = _circle_at(cx, cy, zb, rout)
            inner = _circle_at(cx, cy, zb, rin)
            vo, vi = (cx + rout, cy, zb), (cx + rin, cy, zb)
            faces.append(Face(Plane((F(0), F(0), F(1)), zb),
                              (Loop((Edge(outer, vo, vo),)),
                               Loop((Edge(inner, vi, vi),))), up))
    return Body(tuple(faces))


def from_sphere(s) -> Body:
    # coerce through F(): the quadric dataclasses annotate Fraction but do not
    # enforce it, so a directly-constructed Sphere(0,0,0,6) holds ints and
    # 4·r³/3 would silently become a FLOAT — the one thing the charter forbids.
    return Body((Face(SphereS((F(s.cx), F(s.cy), F(s.cz)), F(s.r)), (), True),))


def from_cone(cone) -> Body:
    """A quadric ``Cone`` frustum (r1 at z0, r2 at z1) as canonical faces.

    Equal radii is a CYLINDER, not a degenerate cone: the apex runs off to
    infinity and ``tan_half`` would be zero, so every axial measurement
    divides by nothing. Route it to the cylinder converter rather than emit a
    surface whose apex does not exist.
    """
    r1, r2 = F(cone.r1), F(cone.r2)
    z0, z1 = F(cone.z0), F(cone.z1)
    cx, cy = F(cone.cx), F(cone.cy)
    if z1 < z0:
        z0, z1, r1, r2 = z1, z0, r2, r1
    if z0 == z1:
        raise ValueError(
            "a cone of zero height has no apex and no slope — refuse rather "
            "than divide by it")
    if r1 == r2:
        from forgekernel.quadric import Cyl

        return from_cyl(Cyl(cx, cy, r1, z0, z1))

    slope = (r2 - r1) / (z1 - z0)               # dr/dz, exact
    z_apex = z0 - r1 / slope
    axis = (F(0), F(0), F(1))
    apex = (cx, cy, z_apex)
    t = slope if slope > 0 else -slope

    faces = []
    for z, r, up in ((z0, r1, False), (z1, r2, True)):
        if r == 0:
            continue                            # the cone closes to a point
        n = (F(0), F(0), F(1) if up else F(-1))
        loop = Loop((Edge(_circle_at(cx, cy, z, r),
                          (cx + r, cy, z), (cx + r, cy, z)),))
        faces.append(Face(Plane(n, z if up else -z), (loop,), True))

    rims = []
    for z, r in ((z0, r1), (z1, r2)):
        if r != 0:
            rims.append(Edge(_circle_at(cx, cy, z, r),
                             (cx + r, cy, z), (cx + r, cy, z)))
    faces.append(Face(Cone(apex, axis, t), (Loop(tuple(rims)),), True))
    return Body(tuple(faces))


def from_revolve(rev) -> Body:
    """A lathed (r, z) profile as canonical faces — one per profile segment.

    With ``Cone`` in the surface set every segment has an exact analytic
    surface: constant z is an annular PLANE, constant r a CYLINDER band, and
    anything else a cone frustum. Nothing here is faceted.

    Orientation comes from the profile's own winding. ``RevolveSolid``
    normalises the loop so its volume is positive, which makes the outward
    normal the right-hand perpendicular ``(dz, -dr)`` in the half-plane — so a
    segment travelling UP faces outward and one travelling DOWN faces in, which
    is exactly what distinguishes a tube's bore from its outside.
    """
    cx, cy = F(rev.cx), F(rev.cy)
    faces = []
    for (r1, z1), (r2, z2) in rev._edges():
        r1, z1, r2, z2 = F(r1), F(z1), F(r2), F(z2)
        if r1 == r2 == 0 or (r1 == r2 and z1 == z2):
            continue                            # on the axis, or degenerate
        if z1 == z2:                            # annular disk
            up = r2 < r1                        # (0, -dr) points +z when dr<0
            lo, hi = (r2, r1) if up else (r1, r2)
            n = (F(0), F(0), F(1) if up else F(-1))
            loops = [Loop((Edge(_circle_at(cx, cy, z1, hi),
                                (cx + hi, cy, z1), (cx + hi, cy, z1)),))]
            if lo > 0:
                loops.append(Loop((Edge(_circle_at(cx, cy, z1, lo),
                                        (cx + lo, cy, z1), (cx + lo, cy, z1)),)))
            faces.append(Face(Plane(n, z1 if up else -z1), tuple(loops), True))
            continue
        rims = tuple(Edge(_circle_at(cx, cy, z, r), (cx + r, cy, z),
                          (cx + r, cy, z))
                     for r, z in ((r1, z1), (r2, z2)) if r > 0)
        sense = z2 > z1
        if r1 == r2:
            # anchor the axis origin at the band, not at z=0: faces_info reads
            # the centroid as origin + axis*h/2, so a band from z=5..9 on a
            # z=0 origin reported z=2 instead of 7
            surf = Cylinder((cx, cy, min(z1, z2)), (F(0), F(0), F(1)), r1)
        else:
            slope = (r2 - r1) / (z2 - z1)
            apex = (cx, cy, z1 - r1 / slope)
            surf = Cone(apex, (F(0), F(0), F(1)),
                        slope if slope > 0 else -slope)
        faces.append(Face(surf, (Loop(rims),), sense))
    return Body(tuple(faces))


def from_rounded_box(rb) -> Body:
    """A box with every edge and corner filleted — the Minkowski sum of the
    core box with a ball — as 6 planes, 12 quarter-cylinders and 8 octants.

    This is the first TRIMMED-quadric body: its bands sweep a right angle
    rather than a full turn, and its spherical patches are corner octants.
    Both stay in ℚ[π] precisely because the trims are at right angles, where
    sin and cos are 0 and ±1 (see ``_quarter_index``).
    """
    r = F(rb.r)
    ox, oy, oz = (F(v) for v in rb.origin)
    lo = (ox + r, oy + r, oz + r)
    hi = (ox + F(rb.a) - r, oy + F(rb.b) - r, oz + F(rb.c) - r)
    out = (ox, oy, oz), (ox + F(rb.a), oy + F(rb.b), oz + F(rb.c))
    E = [(F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))]
    faces = []

    def corner(i, si, j, sj, k, t):
        """Point on the core box: extreme in axes i and j by the given signs,
        at parameter t along axis k."""
        pt = [None, None, None]
        pt[i] = hi[i] if si > 0 else lo[i]
        pt[j] = hi[j] if sj > 0 else lo[j]
        pt[k] = t
        return tuple(pt)

    # --- 6 flats, each the core rectangle pushed out to the true surface
    for ax in range(3):
        for sgn_ in (1, -1):
            n = tuple(sgn_ * x for x in E[ax])
            level = out[1][ax] if sgn_ > 0 else out[0][ax]
            u, v = (ax + 1) % 3, (ax + 2) % 3
            if sgn_ < 0:
                u, v = v, u
            ring = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                pt = [None, None, None]
                pt[ax] = level
                pt[u] = hi[u] if du else lo[u]
                pt[v] = hi[v] if dv else lo[v]
                ring.append(tuple(pt))
            edges = tuple(Edge(Line(ring[i], sub(ring[(i + 1) % 4], ring[i])),
                               ring[i], ring[(i + 1) % 4]) for i in range(4))
            faces.append(Face(Plane(n, level * sgn_), (Loop(edges),), True))

    # --- 12 quarter-cylinders, one per box edge
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        for si in (1, -1):
            for sj in (1, -1):
                axis = E[k]
                c0 = corner(i, si, j, sj, k, lo[k])
                c1 = corner(i, si, j, sj, k, hi[k])
                u = tuple(si * x for x in E[i])
                w = tuple(sj * x for x in E[j])
                # the band must sweep CCW about +axis from u to w, so pick the
                # ref that makes it so; cross(axis, u) is w or -w
                fwd = cross(axis, u) == w
                ref0, ref1 = (u, w) if fwd else (w, u)
                a0 = Circle(c0, axis, ref0, r)
                a1 = Circle(c1, tuple(-x for x in axis), ref1, r)
                p0, p1 = tuple(c0[t] + r * ref0[t] for t in range(3)),                     tuple(c0[t] + r * (w if fwd else u)[t] for t in range(3))
                q0, q1 = tuple(c1[t] + r * ref0[t] for t in range(3)),                     tuple(c1[t] + r * (w if fwd else u)[t] for t in range(3))
                loop = Loop((
                    Edge(a0, p0, p1),
                    Edge(Line(p1, sub(q1, p1)), p1, q1),
                    Edge(a1, q1, q0),
                    Edge(Line(q0, sub(p0, q0)), q0, p0),
                ))
                faces.append(Face(Cylinder(c0, axis, r), (loop,), True))

    # --- 8 corner octants
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                cen = (hi[0] if sx > 0 else lo[0], hi[1] if sy > 0 else lo[1],
                       hi[2] if sz > 0 else lo[2])
                pts = [tuple(cen[t] + r * (s * E[ax][t]) for t in range(3))
                       for ax, s in ((0, sx), (1, sy), (2, sz))]
                edges = tuple(
                    Edge(Circle(cen, cross(sub(pts[m], cen), sub(pts[(m + 1) % 3], cen)),
                                tuple(x / r for x in sub(pts[m], cen)), r),
                         pts[m], pts[(m + 1) % 3]) for m in range(3))
                faces.append(Face(SphereS(cen, r), (Loop(edges),), True))
    return Body(tuple(faces))


def lathe_body(segs, cx, cy) -> Body:
    """Build the canonical B-rep of a lathed profile whose segments may be
    ARCS as well as lines: an arc sweeps a torus, which is what a fillet on a
    turned rim actually is."""
    axis = (F(0), F(0), F(1))
    faces = []

    def circ(z, rad):
        return Edge(_circle_at(cx, cy, z, rad),
                      (cx + rad, cy, z), (cx + rad, cy, z))

    for sg in segs:
        if sg[0] == "arc":
            cen, p0, p1 = sg[1], sg[2], sg[3]
            a = max(abs(p0[0] - cen[0]), abs(p0[1] - cen[1]))
            def quarter(pt):
                co, si = (pt[0] - cen[0]) / a, (pt[1] - cen[1]) / a
                q = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}.get((co, si))
                if q is None:
                    raise ValueError(
                        "a lathe arc that does not start and end on a quarter "
                        "turn is outside ℚ[π] (K3.7)")
                return q
            k0 = quarter(p0)
            span = (quarter(p1) - k0) % 4 or 4
            if span == 3:
                # a CLOCKWISE quarter reads as three counter-clockwise ones.
                # fillet never produces it (RevolveSolid normalises winding),
                # but lathe_body is public API and must not guess.
                raise ValueError(
                    "a lathe arc traversed clockwise about its own centre is "
                    "ambiguous — carry it counter-clockwise (K3.7)")
            # #130: the rims are the face's BOUNDARY, and they must be on the
            # face. Emitting `()` left every rim circle used by ONE face (the
            # neighbour), so `manifold_violations` reported every filleted
            # lathe as open and the answer audit had to exempt any body
            # carrying a torus — a blanket pardon for exactly the bodies most
            # likely to be assembled wrong. A rim of radius zero sits ON the
            # axis: a singular point on no edge (the pointed-cone precedent).
            rims = tuple(circ(p[1], p[0]) for p in (p0, p1) if p[0] > 0)
            faces.append(Face(
                Torus((cx, cy, cen[1]), axis, cen[0], a, k0, span),
                (Loop(rims),) if rims else (), True))
            continue
        (r1, z1), (r2, z2) = sg[1], sg[2]
        if r1 == r2 == 0 or (r1 == r2 and z1 == z2):
            continue
        if z1 == z2:
            up = r2 < r1
            lo, hi = (r2, r1) if up else (r1, r2)
            n = (F(0), F(0), F(1) if up else F(-1))
            loops = [Loop((circ(z1, hi),))]
            if lo > 0:
                loops.append(Loop((circ(z1, lo),)))
            faces.append(Face(Plane(n, z1 if up else -z1),
                                tuple(loops), True))
            continue
        rims = tuple(circ(z, r) for r, z in ((r1, z1), (r2, z2)) if r > 0)
        sense = z2 > z1
        if r1 == r2:
            surf = Cylinder((cx, cy, min(z1, z2)), axis, r1)
        else:
            slope = (r2 - r1) / (z2 - z1)
            surf = Cone((cx, cy, z1 - r1 / slope), axis,
                          slope if slope > 0 else -slope)
        faces.append(Face(surf, (Loop(rims),), sense))
    return Body(tuple(faces))


def _horiz_face_z(f):
    """z of a horizontal planar face (exact), else None."""
    if not isinstance(f.surface, Plane) or not f.loops:
        return None
    n = f.surface.n
    if n[0] != 0 or n[1] != 0:
        return None
    return f.surface.d / n[2]


def _disc_strictly_on_face(f, cx, cy, r) -> bool:
    """Exact: the disc (cx, cy, radius r) lies strictly inside the face's
    outer loop and strictly clear of every inner loop (existing holes).
    Strictness keeps tangent contacts on the merged-shells fallback rather
    than minting a degenerate zero-width sliver of face."""
    from forgekernel.quadric import _dist2_point_seg

    outer = f.loops[0].edges
    if len(outer) == 1 and isinstance(outer[0].curve, Circle):
        cc = outer[0].curve                   # a disk cap (e.g. a wider boss)
        if cc.r <= r:
            return False
        if ((cx - cc.c[0]) ** 2 + (cy - cc.c[1]) ** 2) >= (cc.r - r) ** 2:
            return False
    else:
        if any(isinstance(e.curve, Circle) for e in outer):
            return False                      # mixed boundary: out of scope
        pts = [(e.v0[0], e.v0[1]) for e in outer]
        m = len(pts)
        inside = False
        for i in range(m):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % m]
            if (y1 > cy) != (y2 > cy):
                if cx < x1 + (cy - y1) * (x2 - x1) / (y2 - y1):
                    inside = not inside
        if not inside:
            return False
        r2 = r * r
        for i in range(m):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % m]
            if _dist2_point_seg(cx, cy, x1, y1, x2, y2) <= r2:
                return False
    for lp in f.loops[1:]:
        if len(lp.edges) == 1 and isinstance(lp.edges[0].curve, Circle):
            h = lp.edges[0].curve
            # the disc must sit wholly on MATERIAL: strictly clear of the
            # hole in both directions (no overlap, no containment, no touch)
            if ((cx - h.c[0]) ** 2 + (cy - h.c[1]) ** 2) <= (r + h.r) ** 2:
                return False
        else:
            return False                      # unknown hole shape: no stitch
    return True


def _union_stitches(members, bodies) -> list:
    """Cap-on-face contacts inside a DisjointUnion, at the BODY level:
    [(face_owner_idx, face_idx, cyl_idx, z)] where members[cyl_idx] is a
    Cyl whose bottom (or top) cap sits at exactly the z of a horizontal
    planar face of bodies[face_owner_idx] with its full disc strictly on
    that face. All predicates exact. The union's own disjointness proof
    guarantees the member behind such a face lies on the far side, so a
    z + containment match IS the contact.

    Anything more entangled — several discs on one face, one cap meeting
    several faces — is dropped (both counters must read exactly one), and
    the converter keeps the merged-shells fallback for those members."""
    from collections import Counter

    from forgekernel.quadric import Cyl

    cands = []
    for ci, c in enumerate(members):
        if not isinstance(c, Cyl):
            continue
        for si, sb in enumerate(bodies):
            if si == ci:
                continue
            for fi, f in enumerate(sb.faces):
                try:
                    z = _horiz_face_z(f)
                    if z is None:
                        continue
                    if F(c.z0) == z:
                        end = "lo"
                    elif F(c.z1) == z:
                        end = "hi"
                    else:
                        continue
                    if not _disc_strictly_on_face(f, F(c.cx), F(c.cy),
                                                  F(c.r)):
                        continue
                except (TypeError, ZeroDivisionError, AttributeError):
                    continue                  # exotic coordinates: no stitch
                cands.append((si, fi, ci, end, z))
    per_face = Counter((si, fi) for si, fi, _, _, _ in cands)
    per_cap = Counter((ci, end) for _, _, ci, end, _ in cands)
    return [(si, fi, ci, z) for si, fi, ci, end, z in cands
            if per_face[(si, fi)] == 1 and per_cap[(ci, end)] == 1]


def _find_cap_disk(face_list, c, z):
    """Index of the cylinder body's full-disk cap face at z, else None."""
    for i, f in enumerate(face_list):
        if f is None or not isinstance(f.surface, Plane):
            continue
        if len(f.loops) != 1 or len(f.loops[0].edges) != 1:
            continue
        e = f.loops[0].edges[0]
        if (isinstance(e.curve, Circle) and e.curve.r == F(c.r)
                and e.curve.c == (F(c.cx), F(c.cy), z)):
            return i
    return None


def to_body(shape) -> Body:
    """Convert any forge representation to the canonical B-rep, or raise."""
    from forgekernel.brep import Solid
    from forgekernel.quadric import (Cone as QCone, Cyl, DisjointUnion,
                                     DrilledSolid, RevolveSolid, RoundedBox,
                                     Sphere)

    from forgekernel.surdrev import NapkinRing, SphereBlindBore

    if isinstance(shape, Body):
        return shape
    if isinstance(shape, NapkinRing):
        return from_napkin_ring(shape)
    if isinstance(shape, SphereBlindBore):
        return from_sphere_blind_bore(shape)
    if isinstance(shape, Solid):
        return from_solid(shape)
    if isinstance(shape, QCone):
        return from_cone(shape)
    if isinstance(shape, RevolveSolid):
        return from_revolve(shape)
    if isinstance(shape, RoundedBox):
        return from_rounded_box(shape)
    if isinstance(shape, Cyl):
        return from_cyl(shape)
    if isinstance(shape, DrilledSolid):
        return from_drilled(shape)
    if isinstance(shape, Sphere):
        return from_sphere(shape)
    if isinstance(shape, DisjointUnion):
        # Concatenating the member bodies' faces verbatim left the contact
        # patches BURIED: the flagship boss-on-plate carried the plate's
        # full top rectangle AND the boss's bottom cap, coplanar inside
        # material — volume and per-shell pairing stayed right while every
        # consumer of the canonical form (the one mesher, ADR-0021) saw
        # internal walls (W17). Where a Cyl cap sits on a member's face
        # with its whole disc strictly on material, stitch: punch the
        # contact circle into the face as an inner loop (the very Circle
        # record the cylinder wall's rim carries, so edge pairing closes)
        # and drop the buried cap disk. Everything else keeps the
        # merged-shells fallback, honest residual until general boolean
        # face classification lands.
        bodies = [to_body(m) for m in shape.members]
        face_lists = [list(b.faces) for b in bodies]
        for si, fi, ci, z in _union_stitches(shape.members, bodies):
            c = shape.members[ci]
            di = _find_cap_disk(face_lists[ci], c, z)
            if di is None:
                continue        # unexpected cylinder body shape: no stitch
            f = face_lists[si][fi]
            circ = _circle_at(c.cx, c.cy, z, c.r)
            v = (F(c.cx) + F(c.r), F(c.cy), z)
            face_lists[si][fi] = Face(
                f.surface, f.loops + (Loop((Edge(circ, v, v),)),), f.sense)
            face_lists[ci][di] = None
        return Body(tuple(f for fl in face_lists for f in fl
                          if f is not None))
    raise ValueError(
        f"no canonical-B-rep converter for {type(shape).__name__} yet "
        "(ADR-0021 converters land per representation)")


def vector_area_defect(body: "Body"):
    """Closed-shell orientation check: sum of face vector areas must be zero.

    ``manifold_violations`` counts how many faces use each edge. It does not
    look at DIRECTION, so a face whose loop is traversed the wrong way still
    reports every edge used exactly twice — measured, not assumed: reversing
    one face of a box gives 0 violations and this function gives (0, 0, 800).

    The identity is Gauss': for a closed surface, sum over faces of the vector
    area is the zero vector. Vector area is a BOUNDARY integral,
    ``(1/2) * contour integral of r x dr``, so it needs only the loop — no
    surface parametrisation, no area formula per surface type, and it is exact
    in Q for straight edges.

    Returns ``None`` when the check does not apply, and the caller must treat
    that as "not checked" rather than "passed". Today that means any body
    carrying a circular edge: the arc term is
    ``(1/2)[c x (p1 - p0) + r^2 * dtheta * n_hat]``, derived and exact in Q[pi]
    via the twelfth span, but it is NOT implemented here yet, and a partial sum
    would be worse than no sum — it would be nonzero for a perfectly good
    solid. So a body with any arc is skipped WHOLE.
    """
    total = [F(0), F(0), F(0)]
    for f in body.faces:
        sgn = 1 if getattr(f, "sense", 1) in (1, True) else -1
        for lp in f.loops:
            for e in lp.edges:
                if isinstance(e.curve, Circle):
                    return None              # see docstring: skip whole, not part
                c = cross(e.v0, e.v1)
                for i in range(3):
                    total[i] += sgn * F(c[i]) / 2
    return tuple(total)
