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
class SphereS:
    c: Vec
    r: Fraction

    def transformed(self, m):
        return SphereS(m.point(self.c), m.scale_len(self.r))


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
    if dot(c.n, c.n) != 1 or dot(c.ref, c.ref) != 1 or dot(c.ref, c.n) != 0:
        raise ValueError(
            "circle frame is not exactly orthonormal — a trimmed quadric "
            "needs exact quarter positions (K3.7)")
    return c.ref, cross(c.n, c.ref)


def _quarter_depth(r: float, deflection: float) -> int:
    """Subdivision depth for a quarter arc, shared by trimmed bands and corner
    octants so that they agree vertex for vertex along the arcs they share."""
    d = 1
    while d < 7 and r * (1 - math.cos(math.pi / 4 / (2 ** d))) > deflection:
        d += 1
    return d


def _quarter_index(c: Circle, p):
    """Which quarter turn from `ref` the point sits at (0..3), or None.

    Trimmed quadrics stay in ℚ[π] only at right angles, where sin and cos are
    0 and ±1. Anywhere else the band's ∫n̂ dθ term is a transcendental that
    leaves the field, so this is the predicate that decides exact-or-refuse.
    """
    u, w = _circle_frame(c)
    rel = sub(p, c.c)
    for k, v in enumerate((u, w, tuple(-x for x in u), tuple(-x for x in w))):
        if rel == tuple(c.r * x for x in v):
            return k
    return None


def _arc_quarters(c: Circle, v0, v1):
    """(start index, number of quarter turns) for an arc, counter-clockwise
    about the circle's normal. A closed edge is the whole four."""
    k0, k1 = _quarter_index(c, v0), _quarter_index(c, v1)
    if k0 is None or k1 is None:
        raise ValueError(
            "arc endpoint is not at a quarter turn — a trimmed quadric is "
            "exact only at right angles (K3.7)")
    span = (k1 - k0) % 4
    return k0, (4 if span == 0 else span)


def _quarter_antiderivative(c: Circle, k: int):
    """∫n̂ dθ evaluated at θ = k·π/2, i.e. sin(θ)u − cos(θ)w. Rational."""
    u, w = _circle_frame(c)
    sin_, cos_ = ((0, 1), (1, 0), (0, -1), (-1, 0))[k % 4]
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
    from forgekernel.quadric import PiVal

    total = PiVal(0, 0)
    for face in body.faces:
        total = total + _face_volume_term(face)
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
            if circ is None:
                if _loop_has_arcs(lp):
                    raise ValueError(
                        "planar loop mixing arcs and lines (a slot end, a "
                        "D-profile) — its area is not in ℚ[π] with the chord "
                        "treatment, so refuse rather than under-report (K3.7)")
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
        if float(_as_fraction(area_rat) or 0) + math.pi * float(
                _as_fraction(area_pi) or 0) < -1e-12:
            raise ValueError(
                "face has negative area — an inner loop is larger than the "
                "boundary carrying it (a hole attached to the wrong face)")
        return PiVal(_exact(sgn * rat, "planar area"),
                     _exact(sgn * pi, "circular area"))
    if isinstance(s, Cylinder):
        # x·n̂ = p·N̂ + r on the surface, so ∮x·n̂ dA = r·h·(p·∫N̂dθ + r·Δθ).
        # Over a FULL band ∫N̂dθ vanishes and this is the familiar
        # (1/3)(2πr²h); over a TRIMMED one it does not, and the leftover term
        # is rational exactly when the arc ends on quarter turns — which is
        # what keeps a rounded box's edge fillets inside ℚ[π].
        h = _band_height(face, s)
        arcs = _band_arc(face)
        if arcs is None:
            return PiVal(0, _exact(sgn * 2 * s.r * s.r * h / 3, "cylinder band"))
        if len({sp for _c, _k, sp in arcs}) != 1:
            raise ValueError(
                "a trimmed band whose rims sweep different angles is not one "
                "face (K3.7)")
        # read the sweep off the rim whose normal runs WITH the axis: the
        # other rim carries -n, because an Edge on a Circle is CCW about that
        # circle's own normal and the two rims run opposite ways
        forward = [x for x in arcs if dot(x[0].n, s.d) > 0]
        circ, k0, span = (forward or arcs)[0]
        vint = sub(_quarter_antiderivative(circ, k0 + span),
                   _quarter_antiderivative(circ, k0))
        rat = sgn * s.r * h * dot(s.p, vint) / 3
        pi = sgn * s.r * h * s.r * Fraction(span, 2) / 3          # Δθ = span·π/2
        return PiVal(_exact(rat, "trimmed band offset"),
                     _exact(pi, "trimmed band sweep"))
    if isinstance(s, SphereS):
        # x·n̂ = r + c·n̂. Over the whole sphere ∮c·n̂ dA = 0, leaving
        # (1/3)r·4πr². Over one OCTANT neither term vanishes: the area is
        # πr²/2 and ∮n̂ dA is (πr²/4) times the octant's sign vector.
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
        return PiVal(0, _exact(sgn * dot(s.p, s.d) / ln * nax_area / 3,
                               "cone band"))
    raise ValueError(
        f"no exact volume term for {type(s).__name__} yet (K3.7)")


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
    c = face.surface.c
    acc = [F(0), F(0), F(0)]
    for q in pts:
        for i in range(3):
            acc[i] += q[i] - c[i]
    out = tuple(F(1) if x > 0 else (F(-1) if x < 0 else F(0)) for x in acc)
    if 0 in out or any(abs(a) != face.surface.r for a in acc):
        raise ValueError(
            "spherical patch corners are not on the signed axes — not an "
            "octant (K3.7)")
    return out


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
                else:
                    _refuse_arc_loop(lp)
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
            r2h = math.pi * float(s.r) ** 2 * h
            for k in range(3):
                perp = mid[k] - md * d[k]
                m[k] += sgn * 0.25 * r2h * (2 * mid[k] + perp)
        elif isinstance(s, SphereS):
            v = 4 / 3 * math.pi * float(s.r) ** 3
            for k in range(3):
                m[k] += sgn * v * float(s.c[k])
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
        else:
            raise ValueError(
                f"centroid of a {type(s).__name__} face is not implemented "
                "yet — arrives with K3.7")
    v = float(volume(body))
    if v == 0:
        raise ValueError("centroid of a zero-volume body is undefined")
    return tuple(x / v for x in m)


def bbox(body: Body):
    """Float bbox of the body (a bound, not a topological decision)."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for f in body.faces:
        if isinstance(f.surface, SphereS):      # a whole sphere carries no loops
            c, r = f.surface.c, float(f.surface.r)
            for i in range(3):
                lo[i] = min(lo[i], float(c[i]) - r)
                hi[i] = max(hi[i], float(c[i]) + r)
            continue
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

    def _segs_for(r: float) -> int:
        if r <= deflection:
            return 24
        return max(24, int(math.ceil(
            math.pi / math.acos(max(-1.0, 1 - deflection / r)))))

    # ONE segment count per AXIS, taken from the widest circle on it. Deriving
    # it per circle tears every taper: a frustum's r=2 and r=5 rims want
    # different counts, so the wall and the caps stop sharing vertices. Coaxial
    # bores in a counterbore have the same problem.
    axis_segs: dict = {}
    for f in body.faces:
        for lp in f.loops:
            for e in lp.edges:
                if isinstance(e.curve, Circle):
                    k = _line_key(e.curve.c, e.curve.n)
                    axis_segs[k] = max(axis_segs.get(k, 0),
                                       _segs_for(float(e.curve.r)))

    def circle_pts(c: Circle, n=None):
        r = float(c.r)
        segs = n or axis_segs.get(_line_key(c.c, c.n)) or _segs_for(r)
        u = _unit(tuple(float(x) for x in c.ref))
        w = _unit(_cross_f(tuple(float(x) for x in c.n), u))
        cc = tuple(float(x) for x in c.c)
        return [tuple(cc[i] + r * (math.cos(2 * math.pi * k / segs) * u[i]
                                   + math.sin(2 * math.pi * k / segs) * w[i])
                      for i in range(3)) for k in range(segs)], segs

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
                            ring.extend(_arc_pts(e.curve, e.v0, e.v1,
                                                 deflection))
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
            circ, k0, span = fwd[0]
            u, w = (tuple(float(x) for x in v) for v in _circle_frame(circ))
            rr = float(s.r)
            # a trimmed band and the corner octants next to it SHARE their
            # quarter arcs, so both must land on the same points. Projecting a
            # chord midpoint onto a circle lands on the ANGULAR midpoint, so a
            # depth-d octant subdivision gives 2^d uniform steps per quarter —
            # match the band to that or the shell tears along every seam.
            per = 2 ** _quarter_depth(rr, deflection)
            nseg = per * span
            axis = _unit(tuple(float(x) for x in s.d))
            ends = sorted({sum((float(c.c[t]) - float(s.p[t])) * axis[t]
                                for t in range(3))
                           for c, _k, _sp in arcs})
            base = tuple(float(x) for x in s.p)
            def at(tt, ang):
                th = (k0 + ang) * math.pi / 2
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
        elif isinstance(s, SphereS) and face.loops:
            oct_ = _sphere_octant(face)
            rr = float(s.r)
            cc = tuple(float(x) for x in s.c)
            corners = [tuple(cc[t] + rr * float(oct_[ax]) * (1.0 if ax == t else 0.0)
                             for t in range(3)) for ax in range(3)]
            depth = _quarter_depth(rr, deflection)

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
                    key = ("circle", tuple(float(x) for x in c.c),
                           tuple(float(x) for x in c.n), float(c.r))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"curve": "circle", "radius": float(c.r),
                                "centroid": [float(x) for x in c.c],
                                "length": 2 * math.pi * float(c.r),
                                "axis": list(_unit(tuple(float(y) for y in c.n)))})
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
                else:
                    _refuse_arc_loop(lp)
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
            out.append({"surface": "cylinder", "radius": float(s.r),
                        "axis_dir": list(axis), "axis_origin": list(base),
                        "area": 2 * math.pi * float(s.r) * h,
                        "centroid": [base[i] + axis[i] * h / 2 for i in range(3)]})
        elif isinstance(s, SphereS):
            out.append({"surface": "sphere", "radius": float(s.r),
                        "centroid": [float(x) for x in s.c],
                        "area": 4 * math.pi * float(s.r) ** 2})
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
        zlo, zhi = bands[0][0], bands[-1][1]
        # cap holes: only on a fragment this stack actually passes through
        for i in cap_faces:
            f = faces[i]
            zc = f.loops[0].edges[0].v0[2]
            r = (bands[-1][2] if zc == zhi else
                 bands[0][2] if zc == zlo else None)
            if r is None or not _face_contains_xy(f, cx, cy):
                continue
            circ = _circle_at(cx, cy, zc, r)
            v = (cx + r, cy, zc)
            faces[i] = Face(f.surface, f.loops + (Loop((Edge(circ, v, v),)),),
                            f.sense)
        for za, zb, r in bands:                       # one wall per band
            lo_c, hi_c = _circle_at(cx, cy, za, r), _circle_at(cx, cy, zb, r)
            a, b = (cx + r, cy, za), (cx + r, cy, zb)
            faces.append(Face(Cylinder((cx, cy, za), (F(0), F(0), F(1)), r),
                              (Loop((Edge(lo_c, a, a), Edge(hi_c, b, b))),),
                              False))
        for (za, zb, r0), (zb2, zc, r1) in zip(bands, bands[1:]):
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
        if zlo > F(bz0):                              # blind at the bottom
            faces.append(_disk_face(cx, cy, zlo, bands[0][2], True))
        if zhi < F(bz1):                              # blind at the top
            faces.append(_disk_face(cx, cy, zhi, bands[-1][2], False))
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


def to_body(shape) -> Body:
    """Convert any forge representation to the canonical B-rep, or raise."""
    from forgekernel.brep import Solid
    from forgekernel.quadric import (Cone as QCone, Cyl, DisjointUnion,
                                     DrilledSolid, RevolveSolid, RoundedBox,
                                     Sphere)

    if isinstance(shape, Body):
        return shape
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
        faces = []
        for m in shape.members:
            faces.extend(to_body(m).faces)
        return Body(tuple(faces))
    raise ValueError(
        f"no canonical-B-rep converter for {type(shape).__name__} yet "
        "(ADR-0021 converters land per representation)")
