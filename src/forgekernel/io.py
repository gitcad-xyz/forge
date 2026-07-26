"""Native serialization — exact, canonical, hashable (ADR-0018).

Rationals serialize as "num/den" strings, so a round-trip is BIT-exact
and two equal solids produce identical bytes: geometry identity by
hash, the property OCCT never offered.
"""

from __future__ import annotations

import json
from fractions import Fraction

from forgekernel.brep import Polygon, Solid

SCHEMA = "forge/solid@1"


def _fr(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}"


def _unfr(s: str) -> Fraction:
    n, d = s.split("/")
    return Fraction(int(n), int(d))


def dumps(solid: Solid) -> str:
    doc = {"schema": SCHEMA, "polys": [
        {"source": p.source,
         "verts": [[_fr(v[0]), _fr(v[1]), _fr(v[2])] for v in p.verts]}
        for p in solid.polys]}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"


def loads(text: str) -> Solid:
    doc = json.loads(text)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"unsupported solid schema {doc.get('schema')!r}")
    return Solid([Polygon([(_unfr(a), _unfr(b), _unfr(c))
                           for a, b, c in p["verts"]], p["source"])
                  for p in doc["polys"]])


BODY_SCHEMA = "forge/body@1"


def _num(v) -> str:
    """Exact scalar → canonical string.

    A ``Solid``'s coordinates are rational, but a ROTATED one carries ℚ[√d]
    — so a format that only knows ``num/den`` can describe a box and not the
    same box turned 45°. Both encodings are exact and both round-trip to the
    identical value, which is what makes the bytes a geometry identity.
    """
    from forgekernel.surd import SurdVal

    if isinstance(v, SurdVal):
        if v.b == 0:
            # SurdVal(a, 0, d) IS the rational a. A rigid map leaves every
            # coordinate SurdVal-TYPED even when its surd part is zero, and
            # branching on the TYPE alone respelled all of them `S:a/1:0/1:1/1`
            # — a 360° rotation rewrote the whole committed file (16838 →
            # 30998 bytes) for a geometric no-op, which is exactly what
            # ADR-0004's byte-canonical rule exists to prevent (docket S1).
            return _fr(v.a)
        return f"S:{_fr(v.a)}:{_fr(v.b)}:{_fr(v.d)}"
    return _fr(Fraction(v))


def _unnum(s: str):
    if s.startswith("S:"):
        from forgekernel.surd import SurdVal

        _, a, b, d = s.split(":")
        return SurdVal(_unfr(a), _unfr(b), _unfr(d))
    return _unfr(s)


def _vec(v):
    return [_num(x) for x in v]


def dumps_body(body) -> str:
    """The canonical B-rep as exact, byte-canonical text (ADR-0004/0021).

    Surfaces and curves are stored ANALYTICALLY — a bore is a cylinder with an
    exact radius, never a fan of facets — so the text says what the shape IS,
    and a diff between two revisions reads as a change of geometry rather than
    a reshuffle of triangles.
    """
    from forgekernel.body import Circle, Cone, Cylinder, Line, Plane, SphereS

    def surface(s):
        if isinstance(s, Plane):
            return {"kind": "plane", "n": _vec(s.n), "d": _num(s.d)}
        if isinstance(s, Cylinder):
            return {"kind": "cylinder", "p": _vec(s.p), "d": _vec(s.d),
                    "r": _num(s.r)}
        if isinstance(s, Cone):
            return {"kind": "cone", "p": _vec(s.p), "d": _vec(s.d),
                    "tan_half": _num(s.tan_half)}
        if isinstance(s, SphereS):
            out = {"kind": "sphere", "c": _vec(s.c), "r": _num(s.r)}
            if s.pole is not None:
                # the pole TRIM is part of what the surface is (a blind
                # bore's one-rim face) — dropping it would make the document
                # describe a different solid, silently
                out["pole"] = _vec(s.pole)
            return out
        raise ValueError(
            f"no text encoding for a {type(s).__name__} surface yet (K3.7)")

    def curve(c):
        if isinstance(c, Line):
            return {"kind": "line", "p": _vec(c.p), "d": _vec(c.d)}
        if isinstance(c, Circle):
            return {"kind": "circle", "c": _vec(c.c), "n": _vec(c.n),
                    "ref": _vec(c.ref), "r": _num(c.r)}
        raise ValueError(
            f"no text encoding for a {type(c).__name__} curve yet (K3.7)")

    def face_doc(f):
        # NOTE: (n, d, sense) and (-n, -d, not sense) name the same face, and
        # this format still admits both. Folding the sign into `sense` alone is
        # NOT enough — the loops are wound about n, so flipping it inverts
        # every loop's area and the face reads as having negative area. Doing
        # it properly means reversing each loop and each circle's normal too;
        # until that is written and verified, two converters can spell one
        # plane two ways. Face ORDER, which was the larger half of the
        # problem, is canonical below.
        return {"surface": surface(f.surface), "sense": bool(f.sense),
                "loops": [[{"curve": curve(e.curve), "v0": _vec(e.v0),
                            "v1": _vec(e.v1)} for e in lp.edges]
                          for lp in f.loops]}

    # Face ORDER is not geometry either: from_cyl and from_revolve build the
    # same cylinder's faces in different sequences. Sort by the encoded face,
    # so equal solids produce equal bytes -- the whole point of the format.
    doc = {"schema": BODY_SCHEMA,
           "faces": sorted((face_doc(f) for f in body.faces),
                           key=lambda d: json.dumps(d, sort_keys=True))}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"


def _bad(msg):
    raise ValueError(f"malformed body document: {msg}")


def _lies(msg):
    raise ValueError(f"inconsistent body document: {msg}")


def _check_geometry(body) -> None:
    """Refuse a document whose faces contradict their own rims (docket W13/R3).

    The schema pass catches malformed SHAPE; this catches well-formed documents
    describing impossible GEOMETRY. It matters because ``volume()`` blends the
    surface record (lateral sweep term) with the rim circles (cap terms): a
    cone whose ``tan_half`` was edited 3/10 → 9/10 while its rims still said
    r=6 and r=3 loaded clean and reported +171.4% with zero manifold
    violations, and a sphere radius edited to −6 reported a NEGATIVE volume
    and an inverted bbox. Every question below is exact arithmetic in ℚ or
    ℚ[√d] — no tolerance decides anything (ADR-0019). A document is UNTRUSTED
    input (ADR-0006/0007): validate, never blend.
    """
    from forgekernel.body import (Circle, Cone, Cylinder, Line, Plane, SphereS)
    from forgekernel.exact import cross, dot, sub

    def zero3(v):
        return all(c == 0 for c in v)

    def check_surface(s):
        if isinstance(s, Plane) and zero3(s.n):
            _lies("a plane normal must be nonzero")
        if isinstance(s, (Cylinder, Cone)) and zero3(s.d):
            _lies(f"a {type(s).__name__.lower()} axis must be nonzero")
        if isinstance(s, Cylinder) and not s.r > 0:
            _lies(f"a cylinder radius must be positive, got {s.r}")
        if isinstance(s, Cone) and not s.tan_half > 0:
            _lies(f"a cone tan_half must be positive, got {s.tan_half}")
        if isinstance(s, SphereS) and not s.r > 0:
            _lies(f"a sphere radius must be positive, got {s.r}")

    def check_curve(c):
        if isinstance(c, Line) and zero3(c.d):
            _lies("a line direction must be nonzero")
        if isinstance(c, Circle):
            if zero3(c.n):
                _lies("a circle normal must be nonzero")
            if not c.r > 0:
                _lies(f"a circle radius must be positive, got {c.r}")
            if zero3(c.ref) or dot(c.ref, c.n) != 0:
                _lies("a circle's ref direction must be nonzero and "
                      "perpendicular to its normal")

    def on_surface(pt, s):
        """Does the point satisfy the surface's implicit equation, exactly?

        Axis forms are cleared of division: with u = pt − p and d the (not
        necessarily unit) axis, radial²·|d|² = |u|²|d|² − (u·d)².
        """
        if isinstance(s, Plane):
            return dot(s.n, pt) == s.d
        if isinstance(s, Cylinder):
            u = sub(pt, s.p)
            d2, ax = dot(s.d, s.d), dot(u, s.d)
            return dot(u, u) * d2 - ax * ax == s.r * s.r * d2
        if isinstance(s, Cone):
            u = sub(pt, s.p)
            d2, ax = dot(s.d, s.d), dot(u, s.d)
            return (dot(u, u) * d2 - ax * ax
                    == s.tan_half * s.tan_half * ax * ax)
        if isinstance(s, SphereS):
            u = sub(pt, s.c)
            return dot(u, u) == s.r * s.r
        return True                              # unknown surface: no claim

    def circle_on_surface(cv, s):
        """Does the WHOLE circle lie on the surface? Closed forms, no
        parametrization: a circle lies on a quadric-of-revolution iff it is a
        cross-section (normal ∥ axis, centre on axis, radius matching), and on
        a sphere iff its axis passes through the centre with h² + r² = R²."""
        if isinstance(s, Plane):
            return zero3(cross(cv.n, s.n)) and dot(s.n, cv.c) == s.d
        if isinstance(s, Cylinder):
            u = sub(cv.c, s.p)
            d2, ax = dot(s.d, s.d), dot(u, s.d)
            return (zero3(cross(cv.n, s.d)) and cv.r == s.r
                    and dot(u, u) * d2 == ax * ax)
        if isinstance(s, Cone):
            u = sub(cv.c, s.p)
            d2, ax = dot(s.d, s.d), dot(u, s.d)
            return (zero3(cross(cv.n, s.d))
                    and dot(u, u) * d2 == ax * ax
                    and cv.r * cv.r * d2 == s.tan_half * s.tan_half * ax * ax)
        if isinstance(s, SphereS):
            u = sub(cv.c, s.c)
            return (zero3(cross(u, cv.n))
                    and dot(u, u) + cv.r * cv.r == s.r * s.r)
        return True

    def on_curve(pt, cv):
        if isinstance(cv, Line):
            return zero3(cross(sub(pt, cv.p), cv.d))
        if isinstance(cv, Circle):
            u = sub(pt, cv.c)
            return dot(u, cv.n) == 0 and dot(u, u) == cv.r * cv.r
        return True

    for f in body.faces:
        check_surface(f.surface)
        kind = type(f.surface).__name__.lower()
        for lp in f.loops:
            for e in lp.edges:
                check_curve(e.curve)
                for v in (e.v0, e.v1):
                    if not on_curve(v, e.curve):
                        _lies("an edge endpoint does not lie on its own "
                              f"{type(e.curve).__name__.lower()}")
                    if not on_surface(v, f.surface):
                        _lies("an edge endpoint does not lie on the face's "
                              f"{kind} surface")
                if (isinstance(e.curve, Circle)
                        and not circle_on_surface(e.curve, f.surface)):
                    _lies(f"a rim circle does not lie on the face's {kind} "
                          "surface")


def loads_body(text: str):
    """Parse a ``forge/body@1`` document.

    A document is UNTRUSTED input (ADR-0006/0007). Every shape error here used
    to escape as a bare KeyError/TypeError/ZeroDivisionError, and some nonsense
    loaded SILENTLY — a negative surd radicand, a one-component plane normal,
    a ``sense`` of ``"yes"`` that is merely truthy and then reported a volume
    as fact. Validate the shape, then refuse by name.
    """
    from forgekernel.body import (Body, Circle, Cone, Cylinder, Edge, Face,
                                  Line, Loop, Plane, SphereS)
    from forgekernel.surd import SurdVal

    doc = json.loads(text)
    if not isinstance(doc, dict):
        _bad("the document is not an object")
    if doc.get("schema") != BODY_SCHEMA:
        raise ValueError(f"unsupported body schema {doc.get('schema')!r}")

    def need(d, key, kind):
        if not isinstance(d, dict) or key not in d:
            _bad(f"missing {key!r}")
        v = d[key]
        if not isinstance(v, kind) or (kind is not bool and isinstance(v, bool)):
            _bad(f"{key!r} has the wrong type")
        return v

    def scalar(raw):
        if not isinstance(raw, str):
            _bad("an exact number must be a string")
        try:
            out = _unnum(raw)
        except (ValueError, ZeroDivisionError, IndexError) as exc:
            _bad(f"unreadable exact number {raw!r} ({exc})")
        if isinstance(out, SurdVal) and out.d <= 0:
            _bad(f"a surd radicand must be positive, got {out.d}")
        return out

    def vec(raw):
        if not isinstance(raw, list) or len(raw) != 3:
            _bad("a vector needs exactly 3 components")
        return tuple(scalar(c) for c in raw)

    def surface(s):
        k = need(s, "kind", str)
        if k == "plane":
            return Plane(vec(need(s, "n", list)), scalar(need(s, "d", str)))
        if k == "cylinder":
            return Cylinder(vec(need(s, "p", list)), vec(need(s, "d", list)),
                            scalar(need(s, "r", str)))
        if k == "cone":
            return Cone(vec(need(s, "p", list)), vec(need(s, "d", list)),
                        scalar(need(s, "tan_half", str)))
        if k == "sphere":
            return SphereS(vec(need(s, "c", list)), scalar(need(s, "r", str)),
                           vec(need(s, "pole", list)) if "pole" in s else None)
        raise ValueError(f"unknown surface kind {k!r}")

    def curve(c):
        k = need(c, "kind", str)
        if k == "line":
            return Line(vec(need(c, "p", list)), vec(need(c, "d", list)))
        if k == "circle":
            return Circle(vec(need(c, "c", list)), vec(need(c, "n", list)),
                          vec(need(c, "ref", list)), scalar(need(c, "r", str)))
        raise ValueError(f"unknown curve kind {k!r}")

    def loop(lp):
        if not isinstance(lp, list):
            _bad("a loop must be a list of edges")
        return Loop(tuple(Edge(curve(need(e, "curve", dict)),
                               vec(need(e, "v0", list)),
                               vec(need(e, "v1", list))) for e in lp))

    def face(f):
        return Face(surface(need(f, "surface", dict)),
                    tuple(loop(lp) for lp in need(f, "loops", list)),
                    need(f, "sense", bool))

    body = Body(tuple(face(f) for f in need(doc, "faces", list)))
    _check_geometry(body)
    return body


def to_stl(solid: Solid, name: str = "forge") -> str:
    """ASCII STL from the exact tessellation (floats at the boundary)."""
    mesh = solid.tessellate()
    v = mesh["vertices"]
    out = [f"solid {name}"]
    for a, b, c in mesh["triangles"]:
        out += ["facet normal 0 0 0", "outer loop",
                f"vertex {v[a][0]:.9g} {v[a][1]:.9g} {v[a][2]:.9g}",
                f"vertex {v[b][0]:.9g} {v[b][1]:.9g} {v[b][2]:.9g}",
                f"vertex {v[c][0]:.9g} {v[c][1]:.9g} {v[c][2]:.9g}",
                "endloop", "endfacet"]
    out.append(f"endsolid {name}")
    return "\n".join(out) + "\n"
