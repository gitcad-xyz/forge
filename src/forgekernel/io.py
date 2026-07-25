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
            return {"kind": "sphere", "c": _vec(s.c), "r": _num(s.r)}
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
            return SphereS(vec(need(s, "c", list)), scalar(need(s, "r", str)))
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

    return Body(tuple(face(f) for f in need(doc, "faces", list)))


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
