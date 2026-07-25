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

    doc = {"schema": BODY_SCHEMA, "faces": [
        {"surface": surface(f.surface), "sense": bool(f.sense),
         "loops": [[{"curve": curve(e.curve), "v0": _vec(e.v0),
                     "v1": _vec(e.v1)} for e in lp.edges] for lp in f.loops]}
        for f in body.faces]}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"


def loads_body(text: str):
    from forgekernel.body import (Body, Circle, Cone, Cylinder, Edge, Face,
                                  Line, Loop, Plane, SphereS)

    doc = json.loads(text)
    if doc.get("schema") != BODY_SCHEMA:
        raise ValueError(f"unsupported body schema {doc.get('schema')!r}")

    def vec(v):
        return tuple(_unnum(x) for x in v)

    def surface(s):
        k = s["kind"]
        if k == "plane":
            return Plane(vec(s["n"]), _unnum(s["d"]))
        if k == "cylinder":
            return Cylinder(vec(s["p"]), vec(s["d"]), _unnum(s["r"]))
        if k == "cone":
            return Cone(vec(s["p"]), vec(s["d"]), _unnum(s["tan_half"]))
        if k == "sphere":
            return SphereS(vec(s["c"]), _unnum(s["r"]))
        raise ValueError(f"unknown surface kind {k!r}")

    def curve(c):
        if c["kind"] == "line":
            return Line(vec(c["p"]), vec(c["d"]))
        return Circle(vec(c["c"]), vec(c["n"]), vec(c["ref"]), _unnum(c["r"]))

    return Body(tuple(
        Face(surface(f["surface"]),
             tuple(Loop(tuple(Edge(curve(e["curve"]), vec(e["v0"]),
                                   vec(e["v1"])) for e in lp))
                   for lp in f["loops"]),
             f["sense"])
        for f in doc["faces"]))


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
