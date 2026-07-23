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
