"""The K1 facade — pure-Python exact planar kernel operations.

gitcad adapts this to its Kernel seam in a thin shim (gitcad.kernel.ref);
nothing here depends on gitcad. Unsupported operator classes raise
NotImplementedError with the stage that will bring them — honest refusal
is part of the contract.
"""

from __future__ import annotations

from forgekernel import csg
from forgekernel.brep import Solid
from forgekernel.exact import F, vec

__all__ = ["Solid", "box", "prism", "boolean", "translate", "scale",
           "mirror", "rotate_quarter"]


def box(dx, dy, dz) -> Solid:
    return Solid.box(dx, dy, dz)


def prism(loop_xy, height, source_prefix: str = "prism") -> Solid:
    return Solid.prism(loop_xy, height, source_prefix)


def boolean(op: str, a: Solid, b: Solid) -> Solid:
    fn = {"union": csg.union, "cut": csg.cut, "intersect": csg.intersect}.get(op)
    if fn is None:
        raise ValueError(f"unknown boolean op {op!r} (union|cut|intersect)")
    out = fn(a, b)
    bad = out.watertight_violations()
    if bad:
        raise ArithmeticError(f"boolean produced non-watertight result: {bad}")
    return out


def translate(s: Solid, x, y, z) -> Solid:
    return s.translated(vec(F(x), F(y), F(z)))


def scale(s: Solid, fx, fy=None, fz=None) -> Solid:
    return s.scaled(fx, fx if fy is None else fy, fx if fz is None else fz)


def mirror(s: Solid, axis: str) -> Solid:
    return s.mirrored(axis)


def rotate_quarter(s: Solid, axis: str, quarters: int) -> Solid:
    return s.rotated_quarter(axis, quarters)


def chamfer(s: Solid, distance) -> Solid:
    """Exact chamfer on all convex rational-normal edges, with industrial
    corner-triangle vertex truncation (oracle-matched semantics)."""
    from forgekernel.brep import chamfer_corners, chamfer_planar, logical_edges

    edges = logical_edges(s)
    out = chamfer_planar(s, distance, edges)
    return chamfer_corners(out, distance, edges)


def draft(s: Solid, angle_deg: float, neutral_z=0, faces=None) -> Solid:
    """Draft vertical faces by angle (tan converted exactly at input)."""
    import math as _m

    from forgekernel.brep import draft_box
    from forgekernel.exact import F

    t = F(_m.tan(_m.radians(angle_deg)))
    return draft_box(s, t, F(neutral_z))


def shell(s: Solid, thickness) -> Solid:
    """Hollow to a wall thickness (closed shell, exact for box solids)."""
    from forgekernel.brep import shell_box

    return shell_box(s, thickness)
