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


def _cos_sin_deg(deg):
    """Exact (cos, sin) of an angle in degrees, in ℚ[√d] — or None if the angle
    needs a larger field. Covers the multiples of 30° and 45° (so circular
    patterns of 3/4/6/8/12 and the diagonal sketch-plane rotations are exact)."""
    from forgekernel.surd import SurdVal

    from fractions import Fraction

    d = int(deg) % 360
    if deg != int(deg):
        return None
    if d % 90 == 0:
        return [(F(1), F(0)), (F(0), F(1)), (F(-1), F(0)), (F(0), F(-1))][d // 90]
    h = Fraction(1, 2)
    r3, r2 = SurdVal(0, h, 3), SurdVal(0, h, 2)      # √3/2, √2/2
    table = {30: (r3, h), 60: (h, r3), 120: (-h, r3), 150: (-r3, h),
             210: (-r3, -h), 240: (-h, -r3), 300: (h, -r3), 330: (r3, -h),
             45: (r2, r2), 135: (-r2, r2), 225: (-r2, -r2), 315: (r2, -r2)}
    return table.get(d)


def _rotation_matrix(axis, deg):
    """Exact Rodrigues rotation matrix (list of 3 rows) about ``axis`` by
    ``deg``, over ℚ[√d]. Raises ValueError if the angle is not representable."""
    from forgekernel.surd import SurdVal, sqrt_rational

    cs = _cos_sin_deg(deg)
    if cs is None:
        raise ValueError(
            f"rotation by {deg}° is not exactly representable in ℚ[√d] "
            "(only multiples of 30° and 45°)")
    c, s = cs
    ax, ay, az = F(axis[0]), F(axis[1]), F(axis[2])
    n2 = ax * ax + ay * ay + az * az                 # |axis|², rational
    if n2 == 0:
        raise ValueError("zero rotation axis")
    one = SurdVal(1, 0, 1)
    s_over_n = (s if isinstance(s, SurdVal) else SurdVal(s)) / sqrt_rational(n2)
    k2c = (one - (c if isinstance(c, SurdVal) else SurdVal(c))) / n2
    k = [[F(0), -az, ay], [az, F(0), -ax], [-ay, ax, F(0)]]          # [axis]×
    k2 = [[sum(k[i][t] * k[t][j] for t in range(3)) for j in range(3)]
          for i in range(3)]
    return [[(F(1) if i == j else F(0)) + s_over_n * k[i][j] + k2c * k2[i][j]
             for j in range(3)] for i in range(3)]


def rotate(s: Solid, axis, deg) -> Solid:
    """Exact rotation of a planar solid about an arbitrary axis by an angle in
    ℚ[√d] (multiples of 30°/45°). Rigid motion — volume is preserved exactly and
    the result is a Solid with exact ℚ[√d] coordinates. Raises ValueError for
    angles a larger field would be needed for (the seam turns that into an
    honest refusal)."""
    r = _rotation_matrix(axis, deg)

    def fn(v):
        x, y, z = v
        return (r[0][0] * x + r[0][1] * y + r[0][2] * z,
                r[1][0] * x + r[1][1] * y + r[1][2] * z,
                r[2][0] * x + r[2][1] * y + r[2][2] * z)

    return s.mapped(fn)


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


def fillet_box(a, b, c, r, origin=(0, 0, 0)):
    """Rounded box (all edges+corners filleted r) — exact Q[pi] volume."""
    from forgekernel.quadric import RoundedBox

    return RoundedBox(a, b, c, r, origin)


def sweep(profile_area, path):
    """Mitered sweep of a convex profile — exact volume in Q[sqrt d]."""
    from forgekernel.quadric import MiteredSweep

    return MiteredSweep(profile_area, path)
