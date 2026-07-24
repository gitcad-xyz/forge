"""Tessellation of the analytic composites (K2.x) for the viewer.

Curved solids are exact objects; a mesh is a bounded-error VIEW of them.
``deflection`` is the max chord error (mm) at the boundary — the one
documented place a float enters, and only for display. Segment count is
derived so the chord error stays under deflection: for radius r,
N = ceil(pi / arccos(1 - deflection/r)).
"""

from __future__ import annotations

import math


def _nseg(radius: float, deflection: float) -> int:
    r = max(radius, 1e-9)
    if deflection >= r:
        return 8
    return max(8, math.ceil(math.pi / math.acos(max(-1.0, 1 - deflection / r))))


def lathe(profile_rz: list[tuple], deflection: float = 0.2,
          cx: float = 0.0, cy: float = 0.0) -> dict:
    """Mesh a surface of revolution from a closed (r, z) profile revolved
    about the z axis through (cx, cy). Deterministic; returns
    {vertices, triangles}."""
    rmax = max(r for r, _ in profile_rz) or 1.0
    n = _nseg(rmax, deflection)
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    ring_start: list[int] = []
    for r, z in profile_rz:
        base = len(verts)
        ring_start.append(base)
        if r == 0:
            verts.append([cx, cy, float(z)])
        else:
            for k in range(n):
                a = 2 * math.pi * k / n
                verts.append([cx + r * math.cos(a), cy + r * math.sin(a),
                              float(z)])
    m = len(profile_rz)
    for i in range(m):
        j = (i + 1) % m
        ra = profile_rz[i][0]
        rb = profile_rz[j][0]
        a0, b0 = ring_start[i], ring_start[j]
        if ra == 0 and rb == 0:
            continue
        for k in range(n):
            kn = (k + 1) % n
            if ra == 0:
                tris.append([a0, b0 + k, b0 + kn])
            elif rb == 0:
                tris.append([a0 + k, b0, a0 + kn])
            else:
                tris.append([a0 + k, b0 + k, b0 + kn])
                tris.append([a0 + k, b0 + kn, a0 + kn])
    # orient outward: the revolved profile's winding depends on its direction,
    # so flip the whole mesh if the signed volume came out negative (inward).
    if _signed_volume(verts, tris) < 0:
        tris = [[t[0], t[2], t[1]] for t in tris]
    return {"vertices": verts, "triangles": tris}


def _signed_volume(verts: list, tris: list) -> float:
    total = 0.0
    for a, b, c in tris:
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        total += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx)
                  + az * (bx * cy - by * cx))
    return total


def mesh_volume(mesh: dict) -> float:
    """Signed volume of a triangle mesh (divergence theorem) — the test
    hook proving the mesh approximates the exact solid."""
    v = mesh["vertices"]
    total = 0.0
    for a, b, c in mesh["triangles"]:
        ax, ay, az = v[a]
        bx, by, bz = v[b]
        cx, cy, cz = v[c]
        total += (ax * (by * cz - bz * cy)
                  - ay * (bx * cz - bz * cx)
                  + az * (bx * cy - by * cx))
    return abs(total) / 6.0
