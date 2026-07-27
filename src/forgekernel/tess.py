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


def trimmed_shell_mesh(shell, depth: int = 4) -> dict:
    """Coarse, CERTIFIED-SAFE triangulation of a
    :class:`~forgekernel.trimshell.TrimmedShell` for the viewer.

    Per face, the dyadic 2^-depth grid is walked: a cell overlapping
    the face's SSI strip is never meshed — its membership fraction is
    unknown — and is reported as an explicit ``gaps`` quad instead; a
    strip-free cell is meshed iff the face's exact membership predicate
    keeps it. So no triangle covers a point whose membership is
    uncertain: the mesh shows exactly what is certified, and shows the
    uncertainty band as a band. NOT watertight by design (the honest
    K7 caveat rendered visible); a render artifact, never a measure —
    ``provenance`` says ``"render"``.

    Pass the boolean's own ``depth`` so grid and strip cells align and
    the gap band is exactly one quad per strip cell."""
    from fractions import Fraction as F

    verts: list = []
    tris: list = []
    gaps: list = []
    for face in shell.faces:
        (u0, u1), (v0, v1) = face.surface.domain()
        u0, v0 = F(u0), F(v0)
        n = 1 << depth
        wu = (F(u1) - u0) / n
        wv = (F(v1) - v0) / n
        strip = [tuple(F(c) for c in cell) for cell in face.strip]

        corner_idx: dict = {}

        def corner(i, j, face=face, u0=u0, v0=v0, wu=wu, wv=wv,
                   corner_idx=corner_idx):
            key = (i, j)
            if key not in corner_idx:
                p = face.surface.eval(u0 + wu * i, v0 + wv * j)
                corner_idx[key] = len(verts)
                verts.append([float(c) for c in p])
            return corner_idx[key]

        for i in range(n):
            for j in range(n):
                c0, c1 = u0 + wu * i, u0 + wu * (i + 1)
                d0, d1 = v0 + wv * j, v0 + wv * (j + 1)
                if any(s0 < c1 and c0 < s1 and t0 < d1 and d0 < t1
                       for (s0, s1, t0, t1) in strip):
                    quad = [face.surface.eval(uu, vv)
                            for uu, vv in ((c0, d0), (c1, d0),
                                           (c1, d1), (c0, d1))]
                    gaps.append([[float(c) for c in p] for p in quad])
                    continue
                if not face.inside((c0 + c1) / 2, (d0 + d1) / 2):
                    continue
                a = corner(i, j)
                b = corner(i + 1, j)
                c = corner(i + 1, j + 1)
                d = corner(i, j + 1)
                if face.sense > 0:
                    tris.append([a, b, c])
                    tris.append([a, c, d])
                else:
                    tris.append([a, c, b])
                    tris.append([a, d, c])
    return {"vertices": verts, "triangles": tris, "gaps": gaps,
            "provenance": "render"}


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
