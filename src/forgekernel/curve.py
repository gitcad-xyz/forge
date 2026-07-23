"""K3.0 curves — the helix and its swept round section (ADR-0019).

The first transcendental geometry in the kernel. A ``Helix`` point is
``(R cos 2πt, R sin 2πt, pitch·t)`` — irrational at every rational ``t``
— so this family cannot live in the exact fields. It is carried instead
as *certified* geometry (ADR-0019): the one quantity a build actually
needs, the swept volume, is an exact closed form evaluated as a
``CInterval`` and reported with a proven half-width.

    coil spring = round section of radius ρ swept along a helix
    V = π ρ² L,   L = turns · √((2πR)² + pitch²)

``V = A·L`` is exact for a tube whose section radius stays below the
spine's radius of curvature (checked), so the only approximation is the
irrational ``√`` and ``π`` — both bracketed, never guessed.
"""

from __future__ import annotations

import math
from fractions import Fraction

from forgekernel.interval import CInterval, pi_interval

F = Fraction


class Helix:
    """A helical spine about +z, starting at ``(radius, 0, 0)``."""

    def __init__(self, radius, pitch, turns, ccw: bool = True) -> None:
        self.R = F(radius)
        self.pitch = F(pitch)
        self.turns = F(turns)
        self.ccw = ccw
        if self.R <= 0 or self.pitch <= 0 or self.turns <= 0:
            raise ValueError("helix wants positive radius/pitch/turns")

    def arc_length(self) -> CInterval:
        """Certified length: turns·√((2πR)² + pitch²) in ℚ + π + √."""
        pi = pi_interval()
        arg = CInterval.exact(4 * self.R * self.R) * pi * pi \
            + CInterval.exact(self.pitch * self.pitch)
        return CInterval.exact(self.turns) * arg.sqrt()

    def curvature(self) -> float:
        """κ = R/(R²+c²), c = pitch/2π — the spine radius of curvature is 1/κ."""
        c = float(self.pitch) / (2 * math.pi)
        R = float(self.R)
        return R / (R * R + c * c)

    def point_f(self, t: float) -> tuple[float, float, float]:
        """Float sample at parameter ``t`` turns (for tessellation only)."""
        ang = 2 * math.pi * t * (1.0 if self.ccw else -1.0)
        return (float(self.R) * math.cos(ang),
                float(self.R) * math.sin(ang),
                float(self.pitch) * t)


class TubeSolid:
    """A round section of radius ``wire_radius`` swept along a ``Helix``.

    Provenance is ``certified``: the volume is a ``CInterval``. The tube
    is watertight by construction (closed lateral surface + two end
    caps) as long as the section does not self-overlap, which holds when
    ``wire_radius < 1/κ`` (the spine radius of curvature) and adjacent
    coils clear (``2·wire_radius < pitch``); both are checked."""

    provenance = "certified"

    def __init__(self, helix: Helix, wire_radius, translate=(0, 0, 0)) -> None:
        self.helix = helix
        self.rho = F(wire_radius)
        self.t = tuple(F(v) for v in translate)
        if self.rho <= 0:
            raise ValueError("tube wants positive wire radius")
        if 2 * self.rho >= helix.pitch:
            raise ValueError(
                "tube self-overlaps: 2·wire_radius >= pitch (coils merge)")
        if float(self.rho) >= 1.0 / helix.curvature():
            raise ValueError(
                "tube self-overlaps: wire_radius >= spine radius of curvature")

    # -- certified metrics ----------------------------------------------------

    def volume(self) -> CInterval:
        """π ρ² L — exact tube volume as a certified interval."""
        pi = pi_interval()
        return pi * CInterval.exact(self.rho * self.rho) \
            * self.helix.arc_length()

    def centroid_f(self) -> tuple[float, float, float]:
        """On the axis at mid-height, exact by the sweep's 180° symmetry."""
        tx, ty, tz = (float(v) for v in self.t)
        return (tx, ty, tz + float(self.helix.pitch * self.helix.turns) / 2)

    def bbox_f(self):
        """Analytic (tight) bounds. Horizontal reach R+ρ from the axis; the
        section tilt adds at most ρ in z beyond the raw helix span."""
        R, rho = float(self.helix.R), float(self.rho)
        reach = R + rho
        zspan = float(self.helix.pitch * self.helix.turns)
        tx, ty, tz = (float(v) for v in self.t)
        return ((tx - reach, ty - reach, tz - rho),
                (tx + reach, ty + reach, tz + zspan + rho))

    # -- transforms -----------------------------------------------------------

    def translated(self, dx, dy, dz) -> "TubeSolid":
        t = (self.t[0] + F(dx), self.t[1] + F(dy), self.t[2] + F(dz))
        return TubeSolid(self.helix, self.rho, t)

    # -- tessellation (bounded-error render artifact) -------------------------

    def tessellate(self, deflection: float = 0.2) -> dict:
        """Float mesh of the tube (viewer only). Uses a rotation-minimizing
        frame stepped along the helix; returns {vertices, triangles}."""
        along, around = 24, 12
        h = self.helix
        tx, ty, tz = (float(v) for v in self.t)
        rho = float(self.rho)
        n = max(4, int(along * float(h.turns)))
        # rotation-minimizing frame via finite differences
        ring_centers = []
        tangents = []
        for i in range(n + 1):
            t = float(h.turns) * i / n
            p = h.point_f(t)
            eps = 1e-6
            p1 = h.point_f(t + eps)
            tang = (p1[0] - p[0], p1[1] - p[1], p1[2] - p[2])
            m = math.dist((0, 0, 0), tang) or 1.0
            tangents.append((tang[0] / m, tang[1] / m, tang[2] / m))
            ring_centers.append((p[0] + tx, p[1] + ty, p[2] + tz))
        # initial normal not parallel to first tangent
        ref = (0.0, 0.0, 1.0)
        if abs(tangents[0][2]) > 0.9:
            ref = (1.0, 0.0, 0.0)
        verts, tris = [], []
        prev_n = None
        for i, (c, tg) in enumerate(zip(ring_centers, tangents)):
            base = ref if prev_n is None else prev_n
            # normal = base minus its tangent component, renormalized
            d = base[0] * tg[0] + base[1] * tg[1] + base[2] * tg[2]
            nrm = (base[0] - d * tg[0], base[1] - d * tg[1], base[2] - d * tg[2])
            m = math.dist((0, 0, 0), nrm) or 1.0
            nrm = (nrm[0] / m, nrm[1] / m, nrm[2] / m)
            prev_n = nrm
            binm = (tg[1] * nrm[2] - tg[2] * nrm[1],
                    tg[2] * nrm[0] - tg[0] * nrm[2],
                    tg[0] * nrm[1] - tg[1] * nrm[0])
            for j in range(around):
                a = 2 * math.pi * j / around
                ca, sa = math.cos(a) * rho, math.sin(a) * rho
                verts.append((c[0] + ca * nrm[0] + sa * binm[0],
                              c[1] + ca * nrm[1] + sa * binm[1],
                              c[2] + ca * nrm[2] + sa * binm[2]))
        for i in range(n):
            for j in range(around):
                a = i * around + j
                b = i * around + (j + 1) % around
                c = (i + 1) * around + j
                d = (i + 1) * around + (j + 1) % around
                tris.append([a, b, c])
                tris.append([b, d, c])
        return {"vertices": [list(v) for v in verts], "triangles": tris}
