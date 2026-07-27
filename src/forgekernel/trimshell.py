"""K7 gap 1 — TrimmedShell: a closed freeform solid of trimmed faces.

A boolean over freeform solids produces faces that are *pieces* of
surfaces — (surface, trim-loops) pairs. This module is the closed-solid
type those pieces assemble into, with the audit that stops a lying shell
at the door: the freeform analogue of the planar ``Body`` audit trio
(edge pairing / vector area / volume sign, see
:func:`forgekernel.body.manifold_violations` and
:func:`forgekernel.body.vector_area`).

**Identity is the object.** A trim edge is shared between the two faces
it bounds, but the faces read it in different parameter domains — and
their surface evaluations at a certified SSI point agree only to the
residual bound (≤ 1e-10), so no geometric edge key can pair freeform
edges exactly (the planar ``_edge_key`` idiom is unavailable). Instead a
:class:`TrimVertex` is minted once per intersection point and *bound* to
each face with that face's exact (u, v); loops are sequences of these
shared objects, and pairing is checked by identity — exact, no
tolerance, no interval.

**The three oracles** (each sees a class the others structurally miss):

1. *Edge pairing* (exact, at construction): every trim edge must be used
   by exactly two faces, and in opposite outward-boundary directions.
   The outward direction of a face's loop is its canonical parameter
   winding (region-on-left: outer CCW, holes CW) composed with the
   face's ``sense`` — the outward normal is ``sense·(S_u×S_v)``, and a
   sense of −1 flips the induced boundary orientation. A single flipped
   face, a dropped face, or an edge minted twice all fail here.
2. *Σ-flux orientation* (certified interval): ∮ n̂ dA over any closed
   surface is exactly zero, so Σ over faces of
   ``sense · certified_vector_area`` must CONTAIN (0,0,0). The trim
   boundary is only known as an SSI cell enclosure, so exact zero is
   impossible — the check is interval-aware: a bracket that certifiably
   excludes zero is a *proven* mis-orientation or mis-membership, while
   a true shell can never fail (every bracket contains its true value,
   and the truth is 0). This catches what pairing cannot: a face whose
   membership predicate disagrees with its loops (pairing never looks at
   the measure), or coherent-but-wrong region claims.
3. *Volume positivity* (certified interval): the certified volume
   bracket must exclude zero from below. An inside-out shell — every
   sense flipped, which pairing and vector area both accept by symmetry
   — is certified negative here. A bracket that straddles zero refuses
   (:class:`ShellAuditUncertified`, "raise depth"), never passes:
   tighten-or-refuse, ADR-0019.

Volume is Σ ``sense · certified_trim_flux`` over the faces — always a
``CInterval`` ("certified ± e"), never a bare float; provenance is
``"certified"`` the moment a trim strip exists (the exact tier has
nothing to say about a strip-bounded region).
"""

from __future__ import annotations

from fractions import Fraction

from forgekernel.bsolid import certified_trim_flux, certified_vector_area

F = Fraction


class ShellAuditError(ValueError):
    """A TrimmedShell audit found a CERTIFIED violation — unpaired or
    co-directed trim edges, a Σ-flux bracket excluding zero, or a
    certified non-positive volume. The shell is proven wrong, not merely
    unproven."""


class ShellAuditUncertified(ValueError):
    """Structured refusal: an audit check could neither pass nor fail at
    this depth (the certified bracket straddles the decision value).
    Tighten — raise the depth — or refuse; never a silent pass."""


class TrimVertex:
    """One shared point of an intersection curve. Identity is the
    object: the same vertex is bound to every face whose loop passes
    through it, each binding carrying that face's exact (u, v). Binding
    the same face twice with different coordinates refuses — a vertex
    that moves is two vertices."""

    __slots__ = ("_uv",)

    def __init__(self) -> None:
        self._uv: dict = {}

    def bind(self, face: "ShellFace", u, v) -> None:
        uv = (F(u), F(v))
        old = self._uv.get(id(face))
        if old is not None and old != uv:
            raise ValueError(
                f"TrimVertex already bound to this face at "
                f"({old[0]},{old[1]}) — refusing rebind to ({uv[0]},{uv[1]})")
        self._uv[id(face)] = uv

    def uv(self, face: "ShellFace"):
        try:
            return self._uv[id(face)]
        except KeyError:
            raise ValueError(
                "TrimVertex not bound to this face — every face using a "
                "vertex must bind its own exact (u, v)") from None


class ShellFace:
    """One trimmed face of a closed shell.

    ``surface`` — any patch the certified flux machinery accepts;
    ``sense`` — +1 if S_u×S_v is the outward normal, −1 if inward;
    ``strip`` — the SSI cell enclosure of this face's trim boundary on
    its own parameter domain (:func:`forgekernel.ssi.ssi_strips`), the
    rigorous object every certified measure is bracketed through;
    ``inside(u, v)`` — EXACT ℚ membership for points off the strip
    (e.g. which side of the other solid), constant on every strip-free
    connected region by the enclosure property.

    Loops (added via :meth:`add_loop`) are the shared-vertex topology:
    ``loops[0]`` the outer boundary, the rest holes, in the
    :class:`~forgekernel.trim.TrimmedPatch` convention. They carry the
    pairing audit; the *measure* is carried by strip + inside — the
    polyline is never integrated (the honest K7 caveat)."""

    __slots__ = ("surface", "sense", "strip", "inside", "loops", "roles")

    def __init__(self, surface, sense: int, strip, inside) -> None:
        if sense not in (1, -1):
            raise ValueError("sense must be +1 (S_u×S_v outward) or -1")
        self.surface = surface
        self.sense = sense
        self.strip = set(tuple(F(c) for c in cell) for cell in strip)
        self.inside = inside
        self.loops: list[list[TrimVertex]] = []
        self.roles: list[bool] = []

    def add_loop(self, verts_with_uv, outer: bool | None = None) -> None:
        """Append a trim loop as ``[(vertex, (u, v)), …]``; binds each
        vertex to this face at its exact coordinates.

        ``outer`` names the loop's role for the outward-boundary
        direction: ``True`` — the kept region lies INSIDE the loop
        (canonical CCW winding); ``False`` — the kept region lies
        outside and the loop is a hole in it (canonical CW). A boolean
        keeps either side of an intersection curve, so the role is data,
        not position; the default (``None``) keeps the original
        positional convention — first loop outer, the rest holes."""
        items = list(verts_with_uv)
        if len(items) < 3:
            raise ValueError("a trim loop needs >= 3 vertices")
        loop = []
        for vx, (u, v) in items:
            vx.bind(self, u, v)
            loop.append(vx)
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if a is b:
                raise ValueError("consecutive duplicate vertex in trim loop")
        self.roles.append(bool(outer) if outer is not None
                          else not self.loops)
        self.loops.append(loop)

    def _signed_area(self, loop) -> Fraction:
        s = F(0)
        m = len(loop)
        for k in range(m):
            (au, av) = loop[k].uv(self)
            (bu, bv) = loop[(k + 1) % m].uv(self)
            s += au * bv - bu * av
        return s / 2

    def outward_edges(self):
        """The face's directed trim edges in OUTWARD-boundary order:
        canonical region-on-left parameter winding (outer CCW, holes CW,
        per the loop's ``role``), reversed when ``sense`` is −1. Yields
        (vertex, vertex) pairs — object identity is the edge name."""
        for i, loop in enumerate(self.loops):
            ar = self._signed_area(loop)
            if ar == 0:
                raise ShellAuditError(
                    f"face loop {i}: zero parameter-space area — "
                    f"degenerate trim loop")
            want_ccw = self.roles[i]                 # outer CCW, holes CW
            seq = list(loop) if ((ar > 0) == want_ccw) else list(reversed(loop))
            if self.sense < 0:
                seq = list(reversed(seq))
            for a, b in zip(seq, seq[1:] + seq[:1]):
                yield a, b


class TrimmedShell:
    """A closed, outward-oriented solid bounded by trimmed faces.

    Construction runs the exact edge-pairing audit and REFUSES a shell
    that lies about being closed (the CLOSED_SHELL precedent: audit
    before anything downstream trusts the type). :meth:`audit` adds the
    certified Σ-flux and volume-sign oracles; :meth:`volume` is the
    certified bracket, never a bare float."""

    provenance = "certified"

    def __init__(self, faces) -> None:
        self.faces = list(faces)
        if not self.faces:
            raise ValueError("TrimmedShell needs at least one face")
        bad = self.pairing_violations()
        if bad:
            raise ShellAuditError(
                "trim-edge pairing failed — the shell is not a coherent "
                "closed surface:\n  " + "\n  ".join(bad[:12]))

    def pairing_violations(self) -> list[str]:
        """Every trim edge must be used by exactly two faces, once in
        each direction (outward boundaries of adjacent faces traverse a
        shared edge oppositely). Exact, by vertex object identity."""
        directed: dict = {}
        for fi, face in enumerate(self.faces):
            for a, b in face.outward_edges():
                directed.setdefault((id(a), id(b)), []).append((fi, a, b))
        bad = []
        seen = set()
        for (ia, ib), uses in sorted(directed.items()):
            key = (min(ia, ib), max(ia, ib))
            if key in seen:
                continue
            seen.add(key)
            rev = directed.get((ib, ia), [])
            total = len(uses) + len(rev)
            fi, a, b = uses[0]
            where = a.uv(self.faces[fi])
            if total != 2:
                bad.append(
                    f"edge at face {fi} uv ({where[0]},{where[1]}) used by "
                    f"{total} face-loop(s), not 2")
            elif not rev:
                bad.append(
                    f"edge at face {fi} uv ({where[0]},{where[1]}) traversed "
                    f"in the same direction by both faces — one face's "
                    f"orientation (sense or winding) is wrong")
        return bad

    def edge_count(self) -> int:
        seen = set()
        for face in self.faces:
            for a, b in face.outward_edges():
                seen.add((min(id(a), id(b)), max(id(a), id(b))))
        return len(seen)

    def volume(self, depth: int = 4):
        """Certified volume bracket: Σ sense·certified_trim_flux over
        the faces (each face's flux bracketed through its SSI strip).
        A ``CInterval`` — "certified ± e", per ADR-0019."""
        from forgekernel.interval import CInterval
        total = CInterval.exact(0)
        for face in self.faces:
            iv = certified_trim_flux(face.surface, face.strip, face.inside,
                                     depth=depth)
            total = total + iv * CInterval.exact(face.sense)
        return total

    def audit(self, depth: int = 4) -> dict:
        """The full three-oracle audit. Raises :class:`ShellAuditError`
        on a certified violation, :class:`ShellAuditUncertified` when a
        decision straddles at this depth; returns the report dict
        (faces, edges, vector_area brackets, volume bracket) on pass."""
        from forgekernel.interval import CInterval

        bad = self.pairing_violations()          # re-run: faces may mutate
        if bad:
            raise ShellAuditError(
                "trim-edge pairing failed:\n  " + "\n  ".join(bad[:12]))

        va = [CInterval.exact(0)] * 3
        for face in self.faces:
            comps = certified_vector_area(face.surface, face.strip,
                                          face.inside, depth=depth)
            sgn = CInterval.exact(face.sense)
            va = [a + c * sgn for a, c in zip(va, comps)]
        for axis, comp in zip("xyz", va):
            if comp.lo > 0 or comp.hi < 0:
                raise ShellAuditError(
                    f"vector area check failed: Σ ∮ n̂ dA {axis}-component "
                    f"is certifiably nonzero "
                    f"([{float(comp.lo):.6g}, {float(comp.hi):.6g}]) — a "
                    f"face is mis-oriented or its membership disagrees "
                    f"with its loops")

        vol = self.volume(depth=depth)
        if vol.hi <= 0:
            raise ShellAuditError(
                f"volume is certified non-positive "
                f"([{float(vol.lo):.6g}, {float(vol.hi):.6g}]) — the shell "
                f"is inside-out")
        if vol.lo <= 0:
            raise ShellAuditUncertified(
                f"volume bracket [{float(vol.lo):.6g}, {float(vol.hi):.6g}] "
                f"straddles zero at depth {depth} — positivity is not "
                f"certified; raise depth (tighten-or-refuse, never a "
                f"silent pass)")

        return {"faces": len(self.faces), "edges": self.edge_count(),
                "vector_area": tuple(va), "volume": vol}
