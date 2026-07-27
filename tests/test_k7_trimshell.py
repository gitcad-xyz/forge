"""K7 gap 1 — TrimmedShell: a closed freeform solid of trimmed faces.

The type carries (surface, trim-loops) faces whose trim-loop vertices are
SHARED OBJECTS across faces (identity-by-construction — the two surfaces'
evaluations at a certified SSI point differ by <= 1e-10, so no geometric
edge key could pair freeform edges exactly). Three oracles, the freeform
analogues of the planar Body audit trio:

  1. edge pairing — every trim edge is used by exactly two faces, in
     opposite outward-boundary directions (exact, by object identity);
  2. Σ-flux orientation — Σ over faces of ∮∮ sense·(S_u×S_v) du dv must
     vanish on a closed shell; computed as a certified interval through
     the SSI strip, so the check is CONTAINS-ZERO (interval-aware where
     exact is impossible) and a certified exclusion of zero is a proven
     mis-orientation;
  3. volume positivity — the certified volume bracket must exclude zero
     from below; an inside-out shell is certified negative.

Verification: one TrimmedShell hand-built from the dome ∩ half-space
trim loops (the stage-1 case). Its audit passes and its certified volume
bracket contains the independent stage-1 reference 1.0794724554159079.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from forgekernel.interval import CInterval
from forgekernel.nurbs import bezier_surface
from forgekernel.bsolid import certified_vector_area, patch_flux
from forgekernel.ssi import ssi_strips, ssi_trim_loops
from forgekernel.trimshell import (ShellAuditError, ShellAuditUncertified,
                                   ShellFace, TrimmedShell, TrimVertex)

# the derivation's dome: z = 24 u(1-u) v(1-v) over [0,4]^2, peak 1.5
DOME = bezier_surface([[(0, 0, 0), (0, 2, 0), (0, 4, 0)],
                       [(2, 0, 0), (2, 2, 6), (2, 4, 0)],
                       [(4, 0, 0), (4, 2, 0), (4, 4, 0)]])
PLANE = bezier_surface([[(0, 0, 1), (0, 4, 1)], [(4, 0, 1), (4, 4, 1)]])

# independent high-accuracy reference for the cap volume (stage 1):
# V = (8/3)·∫ a(u)·s(u)^3 du, a = 24u(1-u), s = √(1-4/a), Simpson 2·10^6
CAP_REF = F(10794724554159, 10 ** 13)          # 1.0794724554159079


def _above(u, v) -> bool:
    """Exact membership vs the half-space z >= 1 (dome and plane share
    the same parameter footprint: (u,v) and (s,t) map to the same (x,y))."""
    return DOME.eval(u, v)[2] > 1


def _cap_pieces(depth: int = 5):
    out = ssi_trim_loops(DOME, PLANE, depth=depth)
    assert len(out["loops"]) == 1 and out["loops"][0]["closed"]
    chain = out["loops"][0]["points"]
    a_cells, b_cells = ssi_strips(DOME, PLANE, depth=depth)
    return chain, a_cells, b_cells


def _cap_shell(depth: int = 5, plane_sense: int = -1,
               plane_inside=None) -> TrimmedShell:
    """The dome ∩ half-space cap as a TrimmedShell: dome face (outward
    +z, sense +1) and plane face (outward -z, sense -1), sharing one
    closed trim loop of TrimVertex objects."""
    chain, a_cells, b_cells = _cap_pieces(depth)
    verts = [TrimVertex() for _ in chain]
    top = ShellFace(DOME, +1, a_cells, _above)
    bot = ShellFace(PLANE, plane_sense, b_cells,
                    plane_inside if plane_inside is not None else _above)
    top.add_loop([(vx, (u, v)) for vx, (u, v, _, _) in zip(verts, chain)])
    bot.add_loop([(vx, (s, t)) for vx, (_, _, s, t) in zip(verts, chain)])
    return TrimmedShell([top, bot])


# -- TrimVertex: identity is the object ---------------------------------------

def test_vertex_rebinding_a_face_to_a_different_uv_refuses() -> None:
    vx = TrimVertex()
    face = ShellFace(PLANE, -1, set(), _above)
    vx.bind(face, F(1, 2), F(1, 2))
    vx.bind(face, F(1, 2), F(1, 2))                 # same value: fine
    with pytest.raises(ValueError, match="already bound"):
        vx.bind(face, F(1, 3), F(1, 2))


def test_vertex_uv_on_unbound_face_refuses_by_name() -> None:
    vx = TrimVertex()
    face = ShellFace(PLANE, -1, set(), _above)
    with pytest.raises(ValueError, match="not bound"):
        vx.uv(face)


# -- oracle 1: edge pairing (exact, object identity) --------------------------

def test_open_shell_refuses_at_construction() -> None:
    # the cap top alone: every trim edge used once, not twice
    chain, a_cells, _ = _cap_pieces(depth=4)
    verts = [TrimVertex() for _ in chain]
    top = ShellFace(DOME, +1, a_cells, _above)
    top.add_loop([(vx, (u, v)) for vx, (u, v, _, _) in zip(verts, chain)])
    with pytest.raises(ShellAuditError, match="used by 1 face"):
        TrimmedShell([top])


def test_one_flipped_sense_breaks_pairing_at_construction() -> None:
    # plane face claiming outward +z: both faces then traverse the shared
    # loop the same way — the pairing audit sees it exactly
    with pytest.raises(ShellAuditError, match="direction"):
        _cap_shell(depth=4, plane_sense=+1)


def test_degenerate_loop_refuses() -> None:
    face = ShellFace(PLANE, -1, set(), _above)
    verts = [TrimVertex(), TrimVertex()]
    with pytest.raises(ValueError, match=">= 3"):
        face.add_loop([(verts[0], (0, 0)), (verts[1], (1, 0))])


# -- oracle 2: Σ-flux orientation (certified interval) ------------------------

def test_certified_vector_area_exact_when_strip_empty() -> None:
    # full dome patch, no trim: ∮∮ S_u×S_v du dv, z-component exactly
    # (domain area 1 mapped over the 4×4 footprint ⇒ 16), zero width
    va = certified_vector_area(DOME, [], lambda u, v: True, depth=2)
    assert all(isinstance(c, CInterval) for c in va)
    assert va[2].width == 0 and va[2].lo == 16
    assert va[0].width == 0 and va[0].lo == 0
    assert va[1].width == 0 and va[1].lo == 0


def test_cap_shell_audit_passes_and_reports() -> None:
    shell = _cap_shell(depth=5)
    report = shell.audit(depth=5)
    assert report["faces"] == 2
    assert report["edges"] > 0
    # a true closed shell can never certified-fail: each component's
    # bracket contains its true value, and the true Σ is exactly 0
    for comp in report["vector_area"]:
        assert comp.lo <= 0 <= comp.hi
    assert report["volume"].lo > 0


def test_lying_membership_is_caught_by_sigma_flux() -> None:
    # pairing cannot see this: the loops are untouched, but the plane
    # face claims its whole domain (disk ignored). Σ n̂ dA then excludes
    # zero — a certified mis-orientation/mis-measure.
    shell = _cap_shell(depth=5, plane_inside=lambda s, t: True)
    with pytest.raises(ShellAuditError, match="vector area"):
        shell.audit(depth=5)


# -- oracle 3: volume positivity ----------------------------------------------

def test_inside_out_shell_is_certified_negative() -> None:
    # both senses flipped: pairing stays coherent (still opposite), the
    # vector area still contains zero — only the volume sign sees it
    chain, a_cells, b_cells = _cap_pieces(depth=5)
    verts = [TrimVertex() for _ in chain]
    top = ShellFace(DOME, -1, a_cells, _above)
    bot = ShellFace(PLANE, +1, b_cells, _above)
    top.add_loop([(vx, (u, v)) for vx, (u, v, _, _) in zip(verts, chain)])
    bot.add_loop([(vx, (s, t)) for vx, (_, _, s, t) in zip(verts, chain)])
    shell = TrimmedShell([top, bot])
    with pytest.raises(ShellAuditError, match="volume"):
        shell.audit(depth=5)


def test_audit_refuses_rather_than_pass_when_positivity_uncertified() -> None:
    # at depth 4 the bracket width (~4) straddles zero: tighten-or-refuse,
    # never a silent pass (ADR-0019)
    shell = _cap_shell(depth=4)
    vol = shell.volume(depth=4)
    if vol.lo > 0:                       # tightness improved: nothing to test
        pytest.skip("depth-4 bracket already certifies positivity")
    with pytest.raises(ShellAuditUncertified, match="raise depth"):
        shell.audit(depth=4)


# -- the deliverable: certified volume bracket contains the reference ---------

def test_cap_volume_bracket_contains_stage1_reference() -> None:
    shell = _cap_shell(depth=5)
    assert shell.provenance == "certified"
    widths = []
    for depth in (4, 5):
        vol = shell.volume(depth=depth)
        assert isinstance(vol, CInterval)
        assert vol.lo <= CAP_REF <= vol.hi
        widths.append(vol.width)
    assert widths[1] < widths[0]         # deeper strip → tighter bracket


def test_shell_volume_matches_direct_certified_trim_flux() -> None:
    # the shell's volume is exactly the certified_trim_flux assembly the
    # stage-1 test pinned — same numbers, now behind an audited type
    from forgekernel.bsolid import certified_trim_flux
    shell = _cap_shell(depth=4)
    a_cells, b_cells = ssi_strips(DOME, PLANE, depth=4)
    f_dome = certified_trim_flux(DOME, a_cells, _above, depth=4)
    f_plane = certified_trim_flux(PLANE, b_cells, _above, depth=4)
    ref = f_dome - f_plane
    vol = shell.volume(depth=4)
    assert vol.lo == ref.lo and vol.hi == ref.hi
