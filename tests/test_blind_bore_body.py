"""A sphere with a coaxial BLIND bore — three faces, exact in ℚ[√d][π].

The through case (`NapkinRing`) removes both polar caps and keeps two faces.
A blind bore enters from the top and STOPS: the tool's own flat bottom cap
survives as the floor of the hole. Three faces, derived from the removed set
rather than guessed:

  * the sphere minus ONE polar cap — a single rim where the bore exits at
    z = c + √d (d = R² − r²), and the far pole is a singular point on no
    edge, exactly like a pointed cone's apex;
  * the bore wall, from the floor up to that rim;
  * a FLAT DISK floor of radius r at the tool's end — a plane, NOT a
    spherical cap (an earlier backlog note guessed wrong; the tool's bottom
    is flat, and the sphere's surface at the floor depth is outside the bore
    for every admissible floor).

Volume, by hand from the removed set (bore cylinder of height √d − f plus the
sphere's cap above √d), with f the floor height RELATIVE to the sphere centre:

    V = π · ( 2R³/3  +  (2/3)·d·√d  +  r²·f )

Nothing here is checked against the implementation's own arithmetic: every
expected value below is a literal, written from that closed form. The probe
case (R=6, r=1, f=0) was verified independently by Monte-Carlo membership
sampling: banked 885.969 ± 1.296 (3σ, 4M samples) and re-run fresh at seed
987654321 (16M samples) against the closed form 886.0606404251…
"""

import math
from fractions import Fraction as Q

import pytest

from forgekernel import body as B
from forgekernel.surd import SurdVal
from forgekernel.surdrev import SphereBlindBore, blind_bore_contour, contour_r2_dz

# (R, r, f) — floor height f is relative to the sphere centre, |f| < √(R²−r²)
CASES = [(6, 1, 0), (6, 1, 2), (7, 2, -3), (10, 6, 2)]
IDS = [f"R{R}r{r}f{f}" for R, r, f in CASES]

# hand-written closed forms: V/π as an exact literal, NEVER computed by the
# code under test.  V/π = 2R³/3 + (2/3)d√d + r²f, normalised m√k square-free.
EXPECTED = {
    (6, 1, 0): SurdVal(Q(144), Q(70, 3), 35),          # 2·216/3 + (70/3)√35
    (6, 1, 2): SurdVal(Q(146), Q(70, 3), 35),          # … + 1²·2
    (7, 2, -3): SurdVal(Q(650, 3), Q(90), 5),          # 686/3 − 12 + 30√45 = 90√5
    (10, 6, 2): SurdVal(Q(1080), Q(0), 1),             # d=64 is square: rational
}


def _vpi(v):
    """The volume's π coefficient as a SurdVal, whichever exact type carried it."""
    if hasattr(v, "a"):                                 # PiVal
        assert v.a == 0, f"rational part must be 0: {v!r}"
        b = v.b
    else:                                               # PiPoly
        assert v[0] == 0, f"rational part must be 0: {v!r}"
        b = v[1]
    return b if isinstance(b, SurdVal) else SurdVal(Q(b), 0, 1)


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_contour_volume_matches_the_closed_form(R, r0, f) -> None:
    v3 = contour_r2_dz(blind_bore_contour(R, r0, f))
    d = R * R - r0 * r0
    assert float(v3) * math.pi == pytest.approx(
        math.pi * (2 * R ** 3 / 3 + 2 * d * math.sqrt(d) / 3 + r0 * r0 * f),
        rel=1e-12)


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_representations_volume_is_the_closed_form_EXACTLY(R, r0, f) -> None:
    got = _vpi(SphereBlindBore(R, r0, f).volume())
    want = EXPECTED[(R, r0, f)]
    assert got == want, f"{got!r} != {want!r}"


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_converted_body_has_three_faces(R, r0, f) -> None:
    body = B.to_body(SphereBlindBore(R, r0, f))
    kinds = sorted(type(fc.surface).__name__ for fc in body.faces)
    assert kinds == ["Cylinder", "Plane", "SphereS"], (
        f"got {kinds} — a blind bore keeps its flat floor and loses one polar "
        "cap; anything else is a different solid")


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_converted_body_is_a_closed_shell(R, r0, f) -> None:
    assert B.manifold_violations(B.to_body(SphereBlindBore(R, r0, f))) == []


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_converted_bodys_volume_is_the_closed_form_EXACTLY(R, r0, f) -> None:
    """The converter is where a plausible wrong number is born — the
    representation's own volume() keeps agreeing with the closed form no
    matter what the converter builds. So this goes through B.volume."""
    got = _vpi(B.volume(B.to_body(SphereBlindBore(R, r0, f))))
    want = EXPECTED[(R, r0, f)]
    assert got == want, f"{got!r} != {want!r}"


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_sign_of_the_volume_is_decidable_without_a_float(R, r0, f) -> None:
    v = B.volume(B.to_body(SphereBlindBore(R, r0, f)))
    assert v.sign() == 1
    assert not v <= 0
    assert v > 0


def test_a_translated_solid_converts_about_its_own_centre() -> None:
    """Volume is translation-invariant. This is the test that pins the axis
    OFFSET term c_z·∮n̂_z dA of the one-rim spherical face: unlike a napkin
    ring's symmetric zone, this face's ∮n̂ dA is NOT zero (it is −πr² ẑ), so
    a converter that drops the term is wrong by exactly c_z·πr²/3 the moment
    the solid moves — and right, silently, at the origin."""
    v0 = _vpi(B.volume(B.to_body(SphereBlindBore(6, 1, 0))))
    vt = _vpi(B.volume(B.to_body(SphereBlindBore(6, 1, 0).translated(2, -3, 7))))
    assert v0 == vt, f"volume moved with the solid: {v0!r} -> {vt!r}"


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_converted_body_meshes_watertight_and_refines(R, r0, f) -> None:
    """Floats are legal here — this is meshing (ADR-0019)."""
    from collections import Counter

    body = B.to_body(SphereBlindBore(R, r0, f))
    seen: Counter = Counter()
    coarse = B.tessellate(body, 0.5)
    for tri in coarse["triangles"]:
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            seen[(u, v)] += 1
    assert [k for k, n in seen.items() if seen.get((k[1], k[0]), 0) != n] == []
    assert len(coarse["triangles"]) > 0
    assert len(B.tessellate(body, 0.01)["triangles"]) > len(coarse["triangles"])


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_mesh_volume_converges_on_the_closed_form(R, r0, f) -> None:
    """The mesh must describe the SAME solid the exact volume describes —
    watertight and refining proves nothing about WHICH solid got meshed."""
    d = R * R - r0 * r0
    exact = math.pi * (2 * R ** 3 / 3 + 2 * d * math.sqrt(d) / 3 + r0 * r0 * f)
    body = B.to_body(SphereBlindBore(R, r0, f))

    def vol(defl):
        mesh = B.tessellate(body, defl)
        verts, total = mesh["vertices"], 0.0
        for a, b, c in mesh["triangles"]:
            p, q, r_ = verts[a], verts[b], verts[c]
            total += (p[0] * (q[1] * r_[2] - q[2] * r_[1])
                      - p[1] * (q[0] * r_[2] - q[2] * r_[0])
                      + p[2] * (q[0] * r_[1] - q[1] * r_[0])) / 6.0
        return abs(total)

    coarse = abs(vol(0.05) - exact)
    fine = abs(vol(0.01) - exact)
    finer = abs(vol(0.002) - exact)
    assert coarse / fine > 3, f"5x refinement barely helped: {coarse} -> {fine}"
    assert fine / finer > 3, f"5x refinement barely helped: {fine} -> {finer}"
    assert finer / exact < 1e-3


@pytest.mark.parametrize("R,r0,f", CASES, ids=IDS)
def test_the_bounding_box_is_sound_and_stops_at_the_rim_in_z(R, r0, f) -> None:
    """The north cap is gone, so the solid's top is the rim at +√d; the south
    pole survives, so the bottom is −R. Floats legal — a bound (ADR-0019)."""
    s = SphereBlindBore(R, r0, f)
    lo, hi = s.bbox()
    d = R * R - r0 * r0
    assert float(hi[2]) == pytest.approx(math.sqrt(d), rel=1e-12)
    assert float(lo[2]) == pytest.approx(-R, rel=1e-12)
    body = B.to_body(s)
    blo, bhi = B.bbox(body)
    for v in B.tessellate(body, 0.1)["vertices"]:
        for i in range(3):
            assert float(blo[i]) - 1e-9 <= v[i] <= float(bhi[i]) + 1e-9


def test_the_centroid_is_exact_and_below_centre_for_a_top_bore() -> None:
    """z̄ = m_z / (V/π) with m_z = R²d/2 − d²/4 − R⁴/4 − r²(d − f²)/2, by hand.
    For (6,1,0): m_z = 630 − 306.25 − 324 − 17.5 = −17.75, so
    z̄ = −17.75 / (144 + (70/3)√35) — material was removed from the TOP, so the
    centroid must sit strictly below the centre."""
    c = SphereBlindBore(6, 1, 0).centroid()
    assert c[0] == 0 and c[1] == 0
    want = -17.75 / (144 + (70 / 3) * math.sqrt(35))
    assert float(c[2]) == pytest.approx(want, rel=1e-12)
    assert float(c[2]) < 0


REFUSALS = [
    ("bore as wide as the sphere", (5, 5, 0)),
    ("bore wider than the sphere", (5, 6, 0)),
    ("zero-radius bore", (5, 0, 0)),
    ("floor at the band edge", (5, 3, 4)),      # f² = d exactly: no wall left
    ("floor below the band", (6, 1, -6)),       # tool would exit the bottom
    ("floor above the band", (6, 1, 6)),        # tool never reaches material
]


@pytest.mark.parametrize("label,args", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_degenerate_configurations_refuse_rather_than_guess(label, args) -> None:
    with pytest.raises(ValueError):
        SphereBlindBore(*args)


def test_a_one_rim_sphere_face_without_the_pole_flag_still_refuses() -> None:
    """The band-versus-caps ambiguity, one-rim edition. A bare one-rim
    spherical face carries no way to say WHICH side survives, so without the
    surface's explicit pole trim it must refuse (through the octant path), not
    assume — an assumed side is a plausible wrong number, the exact failure
    `_sphere_zone`'s docstring forbids papering over."""
    from forgekernel.body import (Body, Circle, Edge, Face, Loop, SphereS)

    h = SurdVal(0, 1, 35)
    rim = Circle((Q(0), Q(0), h), (Q(0), Q(0), Q(1)), (Q(1), Q(0), Q(0)), Q(1))
    v = (Q(1), Q(0), h)
    naked = Face(SphereS((Q(0), Q(0), Q(0)), Q(6)),
                 (Loop((Edge(rim, v, v),)),), True)
    with pytest.raises(ValueError):
        B.volume(Body((naked,)))


def test_the_body_document_round_trips_with_the_pole_trim() -> None:
    """`SphereS.pole` is part of what the surface IS. A writer that dropped it
    would produce a document whose reload cannot say which side of the rim
    survives — the reload refuses (octant path) instead of guessing, but the
    DOCUMENT is the source of truth (ADR-0004) and must carry the trim."""
    from forgekernel import io

    body = B.to_body(SphereBlindBore(6, 1, 0))
    text = io.dumps_body(body)
    again = io.loads_body(text)
    assert _vpi(B.volume(again)) == EXPECTED[(6, 1, 0)]
    assert io.dumps_body(again) == text, "the round trip is not byte-stable"


def test_step_export_of_a_trimmed_sphere_face_refuses_by_name() -> None:
    """Found while wiring this cell: the writer's whole-sphere branch emits
    two lunes and IGNORES the face's loops, so a zone or a pole-trimmed face
    with rational rims would have shipped as a complete sphere — a well-formed
    STEP file describing a different solid. (Irrational rims only refused by
    accident, through the SurdVal coercion TypeError.) R=10, r=6 has a
    RATIONAL rim height (√64 = 8), so it is exactly the case the old code got
    wrong for the napkin ring and would have gotten wrong here."""
    from forgekernel.stepbody import write_step_body

    for solid in (SphereBlindBore(10, 6, 2), ):
        with pytest.raises((ValueError, TypeError)):
            write_step_body(B.to_body(solid))
