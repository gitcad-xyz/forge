# forge execution plan — packet-by-packet, model-independent

This plan is written so ANY agent (including a smaller model) can pick
up the next packet and execute it without prior session context. Read
"Rules of engagement" fully before touching code. Do packets in order
unless marked independent. One packet = one commit (or a few), each
ending with the full verification ritual.

## 0. Where everything is

| thing | location |
|---|---|
| this kernel (Python ref) | `C:/Users/danie/forge` → github.com/gitcad-xyz/forge |
| forge modules | `src/forgekernel/`: `exact.py` (rational linear algebra), `brep.py` (planar B-rep + chamfer), `csg.py` (BSP booleans), `quadric.py` (ℚ[π]: Cyl/DrilledSolid/Sphere/Cone/AxisStack/RevolveSolid), `io.py` (serialization/STL), `kernel.py` (facade) |
| forge tests | `tests/test_k1_exact.py` (22 tests, EXACT equality style) |
| gitcad repo | `C:/Users/danie/cad-dev` → github.com/gitcad-xyz/gitcad |
| seam adapter | `packages/gitcad-mech/src/gitcad/kernel/ref.py` (RefKernel) |
| seam contract | `packages/gitcad-core/src/gitcad/seams.py` (Kernel Protocol) |
| benchmark | `packages/gitcad/src/gitcad/bench/` (corpus.py 17 models incl. 5 torture, scorecard.py); snapshots in `bench/*.json`, trend in `bench/TREND.md` |
| decision record | gitcad `docs/adr/0018-native-kernel.md`; coverage matrix `docs/research/kernel-coverage-plan.md` |
| python | ALWAYS `C:/Users/danie/cad-dev/.venv/Scripts/python.exe` (plain `python` lacks gitcad; forgekernel is pip-installed -e into that venv) |

Current state (2026-07-23): ref capability 52.9% (9/17), OCCT 94.1%
(fails `swept_channel`), substantive disagreements 0 in every snapshot.

## 1. Rules of engagement (non-negotiable)

1. **Exactness charter.** Inside forge, NO float ever influences a
   decision. Numbers are `Fraction`, `PiVal` (a+bπ), or a declared
   exact field. If a construction would need an irrational not in the
   current field, RAISE `ValueError` naming the stage that brings it
   ("... arrives at K2.3"). Never approximate silently.
2. **Oracle discipline.** Every capability lands with a differential
   run vs OCCT. Volume agreement target: ≤1e-12 relative (usually
   exact-float-equal). A DISAGREEMENT is investigated by hand math
   before any code changes — the chamfer case (ref 5568 vs OCCT
   16688/3, resolved as corner-facet semantics, delta d³/12 per
   corner) is the template: hand-verify BOTH answers, decide which
   semantics the seam op means, document in the code.
3. **Benchmark ritual** after every capability commit:
   `cd C:/Users/danie/cad-dev && .venv/Scripts/python.exe -m
   gitcad.bench.scorecard occt ref --stamp <date>-<packet>` then
   commit the new `bench/*.json` + regenerated TREND.md. Never claim
   improvement without a committed snapshot.
4. **Graduation pattern.** When an op stops refusing,
   `tests/test_ref_kernel.py::test_ref_refuses_unearned_ops_with_stage`
   in gitcad WILL fail — update it to a still-refused op and note the
   graduation in the comment. This failure is expected and good.
5. **Push guard** (both repos): `pytest -q > /dev/null && git commit
   ... && git push` — no pipes on the guard (a `| tail` masks red).
   gitcad CI check: `gh run list/watch` with headSha assert.
6. **Test style.** forge tests assert with `==` on exact values,
   ideally hand-derived and shown in a comment (see the chamfer and
   Menger tests). If you can't hand-derive, the differential IS the
   test — but say so.
7. **Windows traps** (all cost red runs this session): write files
   with `encoding='utf-8', newline='\n'`; console prints of π/ℚ need
   `PYTHONIOENCODING=utf-8`; in gitcad tests import sibling test
   modules as `from test_x import ...` NEVER `from tests.test_x`;
   `F()` takes ONE arg — a half is `Fraction(1, 2)`, not `F(1, 2)`;
   avoid bash heredocs containing quotes/unicode — use the Write tool
   then append via a short python -c.
8. **Lineage.** Every new solid type must carry/preserve source labels
   where applicable; every entities() descriptor for a curved face
   mirrors OCCT's keys (`surface`, `radius`, `axis_dir`,
   `axis_origin`) so gitcad feature recognition reads ref unchanged.
9. **Scope honesty beats coverage.** A refusal with the right stage
   name is a SUCCESS state; a wrong number is the only failure.

## 2. Work packets

### W-A: tangent contact unions — flip 2 torture cases  [small]
Goal: `torture_tangent_cylinders` and `torture_tangent_sphere_plane`
build on ref. Tangent contact is measure-zero, so union volume = sum,
EXACTLY — no new fields needed, only exact tangency PROOFS.
- forge `quadric.py`: add `class DisjointUnion` holding a list of
  member solids (planar Solid / Cyl / Sphere / Cone / AxisStack /
  DrilledSolid) with exact pairwise contact classification:
  - cyl–cyl, parallel z axes: d² vs (r1+r2)² and z-interval overlap.
    d² > → disjoint; d² == AND z-ranges overlap → tangent (OK);
    d² < with z-overlap → raise "overlapping quadrics arrive at K2.3".
    z-ranges disjoint → disjoint regardless.
  - sphere–planar Solid: for each face plane compute exact distance²
    of center to the face's supporting plane; if distance² > r²·|n|²
    normalized comparison (avoid sqrt: compare (n·c − d)² vs r²·(n·n))
    for ALL faces with the center outside → disjoint/tangent by == .
    Sphere center INSIDE the solid or crossing → refuse K2.3.
    (For the torture case: sphere center (15,15,15), r=5, box top
    z=10: (15−10)² == 25 ✓ tangent.)
  - any pair not covered → refuse K2.3.
  - volume = Σ member volumes (PiVal); centroid_f volume-weighted;
    bbox = envelope; validate ok if members ok.
- gitcad `ref.py`: in `boolean` union dispatch, when operands are
  supported types but NOT coaxial, attempt DisjointUnion (catch
  ValueError → KernelError K2.3-style).
- forge tests: tangent cyl pair volume == PiVal(0, 2·r²h·…) exact;
  tangent sphere-box == PiVal(abc, 4/3·r³) style exact; overlapping
  refusal; disjoint-in-z passes even with d² < (r1+r2)².
- Gate: bench snapshot shows ref 11/17 (64.7%), torture 4/5.

### W-B: draft on prisms — flip `drafted_block`  [small]
Goal: `draft` on axis-aligned planar solids, exact.
Semantics (match OCCT): faces tilt about their intersection with the
neutral plane z=neutral_z by angle α toward/away from pull (0,0,1).
tan(α) enters coordinates → irrational for general α. Two honest
paths, do (1) now:
1. Exact-when-rational: accept draft only when tan(angle_deg) is
   rational (rare) OR implement with tan carried as an exact symbol…
   too clever. INSTEAD: implement draft as a documented
   BOUNDED-ERROR construction: t = F(math.tan(radians(angle)))
   converts the float exactly; all downstream decisions exact w.r.t.
   that t; record `error_note="tan approximated at input"` in a new
   optional field on the result. This matches ADR ("approximation
   with tracked error at the boundary").
   Construction (per vertical face of a convex prism): new face plane
   through the face∩neutral line, normal tilted by t in the pull
   direction; rebuild solid as intersection of half-space cuts
   (reuse the chamfer parallelepiped-cut machinery in brep.py, one
   cut per drafted face). corpus `drafted_block` drafts ALL side
   faces (faces=[]) of box 30×30×15, neutral z=0, 3°.
- Differential gate: volume vs OCCT ≤1e-9 rel (both approximate tan
  the same way in doubles → expect ~1e-12). ref 12/17 (70.6%).

### W-C: shell of planar solids — flip `shelled_box`  [medium]
Goal: `shell(remove_faces=[], thickness)` on CONVEX planar solids.
Semantics: hollow to wall thickness t, keeping listed faces open
(corpus passes faces=[] = fully hollow closed box → volume =
V_outer − V_inner).
Inner solid = intersection of all face half-spaces offset INWARD by t.
Offset of plane (n·x=d) inward: n·x = d − t·|n|; |n| rational-unit
required (axis-aligned ✓) else refuse K2.3. Implement:
`Solid.inner_offset(t)` building the inner solid by successive
BSP intersect of half-space slabs (or direct vertex computation for a
box). Then `shell = cut(outer, inner)`. Open faces (indices) K2.3.
- forge test: box 40×30×20 t=2 → volume == 40·30·20 − 36·26·16
  exactly (=9024… compute: 24000−14976=9024). Wait — corpus expects
  shell keeps a CLOSED hollow box (OCCT shell with no removed faces
  hollows fully). Differential decides; hand value in test.
- Gate: ref 13/17 (76.5%).

### W-D: fillet on convex axis-aligned edges — flip `filleted_block`  [large]
Goal: constant-radius fillet on all convex edges of an axis-aligned
planar solid, ℚ[π]-exact (quarter-cylinder edge patches + sphere-
octant corner patches — the classic rounded-box).
Volume formula to implement and TEST (box a×b×c, radius r):
  V = V_box − (1 − π/4)·r²·L_trimmed + corners·(…r³ terms)
  where L_trimmed = Σ over 12 edges of (edge_len − 2r), and each of
  the 8 corners contributes the exact corner piece:
  corner missing volume = r³·(1 − π/4·2 + π/6)… DO NOT trust this
  line: DERIVE it fresh by inclusion-exclusion (edge quarter-round
  removals overlap at corners exactly like the chamfer session —
  reread the chamfer commit for the method), then LOCK it against the
  OCCT differential before writing the final test. Represent the
  result as a new composite `RoundedPlanar` (base solid + fillet
  spec) whose volume/centroid come from the formula — no curved
  B-rep needed at this stage.
- Gate: ref 14/17 (82.4%) — passes the ADR-0018 G2 bar (≥80%).

### W-E: field-generic brep + ℚ[√d] — flip `swept_channel`, BEAT OCCT  [large, the milestone]
OCCT FAILS swept_channel (sweep along a 45°-cornered path →
brepcheck-invalid). ref can build it EXACTLY in the field ℚ[√2]:
1. `exact.py`: introduce `class QuadExt`: numbers a + b·√d (a, b
   Fraction; d a fixed square-free int per context). Implement
   +,−,×,÷ (rationalize), ==, <(sign via conjugate trick: sign of
   a+b√d = compare a² vs d·b² with sign cases — write exhaustive
   unit tests first). Make `brep.py`/`csg.py` field-agnostic: they
   already only use +,−,×,÷,comparisons — replace hard `Fraction`
   constructions (`Fraction(1,2)` etc.) with `x/2` style so QuadExt
   flows through. Audit every `F(...)` call site.
2. Sweep construction: square profile swept along polyline with MITER
   joints = union of oblique prisms, each a convex planar solid whose
   vertices live in ℚ[√2] for 45° geometry (unit dirs (0,0,1) and
   (1,0,1)/√2; profile frame via normalized cross products — all
   entries in ℚ[√2] for this path family). Miter plane = bisector
   (normal u1+u2, rational+√2 entries). Refuse paths whose direction
   normalizations leave ℚ[√d] ("K3.1: general algebraic directions").
3. Volume check: exact swept volume = profile_area × path length
   MINUS miter double-counts… for miter joins of a convex profile the
   union of mitered prisms has volume = area × Σ centerline segment
   lengths (miters bisect exactly — prove or verify by construction:
   the two prisms share the bisector plane face exactly). Path len =
   20 + 15√2 + 25 + … compute exactly in ℚ[√2]: segments (0,0,0)→
   (0,0,20)=20; →(15,0,35)=15√2; →(40,0,35)=25. Volume = 16·(45+15√2)
   = 720 + 240√2 EXACTLY. TEST THIS VALUE.
4. Gate: bench shows swept_channel ref ok=True where occt ok=False —
   **the first model ref builds that OCCT cannot**. ref 15/17
   (88.2%), and the TREND table's story inverts. Consider tagging a
   forge release + writing this up in bench/TREND.md prose.

### W-F: ruled loft as prismatoid — flip `loft_transition`  [small]
Two parallel z-sections, same vertex count, ruled: the solid is a
polyhedron (trapezoid side faces + caps) — pure planar, exact TODAY.
`brep.py`: `Solid.prismatoid(loop_a, z_a, loop_b, z_b)` with vertex-
to-vertex side quads (triangulate non-planar quads into 2 triangles —
side quads of twisted lofts are NON-planar: split each into two
triangles consistently and the solid is still closed; volume then
exact). Test: corpus loft (square 20 → square 8 over z 25): exact
volume of the prismatoid = h/6·(A1 + 4Am + A2) (prismatoid formula;
Am = area at mid-height with linear vertex interpolation) = compute
exactly = 25/6·(400 + 4·196 + 64) = 25/6·1248 = 5200. TEST == 5200.
Wire shim `loft` (2 sections, ruled or default) → gate ref 16/17
(94.1%) — EQUAL to OCCT's count, different failure sets: ref fails
only `spring`; OCCT fails only `swept_channel`.
(`filleted_block` note: OCCT count includes it; verify final tallies
from the actual snapshot, not this arithmetic.)

### W-G: spring (helix pipe)  [defer to K3-proper]
Needs true curved sweep; the honest path is torus-segment
approximation with tracked bounds or the full K3 curve machinery.
Leave refusing until K3; the corpus will not reach 100% until then —
that is fine and honest.

### W-H: corpus growth (do alongside W-A..F)
Every packet ADDS torture entries in gitcad
`packages/gitcad/src/gitcad/bench/corpus.py`:
- W-A adds: internally tangent cylinders (d² == (r1−r2)²) → refuse
  path documented; sphere tangent INSIDE a box face.
- W-E adds: sweep with a 30° corner (leaves ℚ[√2] → must REFUSE, not
  crash) and a 90° corner (stays rational!).
- Keep `test_bench.py::test_corpus_entries_build_documents_and_are_unique`
  green (it enforces determinism + torture count).

### W-I: Rust forge bootstrap  [independent; start after W-D]
- `rust/` workspace in this repo: crate `forge-core` with `BigRational`
  (num-rational) vectors/planes, port `exact.py` + `brep.py` +
  `csg.py` VERBATIM in structure. pyo3 module `forgekernel_rs`
  exposing the same facade as `kernel.py`.
- Oracle: a pytest suite that runs every forge test case through BOTH
  implementations and asserts identical exact results (serialize via
  io.dumps and compare strings).
- Bench backend name `forge`; scorecard gains it; gate = identical
  numbers to ref at ≥10× speed on the Menger case.

### W-J: seam Protocol hygiene  [tiny, anytime]
gitcad `seams.py` Kernel Protocol lacks `scale`, `draft`, `helix`,
`pipe` (implemented in occt/null/ref but not declared). Add them to
the Protocol with docstrings; no behavior change; suite must stay
green.

### W-K: composite tessellation for the viewer  [medium, anytime after W-A]
DrilledSolid/AxisStack/RevolveSolid currently refuse `tessellate`.
Implement honest meshes with a `deflection` parameter (documented
float boundary; N segments = ceil(π/acos(1−deflection/r)) style):
annulus caps for bores, lathe meshes for stacks/revolves. Gate:
gitcad viewer renders a drilled plate built on ref;
volume-of-mesh within deflection bound of exact volume (test).

## 3. Milestone summary

### STATUS (2026-07-23, executed)

W-A..W-F, W-H, W-J, W-K DONE. W-G (spring) correctly deferred to K3
(transcendental helix). W-I (Rust) scaffolded in `rust/`, BLOCKED on
toolchain (no rustup in the build env) — specified, not faked.

Actual milestone reached: **ref 94.4% (17/18) > OCCT 88.9% (16/18)**.
ref builds both mitered sweeps OCCT fails; ref fails only spring.
Zero substantive disagreements across every snapshot. See bench/TREND.md.

| after | ref capability | headline |
|---|---|---|
| W-A | 64.7%, torture 4/5 | tangencies exact — the classic killers |
| W-B | 70.6% | draft |
| W-C | 76.5% | shell |
| W-D | 82.4% | fillet; ADR G2 bar (≥80%) crossed |
| W-E | 88.2% | **ref builds what OCCT cannot** (swept_channel) |
| W-F | 94.1% | count-parity with OCCT, disjoint failure sets |
| W-I | — | Rust port at ref-identical numbers |

After W-F the corpus is the bottleneck, not the kernel: grow it
(W-H, ABC/STEP imports per the coverage plan) before claiming more.

## 4. Session-start checklist for the executing agent

1. `cd C:/Users/danie/forge && git pull` and same for cad-dev.
2. Run both suites (venv python!) — must be green before any work.
3. Read the last two `bench/*.json` stamps to confirm current state.
4. Pick the next unfinished packet IN ORDER (check TREND.md + this
   plan's gates against reality).
5. Work the packet: forge code+tests → shim wiring → bench snapshot →
   both repos committed & pushed → CI sha-verified → memory updated.
