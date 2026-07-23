# forge (Rust) — the production port

**Status: scaffolded, blocked on toolchain.** This environment has no
`rustup`/`cargo` (`cargo --version` → not found), so the Rust port is
specified here and ready to build the moment a toolchain is available.
Per the exactness/oracle charter it is NOT faked — an empty crate that
"passes" would be dishonest.

## The packet (W-I), ready to execute

1. `rustup` + `cargo new --lib forge-core`; deps `num-rational`,
   `num-bigint`, `pyo3` (extension-module).
2. Port, structure-for-structure, from `../src/forgekernel/`:
   - `exact.rs` ← `exact.py` (BigRational Vec3/Plane, exact predicates)
   - `brep.rs` ← `brep.py` (Polygon/Solid, signed-tetra volume, exact
     line-coverage closure, ear-clip, chamfer)
   - `csg.rs`  ← `csg.py` (BSP union/cut/intersect, exact split)
   - pyo3 module `forgekernel_rs` exposing the same facade as `kernel.py`.
3. **Oracle test** (the whole point): a pytest suite that runs every
   case in `../tests/test_k1_exact.py` through BOTH `forgekernel` (ref)
   and `forgekernel_rs`, asserting identical results by comparing
   `io.dumps()` strings — bit-exact structural equality, not float
   tolerance. ref is forge's oracle exactly as OCCT is ref's.
4. Bench: add backend name `forge` to `gitcad.bench.scorecard`; gate =
   identical numbers to ref at ≥10× speed on the Menger case (the deep-
   boolean model where big rationals cost ref the most).

## Why this order

ref (Python, exact) proved the semantics; the Rust port only makes them
fast. It never invents behavior — every forge answer must already be a
ref answer. That is why W-I comes AFTER the exact reference is complete
through K2, and why its acceptance test is "identical to ref", not
"looks right".

## Status update (2026-07-23): K1 core BUILT and oracle-verified

Toolchain installed (rustup 1.29 / rustc 1.97, MSVC). `forge-core`
(`src/lib.rs`) ports the K1 exact planar core — BigRational Vec3/Plane
with exact sign predicates, box + translate, and the full BSP
union/cut/intersect — behind a pyo3 module `forgekernel_rs`. Build:

    cd rust/forge-core && VIRTUAL_ENV=<venv> maturin develop --release

Oracle (`tests/test_rust_oracle.py`, 7 tests, skips if the extension
isn't built): every case runs through BOTH forgekernel_rs and the
Python `forgekernel` reference; asserts IDENTICAL exact volume AND
identical canonical face-set. All pass — the Rust port reproduces ref
bit-for-bit. Performance: ~2.5x faster than the Python ref on a boolean
cut (larger wins expected on deep chains where ref's denominators grow).

Remaining port work (mechanical, ref stays the oracle): prism, scale,
mirror, rotated_quarter, chamfer, exact line-coverage closure, and the
K2 ℚ[π] composites — each added the same way, gated by ref-identity.
