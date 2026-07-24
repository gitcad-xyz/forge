# forge

An exact-arithmetic boundary-representation (B-rep) CAD kernel. Geometric and
topological decisions use exact rational arithmetic (ℚ, ℚ[π], ℚ[√d]) or
certified intervals; no result depends on a floating-point tolerance.

It ships as two PyPI packages:

- `forgekernel` — the kernel, pure Python.
- `forgekernel_rs` — a native (Rust) build of the performance-critical routines,
  installed automatically when a wheel is available and returning the same
  results.

## Capabilities

- Planar solids: rational linear algebra, plane-based B-rep, booleans,
  divergence-theorem mass properties.
- Quadrics (cylinder, cone, sphere, torus) with closed-form intersections.
- NURBS curves and surfaces; surface–surface intersection with complete branch
  detection and certified points.
- Procedural offsets and shells; edge blends (fillets); lofts.
- Exact volume, centroid, and inertia via the divergence theorem.
- STEP (AP203/AP214) import and export.

## Design

A Python reference implementation defines the exact semantics. A Rust build
provides the same operations for production use; each ported routine is
cross-checked against the reference. Design notes and decisions are recorded in
the gitcad repository (ADR-0018).

## Layout

- `src/forgekernel/` — the Python kernel.
- `rust/forge-core/` — the native build (`forgekernel_rs`).

## Install

```
pip install forgekernel          # pure Python
pip install forgekernel[rust]    # add the native build
```

License: Apache-2.0. Source: https://github.com/gitcad-xyz/forge
