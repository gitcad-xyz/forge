# forge — the gitcad-native B-rep kernel

From-scratch, unencumbered (Apache-2.0), built to pass the bar OCCT
sets and then raise it — the plan and the decision record live in
gitcad ([ADR-0018](https://github.com/gitcad-xyz/gitcad/blob/main/docs/adr/0018-native-kernel.md),
[coverage plan](https://github.com/gitcad-xyz/gitcad/blob/main/docs/research/kernel-coverage-plan.md)).

## The three-oracle chain

```
forgekernel.ref   Python + exact rational arithmetic — the executable
                  specification; topological decisions are EXACT, never
                  epsilon-guarded
      ⇅ differential
OCCT              the 30-year-hardened independent oracle, driven from
                  the gitcad benchmark corpus through the Kernel seam
      ⇅ oracle
forge (Rust)      the production port, added operator class by operator
                  class once ref has proven the semantics
```

## How progress is measured

Not vibes: the [benchmark trend](https://github.com/gitcad-xyz/gitcad/blob/main/bench/TREND.md)
in the gitcad repo scores every backend on the shared corpus —
capability %, torture-case pass rate, correctness deltas, wall time.
Day-one baseline: OCCT scores 93.8%, failing `swept_channel`
(sharp-cornered sweep → invalid geometry). Beating that number with
exact arithmetic is the first milestone.

## Roadmap (gates, not dates)

- **K1** exact planar core: rational linear algebra, plane-based
  polyhedral B-rep, epsilon-free booleans, divergence-theorem mass
  properties, native lineage. Gate: planar corpus green vs OCCT.
- **K2** quadrics + torus with closed-form intersectors — most of
  real mech. Gate: ≥80% corpus green.
- **K3** NURBS + branch-complete surface–surface intersection (the
  crown jewel). **K4** procedural offsets/shell. **K5** G2 blends.
  **K6** the surfacing suite — the SolidWorks-class end goal.

## Layout

- `src/forgekernel/` — the Python reference kernel (`ref`)
- `rust/` — the forge port (arrives with K1 stability)
- gitcad consumes both through its `Kernel` seam; nothing here depends
  on gitcad
