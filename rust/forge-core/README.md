# forgekernel_rs — native accelerator for forgekernel

The optional Rust build of the hot paths in
[`forgekernel`](https://pypi.org/project/forgekernel/), the exact-arithmetic
B-rep CAD kernel. It is a **drop-in speed-up, not a dependency**: `forgekernel`
runs pure-Python and produces identical (exact ℚ) results without it, then
transparently uses `forgekernel_rs` for the surface–surface intersection and
predicate inner loops when the extension is importable.

Bit-identical to the Python reference — every ported routine is oracle-checked
against it, so the accelerator never changes an answer, only the wall time.

```bash
pip install forgekernel[rust]     # forgekernel + this accelerator
```

Built with [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs) as an
`abi3-py311` wheel (one wheel per platform serves every CPython ≥ 3.11).
Exact rationals use `num-bigint` / `num-rational`.

Apache-2.0. Source and design: https://github.com/gitcad-xyz/forge
