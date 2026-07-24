# forgekernel_rs

Native (Rust) build of performance-critical routines for
[`forgekernel`](https://pypi.org/project/forgekernel/), an exact-arithmetic
B-rep CAD kernel.

`forgekernel` runs without this package (pure Python). When `forgekernel_rs` is
installed, it is used for the surface–surface intersection and geometric
predicate routines and returns the same results.

## Install

```
pip install forgekernel[rust]
```

## Build

Built with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs) as an
`abi3` wheel (one wheel per platform, CPython 3.11+). Exact arithmetic uses
`num-bigint` / `num-rational`.

License: Apache-2.0. Source: https://github.com/gitcad-xyz/forge
