# Releasing forge to PyPI

Two distributions ship from this repo:

| project | what | build backend | wheels |
|---|---|---|---|
| **`forgekernel`** | pure-Python exact kernel (the reference; works everywhere) | setuptools | one universal `py3-none-any` |
| **`forgekernel_rs`** | native accelerator (optional; identical results, faster) | maturin (PyO3, abi3) | one per platform, CPython ≥ 3.11 |

`forgekernel` runs fully without `forgekernel_rs`; the Rust build only speeds
hot paths. Version is single-sourced from `src/forgekernel/__init__.py`
(`__version__`) for `forgekernel`, and from `pyproject.toml` / `Cargo.toml` for
`forgekernel_rs` — keep the three in lockstep.

## One-time setup (maintainer, gated)

1. **Reserve the names** on PyPI: `forgekernel` and `forgekernel_rs`. (This is
   the "name approval" gate — until both exist, publishing cannot proceed.)
2. **Trusted Publishing** (no API tokens): on each project's PyPI page →
   *Publishing* → add a GitHub publisher:
   - Repository: `gitcad-xyz/forge`, workflow: `release.yml`, environment: `pypi`.
3. In the GitHub repo, create an Environment named `pypi` (Settings →
   Environments) — optionally with required reviewers so a human approves each
   publish.

## Build & validate locally (no upload)

```bash
python -m build                       # forgekernel → dist/*.whl, *.tar.gz
python -m twine check dist/*          # metadata + README render must PASS
# native wheel for the current platform:
pip install maturin
cd rust/forge-core && maturin build --release
```

Both are wired into CI (`.github/workflows/release.yml`) and validated on every
push/PR (build + `twine check`; the Rust matrix builds Linux/macOS/Windows).

## Cut a release

1. Bump `__version__` in `src/forgekernel/__init__.py` **and** the `version` in
   `rust/forge-core/pyproject.toml` + `rust/forge-core/Cargo.toml` to the same value.
2. Commit; then tag and push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
3. The `release` workflow builds `forgekernel` (sdist + wheel) and the
   `forgekernel_rs` platform wheels + sdist, then the `publish` job (tags only)
   uploads both to PyPI via Trusted Publishing.

## Downstream

`gitcad-mech` depends on `forgekernel_rs` **by default** (via a
`platform_machine` marker), so `pip install gitcad` delivers the native kernel on
mainstream architectures. Publish `forgekernel` / `forgekernel_rs` **before** the
matching gitcad release so that dependency resolves.
