# AGENTS.md

Working agreement for agents operating in this repository. Architecture,
design rationale, and repository structure live in `docs/architecture.md`;
do not duplicate them here. Usage tutorials live in `docs/usage-*.md`;
contribution conventions live in `docs/development.md`; the release pipeline
is documented in `docs/release-process.md`.

## Toolchain

- Package/dependency management: `uv` (lockfile `uv.lock` is authoritative —
  never hand-edit dependencies without going through `uv add`/`uv sync`).
- Lint/format: `ruff check .` / `ruff format .`, config in `.ruff.toml`.
- Type checking: `mypy src/tbls` (strict; see `pyproject.toml` `[tool.mypy]`).
- Tests: `pytest`, config in `pyproject.toml` `[tool.pytest.ini_options]`.
- Pre-commit hooks are configured in `.pre-commit-config.yaml`; keep them
  passing before committing.

## Conventions

- Docstrings: Google style, English, on every public class/function
  (enforced by ruff's `D` rules).
- Comments: English.
- Every ML estimator shipped in the `tbls` package must be sklearn-compatible
  (`BaseEstimator` + `ClassifierMixin`/`RegressorMixin`, full `get_params`/
  `set_params`/`fit`/`predict`/`predict_proba`). See `docs/architecture.md`
  section 5 for the full contract, including the two-view exception for
  `PairwiseKCCA`/`GraphFuzzyKCCA`.
- Large/generated files never get committed: see `.gitignore`. In particular,
  dataset `.pkl` files live under `experiments/datasets/` and are always
  git-ignored — never `git add` them, even by accident via `git add -A`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `build:`/`ci:`, `chore:`, `refactor:`,
  `perf:`, `revert:`). `cliff.toml` parses these prefixes to generate the
  changelog automatically on release — an unconventional commit message is
  silently dropped from the changelog, not just a style nit.
- Every English doc under `docs/` (and the root `README.md`) starts with an
  `English | [简体中文](./<name>.zh-CN.md)` header line. English is the
  source of truth; translations are maintained separately.

## Scope discipline

- `src/tbls/` is the published PyPI package: dependency-light
  (numpy/scipy/scikit-learn only), stable public API. `experiments/` is the
  training/evaluation pipeline for this repository's own use, never
  published, and free to depend on heavier packages (`pandas`,
  `imbalanced-learn`, `xgboost`, `typer`, ...). See `docs/architecture.md`
  section 3 before deciding where a change belongs.
- `tbls.genoptim` and `tbls.ensemble` are experimental subpackages (see
  `docs/experimental-modules.md`). Do not silently "fix" their TBLS-coupled
  functions by patching over `AttributeError`s — the gap is a real missing
  capability on `TBLS`, not a naming bug; closing it is new estimator
  functionality and needs its own explicit scope.
- Do not expand a change's scope (e.g. implementing new estimator
  capabilities, Cython kernels, or fixing unrelated legacy behavior)
  silently inside an unrelated change; call it out explicitly instead.

## Releasing

Do not push a `v*` tag unless explicitly asked to cut a release — pushing one
triggers the full CI/CD pipeline in `.github/workflows/release.yml`,
culminating in a PyPI publish that cannot be undone. See
`docs/release-process.md` for the full procedure and the pre-conditions
(matching `pyproject.toml`/`src/tbls/__init__.py` versions, `master`
ancestry, etc.) the pipeline enforces.
