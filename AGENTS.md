# AGENTS.md

Working agreement for agents operating in this repository. Package-specific
contracts, schemas, and decisions live in `docs/design.md`; do not duplicate
them here. Execution plans live in `docs/plan/`; acceptance reports in
`docs/plan/reports/`.

## Toolchain

- Package/dependency management: `uv` (lockfile `uv.lock` is authoritative —
  never hand-edit dependencies without going through `uv add`/`uv sync`).
- Lint/format: `ruff check .` / `ruff format .`, config in `.ruff.toml`.
- Type checking: `mypy` on `src/tbls` (see `pyproject.toml` `[tool.mypy]`
  once the refactor in `docs/design.md` lands).
- Tests: `pytest`, config in `pyproject.toml` `[tool.pytest.ini_options]`.
- Pre-commit hooks are configured in `.pre-commit-config.yaml`; keep them
  passing before committing.

## Conventions

- Docstrings: Google style, English, on every public class/function
  (enforced by ruff's `D` rules).
- Comments: English.
- Every ML estimator shipped in the `tbls` package must be sklearn-compatible
  (`BaseEstimator` + `ClassifierMixin`/`RegressorMixin`, full `get_params`/
  `set_params`/`fit`/`predict`/`predict_proba`).
- Large/generated files never get committed: see `.gitignore`. In particular,
  dataset `.pkl` files live under `experiments/datasets/` and are always
  git-ignored — never `git add` them, even by accident via `git add -A`.
- Commit messages follow Conventional Commits.

## Planning workflow

- Design source of truth: `docs/design.md`. Update it before writing or
  revising a plan when the architecture itself changes.
- Execution plans: `docs/plan/NN-kebab-case-title.md`. Once a plan has been
  handed to an execution agent or execution has begun, it is immutable
  history — corrections go into a new, next-numbered compensating plan, not
  edits to the original.
- Acceptance reports: `docs/plan/reports/`, one per completed plan, citing
  actual command output as evidence — never claim a check passed without it.

## Scope discipline

- This is a greenfield refactor of an existing, working algorithm
  implementation (`othercode/`) into a publishable package. Treat it as free
  of backward-compatibility obligations: no shims/aliases for the legacy
  root-level modules being deleted, unless a plan explicitly says otherwise.
- Do not expand a plan's scope (e.g. implementing new estimator capabilities,
  Cython kernels, or fixing unrelated legacy bugs) without a new plan.
