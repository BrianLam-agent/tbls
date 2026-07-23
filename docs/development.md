English | [简体中文](./development.zh-CN.md)

# Development guide

This is the "二次开发教程" — everything you need to set up a local
development environment, run the checks CI runs, and extend the package
correctly.

## 1. Setup

```bash
git clone https://github.com/BrianLam-agent/tbls.git
cd tbls
uv sync --group dev --group experiments   # dev tools + experiments/ dependencies
```

`dev` is a `uv` dependency group (`pytest`, `pytest-cov`, `ruff`, `mypy`) and
is synced by default whenever you run `uv run ...` — you rarely need to pass
`--group dev` explicitly. `experiments` is not a default group and must be
requested explicitly (`uv sync --group experiments` or
`uv run --group experiments ...`) since it pulls in `pandas`/`xgboost`/etc.
that only the training pipeline needs.

## 2. Everyday commands

```bash
uv run pytest tests/ -v                 # full test suite
uv run pytest tests/test_tbls.py -v     # one file
uv run ruff check .                     # lint
uv run ruff format .                    # format (in place)
uv run ruff format --check .            # format (check only, what CI runs)
uv run mypy src/tbls                    # type check (strict, package only)
```

Pre-commit hooks (`.pre-commit-config.yaml`) run `ruff --fix`, `ruff-format`,
`pyproject-fmt`, and `yamlfmt` automatically:

```bash
uv run pre-commit install     # one-time, installs the git hook
uv run pre-commit run --all-files
```

## 3. Project conventions

- **Docstrings**: Google style, English, on every public class/function.
  Enforced by ruff's `D` rule group (`convention = "google"` in
  `.ruff.toml`) — `ruff check .` will fail on missing/malformed docstrings.
- **Type hints**: full type hints on every public signature in `src/tbls/`;
  `mypy --strict` must pass. `experiments/` is not held to the same bar (it's
  an internal tool, not a published API), though its docstrings *are* (see the
  next point — the Google-style `D` rules are enforced on `experiments/` too,
  with only `ARG001` for typer-CLI options relaxed).
- **Estimator contract**: any new classifier added to `src/tbls/` must be a
  full `sklearn.base.BaseEstimator` + `ClassifierMixin` (see
  [`architecture.md` §5](./architecture.md#5-estimator-contract) for the
  exact checklist). Any new feature extractor should default to standard
  single-`X` `fit`/`transform` (`TransformerMixin`) *unless* it is inherently
  multi-view like `PairwiseKCCA`/`GraphFuzzyKCCA` — see that section for why
  those two are an intentional exception, not a precedent to casually repeat.
- **Comments**: English.
- **Commit messages**: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `build:`/`ci:`, `chore:`, `refactor:`,
  `perf:`, `revert:`). This is not just style — `cliff.toml` parses these
  prefixes to build the changelog automatically; an unconventional commit
  message is silently dropped from the changelog (see
  [`release-process.md`](./release-process.md)).

## 4. Adding a new estimator or feature to `src/tbls`

1. Implement it as its own module (`src/tbls/<name>.py`), reusing
   `tbls._kernel`/`tbls._ifs`/`tbls._graph` instead of re-deriving
   RBF-kernel/IFS/graph math — see
   [`architecture.md` §4](./architecture.md#4-package-internals-shared-modules).
2. Satisfy the estimator contract (§3 above).
3. Export it from `src/tbls/__init__.py`'s `from .<module> import ...` and add
   it to `__all__`.
4. Add tests: at minimum, `fit`/`predict`/`predict_proba` on synthetic data
   (see `tests/conftest.py`'s fixtures), a `sklearn.base.clone()` round-trip,
   and a `cross_val_score`/`GridSearchCV` smoke test (matching the existing
   `tests/test_tbls.py`/`test_bls.py` pattern). If the new code touches
   `_kernel`/`_ifs`/`_graph`, add a **direct** unit test in
   `tests/test_shared_modules.py` that checks numerical output, not just
   shape/finiteness — see the regression story in
   [`architecture.md` §4](./architecture.md#a-note-on-numerical-fidelity) for
   why shape-only tests are not enough for this code.
5. Document it: add a `docs/usage-<name>.md` (with the
   `English | [简体中文](...)` header line) and link it from the root
   `README.md`'s documentation index.
6. Run the full check suite (§2) before opening a PR.

## 5. Documentation structure and translations

Every doc under `docs/` (and the root `README.md`) starts with:

```markdown
English | [简体中文](./<same-name>.zh-CN.md)
```

English docs are the source of truth and are maintained alongside code
changes. Simplified Chinese translations (`*.zh-CN.md`) are maintained
separately — if you add or restructure an English doc, add the header line
pointing at the (possibly not-yet-existing) `.zh-CN.md` counterpart, but you
are not required to write the translation yourself.

## 6. `experiments/` vs. `src/tbls/`

If your change is about *training/evaluating* on real data (a new data
loader, a new comparison classifier, a new CLI flag), it belongs in
`experiments/`, not `src/tbls/` — see
[`architecture.md` §3](./architecture.md#3-why-the-packageexperiments-split).
If you're unsure which side a change belongs on, ask: "would someone who just
ran `pip install tbls` need this?" If no, it's `experiments/`.

## 7. Releasing

Not part of day-to-day development — see
[`release-process.md`](./release-process.md) if you have maintainer access
and need to cut a release.
