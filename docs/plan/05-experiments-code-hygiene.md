# Plan 05: `experiments/` code hygiene (English-only, Google-style docstrings)

> Status: final, ready to hand off. No hard predecessor. Purely mechanical —
> **must not change any behavior**; every change in this plan is a
> comment/docstring/identifier-language change only.

## Goal

Audit every file under `experiments/` for: (a) any non-English comment,
docstring, log message, or string literal meant for humans (config keys/data
values are not in scope — only human-facing text), and (b) docstrings not
already in Google style (`Args:`/`Returns:`/`Raises:` sections, matching
`src/tbls/`'s existing convention). Bring everything to one consistent
standard. This is cleanup precipitated by `experiments/classifiers.py` in
particular, which predates the package refactor and was migrated largely
as-is.

## Design references

- `src/tbls/*.py` — the Google-docstring convention already in force there;
  match it exactly (see e.g. `src/tbls/_ifs.py` for the target style).
- `docs/development.md` — if it documents a docstring/lint convention,
  update it to state the convention now also applies to `experiments/`
  (check first; do not duplicate if already stated generally).

## Non-goals

- Any change to `experiments/`'s behavior, function signatures, CLI options,
  or output format. If a docstring is found to be describing behavior that's
  actually wrong (stale), fix the docstring to match the real behavior — do
  not change the behavior to match a stale docstring. Flag any such
  correction explicitly in the acceptance report (don't silently "fix" both).
- Adding new docstrings to already-undocumented private helpers that have
  no docstring at all is in scope only if trivial (one line); do not invent
  elaborate documentation for code this plan doesn't otherwise touch — keep
  the diff about *language and format*, not about expanding documentation
  depth (that's Plan 06's job where it overlaps with metrics/logging).

## Implementation steps

### Step 1 — Inventory

`grep -rn` across `experiments/*.py` for non-ASCII characters (a reasonable
proxy for non-English text, given the codebase's identifiers are ASCII) to
build a concrete list of files/lines needing translation before touching
anything, so the acceptance report can state exactly what was found (not just
"looked fine").

### Step 2 — Translate to English

Translate any non-English comment/docstring/log-message text found, file by
file. `experiments/classifiers.py` (1497 lines, the largest file, predates
the refactor) is the most likely to need this — read it in full rather than
assuming it's clean.

### Step 3 — Google-style docstrings

For every public function/class/method in `experiments/*.py` lacking a
Google-style docstring (or having one in a different style — e.g. NumPy-style,
plain prose, or none), bring it to the `Args:`/`Returns:`/`Raises:` format,
matching `src/tbls/`'s existing convention (see any function there for the
exact formatting: one blank line after the summary, `Args:` with each
parameter as `name: description.`, etc.). Private helpers (leading `_`) get a
one-line docstring at minimum if they don't already have one; a full
`Args:`/`Returns:` block for private helpers is encouraged but not mandatory
if the function is small and the existing one-line description remains
accurate.

### Step 4 — Ruff/mypy docstring lint check

If `.ruff.toml` doesn't already enforce a docstring convention for
`experiments/` (check `exclude`/`per-file-ignores` — `experiments/` may
currently be exempted from some `D`-rule (pydocstyle) checks that
`src/tbls/` enforces), consider whether to extend the same ruleset to
`experiments/` now that it's been brought up to the same standard. If this
is a nontrivial ruleset change (i.e. would require touching many additional
lines beyond what Steps 2-3 already fixed), note it in the acceptance report
as a deliberate scope decision rather than silently expanding this plan's
diff further.

## Verification commands

```bash
grep -rPn '[^\x00-\x7F]' experiments/*.py   # should show zero human-facing non-English text
                                             # (data literals / dataset key names excluded)
uv run ruff check experiments/
uv run ruff format --check experiments/
uv run pytest tests/ -v                      # must still be 100% green -- zero behavior change
uv run mypy src/tbls                         # unaffected scope, run anyway as a sanity check
```

## Acceptance checklist

- [ ] Inventory (Step 1's grep output) included in the acceptance report,
      with a clear statement of what was translated.
- [ ] Every public `experiments/*.py` function/class/method has a
      Google-style docstring.
- [ ] Zero behavior change: full test suite passes unmodified; if any
      docstring correction revealed a real behavior/docstring mismatch, it
      is called out explicitly (not silently resolved either direction).
- [ ] `ruff check`/`ruff format --check` clean.

## Suggested commits

1. `docs(experiments): translate remaining non-English text to English`
2. `docs(experiments): Google-style docstrings across experiments/*.py`
