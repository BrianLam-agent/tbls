# Execution graph

Tracks the status of execution plans under `docs/plan/`. Status values:
`READY`, `IN_PROGRESS`, `IMPLEMENTED` (work complete + verified, pending
reviewer acceptance), `ACCEPTED`, `BLOCKED`, `CONDITIONAL`, `REJECTED`.

Implementation agents set `IN_PROGRESS` -> `IMPLEMENTED`; only the reviewer
process (see `AGENTS.md`) sets `ACCEPTED`. Do not release downstream plans while
a node is only `IMPLEMENTED`.

| Plan | Node | Hard predecessors | Status | Implementation commit | Report |
|------|------|-------------------|--------|-----------------------|--------|
| `01-package-refactor-and-real-dataset-verification.md` | `01-package-refactor` | none | `IMPLEMENTED` | `a94c64d`..`d664ab6` (master) | `reports/01-package-refactor-and-real-dataset-verification.md` |
