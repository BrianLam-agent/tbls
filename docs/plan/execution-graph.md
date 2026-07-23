# Execution graph

Tracks the status of execution plans under `docs/plan/`. Status values:
`READY`, `IN_PROGRESS`, `IMPLEMENTED` (work complete + verified, pending
reviewer acceptance), `ACCEPTED`, `BLOCKED`, `CONDITIONAL`, `REJECTED`.

Implementation agents set `IN_PROGRESS` -> `IMPLEMENTED`; only the reviewer
process (see `AGENTS.md`) sets `ACCEPTED`. Do not release downstream plans while
a node is only `IMPLEMENTED`.

| Plan | Node | Hard predecessors | Status | Implementation commit | Report |
|------|------|-------------------|--------|-----------------------|--------|
| `01-tbls-graph-ifs-strategy-and-grid-search.md` | `01-tbls-graph-ifs-grid` | none | `IMPLEMENTED` | `8effeaf` (master) | *(none - flagged: Plan 01 has no acceptance report; reviewer to close)* |
| `02-multiview-cca-gfcca-fusion-convention.md` | `02-multiview-fusion` | `01-tbls-graph-ifs-grid` | `IMPLEMENTED` | (see `git log`, master) | `reports/02-multiview-cca-gfcca-fusion-convention.md` |
