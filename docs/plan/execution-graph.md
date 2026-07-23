# Execution graph

Tracks the status of execution plans under `docs/plan/`. Status values:
`READY`, `IN_PROGRESS`, `IMPLEMENTED` (work complete + verified, pending
reviewer acceptance), `ACCEPTED`, `BLOCKED`, `CONDITIONAL`, `REJECTED`.

Implementation agents set `IN_PROGRESS` -> `IMPLEMENTED`; only the reviewer
process (see `AGENTS.md`) sets `ACCEPTED`. Do not release downstream plans while
a node is only `IMPLEMENTED`.

| Plan | Node | Hard predecessors | Status | Implementation commit | Report |
|------|------|-------------------|--------|-----------------------|--------|
| `01-tbls-graph-ifs-strategy-and-grid-search.md` | `01-tbls-graph-ifs-grid` | none | `ACCEPTED` | `8effeaf` (master) | `reports/01-tbls-graph-ifs-strategy-and-grid-search.md` |
| `02-multiview-cca-gfcca-fusion-convention.md` | `02-multiview-fusion` | `01-tbls-graph-ifs-grid` | `ACCEPTED` | `8effeaf`..`2b8c258` (master; `2b8c258` = reviewer fix for the `view_groups=None` default bug) | `reports/02-multiview-cca-gfcca-fusion-convention.md` (implementer), `reports/02-multiview-cca-gfcca-fusion-convention-review.md` (reviewer, ACCEPTED) |
