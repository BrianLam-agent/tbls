# Execution graph

Tracks the status of execution plans under `docs/plan/`. Status values:
`READY`, `IN_PROGRESS`, `IMPLEMENTED` (work complete + verified, pending
reviewer acceptance), `ACCEPTED`, `BLOCKED`, `CONDITIONAL`, `REJECTED`.

Implementation agents set `IN_PROGRESS` -> `IMPLEMENTED`; only the reviewer
process (see `AGENTS.md`) sets `ACCEPTED`. Do not release downstream plans
while a node is only `IMPLEMENTED`. Accepted plans are removed from this
directory once their history is no longer needed (see git log for prior
accepted plans 01/02).

| Plan | Node | Hard predecessors | Status |
|------|------|-------------------|--------|
| `03-worked-examples.md` | `03-examples` | none | `IN_PROGRESS` |
| `04-ablation-variants-and-vectorization.md` | `04-ablation-vectorize` | none (touches same files as 01, already accepted) | `IMPLEMENTED` |
| `05-experiments-code-hygiene.md` | `05-experiments-hygiene` | none | `READY` |
| `06-metrics-logging-visualization.md` | `06-metrics-logging-viz` | `05-experiments-hygiene` (sequencing preference, not a correctness dependency) | `READY` |
