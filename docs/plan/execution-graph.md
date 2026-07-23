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
| `03-worked-examples.md` | `03-examples` | none | `ACCEPTED` |
| `04-ablation-variants-and-vectorization.md` | `04-ablation-vectorize` | none (touches same files as 01, already accepted) | `ACCEPTED` |
| `05-experiments-code-hygiene.md` | `05-experiments-hygiene` | none | `ACCEPTED` |
| `06-metrics-logging-visualization.md` | `06-metrics-logging-viz` | `05-experiments-hygiene` (sequencing preference, not a correctness dependency) | `ACCEPTED` |
| `07-fix-ifs-simple-membership-bandwidth-collapse.md` | `07-ifs-simple-fix` | none | `READY` |

All four `ACCEPTED` as of `c8fae94`. See `docs/plan/reports/0{3,4,5,6}-*.md`
(implementer) and `docs/plan/reports/0{3,4,6}-*-review.md` (reviewer) for
full evidence, two reviewer-found-and-fixed bugs, and one open follow-up
(the GFTBLS collapse finding -- tracked separately, see the review reports,
not yet its own plan file pending user prioritization).
