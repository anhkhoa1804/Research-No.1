# Notes index

Classification of the `.tex` files in this directory, added during the
infrastructure cleanup pass. Nothing below was deleted or rewritten — all
six files are preserved verbatim, per the "do not delete historical
context" principle.

| File | Status | What it is |
|---|---|---|
| `current_status.tex` | CURRENT | Concise implementation snapshot + report-writing guidance; includes a metric table honestly placing PURE's local numbers next to published VG150 baselines (ResCAGCN+PUM, Motifs+CFA, SBG, Hydra-SGG) |
| `paper1_pure.tex` | CURRENT | PURE architecture, phase curriculum, three training pillars, reporting rules |
| `paper2_core.tex` | CURRENT | CORE dataset status (generated, several thousand images, six relation groups) and its intended future role — explicitly states CORE is *not yet* used as PURE's training/eval benchmark |
| `diagrams.tex` | CURRENT | TikZ figures matching the current architecture description in `paper1_pure.tex` |
| `draft.tex` | PLANNING | Self-describing index/pointer document for this directory; also states the "do not overclaim" policy these notes follow |
| `draft.bib` | CURRENT | Supporting bibliography for the metric-table citations in `current_status.tex` |

No file here is HISTORICAL or OBSOLETE in the sense of describing a
superseded architecture — `git log` (87 commits) shows this is a
fast-moving but not yet long-lived project, and these six notes are
consistently the *current* narrative across that history.

## Known gap: two files referenced elsewhere that don't exist here — RESOLVED in `README.md`

**Status at HEAD `140e163f`:** the dead links are gone from `README.md`
(`grep`-verified — the only remaining mention is `README.md:23`, which
explicitly documents the removal). This section is retained as the record
of *why* those files are absent, not as an open issue.

The root `README.md` (before this cleanup pass) referenced two files that
were never committed to this repository:

- `notes/breakthrough_branch_plan.md`
- `notes/pure_conference_upgrade_roadmap.md`

Neither is fabricated or recreated here — recreating their content would be
guessing at what they were meant to say. The dead links have been removed
from `README.md` as part of this cleanup pass (a documentation-honesty fix,
not a content rewrite); if these planning documents exist somewhere else
(a different branch, a training VM, a local machine), they should be
committed for real rather than referenced from a distance.
