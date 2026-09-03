# `p64` — `p41` (geometry on the field's own metric) replicated on the held-out TEST split

Run: `runs/p64_test_geometry_sgg`, exit 0, 156 s, CPU only, no GPU, no training.
Identical to `p41` in every argument except `--dump`, which points at the
`p54` TEST cache (`runs/p54_test_relfeat_cache/pair_logits_relfeat.pt`,
10,403 images, 132,334 GT rows) instead of the `p24` VALIDATION cache
(`runs/p24_full_val_cache/pair_logits.pt`, 10,401 images, 132,556 GT rows).
`--prior` and `--train-jsonl` are unchanged (`frequency_prior_train.json`,
`train.jsonl`) — the geometry probe is fitted on TRAIN in both runs, so
nothing about this replication changes what the probe was allowed to see.

**This closes a real gap, not an already-covered one.** `docs/TEST_SPLIT_REPLICATION_RESULT.md`
(`p54`/`p59`/`p61`/`p62`) replicated the WPRD-side findings (items 1, 2, 3, 6,
7 of its pre-registration) on test. None of those items is `p41`'s claim:
`p41` is the one result in this cycle measured on the field's own composed
R@50/mR@50/Pareto metric instead of WPRD, and `docs/HYPOTHESIS_MATRIX_LIVE.md`
H7 still listed it as untested on held-out data ("Next test: cross-model
geometry baseline" — no held-out-test entry). This run is that missing
replication. It was not formally pre-registered before running (unlike
`p54`'s items), so it is reported with that caveat attached, not as a
pre-registered pass/fail.

## Input verification, done before running

- `p54` cache present, 338 MB, contains `obj_boxes`, `text_logits`,
  `cls_logits`, `gt_pred`, `pair_index` etc. — the same schema `p59`/`p60`
  already validated on this exact file (`Mech` subclasses `CPA.Bench`, which
  both tools also use).
- `datasets_vg150_clean/train.jsonl` and `frequency_prior_train.json` present,
  identical files used by `p41` (same paths, no copy/versioning needed since
  both are TRAIN-derived and untouched by which split is being *evaluated*).
- **No validation-derived selection leaks into test**: the geometry probe's
  weights come only from `train.jsonl` (disjoint from both validation and
  test); the mixing weight `w` and the tau grid are architectural constants
  fixed in `p41`'s command line, not re-tuned here. The one place a weight is
  chosen by looking at the eval split at all is the same one `p41` already
  named as a limitation — "`w` in the grid IS selected on validation, which
  favours geometry" — and on this run it is selected on test, its own exact
  analogue, not on a different split smuggled in.
- model-term identity gate: `4.292e-06` (test) vs `3.576e-06` (val, from
  `p41`'s log) — both pass the `<1e-4` assert; this is the same check `p33`
  onward has always run before trusting a cache.

## Result — every qualitative conclusion replicates

| tau | split | prior-only R/mR | prior+MODEL R/mR/Pareto | best prior+GEOMETRY (w) | winner | margin |
|---|---|---|---|---|---|---|
| 0.00 | val  | 66.593 / 22.304 | 67.168 / 23.204 / **+0.901** | −1.048 (w=0.25) | **MODEL** | +1.948 |
| 0.00 | test | 66.636 / 22.027 | 67.013 / 22.401 / **+0.374** | −1.241 (w=0.25) | **MODEL** | +1.615 |
| 0.05 | val  | 66.358 / 24.619 | 66.834 / 25.040 / **+2.736** | +1.020 (w=0.25) | **MODEL** | +1.716 |
| 0.05 | test | 66.420 / 24.188 | 66.673 / 24.096 / **+2.069** | +0.907 (w=0.25) | **MODEL** | +1.162 |
| 0.10 | val  | 66.039 / 26.419 | 66.177 / 26.508 / +0.864 | **+2.690** (w=0.25) | **GEOMETRY** | +1.826 |
| 0.10 | test | 66.103 / 26.100 | 66.081 / 25.809 / −0.303 | **+2.653** (w=0.25) | **GEOMETRY** | +2.956 |
| 0.20 | val  | 61.773 / 28.803 | 62.993 / 29.193 / +1.071 | **+3.353** (w=1.00) | **GEOMETRY** | +2.281 |
| 0.20 | test | 61.816 / 28.375 | 62.995 / 28.959 / +1.210 | **+3.050** (w=1.00) | **GEOMETRY** | +1.841 |

Four for four: the winner at every tau is identical between splits, and the
best geometry mixing weight `w` is identical at every tau (0.25, 0.25, 0.25,
1.00). The magnitude drifts by 0.3–0.5 Pareto points at tau ∈ {0, 0.05, 0.2}
and by more at tau = 0.10, where the model's own Pareto gap flips sign
(val +0.864 → test −0.303) — this makes the geometry-wins verdict at
tau = 0.10 *stronger* on test, not weaker. `prior+MODEL+GEOMETRY` and the
"model adds X on top of geometry" reading (full numbers in
`runs/p64_test_geometry_sgg/base.json`) show the same pattern: geometry adds
essentially nothing at tau ≤ 0.05 and matters increasingly at tau ≥ 0.10, on
both splits.

**Adjudication: REPLICATED.** Every sign, every ranking, every selected `w`
carries over. The central claim — *at the checkpoint's actual operating
region (tau ≤ 0.05), the model beats geometry on the field's metric despite
geometry's WPRD superiority; discrimination is not calibration* — holds on
held-out data with no test-set-specific re-tuning beyond the one leakage
mode (`w` chosen on the eval split) that `p41` already disclosed and that
this run inherits identically, applied to test instead of to validation.

This was not a pre-registered replication (no threshold table existed before
the run), so "REPLICATED" here is a descriptive judgment against `p41`'s own
numbers, not a pass against a committed criterion — flagged per the
project's evidence-class discipline. `docs/TEST_SPLIT_REPLICATION_PREREGISTRATION.md`
should be read as covering items 1–7 (WPRD side) only; this result is
additional, not part of that pre-registration.

## The known bug in the nested arm — reproduced faithfully, then corrected separately

`geometry_sgg_baseline.py`'s nested-CV arm selects the geometry weight `w`
inside each fold with `key = (m["mR"],)` — mR alone, no R@50 floor. Geometry
always costs R@50 relative to `prior+MODEL` at every tau tested, so this key
picks `w = 0.0` on **every fold at every tau**, making the nested arm read
exactly `prior_only` and its Pareto gap exactly `0.000`. This was already
named as broken in `docs/GEOMETRY_SGG_BASELINE_RESULT.md` limitation 1
("reported as broken rather than re-run"). Reproduced here, bit for bit, on
test:

```
tau=0.0   nested pareto +0.000  w=[0,0,0,0,0]   |  prior+MODEL pareto +0.374
tau=0.05  nested pareto +0.000  w=[0,0,0,0,0]   |  prior+MODEL pareto +2.069
tau=0.1   nested pareto +0.000  w=[0,0,0,0,0]   |  prior+MODEL pareto -0.303
tau=0.2   nested pareto +0.000  w=[0,0,0,0,0]   |  prior+MODEL pareto +1.210
```

Identical structurally to `p41`'s own nested log on validation (also
`w=[0,0,0,0,0]` at every tau). The bug is confirmed split-independent — it is
a property of the selection rule, not of either dataset.

**Per instruction, the original (bug-faithful) computation above is kept as
the record of what `p41` reported, unmodified. A separate, corrected script**
(`tools/geometry_sgg_nested_corrected.py`, new this run, does not touch
`geometry_sgg_baseline.py`) **fixes the key to `(R@50 >= floor, mR@50)`**,
matching the convention used everywhere else in this project
(`tools/candidate_scorer_probe.py`), with `floor` set to prior-only R@50 *at
that tau* (the arm must not cost recall relative to the prior alone — the
same invariant `docs/RESEARCH_STATE.md` §9 states for the oracle's floor).
A fixed literal floor (e.g. `candidate_scorer_probe.py`'s 0.665) is not
usable here because prior-only R@50 itself falls to 61.8% at tau=0.2.

Run on both caches, `runs/p64_test_geometry_sgg_nested_corrected` (test) and
`runs/p64_val_geometry_sgg_nested_corrected` (val), same probe fit, same
5-fold split by image (`fold_of_image`, salt 0):

| tau | split | nested GEOMETRY (corrected) Pareto | prior+MODEL Pareto | winner |
|---|---|---|---|---|
| 0.00 | val  | −0.311 | +0.901 | **MODEL** |
| 0.00 | test | −0.493 | +0.374 | **MODEL** |
| 0.05 | val  | +1.456 | +2.736 | **MODEL** |
| 0.05 | test | +1.396 | +2.069 | **MODEL** |
| 0.10 | val  | +0.652 | +0.864 | **MODEL** |
| 0.10 | test | +0.793 | −0.303 | **GEOMETRY** |
| 0.20 | val  | −0.007 | +1.071 | **MODEL** |
| 0.20 | test | +0.208 | +1.210 | **MODEL** |

**This is a new, non-pre-registered finding, reported separately and not as
a replacement for `p41`'s "best fixed w" number above.** Once the geometry
weight is chosen without looking at the evaluation split at all (cross-fitted
within training folds, floor-respecting), MODEL wins 7 of 8 val+test cells,
including tau = 0.10 on validation and tau = 0.20 on both splits, where the
"best fixed w" analysis (which explicitly picks `w` by looking at the eval
split, a limitation `p41` itself disclosed) had shown GEOMETRY winning by
+1.8 to +2.3 Pareto points. The one exception is test at tau = 0.10, where
GEOMETRY still wins even under the corrected, leakage-free selection
(+0.793 vs −0.303).

**Reading this carefully:** this does not mean "`p41`'s geometry-wins-at-high-tau
claim is false" — `p41`'s own numbers are exactly reproduced above and stand
as reported, with their disclosed limitation attached. It means the *size* of
geometry's advantage at tau ≥ 0.1 was inflated by evaluation-split leakage in
how `w` was chosen, and a leakage-free selection narrows or reverses it in
7 of 8 cells. The narrowest, most defensible statement combining both: **the
model beats geometry at its own operating region (tau ≤ 0.05) under either
selection procedure; at tau ≥ 0.1 geometry's apparent advantage shrinks
substantially, and mostly disappears, once the mixing weight stops being
chosen by looking at the split being scored.**

## Files

- `runs/p64_test_geometry_sgg/{base.json,stdout.log,command.txt,provenance.json,result.json}`
- `runs/p64_test_geometry_sgg_nested_corrected/{nested.json,stdout.log,...}`
- `runs/p64_val_geometry_sgg_nested_corrected/{nested.json,stdout.log,...}` (validation-side comparison run, same corrected script)
- `tools/geometry_sgg_nested_corrected.py` (new; does not modify `tools/geometry_sgg_baseline.py`)
