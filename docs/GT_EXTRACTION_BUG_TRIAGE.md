# Bug triage: GT-triplet extraction index misalignment in `eval_sgg_standard`

Status: **investigation and specification only — no production code changed.**
Not committed as of this writing (this file itself is documentation-only and
safe to commit later if you want it tracked, but no commit has been made).

Severity: **P0 — confirmed scientific-validity defect in the default,
maintained evaluation path.** Discovered during the Phase 7 prop@K protocol
audit; prop@K turned out to be one of several consumers of a shared, broken
ground-truth-reconstruction helper.

---

## 1. Root cause

Three structures are involved, each produced/consumed at a different point
in the pipeline:

### 1a. Producers

`ex["pairs"]`, `ex["rel_preds"]`, `ex["rel_pos_mask"]` are all originally
produced **together, index-aligned by construction**, by
`_build_relation_entries` (`openvocab_rel/datasets/vg150_loader.py:142-211`),
called from `VG150JSONLDataset.__getitem__` (and the other dataset variants,
lines 610, 817, 950, 1229) at lines 654-660 / 852-856 / 985-989 / 1260-1266.

`_build_relation_entries`'s ordering invariant (`vg150_loader.py:156-210`):
1. All **positive** (annotated) relationships come first, in the order they
   appear in the raw `relationships` list for that image (lines 161-171,
   `positive_entries`/`positive_preds`).
2. If `use_all_pairs=True` (`TrainConfig.use_all_pairs`, default `True`,
   `config.py:186`), negative (non-annotated) pairs are appended **after**
   all positives, in row-major `(subj_idx, obj_idx)` order excluding
   self-pairs and already-positive pairs, capped by `negative_pair_ratio`
   (default `2.0`, `config.py:187`) and/or `max_pairs` (lines 186-210).
3. If `use_all_pairs=False`, only positives are returned (lines 173-179) —
   `pairs`/`rel_preds`/`rel_pos_mask` (all `True`) are shorter but still
   mutually aligned.

In both cases, `pairs[i]`, `rel_preds[i]`, `rel_pos_mask[i]` (and
`rel_is_pos[i]`) refer to the **same** relationship at every index `i` —
this is the invariant every downstream consumer implicitly assumes.

### 1b. Where the invariant is broken

`eval_sgg_standard` (`openvocab_rel/evals.py:1333`) needs to run the model
over a **different, evaluation-specific candidate pair set** than whatever
the dataset item happened to be constructed with (because eval must score
*every* candidate pair for R@K/mR@K, not the training-time positives +
sampled-negatives set). It does this at lines 1577-1594:

```python
for ex in batch:
    boxes = ex.get("obj_boxes", None)
    n_obj = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else 0
    pair_list = _extract_gt_pairs(ex) if use_gt_pairs else _build_all_ordered_pairs(n_obj)   # 1587
    pair_lists.append(pair_list)
    valid = n_obj > 1 and len(pair_list) > 0
    valid_mask.append(valid)
    ex_eval = dict(ex)                # 1591 -- shallow copy
    ex_eval["pairs"] = pair_list      # 1592 -- OVERWRITES "pairs" only
    ex_eval["_pred_vocab"] = pred_vocab
    eval_batch.append(ex_eval)
```

`use_gt_pairs = bool(getattr(cfg, "eval_sgg_use_gt_pairs", False))`
(`evals.py:1393`); `TrainConfig.eval_sgg_use_gt_pairs` defaults to `False`
(`config.py:383`) and every maintained eval entrypoint keeps that default
(`scripts/eval/eval_l4_phase34.sh:18`,
`EVAL_SGG_USE_GT_PAIRS="${EVAL_SGG_USE_GT_PAIRS:-false}"`).

Under this default, `pair_list = _build_all_ordered_pairs(n_obj)`
(`evals.py:43-44`):

```python
def _build_all_ordered_pairs(n_obj: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(int(n_obj)) for j in range(int(n_obj)) if i != j]
```

— a **fresh, strict row-major ordering**, built from scratch, with **no
relationship to the order `_build_relation_entries` used**. Only the
`"pairs"` key is overwritten on the shallow copy `ex_eval`; `"rel_preds"`
and `"rel_pos_mask"` are carried over **unchanged**, still in the original
positives-first-then-row-major-negatives order.

`ex_eval` (aliased as `ex` in the per-image loop, since
`eval_batch.append(ex_eval)` at line 1594 and the loop at line 1629 does
`for ex, pair_list, rel_feat, valid in zip(eval_batch, pair_lists, rels, valid_mask)`)
is then passed to `_collect_gt_triplets(ex, device)` at line 1633.

### 1c. The consumer that assumes the (now-broken) invariant

`_collect_gt_triplets` (`evals.py:518-565`):

```python
def _collect_gt_triplets(ex, device):
    pairs = ex.get("pairs", [])          # 519 -- NOW the fresh row-major list
    preds = ex.get("rel_preds", [])      # 520 -- STILL the original positives-first list
    pos_mask = ex.get("rel_pos_mask", None)   # 521 -- STILL the original list
    ...
    n = min(len(pairs), len(preds))      # 533
    for idx in range(n):                 # 534
        if pos_mask is not None and idx < len(pos_mask) and not bool(pos_mask[idx]):
            continue
        pair = pairs[idx]                # 537 -- reads from the NEW ordering
        ...
        pred = str(preds[idx]).strip().lower()   # 546 -- reads from the OLD ordering, same idx
        ...
```

It zips `pairs[idx]` against `preds[idx]`/`pos_mask[idx]` **positionally**.
This is exactly the invariant `_build_relation_entries` guarantees at
production time — but it no longer holds once `evals.py:1592` replaces
`pairs` with an independently-ordered list while leaving `preds`/`pos_mask`
untouched.

### 1d. Data-flow summary

```
vg150_loader.py:_build_relation_entries()
   → pairs, rel_preds, rel_pos_mask   (mutually aligned by construction)
        ↓ (dataset __getitem__, unchanged, batched by VG150DataLoader)
evals.py:1577-1594  eval_sgg_standard(), per-image prep loop
   → ex_eval["pairs"] = _build_all_ordered_pairs(n_obj)   (NEW ordering)
     ex_eval["rel_preds"], ex_eval["rel_pos_mask"]         (OLD ordering, untouched)
        ↓
evals.py:1633  gt = _collect_gt_triplets(ex_eval, device)  (BROKEN: positional zip of two
                                                             independently-ordered lists)
        ↓ (single "gt" dict, computed once per image, shared by every task branch)
evals.py:1677  _update_pair_proposal_diag(gt, ...)          [prop@K]
evals.py:1687  _update_pair_rank_diag(gt, ...)              [pair-rank diagnostic]
evals.py:1694-1720  role-swap diagnostic loop               [role_swap_diag]
evals.py:1724  matches = _compute_pred_matches(triplets, gt, ...)  [R@K / mR@K, all 6 tasks]
```

**VERIFIED FROM CODE** for every claim in this section (file/line citations
given inline).

---

## 2. Minimal counterexample

Smallest reproducible case: **3 objects, 2 relationships**, sharing one
endpoint (object 1). Reproduced using the actual repository functions
(`_build_relation_entries`, `_build_all_ordered_pairs`, `_collect_gt_triplets`
imported directly from `openvocab_rel.datasets.vg150_loader` /
`openvocab_rel.evals` — nothing reimplemented). Script:
`bug_triage_repro.py` (scratchpad, reproducible from this report).

```python
obj_boxes = [[0,0,10,10], [20,20,30,30], [40,40,50,50]]   # man, shirt, dog
relationships = [
    {"subject_id": 0, "object_id": 1, "predicate": "wearing"},
    {"subject_id": 1, "object_id": 2, "predicate": "near"},
]
```

Default config (`use_all_pairs=True`, `eval_sgg_use_gt_pairs=False`):

| Structure | Value |
|---|---|
| loader `pairs` (positives-first) | `[(0,1), (1,2), (0,2), (1,0), (2,0), (2,1)]` |
| loader `rel_preds` | `['wearing', 'near', 'relation', 'relation', 'relation', 'relation']` |
| loader `rel_pos_mask` | `[True, True, False, False, False, False]` |
| eval-time `pairs` (row-major, replaces loader's) | `[(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]` |
| **Expected GT** (the real annotations) | `[(0,1,'wearing'), (1,2,'near')]` |
| **`_collect_gt_triplets` actually returns** | `[(0,1,'wearing'), (0,2,'near')]` |

The second relationship is silently reattributed from the real pair `(1,2)`
to the unrelated pair `(0,2)` — `(0,2)` (man, dog) has no annotation at all
in this example, yet is reported as ground truth `"near"`.

**VERIFIED FROM EXPERIMENT** — full output including 4 additional cases
(single relation, zero relations, `use_all_pairs=False`, finite
`negative_pair_ratio`) is in section 4 below; all were executed this
session via `py -3 bug_triage_repro.py` against the live repo code.

---

## 3. Blast radius

`_collect_gt_triplets` is called **exactly once per image**
(`evals.py:1633`; confirmed by repo-wide search — it has no other call
site). The single resulting `gt` dict is then reused, unmodified, by every
task branch for that image:

| Consumer | Location | Status |
|---|---|---|
| `predcls` recall (`_process_task("predcls", ...)` → `_compute_pred_matches(triplets, gt, ...)`) | `evals.py:1724, 1746` | **Definitely affected** |
| `predcls_nogc` | `evals.py:1750` | **Definitely affected** (same shared `gt`) |
| `sgcls` | `evals.py:1772` | **Definitely affected** |
| `sgcls_nogc` | `evals.py:1776` | **Definitely affected** |
| `sgdet` | `evals.py:1827` | **Definitely affected** (predictions come from a separate Grounding-DINO detection path via `_make_detected_example`, `evals.py:809-827`, but `gt` itself is still the same corrupted per-image dict) |
| `sgdet_nogc` | `evals.py:1831` | **Definitely affected** |
| `prop@K` (`gt_pair_recall@K`, `gt_pair_pruned_rate@K`, `gt_triplet_pair_recall@K`) | `_update_pair_proposal_diag`, `evals.py:1461-1503`, called at `1677` | **Definitely affected** — this is what the Phase 7 audit was investigating when the bug was found |
| Pair-rank diagnostic | `_update_pair_rank_diag`, `evals.py:1505+`, called at `1687` | **Definitely affected** |
| Role-swap consistency diagnostic (`role_swap_diag`) | `evals.py:1689-1720` | **Definitely affected** — uses `gt["subj_idx"]`/`gt["obj_idx"]` directly (lines 1700-1701) |
| R@20/50/100, mR@20/50/100 (all computed from the same `_process_task`/`_compute_pred_matches` path, per task above) | — | **Definitely affected**, for every task listed above |
| Predicate-confusion diagnostics (`predicate_diag`, `global_gt_counts`, `zs_gt_counts`) | `evals.py:1637-1673` | **Definitely affected** — these read `gt["pred_labels"]` directly, so counts are inflated/attributed to the wrong images' predicate mix in aggregate (though predicate *strings* themselves are not corrupted, only which pair/box they're attached to) |
| Training (loss computation, `train.py`) | `train.py:1852` (`batch_pairs = [ex["pairs"] for ex in batch]`) | **Unaffected** — training reads `ex["pairs"]` directly from the DataLoader output; the corrupting overwrite exists only inside `eval_sgg_standard`'s local `ex_eval` copies and never reaches `train.py` |
| `eval_query_grounding` (the separate cosine-retrieval eval path, `evals.py:~3513`) | not traced in this triage (out of scope — does not call `_collect_gt_triplets`) | **Not verified either way — out of scope for this triage; flagged for a follow-up check if that path is ever relied on for a reported number** |
| `geometry.py:reconstruct_gt_edges_from_example` / `build_candidate_tensors_from_gt` | `geometry.py:167-214` | **Unaffected but irrelevant** — these implement the *correct* dict-keyed-by-`(s,o)` reconstruction pattern, but grep confirms **zero call sites** anywhere in `openvocab_rel/` or `tests/` — this is unused, unreachable code, not a safety net |

**Nothing in this repo's training path is affected.** The bug is 100%
contained to `eval_sgg_standard`'s ground-truth reconstruction, but within
that function it is total — every metric and diagnostic that function
reports is downstream of the same corrupted `gt`.

---

## 4. Conditionality

Confirmed by direct execution (`bug_triage_repro.py`, all cases run against
live repo code this session):

| Case | `use_all_pairs` (loader) | `eval_sgg_use_gt_pairs` | Result |
|---|---|---|---|
| 2 relations, default config | `True` | `False` (**default**) | **BROKEN** — `(0,2,'near')` fabricated |
| Same data | `True` | `True` (non-default) | **Correct** — `_extract_gt_pairs` preserves order, invariant holds |
| 2 relations, positives-only loader | `False` | `False` (**default**) | **BROKEN** — identical corruption; loader's `use_all_pairs` setting is irrelevant to whether the bug triggers |
| **1 relation only** | `True` | `False` | **BROKEN** — `(1,2,'near')` misreported as `(0,1,'near')`. This corrects an earlier hypothesis: a single relation is *not* inherently safe; it only survives if that one relationship happens to occupy the exact same position in both orderings (which for a lone relation means it must literally be the row-major-first pair, `(0,1)`) |
| 0 relations | `True` | `False` | **Correct** — both lists are empty/trivial, no misalignment possible |
| Finite `negative_pair_ratio=3.0` | `True` | `False` | **BROKEN** — same as the unlimited case; the cap changes how many negatives are appended but not the ordering divergence |

**Root condition**: the bug triggers whenever (a) `eval_sgg_use_gt_pairs=False`
(the default — line 1587 takes the `_build_all_ordered_pairs` branch) **and**
(b) the image has at least one relationship whose positional index under
row-major ordering differs from its positional index under
positives-first ordering. In practice, with VG150 images averaging ~12.6
relationships per image (this session's own diagnostics on the freshly
prepared `datasets_vg150_clean/`), condition (b) is satisfied for
essentially every image with more than a trivial one relationship in the
"lucky" position.

**No other code path found that preserves the invariant** under the
default configuration. The only escape is `eval_sgg_use_gt_pairs=True`,
which is not what any shipped script uses by default.

**VERIFIED FROM CODE + VERIFIED FROM EXPERIMENT.**

---

## 5. Historical analysis

```
git blame -L 1590,1593 -- openvocab_rel/evals.py   →  ^31e89601 (anhkhoa1804, 2026-05-19)
git blame -L 518,520   -- openvocab_rel/evals.py   →  ^31e89601 (anhkhoa1804, 2026-05-19)
```

The `^` prefix marks `31e89601` as a **boundary commit** — the earliest
commit in this repository's history where these lines exist, i.e. they were
present in the very first commit of the project (`git log` shows 87+
commits from `2026-05-19` onward per the earlier repo-cleanup audit). This
predates:

- All 6 commits in the prior validity-fix phase (`c71a4d56` →
  `08d0d1a6`, see `BEHAVIORAL_CHANGE_TABLE.md`). That document's claim that
  "`_build_relation_entries`, the model forward pass, every loss function,
  and the R@K/mR@K computation are all byte-for-byte unchanged across all 6
  commits" **remains accurate** — it correctly means this bug is present
  **identically** in both the pre-fix (`c71a4d56`) and post-fix (`08d0d1a6`)
  reference commits. Nothing in this triage contradicts that document; it
  surfaces a defect neither of those 6 commits touched, introduced, or
  fixed.
- This entire "EXPERIMENT READINESS" phase's own dataset-preparation work
  (today's session).

No commit was found (via the call-site search in section 3) that depends on
`_collect_gt_triplets`'s current positional-zip behavior in a way that
would make it an intentional design choice — no test exercises it, no
other function reads its output except the six task branches and two
diagnostics already listed in section 3, all of which expect *correct* GT,
not this specific corrupted-but-stable-in-shape behavior. There is no
evidence any downstream code has been tuned around the bug's *specific*
wrong outputs (e.g. no metric threshold was suspiciously chosen to
compensate). **INFERENCE**: this reads as an unnoticed oversight — the
person who wrote `evals.py:1592`'s pairs-overwrite most likely intended
`_collect_gt_triplets` to read from the *original*, un-overwritten `ex`,
or intended the overwrite to also carry `rel_preds`/`rel_pos_mask`, and
this was never caught because the repository has no checkpoint to run this
function against yet (confirmed by every earlier phase's `NOT YET
MEASURABLE` findings) — a bug in an evaluation function that has,
apparently, never actually been run end-to-end against a trained model on
this machine.

---

## 6. Proposed fix (NOT implemented)

Three candidate approaches:

**A. Preserve the original `ex["pairs"]` ordering; don't overwrite it.**
Rejected: `eval_sgg_standard` genuinely needs to score a *different*,
typically larger, candidate set than what training's positives+sampled-
negatives representation provides (that's the entire reason the overwrite
exists at line 1587-1592). Keeping the original ordering would mean losing
the ability to evaluate over all pairs; not a correctness fix, a
functionality regression.

**B. When overwriting `ex_eval["pairs"]`, also rebuild `rel_preds`/
`rel_pos_mask` in the same new order.** Feasible: build a
`{(s,o): predicate}` dict from the *original* `pairs`/`rel_preds`/
`rel_pos_mask` (this is exactly what `geometry.py:reconstruct_gt_edges_from_example`
already does, lines 167-184 — currently dead code, per section 3) before
the overwrite, then re-derive `rel_preds`/`rel_pos_mask` for the *new*
`pair_list` by dict lookup (`gt_edges.get((s,o), "relation")`,
positive iff found). This keeps `_collect_gt_triplets`'s positional-zip
logic unchanged and untouched by callers.

**C. Reconstruct GT directly from canonical annotation structures inside
`_collect_gt_triplets` itself, independent of whatever `ex["pairs"]`
currently holds** — e.g. have `_collect_gt_triplets` take the *original*
un-overwritten example (or a separately-stored `ex["_gt_pairs"]` /
`ex["_gt_preds"]` snapshot taken before the line-1592 overwrite) as its
explicit ground-truth source, rather than reading the same mutable
`"pairs"` key that eval also repurposes for candidate scoring.

**Recommendation: Approach C**, with a minimal implementation shape:
before line 1592's overwrite, snapshot the original GT under new keys
(e.g. `ex_eval["_gt_pairs"] = ex["pairs"]`, `ex_eval["_gt_preds"] =
ex["rel_preds"]`, `ex_eval["_gt_pos_mask"] = ex["rel_pos_mask"]`), and
change `_collect_gt_triplets` to read those `_gt_*` keys instead of
`"pairs"`/`"rel_preds"`/`"rel_pos_mask"`. This is safer than Approach B
because:
- It requires no re-derivation logic (no dict-building, no risk of a
  second, subtly-different bug in the reconstruction step).
- It makes the two uses of `"pairs"` (candidate-scoring input vs.
  ground-truth source) structurally distinct instead of aliased through
  the same dict key — the exact ambiguity that caused this bug — so the
  same class of bug cannot recur if a third piece of code later needs to
  overwrite `"pairs"` again for some other purpose.
- It touches only `evals.py` (the eval-time snapshot line, plus
  `_collect_gt_triplets`'s three `.get(...)` calls) — zero changes to
  `vg150_loader.py`, `train.py`, or any training-path code, keeping the
  fix's blast radius exactly matched to the bug's actual blast radius
  (section 3).

Approach B is a reasonable second choice if, for some reason, keeping the
`"pairs"`/`"rel_preds"`/`"rel_pos_mask"` key names stable end-to-end is
preferred — but it duplicates logic that `geometry.py` already has
(currently unused), which is either an opportunity to finally wire that
helper in, or a sign it was written for exactly this purpose and never
connected.

**RECOMMENDATION, not implemented.**

---

## 7. Regression test plan (design only, not implemented)

All tests should live in a new `tests/test_eval_gt_extraction.py`, using
tiny synthetic in-memory `ex` dicts (no real dataset, no CLIP, no GPU —
matching this repo's existing smoke-test philosophy).

1. **`test_multi_relation_gt_recovered_exactly`** (the most important
   test): 3 objects, 2 relationships sharing an endpoint (the section-2
   counterexample). Build `ex` via the real `_build_relation_entries`,
   simulate the eval-time overwrite exactly as `evals.py:1587-1592` does,
   call `_collect_gt_triplets`, and assert **semantic equality of the full
   GT triplet set** — `{(subj_label, pred, obj_label) for each recovered
   entry}` compared against the true annotation set — not just tensor
   shapes or counts. This is the test that fails today and must pass after
   the fix.
2. **`test_single_relation_gt_recovered`**: 1 relationship, chosen
   deliberately *not* at row-major position 0 (e.g. `(1,2)` on a 3-object
   image) — proves the "lucky first case" isn't silently masking the fix's
   correctness.
3. **`test_zero_relation_image_yields_empty_gt`**: confirms the already-
   correct empty case stays correct after the fix (no regression).
4. **`test_dense_multi_object_sparse_relations`**: 5+ objects, only 2-3
   relationships, objects with zero incident relations present — proves
   the fix doesn't accidentally invent GT for unrelated pairs and doesn't
   drop legitimate GT among many negatives.
5. **`test_use_gt_pairs_true_still_correct_after_fix`**: confirms the
   already-correct `eval_sgg_use_gt_pairs=True` path (section 4) is not
   broken by whatever change lands for the `False` path.
6. **`test_use_gt_pairs_false_matches_true_on_same_data`**: the strongest
   cross-check — for the same synthetic image, assert that
   `_collect_gt_triplets`'s recovered GT set is **identical** whether
   `eval_sgg_use_gt_pairs` is `True` or `False` (only the *candidate* pair
   set should differ between the two modes; the *ground truth* must not
   depend on which candidate-generation mode is active). This single test
   directly encodes "GT must not depend on eval configuration," which is
   the actual invariant this bug violates.
7. **`test_duplicate_relationship_pairs_same_endpoints_different_predicate`**
   (edge case carried over from the earlier prop@K-audit requirements):
   confirms behavior is defined (not silently dropped or duplicated) when
   two GT relationships share the same `(s,o)` pair with different
   predicates — relevant since `datasets_vg150_clean`'s own prep-tool
   diagnostics (`data/manifests/vg150_clean_validation_report.md`) show
   this is already de-duplicated at data-prep time, so this test documents
   an assumption the eval code currently relies on implicitly.

Each test should construct its `ex` using the **real** `_build_relation_entries`
and the **real** eval-time overwrite logic (either by importing
`_build_all_ordered_pairs`/`_extract_gt_pairs` directly, or, if the fix
introduces a small reusable helper, by calling that helper) — not a
hand-rolled reimplementation — so the tests exercise the actual code path,
matching this triage's own verification method.

---

## 8. Metric regression plan

**No PURE checkpoint exists anywhere in this environment** (confirmed by
every earlier phase in this engagement — `runs/pre_fix_baseline/STATUS.md`,
`COMPUTE_REQUIREMENTS.md`). This triage does not invent one, and does not
estimate what R@K/mR@K "would have been" pre- vs. post-fix — that is
explicitly `NOT YET MEASURABLE`.

What **can** be validated right now, without a checkpoint:
- The regression tests in section 7, which operate purely on synthetic
  `ex` dicts and the GT-extraction code path — no model forward pass
  needed. These can be written and run (`pytest`) today, on CPU, and would
  conclusively prove old-code-fails / new-code-passes for the *extraction*
  logic itself.
- A synthetic, fully-controlled `eval_sgg_standard` smoke run using
  `tests/fixtures/tiny_vg150/` (already exists from the earlier cleanup
  phase) with a randomly-initialized (untrained) model, purely to confirm
  `eval_sgg_standard` doesn't crash and that its reported `global_gt`
  count (a diagnostic already exposed by `accum[task]["global_gt"]`,
  `evals.py:1742`) changes between old and new code for a multi-relation
  fixture image — this validates the *plumbing*, not the *metric value*
  (an untrained model's R@K is meaningless either way).

What **cannot** be validated without a checkpoint:
- Any actual before/after R@K, mR@K, or prop@K number. Any such number
  would require a trained model, which requires the GPU compute this
  entire phase has been explicitly forbidden from provisioning (Rule 1).
- Whether the fix changes *previously reported* numbers in
  `notes/current_status.tex` / README — those numbers were produced by
  some earlier, external training run this repository has no artifact for
  (checkpoint, logs, or config snapshot); this triage cannot and does not
  attempt to reconcile them.

**Sequencing recommendation**: once compute is available and Experiment 0
(environment/smoke test, already planned in this phase's broader scope) is
running, this fix should land and be verified via section 7's tests
**before** Experiment 1 (the pre-fix controlled baseline) is trained —
otherwise "pre-fix" and "post-fix" checkpoints would both be evaluated
through the same broken GT-extraction path, and the controlled A/B
comparison in `EXPERIMENT_MATRIX.md` would silently inherit this defect
rather than testing what it's meant to test (the validity-fix commits, not
this newly-found one). This is a **scope change** to that matrix worth
flagging explicitly to whoever runs Experiment 1/2: this bug is orthogonal
to, and must be fixed before, *both* Experiment A and Experiment B, or
neither run's reported R@K/mR@K/prop@K numbers will mean what the protocol
says they mean.

---

## 9. Impact assessment

- **Severity: P0.** It corrupts the ground truth for every recall-based
  metric (R@K, mR@K, all 6 tasks) and every pair-proposal diagnostic
  (prop@K, pair-rank, role-swap) in the single evaluation function this
  entire repository relies on for reporting SGG results, under the
  default, documented, only-shipped configuration.
- **Scientific validity impact: severe.** Any number this function has
  ever produced or will produce under `eval_sgg_use_gt_pairs=False` (the
  default) reflects recall against a *partially fabricated* ground truth
  — some fraction of "GT" triplets are attached to the wrong object pair.
  The fraction grows with the number of relationships per image (VG150
  averages ~12.6/image), so this is not a rare-edge-case defect; it is the
  common case.
- **Reproducibility impact: contained but real.** Training is unaffected
  (section 3) — model weights, losses, and checkpoints produced by
  `train.py` are not touched by this bug. Only *evaluation numbers* are
  affected. This means re-evaluating an existing checkpoint with a fixed
  `eval_sgg_standard` is sufficient to get trustworthy numbers — no
  retraining is required to recover from this specific bug once it's
  fixed.
- **Should historical reported results (README / `notes/current_status.tex`)
  under the default path be considered trustworthy?** No — not until
  either (a) they are re-produced through a fixed `eval_sgg_standard`
  against the same checkpoint(s), or (b) it's confirmed those specific
  numbers were generated with `eval_sgg_use_gt_pairs=True` (which this
  triage found to be bug-free, section 4) rather than the shipped default.
  This repository has no artifact recording which configuration produced
  any currently-documented number (a gap already flagged in
  `docs/known_issues.md` from the earlier cleanup phase, re-surfaced here
  as now being load-bearing for this specific question), so this can only
  be answered as **`NOT YET MEASURABLE`** from the current checkout, not
  assumed either way.

---

## 10. Required follow-up: implementation spec

For whoever implements the fix later (Category A/VALIDITY, requires
explicit go-ahead per the user's instructions to this triage — not
authorized by this document alone):

1. In `openvocab_rel/evals.py`, immediately before line 1592
   (`ex_eval["pairs"] = pair_list`), add:
   ```python
   ex_eval["_gt_pairs"] = ex.get("pairs", [])
   ex_eval["_gt_preds"] = ex.get("rel_preds", [])
   ex_eval["_gt_pos_mask"] = ex.get("rel_pos_mask", None)
   ```
2. In `_collect_gt_triplets` (`evals.py:518-521`), change the three reads
   from `ex.get("pairs", [])` / `ex.get("rel_preds", [])` /
   `ex.get("rel_pos_mask", None)` to `ex.get("_gt_pairs", [])` /
   `ex.get("_gt_preds", [])` / `ex.get("_gt_pos_mask", None)`, falling back
   to the old keys only if the new ones are absent (keeps any other,
   not-yet-found caller of `_collect_gt_triplets` outside `eval_sgg_standard`
   working against un-overwritten `ex` dicts without change).
3. Add `tests/test_eval_gt_extraction.py` per section 7, all 7 cases,
   before merging — the multi-relation test (7.1) and the
   `use_gt_pairs=True`-vs-`False` cross-check (7.6) are the two that must
   fail on current `main` and pass after the two-line change above.
4. Do **not** touch `vg150_loader.py`, `train.py`, `geometry.py`, or any
   loss/model file — this fix is fully contained to `evals.py`.
5. Update `docs/known_issues.md` to remove/resolve this entry once fixed
   (it should be added there now, in the interim, as an open P0 item —
   recommended as a small, separate, documentation-only commit, distinct
   from the eventual code fix commit).
6. Per section 8: this fix must land and be verified **before** Experiment
   1/2 (the controlled pre-fix/post-fix baseline training runs) are run,
   not after — otherwise both runs' evaluation numbers inherit this defect
   equally and the comparison stops measuring what `EXPERIMENT_MATRIX.md`
   says it measures.

Every conclusion above is labeled inline; summary: sections 1, 3 (except
the explicitly-flagged `eval_query_grounding` line), 4, and 5's dates are
`VERIFIED FROM CODE`; section 2 and the section-4 table are `VERIFIED FROM
EXPERIMENT`; section 5's motive discussion and section 9's "common case"
framing are `INFERENCE`; sections 6, 7, and 10 are `RECOMMENDATION`;
section 8's explicit numeric claims are marked `NOT YET MEASURABLE` where
they require a checkpoint that does not exist.
