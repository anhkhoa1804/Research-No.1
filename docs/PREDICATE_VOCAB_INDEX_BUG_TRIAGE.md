# STOP finding: predicate-vocabulary index mismatch corrupts classifier-based predicate scoring

Discovered while running Experiment A (PredCls, GT pairs, raw model) against
the recovered checkpoint. Not fixed — investigation and specification only,
per the same STOP discipline as `docs/GT_EXTRACTION_BUG_TRIAGE.md`.

Severity: **P0/P1 — corrupts predicate-classifier-based scoring for every
current/future eval and training run that resolves its predicate vocabulary
against a `vg150_root` carrying this vocabulary file**, not specific to the
recovered checkpoint.

## The symptom that surfaced it

Experiment A (`eval_sgg_use_gt_pairs=True`, `eval_sgg_predicate_score_mode=
"classifier"`, no calibration/ensemble/freq-bias overlays) on 16 real
validation images: **R@50 = 1.4%, mR@50 = 0.94%** (3 of 213 GT relations
recovered) — far below what a genuinely raw, uncalibrated but *correctly
wired* classifier should produce, and incompatible with the checkpoint's
self-reported historical numbers (R@50=67.09%, mR@50=22.64%, albeit under a
different, fully-calibrated configuration).

`predicate_diag` from the same run is the real tell: **ground truth's
dominant class is "on" (83/213, ~39%)**, the single most common VG150
predicate by a wide margin — **yet "on" never appears in the model's
top-8 predicted classes at all.** The model's most-predicted class is
"holding" (47), followed by **"flying in" (39)** — a rare, low-frequency
predicate. A raw CE-trained classifier evaluated on imbalanced data
over-predicts the majority class; it does not spontaneously prefer a tail
class over the head class it saw 39% of the time in this very sample. That
inversion is the signature of a **scrambled logit-to-label index mapping**,
not a weak model.

## Root cause (`VERIFIED FROM DATA`)

`datasets_vg150_clean/vocabulary/predicates.json` — the file
`_load_vg150_vocab`/`scan_vg150_predicate_vocab` resolve the predicate
classifier's index-to-label mapping from (`openvocab_rel/datasets/vg150_loader.py:286-346`,
`:1129-1137`) — contains:

```
1:above 2:across 3:against 4:along 5:and 6:at 7:attached to 8:behind
9:belonging to 10:between 11:carrying 12:covered in 13:covering 14:eating
15:flying in 16:for 17:from 18:growing on 19:hanging from 20:has 21:holding
22:in 23:in front of 24:laying on 25:looking at 26:lying on 27:made of
28:mounted on 29:near 30:of 31:on 32:on back of 33:over 34:painted on
35:parked on 36:part of 37:playing 38:riding 39:says 40:sitting on
41:standing on 42:to 43:under 44:using 45:walking in 46:walking on
47:watching 48:wearing 49:wears 50:with
```

Compared against `STANDARD_VG150_PREDICATES`
(`tools/prepare_vg150_subset.py`) — the canonical 50-predicate set this
repo actually uses to *filter* relationships during data preparation
(`_build_relationships`, `tools/prepare_vg150_drive_clean.py:171-207`):

- **Extra, non-canonical entries in the vocab file**: `"growing on"`
  (index 18), `"says"` (index 39) — neither is in `STANDARD_VG150_PREDICATES`.
- **Missing canonical entries**: `"next to"`, `"wrapped around"` — both
  are in `STANDARD_VG150_PREDICATES` and both are **real, common**
  predicates in the actual data.

**Confirmed directly against the cleaned data** (`datasets_vg150_clean/train.jsonl`):

| Predicate | Occurrences in train.jsonl |
|---|---|
| `next to` | **19,410** |
| `wrapped around` | **4,478** |
| `growing on` | 0 |
| `says` | 0 |

So the vocabulary file has two dead slots for predicates that occur **zero**
times in the actual filtered data, and has **no valid index at all** for two
predicates that occur a combined **23,888 times** in `train.jsonl` alone.
Because index assignment is positional (`{name: idx for idx, name in
enumerate(names)}`, `vg150_loader.py:313/323/333`), inserting `"growing
on"` at position 18 shifts every alphabetically-later predicate's index by
one relative to the true canonical ordering, and `"says"`'s insertion
compounds a second shift further down the alphabet — corrupting the
index-to-label correspondence for a large contiguous stretch of the
vocabulary, not just the 4 directly-affected predicates.

**Not introduced by this session's work**: `diff` confirms
`datasets_vg150_clean/vocabulary/predicates.json` is byte-identical to
`datasets/vg_raw/vocabulary/predicates.json` — a pre-existing raw file on
this machine, not created by any tool run this phase.
`tools/prepare_vg150_drive_clean.py`'s `_copy_vocab` step copies this file
verbatim into every prepared output directory without validating it
against `STANDARD_VG150_PREDICATES` — the same set the tool uses one
function away to filter relationships. **The filtering logic and the
vocabulary file it ships alongside the filtered data are drawn from two
different, silently-diverged sources.**

This is the third instance, discovered across this and the immediately
preceding phase, of exactly this pattern: multiple independent "canonical
VG150 predicate list" artifacts in this repository that have silently
drifted apart —
1. `configs/predicate_metadata_vg150.json` (52 keys: extra `"around"`,
   `"growing on"`, `"says"`; missing `"wrapped around"` — found earlier
   this session, not yet fixed, documented but non-crashing because
   `load_predicate_metadata` defensively pre-populates defaults first).
2. `datasets/vg_raw/vocabulary/predicates.json` → copied verbatim into
   every `tools/prepare_vg150_drive_clean.py` output (this finding) — extra
   `"growing on"`, `"says"`; missing `"next to"`, `"wrapped around"` —
   **crashing/corrupting, not merely cosmetic**, because this one drives
   actual classifier index assignment.
3. `STANDARD_VG150_PREDICATES` (`tools/prepare_vg150_subset.py`) — the one
   *correct*, actually-used-for-filtering source.

## Blast radius

- **Predicate-classifier-based scoring** (`eval_sgg_predicate_score_mode
  in {"classifier","ensemble"}` when `eval_sgg_use_predicate_classifier=True`,
  which is this repo's default combination): index-to-label mapping is
  wrong for a large contiguous range of the vocabulary. **Definitely
  affected** — directly demonstrated by Experiment A's results and
  `predicate_diag` above.
- **Text-based (CLIP cosine) scoring** (`eval_sgg_predicate_score_mode="text"`):
  encodes predicate *strings* via the CLIP text encoder; content, not
  position, determines each embedding, so the 48 correctly-named entries
  keep correct embeddings regardless of index order. **Degrades gracefully,
  not catastrophically** — but `"next to"`/`"wrapped around"` are still
  entirely absent from the vocabulary (never encoded, never scorable at
  all under this path either), and `"growing on"`/`"says"` still waste a
  text-embedding slot on predicates that never occur.
- **Training** (`train.py`): uses its own predicate-vocab resolution path;
  not traced in this triage (out of scope — this was found via evaluation,
  not training, and the user's instructions for this phase are audit +
  baseline, not architecture/training changes). **Flagging as
  UNVERIFIED, not "unaffected"** — this needs the same check before any
  future training run is trusted.
- **The GT-extraction fix from the prior phase (commit `220c5c2e`) is
  unaffected** — its regression tests compare predicate *strings* via set
  equality, never through a classifier index, so that fix and its tests
  remain valid and correct independent of this finding.
- **Whether the original GCP training run (that produced
  `pure_best_adapt_light_mR50.pt`) suffered from this same defect is
  unknown** — its own `vg150_root='datasets'` directory (a different path,
  a different point in time) no longer exists on this machine, so its
  `vocabulary/predicates.json` cannot be inspected. Two explanations remain
  open, and this triage does not choose between them:
  1. The original training/eval environment had the *same* defective
     vocab file, in which case the historical 67%/22.6% numbers may
     themselves be measuring a scrambled-index classifier (whether the
     `"ensemble"` score mode's text-scoring component compensated enough
     to still produce plausible numbers is untested).
  2. The original environment had a *correct* vocab file (this defect is
     specific to whatever produced the copy of `datasets/vg_raw/` present
     on this machine today), in which case the historical numbers are
     unaffected but **this machine's current `datasets_vg150_clean/`
     cannot currently reproduce them** regardless of which checkpoint is
     evaluated.

## Why Experiment A's numbers must not be reported as a baseline

R@50=1.4%/mR@50=0.94% are **real, reproducible outputs of the current
pipeline** (`VERIFIED FROM EXPERIMENT`) but are confounded by this
vocabulary defect to an unknown degree — likely the dominant cause of the
near-total collapse, per the `predicate_diag` evidence above, but not
formally isolated from other possible contributors (sample size of 16
images; the `explicit_spoa_enabled=False` compatibility override; general
raw/uncalibrated weakness) without a controlled re-run on a corrected
vocabulary. Reporting these numbers as "the post-fix baseline" would
repeat exactly the mistake this whole engagement has been structured to
avoid — treating an artifact of a pipeline bug as if it were a measurement
of the model.

## Performance note (separate from correctness)

The same run took **22,021 seconds (~6.1 hours) for 16 images** (~23
minutes/image) on this CPU-only machine — CLIP loading was 169s, state-dict
loading ~2s, the remainder was `eval_sgg_standard` itself. This is far
beyond what a single CLIP ViT-L/14-336 forward pass plus relation decoding
should cost even on CPU, and is almost certainly dominated by
`_predict_object_label_candidates_from_clip`'s per-object,
prompt-ensembled CLIP calls for the SGCLS/SGDET task branches (up to 32
objects/image, each requiring its own crop + multi-prompt CLIP text/image
scoring against a 150-class vocabulary) — not investigated further in this
triage since it's a performance question, not a correctness one, but
**flagging it now because it makes any larger-sample "trustworthy
baseline" impractical on this machine without either GPU access or
restricting future smoke runs to `predcls`-only paths** (skipping
SGCLS/SGDET) to avoid paying this cost for tasks not currently needed.

## Not implemented (per RULE 2 / explicit STOP instructions)

No fix has been applied. The two most obvious candidate fixes — (a)
regenerate `vocabulary/predicates.json` from `STANDARD_VG150_PREDICATES`
inside `tools/prepare_vg150_drive_clean.py`'s `_copy_vocab` step instead of
copying the raw source verbatim, or (b) validate the copied vocab file
against `STANDARD_VG150_PREDICATES` at prepare-time and fail loudly on
mismatch (matching this repo's existing "fail loud, don't silently
degrade" convention from the earlier frequency-prior fix) — are both
plausible, but neither is authorized by this triage. Choosing between them,
and deciding whether `configs/predicate_metadata_vg150.json`'s parallel
defect should be fixed in the same pass, is a decision for the next
explicitly-authorized fix phase.
