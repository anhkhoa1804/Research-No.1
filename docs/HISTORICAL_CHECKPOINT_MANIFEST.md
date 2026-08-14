# Historical checkpoint manifest — canonical artifact freeze

**This is the single source of truth for the recovered historical
checkpoint and everything required to evaluate it.** Machine-readable
counterpart: `data/manifests/historical_checkpoint_v1.yaml`, which
`tools/gcp_preflight.py` reads and enforces.

If this document and any other document disagree, this one wins for
*artifact identity and load compatibility*; `docs/PROJECT_STATUS.md` wins
for overall project state.

Related, deliberately not merged into this file (they are the
*investigation record*, this is the *freeze*):
`docs/HISTORICAL_CHECKPOINT_PROVENANCE.md` (how it was identified),
`docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md` (the forensic
investigation, including its own retraction banner).

---

## 0. The three-way separation this document exists to enforce

The single most important thing here. These are three different things and
must never be conflated in any report, commit message, or paper:

| | What it is | Epistemic status |
|---|---|---|
| **1. Historical claim** | `R@50 = 67.09 %`, `mR@50 = 22.64 %` | `HISTORICAL EVIDENCE` — self-reported by the run that produced it, in a config file. **Single-source. Unreproduced. Never independently checked.** |
| **2. Current verified behavior** | `R@50 = 25.26 %`, `mR@50 = 5.84 %` at n=50, **uncalibrated** | `VERIFIED FROM EXPERIMENT` on current code and current data — but a 50-image diagnostic, not a result |
| **3. Current reproduction target** | the full historical protocol: text path **+** frequency prior α=3.75 | **NOT YET ATTEMPTED AT ANY SAMPLE SIZE** |

Row 1 is **not** a "baseline", **not** "verified", **not** "reproduced",
**not** a "target to beat", and **not** a number this project has earned
the right to cite. It is a claim awaiting a test.

Row 2 is not comparable to row 1 — it deliberately ran with calibration
**off**, and the historical claim is explicitly a *calibrated* number
(`MODEL_NOTE=adapt_light_best_mR50 calibrated with frequency prior alpha
3.75`).

Row 3 is the experiment.

---

## 1. Artifacts and hashes

None of these are in git. All are gitignored and must be transferred out
of band — **`git clone` does not give you a runnable experiment** (§6).

| Artifact | SHA256 | Size | Req'd |
|---|---|---:|:---:|
| `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` | `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442` | 931,057,422 | ✅ |
| `checkpoints/demo_best/frequency_prior.json` | `144d9f928ed9cc213acbe081b1a8791488a92ddab0ed885da7fd6bb6058c6e6a` | 101,944,045 | ✅ |
| `checkpoints/demo_best/demo_config.env` | `c73180698b5b5f6d3a58e5e6b78a39fcb8a81b6250478145d0538a5a107226a6` | 355 | — |
| `datasets_vg150_clean/train.jsonl` | `36bc2923b1a3ddb56e331e9b645607606003bd1c8298d1c950b2cdcca7e31ae5` | 230,887,586 | ✅ |
| `datasets_vg150_clean/validation.jsonl` | `4348ddbb3ce85160d0ebc7522634c68f197b0c99906794e0e4731156740f3412` | 28,950,612 | ✅ |
| `datasets_vg150_clean/vocabulary/predicates.json` | `e4e88e87c3c26bf65426957ae4028b45f03d33643d5520b6ff9b42b7b60d4dc5` | 1,124 | ✅ |
| `datasets_vg150_clean/vocabulary/objects.json` | `9f536376981e758767eab051028568d0ecab0972a7330dc616801878aa359a1b` | 3,056 | — |

The checkpoint has been re-verified byte-identical at **four** separate
points across this engagement. It has never been overwritten, converted, or
modified in place, and must never be.

**Why `train.jsonl` is required for an evaluation-only run** (a
non-obvious dependency): `adaptive_calibration_enabled=true` makes
`_predicate_log_prior_for_eval` read train-split statistics at eval time.
Since the `65686b5f` validity fix, a missing train split raises
`MissingTrainStatisticsError` rather than silently falling back to the
split being scored. Omit `train.jsonl` and the run dies loudly — which is
correct, but only if you know to bring the file.

### 1.1 Frequency prior — structural expectations

`_load_frequency_bias` (`evals.py`) returns `None` — meaning *silently no
calibration* — on six separate conditions and warns on none of them. The
preflight therefore checks each condition independently:

| Property | Expected | Why it matters |
|---|---|---|
| parses as JSON | yes | bare `except Exception: return None` |
| `predicate_vocab` length | 50 | empty ⇒ `return None` |
| `predicate_vocab` value | `== sorted(STANDARD_VG150_PREDICATES)` | mismatched labels are silently filled with `default_log_prob` |
| `global_log_probs` length | 50 | wrong length ⇒ `return None` |
| `pair_log_probs` rows | all length 50 | wrong-length rows are silently dropped |
| `subject_log_probs` / `object_log_probs` rows | all length 50 | same |

Counts as recovered: 74,884 pair entries, 12,030 subject, 9,689 object;
`smoothing = 1.0`; `default_log_prob = -3.912023005428146`;
`object_vocab` = 25,918 **raw, pre-canonicalization** label strings (not
the 150-word canonical object vocabulary — a materially richer and
different artifact than the filename suggests).

**Provenance caveat**: `stats.num_relationships = 251,126` does **not**
match `datasets_vg150_clean/train.jsonl` (1,046,427 relationships). This
prior was **not** built from the dataset currently on disk. `INFERENCE`: it
derives from the original run's own `datasets/train.jsonl`, which no
longer exists on this machine. It **cannot be independently verified as
train-only** — there is no way to check for held-out contamination without
the original raw file. That is a real, unresolvable caveat on any number
the prior contributes to.

---

## 2. Provenance

| Field | Value | Status |
|---|---|---|
| Recovered from | `checkpoints/demo_best/` (curated folder) | — |
| mtime (all three files) | `2026-05-27 08:30` | `VERIFIED` |
| Training commit | **`[5ed4e429, 215fd6d0)`** | `VERIFIED FROM GIT HISTORY` |
| Training commit point estimate | `2b7ba82f` | **`INFERENCE` — do not cite as fact** |
| Embedded run name | `core_l3_balanced_adapt_light` | `VERIFIED FROM CHECKPOINT` |
| Embedded epoch | 5 (local; `reset_epoch=True`) | `VERIFIED FROM CHECKPOINT` |
| Parameters | 79,931,906 (~79.9M), 192 tensors | `VERIFIED FROM CHECKPOINT` |
| Embedded config fields | 174 | `VERIFIED FROM CHECKPOINT` |
| Lineage predecessor | `l3_counterfactual_recovery_l4_best_mR50.pt` | **does not exist on this machine** |
| Reproducible? | **No** | see below |

**Commit bound evidence** (feature presence/absence, not guesswork):
- *Lower*: `5ed4e429` is the commit that **adds**
  `adaptive_calibration_enabled`; the checkpoint's embedded config has it
  `True`.
- *Upper*: `215fd6d0` is the only commit that adds the `"experiment"` key
  to saved checkpoints, present unchanged ever since; this checkpoint has
  **no** `"experiment"` key.

The `2b7ba82f` point estimate rests only on wall-clock proximity (3
minutes before the checkpoint mtime). A training job's code checkout can
predate its checkpoint write by hours or days, so **the bounded range is
the defensible claim**.

**Not reproducible**: the training commit is bounded not pinned; the
original dataset preparation, alias-map version, and `vg150_root='datasets'`
corpus no longer exist. Treat the `.pt` as **irreplaceable historical
evidence**, not a regenerable artifact. This is why it must never be
converted in place.

**Filename mismatch**: the file is `pure_best_adapt_light_mR50.pt` but its
embedded `save_path` says `checkpoints/core_l3_balanced_adapt_light.pt`.
`INFERENCE`: manually copied and renamed. **Content is ground truth, never
the filename.**

---

## 3. Load compatibility

**59 missing keys / 0 unexpected keys** under `strict=False`. All 59 traced
to architecture added *after* this checkpoint's training window and
confirmed inactive under the overrides below — `VERIFIED FROM EXPERIMENT`
(`docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md` Phase F).

Zero unexpected keys is the important half: the checkpoint contains
nothing the current architecture cannot place.

---

## 4. Required compatibility overrides — and exactly why each exists

Every row is **mandatory**. The `--stage 3` column is what the flag
resolves to if you *don't* pass it — verified by executing
`apply_stage_config` at this HEAD, not read off the dataclass.

| Flag | Must be | `--stage 3` gives | Why the stage default is wrong here |
|---|---|---|---|
| `explicit_spoa_enabled` | `false` | `true` | Checkpoint **predates SPOA branches entirely**. The one override without which the load is not faithful. |
| `text_conditioned_projection_enabled` | `false` | `true` | Forced on by `config.py:742`. No trained weights exist. |
| `relationness_enabled` | `false` | `true` | Forced on by `config.py:744`. Head would be at random init. |
| `eval_sgg_use_relationness` | `false` | `true` | Forced on by `config.py:746`. Would **prune candidate pairs using random scores**, corrupting pair proposal. |
| `clip_input_res` | `336` | `448` | Checkpoint's `clip_name` is `…-patch14-336`. 448 is a resolution it never saw. |
| `eval_sgg_predicate_score_mode` | `ensemble` | `ensemble` | Matches `demo_config.env`. Same as default, stated explicitly because it is meaningless without α. |
| `eval_sgg_predicate_ensemble_alpha` | `0.0` | `0.45` | **The single most consequential flag.** α=0 ⇒ `0.0·classifier + 1.0·text` = 100 % CLIP text-cosine. This checkpoint's classifier head is at/near random init, so any α>0 mixes in noise. This one value is the difference between mR@50 0.94 % and 5.84 %. |
| `eval_sgg_use_gt_pairs` | `true` | `false` | `demo_config.env` `PROTOCOL=PredCls_GT_pair`. |
| `adaptive_calibration_enabled` | `true` | `false` | Checkpoint's own config has it `True` with `adaptive_prior_enabled=True`; its trained calibration parameters would go unused. Note stage 3 does **not** set this while stages 1–2 do. |
| `bayes_calibration_weight` | `0.0` | `0.0` | Distinct mechanism from `freq_bias`. Held at 0 so the **only** calibration applied is the recovered prior. |
| `freq_bias_enabled` | `true` | `false` | **Not recorded in `demo_config.env`** — the documented reconstruction gap. `_load_frequency_bias` needs both this and `alpha>0`; the checkpoint's own config says `False`. `INFERENCE`: passed as a bare CLI flag in the original invocation. |
| `freq_bias_alpha` | `3.75` | `0.5` | `demo_config.env` `FREQ_BIAS_ALPHA=3.75`. |
| `eval_sgg_grounding_dino_enabled` | `false` | `true` | PredCls over GT boxes needs no detector. Leaving it on downloads Grounding-DINO and runs an SGDet path carrying its own registered P1 silent fallback. |

### 4.1 Do explicit flags actually win? Yes — verified

`train.py` `main()` order (`VERIFIED FROM CODE`):

```
apply_stage_config(base_cfg)      # stage-3 forces the flags above ON
  → merge explicit CLI args        # your flags land
  → apply_gpu_preset(...)          # may re-clobber batch_size/num_workers
  → _reapply_explicit_cli_args()   # train.py:1326 — your flags win, LAST
```

**But there is no safety net on flag *names*.** `main()` uses
`parse_known_args`, which silently discards unrecognized flags — verified
by execution: `--totally_bogus_flag true` produces no error, no warning,
no output. A typo like `--explicit_spoa_enable` would leave stage-3's
`true` in place and load the checkpoint into the wrong architecture with
**no runtime signal whatsoever**.

Two mitigations, because this cannot be caught at runtime:
1. `scripts/eval/eval_historical_checkpoint.sh` is the only supported
   entrypoint and hardcodes every flag.
2. `tests/test_historical_eval_protocol.py` extracts every `--flag` from
   that script and asserts each exists in `build_argparser()` **and** is a
   real `TrainConfig` field — so a typo fails at commit time, not on a GPU.

---

## 5. The historical claim, stated with its limits

**Source**: `checkpoints/demo_best/demo_config.env` — a 355-byte config
file, self-reported by the run that produced it.

```
BEST_FULL_PREDCLS_R50=0.6709      # 67.09 %
BEST_FULL_PREDCLS_MR50=0.2264     # 22.64 %
```

**Recorded protocol**: `EVAL_SCORE_MODE=ensemble`,
`EVAL_ENSEMBLE_ALPHA=0.0`, `FREQ_BIAS_ALPHA=3.75`,
`PROTOCOL=PredCls_GT_pair`.

**`UNKNOWN` — and possibly permanently so:**
- whether `FREQ_BIAS_ENABLED` was true (inferred, not recorded)
- **which split** it was measured on
- **how many images** it covered
- **pooled vs. per-image-averaged** R@K aggregation — this repo computes
  both (`R@50` and `image_mean_R@50`) and they differ materially
- what `clip_input_res` was used at eval time

**Corroboration: none.** Neither number, nor the run name
`core_l3_balanced_adapt_light`, appears anywhere in `notes/`, `README.md`,
or any script (grep-verified). `notes/current_status.tex` — last edited
four days *before* the checkpoint was written — documents a *lower* best
mR@50 of 18.80. This is an experiment that was run but never written up.

**Consequence for the reproduction**: a gap between the reproduction and
22.64 % has at least five innocent explanations (split, sample size,
aggregation, resolution, `freq_bias_enabled`) before "the checkpoint is
worse than claimed" becomes the leading hypothesis. **A failed
reproduction is not automatically a model defect**, and must not be
reported as one.

---

## 6. Transfer to a fresh machine

`git clone` gives you **code only**. Everything in §1 is gitignored.

Out of band, **1.20 GiB** (measured) excluding images:

```
checkpoints/demo_best/pure_best_adapt_light_mR50.pt    931,057,422
checkpoints/demo_best/frequency_prior.json             101,944,045
checkpoints/demo_best/demo_config.env                          355
datasets_vg150_clean/train.jsonl                       230,887,586
datasets_vg150_clean/validation.jsonl                   28,950,612
datasets_vg150_clean/vocabulary/predicates.json              1,124
datasets_vg150_clean/vocabulary/objects.json                 3,056
                                                     -------------
                                                     1,292,844,200
```

Plus `datasets_vg150_clean/images/` (~14.6 GB).

> **⚠ `images/` is an NTFS junction on the source machine**, pointing at
> `datasets\vg_raw\images` — verified (`LinkType=Junction`). It is **not a
> copy**. Whether an archive follows it depends on the tool and its flags.
> After transfer, verify the image count on the target rather than assuming
> the directory came across; a junction archived as an empty directory
> yields a run where every image silently falls back to a gray placeholder
> (`VG150JSONLDataset._resolve_image`).

Fetched at runtime from HF: `openai/clip-vit-large-patch14-336`.
**Not needed**: `IDEA-Research/grounding-dino-tiny` (detector disabled).

Verify everything landed intact:

```bash
python tools/gcp_preflight.py --strict
```

---

## 7. What must never change

- **`checkpoints/demo_best/*`** — never modify, convert, overwrite, or
  delete. If a compatibility conversion is ever needed, the original must
  be preserved alongside the derived artifact.
- **`datasets_vg150_clean/vocabulary/predicates.json` ordering** — the
  checkpoint's index mapping *and* the frequency prior both depend on it.
  It is canonical now; regenerating it in a different order breaks both.
- **The 13 override values in §4** — they define the experiment.
- **Row 1 of §0's status** — until a reproduction actually runs, 67.09 /
  22.64 stays `HISTORICAL EVIDENCE`. Do not promote it in a paper, a
  README, or a commit message.
