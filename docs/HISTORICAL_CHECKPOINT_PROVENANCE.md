# Historical checkpoint provenance: `checkpoints/demo_best/`

Not committed as of this writing (investigation output; commit only if
explicitly requested, per the same policy as
`docs/GT_EXTRACTION_BUG_TRIAGE.md`). The checkpoint binaries themselves are
gitignored (`/checkpoints/`, verified via `git check-ignore -v`) and must
never be committed.

## What was recovered

```
checkpoints/demo_best/
├── demo_config.env              355 bytes
├── frequency_prior.json         101,944,045 bytes (~97 MiB)
└── pure_best_adapt_light_mR50.pt 931,057,422 bytes (~888 MiB)
```

All three files share a filesystem mtime of `2026-05-27 08:30`.

SHA256:
```
pure_best_adapt_light_mR50.pt  8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442
frequency_prior.json           144d9f928ed9cc213acbe081b1a8791488a92ddab0ed885da7fd6bb6058c6e6a
demo_config.env                c73180698b5b5f6d3a58e5e6b78a39fcb8a81b6250478145d0538a5a107226a6
```
Re-verified identical after all inspection work in this phase — the
checkpoint was never overwritten, converted, or modified in place.

**Correction to initial assumptions**: the checkpoint is ~888 MiB, not the
~909 KB initially estimated (roughly a 1000x discrepancy) — consistent with
a real 79.9M-parameter model plus full AdamW optimizer state, not a
lightweight artifact. `frequency_prior.json` at ~97 MiB is a full
pair/subject/object-conditioned frequency table (74,884 unique pair
entries), not a small ~50-entry marginal table — a materially different
and richer artifact than what its filename alone suggests.

## `demo_config.env` — full field classification

| Field | Value | Classification |
|---|---|---|
| `PURE_BEST_CKPT` | `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` | PATH — safe |
| `FREQ_BIAS_PATH` | `checkpoints/demo_best/frequency_prior.json` | PATH — safe |
| `FREQ_BIAS_ALPHA` | `3.75` | HYPERPARAMETER — safe |
| `EVAL_SCORE_MODE` | `ensemble` | EXPERIMENT CONFIG — safe |
| `EVAL_ENSEMBLE_ALPHA` | `0.0` | HYPERPARAMETER — safe |
| `PROTOCOL` | `PredCls_GT_pair` | EXPERIMENT CONFIG — safe (human-readable label only; not read by any script — grep-verified) |
| `BEST_FULL_PREDCLS_R50` | `0.6709` | Historical result — safe |
| `BEST_FULL_PREDCLS_MR50` | `0.2264` | Historical result — safe |
| `MODEL_NOTE` | free text | EXPERIMENT CONFIG — safe |

**No secrets, credentials, tokens, or sensitive paths found.** Nothing in
this file requires redaction.

**Reconstruction gap**: the file does not set `FREQ_BIAS_ENABLED`.
`_load_frequency_bias` (`openvocab_rel/evals.py:1008-1013`) requires
`freq_bias_enabled=True` *and* `freq_bias_alpha>0` for the frequency bias
to have any effect; the checkpoint's own embedded config has
`freq_bias_enabled=False`. Reproducing `BEST_FULL_PREDCLS_MR50=0.2264`
therefore requires an explicit `--freq_bias_enabled true` override not
recorded anywhere in this file (`INFERENCE`: most likely passed as a bare
CLI flag in whatever invocation produced these numbers, not captured by
this env file).

## `frequency_prior.json` — inventory

```json
{
  "predicate_vocab": [... 50 entries, matches the canonical VG150 set exactly ...],
  "object_vocab": [... 25,918 entries — raw, pre-canonicalization label strings, not the 150-word canonical object vocabulary ...],
  "smoothing": 1.0,
  "default_log_prob": -3.912023005428146,
  "global_log_probs": [... 50 floats ...],
  "pair_log_probs": {... 74,884 keys ...},
  "subject_log_probs": {... 12,030 keys ...},
  "object_log_probs": {... 9,689 keys ...},
  "stats": {
    "vocab_source": "fixed_or_scanned",
    "num_predicates": 50,
    "num_object_labels": 25918,
    "num_pairs": 74884,
    "num_relationships": 251126
  }
}
```

- Schema matches `_load_frequency_bias`'s expected format field-for-field
  (`VERIFIED FROM CODE`).
- `tools/build_vg150_frequency_prior.py --train_jsonl <path>` is the only
  generator in this repo; `--train_jsonl` is a required argument with no
  val/test alternative in the tool's own CLI surface — structurally, this
  file can only have been built by pointing the tool at a file its caller
  designated as "train" (`VERIFIED FROM CODE`, tool design).
- `num_relationships=251,126` does **not** match this session's freshly
  prepared `datasets_vg150_clean/train.jsonl` (1,046,427 relationships) —
  this file was **not** built from the canonical dataset prepared this
  session. It almost certainly derives from the original GCP run's own
  `datasets/train.jsonl`, which no longer exists on this machine
  (`INFERENCE`).
- **Cannot be independently verified as train-only** in the strict sense
  (no way to check for held-out contamination without the original raw
  file) — `VERIFIED FROM DATA` (schema/vocab/tool-design evidence) +
  `INFERENCE` (train-derived), explicitly **not** `VERIFIED FROM
  EXPERIMENT`.
- **This is a different mechanism from `adaptive_calibration_enabled`** /
  `_predicate_log_prior_for_eval` (the mechanism fixed for the eval-split
  leak in the earlier validity-fix phase, `MissingTrainStatisticsError`).
  That mechanism reads `train.jsonl` directly at eval time and is entirely
  separate from `freq_bias_path`/`frequency_prior.json`. Both exist in
  this codebase; do not conflate them. The checkpoint's embedded config has
  `adaptive_calibration_enabled=True` (used during training-time eval,
  against the original run's own `vg150_root='datasets'`) and
  `freq_bias_enabled=False` (this second mechanism was off during
  training-time eval, only enabled for the separately-reported "best" eval
  per `demo_config.env`).

## Checkpoint inventory

Top-level keys: `epoch`, `model`, `optim`, `scaler`, `cfg` — **no
`experiment` key** (see provenance bound below).

- `epoch`: `5`
- `model`: `OrderedDict`, 192 tensors, 79,931,906 total scalar parameters (~79.9M)
- `optim`: AdamW state (`state` + `param_groups`, 179 state entries, 2 param groups — lr=1e-7/1e-8, `decoupled_weight_decay=True`)
- `scaler`: AMP `GradScaler` state
- `cfg`: a full 174-field `TrainConfig.__dict__` snapshot

No git commit hash, config hash, or predicate-vocab hash is embedded
anywhere in this checkpoint (unlike current `train.py`, which embeds all
three via `_experiment_snapshot`) — see provenance section below for why.

### Selected embedded config fields (`VERIFIED FROM CHECKPOINT`)

```
model_name=pure          pure_phase=core           stage=3
run_name=core_l3_balanced_adapt_light
out_dir=runs/core_l3_balanced_adapt_light
save_path=checkpoints/core_l3_balanced_adapt_light.pt
resume=True               resume_from=checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt
reset_epoch=True          seed=0                    epochs=6
vg150_root=datasets       vg150_source=local-jsonl
hf_dataset_id=anhkhoa1804/VG150-SGG-Standard
max_images=20000          samples_per_epoch=20000
freeze_clip=True          clip_name=openai/clip-vit-large-patch14-336   emb_dim=768
eval_sgg_use_gt_pairs=True                 (matches demo_config.env PROTOCOL=PredCls_GT_pair)
eval_sgg_predicate_score_mode=classifier   (differs from demo_config.env EVAL_SCORE_MODE=ensemble --
                                             this is the TRAINING run's own periodic eval setting, NOT
                                             necessarily what produced the "best" reported numbers)
adaptive_calibration_enabled=True   adaptive_prior_enabled=True   adaptive_prior_scale=0.5
freq_bias_enabled=False    freq_bias_alpha=1.0   (both overridden by demo_config.env for the "best" eval)
predicate_classifier_classes=51   predicate_classifier_enabled=True
use_all_pairs=True   negative_pair_ratio=2.0   use_rfs=True   use_geom_bias=True
relationness_enabled=False   (not present as a field at all -- see compatibility audit)
```

**Filename vs. embedded `save_path` mismatch**: the file is named
`pure_best_adapt_light_mR50.pt`, but the embedded config's own
`save_path` field says `checkpoints/core_l3_balanced_adapt_light.pt`.
Current `train.py`'s best-checkpoint naming convention would produce
`core_l3_balanced_adapt_light_best_mR50.pt` for a best-mR@50 save — not
`pure_best_adapt_light_mR50.pt`. `INFERENCE`: the file most likely was
manually copied/renamed into the `demo_best/` directory (itself an
evidently manually-curated folder name) for convenience, rather than
reflecting the original training run's own save-path naming exactly. This
does not affect the checkpoint's validity — the *content* (the `cfg` dict,
`model` weights) is the ground truth, not the filename, exactly as
warned against in this phase's brief.

### `resume_from` lineage

`resume_from=checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt` — this
run was **not** trained from scratch; it continued from an earlier
checkpoint of that name, combined with `reset_epoch=True` (so `epoch=5` is
this run's own local epoch count, not cumulative from the lineage
predecessor). That predecessor checkpoint does **not** exist anywhere on
this machine (confirmed via filesystem search) — only the final
`adapt_light` checkpoint was recovered, not its training lineage.

## Bounding the training commit (`VERIFIED FROM GIT HISTORY`)

Two hard constraints, both derived from feature-presence/absence rather
than guesswork:

1. **Lower bound**: `notes/current_status.tex` was last touched at commit
   `5ed4e429` (`2026-05-23`), whose own commit message is *"Add adaptive
   calibration and metrics logging: implement optional adaptive
   calibration path..."* — i.e. this is the commit that **adds**
   `adaptive_calibration_enabled` to `train.py`. The checkpoint's embedded
   config has `adaptive_calibration_enabled=True`, so the training commit
   must be **at or after `5ed4e429`**.
2. **Upper bound**: `git log --oneline --all -S'"experiment": _experiment_snapshot'
   -- openvocab_rel/train.py` returns exactly one commit, `215fd6d0`
   (`2026-06-13 21:12`) — the commit that adds the `"experiment"` key to
   every saved checkpoint dict, a line that has been present, unchanged,
   in every commit since. The recovered checkpoint has no `"experiment"`
   key, so the training commit must be **strictly before `215fd6d0`**.

Combined with the file mtime (`2026-05-27 08:30`), the closest commit by
wall-clock time is `2b7ba82f` (`2026-05-27 08:27`, *"Enhance CORE
fine-tuning and ablation process..."*), 3 minutes before the checkpoint
was written; `b7c9470f` follows 26 minutes later. **`INFERENCE`, not
proof**: a GCP training job's actual code checkout can predate its last
checkpoint write by hours or days, so any commit in the
`5ed4e429..2b7ba82f` window remains a possible true training commit — the
rigorous, defensible claim is the bounded range
`[5ed4e429, 215fd6d0)`, with `2b7ba82f` as the single most plausible point
estimate.

## Undocumented result

`notes/current_status.tex` (as of its last edit, `5ed4e429`, `2026-05-23`)
documents "the strongest current PURE number for mean recall" as **PredCls
mR@50 = 18.80** ("PURE L3 final"), with a separate sweep row at R@50=63.41/
mR@50=17.20. `demo_config.env`'s self-reported
`BEST_FULL_PREDCLS_MR50=0.2264` (22.64%) and `BEST_FULL_PREDCLS_R50=0.6709`
(67.09%) are **both higher** than anything documented in
`current_status.tex`, and neither number, nor the run name
`core_l3_balanced_adapt_light`/`adapt_light`, appears anywhere in
`notes/`, `README.md`, or any script in this repository
(`grep`-verified). Since the checkpoint postdates `current_status.tex`'s
last edit by 4 days, this is simply an experiment that was run but never
written up — not a discrepancy requiring explanation, but also not yet
independently verified. See the Experiment A/B/C results below for the
current-code, current-data verification of these self-reported numbers.
