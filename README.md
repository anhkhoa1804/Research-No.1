# PURE: Scene Graph Generation for Relation-Aware Retrieval

PURE (**Predicate-aware Uncropped Relation Embedding**) is a compact Scene Graph Generation (SGG) codebase for learning visual relations on VG150-style data. The thesis framing is **SGG-first**: the primary task is predicate-aware scene graph prediction, while text-based image retrieval is a downstream application built from predicted triplets.

The project currently has two main research components:

1. **PURE model**: a compact ordered-pair predicate model using dense uncropped visual evidence, box-conditioned routing, geometry-aware relation embeddings, and transparent calibrated evaluation.
2. **CORE benchmark**: a compositional relation evaluation benchmark for diagnosing role swaps, object swaps, spatial contradictions, predicate confusability, and other relation-level failures.

> An infrastructure/documentation cleanup pass reorganized `scripts/` into `scripts/{train,eval,notebooks}/`, and added `data/`, `docs/`, and `tests/`. No research logic changed. See `docs/known_issues.md` for issues found but deliberately left unfixed during that pass, and `docs/reproducibility.md` for the current reproducibility status.

## 1. What this repository is

A single research codebase, not a package with external contributors: rapid iteration on one architecture (PURE) plus a currently-inactive diagnostic dataset/benchmark (CORE). See `docs/architecture/overview.md` for what the code actually implements, traced from source rather than described aspirationally.

**Breakthrough branch direction.** The active direction is moving beyond incremental all-pairs runs toward full all-pairs/SGCls/SGDet readiness. Recent all-pairs experiments showed that small pair-rank or triplet-rank losses can increase aggregate R@50 while collapsing mR/tail into head predicates such as `on` and `has` — those experiments are retained as diagnostics, not the final path. The staged plan:

1. **Phase A/B — stabilize and freeze evidence:** keep GT-pair reporting reproducible, keep experimental ranking hooks default-off, summarize all-pairs failures.
2. **Phase C — pair proposal redesign:** train a dedicated pair proposal objective and gate on GT-pair recall@K before optimizing triplet R@50.
3. **Phase D — object bridge:** improve object-label top-k accuracy before SGCls/SGDet headline claims.
4. **Phase E — full graph scoring:** train graph-level triplet scoring only after pair proposal and object bridge gates pass.

*(A dedicated `notes/breakthrough_branch_plan.md` was referenced here previously but does not exist in this checkout — removed rather than left as a dead link; see `notes/INDEX.md`.)*

**Problem definition.** The core task is Scene Graph Generation. Given an image and, in the main reported setting, ground-truth object boxes and object labels:

```text
Input:  image I, boxes B = {b_i}, object labels O = {o_i}
Output: ranked scene graph G(I) = {(subject, predicate, object, score)}
```

The main evaluation setting is currently **PredCls / all-pairs over GT boxes**: boxes and object labels are provided, and the model focuses on predicate ranking. SGCls and SGDet diagnostics exist, but they are not the current headline claims.

## 2. Current status

The maintained thesis direction is:

- Keep the SGG problem as the core task.
- Treat retrieval as an application layer over cached scene graph triplets.
- Report raw classifier capacity separately from calibrated evaluation.
- Use VG150 for main PredCls / GT-pair reporting.
- Use CORE as a diagnostic benchmark and optional robustness-training ablation, not as a replacement for VG150.

Current local headline PredCls results retained for experiment context:

| Setting | R@50 | mR@50 | Notes |
| --- | ---: | ---: | --- |
| Adapt-light + fixed prior `fa45` | **67.20** | 22.38 | Best calibrated R@50 |
| Adapt-light + fixed prior `fa375` | 67.09 | **22.64** | Best calibrated mR@50 |
| Ultra-light + fixed prior `fa35` | 66.92 | 22.58 | Strong calibrated point |
| Ultra-light raw e2 | 35.22 | 20.35 | Best raw training mR@50 |
| Adapt-light raw e5 | 33.71 | 20.03 | Tail improves across epochs |
| Prior-only raw e5 | 33.50 | 19.50 | Prior-only reference |

**Important reporting rule:** calibrated scores are system-level evaluation results. They should not be used as evidence that the raw classifier alone learned all tail predicates. These numbers also come from external runs not reproducible from this checkout alone — see `docs/reproducibility.md`.

## 3. Current supported experiments

**PURE architecture** (maintained, compact):

1. **Dense visual field**: encode the uncropped image once with CLIP patch tokens.
2. **Box-conditioned object routing**: initialize object queries from boxes and sample evidence from the dense visual field.
3. **Ordered relation embedding**: build subject-object pair features with explicit direction and geometry.
4. **Predicate scoring head**: predict relation logits for each ordered pair.
5. **Evaluation-time calibration**: optionally add a fixed frequency-prior term during evaluation.

| Component | Status | Reason |
| --- | --- | --- |
| Dense CLIP tokens | Kept | Stable visual evidence for relations |
| Deformable box routing | Kept | Grounds object features in the uncropped image |
| Geometry-aware pair decoder | Kept | Preserves subject-object direction |
| Fixed frequency / Bayesian calibration | Eval-time only | Report raw and calibrated metrics separately |
| Object-language anchors | Ablation-only | Unstable calibrated mR in local ablations |
| Relation context / bilinear mixing | Ablation-only | Added complexity without reliable mR gain |
| Visual hard negatives | Ablation-only | Useful diagnostic, not headline path |

**Training objective** — three losses:

```text
L_PURE = λ_ce L_PredCE-LA + λ_spoa L_CF-SPOA + λ_g L_Ground
```
- **PredCE-LA**: predicate cross-entropy with logit adjustment for long-tail predicate frequency.
- **Counterfactual-SPOA**: semantic alignment with role/predicate counterfactual negatives.
- **Dense Grounding**: keeps subject, object, and pair features visually anchored to dense image evidence.

Calibration is not counted as a fourth loss because it is applied at evaluation time. Full loss-assembly detail (including 10 additional optional terms, all off by default or ablation-only): `docs/architecture/training.md`.

**Phase 1/2 upgrades:** `--adaptive_calibration_enabled true` trains bounded pair-conditioned prior gates and bias residuals (optionally regularized via `--lambda_calibration_kl`/`--lambda_calibration_rank`). `--explicit_spoa_enabled true` separates subject/predicate/object/attribute branches with role-asymmetric projections. `--predicate_label_relaxation_enabled true` applies CLIP-lattice soft targets for semantically ambiguous predicate negatives instead of hard-masking them. Stage presets in `openvocab_rel/config.py` turn on Phase 1 for Stage 1, Phase 1+2 for Stage 2, and Stage 3 adds text-conditioned predicate scoring, relationness supervision, relationness-pruned SGDet evaluation, and object-uncertainty-aware triplet scoring. YAML files documenting these knobs: `pure_phase1_calibrated_spoa`, `pure_phase2_spoa_relaxed`, `l4_24gb_phase34`, `a100_40gb_phase34` in `configs/presets.yaml` — **note that file is documentation only and is never loaded by any code**; the actual behavior comes from `openvocab_rel/config.py`.

**Five-phase extension hooks:**

| Phase | Status in code | Main entry points |
| --- | --- | --- |
| SGCls bridge | Available | CLIP object-label candidates in `eval_sgg_standard` |
| SGDet bridge | Available | Grounding-DINO proposal path in `eval_sgg_standard` |
| One-stage facade | Available | `RelationalModel.forward_one_stage_facade` for detector proposals |
| Open-vocabulary predicates | Available | `RelationalModel.predicate_scores(..., mode="text")` and `--open_vocab_predicate_primary` |
| Retrieval index | Available, **not wired into evals.py** | `TripletRetrievalIndex` and `build_triplet_records` exist in `retrieval.py` but are never called from the evaluation pipeline today |

These hooks do not change the main claim: the maintained and best-validated setting remains PredCls / all-pairs over GT boxes. SGCls, SGDet, the one-stage facade, open-vocabulary predicate scoring, and the retrieval index are extension paths that can be enabled and reported separately.

**Metrics and reporting policy.** Report every serious checkpoint in separate tiers: (1) raw classifier-only — no frequency prior, no text ensemble; (2) fixed-prior calibrated — checkpoint-specific alpha sweep; (3) adaptive-calibrated — only for checkpoints trained with adaptive calibration enabled; (4) CORE diagnostic — relation-composition stress test, not a protocol-matched VG150 SOTA comparison. Core metrics: **R@K** (aggregate recall of ground-truth triplets in the top-K predictions — note the headline field is a dataset-global-pooled recall, not the per-image-averaged variant most VG150 papers report under the same name; see `docs/known_issues.md`) and **mR@K** (mean recall across predicate classes, exposing long-tail behavior). Do not mix raw and calibrated numbers in the same claim.

**CORE benchmark.** CORE-specific conversion, merge, evaluation, and ablation helpers have been archived from the active workspace. The maintained code path is VG150 training/evaluation through the L4 scripts below. Historical CORE numbers in older notes should be treated as diagnostic context only, not as an active reproducibility path in this checkout.

**SGCls / SGDet diagnostics** are available through the maintained evaluation script; PredCls remains the main reported setting unless a clean SGCls/SGDet rerun is produced:
```bash
CKPT=checkpoints/pure_l4_phase34.pt EVAL_SGDET=true EVAL_BATCHES=100 bash scripts/eval/eval_l4_phase34.sh
```

## 4. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # only if you need diffusion rendering / VLM filtering
pip install -r requirements-dev.txt        # only to run tests/
```
`requirements.txt`/`requirements-optional.txt` pin lower-bound versions anchored to a known-working environment snapshot — see the header comment in each file and `docs/reproducibility.md` for what's and isn't verified.

Do not create new Python virtual environments elsewhere in this workspace.

## 5. External data requirements

Only **VG150** is consumed by any code path in this repo (`openvocab_rel/datasets/vg150_loader.py`). Full requirements, expected layout, and vocabulary sizes: **`data/README.md`** and **`data/manifests/vg150.yaml`**.

Preferred VM workflow: download the maintained Google Drive VG JSONL archive and filter it into a VG150-clean local dataset. The raw archive contains 150-object vocab files but raw predicate strings; this helper maps aliases, filters to the 50 VG150 predicates, drops invalid relationships, writes diagnostics, and symlinks/copies images.

```bash
python3 tools/prepare_vg150_drive_clean.py \
  --out_dir datasets_vg150_clean
```

If the archive was already downloaded/extracted, reuse it without another Drive download:

```bash
python3 tools/prepare_vg150_drive_clean.py \
  --skip_download \
  --skip_extract \
  --jsonl_root datasets/vg_drive_raw \
  --out_dir datasets_vg150_clean
```

Check diagnostics:

```bash
python3 tools/check_vg150_diagnostics.py \
  --diagnostics datasets_vg150_clean/diagnostics.json \
  --min_train_rows 50000 \
  --min_val_rows 5000 \
  --min_predicate_coverage 50 \
  --require_no_validation_issues
```

Alternative HF subset preparation remains available for small smoke datasets:

```bash
python3 tools/prepare_vg150_subset.py \
  --dataset_id anhkhoa1804/VG150-SGG-Standard \
  --out_dir datasets \
  --train_images 5000 \
  --val_images 500 \
  --max_objects 32 \
  --min_relationships 1 \
  --min_predicate_coverage 50
```

Build the frequency prior used for calibrated evaluation:

```bash
python3 tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets_vg150_clean/train.jsonl \
  --out_path datasets_vg150_clean/frequency_prior.json \
  --vg150_root datasets_vg150_clean \
  --smoothing 1.0
```

## 6. Quick validation

On a new VM, verify the local dataset/checkpoint layout before launching any long run:

```bash
python3 tools/validate_dataset.py --dataset vg150 --vg150_root datasets_vg150_clean
```
Pre-flight structural check (fails loudly with a specific message on a missing/incomplete root, rather than surfacing confusingly deep into a training run). Then:

```bash
ls -lh datasets_vg150_clean/train.jsonl datasets_vg150_clean/validation.jsonl
ls -lh datasets_vg150_clean/frequency_prior.json || python3 tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets_vg150_clean/train.jsonl \
  --out_path datasets_vg150_clean/frequency_prior.json \
  --vg150_root datasets_vg150_clean \
  --smoothing 1.0
ls -lh checkpoints/*.pt | tail
```

The maintained L4 scripts auto-select `datasets_vg150_clean` when it contains `train.jsonl` and `validation.jsonl`; otherwise they fall back to `datasets`. Override explicitly with `DATA_ROOT=...`.

For budget-safe validation, prefer smoke settings first:

```bash
MAX_IMAGES=64 EVAL_BATCHES=1 BATCH_SIZE=64 NUM_WORKERS=8
```

Do not run full `EVAL_BATCHES=300/500` until the smoke report confirms the checkpoint, dataset, and score mode are correct. To exercise the code path itself with no external data or GPU access at all, run the smoke-test suite: `pytest tests/`.

## 7. Training

The maintained L4 workflow targets the current Phase 3/4 configuration.

Train with default L4 settings:

```bash
bash scripts/train/train_l4_phase34.sh
```

Resume from a checkpoint:

```bash
RESUME_FROM=checkpoints/previous.pt RUN_NAME=pure_l4_phase34_resume bash scripts/train/train_l4_phase34.sh
```

Run a small pipeline smoke test:

```bash
EPOCHS=1 SAMPLES_PER_EPOCH=1000 EVAL_BATCHES=20 MAX_IMAGES=1000 bash scripts/train/train_l4_phase34.sh
```

Use `scripts/train/run_pure_next.sh` only when you need the lower-level configurable entrypoint. `scripts/notebooks/kaggle-pure-full-train.ipynb` is a Kaggle-specific alternate entrypoint.

**Phase C pair-proposal gate.** The active breakthrough direction is Phase C: measure and improve candidate pair retention before optimizing triplet R@50. Run the dedicated smoke gate:

```bash
bash scripts/eval/eval_phasec_pairgate_smoke.sh
```

This writes `runs/phasec_pairgate_smoke/metrics.jsonl` and immediately summarizes it with:

```bash
python3 tools/sgg_gate_report.py runs/phasec_pairgate_smoke/metrics.jsonl
```

Key Phase C fields are `prop@64`, `prop@96`, `prop@128`, and `pdrop96`. If `prop@96` is low, train or redesign the pair proposal/relationness objective before interpreting all-pairs R@50/mR@50.

For a budget-safe Phase C pilot, train only the relationness/pair-proposal head from the strongest predicate checkpoint:

```bash
bash scripts/train/train_phasec_pair_proposal.sh
bash scripts/eval/eval_phasec_pairgate_smoke.sh CKPT=checkpoints/phasec_pair_proposal_l4_smoke.pt
```

The Phase C pilot freezes CLIP and non-relationness model parameters, disables predicate/triplet/object losses, and optimizes BCE relationness, image-wise hard-negative ranking, and a top-K retention surrogate. Increase `SAMPLES_PER_EPOCH`, `EPOCHS`, and `MAX_IMAGES` only after the smoke gate improves `prop@96`.

First PURE-next modeling pilot:

```bash
bash scripts/train/train_l4_phase34.sh --asymmetric_pair_fusion_enabled true --eval_sgg_role_swap_metric_enabled true --eval_sgg_compare_score_modes classifier,text,ensemble
```

This keeps the protocol PredCls-focused while testing whether asymmetric pair fusion improves role-swap margins and raw/ensemble mR@50.

## 8. Evaluation

Evaluate a checkpoint:

```bash
CKPT=checkpoints/pure_l4_phase34.pt EVAL_BATCHES=500 bash scripts/eval/eval_l4_phase34.sh
```

Create the standard report card after a run:

```bash
python3 tools/model_report_card.py runs/*/metrics.jsonl \
  --train_jsonl datasets/train.jsonl
```

Use this report as the accepted summary format for comparing raw classifier, text-only, ensemble, and calibrated score modes. If the metrics include per-predicate recall, the tool also reports head/body/tail mR@50 and worst predicates. `tools/predicate_delta_report.py` compares per-predicate recall between two metric rows.

See §3 above for the SGCls/SGDet diagnostic command and §7 for the Phase C gate. Full evaluation-pipeline detail: `docs/architecture/evaluation.md`.

## 9. Reproducibility

Full walkthrough: **`docs/reproducibility.md`**. Status, stated plainly: **not yet reproducible end-to-end** — this checkout ships zero committed checkpoints, run logs, or prepared datasets (see §11). What *is* reproducible: the code path itself (`pytest tests/`, no external data or GPU needed), and, given the external VG150 data and CLIP/Grounding-DINO access described in §5, the training/eval commands run exactly as documented.

Every checkpoint and every `metrics.jsonl` row embeds a git commit hash, a config hash, and a predicate-vocabulary hash, so you can tell after the fact exactly what produced a given number — a genuine existing strength of the training loop, not something added by any cleanup.

When reporting numbers, include protocol, score mode, checkpoint, alpha/calibration setting, and evaluation batch count. Prefer concise experiment notes under `notes/` (see `notes/INDEX.md` for what's there and its status).

## 10. Repository structure

```text
openvocab_rel/train.py                    main train/eval loop
openvocab_rel/config.py                   TrainConfig and runtime knobs (field-group index at the top of the file)
openvocab_rel/models/relational_model.py  PURE relation model and decoder
openvocab_rel/datasets/vg150_loader.py    VG150 JSONL/HF loaders and pair construction
openvocab_rel/evals.py                    PredCls/SGCls/SGDet-style evaluation utilities
openvocab_rel/losses.py                   predicate, counterfactual, and grounding losses
openvocab_rel/clip_utils.py               CLIP setup and image/text helpers
openvocab_rel/geometry.py                 geometry utilities
openvocab_rel/prompts.py                  text prompt helpers
openvocab_rel/text_cache.py               text embedding cache utilities
openvocab_rel/retrieval.py                triplet retrieval index utilities (not wired into evals.py)
openvocab_rel/phase_audit.py              five-phase readiness audit
configs/presets.yaml                      GPU/runtime presets -- documentation only, never loaded by code
configs/predicate_metadata_vg150.json     the 50 VG150 predicates' {group, symmetric} table (live, used)
scripts/train/                            training entrypoints (train_l4_phase34.sh, run_pure_next.sh, train_phasec_pair_proposal.sh)
scripts/eval/                             evaluation entrypoints (eval_l4_phase34.sh, eval_calibration_sweep_l4.sh, eval_phasec_pairgate_smoke.sh, report_breakthrough_phase_ab.sh)
scripts/notebooks/                        kaggle-pure-full-train.ipynb (Kaggle-specific alternate entrypoint)
tools/prepare_vg150_drive_clean.py        Drive VG JSONL -> VG150-clean local data
tools/prepare_vg150_subset.py             HF -> local VG150 JSONL/images smoke subsets
tools/check_vg150_diagnostics.py          dataset diagnostics guard (validates prepare_* output)
tools/validate_dataset.py                 dataset pre-flight readiness check (validates input, before a run)
tools/build_vg150_frequency_prior.py      subject-object predicate prior builder (train-split only)
tools/build_vg150_clean_vocab.py          clean VG150 object/predicate vocab builder
tools/model_report_card.py                standardized raw/text/ensemble/calibrated report card
tools/sgg_gate_report.py                  Phase-C pair-proposal gate summarizer
tools/predicate_delta_report.py           per-predicate recall delta between two metric rows
data/README.md, data/manifests/vg150.yaml documentation-only dataset requirements (no data committed)
docs/architecture/                        overview.md, training.md, evaluation.md, data_flow.md
docs/reproducibility.md                   environment/dataset/model setup through experiment identity
docs/known_issues.md                      bugs/drift found during cleanup, deliberately left unfixed
notes/                                    current architecture/status notes -- see notes/INDEX.md
tests/                                    smoke-test suite + tiny synthetic fixtures (no real data/CLIP needed)
```

## 11. What is intentionally excluded from Git

Datasets, pretrained/fine-tuned model weights (CLIP, Grounding-DINO), checkpoints, run outputs, logs, and generated caches/embeddings are all excluded via `.gitignore` (`datasets/`, `runs/`, `checkpoints/`, `*.pt`/`*.pth`/`*.ckpt`, `*.h5`, `*.log`, etc.) — large, regenerable, and would make the repository unusable to clone if committed. Their absence is made explicit instead: `data/README.md` + `data/manifests/vg150.yaml` document what VG150 setup is expected, and `tools/validate_dataset.py`/`tools/check_vg150_diagnostics.py` check for it. Every checkpoint embeds enough metadata (git commit, config hash, predicate-vocab hash) to identify what produced it even though the checkpoint file itself isn't tracked. See `docs/reproducibility.md` §10 for the full artifact-by-artifact policy.

## 12. Known limitations

Full list, with file/function references and severity: **`docs/known_issues.md`**. Headlines:

- This checkout is not independently reproducible end-to-end (§9) — every number above is a claim from an external run.
- The headline `R@K`/`mR@K` fields are dataset-global-pooled, a different statistic from the per-image-averaged recall most VG150 literature reports under the same name (the literature-standard variant is also computed, as `image_mean_R@K`).
- The default pair-fusion path (`asymmetric_pair_fusion_enabled=False`) is symmetric/order-invariant, a partial gap against the "ordered relation embedding" architecture claim.
- `configs/presets.yaml` is documentation only and can drift from the actual hardcoded presets in `openvocab_rel/config.py`.
- `use_rfs`/`rfs_t` are a silent no-op under the maintained `local-jsonl` data loader.

### Paper table snippet

```latex
\begin{table}[h]
\centering
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrl}
\toprule
Setting & R@50 & mR@50 & Notes \\
\midrule
PURE adapt-light fa45 & \textbf{67.20} & 22.38 & best calibrated R@50 \\
PURE adapt-light fa375 & 67.09 & \textbf{22.64} & best calibrated mR@50 \\
PURE ultra-light fa35 & 66.92 & 22.58 & strong calibrated point \\
PURE ultra-light raw e2 & 35.22 & 20.35 & best raw mR@50 \\
\bottomrule
\end{tabular}%
}
\caption{Current local PURE PredCls results. Values are percentages. \textbf{Raw and calibrated rows are not the same measurement and should never be presented as if they were} -- the first row (raw e2) reports the unaided classifier; the other three add a fixed frequency prior at evaluation time. Numbers also reflect this repository's dataset-global-pooled R@K aggregation, not the per-image-averaged variant most VG150 literature reports under the same name -- see docs/known_issues.md before citing these next to external baselines.}
\end{table}
```

### Recommended final reporting order

1. State the thesis as **SGG-first** with retrieval as an application.
2. Define the SGG input/output and PredCls / all-pairs protocol.
3. Explain PURE architecture and the three maintained losses.
4. Report raw classifier metrics before calibrated scores.
5. Present calibrated R@50/mR@50 Pareto points.
6. Use CORE for diagnostic failure analysis and compositional robustness.
7. Keep SGCls/SGDet as diagnostics unless object-vocabulary collapse is fixed.
