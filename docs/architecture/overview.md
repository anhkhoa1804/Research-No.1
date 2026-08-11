# Architecture overview

What this codebase actually implements today, not the aspirational version.
See `docs/known_issues.md` for gaps between this description and
`README.md`'s claims, where any exist.

## What it is

PURE ("Predicate-aware Uncropped Relation Embedding") is a Scene Graph
Generation model for VG150. The maintained, reported protocol is **PredCls**
(ground-truth boxes and labels given; predict the predicate) over all
N·(N−1) ordered candidate pairs per image. SGCls, SGDet, a one-stage
detector facade, open-vocabulary predicate scoring, and a triplet-retrieval
index all exist as code but are extension paths, not the headline claim
(`README.md`'s own framing, confirmed by the code: `retrieval.py`'s
`TripletRetrievalIndex` is never imported from `evals.py`, for instance).

## Forward path (`openvocab_rel/models/relational_model.py`)

```text
CLIP ViT-L/14-336 patch tokens (dense, uncropped, never per-object cropped)
  -> DeformableObjectRouter: box-conditioned query samples 8 learned offset
     points from the dense grid via bilinear grid_sample per object
  -> pair features: Fourier-embedded box geometry (fixed random basis,
     ProgressiveRelationalDecoder.geom_B, non-trainable)
     + semantic pair fusion (sub_norm + obj_norm by default -- see
       docs/known_issues.md, this default path is symmetric/order-invariant;
       asymmetric_pair_fusion_enabled, off by default, fixes this)
     + explicit SPOA branches (subject-role, subject-attribute, predicate,
       object-role, object-attribute -- 5-way factorization, on by default)
  -> ProgressiveEdgeConditionedLayer x2 (default): gated cross-attention
     re-queries the dense visual field per pair, conditioned on the
     current relation hypothesis
  -> [off by default: DynamicBilinearLayer, relation-context Transformer]
  -> heads: predicate_classifier (51-way: 50 VG150 predicates + a
     synthetic "relation" background class), relationness_head (binary,
     pair-proposal signal), calibration_gate + bias_residual_head
     (adaptive, trained calibration), text_space_projection (CLIP-text
     ensemble scoring path)
```

## The three training-loss pillars

```text
L_PURE = lambda_ce * L_PredCE-LA + lambda_spoa * L_Counterfactual-SPOA + lambda_ground * L_Dense-Grounding
```
implemented across `openvocab_rel/train.py` (loss assembly, see
`docs/architecture/training.md`) and `openvocab_rel/losses.py` (InfoNCE
variants, queue-based negative bank, counterfactual hard-negative
composition). Ten additional optional loss terms exist (calibration
KL/rank, text-CE, relationness BCE+rank, pair-topk-surrogate,
pair-balanced-topk, object-bridge, triplet-rank, role-swap-rank), each
gated by its own `lambda_*` (default 0 for most) — see the field-group
index at the top of `openvocab_rel/config.py`.

## Extension surface, current status

| Component | Default | Notes |
|---|---|---|
| Bilinear mixing | off | README/notes: "added complexity without reliable mR gain" |
| Relation-context Transformer | off | same |
| Object-language anchor | off | README: "unstable calibrated mR in local ablations" |
| Adaptive calibration | on (stage 2/3 presets) | trainable, pair-conditioned; reported separately from raw classifier metrics per `README.md`'s reporting policy |
| Relationness head / Phase-C pair-proposal losses | on (stage 3 only) | most recently active research direction per git history |
| One-stage facade, retrieval index, open-vocab-primary scoring | off | extension hooks, see `openvocab_rel/phase_audit.py` |

## Curriculum

`--stage {1,2,3}` applies a preset curriculum (`openvocab_rel/config.py`,
`apply_stage_config`): stage 1 freezes CLIP and disables most auxiliary
losses/calibration; stage 2 progressively unfreezes CLIP and enables
label/attribute relaxation; stage 3 enables text-conditioned predicate
scoring, relationness supervision, and adaptive calibration together. GPU
presets (`--gpu_preset`) separately control batch size/gradient
checkpointing per hardware target — see `configs/presets.yaml`'s header for
why that file is documentation, not the actual mechanism.

## Where to read more

- `docs/architecture/training.md` — training loop, loss assembly, optimizer/DDP
- `docs/architecture/evaluation.md` — PredCls/SGCls/SGDet, calibration, metrics
- `docs/architecture/data_flow.md` — end-to-end tensor flow, both directions
