# Training pipeline

Traced from `openvocab_rel/train.py`. Line numbers are approximate to the
version this doc was written against — if they drift, trust the code and
treat this as a map, not ground truth.

## Entry point

```bash
python -m openvocab_rel.train --stage 3 --gpu_preset l4_24gb ...
```
`build_argparser()` exposes one CLI flag per `TrainConfig` field (all 255,
verified). `main()` builds `TrainConfig()`, applies `apply_stage_config`
(stage presets) then `apply_gpu_preset`/`apply_ddp_a100_preset` (hardware
presets), then re-applies any *explicitly passed* CLI flags last so they
always win over both preset layers.

## Setup

- `seed_all(cfg.seed)` seeds `random`/`numpy`/`torch` CPU+CUDA RNGs.
  `cudnn.deterministic` is never set and `cudnn.benchmark=True` is the
  default — reproducibility is not bit-exact by design (throughput is
  prioritized).
- CLIP loaded via `configure_clip`; `cfg.emb_dim` can be silently
  overwritten by CLIP's actual projection dimension.
- Freezing: `freeze_clip`, `freeze_predicate_head`, `freeze_non_relationness`
  each gate `requires_grad_(False)` on different submodule sets — used
  together by the Phase-C pilot script to train only the relationness head.
- DDP: model always wrapped when `WORLD_SIZE>1`; CLIP wrapped only if
  unfrozen. No `find_unused_parameters=True` anywhere.
- Optimizer: single `AdamW`, two param groups — model params at `cfg.lr`,
  CLIP params at a **hardcoded 0.1x cfg.lr** with zero weight decay.
- Grad clipping: `clip_grad_norm_` applied to `model.parameters()` only —
  CLIP gradients are never clipped once unfrozen (see
  `docs/known_issues.md`).

## Dataset / dataloader

`VG150DataLoader` (`openvocab_rel/datasets/vg150_loader.py`) — see
`docs/architecture/data_flow.md` for the full trace. Only VG150 mode is
supported; the multi-task path was removed (`main()` raises
`NotImplementedError` otherwise).

## Predicate-text caching

`PrecomputedVariantCache` (`openvocab_rel/text_cache.py`) pre-computes CLIP
text embeddings for every predicate-prompt variant once per epoch (both
subject->object and object->subject directions), so the training loop does
O(1) dict lookups instead of a CLIP text forward pass per batch.

## Forward pass

CLIP vision tower (outside autocast is not the case — it's inside
`torch.amp.autocast`) -> dense feature map -> `RelationalModel.
forward_from_featmap(feat_map, obj_boxes_224, pairs, force_keep=<GT-positive
indices>, obj_semantic_feats)`. `force_keep` guarantees GT-positive pairs
survive the model's internal geometric top-K pruning
(`_compute_kept_indices` — a nearest-neighbor-by-center-distance heuristic,
independent of the relationness head).

## Loss assembly

Exact summed formula (all terms read via `float(getattr(cfg, "lambda_X",
0.0))` immediately before multiplication, so an unset/zero lambda
contributes exactly zero regardless of whether its underlying tensor was
even computed):

```text
L = 1[spoa]*lambda_spoa*L_SPOA + 1[ground]*lambda_ground*L_Ground + 1[predce]*lambda_predce*L_PredCE
  + lambda_calkl*L_CalKL + lambda_calrank*L_CalRank + lambda_textce*L_TextCE
  + lambda_relness*L_Relness + lambda_relnessrank*L_RelnessRank
  + lambda_pairtopk*L_PairTopK + lambda_pairbaltopk*L_PairBalTopK
  + lambda_objbridge*L_ObjBridge + lambda_tripletrank*L_TripletRank + lambda_roleswap*L_RoleSwap
  + [lambda_calreg * calibration_regularizer()  if adaptive_calibration_enabled]
  + [lambda_gate  * gate_reg                    if the decoder produced one]
, then loss = L / accum_steps
```

`L_SPOA` itself already nests the counterfactual-hard-negative
(`lambda_counterfactual`) and visual-hard-negative (`lambda_visual_hard_negative`)
terms *inside* it before the outer `lambda_spoa` multiplies — those two are
not top-level additive terms despite the README's simplified 3-pillar
equation. `train_objective` (default `"full"`) gates which of
spoa/ground/pred_ce are nonzero at all (`predicate_warmup`, `ce_only`,
`no_spoa`, `ground_ce`, and the `object_bridge`-family objectives each zero
out a different subset).

## LR schedule

Linear warmup (`warmup_steps` or `warmup_epochs`-derived) then optional
cosine decay to a `1e-7` floor. Both param groups pinned to the floor at
step 0.

## AMP / precision

bf16 preferred, falls back to fp16 if the GPU doesn't support bf16.
`GradScaler` used unconditionally when `cfg.amp=True` regardless of dtype.
FP8 (`fp8_autocast`) wraps only the model forward call and is force-disabled
by every shipped GPU preset (A100/L4 only — FP8 is Hopper-only).

## Checkpointing

Every epoch (rank 0 only): full state dict + optimizer + scaler + full
config dict + an experiment snapshot (git commit hash, config hash,
predicate-vocab hash) saved to `cfg.save_path`, overwriting the "latest".
Up to 4 additional named-best snapshots (`_best_R50`, `_best_mR50`,
`_best_tail_mR50`, `_best_selection`) are independently tracked by running
maxima and re-saved whenever that epoch's eval improves the corresponding
metric — this only updates on `eval_every`-triggered epochs.

## Logging

Per-step console log (every `log_every` steps): total loss + every
individual weighted loss component, gate value, current LR, optional timing
breakdown. Per-epoch: `metrics.jsonl` (appended), `epoch_metrics/epoch_NNN.json`
(full snapshot), `latest_metrics.json` (overwritten). See
`docs/reproducibility.md` §8 for the on-disk layout.
