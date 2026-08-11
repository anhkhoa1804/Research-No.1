# Data flow

End-to-end tensor/data flow, both directions. See
`docs/architecture/{training,evaluation}.md` for the surrounding loop logic
this sits inside.

## Training direction

```text
scripts/train/train_l4_phase34.sh (env-var-driven bash wrapper)
  -> python -m openvocab_rel.train --stage 3 --gpu_preset l4_24gb ...
  -> TrainConfig() defaults -> apply_stage_config -> apply_gpu_preset
     -> explicit CLI flags re-applied last (always win)
  -> configure_clip(cfg.clip_name, device) -> CLIP ViT-L/14-336
  -> RelationalModel(cfg, clip_vision_dim, text_dim)
  -> freeze_clip / freeze_predicate_head / freeze_non_relationness applied
  -> DDP-wrap model always; CLIP only if unfrozen
  -> AdamW: model params @ lr, CLIP params @ 0.1x lr (hardcoded)
  -> VG150DataLoader(split="train", source="local-jsonl") -> VG150JSONLDataset
     reads train.jsonl, builds obj boxes/names, builds GT-positive +
     sampled-negative pairs (use_all_pairs, negative_pair_ratio -- see
     docs/known_issues.md for the shipped-script-vs-dataclass-default
     mismatch here), geometry feats, predicate-prompt text variants
  -> encode_predicate_vocab(...) once before the epoch loop;
     PrecomputedVariantCache pre-computes all predicate-prompt-variant CLIP
     text embeddings for O(1) lookup during the epoch
  -> per step: CLIP vision tower on raw pixels -> dense feat_map
     -> RelationalModel.forward_from_featmap(feat_map, obj_boxes_224, pairs,
        force_keep=GT-positive-idx, obj_semantic_feats)
     -> per-object features (regs), per-pair relation features
        (rels + role-swapped rel_swaps), pruning-kept indices
  -> losses computed on the flattened kept-pair set (see training.md for
     the exact summed formula)
  -> scaler.scale(loss).backward() -> grad clip (model params only)
     -> AdamW step -> cosine LR schedule
  -> every eval_every epochs: evals.py.eval_sgg_standard(...) on the
     validation split -> metrics merged -> appended to metrics.jsonl
  -> checkpoint saved every epoch + up to 4 best-tracked snapshots
```

## Evaluation direction (PredCls)

```text
checkpoint (--resume_from) -> TrainConfig merged from ckpt["cfg"] + CLI overrides
  -> eval_sgg_standard(cfg, model, clip_model, processor, val_loader, device, max_batches)
  -> per example: obj_boxes = ex["obj_boxes"] (GT, untouched),
     gt_labels = ex["obj_labels"] (GT)
  -> pair_list = _build_all_ordered_pairs(n_obj)  [default; all N*(N-1) pairs]
     OR _extract_gt_pairs(ex)  [if eval_sgg_use_gt_pairs=True -- easier, non-standard]
  -> _forward_eval_batch -> CLIP vision tower -> out.forward_from_featmap(...)
     (model never sees GT predicate labels or object labels at this stage)
  -> _relation_predicate_logits(...): raw text-cosine logits (always) +
     classifier logits (predicate_logits or calibrated_predicate_logits)
     -> combined per eval_sgg_predicate_score_mode
  -> optional relationness-score pruning and/or eval-time frequency-prior
     fusion (both off by default)
  -> _make_triplet_predictions[_nogc] -> score = pair_score * predicate_prob
     -> _compute_pred_matches (exact label-triple match + IoU>=0.5, IoU
        always 1.0 here since boxes==GT) -> recall accumulation
  -> R@20/50/100 (dataset-global-pooled, the headline field) +
     image_mean_R@K (per-image-averaged, the literature-standard variant) +
     mR@K + head/body/tail bucketed mR@50 + role-swap consistency +
     pair_proposal_diag (prop@K) + pair_rank_diag
```

SGCls and SGDet reuse this exact scoring/matching/metric machinery,
differing only in *what supplies the boxes and labels* (see
`docs/architecture/evaluation.md`'s protocol table).

## What is NOT in this flow

`retrieval.py`'s `TripletRetrievalIndex`/`build_triplet_records` — confirmed
by grep to be unreferenced from `evals.py`, so no retrieval evaluation is
currently wired into either direction above.
