from __future__ import annotations
import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .clip_utils import clip_text_features
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:
    if mp.get_sharing_strategy() != "file_system":
        mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass
from .config import (
    TrainConfig,
    apply_ddp_a100_preset,
    apply_gpu_preset,
    apply_stage_config,
    resolve_loader_num_workers,
    resolve_loader_pin_memory,
    warn_if_low_shm,
)
from .prompts import (
    PredicateCanonicalizer,
    build_predicate_prompt_candidates,
    pred_prompt,
    pred_prompt_roles,
)
def _timing_sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    sv = str(v).strip().lower()
    if sv in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if sv in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _iter_predicates_from_dataset_for_weights(dataset: Any, max_rows: int = 50000) -> Counter:
    counts: Counter = Counter()
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list):
        for row in rows[: max(0, int(max_rows))]:
            rels = row.get("relationships", row.get("relations", row.get("rels", []))) if isinstance(row, dict) else []
            for rel in rels if isinstance(rels, list) else []:
                if isinstance(rel, dict):
                    pred = str(rel.get("predicate", rel.get("pred", ""))).strip().lower()
                    if pred:
                        counts[pred] += 1
        return counts
    scene_graphs = getattr(dataset, "scene_graphs", None)
    images = getattr(dataset, "images", None)
    valid_indices = getattr(dataset, "valid_indices", None)
    if isinstance(scene_graphs, dict) and isinstance(images, list) and isinstance(valid_indices, list):
        for img_idx in valid_indices[: max(0, int(max_rows))]:
            if int(img_idx) >= len(images):
                continue
            img_id = images[int(img_idx)].get("image_id")
            sg = scene_graphs.get(img_id, {})
            for rel in sg.get("relationships", []) if isinstance(sg, dict) else []:
                pred = str(rel.get("predicate", "")).strip().lower()
                if pred:
                    counts[pred] += 1
        return counts
    inner_dataset = getattr(dataset, "dataset", None)
    if inner_dataset is not None:
        try:
            limit = min(len(inner_dataset), max(0, int(max_rows)))
        except Exception:
            limit = 0
        for idx in range(limit):
            try:
                row = inner_dataset[int(idx)]
            except Exception:
                continue
            rels = row.get("relationships", row.get("relations", row.get("rels", []))) if isinstance(row, dict) else []
            for rel in rels if isinstance(rels, list) else []:
                if isinstance(rel, dict):
                    pred = str(rel.get("predicate", rel.get("pred", ""))).strip().lower()
                    if pred:
                        counts[pred] += 1
    return counts


def _build_predicate_ce_weights(cfg: TrainConfig, pred_vocab: List[str], pred_freq: Dict[str, int], device: torch.device) -> torch.Tensor:
    counts = torch.tensor([max(1.0, float(pred_freq.get(str(p), 1))) for p in pred_vocab], dtype=torch.float32, device=device)
    weights = counts.mean().clamp(min=1.0) / counts
    weights = torch.pow(weights, float(getattr(cfg, "predicate_ce_weight_power", 0.5)))
    weights = weights / weights.mean().clamp(min=1e-6)
    weights = weights.clamp(max=float(getattr(cfg, "predicate_ce_max_weight", 5.0)))
    for bg_name in ("relation", "no relation", "no interaction", "background", "__background__"):
        if bg_name in pred_vocab:
            bg_idx = int(pred_vocab.index(bg_name))
            weights[bg_idx] = float(getattr(cfg, "predicate_background_weight", 0.1))
    weights = weights / weights.mean().clamp(min=1e-6)
    return weights.detach()


def _predicate_ce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    mode = str(getattr(cfg, "predicate_ce_loss", "weighted_ce")).strip().lower()
    if logits.numel() == 0 or targets.numel() == 0:
        return logits.sum() * 0.0
    use_weights = weights if mode in {"weighted_ce", "focal"} and int(weights.numel()) == int(logits.shape[-1]) else None
    ce = F.cross_entropy(logits.float(), targets.long(), weight=use_weights, reduction="none")
    if mode == "focal":
        pt = torch.exp(-ce).clamp(0.0, 1.0)
        ce = torch.pow(1.0 - pt, float(getattr(cfg, "predicate_ce_gamma", 1.5))) * ce
    return ce.mean()

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("openvocab-rel SPOA one-stage (3-stage training: --stage 1|2|3)")
    p.add_argument("--stage", type=int, default=TrainConfig.stage)
    p.add_argument("--a100_ddp_preset", type=str, default=TrainConfig.a100_ddp_preset)
    p.add_argument("--gpu_preset", type=str, default=TrainConfig.gpu_preset)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--run_name", type=str, default=TrainConfig.run_name)
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir)
    p.add_argument("--save_metrics_json", type=str, default=TrainConfig.save_metrics_json)
    p.add_argument("--hf_dataset_id", type=str, default=TrainConfig.hf_dataset_id)
    p.add_argument("--hf_train_split", type=str, default=TrainConfig.hf_train_split)
    p.add_argument("--hf_val_split", type=str, default=TrainConfig.hf_val_split)
    p.add_argument("--hf_streaming", type=_str2bool, nargs="?", const=True, default=TrainConfig.hf_streaming)
    p.add_argument("--hf_shuffle_buffer_size", type=int, default=TrainConfig.hf_shuffle_buffer_size)
    p.add_argument("--samples_per_epoch", type=int, default=TrainConfig.samples_per_epoch)
    p.add_argument("--max_images", type=int, default=TrainConfig.max_images)
    p.add_argument("--max_objects", type=int, default=TrainConfig.max_objects)
    p.add_argument("--max_pairs", type=int, default=TrainConfig.max_pairs)
    p.add_argument("--use_all_pairs", type=_str2bool, nargs="?", const=True, default=TrainConfig.use_all_pairs)
    p.add_argument("--negative_pair_ratio", type=float, default=TrainConfig.negative_pair_ratio)
    p.add_argument("--box_noise", type=float, default=TrainConfig.box_noise)
    p.add_argument("--learned_prune_k", type=int, default=TrainConfig.learned_prune_k)
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--loader_prefetch_factor", type=int, default=TrainConfig.loader_prefetch_factor)
    p.add_argument("--loader_persistent_workers", type=_str2bool, nargs="?", const=True, default=TrainConfig.loader_persistent_workers)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--accum_steps", type=int, default=TrainConfig.accum_steps)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--lr_schedule", type=str, default=TrainConfig.lr_schedule)
    p.add_argument("--train_objective", type=str, default=getattr(TrainConfig, "train_objective", "full"))
    p.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    p.add_argument("--warmup_epochs", type=int, default=TrainConfig.warmup_epochs)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--amp", type=_str2bool, nargs="?", const=True, default=TrainConfig.amp)
    p.add_argument("--amp_dtype", type=str, default=TrainConfig.amp_dtype)
    p.add_argument("--grad_clip_norm", type=float, default=TrainConfig.grad_clip_norm)
    p.add_argument("--progressive_unfreeze", type=_str2bool, nargs="?", const=True, default=TrainConfig.progressive_unfreeze)
    p.add_argument("--clip_unfreeze_after_epochs", type=int, default=TrainConfig.clip_unfreeze_after_epochs)
    p.add_argument("--channels_last", type=_str2bool, nargs="?", const=True, default=TrainConfig.channels_last)
    p.add_argument("--cudnn_benchmark", type=_str2bool, nargs="?", const=True, default=TrainConfig.cudnn_benchmark)
    p.add_argument("--torch_compile", type=_str2bool, nargs="?", const=True, default=TrainConfig.torch_compile)
    p.add_argument("--torch_compile_mode", type=str, default=TrainConfig.torch_compile_mode)
    p.add_argument("--lambda_spoa_alignment", type=float, default=TrainConfig.lambda_spoa_alignment)
    p.add_argument("--lambda_dense_grounding", type=float, default=TrainConfig.lambda_dense_grounding)
    p.add_argument("--lambda_predicate_ce", type=float, default=TrainConfig.lambda_predicate_ce)
    p.add_argument("--lambda_counterfactual", type=float, default=TrainConfig.lambda_counterfactual)
    p.add_argument("--lambda_visual_hard_negative", type=float, default=TrainConfig.lambda_visual_hard_negative)
    p.add_argument("--visual_hard_negative_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.visual_hard_negative_enabled)
    p.add_argument("--predicate_counterfactual_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.predicate_counterfactual_enabled)
    p.add_argument("--gate_regularizer_weight", type=float, default=TrainConfig.gate_regularizer_weight)
    p.add_argument("--fusion_gate_temperature", type=float, default=getattr(TrainConfig, "fusion_gate_temperature", 0.7))
    p.add_argument("--predicate_classifier_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.predicate_classifier_enabled)
    p.add_argument("--predicate_classifier_classes", type=int, default=TrainConfig.predicate_classifier_classes)
    p.add_argument("--eval_sgg_use_predicate_classifier", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_use_predicate_classifier)
    p.add_argument("--eval_sgg_predicate_score_mode", type=str, default=TrainConfig.eval_sgg_predicate_score_mode)
    p.add_argument("--eval_sgg_predicate_ensemble_alpha", type=float, default=TrainConfig.eval_sgg_predicate_ensemble_alpha)
    p.add_argument("--eval_sgg_classifier_temperature", type=float, default=getattr(TrainConfig, "eval_sgg_classifier_temperature", 1.0))
    p.add_argument("--eval_sgg_text_temperature", type=float, default=getattr(TrainConfig, "eval_sgg_text_temperature", 1.0))
    p.add_argument("--eval_sgg_predicate_diag_topk", type=int, default=TrainConfig.eval_sgg_predicate_diag_topk)
    p.add_argument("--eval_sgg_compare_score_modes", type=str, default=getattr(TrainConfig, "eval_sgg_compare_score_modes", ""))
    p.add_argument("--freq_bias_enabled", type=_str2bool, nargs="?", const=True, default=getattr(TrainConfig, "freq_bias_enabled", False))
    p.add_argument("--freq_bias_path", type=str, default=getattr(TrainConfig, "freq_bias_path", ""))
    p.add_argument("--freq_bias_alpha", type=float, default=getattr(TrainConfig, "freq_bias_alpha", 0.5))
    p.add_argument("--freq_bias_smoothing", type=float, default=getattr(TrainConfig, "freq_bias_smoothing", 1.0))
    p.add_argument("--predicate_background_weight", type=float, default=TrainConfig.predicate_background_weight)
    p.add_argument("--predicate_ce_loss", type=str, default=TrainConfig.predicate_ce_loss)
    p.add_argument("--predicate_ce_gamma", type=float, default=TrainConfig.predicate_ce_gamma)
    p.add_argument("--predicate_ce_weight_power", type=float, default=TrainConfig.predicate_ce_weight_power)
    p.add_argument("--predicate_ce_max_weight", type=float, default=TrainConfig.predicate_ce_max_weight)
    p.add_argument("--predicate_ce_positive_only", type=_str2bool, nargs="?", const=True, default=TrainConfig.predicate_ce_positive_only)
    p.add_argument("--predicate_sampler_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.predicate_sampler_enabled)
    p.add_argument("--predicate_sampler_power", type=float, default=TrainConfig.predicate_sampler_power)
    p.add_argument("--predicate_sampler_max_weight", type=float, default=TrainConfig.predicate_sampler_max_weight)
    p.add_argument("--temp", type=float, default=TrainConfig.temp)
    p.add_argument("--ground_num_queries", type=int, default=TrainConfig.ground_num_queries)
    p.add_argument("--rel_queue_size", type=int, default=TrainConfig.rel_queue_size)
    p.add_argument("--rel_queue_min_negatives", type=int, default=TrainConfig.rel_queue_min_negatives)
    p.add_argument("--lattice_loss_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.lattice_loss_enabled)
    p.add_argument("--lattice_min_neg_weight", type=float, default=TrainConfig.lattice_min_neg_weight)
    p.add_argument("--use_rfs", type=_str2bool, nargs="?", const=True, default=TrainConfig.use_rfs)
    p.add_argument("--rfs_t", type=float, default=TrainConfig.rfs_t)
    p.add_argument("--use_geom_bias", type=_str2bool, nargs="?", const=True, default=TrainConfig.use_geom_bias)
    p.add_argument("--vector_fusion_gate", type=_str2bool, nargs="?", const=True, default=TrainConfig.vector_fusion_gate)
    p.add_argument("--geom_fourier_dim", type=int, default=TrainConfig.geom_fourier_dim)
    p.add_argument("--logit_adj_tau", type=float, default=TrainConfig.logit_adj_tau)
    p.add_argument("--eval_logit_adj_tau", type=float, default=TrainConfig.eval_logit_adj_tau)
    p.add_argument("--pure_phase", type=str, default=getattr(TrainConfig, "pure_phase", "core"))
    p.add_argument("--object_language_anchor_enabled", type=_str2bool, nargs="?", const=True, default=getattr(TrainConfig, "object_language_anchor_enabled", False))
    p.add_argument("--object_language_anchor_source", type=str, default=getattr(TrainConfig, "object_language_anchor_source", "auto"))
    p.add_argument("--object_language_anchor_alpha", type=float, default=getattr(TrainConfig, "object_language_anchor_alpha", 0.35))
    p.add_argument("--relation_context_layers", type=int, default=getattr(TrainConfig, "relation_context_layers", 0))
    p.add_argument("--relation_context_heads", type=int, default=getattr(TrainConfig, "relation_context_heads", 8))
    p.add_argument("--bayes_calibration_weight", type=float, default=getattr(TrainConfig, "bayes_calibration_weight", 0.0))
    p.add_argument("--adaptive_calibration_enabled", type=_str2bool, nargs="?", const=True, default=getattr(TrainConfig, "adaptive_calibration_enabled", False))
    p.add_argument("--adaptive_prior_enabled", type=_str2bool, nargs="?", const=True, default=getattr(TrainConfig, "adaptive_prior_enabled", True))
    p.add_argument("--bias_residual_enabled", type=_str2bool, nargs="?", const=True, default=getattr(TrainConfig, "bias_residual_enabled", True))
    p.add_argument("--adaptive_prior_scale", type=float, default=getattr(TrainConfig, "adaptive_prior_scale", 1.0))
    p.add_argument("--bias_residual_scale", type=float, default=getattr(TrainConfig, "bias_residual_scale", 0.25))
    p.add_argument("--lambda_calibration_reg", type=float, default=getattr(TrainConfig, "lambda_calibration_reg", 0.001))
    p.add_argument("--calibration_grad_clip_norm", type=float, default=getattr(TrainConfig, "calibration_grad_clip_norm", 0.0))
    p.add_argument("--emb_dim", type=int, default=TrainConfig.emb_dim)
    p.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    p.add_argument("--pure_local_alpha", type=float, default=TrainConfig.pure_local_alpha)
    p.add_argument("--pure_bridge_layers", type=int, default=TrainConfig.pure_bridge_layers)
    p.add_argument("--deformable_routing_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.deformable_routing_enabled)
    p.add_argument("--deformable_num_points", type=int, default=TrainConfig.deformable_num_points)
    p.add_argument("--deformable_offset_scale", type=float, default=TrainConfig.deformable_offset_scale)
    p.add_argument("--progressive_node_layers", type=int, default=TrainConfig.progressive_node_layers)
    p.add_argument("--progressive_edge_layers", type=int, default=TrainConfig.progressive_edge_layers)
    p.add_argument("--progressive_bilinear_layers", type=int, default=TrainConfig.progressive_bilinear_layers)
    p.add_argument("--bilinear_low_rank", type=_str2bool, nargs="?", const=True, default=TrainConfig.bilinear_low_rank)
    p.add_argument("--bilinear_rank", type=int, default=TrainConfig.bilinear_rank)
    p.add_argument("--bilinear_residual_scale", type=float, default=getattr(TrainConfig, "bilinear_residual_scale", 0.2))
    p.add_argument("--clip_name", type=str, default=TrainConfig.clip_name)
    p.add_argument("--clip_input_res", type=int, default=TrainConfig.clip_input_res)
    p.add_argument("--gradient_checkpointing", type=_str2bool, nargs="?", const=True, default=TrainConfig.gradient_checkpointing)
    p.add_argument("--freeze_clip", type=_str2bool, nargs="?", const=True, default=TrainConfig.freeze_clip)
    p.add_argument("--model_diagnostics_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.model_diagnostics_enabled)
    p.add_argument("--unfreeze_clip_lora", type=_str2bool, nargs="?", const=True, default=TrainConfig.unfreeze_clip_lora)
    p.add_argument("--fp8_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.fp8_enabled)
    p.add_argument("--expandable_segments", type=_str2bool, nargs="?", const=True, default=TrainConfig.expandable_segments)
    p.add_argument("--fp8_recipe", type=str, default=TrainConfig.fp8_recipe)
    p.add_argument("--prompt_aug", type=_str2bool, nargs="?", const=True, default=TrainConfig.prompt_aug)
    p.add_argument("--prompt_aug_prob", type=float, default=TrainConfig.prompt_aug_prob)
    p.add_argument("--prompt_aug_max_paraphrases", type=int, default=TrainConfig.prompt_aug_max_paraphrases)
    p.add_argument("--canon_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.canon_enabled)
    p.add_argument("--canon_model", type=str, default=TrainConfig.canon_model)
    p.add_argument("--paraphrase_cache_path", type=str, default=TrainConfig.paraphrase_cache_path)
    p.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--eval_only", type=_str2bool, nargs="?", const=True, default=False)
    p.add_argument("--eval_batches", type=int, default=TrainConfig.eval_batches)
    p.add_argument("--eval_fast_mode", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_fast_mode)
    p.add_argument("--eval_on_train_split", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_on_train_split)
    p.add_argument("--eval_sgg_iou_thresh", type=float, default=TrainConfig.eval_sgg_iou_thresh)
    p.add_argument("--eval_sgg_use_gt_pairs", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_use_gt_pairs)
    p.add_argument("--eval_sgg_use_clip_obj_classifier", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_use_clip_obj_classifier)
    p.add_argument("--eval_sgg_clip_obj_topk", type=int, default=getattr(TrainConfig, "eval_sgg_clip_obj_topk", 5))
    p.add_argument("--eval_sgg_clip_obj_cache_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_clip_obj_cache_enabled)
    p.add_argument("--eval_sgg_clip_obj_cache_dir", type=str, default=TrainConfig.eval_sgg_clip_obj_cache_dir)
    p.add_argument("--eval_sgg_report_nograph", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_report_nograph)
    p.add_argument("--eval_sgg_use_no_interaction_prior", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_use_no_interaction_prior)
    p.add_argument("--eval_sgg_no_interaction_text", type=str, default=TrainConfig.eval_sgg_no_interaction_text)
    p.add_argument("--eval_sgg_grounding_dino_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_grounding_dino_enabled)
    p.add_argument("--eval_sgg_grounding_dino_model_id", type=str, default=TrainConfig.eval_sgg_grounding_dino_model_id)
    p.add_argument("--eval_sgg_grounding_dino_box_threshold", type=float, default=TrainConfig.eval_sgg_grounding_dino_box_threshold)
    p.add_argument("--eval_sgg_grounding_dino_text_threshold", type=float, default=TrainConfig.eval_sgg_grounding_dino_text_threshold)
    p.add_argument("--eval_sgg_grounding_dino_max_detections", type=int, default=TrainConfig.eval_sgg_grounding_dino_max_detections)
    p.add_argument("--eval_sgg_grounding_dino_cache_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_grounding_dino_cache_enabled)
    p.add_argument("--eval_sgg_grounding_dino_cache_dir", type=str, default=TrainConfig.eval_sgg_grounding_dino_cache_dir)
    p.add_argument("--eval_research_suite", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_research_suite)
    p.add_argument("--eval_latency", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_latency)
    p.add_argument("--eval_latency_batches", type=int, default=TrainConfig.eval_latency_batches)
    p.add_argument("--eval_prune_batches", type=int, default=TrainConfig.eval_prune_batches)
    p.add_argument("--eval_prune_every", type=int, default=TrainConfig.eval_prune_every)
    p.add_argument("--eval_prune_ks", type=str, default=TrainConfig.eval_prune_ks)
    p.add_argument("--eval_paraphrase", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_paraphrase)
    p.add_argument("--paraphrase_n", type=int, default=TrainConfig.paraphrase_n)
    p.add_argument("--eval_debug_grounding", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_debug_grounding)
    p.add_argument("--eval_debug_grounding_max_samples", type=int, default=TrainConfig.eval_debug_grounding_max_samples)
    p.add_argument("--eval_sgg_debug_triplets", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_debug_triplets)
    p.add_argument("--eval_sgg_debug_triplets_max_samples", type=int, default=TrainConfig.eval_sgg_debug_triplets_max_samples)
    p.add_argument("--eval_sgg_use_vg_aliases", type=_str2bool, nargs="?", const=True, default=TrainConfig.eval_sgg_use_vg_aliases)
    p.add_argument("--eval_zs_predicates", type=str, default=TrainConfig.eval_zs_predicates)
    p.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    p.add_argument("--timing_breakdown", type=_str2bool, nargs="?", const=True, default=TrainConfig.timing_breakdown)
    p.add_argument("--timing_warmup_steps", type=int, default=TrainConfig.timing_warmup_steps)
    p.add_argument("--save_path", type=str, default=TrainConfig.save_path)
    p.add_argument("--resume", type=_str2bool, nargs="?", const=True, default=TrainConfig.resume)
    p.add_argument("--resume_from", type=str, default=TrainConfig.resume_from)
    p.add_argument("--reset_epoch", type=_str2bool, nargs="?", const=True, default=TrainConfig.reset_epoch)
    p.add_argument("--max_text_cache", type=int, default=TrainConfig.max_text_cache)
    p.add_argument("--zero_shot_pairs", type=_str2bool, nargs="?", const=True, default=False)
    p.add_argument("--vg150_enabled", type=_str2bool, nargs="?", const=True, default=TrainConfig.vg150_enabled)
    p.add_argument("--vg150_root", type=str, default=TrainConfig.vg150_root)
    p.add_argument("--vg150_source", type=str, default=getattr(TrainConfig, "vg150_source", "auto"))
    return p
def _cfg_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
def _should_use_prompt_aug(cfg: TrainConfig) -> bool:
    return bool(cfg.prompt_aug) and float(cfg.prompt_aug_prob) > 0.0 and int(cfg.prompt_aug_max_paraphrases) > 0
def _should_run_research_eval(cfg: TrainConfig, epoch: int) -> bool:
    return bool(getattr(cfg, "eval_research_suite", False)) and (epoch % int(cfg.eval_prune_every)) == 0


def _objective_uses_spoa(train_objective: str) -> bool:
    return str(train_objective).strip().lower() not in {"predicate_warmup", "ce_only", "pred_ce", "no_spoa", "ground_ce"}


def _objective_uses_grounding(train_objective: str) -> bool:
    return str(train_objective).strip().lower() not in {"predicate_warmup", "ce_only", "pred_ce"}


def _active_branch_report(cfg: TrainConfig, train_objective_name: str) -> Dict[str, Any]:
    uses_spoa = _objective_uses_spoa(train_objective_name) and float(getattr(cfg, "lambda_spoa_alignment", 0.0)) > 0.0
    uses_ground = _objective_uses_grounding(train_objective_name) and float(getattr(cfg, "lambda_dense_grounding", 0.0)) > 0.0
    uses_counterfactual = uses_spoa and bool(getattr(cfg, "predicate_counterfactual_enabled", True)) and float(getattr(cfg, "lambda_counterfactual", 0.0)) > 0.0
    uses_visual_hard = uses_spoa and bool(getattr(cfg, "visual_hard_negative_enabled", False)) and float(getattr(cfg, "lambda_visual_hard_negative", 0.0)) > 0.0
    return {
        "objective": str(train_objective_name),
        "pure_next_pillars": {
            "predce_logit_adjusted": bool(float(getattr(cfg, "lambda_predicate_ce", 0.0)) > 0.0),
            "counterfactual_spoa": bool(uses_spoa),
            "dense_grounding": bool(uses_ground),
        },
        "regularizers": {
            "predicate_counterfactual": bool(uses_counterfactual),
            "gate_regularizer": bool(float(getattr(cfg, "gate_regularizer_weight", 0.0)) > 0.0),
            "lattice_negatives": bool(getattr(cfg, "lattice_loss_enabled", True)),
        },
        "ablation_only": {
            "visual_hard_negative": bool(uses_visual_hard),
            "bilinear_layers": int(getattr(cfg, "progressive_bilinear_layers", 0)),
        },
        "scaling_layers": {
            "pure_phase": str(getattr(cfg, "pure_phase", "core")),
            "object_language_anchor": bool(getattr(cfg, "object_language_anchor_enabled", False)),
            "object_language_anchor_source": str(getattr(cfg, "object_language_anchor_source", "auto")),
            "object_language_anchor_alpha": float(getattr(cfg, "object_language_anchor_alpha", 0.0)),
            "relation_context_layers": int(getattr(cfg, "relation_context_layers", 0)),
            "bayes_calibration_weight": float(getattr(cfg, "bayes_calibration_weight", 0.0)),
            "adaptive_calibration": bool(getattr(cfg, "adaptive_calibration_enabled", False)),
            "adaptive_prior": bool(getattr(cfg, "adaptive_prior_enabled", True)),
            "bias_residual": bool(getattr(cfg, "bias_residual_enabled", True)),
        },
        "architecture": {
            "geom_bias": bool(getattr(cfg, "use_geom_bias", True)),
            "vector_fusion_gate": bool(getattr(cfg, "vector_fusion_gate", True)),
            "deformable_router": bool(getattr(cfg, "deformable_routing_enabled", True)),
            "node_layers": int(getattr(cfg, "progressive_node_layers", 0)),
            "edge_layers": int(getattr(cfg, "progressive_edge_layers", 0)),
            "predicate_classifier": bool(getattr(cfg, "predicate_classifier_enabled", True)),
        },
        "weights": {
            "lambda_predicate_ce": float(getattr(cfg, "lambda_predicate_ce", 0.0)),
            "lambda_spoa_alignment": float(getattr(cfg, "lambda_spoa_alignment", 0.0)),
            "lambda_dense_grounding": float(getattr(cfg, "lambda_dense_grounding", 0.0)),
            "lambda_counterfactual_in_spoa": float(getattr(cfg, "lambda_counterfactual", 0.0)),
            "gate_regularizer_weight": float(getattr(cfg, "gate_regularizer_weight", 0.0)),
            "fusion_gate_temperature": float(getattr(cfg, "fusion_gate_temperature", 1.0)),
            "logit_adj_tau": float(getattr(cfg, "logit_adj_tau", 0.0)),
            "bayes_calibration_weight": float(getattr(cfg, "bayes_calibration_weight", 0.0)),
            "adaptive_calibration": bool(getattr(cfg, "adaptive_calibration_enabled", False)),
            "adaptive_prior": bool(getattr(cfg, "adaptive_prior_enabled", True)),
            "bias_residual": bool(getattr(cfg, "bias_residual_enabled", True)),
        },
    }


def _print_active_branch_report(report: Dict[str, Any]) -> None:
    print("[ActiveBranches] " + json.dumps(report, sort_keys=True), flush=True)

def _object_prompt_for_anchor(name: str) -> str:
    label = str(name).strip().lower()
    if label == "":
        label = "object"
    article = "an" if label[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"a photo of {article} {label}"


def _batch_object_semantic_feats(
    cfg: TrainConfig,
    clip_model: Any,
    processor: Any,
    batch: List[Dict[str, Any]],
    device: torch.device,
) -> Optional[List[torch.Tensor]]:
    if not bool(getattr(cfg, "object_language_anchor_enabled", False)):
        return None
    labels_per_image: List[List[str]] = []
    unique_prompts: List[str] = []
    seen = set()
    for ex in batch:
        labels = [str(x).strip().lower() for x in ex.get("obj_labels", [])]
        labels_per_image.append(labels)
        for label in labels:
            prompt = _object_prompt_for_anchor(label)
            if prompt not in seen:
                seen.add(prompt)
                unique_prompts.append(prompt)
    if len(unique_prompts) == 0:
        return None
    with torch.no_grad():
        feats = clip_text_features(clip_model, processor, unique_prompts, device).detach()
    prompt_to_feat = {prompt: feats[i] for i, prompt in enumerate(unique_prompts)}
    out: List[torch.Tensor] = []
    for labels in labels_per_image:
        if len(labels) == 0:
            out.append(torch.empty((0, int(feats.shape[-1])), device=device, dtype=feats.dtype))
        else:
            out.append(torch.stack([prompt_to_feat[_object_prompt_for_anchor(label)] for label in labels], dim=0))
    return out

def _safe_load_optimizer_state(optim: torch.optim.Optimizer, ckpt: Dict[str, Any]) -> bool:
    if "optim" not in ckpt:
        return False
    try:
        optim.load_state_dict(ckpt["optim"])
        return True
    except ValueError as exc:
        print(f"[System] Skipping optimizer state load: {exc}")
        return False


def _safe_load_scaler_state(scaler: Any, ckpt: Dict[str, Any]) -> bool:
    if "scaler" not in ckpt:
        return False
    try:
        scaler.load_state_dict(ckpt["scaler"])
        return True
    except Exception as exc:
        print(f"[System] Skipping scaler state load: {exc}")
        return False

def _run_core_evals(
    args: argparse.Namespace,
    cfg: TrainConfig,
    metrics_dump: Dict[str, Any],
    model: Any,
    clip_model: Any,
    processor: Any,
    val_loader: Any,
    device: torch.device,
    pred_emb_s2o: torch.Tensor,
    pred_to_idx: Dict[str, int],
    pred_buckets: Dict[str, List[str]],
    canonicalizer: Any,
) -> None:
    from .evals import (
        eval_sgg_standard,
        eval_geometry_hard_negative_stress,
        eval_global_predicate_classification,
        eval_object_leakage_diagnostic,
        eval_prompt_robustness,
        eval_prune_reliability,
        eval_query_grounding,
        eval_query_grounding_split,
        eval_swap_consistency,
    )
    if bool(getattr(cfg, "eval_fast_mode", False)):
        fast_cfg = replace(cfg)
        fast_cfg.eval_sgg_use_clip_obj_classifier = False
        fast_cfg.eval_sgg_grounding_dino_enabled = False
        fast_cfg.eval_sgg_report_nograph = False
        metrics_dump["eval_mode"] = "fast"
        metrics_dump["val_sgg"] = eval_sgg_standard(
            fast_cfg, model, clip_model, processor, val_loader, device, max_batches=int(fast_cfg.eval_batches)
        )
        metrics_dump["val_grounding"] = eval_query_grounding(
            fast_cfg, model, clip_model, processor, val_loader, device, max_batches=int(fast_cfg.eval_batches)
        )
        if str(getattr(cfg, "eval_sgg_compare_score_modes", "")).strip() != "":
            metrics_dump["val_sgg_by_score_mode"] = _run_sgg_score_mode_comparison(
                fast_cfg, model, clip_model, processor, val_loader, device
            )
        return
    metrics_dump["eval_mode"] = "full"
    metrics_dump["val_sgg"] = eval_sgg_standard(
        cfg, model, clip_model, processor, val_loader, device, max_batches=int(cfg.eval_batches)
    )
    metrics_dump["val_grounding"] = eval_query_grounding(
        cfg, model, clip_model, processor, val_loader, device, max_batches=int(cfg.eval_batches)
    )
    if str(getattr(cfg, "eval_sgg_compare_score_modes", "")).strip() != "":
        metrics_dump["val_sgg_by_score_mode"] = _run_sgg_score_mode_comparison(
            cfg, model, clip_model, processor, val_loader, device
        )
    if bool(args.zero_shot_pairs):
        metrics_dump["val_grounding_split_pairs"] = eval_query_grounding_split(
            cfg, model, clip_model, processor, val_loader, device, heldout_pairs=set(), max_batches=int(cfg.eval_batches)
        )
    metrics_dump["val_global_predicate"] = eval_global_predicate_classification(
        cfg,
        model,
        clip_model,
        processor,
        val_loader,
        device,
        pred_emb_s2o,
        pred_to_idx,
        pred_buckets=pred_buckets,
        max_batches=int(cfg.eval_batches),
    )
    metrics_dump["val_object_leakage"] = eval_object_leakage_diagnostic(
        cfg, model, clip_model, processor, val_loader, device, max_batches=int(cfg.eval_batches)
    )
    metrics_dump["val_geom_hardneg"] = eval_geometry_hard_negative_stress(
        cfg, model, clip_model, processor, val_loader, device, hard_k=16, max_batches=int(cfg.eval_batches)
    )
    metrics_dump["val_swap_consistency"] = eval_swap_consistency(
        cfg, model, clip_model, processor, val_loader, device, max_batches=int(cfg.eval_batches)
    )
    if bool(cfg.eval_paraphrase):
        metrics_dump["val_prompt_robustness"] = eval_prompt_robustness(
            cfg,
            model,
            clip_model,
            processor,
            val_loader,
            device,
            canonicalizer=canonicalizer,
            max_batches=int(cfg.eval_batches),
        )
    if int(cfg.learned_prune_k) > 0:
        metrics_dump["val_prune_reliability"] = eval_prune_reliability(
            cfg, model, clip_model, processor, val_loader, device, k=int(cfg.learned_prune_k), max_batches=int(cfg.eval_batches)
        )


def _run_sgg_score_mode_comparison(
    cfg: TrainConfig,
    model: Any,
    clip_model: Any,
    processor: Any,
    val_loader: Any,
    device: torch.device,
) -> Dict[str, Any]:
    from .evals import eval_sgg_standard

    modes = [
        str(x).strip().lower()
        for x in str(getattr(cfg, "eval_sgg_compare_score_modes", "")).split(",")
        if str(x).strip() != ""
    ]
    out: Dict[str, Any] = {}
    for mode in modes:
        cmp_cfg = replace(cfg)
        cmp_cfg.eval_sgg_predicate_score_mode = mode
        cmp_cfg.eval_sgg_use_clip_obj_classifier = False
        cmp_cfg.eval_sgg_grounding_dino_enabled = False
        cmp_cfg.eval_sgg_report_nograph = False
        out[mode] = eval_sgg_standard(
            cmp_cfg,
            model,
            clip_model,
            processor,
            val_loader,
            device,
            max_batches=int(cmp_cfg.eval_batches),
        )
    return out

def _run_research_evals(
    cfg: TrainConfig,
    metrics_dump: Dict[str, Any],
    model: Any,
    clip_model: Any,
    processor: Any,
    val_loader: Any,
    device: torch.device,
) -> None:
    from .evals import (
        eval_latency_throughput,
        eval_prune_tradeoff,
        eval_query_grounding_vs_k,
    )
    if bool(cfg.eval_latency):
        metrics_dump["val_latency"] = eval_latency_throughput(
            cfg, model, clip_model, processor, val_loader, device, max_batches=int(cfg.eval_latency_batches)
        )
    ks = [int(x) for x in str(cfg.eval_prune_ks).split(",") if str(x).strip() != ""]
    ks = [k for k in ks if k > 0]
    if len(ks) == 0:
        ks = [16, 32, 64, 128]
    if int(cfg.learned_prune_k) > 0:
        metrics_dump["val_prune_tradeoff"] = eval_prune_tradeoff(
            cfg, model, clip_model, processor, val_loader, device, ks=ks, max_batches=int(cfg.eval_prune_batches)
        )
    metrics_dump["val_grounding_vs_k"] = eval_query_grounding_vs_k(
        cfg, model, clip_model, processor, val_loader, device, ks=ks, max_batches=min(25, int(cfg.eval_batches))
    )
def _print_epoch_summary(metrics_dump: Dict[str, Any]) -> None:
    sgg = metrics_dump.get("val_sgg", {}) if isinstance(metrics_dump.get("val_sgg", {}), dict) else {}
    predcls = sgg.get("predcls", {}) if isinstance(sgg.get("predcls", {}), dict) else {}
    sgcls = sgg.get("sgcls", {}) if isinstance(sgg.get("sgcls", {}), dict) else {}
    sgdet = sgg.get("sgdet", {}) if isinstance(sgg.get("sgdet", {}), dict) else {}
    grounding = metrics_dump.get("val_grounding", {}) if isinstance(metrics_dump.get("val_grounding", {}), dict) else {}

    def _metric(block: Dict[str, Any], key: str) -> float:
        return float(block.get(key, 0.0))

    eval_mode = str(metrics_dump.get("eval_mode", "unknown"))
    print("\n[Epoch Summary]")
    print("+----------------------+----------+")
    print(f"| {'Train Loss':<20} | {float(metrics_dump['train']['avg_loss']):>8.4f} |")
    print(f"| {'Eval Mode':<20} | {eval_mode[:8]:>8} |")
    if len(grounding) > 0:
        print(f"| {'Grounding R@50':<20} | {_metric(grounding, 'R@50'):>8.4f} |")
    if len(predcls) > 0:
        print(f"| {'PredCls R@50':<20} | {_metric(predcls, 'R@50'):>8.4f} |")
        print(f"| {'PredCls mR@50':<20} | {_metric(predcls, 'mR@50'):>8.4f} |")
    if len(sgcls) > 0:
        print(f"| {'SGCls R@50':<20} | {_metric(sgcls, 'R@50'):>8.4f} |")
        print(f"| {'SGCls mR@50':<20} | {_metric(sgcls, 'mR@50'):>8.4f} |")
    if len(sgdet) > 0 and bool(sgdet.get('available', False)):
        print(f"| {'SGDet R@50':<20} | {_metric(sgdet, 'R@50'):>8.4f} |")
        print(f"| {'SGDet mR@50':<20} | {_metric(sgdet, 'mR@50'):>8.4f} |")
    by_mode = metrics_dump.get("val_sgg_by_score_mode", {})
    if isinstance(by_mode, dict) and len(by_mode) > 0:
        for mode, mode_sgg in by_mode.items():
            mode_predcls = mode_sgg.get("predcls", {}) if isinstance(mode_sgg, dict) and isinstance(mode_sgg.get("predcls", {}), dict) else {}
            if len(mode_predcls) > 0:
                print(
                    f"[ScoreMode {str(mode)}] "
                    f"PredCls R@50={_metric(mode_predcls, 'R@50'):.4f} "
                    f"mR@50={_metric(mode_predcls, 'mR@50'):.4f}"
                )
    predicate_diag = sgg.get("predicate_diag", {}) if isinstance(sgg.get("predicate_diag", {}), dict) else {}
    if len(predicate_diag) > 0:
        mode = str(predicate_diag.get("score_mode", "unknown"))
        alpha = float(predicate_diag.get("ensemble_alpha", 0.0))
        gt_top = ", ".join(
            f"{x.get('label', '')}:{int(x.get('count', 0))}"
            for x in predicate_diag.get("gt_top", [])[:5]
            if isinstance(x, dict)
        )
        pred_top = ", ".join(
            f"{x.get('label', '')}:{int(x.get('count', 0))}"
            for x in predicate_diag.get("pred_top", [])[:5]
            if isinstance(x, dict)
        )
        print(f"| {'Pred Score':<20} | {mode[:8]:>8} |")
        if mode in {"ensemble", "auto"}:
            print(f"| {'Ens Alpha':<20} | {alpha:>8.2f} |")
        if gt_top:
            print(f"[Predicate GT top] {gt_top}")
        if pred_top:
            print(f"[Predicate Pred top] {pred_top}")
    print("+----------------------+----------+\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    
    args, _unknown = parser.parse_known_args(argv)
    
    base_cfg = TrainConfig()
    
    base_cfg.stage = args.stage
    base_cfg = apply_stage_config(base_cfg)
    import sys
    called_args = []
    argv_to_check = argv if argv is not None else sys.argv
    for arg in argv_to_check:
        if arg.startswith("--"):
            called_args.append(arg[2:].split('=')[0]) 
            
    for k, v in vars(args).items():
        if k in called_args:
            setattr(base_cfg, k, v)
        elif not hasattr(base_cfg, k) or getattr(base_cfg, k) is None:
            # Nếu cfg chưa có thì gán giá trị mặc định từ args
            setattr(base_cfg, k, v)
    cfg = base_cfg
    def _reapply_explicit_cli_args(target_cfg: TrainConfig) -> TrainConfig:
        for key in called_args:
            if hasattr(target_cfg, key) and hasattr(args, key):
                setattr(target_cfg, key, getattr(args, key))
        return target_cfg
    from .clip_utils import (
        build_confusable_index,
        clip_text_features,
        clip_text_is_trainable,
        configure_clip,
        encode_predicate_vocab,
    )
    from .datasets import scan_vg150_predicate_vocab
    from .datasets.vg150_loader import VG150DataLoader, VG150LoaderConfig
    from .ddp_utils import ddp_all_gather_tensor, ddp_init, ddp_is_enabled, is_main, seed_all, unwrap_ddp
    from .evals import (
        eval_sgg_standard,
        eval_geometry_hard_negative_stress,
        eval_global_predicate_classification,
        eval_latency_throughput,
        eval_object_leakage_diagnostic,
        eval_prompt_robustness,
        eval_prune_reliability,
        eval_prune_tradeoff,
        eval_query_grounding,
        eval_query_grounding_split,
        eval_query_grounding_vs_k,
        eval_swap_consistency,
        split_predicates_head_mid_tail,
    )
    from .fp8_utils import fp8_autocast, fp8_supported_on_current_device
    
    from .lattice import build_predicate_similarity_matrix
    from .losses import (
        RingQueue,
        counterfactual_hard_negative_loss,
        info_nce_with_queue,
        visual_counterfactual_inbatch_loss,
    )
    from .models.relational_model import RelationalModel
    from .text_cache import PrecomputedVariantCache
    if bool(getattr(cfg, "expandable_segments", False)):
        alloc_conf = str(os.environ.get("PYTORCH_ALLOC_CONF", "")).strip()
        if "expandable_segments" not in alloc_conf:
            os.environ["PYTORCH_ALLOC_CONF"] = (
                "expandable_segments:True" if alloc_conf == "" else f"{alloc_conf},expandable_segments:True"
            )
    seed_all(int(cfg.seed))
    rank, _world = ddp_init()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if str(getattr(cfg, "gpu_preset", "")).strip() != "":
        cfg = apply_gpu_preset(cfg, str(cfg.gpu_preset))
    if str(getattr(cfg, "a100_ddp_preset", "")).strip() != "":
        cfg = apply_ddp_a100_preset(cfg, str(cfg.a100_ddp_preset))
    cfg = _reapply_explicit_cli_args(cfg)
    if bool(cfg.amp) and device.type != "cuda":
        cfg.amp = False
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark) and device.type == "cuda"
    if bool(cfg.fp8_enabled) and not fp8_supported_on_current_device():
        if is_main():
            print("[Runtime] Disabling FP8: current GPU is not Hopper-class; using bf16/tf32 path.")
        cfg.fp8_enabled = False
    warn_if_low_shm()
    canonicalizer = PredicateCanonicalizer(
        enabled=bool(cfg.canon_enabled),
        model_name=str(cfg.canon_model),
        device=str(device),
        cache_path=str(cfg.paraphrase_cache_path),
    )
    print("[System] Loading CLIP model, please wait...")
    clip_model, processor, clip_vision_dim, text_dim = configure_clip(cfg.clip_name, device)
    if cfg.gradient_checkpointing:
        clip_model.gradient_checkpointing_enable()
    cm = unwrap_ddp(clip_model)
    clip_proj_dim = int(
        getattr(getattr(cm, "config", object()), "projection_dim", 0)
        or getattr(getattr(cm, "text_projection", object()), "out_features", 0)
        or int(cfg.emb_dim)
    )
    cfg.emb_dim = int(clip_proj_dim)
    model = RelationalModel(cfg, clip_vision_dim=clip_vision_dim, text_dim=text_dim)
    model = model.to(device)
    if device.type == "cuda" and bool(cfg.channels_last):
        clip_model = clip_model.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
    if bool(cfg.torch_compile) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=str(cfg.torch_compile_mode))
            if is_main():
                print(f"[Runtime] Compiled relational model with torch.compile(mode={cfg.torch_compile_mode}).")
        except Exception as exc:
            if is_main():
                print(f"[Runtime] torch.compile skipped: {exc}")
    if bool(cfg.freeze_clip):
        for p in clip_model.parameters():
            p.requires_grad_(False)
    if ddp_is_enabled():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = model.to(local_rank) 
        model = DDP(model, device_ids=[local_rank])
        if not cfg.freeze_clip:
            clip_model = DDP(clip_model, device_ids=[local_rank])
    base_lr = float(cfg.lr)
    cfg.lr = base_lr
    train_objective_name = str(getattr(cfg, "train_objective", "full")).strip().lower()
    active_branch_report = _active_branch_report(cfg, train_objective_name)
    if is_main():
        _print_active_branch_report(active_branch_report)
    
    model_params = [p for p in model.parameters() if p.requires_grad]
    clip_params = [p for p in clip_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        [
            {"params": model_params, "lr": float(cfg.lr), "weight_decay": float(cfg.weight_decay)}, 
            {"params": clip_params, "lr": float(cfg.lr) * 0.1, "weight_decay": 0.0} 
        ],
        lr=float(cfg.lr),
    )
    
    start_epoch = 0 
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp) and device.type == "cuda")
    # Load data based on mode (VG150 or multi-task streaming)
    if bool(cfg.vg150_enabled):
        if is_main():
            gpu_preset_name = str(getattr(cfg, "gpu_preset", "")).strip() or "none"
            print(
                f"[Data] Preparing VG150 loader, source={str(getattr(cfg, 'vg150_source', 'auto'))}, "
                f"requested_streaming={bool(cfg.hf_streaming)}, workers={int(cfg.num_workers)}, "
                f"root={str(cfg.vg150_root)}, gpu_preset={gpu_preset_name}"
            )
        vg150_cfg = VG150LoaderConfig(
            vg150_root=str(cfg.vg150_root),
            source=str(getattr(cfg, "vg150_source", "auto")),
            split="train",
            streaming=bool(cfg.hf_streaming),
            hf_dataset_id=str(cfg.hf_dataset_id),
            batch_size=int(cfg.batch_size),
            num_workers=resolve_loader_num_workers(int(cfg.num_workers), validation=False),
            pin_memory=resolve_loader_pin_memory(str(device)),
            persistent_workers=bool(cfg.loader_persistent_workers),
            prefetch_factor=int(cfg.loader_prefetch_factor),
            drop_last=True,
            seed=int(cfg.seed),
            shuffle_buffer_size=int(cfg.hf_shuffle_buffer_size),
            samples_per_epoch=(0 if int(cfg.samples_per_epoch) <= 0 else int(cfg.samples_per_epoch)),
            max_images=int(cfg.max_images),
            max_objects=int(cfg.max_objects),
            max_pairs=int(cfg.max_pairs),
            use_rfs=bool(cfg.use_rfs),
            rfs_t=float(cfg.rfs_t),
            use_all_pairs=bool(cfg.use_all_pairs),
            negative_pair_ratio=float(cfg.negative_pair_ratio),
            predicate_sampler_enabled=bool(getattr(cfg, "predicate_sampler_enabled", False)),
            predicate_sampler_power=float(getattr(cfg, "predicate_sampler_power", 0.75)),
            predicate_sampler_max_weight=float(getattr(cfg, "predicate_sampler_max_weight", 20.0)),
        )
        joint_loader = VG150DataLoader(cfg=vg150_cfg, split="train", shuffle=True, processor=processor, clip_input_res=int(cfg.clip_input_res))
        if is_main():
            print(
                f"[Data] VG150 train loader source={getattr(joint_loader, 'source_mode', 'unknown')}, "
                f"workers={int(vg150_cfg.num_workers)}, prefetch={int(vg150_cfg.prefetch_factor)}, "
                f"predicate_sampler={bool(getattr(joint_loader, 'predicate_sampler_active', False))}"
            )
        
        eval_split = "train" if bool(getattr(cfg, "eval_on_train_split", False)) else "validation"
        vg150_val_cfg = VG150LoaderConfig(
            vg150_root=str(cfg.vg150_root),
            source=str(getattr(cfg, "vg150_source", "auto")),
            split=eval_split,
            streaming=bool(cfg.hf_streaming),
            hf_dataset_id=str(cfg.hf_dataset_id),        
            batch_size=int(cfg.batch_size),
            num_workers=resolve_loader_num_workers(int(cfg.num_workers), validation=True),
            pin_memory=resolve_loader_pin_memory(str(device)),
            persistent_workers=bool(cfg.loader_persistent_workers),
            prefetch_factor=max(2, int(cfg.loader_prefetch_factor // 2)),
            drop_last=False,
            seed=int(cfg.seed) + 13,
            shuffle_buffer_size=max(1000, int(cfg.hf_shuffle_buffer_size // 2)),
            samples_per_epoch=(0 if int(cfg.samples_per_epoch) <= 0 else int(cfg.samples_per_epoch)),
            max_images=int(cfg.max_images),
            max_objects=int(cfg.max_objects),
            max_pairs=int(cfg.max_pairs),
            use_rfs=False,
            rfs_t=float(cfg.rfs_t),
            use_all_pairs=bool(cfg.use_all_pairs),
            negative_pair_ratio=-1.0,
        )
        val_vg_loader = VG150DataLoader(cfg=vg150_val_cfg, split=eval_split, shuffle=False, processor=processor, clip_input_res=int(cfg.clip_input_res))
        if is_main():
            print(
                f"[Data] VG150 val loader split={eval_split}, source={getattr(val_vg_loader, 'source_mode', 'unknown')}, "
                f"workers={int(vg150_val_cfg.num_workers)}, prefetch={int(vg150_val_cfg.prefetch_factor)}"
            )
        val_loader = val_vg_loader.loader
        
        # Scan VG150 predicate vocabulary (50 predicates)
        global_pred_pool = scan_vg150_predicate_vocab(vg150_root=str(cfg.vg150_root))
        pred_freq_scan = _iter_predicates_from_dataset_for_weights(getattr(joint_loader.loader, "dataset", None))
    else:
        raise NotImplementedError(
            "Only VG150 mode is supported in the cleaned codebase. "
            "Set --vg150_enabled true."
        )
    pred_freq = {p: int(pred_freq_scan.get(p, 1)) for p in global_pred_pool} if "pred_freq_scan" in locals() else {p: 1 for p in global_pred_pool}
    global_pred_pool = [str(p).strip().lower() for p in global_pred_pool if str(p).strip() != ""]
    global_pred_pool = list(dict.fromkeys(global_pred_pool))
    if "relation" not in global_pred_pool:
        global_pred_pool.append("relation")
    pred_buckets = split_predicates_head_mid_tail(global_pred_pool, pred_freq)
    _, pred_emb_s2o = encode_predicate_vocab(
        clip_model,
        processor,
        global_pred_pool,
        device,
        prompt_fn=pred_prompt_roles,
        direction="s2o",
    )
    pred_to_idx = {p: i for i, p in enumerate(global_pred_pool)}
    pred_ce_weights = _build_predicate_ce_weights(cfg, global_pred_pool, pred_freq, device)
    pred_freq_tensor = torch.tensor(
        [max(1.0, float(pred_freq.get(str(p), 1))) for p in global_pred_pool],
        dtype=torch.float32,
        device=device,
    )
    pred_log_prior = torch.log(pred_freq_tensor / pred_freq_tensor.sum().clamp_min(1.0)).detach()
    if is_main():
        top_pred_freq = sorted(pred_freq.items(), key=lambda kv: int(kv[1]), reverse=True)[:8]
        print(f"[Data] Predicate CE mode={cfg.predicate_ce_loss}, weight_power={float(cfg.predicate_ce_weight_power):.2f}, top_freq={top_pred_freq}", flush=True)
    confusable_idx = build_confusable_index(pred_emb_s2o, topm=min(32, max(1, len(global_pred_pool) - 1)))
    pred_sim_matrix = (
        build_predicate_similarity_matrix(pred_emb_s2o).to(device)
        if bool(getattr(cfg, "lattice_loss_enabled", True))
        else torch.empty((0, 0), device=device)
    )
    rel_queue = RingQueue(int(cfg.emb_dim), int(cfg.rel_queue_size), device=device)
    os.makedirs(str(cfg.out_dir), exist_ok=True)
    resume_path = str(getattr(cfg, "resume_from", "")).strip() or str(getattr(cfg, "save_path", ""))
    if bool(getattr(cfg, "resume", False)) and not os.path.exists(resume_path):
        raise FileNotFoundError(f"--resume true but checkpoint was not found: {resume_path}")
    if bool(getattr(cfg, "resume", False)) and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        mdl = model.module if isinstance(model, DDP) else model
        mdl.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if "clip" in ckpt:
            unwrap_ddp(clip_model).load_state_dict(ckpt["clip"], strict=False)
            print("[System] Loaded fine-tuned CLIP weights.")
        if not bool(getattr(cfg, "reset_epoch", False)):
            optim_loaded = _safe_load_optimizer_state(optim, ckpt)
            scaler_loaded = _safe_load_scaler_state(scaler, ckpt)
            if optim_loaded:
                start_epoch = int(ckpt.get("epoch", 0)) + 1
            else:
                print("[System] Optimizer state not restored; restarting epoch count for this stage.")
            _ = scaler_loaded

        if int(cfg.stage) == 2 and start_epoch == 0:
            print("[System] Stage 2 detected: Purging stale Stage 1 queue to prevent divergence.")
            rel_queue = RingQueue(int(cfg.emb_dim), int(cfg.rel_queue_size), device=device)
    if is_main():
        print(
            "[Config] "
            f"stage={int(cfg.stage)} objective={train_objective_name} "
            f"spoa={_objective_uses_spoa(train_objective_name)} "
            f"grounding={_objective_uses_grounding(train_objective_name)} "
            f"eval_split={eval_split} score_mode={str(getattr(cfg, 'eval_sgg_predicate_score_mode', 'unknown'))} "
            f"gt_pairs={bool(getattr(cfg, 'eval_sgg_use_gt_pairs', False))} "
            f"lr={float(cfg.lr):.2e} batch={int(cfg.batch_size)} accum={int(cfg.accum_steps)} "
            f"queue_min={int(getattr(cfg, 'rel_queue_min_negatives', 1024))}"
        )
    total_updates = max(1, (int(cfg.epochs) * len(joint_loader) + max(1, int(cfg.accum_steps)) - 1) // max(1, int(cfg.accum_steps)))
    warmup_steps = max(
        int(cfg.warmup_steps),
        int(cfg.warmup_epochs) * max(1, len(joint_loader) // max(1, int(cfg.accum_steps))),
    )
    min_lr = 1e-7
    def _lr_at_step(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return min_lr + (base_lr - min_lr) * (float(step_idx + 1) / float(warmup_steps))
        if str(cfg.lr_schedule).strip().lower() not in {"cosine", "cos"}:
            return base_lr
        if total_updates <= warmup_steps:
            return base_lr
        prog = float(step_idx - warmup_steps) / float(max(1, total_updates - warmup_steps))
        prog = min(1.0, max(0.0, prog))
        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
    for pg in optim.param_groups:
        pg["lr"] = float(min_lr)
    use_gpu = torch.cuda.is_available()
    amp_dtype_name = str(getattr(cfg, "amp_dtype", "bf16")).strip().lower()
    if amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    elif amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16 if (use_gpu and torch.cuda.is_bf16_supported()) else torch.float16
    else:
        amp_dtype = torch.bfloat16 if (use_gpu and torch.cuda.is_bf16_supported()) else torch.float16
    
    global_step = start_epoch * len(joint_loader)
    update_step = start_epoch * (len(joint_loader) // max(1, int(cfg.accum_steps)))
    use_prompt_aug = _should_use_prompt_aug(cfg)
    def _set_clip_trainable(trainable: bool) -> None:
        for p in unwrap_ddp(clip_model).parameters():
            p.requires_grad_(bool(trainable))
    def _apply_progressive_unfreeze(epoch_idx: int) -> None:
        if not bool(cfg.progressive_unfreeze):
            return
        cm = unwrap_ddp(clip_model)
        for p in cm.parameters():
            p.requires_grad_(False)
        layer_groups: List[torch.nn.Module] = []
        for root_name in ["vision_model", "text_model"]:
            root = getattr(cm, root_name, None)
            enc = getattr(root, "encoder", None) if root is not None else None
            layers = getattr(enc, "layers", None) if enc is not None else None
            if layers is not None:
                layer_groups.extend(list(layers))
        if len(layer_groups) == 0:
            _set_clip_trainable(True)
            return
        progress = float(epoch_idx + 1) / float(max(1, int(cfg.epochs)))
        n_unfreeze = max(1, int(round(progress * float(len(layer_groups)))))
        for layer in layer_groups[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for proj_name in ["visual_projection", "text_projection"]:
            proj = getattr(cm, proj_name, None)
            if proj is not None and hasattr(proj, "parameters"):
                for p in proj.parameters():
                    p.requires_grad_(True)
        if hasattr(cm, "logit_scale") and isinstance(cm.logit_scale, torch.Tensor):
            cm.logit_scale.requires_grad_(True)
    if int(cfg.epochs) == 0 and is_main():
        print("\n[Evaluation Only Mode]")
        model.eval()
        unwrap_ddp(clip_model).eval()
        
        metrics_dump = {"epoch": 0, "train": {"avg_loss": 0.0}}
        _run_core_evals(
            args,
            cfg,
            metrics_dump,
            model,
            clip_model,
            processor,
            val_loader,
            device,
            pred_emb_s2o,
            pred_to_idx,
            pred_buckets,
            canonicalizer,
        )
        
        sgg = metrics_dump.get("val_sgg", {})
        predcls = sgg.get("predcls", {}) if isinstance(sgg.get("predcls", {}), dict) else {}
        sgcls = sgg.get("sgcls", {}) if isinstance(sgg.get("sgcls", {}), dict) else {}
        sgdet = sgg.get("sgdet", {}) if isinstance(sgg.get("sgdet", {}), dict) else {}
        predcls_ng = sgg.get("predcls_nogc", {}) if isinstance(sgg.get("predcls_nogc", {}), dict) else {}
        sgcls_ng = sgg.get("sgcls_nogc", {}) if isinstance(sgg.get("sgcls_nogc", {}), dict) else {}
        sgdet_ng = sgg.get("sgdet_nogc", {}) if isinstance(sgg.get("sgdet_nogc", {}), dict) else {}
        print("\n[Evaluation Summary]")
        print("+--------------------+------------+")
        print(f"| {'PredCls R@20':<18} | {float(predcls.get('R@20', 0.0)):>10.4f} |")
        print(f"| {'PredCls R@50':<18} | {float(predcls.get('R@50', 0.0)):>10.4f} |")
        print(f"| {'PredCls R@100':<18} | {float(predcls.get('R@100', 0.0)):>10.4f} |")
        print(f"| {'PredCls mR@20':<18} | {float(predcls.get('mR@20', 0.0)):>10.4f} |")
        print(f"| {'PredCls mR@50':<18} | {float(predcls.get('mR@50', 0.0)):>10.4f} |")
        print(f"| {'PredCls mR@100':<18} | {float(predcls.get('mR@100', 0.0)):>10.4f} |")
        if bool(predcls_ng.get('available', False)):
            print(f"| {'PredCls ngR@50':<18} | {float(predcls_ng.get('R@50', 0.0)):>10.4f} |")
        print(f"| {'SGCls R@50':<18} | {float(sgcls.get('R@50', 0.0)):>10.4f} |")
        print(f"| {'SGCls mR@50':<18} | {float(sgcls.get('mR@50', 0.0)):>10.4f} |")
        if bool(sgcls_ng.get('available', False)):
            print(f"| {'SGCls ngR@50':<18} | {float(sgcls_ng.get('R@50', 0.0)):>10.4f} |")
        if bool(sgdet.get('available', False)):
            print(f"| {'SGDet R@50':<18} | {float(sgdet.get('R@50', 0.0)):>10.4f} |")
            print(f"| {'SGDet mR@50':<18} | {float(sgdet.get('mR@50', 0.0)):>10.4f} |")
            if bool(sgdet_ng.get('available', False)):
                print(f"| {'SGDet ngR@50':<18} | {float(sgdet_ng.get('R@50', 0.0)):>10.4f} |")
        else:
            print(f"| {'SGDet':<18} | {'N/A':>10} |")
        print("+--------------------+------------+\n")

        out_dir = str(cfg.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        path = str(cfg.save_metrics_json).strip()
        if path == "":
            path = os.path.join(out_dir, "metrics.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_dump) + "\n")
        per_epoch_dir = os.path.join(out_dir, "epoch_metrics")
        os.makedirs(per_epoch_dir, exist_ok=True)
        with open(os.path.join(per_epoch_dir, "epoch_000.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_dump, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "latest_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_dump, f, ensure_ascii=False, indent=2)

        import sys
        sys.exit(0)
        
    best_predcls_r50 = float("-inf")
    best_predcls_mr50 = float("-inf")

    for epoch in range(start_epoch, int(cfg.epochs)):
            
        if hasattr(joint_loader, "sampler") and callable(getattr(joint_loader.sampler, "set_epoch", None)):
            joint_loader.sampler.set_epoch(epoch)
        elif hasattr(joint_loader, "dataset") and callable(getattr(joint_loader.dataset, "set_epoch", None)):
            joint_loader.dataset.set_epoch(epoch)
        elif hasattr(joint_loader, "loader") and hasattr(joint_loader.loader, "sampler") and callable(getattr(joint_loader.loader.sampler, "set_epoch", None)):
            joint_loader.loader.sampler.set_epoch(epoch)
        
        clip_warmup_frozen = int(epoch) < int(getattr(cfg, "clip_unfreeze_after_epochs", 0))
        if clip_warmup_frozen:
            _set_clip_trainable(False)
        elif bool(cfg.progressive_unfreeze):
            _apply_progressive_unfreeze(epoch - int(getattr(cfg, "clip_unfreeze_after_epochs", 0)))
        else:
            _set_clip_trainable(not bool(cfg.freeze_clip))
        model.train()
        clip_model.train((not bool(cfg.freeze_clip)) and (not clip_warmup_frozen))
        optim.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_l_spoa = 0.0
        epoch_l_ground = 0.0
        epoch_l_pred_ce = 0.0
        epoch_batches = 0
        epoch_positive_pairs = 0
        epoch_candidate_pairs = 0
        epoch_vis_mined = 0
        epoch_vis_used = 0
        epoch_vis_batches = 0
        epoch_vis_active_batches = 0
        if clip_text_is_trainable(clip_model):
            _, pred_emb_s2o = encode_predicate_vocab(
                clip_model,
                processor,
                global_pred_pool,
                device,
                prompt_fn=pred_prompt,
                direction="s2o",
            )
            confusable_idx = build_confusable_index(pred_emb_s2o, topm=min(32, max(1, len(global_pred_pool) - 1)))
            pred_sim_matrix = (
                build_predicate_similarity_matrix(pred_emb_s2o).to(device)
                if bool(getattr(cfg, "lattice_loss_enabled", True))
                else torch.empty((0, 0), device=device)
            )
        cache_verbose = True if epoch == start_epoch else is_main()
        
        variant_cache = PrecomputedVariantCache(
            clip_model=clip_model, processor=processor, pred_vocab=global_pred_pool,
            device=device, prompt_fn=pred_prompt_roles, candidate_fn=build_predicate_prompt_candidates,
            canonicalizer=canonicalizer, max_paraphrases=(int(cfg.prompt_aug_max_paraphrases) if use_prompt_aug else 0),
            direction="s2o", lru_size=int(getattr(args, "max_text_cache", 256)), verbose=cache_verbose,
        )
        
        variant_cache_o2s = PrecomputedVariantCache(
            clip_model=clip_model, processor=processor, pred_vocab=global_pred_pool,
            device=device, prompt_fn=pred_prompt_roles, candidate_fn=build_predicate_prompt_candidates,
            canonicalizer=canonicalizer, max_paraphrases=(int(cfg.prompt_aug_max_paraphrases) if use_prompt_aug else 0),
            direction="o2s", lru_size=int(getattr(args, "max_text_cache", 256)), verbose=False,
        )
        
        variant_cache.cache.update(variant_cache_o2s.cache)
        if is_main():
            print(f"[Epoch {epoch}] Text cache stats: {variant_cache.stats()}", flush=True)
        timing_enabled = bool(cfg.timing_breakdown)
        timing_warmup_steps = max(0, int(getattr(cfg, "timing_warmup_steps", 0)))
        timing_acc = {
            "data_wait": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "optim": 0.0,
            "queue": 0.0,
            "step_total": 0.0,
            "measured_steps": 0,
        }
        last_step_end = time.perf_counter()
        for step, batch in enumerate(joint_loader):
            step_wall_start = time.perf_counter()
            data_wait_time = step_wall_start - last_step_end
            _timing_sync(device, timing_enabled)
            step_compute_start = time.perf_counter()
            # Pre-compute batch metadata OUTSIDE autocast (CPU work, not on critical path)
            batch_pairs = [ex["pairs"] for ex in batch]
            batch_rel_pred_ids = [ex.get("rel_pred_ids", torch.empty((0,), dtype=torch.long)) for ex in batch]
            batch_rel_pair_hashes = [ex.get("rel_pair_hashes", torch.empty((0,), dtype=torch.long)) for ex in batch]
            batch_rel_pos_mask = [ex.get("rel_pos_mask", torch.empty((0,), dtype=torch.bool)) for ex in batch]
            batch_rel_non_sym_mask = [ex.get("rel_non_sym_mask", torch.empty((0,), dtype=torch.bool)) for ex in batch]
            batch_rel_texts = [ex.get("rel_texts", []) for ex in batch]
            batch_rel_role_swap_texts = [ex.get("rel_role_swap_texts", []) for ex in batch]
            batch_rel_cf_attr_texts = [ex.get("rel_cf_attr_texts", []) for ex in batch]
            batch_force_keep = [
                torch.nonzero(ex.get("rel_pos_mask", torch.zeros((0,), dtype=torch.bool)), as_tuple=False).squeeze(1)
                if ex.get("rel_pos_mask", None) is not None
                else None
                for ex in batch
            ]
            # Stack preprocessed tensors from workers (GPU-ready)
            pixel = torch.stack([ex["pixel_values"] for ex in batch], dim=0).to(device, non_blocking=True)
            if device.type == "cuda" and bool(cfg.channels_last):
                pixel = pixel.contiguous(memory_format=torch.channels_last)
            obj_boxes_224 = [ex["obj_boxes_224"].to(device, non_blocking=True) for ex in batch]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(cfg.amp)):
                cm = unwrap_ddp(clip_model)
                vision_out = cm.vision_model(pixel_values=pixel)
                tokens = vision_out.last_hidden_state[:, 1:, :]
                bsz, n_tok, d_ch = tokens.shape
                side = int(math.sqrt(n_tok))
                feat_map = tokens.transpose(1, 2).reshape(bsz, d_ch, side, side)
                out = model.module if isinstance(model, DDP) else model
                obj_semantic_feats = _batch_object_semantic_feats(cfg, clip_model, processor, batch, device)
                with fp8_autocast(enabled=bool(cfg.fp8_enabled)):                   
                    regs, rels, rel_swaps, assigns, gates, kept_idx = out.forward_from_featmap(
                        feat_map,
                        obj_boxes_224=obj_boxes_224,
                        pairs=batch_pairs,
                        return_swapped=True,
                        return_kept=True,
                        force_keep=batch_force_keep,
                        obj_semantic_feats=obj_semantic_feats,
                    )
            kept_cpu = [
                (k.detach().cpu().long() if isinstance(k, torch.Tensor) else torch.tensor(k, dtype=torch.long))
                for k in kept_idx
            ]
            keep_sizes = [
                min(
                    int(r.shape[0]),
                    int(rs.shape[0]),
                    int(kept_cpu[i].numel()),
                    int(batch_rel_pred_ids[i].shape[0]),
                    int(batch_rel_pair_hashes[i].shape[0]),
                    int(batch_rel_pos_mask[i].shape[0]),
                    int(batch_rel_non_sym_mask[i].shape[0]),
                    len(batch_rel_texts[i]),
                    len(batch_rel_role_swap_texts[i]),
                    len(batch_rel_cf_attr_texts[i]),
                )
                for i, (r, rs) in enumerate(zip(rels, rel_swaps))
            ]
            keep_sizes_t = torch.tensor(keep_sizes, device=device, dtype=torch.long)
            if int(keep_sizes_t.sum().item()) == 0:
                continue
            kept_trimmed = [kept_cpu[i][: int(keep_sizes[i])] for i in range(len(keep_sizes))]
            rel_chunks = [rels[i][: int(keep_sizes[i])] for i in range(len(keep_sizes)) if int(keep_sizes[i]) > 0]
            rs_chunks = [rel_swaps[i][: int(keep_sizes[i])] for i in range(len(keep_sizes)) if int(keep_sizes[i]) > 0]
            pos_mask_chunks = [
                batch_rel_pos_mask[i].index_select(0, kept_trimmed[i]).to(device)
                for i in range(len(keep_sizes))
                if int(keep_sizes[i]) > 0
            ]
            pair_hash_chunks = [
                batch_rel_pair_hashes[i].index_select(0, kept_trimmed[i]).to(device)
                for i in range(len(keep_sizes))
                if int(keep_sizes[i]) > 0
            ]
            pred_id_chunks = [
                batch_rel_pred_ids[i].index_select(0, kept_trimmed[i]).to(device)
                for i in range(len(keep_sizes))
                if int(keep_sizes[i]) > 0
            ]
            non_sym_chunks = [
                batch_rel_non_sym_mask[i].index_select(0, kept_trimmed[i]).to(device)
                for i in range(len(keep_sizes))
                if int(keep_sizes[i]) > 0
            ]
            rel_texts_all = [
                batch_rel_texts[i][int(j)]
                for i in range(len(keep_sizes))
                for j in kept_trimmed[i].tolist()
            ]
            cf_attr_texts_all = [
                batch_rel_cf_attr_texts[i][int(j)]
                for i in range(len(keep_sizes))
                for j in kept_trimmed[i].tolist()
            ]
            role_texts_all = [
                batch_rel_role_swap_texts[i][int(j)]
                for i in range(len(keep_sizes))
                for j in kept_trimmed[i].tolist()
            ]
            flat_r = torch.cat(rel_chunks, dim=0)
            flat_rs = torch.cat(rs_chunks, dim=0)
            flat_pos_mask = torch.cat(pos_mask_chunks, dim=0).bool()
            flat_pair_hashes = torch.cat(pair_hash_chunks, dim=0)
            flat_pred_ids = torch.cat(pred_id_chunks, dim=0)
            flat_non_sym_mask = torch.cat(non_sym_chunks, dim=0).bool()
            flat_img_ids = torch.repeat_interleave(torch.arange(len(keep_sizes), device=device), keep_sizes_t)
            pos_idx = torch.nonzero(flat_pos_mask, as_tuple=False).squeeze(1)
            if int(pos_idx.numel()) == 0:
                continue
            objective_uses_spoa = _objective_uses_spoa(train_objective_name)
            objective_uses_grounding = _objective_uses_grounding(train_objective_name)
            q_take = min(int(cfg.ground_num_queries), int(pos_idx.numel())) if objective_uses_grounding else 0
            q_texts: List[str] = []
            q_pred_ids_t = torch.empty((0,), device=device, dtype=torch.long)
            q_img_ids_t = torch.empty((0,), device=device, dtype=torch.long)
            if q_take > 0:
                q_pick = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:q_take]]
                q_pick_cpu = q_pick.detach().cpu().tolist()
                q_texts = [rel_texts_all[i] for i in q_pick_cpu]
                q_pred_ids_t = flat_pred_ids.index_select(0, q_pick)
                q_img_ids_t = flat_img_ids.index_select(0, q_pick)
            all_texts = (rel_texts_all + cf_attr_texts_all + role_texts_all if objective_uses_spoa else []) + q_texts
            all_emb = (
                variant_cache.get_batch(
                    all_texts,
                    fallback_fn=lambda missing: clip_text_features(clip_model, processor, missing, device),
                )
                if len(all_texts) > 0
                else torch.empty((0, int(cfg.emb_dim)), device=device)
            )
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(cfg.amp)):
                l_spoa_alignment = torch.tensor(0.0, device=device)
                l_dense_grounding = torch.tensor(0.0, device=device)
                n_spoa = 0
                n_ground = 0
                all_rel_text = []
                all_rel_pred_ids = []
                n_rel = len(rel_texts_all) if objective_uses_spoa else 0
                n_cf_attr = len(cf_attr_texts_all) if objective_uses_spoa else 0
                n_role = len(role_texts_all) if objective_uses_spoa else 0
                y_all = all_emb[:n_rel] if objective_uses_spoa else torch.empty((int(flat_r.shape[0]), int(cfg.emb_dim)), device=device)
                cf_attr_feats_all = all_emb[n_rel : n_rel + n_cf_attr] if n_cf_attr > 0 else torch.empty((0, int(cfg.emb_dim)), device=device)
                role_feats_all = all_emb[n_rel + n_cf_attr : n_rel + n_cf_attr + n_role] if n_role > 0 else torch.empty((0, int(cfg.emb_dim)), device=device)
                q_offset = n_rel + n_cf_attr + n_role
                q_feats = all_emb[q_offset:] if q_take > 0 else torch.empty((0, int(cfg.emb_dim)), device=device)
                x_pos = flat_r[flat_pos_mask]
                y_pos = y_all[flat_pos_mask] if objective_uses_spoa else torch.empty_like(x_pos)
                pos_pair_hashes = flat_pair_hashes[flat_pos_mask]
                pos_pred_ids = flat_pred_ids[flat_pos_mask]
                if objective_uses_spoa:
                    all_rel_text.append(y_pos.detach())
                    all_rel_pred_ids.append(pos_pred_ids.detach())
                if objective_uses_spoa and x_pos.numel() > 0 and y_pos.numel() > 0:
                    l_spoa_alignment = info_nce_with_queue(
                        x_pos,
                        y_pos,
                        rel_queue,
                        float(cfg.temp),
                        anchor_pred_ids=pos_pred_ids,
                        pred_sim_matrix=pred_sim_matrix,
                        lattice_min_weight=float(cfg.lattice_min_neg_weight),
                        min_queue_negatives=int(getattr(cfg, "rel_queue_min_negatives", 1024)),
                    )
                    n_spoa += 1
                if objective_uses_spoa:
                    hard_role_feats = flat_rs[flat_pos_mask]
                    pos_non_sym_mask = flat_non_sym_mask[flat_pos_mask]
                    cf_attr_feats = cf_attr_feats_all[flat_pos_mask] if int(cf_attr_feats_all.shape[0]) > 0 else torch.zeros_like(y_pos)
                    role_feats_pos = role_feats_all[flat_pos_mask] if int(role_feats_all.shape[0]) > 0 else torch.zeros_like(y_pos)
                    cf_role_text_feats = torch.zeros_like(y_pos)
                    cf_role_text_feats[pos_non_sym_mask] = role_feats_pos[pos_non_sym_mask]
                    role_swap_x = torch.zeros_like(hard_role_feats)
                    role_swap_x[pos_non_sym_mask] = hard_role_feats[pos_non_sym_mask]
                    cf_pred_feat = torch.zeros((int(pos_idx.numel()), int(cfg.emb_dim)), device=device)
                if objective_uses_spoa and int(pos_pred_ids.numel()) > 0:
                    safe_pred_ids = pos_pred_ids.clamp(min=0)
                    cand = confusable_idx.index_select(0, safe_pred_ids)
                    mask = cand != safe_pred_ids.unsqueeze(1)
                    mask_any = mask.any(dim=1)
                    first_idx = mask.float().argmax(dim=1)
                    row_idx = torch.arange(int(cand.shape[0]), device=device)
                    conf_ids = cand[row_idx, first_idx]
                    conf_ids = torch.where(mask_any, conf_ids, safe_pred_ids)
                    valid_conf = (pos_pred_ids >= 0) & (conf_ids >= 0) & (conf_ids < int(pred_emb_s2o.shape[0]))
                    if bool(valid_conf.any()):
                        cf_pred_feat[valid_conf] = pred_emb_s2o[conf_ids[valid_conf]].detach()
                if objective_uses_spoa and x_pos.numel() > 0:
                    if bool(getattr(cfg, "predicate_counterfactual_enabled", True)) and float(getattr(cfg, "lambda_counterfactual", 0.0)) > 0.0:
                        l_cf = counterfactual_hard_negative_loss(
                            x_pos,
                            y_pos,
                            role_swap_x=role_swap_x,
                            attr_swap_y=cf_attr_feats,
                            role_swap_y=cf_role_text_feats,
                            predicate_confusable_y=cf_pred_feat,
                            temp=float(cfg.temp),
                        )
                        l_spoa_alignment = l_spoa_alignment + (float(getattr(cfg, "lambda_counterfactual", 1.0)) * l_cf)
                if objective_uses_spoa and bool(getattr(cfg, "visual_hard_negative_enabled", True)) and float(getattr(cfg, "lambda_visual_hard_negative", 1.0)) > 0.0:
                    pool_r = ddp_all_gather_tensor(flat_r).detach()
                    pool_pair_hashes = ddp_all_gather_tensor(flat_pair_hashes).detach()
                    pool_pred_ids = ddp_all_gather_tensor(flat_pred_ids).detach()
                    vis_loss, vis_mined, vis_used = visual_counterfactual_inbatch_loss(
                        x_pos,
                        y_pos,
                        pos_pair_hashes,
                        pos_pred_ids,
                        pool_r,
                        pool_pair_hashes,
                        pool_pred_ids,
                        float(cfg.temp),
                        k_vis=8,
                    )
                    epoch_vis_batches += 1
                    epoch_vis_mined += int(vis_mined)
                    epoch_vis_used += int(vis_used)
                    if vis_used > 0:
                        epoch_vis_active_batches += 1
                        l_spoa_alignment = l_spoa_alignment + (float(getattr(cfg, "lambda_visual_hard_negative", 1.0)) * vis_loss)
                if objective_uses_grounding and q_take > 0 and q_feats.numel() > 0:
                    sim = out.score(flat_r, q_feats)
                    
                    same_pred = flat_pred_ids.unsqueeze(1) == q_pred_ids_t.unsqueeze(0)
                    same_img = flat_img_ids.unsqueeze(1) == q_img_ids_t.unsqueeze(0)
                    
                    sim = sim / float(cfg.temp)
                    sim = sim.masked_fill(~same_img, -10000.0)
                    
                    tgt = flat_pos_mask.unsqueeze(1) & same_pred & same_img
                    fallback = flat_pos_mask.unsqueeze(1) & same_img
                    tgt_sum = tgt.sum(dim=0, keepdim=True)
                    use_tgt = torch.where(tgt_sum > 0, tgt.float(), fallback.float())
                    
                    logp = F.log_softmax(sim, dim=0)
                    denom = use_tgt.sum(dim=0).clamp(min=1e-6)
                    l_g_ex = -((use_tgt * logp).sum(dim=0) / denom)
                    l_dense_grounding = l_g_ex.mean()
                    n_ground += 1
                if n_spoa > 0:
                    l_spoa_alignment = l_spoa_alignment / float(n_spoa)
                if n_ground > 0:
                    l_dense_grounding = l_dense_grounding / float(n_ground)
                l_predicate_ce = torch.tensor(0.0, device=device)
                train_objective = train_objective_name
                logit_adj_tau = float(getattr(cfg, "logit_adj_tau", 0.0))
                if float(getattr(cfg, "lambda_predicate_ce", 0.0)) > 0.0:
                    if bool(getattr(cfg, "predicate_classifier_enabled", True)) and hasattr(out, "predicate_logits"):
                        ce_feats = x_pos if bool(getattr(cfg, "predicate_ce_positive_only", True)) else flat_r
                        ce_targets = pos_pred_ids if bool(getattr(cfg, "predicate_ce_positive_only", True)) else flat_pred_ids
                        valid_ce = (ce_targets >= 0) & (ce_targets < int(pred_emb_s2o.shape[0]))
                        if bool(valid_ce.any()):
                            if bool(getattr(cfg, "adaptive_calibration_enabled", False)) and hasattr(out, "calibrated_predicate_logits"):
                                ce_logits = out.calibrated_predicate_logits(ce_feats[valid_ce], pred_log_prior).float()
                            else:
                                ce_logits = out.predicate_logits(ce_feats[valid_ce]).float()
                            if int(ce_logits.shape[-1]) == int(pred_emb_s2o.shape[0]):
                                if logit_adj_tau > 0.0 and int(pred_log_prior.numel()) == int(ce_logits.shape[-1]):
                                    ce_logits = ce_logits - (logit_adj_tau * pred_log_prior.view(1, -1))
                                l_predicate_ce = _predicate_ce_loss(ce_logits, ce_targets[valid_ce].long(), pred_ce_weights, cfg)
                            elif int(pos_pred_ids.numel()) > 0:
                                valid_pos_ce = (pos_pred_ids >= 0) & (pos_pred_ids < int(pred_emb_s2o.shape[0]))
                                if bool(valid_pos_ce.any()):
                                    ce_logits = out.score(x_pos[valid_pos_ce], pred_emb_s2o.detach()).float()
                                    if logit_adj_tau > 0.0 and int(pred_log_prior.numel()) == int(ce_logits.shape[-1]):
                                        ce_logits = ce_logits - (logit_adj_tau * pred_log_prior.view(1, -1))
                                    l_predicate_ce = _predicate_ce_loss(ce_logits, pos_pred_ids[valid_pos_ce].long(), pred_ce_weights, cfg)
                    elif int(pos_pred_ids.numel()) > 0:
                        valid_ce = (pos_pred_ids >= 0) & (pos_pred_ids < int(pred_emb_s2o.shape[0]))
                        if bool(valid_ce.any()):
                            ce_logits = out.score(x_pos[valid_ce], pred_emb_s2o.detach()).float()
                            if logit_adj_tau > 0.0 and int(pred_log_prior.numel()) == int(ce_logits.shape[-1]):
                                ce_logits = ce_logits - (logit_adj_tau * pred_log_prior.view(1, -1))
                            l_predicate_ce = _predicate_ce_loss(ce_logits, pos_pred_ids[valid_ce].long(), pred_ce_weights, cfg)
                if train_objective in {"predicate_warmup", "ce_only", "pred_ce"}:
                    spoa_term = l_spoa_alignment * 0.0
                    ground_term = l_dense_grounding * 0.0
                elif train_objective in {"no_spoa", "ground_ce"}:
                    spoa_term = l_spoa_alignment * 0.0
                    ground_term = float(cfg.lambda_dense_grounding) * l_dense_grounding
                else:
                    spoa_term = float(cfg.lambda_spoa_alignment) * l_spoa_alignment
                    ground_term = float(cfg.lambda_dense_grounding) * l_dense_grounding
                pred_ce_term = float(getattr(cfg, "lambda_predicate_ce", 0.0)) * l_predicate_ce
                gate_reg = None
                gate_val = 0.5
                decoder = model.module.decoder if isinstance(model, DDP) else model.decoder
                if hasattr(decoder, "last_gate_reg") and decoder.last_gate_reg is not None:
                    gate_reg = decoder.last_gate_reg
                    gate_val = decoder.last_gate_val
                loss_total = spoa_term + ground_term + pred_ce_term
                calib_reg_weight = float(getattr(cfg, "lambda_calibration_reg", 0.0))
                if bool(getattr(cfg, "adaptive_calibration_enabled", False)) and calib_reg_weight > 0.0 and hasattr(out, "calibration_regularizer"):
                    calib_reg = out.calibration_regularizer()
                    if calib_reg is not None:
                        loss_total = loss_total + (calib_reg_weight * calib_reg)
                gate_reg_weight = float(getattr(cfg, "gate_regularizer_weight", 0.01))
                if gate_reg is not None and gate_reg_weight > 0.0:
                    loss_total = loss_total + (gate_reg_weight * gate_reg)
                loss = loss_total / float(max(1, int(cfg.accum_steps)))
            _timing_sync(device, timing_enabled)
            forward_time = time.perf_counter() - step_compute_start
            if not bool(loss.requires_grad):
                last_step_end = time.perf_counter()
                continue
            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            _timing_sync(device, timing_enabled)
            backward_time = time.perf_counter() - backward_start
            optim_time = 0.0
            do_step = (((step + 1) % int(cfg.accum_steps)) == 0) or ((step + 1) == len(joint_loader))
            if do_step:
                optim_start = time.perf_counter()
                scaler.unscale_(optim)
                calib_clip_norm = float(getattr(cfg, "calibration_grad_clip_norm", 0.0))
                if calib_clip_norm > 0.0 and hasattr(unwrap_ddp(model), "calibration_parameters"):
                    torch.nn.utils.clip_grad_norm_(unwrap_ddp(model).calibration_parameters(), calib_clip_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                _timing_sync(device, timing_enabled)
                optim_time = time.perf_counter() - optim_start
                curr_lr_head = _lr_at_step(update_step)
                optim.param_groups[0]["lr"] = curr_lr_head          
                if len(optim.param_groups) > 1:
                    optim.param_groups[1]["lr"] = curr_lr_head * 0.1
                update_step += 1
            epoch_loss += float(loss_total.item())
            epoch_l_spoa += float(l_spoa_alignment.item())
            epoch_l_ground += float(l_dense_grounding.item())
            epoch_l_pred_ce += float(l_predicate_ce.item())
            epoch_batches += 1
            epoch_positive_pairs += int(pos_idx.numel())
            epoch_candidate_pairs += int(flat_r.shape[0])
            queue_start = time.perf_counter()
            with torch.no_grad():
                if len(all_rel_text) > 0:
                    queue_pred_ids = torch.cat(all_rel_pred_ids, dim=0) if len(all_rel_pred_ids) > 0 else None
                    rel_queue.add(torch.cat(all_rel_text, dim=0), queue_pred_ids)
            _timing_sync(device, timing_enabled)
            queue_time = time.perf_counter() - queue_start
            step_total_time = time.perf_counter() - step_wall_start
            last_step_end = time.perf_counter()
            if timing_enabled and step >= timing_warmup_steps:
                timing_acc["data_wait"] += float(data_wait_time)
                timing_acc["forward"] += float(forward_time)
                timing_acc["backward"] += float(backward_time)
                timing_acc["optim"] += float(optim_time)
                timing_acc["queue"] += float(queue_time)
                timing_acc["step_total"] += float(step_total_time)
                timing_acc["measured_steps"] += 1
            if is_main() and (step % int(cfg.log_every) == 0):
                # Log loss ratio and gate mean
                loss_ratio = float(l_dense_grounding.item()) / max(1e-6, float(l_spoa_alignment.item()))
                timing_msg = ""
                if timing_enabled and timing_acc["measured_steps"] > 0:
                    denom = float(timing_acc["measured_steps"])
                    timing_msg = (
                        f" | t_data={timing_acc['data_wait']/denom:.3f}s"
                        f" t_fwd={timing_acc['forward']/denom:.3f}s"
                        f" t_bwd={timing_acc['backward']/denom:.3f}s"
                        f" t_opt={timing_acc['optim']/denom:.3f}s"
                        f" t_q={timing_acc['queue']/denom:.3f}s"
                        f" t_step={timing_acc['step_total']/denom:.3f}s"
                    )
                print(
                    f"ep={epoch} st={step} loss={float(loss_total.item()):.3f} "
                    f"spoa={float(l_spoa_alignment.item()):.3f} ground={float(l_dense_grounding.item()):.3f} "
                    f"pred_ce={float(l_predicate_ce.item()):.3f} "
                    f"[ratio={loss_ratio:.2f}, gate={float(gate_val):.2f}] | lr={float(optim.param_groups[0]['lr']):.2e}"
                    f"{timing_msg}",
                    flush=True,
                )
            global_step += 1
        if is_main():
            os.makedirs(os.path.dirname(str(cfg.save_path)) or ".", exist_ok=True)
            
            ckpt = {
                "epoch": int(epoch),
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg.__dict__,
            }
            if not bool(cfg.freeze_clip):
                ckpt["clip"] = unwrap_ddp(clip_model).state_dict()
                
            torch.save(ckpt, str(cfg.save_path))
            print(f"[System] Saved lightweight checkpoint to {cfg.save_path}", flush=True)
            metrics_dump: Dict[str, Any] = {"epoch": int(epoch), "run_name": str(cfg.run_name), "seed": int(cfg.seed)}
            metrics_dump["active_branches"] = active_branch_report
            metrics_dump["train"] = {
                "avg_loss": float(epoch_loss / max(1, epoch_batches)),
                "avg_counterfactual_spoa": float(epoch_l_spoa / max(1, epoch_batches)),
                "avg_dense_grounding": float(epoch_l_ground / max(1, epoch_batches)),
                "avg_predce_logit_adjusted": float(epoch_l_pred_ce / max(1, epoch_batches)),
                "negative_pair_ratio": float(getattr(cfg, "negative_pair_ratio", -1.0)),
                "lambda_predicate_ce": float(getattr(cfg, "lambda_predicate_ce", 0.0)),
                "lambda_counterfactual_in_spoa": float(getattr(cfg, "lambda_counterfactual", 1.0)),
                "predicate_ce_loss": str(getattr(cfg, "predicate_ce_loss", "ce")),
                "predicate_ce_positive_only": bool(getattr(cfg, "predicate_ce_positive_only", True)),
                "predicate_sampler_enabled": bool(getattr(cfg, "predicate_sampler_enabled", False)),
                "predicate_sampler_power": float(getattr(cfg, "predicate_sampler_power", 0.0)),
                "ablation_visual_hard_negative_enabled": bool(getattr(cfg, "visual_hard_negative_enabled", False)),
                "ablation_lambda_visual_hard_negative": float(getattr(cfg, "lambda_visual_hard_negative", 0.0)),
                "predicate_counterfactual_enabled": bool(getattr(cfg, "predicate_counterfactual_enabled", True)),
                "gate_regularizer_weight": float(getattr(cfg, "gate_regularizer_weight", 0.01)),
                "train_objective": str(getattr(cfg, "train_objective", "full")),
                "logit_adj_tau": float(getattr(cfg, "logit_adj_tau", 0.0)),
                "pure_phase": str(getattr(cfg, "pure_phase", "core")),
                "object_language_anchor_enabled": bool(getattr(cfg, "object_language_anchor_enabled", False)),
                "object_language_anchor_source": str(getattr(cfg, "object_language_anchor_source", "auto")),
                "relation_context_layers": int(getattr(cfg, "relation_context_layers", 0)),
                "bayes_calibration_weight": float(getattr(cfg, "bayes_calibration_weight", 0.0)),
                "adaptive_calibration_enabled": bool(getattr(cfg, "adaptive_calibration_enabled", False)),
                "adaptive_prior_scale": float(getattr(cfg, "adaptive_prior_scale", 1.0)),
                "bias_residual_scale": float(getattr(cfg, "bias_residual_scale", 0.25)),
                "clip_unfreeze_after_epochs": int(getattr(cfg, "clip_unfreeze_after_epochs", 0)),
                "num_batches": int(epoch_batches),
                "positive_pairs": int(epoch_positive_pairs),
                "candidate_pairs": int(epoch_candidate_pairs),
                "avg_positive_pairs_per_batch": float(epoch_positive_pairs / max(1, epoch_batches)),
                "avg_candidate_pairs_per_batch": float(epoch_candidate_pairs / max(1, epoch_batches)),
                "ablation_visual_hard_mined": int(epoch_vis_mined),
                "ablation_visual_hard_used": int(epoch_vis_used),
                "ablation_visual_hard_batches": int(epoch_vis_batches),
                "ablation_visual_hard_active_batches": int(epoch_vis_active_batches),
                "ablation_visual_hard_active_batch_rate": float(epoch_vis_active_batches / max(1, epoch_vis_batches)),
                "effective_lr": float(optim.param_groups[0]["lr"]),
                "eval_split": str(eval_split),
                "eval_sgg_predicate_score_mode": str(getattr(cfg, "eval_sgg_predicate_score_mode", "unknown")),
                "eval_sgg_use_gt_pairs": bool(getattr(cfg, "eval_sgg_use_gt_pairs", False)),
                "prompt_aug_enabled": bool(use_prompt_aug),
                "eval_research_suite": bool(getattr(cfg, "eval_research_suite", False)),
            }
            if int(cfg.eval_every) > 0 and (((epoch + 1) % int(cfg.eval_every)) == 0):
                _run_core_evals(
                    args,
                    cfg,
                    metrics_dump,
                    model,
                    clip_model,
                    processor,
                    val_loader,
                    device,
                    pred_emb_s2o,
                    pred_to_idx,
                    pred_buckets,
                    canonicalizer,
                )
            if _should_run_research_eval(cfg, epoch):
                _run_research_evals(cfg, metrics_dump, model, clip_model, processor, val_loader, device)

            sgg_metrics = metrics_dump.get("val_sgg", {}) if isinstance(metrics_dump.get("val_sgg", {}), dict) else {}
            predcls_metrics = {}
            for predcls_key in ("predcls", "PredCls", "pred_cls", "predicate_classification"):
                candidate = sgg_metrics.get(predcls_key, {}) if isinstance(sgg_metrics, dict) else {}
                if isinstance(candidate, dict) and len(candidate) > 0:
                    predcls_metrics = candidate
                    break
            predcls_r50 = float(predcls_metrics.get("R@50", 0.0) or 0.0)
            predcls_mr50 = float(predcls_metrics.get("mR@50", 0.0) or 0.0)
            save_dir = os.path.dirname(str(cfg.save_path)) or "."
            save_stem = os.path.splitext(os.path.basename(str(cfg.save_path)))[0]
            if predcls_r50 > best_predcls_r50:
                best_predcls_r50 = predcls_r50
                best_path = os.path.join(save_dir, f"{save_stem}_best_R50.pt")
                torch.save(ckpt, best_path)
                print(f"[System] Saved best PredCls R@50 checkpoint to {best_path} ({predcls_r50:.4f})", flush=True)
            if predcls_mr50 > best_predcls_mr50:
                best_predcls_mr50 = predcls_mr50
                best_path = os.path.join(save_dir, f"{save_stem}_best_mR50.pt")
                torch.save(ckpt, best_path)
                print(f"[System] Saved best PredCls mR@50 checkpoint to {best_path} ({predcls_mr50:.4f})", flush=True)
            metrics_dump["best_so_far"] = {
                "PredCls_R@50": float(best_predcls_r50),
                "PredCls_mR@50": float(best_predcls_mr50),
            }

            if is_main():
                _print_epoch_summary(metrics_dump)
            out_dir = str(cfg.out_dir)
            os.makedirs(out_dir, exist_ok=True)
            path = str(cfg.save_metrics_json).strip()
            if path == "":
                path = os.path.join(out_dir, "metrics.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_dump) + "\n")
            per_epoch_dir = os.path.join(out_dir, "epoch_metrics")
            os.makedirs(per_epoch_dir, exist_ok=True)
            epoch_metrics_path = os.path.join(per_epoch_dir, f"epoch_{int(epoch):03d}.json")
            with open(epoch_metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dump, f, ensure_ascii=False, indent=2)
            latest_metrics_path = os.path.join(out_dir, "latest_metrics.json")
            with open(latest_metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dump, f, ensure_ascii=False, indent=2)
    if is_main() and isinstance(canonicalizer, PredicateCanonicalizer):
        canonicalizer.save_cache()
    if ddp_is_enabled():
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
