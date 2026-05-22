from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"

    run_name: str = "pure_onestage"
    out_dir: str = "runs/pure_onestage"
    save_metrics_json: str = ""
    stage: int = 0
    a100_ddp_preset: str = ""
    gpu_preset: str = ""

    # Cloud-native dataset source (Hugging Face Hub). Default to VG150 standard benchmark.
    hf_dataset_id: str = os.environ.get("HF_DATASET_ID", "anhkhoa1804/VG150-SGG-Standard")
    hf_train_split: str = os.environ.get("HF_TRAIN_SPLIT", "train")
    hf_val_split: str = os.environ.get("HF_VAL_SPLIT", "validation")
    hf_streaming: bool = False
    hf_shuffle_buffer_size: int = 20000
    samples_per_epoch: int = 0

    # VG150-specific settings (150 objects, 50 predicates).
    vg150_root: str = "datasets"
    vg150_enabled: bool = True
    vg150_source: str = "auto"  # auto, local, local-jsonl, local-files, hf-map, hf-streaming

    # Dataset construction.
    max_images: int = 0
    max_objects: int = 32
    max_pairs: int = 64
    use_all_pairs: bool = True
    negative_pair_ratio: float = 2.0  # 0 keeps positives only; positive values sample negatives per positive.
    box_noise: float = 0.0
    learned_prune_k: int = 0

    # Training.
    batch_size: int = 24
    num_workers: int = 8
    loader_prefetch_factor: int = 4
    loader_persistent_workers: bool = True
    epochs: int = 5
    accum_steps: int = 1
    lr: float = 2e-4
    lr_schedule: str = "cosine"
    train_objective: str = "full"  # full, predicate_warmup, no_spoa
    warmup_steps: int = 500
    warmup_epochs: int = 0
    weight_decay: float = 1e-2
    amp: bool = True
    amp_dtype: str = "bf16"
    grad_clip_norm: float = 1.0
    progressive_unfreeze: bool = False
    clip_unfreeze_after_epochs: int = 0
    channels_last: bool = True
    cudnn_benchmark: bool = True
    torch_compile: bool = False
    torch_compile_mode: str = "default"

    # PURE-next three pillars: PredCE-LA, Counterfactual-SPOA, Dense Grounding.
    lambda_spoa_alignment: float = 1.0
    lambda_dense_grounding: float = 1.0
    lambda_predicate_ce: float = 0.3
    lambda_counterfactual: float = 0.05
    lambda_visual_hard_negative: float = 0.0
    visual_hard_negative_enabled: bool = False
    predicate_counterfactual_enabled: bool = True
    gate_regularizer_weight: float = 0.01
    fusion_gate_temperature: float = 0.7
    predicate_classifier_enabled: bool = True
    predicate_classifier_classes: int = 51
    eval_sgg_use_predicate_classifier: bool = True
    eval_sgg_predicate_score_mode: str = "ensemble"  # ensemble, classifier, text, auto
    eval_sgg_predicate_ensemble_alpha: float = 0.5
    eval_sgg_classifier_temperature: float = 1.0
    eval_sgg_text_temperature: float = 1.0
    eval_sgg_predicate_diag_topk: int = 8
    eval_sgg_compare_score_modes: str = ""  # comma list, e.g. classifier,text,ensemble
    freq_bias_enabled: bool = False
    freq_bias_path: str = ""
    freq_bias_alpha: float = 0.5
    freq_bias_smoothing: float = 1.0
    predicate_background_weight: float = 0.1
    predicate_ce_loss: str = "focal"  # ce, weighted_ce, focal
    predicate_ce_gamma: float = 1.5
    predicate_ce_weight_power: float = 0.5
    predicate_ce_max_weight: float = 5.0
    predicate_ce_positive_only: bool = True
    predicate_sampler_enabled: bool = False
    predicate_sampler_power: float = 0.75
    predicate_sampler_max_weight: float = 20.0
    temp: float = 0.07
    ground_num_queries: int = 3
    rel_queue_size: int = 32768
    rel_queue_min_negatives: int = 1024
    lattice_loss_enabled: bool = True
    lattice_min_neg_weight: float = 0.05
    
    use_rfs: bool = True
    rfs_t: float = 0.001
    use_geom_bias: bool = True
    vector_fusion_gate: bool = True
    geom_fourier_dim: int = 256
    logit_adj_tau: float = 0.0
    eval_logit_adj_tau: float = -1.0  # negative means reuse logit_adj_tau for eval
    pure_phase: str = "core"  # core, scaling, eval
    object_language_anchor_enabled: bool = False
    object_language_anchor_source: str = "auto"  # auto, gt, clip_pred, none
    object_language_anchor_alpha: float = 0.35
    relation_context_layers: int = 0
    relation_context_heads: int = 8
    bayes_calibration_weight: float = 0.0
    adaptive_calibration_enabled: bool = False
    adaptive_prior_enabled: bool = True
    bias_residual_enabled: bool = True
    adaptive_prior_scale: float = 1.0
    bias_residual_scale: float = 0.25
    lambda_calibration_reg: float = 0.001

    # LoRA fine-tuning.
    unfreeze_clip_lora: bool = False

    # Model.
    emb_dim: int = 512
    model_name: str = "pure"
    pure_local_alpha: float = 0.7
    pure_bridge_layers: int = 4
    deformable_routing_enabled: bool = True
    deformable_num_points: int = 8
    deformable_offset_scale: float = 0.35
    progressive_node_layers: int = 2
    progressive_edge_layers: int = 2
    progressive_bilinear_layers: int = 0
    bilinear_low_rank: bool = False
    bilinear_rank: int = 32
    bilinear_residual_scale: float = 0.2
    clip_name: str = "openai/clip-vit-large-patch14-336"
    clip_input_res: int = 336
    gradient_checkpointing: bool = False
    freeze_clip: bool = True
    model_diagnostics_enabled: bool = False

    # FP8 is Hopper-only and disabled by default; A100 uses bf16/tf32.
    fp8_enabled: bool = False
    expandable_segments: bool = False
    fp8_recipe: str = "hybrid"

    # Text/prompt.
    prompt_aug: bool = True
    prompt_aug_prob: float = 0.5
    prompt_aug_max_paraphrases: int = 3
    canon_enabled: bool = False
    canon_model: str = "google/flan-t5-small"
    paraphrase_cache_path: str = ""

    # Eval.
    eval_every: int = 1
    eval_batches: int = 9999
    eval_fast_mode: bool = False
    eval_on_train_split: bool = False
    eval_sgg_iou_thresh: float = 0.5
    eval_sgg_use_gt_pairs: bool = False
    eval_sgg_use_clip_obj_classifier: bool = True
    eval_sgg_clip_obj_cache_enabled: bool = True
    eval_sgg_clip_obj_cache_dir: str = "runs/clip_obj_cache"
    eval_sgg_report_nograph: bool = True
    eval_sgg_use_no_interaction_prior: bool = False
    eval_sgg_no_interaction_text: str = "no relation or interaction between the objects"
    eval_sgg_grounding_dino_enabled: bool = True
    eval_sgg_grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    eval_sgg_grounding_dino_box_threshold: float = 0.25
    eval_sgg_grounding_dino_text_threshold: float = 0.25
    eval_sgg_grounding_dino_max_detections: int = 64
    eval_sgg_grounding_dino_cache_enabled: bool = True
    eval_sgg_grounding_dino_cache_dir: str = "runs/grounding_dino_cache"
    eval_research_suite: bool = False
    eval_latency: bool = False
    eval_latency_batches: int = 25
    eval_prune_batches: int = 50
    eval_prune_every: int = 1
    eval_prune_ks: str = "16,32,64,128"
    eval_paraphrase: bool = False
    paraphrase_n: int = 5
    eval_debug_grounding: bool = False
    eval_debug_grounding_max_samples: int = 3
    eval_sgg_debug_triplets: bool = False
    eval_sgg_debug_triplets_max_samples: int = 5
    eval_sgg_use_vg_aliases: bool = True
    eval_zs_predicates: str = ""

    # Logging/checkpoint.
    log_every: int = 50
    timing_breakdown: bool = False
    timing_warmup_steps: int = 5
    save_path: str = os.path.join("checkpoints", "pure_onestage.pt")
    resume: bool = False
    resume_from: str = ""
    reset_epoch: bool = False
    max_text_cache: int = 100_000


def resolve_loader_num_workers(num_workers: int, validation: bool = False) -> int:
    if validation:
        return max(0, int(num_workers // 2))
    return int(num_workers)


def resolve_loader_pin_memory(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def warn_if_low_shm(min_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
    shm_path = "/dev/shm"
    if os.name != "posix" or not os.path.isdir(shm_path):
        return
    try:
        stat = os.statvfs(shm_path)
    except OSError:
        return
    available_bytes = int(stat.f_bavail) * int(stat.f_frsize)
    if available_bytes < int(min_bytes):
        logging.warning(
            "Detected only %.2f GiB of /dev/shm; multi-worker loading may crash or stall.",
            available_bytes / float(1024**3),
        )


def apply_gpu_preset(cfg: TrainConfig, preset: str) -> TrainConfig:
    key = str(preset).strip().lower().replace("-", "_")
    if key in {"", "none", "off", "0"}:
        return cfg

    aliases = {
        "a100": "a100_80gb_balanced",
        "a100_80gb": "a100_80gb_balanced",
        "a100_balanced": "a100_80gb_balanced",
        "a100_throughput": "a100_80gb_throughput",
        "high": "a100_80gb_throughput",
        "high_80gb": "a100_80gb_throughput",
        "l4": "l4_24gb",
        "l4_24g": "l4_24gb",
        "l4_lowmem": "l4_22gb_lowmem",
        "l4_22gb": "l4_22gb_lowmem",
        "l4_22g": "l4_22gb_lowmem",
    }
    key = aliases.get(key, key)
    if key not in {"a100_80gb_balanced", "a100_80gb_throughput", "l4_24gb", "l4_22gb_lowmem"}:
        raise ValueError(
            f"Unsupported GPU preset: {preset}. "
            "Use a100_80gb_balanced, a100_80gb_throughput, high_80gb, l4_24gb, or l4_22gb_lowmem."
        )

    cfg.loader_persistent_workers = True
    cfg.channels_last = True
    cfg.cudnn_benchmark = True
    cfg.amp = True
    cfg.amp_dtype = "bf16"
    cfg.fp8_enabled = False

    if key == "a100_80gb_balanced":
        cfg.num_workers = max(int(cfg.num_workers), 8)
        cfg.loader_prefetch_factor = 4
        if int(cfg.stage) == 1:
            cfg.batch_size = 128
            cfg.gradient_checkpointing = False
        elif int(cfg.stage) == 2:
            cfg.batch_size = 16
            cfg.gradient_checkpointing = True
        elif int(cfg.stage) == 3:
            cfg.batch_size = 24
            cfg.gradient_checkpointing = True
    elif key == "a100_80gb_throughput":
        cfg.num_workers = max(int(cfg.num_workers), 10)
        cfg.loader_prefetch_factor = 4
        cfg.rel_queue_size = max(int(cfg.rel_queue_size), 32768)
        if int(cfg.stage) == 1:
            cfg.batch_size = 128
            cfg.gradient_checkpointing = False
        elif int(cfg.stage) == 2:
            cfg.batch_size = 16
            cfg.gradient_checkpointing = True
        elif int(cfg.stage) == 3:
            cfg.batch_size = 32
            cfg.gradient_checkpointing = True
    elif key == "l4_24gb":
        cfg.num_workers = min(max(int(cfg.num_workers), 4), 6)
        cfg.loader_prefetch_factor = 2
        cfg.expandable_segments = True
        if int(cfg.stage) == 1:
            cfg.batch_size = 64
            cfg.gradient_checkpointing = False
        elif int(cfg.stage) == 2:
            cfg.batch_size = 16
            cfg.gradient_checkpointing = True
        elif int(cfg.stage) == 3:
            cfg.batch_size = 12
            cfg.gradient_checkpointing = True
    elif key == "l4_22gb_lowmem":
        cfg.num_workers = min(max(int(cfg.num_workers), 2), 4)
        cfg.loader_prefetch_factor = 1
        cfg.loader_persistent_workers = False
        cfg.expandable_segments = True
        cfg.gradient_checkpointing = True
        cfg.max_pairs = min(int(cfg.max_pairs), 48)
        cfg.rel_queue_size = min(int(cfg.rel_queue_size), 8192)
        if int(cfg.stage) == 1:
            cfg.batch_size = 32
            cfg.clip_input_res = min(int(cfg.clip_input_res), 336)
            cfg.gradient_checkpointing = False
        elif int(cfg.stage) == 2:
            cfg.batch_size = 8
            cfg.clip_input_res = min(int(cfg.clip_input_res), 384)
        elif int(cfg.stage) == 3:
            cfg.batch_size = 6
            cfg.clip_input_res = min(int(cfg.clip_input_res), 384)


    return cfg


def apply_stage_config(cfg):
    """Apply staged curriculum presets for stable PURE training."""
    if int(cfg.stage) == 1:
        cfg.clip_input_res = 336
        cfg.freeze_clip = True
        cfg.progressive_unfreeze = False
        cfg.clip_unfreeze_after_epochs = 0
        cfg.gradient_checkpointing = False
        cfg.lr = 2e-4
        cfg.batch_size = 128
        cfg.epochs = 2
        cfg.num_workers = 8
        cfg.max_pairs = min(int(cfg.max_pairs), 48)
        cfg.negative_pair_ratio = 1.0
        cfg.progressive_bilinear_layers = 0
        cfg.lambda_spoa_alignment = 0.0
        cfg.lambda_dense_grounding = 0.25
        cfg.lambda_predicate_ce = 2.0
        cfg.lambda_counterfactual = 0.0
        cfg.lambda_visual_hard_negative = 0.0
        cfg.visual_hard_negative_enabled = False
        cfg.predicate_counterfactual_enabled = False
        cfg.gate_regularizer_weight = 0.0
        cfg.predicate_ce_loss = "focal"
        cfg.predicate_ce_gamma = 1.5
        cfg.predicate_ce_weight_power = 0.5
        cfg.predicate_ce_max_weight = 5.0
        cfg.ground_num_queries = 1
        cfg.temp = 0.1
        cfg.rel_queue_size = 8192
        cfg.predicate_classifier_enabled = True
        cfg.eval_sgg_use_predicate_classifier = True
        cfg.prompt_aug = False
        cfg.prompt_aug_prob = 0.0
        cfg.prompt_aug_max_paraphrases = 1
        cfg.eval_research_suite = False
        cfg.eval_latency = False
        cfg.eval_fast_mode = False
        cfg.eval_batches = int(cfg.eval_batches)
        cfg.eval_every = max(1, int(cfg.eval_every))
        cfg.eval_prune_every = max(2, int(cfg.eval_prune_every))
        cfg.fp8_enabled = False

    elif int(cfg.stage) == 2:
        cfg.clip_input_res = 448
        cfg.freeze_clip = False
        cfg.progressive_unfreeze = True
        cfg.clip_unfreeze_after_epochs = 1
        cfg.lr = 1e-4
        cfg.learned_prune_k = 64
        cfg.gradient_checkpointing = True
        cfg.batch_size = 16
        cfg.epochs = 8
        cfg.num_workers = 8
        cfg.negative_pair_ratio = 1.5
        cfg.progressive_bilinear_layers = 0
        cfg.lambda_spoa_alignment = 0.5
        cfg.lambda_dense_grounding = 0.25
        cfg.lambda_predicate_ce = 1.5
        cfg.lambda_counterfactual = 0.0
        cfg.lambda_visual_hard_negative = 0.25
        cfg.visual_hard_negative_enabled = True
        cfg.predicate_counterfactual_enabled = False
        cfg.gate_regularizer_weight = 0.01
        cfg.predicate_ce_loss = "focal"
        cfg.predicate_ce_gamma = 1.5
        cfg.predicate_ce_weight_power = 0.5
        cfg.predicate_ce_max_weight = 5.0
        cfg.ground_num_queries = 1
        cfg.temp = 0.07
        cfg.rel_queue_size = 8192
        cfg.predicate_classifier_enabled = True
        cfg.eval_sgg_use_predicate_classifier = True
        cfg.prompt_aug = False
        cfg.prompt_aug_prob = 0.0
        cfg.prompt_aug_max_paraphrases = 1
        cfg.eval_research_suite = False
        cfg.eval_fast_mode = False
        cfg.eval_batches = int(cfg.eval_batches)
        cfg.fp8_enabled = False

    elif int(cfg.stage) == 3:
        cfg.clip_input_res = 448
        cfg.freeze_clip = False
        cfg.progressive_unfreeze = False
        cfg.clip_unfreeze_after_epochs = 0
        cfg.gradient_checkpointing = True
        cfg.lr = 2e-5
        cfg.batch_size = 16
        cfg.num_workers = 8
        cfg.learned_prune_k = 64
        cfg.negative_pair_ratio = 2.0
        cfg.progressive_bilinear_layers = 0
        cfg.lambda_spoa_alignment = 0.75
        cfg.lambda_dense_grounding = 0.25
        cfg.lambda_predicate_ce = 1.2
        cfg.lambda_counterfactual = 0.05
        cfg.lambda_visual_hard_negative = 0.0
        cfg.visual_hard_negative_enabled = False
        cfg.predicate_counterfactual_enabled = True
        cfg.gate_regularizer_weight = 0.01
        cfg.predicate_ce_loss = "focal"
        cfg.predicate_ce_gamma = 1.0
        cfg.predicate_ce_weight_power = 0.5
        cfg.predicate_ce_max_weight = 5.0
        cfg.ground_num_queries = 1
        cfg.temp = 0.07
        cfg.rel_queue_size = 8192
        cfg.predicate_classifier_enabled = True
        cfg.eval_sgg_use_predicate_classifier = True
        cfg.prompt_aug_max_paraphrases = 1
        cfg.eval_research_suite = False
        cfg.fp8_enabled = False

    return cfg


def apply_ddp_a100_preset(cfg: TrainConfig, preset: str) -> TrainConfig:
    key = str(preset).strip().lower()
    if key in {"", "none", "off", "0"}:
        return cfg

    if key not in {"4x", "8x"}:
        raise ValueError(f"Unsupported A100 DDP preset: {preset}")

    cfg.loader_persistent_workers = True
    cfg.loader_prefetch_factor = 4
    cfg.channels_last = True
    cfg.cudnn_benchmark = True
    cfg.amp = True
    cfg.amp_dtype = "bf16"
    cfg.fp8_enabled = False

    if key == "4x":
        cfg.num_workers = max(int(cfg.num_workers), 8)
        if int(cfg.stage) == 1:
            cfg.batch_size = 32
        elif int(cfg.stage) == 2:
            cfg.batch_size = 16
            cfg.gradient_checkpointing = True
        elif int(cfg.stage) == 3:
            cfg.batch_size = 12
            cfg.gradient_checkpointing = True
    elif key == "8x":
        cfg.num_workers = max(int(cfg.num_workers), 6)
        if int(cfg.stage) == 1:
            cfg.batch_size = 24
        elif int(cfg.stage) == 2:
            cfg.batch_size = 12
            cfg.gradient_checkpointing = True
        elif int(cfg.stage) == 3:
            cfg.batch_size = 8
            cfg.gradient_checkpointing = True

    return cfg
