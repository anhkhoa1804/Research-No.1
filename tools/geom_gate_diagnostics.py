"""Empirical diagnostic: what does decoder.fusion_gate (the vector geometry
gate) actually output on real val-split pairs, using the historical checkpoint?

PURELY OBSERVATIONAL. Wraps ProgressiveRelationalDecoder.forward_pairs to
snapshot `self.last_vector_gate` after every call (this attribute already
exists and is populated whenever `capture_diagnostics=True`, which
`eval_sgg_routing_diag_enabled` -- default True -- turns on inside
_forward_eval_batch; this tool changes no forward-pass numerics). Runs the
same eval entry point (`openvocab_rel.train.main`) used to build the p36
rel_feat cache, bounded to a small number of batches via --eval_batches, and
writes gate statistics to a JSON result file. No historical artifact is
touched; output goes under runs/.

geom_alpha is not exercised (use_geom_bias=True, vector_fusion_gate=True per
the checkpoint's own saved cfg dict), so it is not measured here -- see
docs/GEOM_GATE_DIAGNOSTIC_RESULT.md for that source-level finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openvocab_rel.models.relational_model import ProgressiveRelationalDecoder  # noqa: E402
from openvocab_rel import geometry as geometry_mod  # noqa: E402

_GATE_RECORDS: List[Dict[str, Any]] = []
_GEOM_RECORDS: List[torch.Tensor] = []

_orig_forward_pairs = ProgressiveRelationalDecoder.forward_pairs
_orig_geom_feats_torch = geometry_mod.geom_feats_torch


def _patched_forward_pairs(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw):
    out = _orig_forward_pairs(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw)
    gate = getattr(self, "last_vector_gate", None)
    if isinstance(gate, torch.Tensor) and gate.numel() > 0:
        per_pair_mean = gate.mean(dim=-1)
        _GATE_RECORDS.append(
            {
                "n_pairs": int(gate.shape[0]),
                "dim": int(gate.shape[-1]),
                "per_pair_mean": per_pair_mean.cpu().tolist(),
                "flat_mean": float(gate.mean().item()),
                "flat_std": float(gate.std().item()),
                "frac_near_geom": float((gate < 0.1).float().mean().item()),
                "frac_near_sem": float((gate > 0.9).float().mean().item()),
                "geom_alpha_value": float(self.geom_alpha.detach().mean().item())
                if hasattr(self, "geom_alpha")
                else None,
            }
        )
    return out


def _patched_geom_feats_torch(b1, b2):
    feat = _orig_geom_feats_torch(b1, b2)
    _GEOM_RECORDS.append(feat.detach().float().cpu())
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_batches", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--out_dir", type=str, default="runs/p68_geom_gate_pilot")
    ap.add_argument("--run_name", type=str, default="p68_geom_gate_pilot")
    ap.add_argument("--split", type=str, default="val")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ProgressiveRelationalDecoder.forward_pairs = _patched_forward_pairs
    geometry_mod.geom_feats_torch = _patched_geom_feats_torch
    import openvocab_rel.models.relational_model as rel_mod
    rel_mod.geom_feats_torch = _patched_geom_feats_torch

    from openvocab_rel import train as train_mod

    argv = [
        "--stage", "3",
        "--gpu_preset", "l4_24gb",
        "--eval_only", "true",
        "--epochs", "0",
        "--resume", "true",
        "--resume_from", "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
        "--vg150_root", "datasets_vg150_clean",
        "--vg150_enabled", "true",
        "--vg150_source", "local-jsonl",
        "--device", "cuda",
        "--batch_size", str(args.batch_size),
        "--num_workers", "4",
        "--clip_input_res", "336",
        "--eval_batches", str(args.eval_batches),
        "--eval_fast_mode", "false",
        "--explicit_spoa_enabled", "false",
        "--text_conditioned_projection_enabled", "false",
        "--relationness_enabled", "false",
        "--eval_sgg_use_relationness", "false",
        "--eval_sgg_predicate_score_mode", "ensemble",
        "--eval_sgg_predicate_ensemble_alpha", "0.0",
        "--adaptive_calibration_enabled", "true",
        "--bayes_calibration_weight", "0.0",
        "--freq_bias_enabled", "true",
        "--freq_bias_path", "datasets_vg150_clean/frequency_prior_train.json",
        "--freq_bias_alpha", "3.75",
        "--freq_bias_smoothing", "1.0",
        "--eval_sgg_use_gt_pairs", "true",
        "--eval_sgg_grounding_dino_enabled", "false",
        "--run_name", args.run_name,
        "--out_dir", str(out_dir),
        "--save_metrics_json", str(out_dir / "metrics.jsonl"),
    ]
    try:
        # train.main() ends eval-only mode with sys.exit(0) (train.py:1806);
        # swallow it so the captured gate statistics can still be written.
        train_mod.main(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            raise
    finally:
        ProgressiveRelationalDecoder.forward_pairs = _orig_forward_pairs
        geometry_mod.geom_feats_torch = _orig_geom_feats_torch
        rel_mod.geom_feats_torch = _orig_geom_feats_torch

    all_flat_means = [r["flat_mean"] for r in _GATE_RECORDS]
    all_pair_means: List[float] = []
    for r in _GATE_RECORDS:
        all_pair_means.extend(r["per_pair_mean"])
    all_pair_means_arr = np.array(all_pair_means, dtype=np.float64)

    geom_cat = torch.cat(_GEOM_RECORDS, dim=0) if _GEOM_RECORDS else torch.zeros((0, 8))
    geom_np = geom_cat.numpy()
    dx = geom_np[:, 0] if geom_np.shape[0] else np.array([])
    dy = geom_np[:, 1] if geom_np.shape[0] else np.array([])
    geom_mag = np.sqrt(dx ** 2 + dy ** 2) if geom_np.shape[0] else np.array([])

    # Per-column diagnostics of the geometry vector the model ACTUALLY receives
    # (captured in-situ from the live forward pass, after the /img_res
    # normalisation at relational_model.py:725).
    geom_cols = ["dx", "dy", "rw", "rh", "ar1", "ar2", "a1", "a2"]
    geom_col_stats = {}
    for i, name in enumerate(geom_cols):
        if geom_np.shape[0] == 0:
            break
        col = geom_np[:, i]
        geom_col_stats[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "frac_exactly_zero": float((col == 0.0).mean()),
            "n_distinct_approx": int(len(np.unique(np.round(col, 6)))),
        }

    n_pairs_total = int(sum(r["n_pairs"] for r in _GATE_RECORDS))
    corr_gate_vs_geom_mag = None
    if len(all_pair_means_arr) == len(geom_mag) and len(geom_mag) > 1:
        corr_gate_vs_geom_mag = float(np.corrcoef(all_pair_means_arr, geom_mag)[0, 1])

    result = {
        "n_calls": len(_GATE_RECORDS),
        "n_pairs_total": n_pairs_total,
        "geom_alpha_value": _GATE_RECORDS[0]["geom_alpha_value"] if _GATE_RECORDS else None,
        "gate_flat_mean_over_calls": float(np.mean(all_flat_means)) if all_flat_means else None,
        "gate_per_pair_mean_of_means": float(all_pair_means_arr.mean()) if len(all_pair_means_arr) else None,
        "gate_per_pair_std_of_means": float(all_pair_means_arr.std()) if len(all_pair_means_arr) else None,
        "gate_per_pair_min": float(all_pair_means_arr.min()) if len(all_pair_means_arr) else None,
        "gate_per_pair_max": float(all_pair_means_arr.max()) if len(all_pair_means_arr) else None,
        "frac_near_geom_mean": float(np.mean([r["frac_near_geom"] for r in _GATE_RECORDS])) if _GATE_RECORDS else None,
        "frac_near_sem_mean": float(np.mean([r["frac_near_sem"] for r in _GATE_RECORDS])) if _GATE_RECORDS else None,
        "corr_gate_mean_vs_geom_magnitude": corr_gate_vs_geom_mag,
        "geom_column_stats_as_model_receives_them": geom_col_stats,
        "note": (
            "gate=1 means fused_feat leans fully to sem_feat (geometry dropped); "
            "gate=0 means fused_feat leans fully to geom_norm (semantics dropped). "
            "per_pair_std_of_means near 0 would indicate the gate is not "
            "input-dependent in practice, despite being architecturally capable of it."
        ),
    }
    with open(out_dir / "gate_diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
