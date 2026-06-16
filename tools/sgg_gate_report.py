#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from model_report_card import _as_float, _load_json_or_jsonl


def _fmt(value: Any) -> str:
    return f"{_as_float(value, 0.0):.4f}"


def _metric_block(row: Dict[str, Any], task: str) -> Dict[str, Any]:
    val = row.get("val_sgg") if isinstance(row.get("val_sgg"), dict) else {}
    block = val.get(task) if isinstance(val.get(task), dict) else {}
    return block


def _top_pairs(rows: List[Dict[str, Any]], key: str, limit: int) -> str:
    out = []
    for row in rows[:limit]:
        out.append(f"{row.get('gt')}->{row.get('pred')}:{row.get('count')}")
    return ", ".join(out) if out else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PURE SGG gate diagnostics from metrics files.")
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics.jsonl/json files")
    parser.add_argument("--top", type=int, default=8, help="number of confusion rows to print")
    parser.add_argument("--all_rows", action="store_true", help="show every metrics row instead of the latest row per run/mode/alpha")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    summaries: List[Dict[str, Any]] = []
    for path in args.metrics:
        for row in _load_json_or_jsonl(path):
            val = row.get("val_sgg") if isinstance(row.get("val_sgg"), dict) else {}
            cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
            pair = val.get("pair_rank_diag") if isinstance(val.get("pair_rank_diag"), dict) else {}
            obj = val.get("object_diag") if isinstance(val.get("object_diag"), dict) else {}
            predcls = _metric_block(row, "predcls")
            allpairs = predcls if not bool(cfg.get("eval_sgg_use_gt_pairs", False)) else {}
            summaries.append(
                {
                    "file": str(path),
                    "run_name": row.get("run_name", path.parent.name),
                    "epoch": row.get("epoch", "?"),
                    "score_mode": cfg.get("eval_sgg_predicate_score_mode", val.get("settings", {}).get("score_mode", "")),
                    "prune_score_mode": cfg.get("eval_sgg_prune_score_mode", val.get("settings", {}).get("prune_score_mode", "relationness")),
                    "pair_score_mode": cfg.get("eval_sgg_pair_score_mode", val.get("settings", {}).get("pair_score_mode", "relationness")),
                    "alpha": _as_float(cfg.get("bayes_calibration_weight", val.get("settings", {}).get("bayes_calibration_weight", 0.0)), 0.0),
                    "gt_pairs": bool(cfg.get("eval_sgg_use_gt_pairs", False)),
                    "predcls_R50": _as_float(predcls.get("R@50", 0.0), 0.0),
                    "predcls_mR50": _as_float(predcls.get("mR@50", 0.0), 0.0),
                    "predcls_tail": _as_float(predcls.get("tail_mR@50", 0.0), 0.0),
                    "pair_rank_n": int(pair.get("n", 0) or 0),
                    "pair_mean_rank": _as_float(pair.get("mean_gt_predicate_rank_on_pair", 0.0), 0.0),
                    "pair_top1": _as_float(pair.get("top1", 0.0), 0.0),
                    "pair_top5": _as_float(pair.get("top5", 0.0), 0.0),
                    "pair_top10": _as_float(pair.get("top10", 0.0), 0.0),
                    "pair_top50": _as_float(pair.get("top50", 0.0), 0.0),
                    "pair_pruned_rate": _as_float(pair.get("pruned_pair_rate", 0.0), 0.0),
                    "pair_missing_rate": _as_float(pair.get("missing_pair_rate", 0.0), 0.0),
                    "pair_relationness": _as_float(pair.get("mean_gt_pair_relationness", 0.0), 0.0),
                    "clip_top1_object_acc": _as_float(obj.get("clip_top1_object_acc", 0.0), 0.0),
                    "clip_topk_object_acc": _as_float(obj.get("clip_topk_object_acc", 0.0), 0.0),
                    "triplet_endpoint_topk_coverage": _as_float(obj.get("triplet_endpoint_topk_coverage", 0.0), 0.0),
                    "top1_confusions": pair.get("top1_confusions", []),
                    "miss_confusions": pair.get("miss_confusions", []),
                    "allpairs_R50": _as_float(allpairs.get("R@50", 0.0), 0.0),
                    "allpairs_mR50": _as_float(allpairs.get("mR@50", 0.0), 0.0),
                }
            )

    if not args.all_rows:
        latest = {}
        for row in summaries:
            key = (row.get("run_name"), row.get("score_mode"), row.get("alpha"), row.get("gt_pairs"))
            latest[key] = row
        summaries = list(latest.values())

    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    print("SGG gate report")
    header = f"{'run':<36} {'mode':<10} {'prune':<10} {'pairscore':<10} {'alpha':>6} {'gt':>3} {'R50':>7} {'mR50':>7} {'tail':>7} {'pair@1':>7} {'pair@5':>7} {'pair@50':>7} {'drop':>7} {'obj@1':>7} {'obj@k':>7}"
    print(header)
    for row in summaries:
        print(
            f"{str(row['run_name'])[:36]:<36} {str(row['score_mode'])[:10]:<10} {str(row['prune_score_mode'])[:10]:<10} {str(row['pair_score_mode'])[:10]:<10} {row['alpha']:>6.2f} {str(row['gt_pairs']):>3} "
            f"{row['predcls_R50']:>7.4f} {row['predcls_mR50']:>7.4f} {row['predcls_tail']:>7.4f} "
            f"{row['pair_top1']:>7.4f} {row['pair_top5']:>7.4f} {row['pair_top50']:>7.4f} {row['pair_pruned_rate']:>7.4f} "
            f"{row['clip_top1_object_acc']:>7.4f} {row['clip_topk_object_acc']:>7.4f}"
        )
        if row["top1_confusions"]:
            print(f"  top1 confusions: {_top_pairs(row['top1_confusions'], 'count', args.top)}")
        if row["miss_confusions"]:
            print(f"  miss confusions: {_top_pairs(row['miss_confusions'], 'count', args.top)}")


if __name__ == "__main__":
    main()