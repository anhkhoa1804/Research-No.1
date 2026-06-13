#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MetricRow = Dict[str, Any]
MetricBlock = Dict[str, Any]


R_KEYS = ("R@50", "recall@50", "r50", "R50")
MR_KEYS = ("mR@50", "mean_recall@50", "mr50", "mR50")
MODE_ALIASES = {
    "raw": ("raw", "classifier", "classifier_only", "pred", "predicate", "logits"),
    "text": ("text", "clip_text", "text_only"),
    "ensemble": ("ensemble", "ens", "hybrid"),
    "calibrated": ("calibrated", "calibration", "fixed_prior", "adaptive", "bayes"),
}


def _get(data: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _first_float(block: MetricBlock, keys: Sequence[str]) -> float:
    for key in keys:
        if key in block:
            return _as_float(block.get(key), 0.0)
    return 0.0


def _load_json_or_jsonl(path: Path) -> List[MetricRow]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("metrics"), list):
            return [x for x in data["metrics"] if isinstance(x, dict)]
        return [data]
    return []


def _expand_metric_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = [Path(x) for x in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    seen = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _normalize_predicate(value: Any) -> str:
    return str(value).strip().lower()


def _extract_relationship_predicate(rel: Any) -> str:
    if not isinstance(rel, dict):
        return ""
    return _normalize_predicate(rel.get("predicate", rel.get("pred", rel.get("relation", ""))))


def _load_counts_from_train_jsonl(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rels = row.get("relationships", row.get("relations", row.get("rels", []))) if isinstance(row, dict) else []
            for rel in rels if isinstance(rels, list) else []:
                pred = _extract_relationship_predicate(rel)
                if pred:
                    counts[pred] += 1
    return counts


def _load_counts_from_frequency_file(path: Path) -> Counter:
    counts: Counter = Counter()
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                pred = _normalize_predicate(row.get("predicate", row.get("label", row.get("name", ""))))
                count = _as_float(row.get("count", row.get("frequency", row.get("freq", 0))), 0.0)
                if pred:
                    counts[pred] += int(count)
        return counts

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("predicate_counts", "counts", "global_counts", "frequencies"):
            value = data.get(key)
            if isinstance(value, dict):
                for pred, count in value.items():
                    pred_norm = _normalize_predicate(pred)
                    if pred_norm:
                        counts[pred_norm] += int(_as_float(count, 0.0))
                return counts
        vocab = data.get("predicate_vocab")
        probs = data.get("global_log_probs")
        if isinstance(vocab, list) and isinstance(probs, list) and len(vocab) == len(probs):
            for pred, log_prob in zip(vocab, probs):
                pred_norm = _normalize_predicate(pred)
                if pred_norm:
                    counts[pred_norm] += max(1, int(round(math.exp(_as_float(log_prob, -20.0)) * 1_000_000)))
            return counts
        for pred, count in data.items():
            if isinstance(count, (int, float)):
                pred_norm = _normalize_predicate(pred)
                if pred_norm:
                    counts[pred_norm] += int(count)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                pred = _normalize_predicate(item.get("predicate", item.get("label", item.get("name", ""))))
                count = _as_float(item.get("count", item.get("frequency", item.get("freq", 0))), 0.0)
                if pred:
                    counts[pred] += int(count)
    return counts


def _build_buckets(counts: Counter, head_frac: float, tail_frac: float) -> Dict[str, str]:
    items = sorted(((pred, int(count)) for pred, count in counts.items()), key=lambda x: (-x[1], x[0]))
    if not items:
        return {}
    n = len(items)
    head_n = max(1, int(round(n * float(head_frac))))
    tail_n = max(1, int(round(n * float(tail_frac))))
    buckets: Dict[str, str] = {}
    for idx, (pred, _count) in enumerate(items):
        if idx < head_n:
            buckets[pred] = "head"
        elif idx >= n - tail_n:
            buckets[pred] = "tail"
        else:
            buckets[pred] = "body"
    return buckets


def _metric_block_from_candidate(candidate: Any) -> Optional[MetricBlock]:
    if not isinstance(candidate, dict):
        return None
    for key in ("predcls", "PredCls", "pred_cls", "predicate_classification"):
        value = candidate.get(key)
        if isinstance(value, dict):
            return value
    if any(key in candidate for key in R_KEYS + MR_KEYS):
        return candidate
    return None


def _score_modes(row: MetricRow) -> Dict[str, MetricBlock]:
    modes: Dict[str, MetricBlock] = {}
    val_sgg = row.get("val_sgg") if isinstance(row.get("val_sgg"), dict) else {}
    base = _metric_block_from_candidate(val_sgg)
    if base:
        modes["default"] = base
    for container_key in ("score_modes", "score_mode_metrics"):
        container = val_sgg.get(container_key) if isinstance(val_sgg, dict) else None
        if isinstance(container, dict):
            for mode, value in container.items():
                block = _metric_block_from_candidate(value)
                if block:
                    modes[str(mode)] = block
    by_mode = row.get("val_sgg_by_score_mode")
    if isinstance(by_mode, dict):
        for mode, value in by_mode.items():
            block = _metric_block_from_candidate(value)
            if block:
                modes[str(mode)] = block
    return modes


def _mode_family(mode: str) -> str:
    lower = mode.lower()
    for family, aliases in MODE_ALIASES.items():
        if any(alias in lower for alias in aliases):
            return family
    return "default" if lower == "default" else lower


def _combined_score(block: MetricBlock, lambda_mr: float, mu_tail: float) -> float:
    r50 = _first_float(block, R_KEYS)
    mr50 = _first_float(block, MR_KEYS)
    tail = _as_float(block.get("tail_mR@50", block.get("tail_mr50", 0.0)), 0.0)
    return r50 + float(lambda_mr) * mr50 + float(mu_tail) * tail


def _per_predicate_recall(block: MetricBlock) -> Dict[str, float]:
    candidates = [
        block.get("per_predicate_R@50"),
        block.get("per_predicate_recall@50"),
        block.get("predicate_recall@50"),
        block.get("per_predicate", {}).get("R@50") if isinstance(block.get("per_predicate"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, dict):
            return {_normalize_predicate(k): _as_float(v, 0.0) for k, v in value.items()}
        if isinstance(value, list):
            out: Dict[str, float] = {}
            for item in value:
                if isinstance(item, dict):
                    pred = _normalize_predicate(item.get("predicate", item.get("label", "")))
                    recall = _as_float(item.get("R@50", item.get("recall", item.get("value", 0.0))), 0.0)
                    if pred:
                        out[pred] = recall
            if out:
                return out
    return {}


def _bucket_metrics(per_pred: Dict[str, float], buckets: Dict[str, str]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {"head": [], "body": [], "tail": []}
    for pred, recall in per_pred.items():
        bucket = buckets.get(_normalize_predicate(pred))
        if bucket in grouped:
            grouped[bucket].append(float(recall))
    return {f"{bucket}_mR@50": (sum(vals) / len(vals) if vals else 0.0) for bucket, vals in grouped.items()}


def _row_id(row: MetricRow, path: Path) -> str:
    run = str(row.get("run_name", path.parent.name or path.stem))
    epoch = row.get("epoch", "?")
    return f"{run}:epoch={epoch}"


def _summaries(rows_by_file: List[Tuple[Path, MetricRow]], buckets: Dict[str, str], lambda_mr: float, mu_tail: float) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path, row in rows_by_file:
        modes = _score_modes(row)
        for mode, block in modes.items():
            per_pred = _per_predicate_recall(block)
            bucket_values = _bucket_metrics(per_pred, buckets) if per_pred and buckets else {}
            enriched = dict(block)
            enriched.update({k: v for k, v in bucket_values.items() if k not in enriched})
            role_diag = row.get("role_swap_diag", {}) if isinstance(row.get("role_swap_diag", {}), dict) else {}
            if not role_diag:
                role_diag = _get(row, ["val_sgg", "role_swap_diag"], {}) or {}
            summaries.append(
                {
                    "file": str(path),
                    "id": _row_id(row, path),
                    "epoch": row.get("epoch"),
                    "run_name": row.get("run_name", path.parent.name or path.stem),
                    "mode": mode,
                    "family": _mode_family(mode),
                    "R@50": _first_float(enriched, R_KEYS),
                    "mR@50": _first_float(enriched, MR_KEYS),
                    "tail_mR@50": _as_float(enriched.get("tail_mR@50", enriched.get("tail_mr50", 0.0)), 0.0),
                    "head_mR@50": _as_float(enriched.get("head_mR@50", enriched.get("head_mr50", 0.0)), 0.0),
                    "body_mR@50": _as_float(enriched.get("body_mR@50", enriched.get("body_mr50", 0.0)), 0.0),
                    "combined": _combined_score(enriched, lambda_mr, mu_tail),
                    "role_swap_consistency": _as_float(role_diag.get("consistency", 0.0), 0.0) if isinstance(role_diag, dict) else 0.0,
                    "role_swap_margin_mean": _as_float(role_diag.get("margin_mean", 0.0), 0.0) if isinstance(role_diag, dict) else 0.0,
                    "role_swap_failure_rate": _as_float(role_diag.get("failure_rate", 0.0), 0.0) if isinstance(role_diag, dict) else 0.0,
                    "per_predicate_R@50": per_pred,
                    "config": row.get("config", row.get("train_config", row.get("cfg", {}))),
                }
            )
    return summaries


def _best(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    valid = [row for row in rows if math.isfinite(_as_float(row.get(key), float("nan")))]
    if not valid:
        return None
    return max(valid, key=lambda row: _as_float(row.get(key), 0.0))


def _print_best(label: str, row: Optional[Dict[str, Any]]) -> None:
    if not row:
        print(f"{label}: unavailable")
        return
    print(
        f"{label}: {row['id']} mode={row['mode']} "
        f"R@50={row['R@50']:.4f} mR@50={row['mR@50']:.4f} "
        f"head/body/tail={row['head_mR@50']:.4f}/{row['body_mR@50']:.4f}/{row['tail_mR@50']:.4f} "
        f"role_swap={row.get('role_swap_consistency', 0.0):.4f} "
        f"combined={row['combined']:.4f}"
    )


def _print_family_table(rows: List[Dict[str, Any]]) -> None:
    print("\nScore-family best rows:")
    print(f"{'family':<14} {'run/epoch':<34} {'mode':<18} {'R@50':>8} {'mR@50':>8} {'tail':>8} {'role':>8}")
    for family in sorted({row["family"] for row in rows}):
        best = _best([row for row in rows if row["family"] == family], "combined")
        if not best:
            continue
        print(f"{family:<14} {best['id'][:34]:<34} {best['mode'][:18]:<18} {best['R@50']:>8.4f} {best['mR@50']:>8.4f} {best['tail_mR@50']:>8.4f} {best.get('role_swap_consistency', 0.0):>8.4f}")


def _print_worst_predicates(row: Optional[Dict[str, Any]], limit: int) -> None:
    if not row:
        return
    per_pred = row.get("per_predicate_R@50")
    if not isinstance(per_pred, dict) or not per_pred:
        print("\nWorst predicates: unavailable (metrics do not include per-predicate recall).")
        return
    print(f"\nWorst {limit} predicates for {row['id']} mode={row['mode']}:")
    for pred, value in sorted(per_pred.items(), key=lambda kv: (float(kv[1]), kv[0]))[:limit]:
        print(f"  {pred:<28} R@50={float(value):.4f}")


def _print_config_snapshot(row: Optional[Dict[str, Any]]) -> None:
    if not row:
        return
    cfg = row.get("config")
    if not isinstance(cfg, dict) or not cfg:
        print("\nConfig snapshot: unavailable.")
        return
    keys = [
        "run_name",
        "seed",
        "eval_sgg_predicate_score_mode",
        "eval_sgg_predicate_ensemble_alpha",
        "bayes_calibration_weight",
        "adaptive_calibration_enabled",
        "predicate_sampler_enabled",
        "predicate_sampler_power",
        "predicate_label_relaxation_enabled",
    ]
    print("\nConfig snapshot:")
    for key in keys:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standardized PURE metrics report card.")
    parser.add_argument("metrics", nargs="+", help="metrics.jsonl/json files or glob patterns")
    parser.add_argument("--frequency_file", type=Path, default=None, help="predicate frequency JSON/CSV/TSV")
    parser.add_argument("--train_jsonl", type=Path, default=None, help="VG150 train.jsonl used to derive predicate buckets")
    parser.add_argument("--head_frac", type=float, default=0.30, help="top fraction of predicates marked head")
    parser.add_argument("--tail_frac", type=float, default=0.30, help="bottom fraction of predicates marked tail")
    parser.add_argument("--lambda_mr", type=float, default=1.0, help="combined score weight for mR@50")
    parser.add_argument("--mu_tail", type=float, default=1.0, help="combined score weight for tail_mR@50")
    parser.add_argument("--worst", type=int, default=10, help="number of worst predicates to print")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    paths = _expand_metric_inputs(args.metrics)
    if not paths:
        raise SystemExit("No metrics files matched the provided inputs.")

    counts: Counter = Counter()
    if args.frequency_file is not None:
        counts = _load_counts_from_frequency_file(args.frequency_file)
    elif args.train_jsonl is not None:
        counts = _load_counts_from_train_jsonl(args.train_jsonl)
    buckets = _build_buckets(counts, args.head_frac, args.tail_frac)

    rows_by_file: List[Tuple[Path, MetricRow]] = []
    for path in paths:
        for row in _load_json_or_jsonl(path):
            rows_by_file.append((path, row))
    summaries = _summaries(rows_by_file, buckets, args.lambda_mr, args.mu_tail)
    if not summaries:
        raise SystemExit("No usable PredCls/SGG metric rows found.")

    best_r = _best(summaries, "R@50")
    best_mr = _best(summaries, "mR@50")
    best_combined = _best(summaries, "combined")

    payload = {
        "num_files": len(paths),
        "num_rows": len(rows_by_file),
        "num_mode_rows": len(summaries),
        "bucket_counts": dict(Counter(buckets.values())),
        "best_R@50": best_r,
        "best_mR@50": best_mr,
        "best_combined": best_combined,
        "rows": summaries,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    print("PURE model report card")
    print(f"files={len(paths)} rows={len(rows_by_file)} score_rows={len(summaries)} buckets={payload['bucket_counts']}")
    _print_best("Best R@50", best_r)
    _print_best("Best mR@50", best_mr)
    _print_best("Best combined", best_combined)
    _print_family_table(summaries)
    _print_worst_predicates(best_combined, int(args.worst))
    _print_config_snapshot(best_combined)


if __name__ == "__main__":
    main()