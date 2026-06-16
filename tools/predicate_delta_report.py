#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from model_report_card import _as_float, _load_json_or_jsonl, _normalize_predicate, _per_predicate_recall, _score_modes


def _row_name(path: Path, row: Dict[str, Any], mode: str) -> str:
    return f"{row.get('run_name', path.parent.name)}:epoch={row.get('epoch', '?')}:mode={mode}"


def _load_series(paths: Iterable[Path]) -> List[Tuple[str, Dict[str, float]]]:
    series: List[Tuple[str, Dict[str, float]]] = []
    for path in paths:
        for row in _load_json_or_jsonl(path):
            for mode, block in _score_modes(row).items():
                per_pred = _per_predicate_recall(block)
                if per_pred:
                    series.append((_row_name(path, row, mode), per_pred))
    return series


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-predicate R@50 between two PURE metric rows.")
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics.jsonl/json files")
    parser.add_argument("--baseline", default="", help="substring selecting baseline row; defaults to first row")
    parser.add_argument("--candidate", default="", help="substring selecting candidate row; defaults to last row")
    parser.add_argument("--worst", type=int, default=20, help="number of worst/dead predicates to print")
    parser.add_argument("--delta", type=int, default=20, help="number of largest positive/negative deltas to print")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    series = _load_series(args.metrics)
    if not series:
        raise SystemExit("No per-predicate recall found in provided metrics.")

    baseline = next(((name, vals) for name, vals in series if args.baseline and args.baseline in name), series[0])
    candidate = next(((name, vals) for name, vals in series if args.candidate and args.candidate in name), series[-1])
    base_name, base_vals = baseline
    cand_name, cand_vals = candidate
    predicates = sorted(set(base_vals) | set(cand_vals))
    rows = [
        {
            "predicate": pred,
            "baseline": _as_float(base_vals.get(pred, 0.0), 0.0),
            "candidate": _as_float(cand_vals.get(pred, 0.0), 0.0),
            "delta": _as_float(cand_vals.get(pred, 0.0), 0.0) - _as_float(base_vals.get(pred, 0.0), 0.0),
        }
        for pred in predicates
    ]
    dead = [row for row in rows if row["candidate"] <= 0.0]
    summary = {
        "baseline": base_name,
        "candidate": cand_name,
        "num_predicates": len(rows),
        "baseline_mean": _mean([row["baseline"] for row in rows]),
        "candidate_mean": _mean([row["candidate"] for row in rows]),
        "dead_candidate": len(dead),
        "improved": sum(1 for row in rows if row["delta"] > 0.0),
        "regressed": sum(1 for row in rows if row["delta"] < 0.0),
        "dead_predicates": sorted(dead, key=lambda row: (row["baseline"], row["predicate"]))[: args.worst],
        "largest_gains": sorted(rows, key=lambda row: (-row["delta"], row["predicate"]))[: args.delta],
        "largest_losses": sorted(rows, key=lambda row: (row["delta"], row["predicate"]))[: args.delta],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"baseline:  {base_name}")
    print(f"candidate: {cand_name}")
    print(
        f"predicates={summary['num_predicates']} mean={summary['baseline_mean']:.4f}->{summary['candidate_mean']:.4f} "
        f"dead={summary['dead_candidate']} improved={summary['improved']} regressed={summary['regressed']}"
    )
    print(f"\nDead predicates in candidate (first {args.worst}):")
    for row in summary["dead_predicates"]:
        print(f"  {row['predicate']:<28} base={row['baseline']:.4f} cand={row['candidate']:.4f}")
    print(f"\nLargest gains (top {args.delta}):")
    for row in summary["largest_gains"]:
        print(f"  {row['predicate']:<28} delta={row['delta']:+.4f} {row['baseline']:.4f}->{row['candidate']:.4f}")
    print(f"\nLargest losses (top {args.delta}):")
    for row in summary["largest_losses"]:
        print(f"  {row['predicate']:<28} delta={row['delta']:+.4f} {row['baseline']:.4f}->{row['candidate']:.4f}")


if __name__ == "__main__":
    main()