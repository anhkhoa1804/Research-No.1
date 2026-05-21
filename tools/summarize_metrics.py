from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _get(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(description="Print compact training/eval metrics summary.")
    parser.add_argument("metrics", nargs="+", help="metrics.jsonl files")
    args = parser.parse_args()
    for filename in args.metrics:
        path = Path(filename)
        print(f"\n== {path} ==")
        if not path.exists():
            print("missing")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            pred_top = _get(row, ["val_sgg", "predicate_diag", "pred_top"], []) or []
            gt_top = _get(row, ["val_sgg", "predicate_diag", "gt_top"], []) or []
            score_modes = _get(row, ["val_sgg", "score_modes"], {}) or _get(row, ["val_sgg", "score_mode_metrics"], {}) or {}
            best = _get(row, ["best_so_far"], {}) or {}
            print(
                "epoch={epoch} loss={loss:.4f} predcls_R50={r50:.4f} predcls_mR50={mr50:.4f} "
                "ground_R50={gr50:.4f} pos={pos} cand={cand}".format(
                    epoch=row.get("epoch", "?"),
                    loss=float(_get(row, ["train", "avg_loss"], 0.0) or 0.0),
                    r50=float(_get(row, ["val_sgg", "predcls", "R@50"], 0.0) or 0.0),
                    mr50=float(_get(row, ["val_sgg", "predcls", "mR@50"], 0.0) or 0.0),
                    gr50=float(_get(row, ["val_grounding", "R@50"], 0.0) or 0.0),
                    pos=int(_get(row, ["train", "positive_pairs"], 0) or 0),
                    cand=int(_get(row, ["train", "candidate_pairs"], 0) or 0),
                )
            )
            if best:
                print("  best_so_far:", best)
            if isinstance(score_modes, dict) and score_modes:
                compact = {}
                for mode, metrics in score_modes.items():
                    if isinstance(metrics, dict):
                        predcls = metrics.get("predcls", metrics)
                        if isinstance(predcls, dict):
                            compact[mode] = {"R@50": predcls.get("R@50"), "mR@50": predcls.get("mR@50")}
                if compact:
                    print("  score_modes:", compact)
            print("  gt_top:", [(x.get("label"), x.get("count")) for x in gt_top[:5] if isinstance(x, dict)])
            print("  pred_top:", [(x.get("label"), x.get("count")) for x in pred_top[:5] if isinstance(x, dict)])


if __name__ == "__main__":
    main()