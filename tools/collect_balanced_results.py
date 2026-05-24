from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


def get(row: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def infer_kind(path: str) -> tuple[str, str]:
    name = Path(path).parts[-2] if len(Path(path).parts) >= 2 else path
    variant = name
    for prefix in ("reeval_core_l3_balanced_", "core_l3_balanced_", "reeval_core_l3_seed_", "core_l3_seed_"):
        variant = variant.replace(prefix, "")
    variant = re.sub(r"_best_mR50.*", "", variant)
    mode = "raw" if "raw_classifier" in name else "fa" + (re.search(r"_fa([0-9]+)_", name).group(1) if re.search(r"_fa([0-9]+)_", name) else "train")
    return variant, mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect top balanced debias metrics.")
    parser.add_argument("patterns", nargs="*", default=["runs/core_l3_balanced_*/metrics.jsonl", "runs/reeval_core_l3_balanced_*/metrics.jsonl", "runs/core_l3_seed_*/metrics.jsonl", "runs/reeval_core_l3_seed_*/metrics.jsonl"])
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows: list[tuple[float, float, float, str, str, str, str]] = []
    for pattern in args.patterns:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                r50 = get(row, ("val_sgg", "predcls", "R@50"))
                mr50 = get(row, ("val_sgg", "predcls", "mR@50"))
                gr50 = get(row, ("val_grounding", "R@50"), 0.0)
                if r50 is None or mr50 is None:
                    continue
                variant, mode = infer_kind(path)
                rows.append((float(mr50), float(r50), float(gr50 or 0.0), variant, mode, str(row.get("epoch", "?")), path))

    print("mR@50   R@50    G@50    variant                         mode     ep   path")
    print("------  ------  ------  ------------------------------  -------  ---  ----")
    for mr50, r50, gr50, variant, mode, epoch, path in sorted(rows, reverse=True)[: args.top]:
        print(f"{mr50*100:6.2f}  {r50*100:6.2f}  {gr50*100:6.2f}  {variant[:30]:30}  {mode:7}  {epoch!s:>3}  {path}")


if __name__ == "__main__":
    main()