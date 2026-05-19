from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _split(summary: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = summary.get(name, {})
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check VG150 preparation diagnostics before training.")
    parser.add_argument("--diagnostics", default="datasets/diagnostics.json")
    parser.add_argument("--min_train_rows", type=int, default=1)
    parser.add_argument("--min_val_rows", type=int, default=1)
    parser.add_argument("--min_predicate_coverage", type=int, default=50)
    parser.add_argument("--max_fallback_image_ratio", type=float, default=0.05)
    parser.add_argument("--require_no_validation_issues", action="store_true")
    parser.add_argument("--allow_fallback_images", action="store_true")
    args = parser.parse_args()

    path = Path(args.diagnostics)
    if not path.exists():
        raise SystemExit(f"Missing diagnostics file: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))

    failures = []
    for name, min_rows in (("train", int(args.min_train_rows)), ("validation", int(args.min_val_rows))):
        split = _split(summary, name)
        rows = int(split.get("rows", 0))
        coverage = int(split.get("predicate_coverage", 0))
        counters = split.get("filter_counters", {}) if isinstance(split.get("filter_counters", {}), dict) else {}
        issues = split.get("validation_issues", {}) if isinstance(split.get("validation_issues", {}), dict) else {}
        fallback = int(counters.get("fallback_images", 0))
        alias_mapped = int(counters.get("predicate_aliases_mapped", 0))
        unknown_filtered = int(counters.get("unknown_predicates_filtered", 0))
        fallback_ratio = float(fallback) / float(max(1, rows))
        print(
            f"{name}: rows={rows} predicates={coverage} relationships={int(split.get('relationships_total', 0))} "
            f"alias_mapped={alias_mapped} unknown_filtered={unknown_filtered} "
            f"fallback_images={fallback} ({fallback_ratio:.3f}) issues={issues}"
        )
        if rows < min_rows:
            failures.append(f"{name}: rows {rows} < {min_rows}")
        if coverage < int(args.min_predicate_coverage):
            failures.append(f"{name}: predicate coverage {coverage} < {int(args.min_predicate_coverage)}")
        if bool(args.require_no_validation_issues) and len(issues) > 0:
            failures.append(f"{name}: validation issues present: {issues}")
        if not bool(args.allow_fallback_images) and fallback_ratio > float(args.max_fallback_image_ratio):
            failures.append(
                f"{name}: fallback image ratio {fallback_ratio:.3f} > {float(args.max_fallback_image_ratio):.3f}"
            )
    if failures:
        raise SystemExit("Diagnostics check failed:\n" + "\n".join(f"- {x}" for x in failures))


if __name__ == "__main__":
    main()