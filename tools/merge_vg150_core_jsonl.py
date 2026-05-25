from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge VG150 JSONL and converted CORE JSONL for train ablations.")
    parser.add_argument("--vg-root", default="datasets")
    parser.add_argument("--core-root", default="datasets/core_vg150_jsonl")
    parser.add_argument("--out-root", default="datasets/vg150_core_merged")
    parser.add_argument("--core-train-repeat", type=int, default=1)
    parser.add_argument("--include-core-validation", action="store_true")
    args = parser.parse_args()

    vg_root = Path(args.vg_root)
    core_root = Path(args.core_root)
    out_root = Path(args.out_root)

    vg_train = read_jsonl(vg_root / "train.jsonl")
    vg_val = read_jsonl(vg_root / "validation.jsonl") or read_jsonl(vg_root / "val.jsonl")
    core_train = read_jsonl(core_root / "train.jsonl")
    core_val = read_jsonl(core_root / "validation.jsonl")

    merged_train = list(vg_train)
    for _ in range(max(0, int(args.core_train_repeat))):
        merged_train.extend(core_train)
    merged_val = list(vg_val)
    if args.include_core_validation:
        merged_val.extend(core_val)

    write_jsonl(out_root / "train.jsonl", merged_train)
    write_jsonl(out_root / "validation.jsonl", merged_val)

    for aux in ("frequency_prior.json", "VG-SGG-dicts-with-attri.json"):
        src = vg_root / aux
        if src.exists():
            (out_root / aux).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    report: Dict[str, int | str] = {
        "vg_root": str(vg_root),
        "core_root": str(core_root),
        "out_root": str(out_root),
        "vg_train_rows": len(vg_train),
        "core_train_rows": len(core_train),
        "merged_train_rows": len(merged_train),
        "validation_rows": len(merged_val),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()