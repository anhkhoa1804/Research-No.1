from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _iter_rows(root: Path, splits: Iterable[str]) -> Iterable[dict[str, Any]]:
    for split in splits:
        path = root / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def _object_names(row: dict[str, Any]) -> Iterable[str]:
    labels = row.get("obj_labels", row.get("object_labels", []))
    if isinstance(labels, list) and labels:
        for label in labels:
            yield str(label)
        return
    objects = row.get("objects", [])
    if not isinstance(objects, list):
        return
    for obj in objects:
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            names = obj.get("names", obj.get("name", []))
            if isinstance(names, str):
                yield names
            elif isinstance(names, list) and names:
                yield str(names[0])


def _predicate_names(row: dict[str, Any]) -> Iterable[str]:
    rels = row.get("relationships", row.get("relations", row.get("rels", [])))
    if not isinstance(rels, list):
        return
    for rel in rels:
        if isinstance(rel, dict):
            yield str(rel.get("predicate", rel.get("pred", "")))


def _clean(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean local vocabulary files for CORE eval JSONL.")
    parser.add_argument("--jsonl-root", default="datasets/core_vg150_jsonl")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--max-objects", type=int, default=300)
    args = parser.parse_args()

    jsonl_root = Path(args.jsonl_root)
    out_root = Path(args.out_root) if str(args.out_root).strip() else jsonl_root
    vocab_dir = out_root / "vocabulary"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    generic_objects = {"", "__background__", "background", "bg", "object", "objects", "thing", "entity"}
    generic_preds = {"", "__background__", "background", "bg", "relation", "no relation", "no interaction"}
    object_aliases = {
        "people": "person",
        "persons": "person",
        "men": "man",
        "women": "woman",
        "children": "child",
        "kids": "child",
    }
    predicate_aliases = {
        "resting on": "on",
        "rests on": "on",
        "placed on": "on",
        "sits on": "on",
        "inside": "in",
        "placed under": "under",
        "turned towards": "facing",
        "tilted towards": "facing",
        "angled towards": "facing",
        "pointed at": "facing",
        "aimed at": "facing",
    }

    object_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    for row in _iter_rows(jsonl_root, splits):
        for name in _object_names(row):
            clean = object_aliases.get(_clean(name), _clean(name))
            if clean in generic_objects:
                continue
            object_counts[clean] += 1
        for pred in _predicate_names(row):
            clean = predicate_aliases.get(_clean(pred), _clean(pred))
            if clean in generic_preds:
                continue
            pred_counts[clean] += 1

    objects = [name for name, _ in object_counts.most_common(max(1, int(args.max_objects)))]
    predicates = [name for name, _ in pred_counts.most_common()]

    (vocab_dir / "objects.json").write_text(json.dumps(objects, indent=2, ensure_ascii=False), encoding="utf-8")
    (vocab_dir / "predicates.json").write_text(json.dumps(predicates, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "jsonl_root": str(jsonl_root),
        "out_root": str(out_root),
        "num_objects": len(objects),
        "num_predicates": len(predicates),
        "top_objects": object_counts.most_common(30),
        "top_predicates": pred_counts.most_common(50),
    }
    (vocab_dir / "core_eval_vocab_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()