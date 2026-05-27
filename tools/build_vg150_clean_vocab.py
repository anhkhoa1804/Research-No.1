from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


GENERIC_OBJECTS = {"", "__background__", "background", "bg", "object", "objects", "thing", "entity"}
ALIASES = {
    "people": "person",
    "persons": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "kids": "child",
    "trees": "tree",
    "windows": "window",
    "buildings": "building",
    "cars": "car",
    "buses": "bus",
    "planes": "airplane",
    "airplanes": "airplane",
    "bikes": "bicycle",
    "shirts": "shirt",
    "pants": "pant",
    "legs": "leg",
    "hands": "hand",
    "heads": "head",
    "eyes": "eye",
    "ears": "ear",
}


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


def _clean(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean VG150 object vocabulary from JSONL splits.")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--out", default="datasets/vocabulary/objects.json")
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--max-objects", type=int, default=150)
    parser.add_argument("--max-words", type=int, default=2)
    args = parser.parse_args()

    splits = [split.strip() for split in str(args.splits).split(",") if split.strip()]
    counter: Counter[str] = Counter()
    skipped_generic = 0
    skipped_shape = 0

    for row in _iter_rows(Path(args.data_root), splits):
        for raw_name in _object_names(row):
            name = ALIASES.get(_clean(raw_name), _clean(raw_name))
            if name in GENERIC_OBJECTS:
                skipped_generic += 1
                continue
            if any(char.isdigit() for char in name) or len(name.split()) > int(args.max_words):
                skipped_shape += 1
                continue
            counter[name] += 1

    vocab = [name for name, _ in counter.most_common(max(1, int(args.max_objects)))]
    if not vocab:
        print(
            f"[build_vg150_clean_vocab] no valid object labels found under {args.data_root} for splits={splits}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "data_root": str(args.data_root),
        "out": str(out_path),
        "splits": splits,
        "num_objects": len(vocab),
        "skipped_generic": skipped_generic,
        "skipped_shape": skipped_shape,
        "top_objects": counter.most_common(50),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()