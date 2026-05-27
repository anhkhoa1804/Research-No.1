from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from core_utils import (
    SCENE_KEYS,
    discover_core_root,
    entities_for_scene,
    entity_id,
    entity_label,
    get_image_size,
    iter_metadata_files,
    load_metadata,
    grounding_box,
    grounding_entity_id,
    normalized_cxcywh_to_xyxy,
    relation_endpoints,
    relation_predicate,
    relations_for_scene,
    relpath,
    resolve_image_path,
    stable_split_key,
)


def split_for_item(version: str, group: str, pair_id: str, train_ratio: float, val_ratio: float, holdout_v2: bool) -> str:
    if holdout_v2 and version == "v2":
        return "test"
    value = stable_split_key(version, group, pair_id)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "validation"
    return "test"


def build_scene_row(core_root: Path, version: str, group: str, group_dir: Path, item: Dict[str, Any], scene_key: str, item_index: int) -> Dict[str, Any] | None:
    scene = item.get(scene_key)
    if not isinstance(scene, dict):
        return None
    pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
    entities = entities_for_scene(item, scene_key)
    if not entities:
        return None
    image_path = resolve_image_path(group_dir, scene, pair_id=pair_id, scene_key=scene_key, group=group)
    width, height = get_image_size(image_path)

    box_by_entity: Dict[str, List[float]] = {}
    grounding = scene.get("grounding", [])
    if isinstance(grounding, list):
        for grounded in grounding:
            if not isinstance(grounded, dict):
                continue
            raw_id = grounding_entity_id(grounded)
            if raw_id is None:
                continue
            converted = normalized_cxcywh_to_xyxy(grounding_box(grounded), width, height)
            if converted is not None:
                box_by_entity[str(raw_id)] = converted

    kept_objects: List[Dict[str, Any]] = []
    kept_boxes: List[List[float]] = []
    old_to_new: Dict[str, int] = {}
    for idx, entity in enumerate(entities):
        eid = entity_id(entity, idx)
        box = box_by_entity.get(eid)
        if box is None:
            continue
        old_to_new[eid] = len(kept_objects)
        kept_objects.append({"object_id": len(kept_objects), "names": [entity_label(entity)], "core_entity_id": eid})
        kept_boxes.append(box)

    if len(kept_objects) < 2:
        return None

    relationships: List[Dict[str, Any]] = []
    for rel in relations_for_scene(scene):
        subject_id, object_id = relation_endpoints(rel)
        if subject_id not in old_to_new or object_id not in old_to_new:
            continue
        subject_idx = old_to_new[str(subject_id)]
        object_idx = old_to_new[str(object_id)]
        if subject_idx == object_idx:
            continue
        relationships.append({
            "subject_id": subject_idx,
            "object_id": object_idx,
            "predicate": relation_predicate(rel),
            "core_relation": rel,
        })

    if not relationships:
        return None

    return {
        "image_id": f"core_{version}_{group}_{pair_id}_{scene_key}",
        "image": relpath(image_path, core_root) if image_path is not None else "",
        "width": width,
        "height": height,
        "objects": kept_objects,
        "obj_boxes": kept_boxes,
        "relationships": relationships,
        "source_dataset": "CORE",
        "core_version": version,
        "core_group": group,
        "core_pair_id": pair_id,
        "core_scene": scene_key,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_core(args: argparse.Namespace) -> Dict[str, Any]:
    core_root = discover_core_root(Path(args.core_root))
    out_root = Path(args.out_root)
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    skipped = Counter()
    predicates = Counter()

    for version, group, meta_path in iter_metadata_files(core_root):
        items = load_metadata(meta_path)
        for item_index, item in enumerate(items):
            pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
            item.setdefault("group", group)
            split = split_for_item(version, group, pair_id, args.train_ratio, args.val_ratio, args.holdout_v2)
            for scene_key in SCENE_KEYS:
                row = build_scene_row(core_root, version, group, meta_path.parent, item, scene_key, item_index)
                if row is None:
                    skipped[f"{version}/{group}"] += 1
                    continue
                splits[split].append(row)
                for rel in row["relationships"]:
                    predicates[rel["predicate"]] += 1

    for split, rows in splits.items():
        write_jsonl(out_root / f"{split}.jsonl", rows)

    report = {
        "core_root": str(core_root),
        "out_root": str(out_root),
        "rows": {split: len(rows) for split, rows in splits.items()},
        "skipped": dict(skipped),
        "top_predicates": predicates.most_common(50),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "core_conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CORE metadata into PURE/VG150-style JSONL.")
    parser.add_argument("--core-root", default="datasets/core_benchmark")
    parser.add_argument("--out-root", default="datasets/core_vg150_jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-v2", action="store_true", help="Force all v2 rows into test.jsonl.")
    args = parser.parse_args()
    print(json.dumps(convert_core(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()