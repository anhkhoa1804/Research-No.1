#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from openvocab_rel.clip_utils import configure_clip

from core_utils import (
    SCENE_KEYS,
    discover_core_root,
    entities_for_scene,
    entity_id,
    entity_label,
    iter_metadata_files,
    load_metadata,
    relation_endpoints,
    relation_predicate,
    relations_for_scene,
    resolve_image_path,
)


GENERIC_OBJECTS = {"", "object", "objects", "thing", "entity", "unknown", "none", "background", "__background__"}
GENERIC_RELATIONS = {"", "relation", "relationships", "unknown", "none", "background", "__background__", "no relation", "no interaction"}


def _clean(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _scene_text(item: dict[str, Any], scene_key: str) -> str:
    scene = item.get(scene_key, {}) if isinstance(item.get(scene_key), dict) else {}
    entities = entities_for_scene(item, scene_key)
    id_to_label = {
        entity_id(entity, idx): _clean(entity_label(entity))
        for idx, entity in enumerate(entities)
        if _clean(entity_label(entity)) not in GENERIC_OBJECTS
    }
    clauses: list[str] = []
    for rel in relations_for_scene(scene):
        subject_id, object_id = relation_endpoints(rel)
        predicate = _clean(relation_predicate(rel))
        if predicate in GENERIC_RELATIONS:
            continue
        subject = id_to_label.get(str(subject_id), "object")
        obj = id_to_label.get(str(object_id), "object")
        clauses.append(f"{subject} {predicate} {obj}")
    if clauses:
        return "A scene where " + "; ".join(clauses[:4]) + "."
    labels = list(dict.fromkeys(label for label in id_to_label.values() if label not in GENERIC_OBJECTS))
    return "A scene containing " + ", ".join(labels[:8]) + "." if labels else "A scene."


def _iter_pairs(core_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for version, group, meta_path in iter_metadata_files(core_root):
        for item_index, item in enumerate(load_metadata(meta_path)):
            pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
            item.setdefault("group", group)
            image_paths = {
                scene_key: resolve_image_path(meta_path.parent, item.get(scene_key, {}), pair_id=pair_id, scene_key=scene_key, group=group)
                for scene_key in SCENE_KEYS
            }
            if any(path is None for path in image_paths.values()):
                continue
            rows.append({
                "version": version,
                "group": group,
                "pair_id": pair_id,
                "texts": {scene_key: _scene_text(item, scene_key) for scene_key in SCENE_KEYS},
                "images": {scene_key: str(image_paths[scene_key]) for scene_key in SCENE_KEYS},
            })
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CORE pairwise text-image retrieval with CLIP.")
    parser.add_argument("--core-root", default="datasets/core")
    parser.add_argument("--clip-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--out", default="runs/core_text_image_retrieval/metrics.json")
    args = parser.parse_args()

    core_root = discover_core_root(Path(args.core_root))
    pairs = _iter_pairs(core_root)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    device = torch.device(args.device)
    model, processor, _, _ = configure_clip(args.clip_name, device)
    model.eval()

    correct = 0
    total = 0
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    for pair in pairs:
        images = []
        for scene_key in SCENE_KEYS:
            with Image.open(pair["images"][scene_key]) as image:
                images.append(image.convert("RGB"))
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = F.normalize(model.get_image_features(**image_inputs), dim=-1)
        for target_index, scene_key in enumerate(SCENE_KEYS):
            text_inputs = processor(text=[pair["texts"][scene_key]], return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = F.normalize(model.get_text_features(**text_inputs), dim=-1)
            scores = (text_features @ image_features.T).squeeze(0)
            pred_index = int(torch.argmax(scores).item())
            is_correct = pred_index == target_index
            correct += int(is_correct)
            total += 1
            by_group[pair["group"]]["correct"] += int(is_correct)
            by_group[pair["group"]]["total"] += 1
            if len(examples) < 50:
                examples.append({**pair, "query_scene": scene_key, "pred_scene": SCENE_KEYS[pred_index], "scores": [float(x) for x in scores.cpu()]})

    report = {
        "core_root": str(core_root),
        "clip_name": args.clip_name,
        "num_pairs": len(pairs),
        "total_queries": total,
        "accuracy": correct / total if total else 0.0,
        "by_group": {group: {"accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0, **dict(counts)} for group, counts in sorted(by_group.items())},
        "examples": examples,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()