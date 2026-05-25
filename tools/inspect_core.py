from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from core_utils import (
    EXPECTED_CORE_GROUPS,
    SCENE_KEYS,
    discover_core_root,
    entities_for_scene,
    entity_id,
    get_image_size,
    iter_metadata_files,
    load_metadata,
    normalized_cxcywh_to_xyxy,
    relation_endpoints,
    relations_for_scene,
    resolve_image_path,
)


def inspect_core(root: Path) -> Dict[str, Any]:
    core_root = discover_core_root(root)
    counters: Dict[str, Counter] = defaultdict(Counter)
    errors: List[str] = []
    warnings: List[str] = []
    groups_by_version: Dict[str, set] = defaultdict(set)

    for version, group, meta_path in iter_metadata_files(core_root):
        groups_by_version[version].add(group)
        group_dir = meta_path.parent
        try:
            items = load_metadata(meta_path)
        except Exception as exc:
            errors.append(f"{version}/{group}: cannot load metadata.json: {exc}")
            continue

        counters[version]["groups"] += 1
        counters[version]["pairs"] += len(items)
        counters["all"]["pairs"] += len(items)

        for item_index, item in enumerate(items):
            pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
            if group != "Extreme_Compositional_OOD" and "shared_entities" not in item:
                warnings.append(f"{version}/{group}/{pair_id}: non-extreme item lacks shared_entities")
            for scene_key in SCENE_KEYS:
                scene = item.get(scene_key)
                if not isinstance(scene, dict):
                    errors.append(f"{version}/{group}/{pair_id}: missing {scene_key}")
                    continue
                entities = entities_for_scene(item, scene_key)
                valid_ids = {entity_id(entity, idx) for idx, entity in enumerate(entities)}
                if not valid_ids:
                    warnings.append(f"{version}/{group}/{pair_id}/{scene_key}: no entities found")

                image_path = resolve_image_path(group_dir, scene)
                if image_path is None:
                    warnings.append(f"{version}/{group}/{pair_id}/{scene_key}: image not resolved")
                width, height = get_image_size(image_path)
                counters["all"]["scenes"] += 1
                counters[version]["scenes"] += 1

                grounding = scene.get("grounding", [])
                if grounding and not isinstance(grounding, list):
                    errors.append(f"{version}/{group}/{pair_id}/{scene_key}: grounding is not a list")
                    grounding = []
                for idx, grounded in enumerate(grounding if isinstance(grounding, list) else []):
                    if not isinstance(grounded, dict):
                        errors.append(f"{version}/{group}/{pair_id}/{scene_key}: grounding[{idx}] is not object")
                        continue
                    ref = grounded.get("entity_id", grounded.get("id", grounded.get("entity")))
                    if ref is not None and valid_ids and str(ref) not in valid_ids:
                        errors.append(f"{version}/{group}/{pair_id}/{scene_key}: unknown grounding entity {ref}")
                    box = grounded.get("box", grounded.get("bbox"))
                    if normalized_cxcywh_to_xyxy(box, width, height) is None:
                        errors.append(f"{version}/{group}/{pair_id}/{scene_key}: invalid box {box}")
                    counters["all"]["boxes"] += 1
                    counters[version]["boxes"] += 1

                for rel_idx, rel in enumerate(relations_for_scene(scene)):
                    subject_id, object_id = relation_endpoints(rel)
                    if subject_id is None or object_id is None:
                        warnings.append(f"{version}/{group}/{pair_id}/{scene_key}: relation[{rel_idx}] missing endpoints")
                    elif valid_ids and (subject_id not in valid_ids or object_id not in valid_ids):
                        warnings.append(f"{version}/{group}/{pair_id}/{scene_key}: relation[{rel_idx}] endpoints not in entities")
                    counters["all"]["relations"] += 1
                    counters[version]["relations"] += 1

    for version, groups in sorted(groups_by_version.items()):
        for missing in sorted(EXPECTED_CORE_GROUPS - groups):
            warnings.append(f"{version}: missing expected group {missing}")

    return {
        "root": str(core_root),
        "stats": {key: dict(value) for key, value in counters.items()},
        "groups_by_version": {key: sorted(value) for key, value in groups_by_version.items()},
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors[:300],
        "warnings": warnings[:300],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a downloaded CORE dataset folder.")
    parser.add_argument("--core-root", default="datasets/core_benchmark")
    parser.add_argument("--report", default="runs/core_inspect/report.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = inspect_core(Path(args.core_root))
    print("=== CORE INSPECT REPORT ===")
    print(f"Root: {report['root']}")
    print(f"Stats: {json.dumps(report['stats'], ensure_ascii=False)}")
    print(f"Errors: {report['num_errors']}")
    print(f"Warnings: {report['num_warnings']}")
    for line in report["errors"][:20]:
        print(f"ERROR {line}")
    for line in report["warnings"][:20]:
        print(f"WARN {line}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {report_path}")
    if args.strict and report["num_errors"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()