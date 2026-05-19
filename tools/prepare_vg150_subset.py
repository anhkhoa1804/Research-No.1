from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

STANDARD_VG150_PREDICATES = {
    "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to",
    "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from",
    "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on",
    "made of", "mounted on", "near", "next to", "of", "on", "on back of", "over", "painted on",
    "parked on", "part of", "playing", "riding", "sitting on", "standing on", "to", "under",
    "using", "walking in", "walking on", "watching", "wearing", "wears", "with", "wrapped around",
}

VG150_PREDICATE_ALIASES = {
    "are in": "in",
    "are on": "on",
    "around": "wrapped around",
    "beside": "next to",
    "below": "under",
    "has a": "has",
    "has an": "has",
    "have": "has",
    "hanging on": "hanging from",
    "holds": "holding",
    "in a": "in",
    "in an": "in",
    "inside": "in",
    "inside of": "in",
    "of a": "of",
    "of an": "of",
    "on a": "on",
    "on an": "on",
    "on front of": "in front of",
    "on side of": "on",
    "on top of": "on",
    "sitting in": "sitting on",
    "standing in": "standing on",
    "wearing a": "wearing",
    "wearing an": "wearing",
}


def _open_image_candidate(value: Any, download_remote_images: bool = True, timeout: float = 10.0) -> Optional[Image.Image]:
    try:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value)).convert("RGB")
        if isinstance(value, dict):
            for key in ("bytes", "data", "image", "jpg", "jpeg", "png"):
                if key in value:
                    image = _open_image_candidate(
                        value.get(key),
                        download_remote_images=bool(download_remote_images),
                        timeout=float(timeout),
                    )
                    if image is not None:
                        return image
            for key in ("path", "file", "filename", "filepath", "image_path"):
                image_path = value.get(key)
                if image_path and Path(str(image_path)).exists():
                    return Image.open(str(image_path)).convert("RGB")
            if bool(download_remote_images):
                for key in ("url", "image_url", "download_url"):
                    image_url = value.get(key)
                    if image_url:
                        image = _download_image_url(str(image_url), timeout=float(timeout))
                        if image is not None:
                            return image
        if isinstance(value, str):
            if Path(value).exists():
                return Image.open(value).convert("RGB")
            if bool(download_remote_images) and value.startswith(("http://", "https://")):
                return _download_image_url(value, timeout=float(timeout))
    except Exception:
        return None
    return None


def _download_image_url(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
    try:
        request = urllib.request.Request(
            str(url),
            headers={"User-Agent": "Mozilla/5.0 (compatible; PURE-v3-VG150-prep/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            raw = response.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _first_name(value: Any, fallback: str = "object") -> str:
    if isinstance(value, dict):
        names = value.get("names", value.get("synsets", value.get("name", value.get("label", fallback))))
        if isinstance(names, list) and len(names) > 0:
            return str(names[0]).strip().lower() or fallback
        return str(names).strip().lower() or fallback
    return str(value).strip().lower() or fallback


def _image_from_example(
    example: Dict[str, Any],
    download_remote_images: bool = True,
    image_download_timeout: float = 10.0,
) -> Tuple[Image.Image, bool]:
    for key in ("image", "img", "jpg", "jpeg", "png", "image_bytes", "bytes", "image_path", "path", "file"):
        if key in example:
            image = _open_image_candidate(
                example.get(key),
                download_remote_images=bool(download_remote_images),
                timeout=float(image_download_timeout),
            )
            if image is not None:
                return image, True
    for value in example.values():
        image = _open_image_candidate(
            value,
            download_remote_images=bool(download_remote_images),
            timeout=float(image_download_timeout),
        )
        if image is not None:
            return image, True
    return Image.new("RGB", (336, 336), color=(128, 128, 128)), False


def _normalize_predicate_name(predicate: str, map_predicate_aliases: bool = True) -> Tuple[str, bool]:
    normalized = " ".join(str(predicate).strip().lower().split())
    if bool(map_predicate_aliases) and normalized in VG150_PREDICATE_ALIASES:
        return VG150_PREDICATE_ALIASES[normalized], True
    return normalized, False


def _numeric_id(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        for key in ("object_id", "obj_id", "id", "index"):
            out = _numeric_id(value.get(key))
            if out is not None:
                return out
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _normalize_box(box: Any) -> Optional[List[float]]:
    if isinstance(box, dict):
        if all(key in box for key in ("x", "y", "w", "h")):
            box = [box["x"], box["y"], float(box["x"]) + float(box["w"]), float(box["y"]) + float(box["h"])]
        elif all(key in box for key in ("x1", "y1", "x2", "y2")):
            box = [box["x1"], box["y1"], box["x2"], box["y2"]]
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        vals = [float(x) for x in box[:4]]
    except (TypeError, ValueError):
        return None
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def _box_from_object(obj: Any) -> Optional[List[float]]:
    if not isinstance(obj, dict):
        return None
    return _normalize_box(obj.get("box", obj.get("bbox", obj)))


def _raw_relationship_indices(rel: Dict[str, Any]) -> Tuple[Any, Any]:
    subj = rel.get("subject_id", rel.get("subj_id", rel.get("subject", rel.get("subj", -1))))
    obj = rel.get("object_id", rel.get("obj_id", rel.get("object", rel.get("obj", -1))))
    return subj, obj


def _extract_boxes_and_objects(example: Dict[str, Any], max_objects: int) -> Tuple[List[List[float]], List[Dict[str, Any]], Dict[int, int], Counter]:
    diag: Counter = Counter()
    raw_boxes = example.get("obj_boxes", example.get("boxes", example.get("bboxes", [])))
    raw_objects = example.get("objects", example.get("obj_labels", example.get("labels", [])))
    boxes: List[List[float]] = []
    objects: List[Dict[str, Any]] = []
    id_lookup: Dict[int, int] = {}

    candidate_count = max(
        len(raw_boxes) if isinstance(raw_boxes, list) else 0,
        len(raw_objects) if isinstance(raw_objects, list) else 0,
    )
    for raw_idx in range(candidate_count):
        if len(boxes) >= int(max_objects):
            diag["objects_truncated"] += 1
            break
        raw_obj = raw_objects[raw_idx] if isinstance(raw_objects, list) and raw_idx < len(raw_objects) else {}
        raw_box = raw_boxes[raw_idx] if isinstance(raw_boxes, list) and raw_idx < len(raw_boxes) else None
        box = _normalize_box(raw_box) or _box_from_object(raw_obj)
        if box is None:
            diag["bad_boxes"] += 1
            continue
        new_idx = len(boxes)
        boxes.append(box)
        objects.append({"object_id": new_idx, "names": [_first_name(raw_obj)]})
        id_lookup[int(raw_idx)] = new_idx
        if isinstance(raw_obj, dict):
            for key in ("object_id", "obj_id", "id"):
                raw_id = _numeric_id(raw_obj.get(key))
                if raw_id is not None:
                    id_lookup[int(raw_id)] = new_idx
    return boxes, objects, id_lookup, diag


def _resolve_object_ref(value: Any, id_lookup: Dict[int, int]) -> Optional[int]:
    raw_id = _numeric_id(value)
    if raw_id is None:
        return None
    return id_lookup.get(raw_id)


def _normalize_relationships(
    raw_rels: Any,
    id_lookup: Dict[int, int],
    allow_unknown_predicates: bool = False,
    map_predicate_aliases: bool = True,
) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    relationships: List[Dict[str, Any]] = []
    diag: Counter = Counter()
    pred_counter: Counter = Counter()
    seen = set()
    for rel in raw_rels if isinstance(raw_rels, list) else []:
        if not isinstance(rel, dict):
            diag["bad_relationship_records"] += 1
            continue
        subj_raw, obj_raw = _raw_relationship_indices(rel)
        subj = _resolve_object_ref(subj_raw, id_lookup)
        obj = _resolve_object_ref(obj_raw, id_lookup)
        if subj is None or obj is None:
            diag["unresolved_relationship_refs"] += 1
            continue
        if subj == obj:
            diag["self_relationships"] += 1
            continue
        predicate, alias_mapped = _normalize_predicate_name(
            str(rel.get("predicate", rel.get("pred", ""))),
            map_predicate_aliases=bool(map_predicate_aliases),
        )
        if predicate == "":
            diag["empty_predicates"] += 1
            continue
        if alias_mapped:
            diag["predicate_aliases_mapped"] += 1
        if predicate not in STANDARD_VG150_PREDICATES:
            pred_counter[predicate] += 1
            diag["unknown_predicates_filtered"] += 1
            if not bool(allow_unknown_predicates):
                continue
        key = (subj, obj, predicate)
        if key in seen:
            diag["duplicate_relationships"] += 1
            continue
        seen.add(key)
        relationships.append({"subject_id": subj, "object_id": obj, "predicate": predicate})
    return relationships, diag, pred_counter


def _normalize_example(
    example: Dict[str, Any],
    split_name: str,
    index: int,
    image_dir: Path,
    max_objects: int,
    min_relationships: int,
    save_images: bool,
    jpeg_quality: int,
    download_remote_images: bool = True,
    image_download_timeout: float = 10.0,
    allow_unknown_predicates: bool = False,
    map_predicate_aliases: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Counter, Counter]:
    diag: Counter = Counter()
    image_id = str(example.get("image_id", example.get("img_id", index))).strip() or str(index)
    boxes, objects, id_lookup, object_diag = _extract_boxes_and_objects(example, max_objects=max_objects)
    diag.update(object_diag)
    if len(boxes) < 2:
        diag["too_few_objects"] += 1
        return None, diag, Counter()

    raw_rels = example.get("relationships", example.get("relations", example.get("rels", [])))
    relationships, rel_diag, unknown_predicates = _normalize_relationships(
        raw_rels,
        id_lookup,
        allow_unknown_predicates=bool(allow_unknown_predicates),
        map_predicate_aliases=bool(map_predicate_aliases),
    )
    diag.update(rel_diag)
    if len(relationships) < int(min_relationships):
        diag["too_few_relationships"] += 1
        return None, diag, unknown_predicates

    safe_id = Path(image_id).stem.replace("/", "_")
    image_name = f"{split_name}_{safe_id}.jpg"
    image_rel_path = f"images/{image_name}"
    image_path = image_dir / image_name
    image_ok = True
    if save_images and not image_path.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        pil_image, image_ok = _image_from_example(
            example,
            download_remote_images=bool(download_remote_images),
            image_download_timeout=float(image_download_timeout),
        )
        pil_image.save(image_path, quality=int(jpeg_quality))
    if not image_ok:
        diag["fallback_images"] += 1

    return {
        "image_id": image_id,
        "image": image_rel_path,
        "obj_boxes": boxes,
        "objects": objects,
        "relationships": relationships,
    }, diag, unknown_predicates


def _summarize_value(value: Any, depth: int = 0) -> Any:
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, Image.Image):
        return {"type": "PIL.Image", "size": list(value.size), "mode": value.mode}
    if isinstance(value, (bytes, bytearray)):
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys())[:25],
            "items": {str(k): _summarize_value(v, depth + 1) for k, v in list(value.items())[:8]},
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "len": len(value),
            "first": _summarize_value(value[0], depth + 1) if len(value) > 0 else None,
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "len": len(value),
            "first": _summarize_value(value[0], depth + 1) if len(value) > 0 else None,
        }
    text = str(value)
    return {"type": type(value).__name__, "repr": text[:160]}


def _inspect_dataset_schema(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: python3 -m pip install datasets huggingface_hub") from exc

    for split_name in (str(args.train_split), str(args.val_split)):
        dataset = load_dataset(str(args.dataset_id), split=split_name, streaming=True)
        print(f"\n[Schema] split={split_name}")
        for index, example in enumerate(dataset):
            if index >= int(args.inspect_examples):
                break
            image, image_ok = _image_from_example(
                example,
                download_remote_images=not bool(args.no_remote_images),
                image_download_timeout=float(args.image_download_timeout),
            )
            summary = {str(k): _summarize_value(v) for k, v in example.items()}
            print(json.dumps({
                "index": index,
                "keys": list(example.keys()),
                "image_ok": bool(image_ok),
                "image_size": list(image.size),
                "fields": summary,
            }, ensure_ascii=False, indent=2)[:12000])


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _build_split(args: argparse.Namespace, split_name: str, target_count: int) -> Tuple[List[Dict[str, Any]], Counter, Counter, Counter, Counter]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: python3 -m pip install datasets huggingface_hub") from exc

    dataset = load_dataset(str(args.dataset_id), split=split_name, streaming=True)
    rows: List[Dict[str, Any]] = []
    diag: Counter = Counter()
    pred_counter: Counter = Counter()
    obj_counter: Counter = Counter()
    unknown_predicates: Counter = Counter()
    image_dir = Path(args.out_dir) / "images"

    for index, example in enumerate(dataset):
        if int(args.max_source_scan) > 0 and index >= int(args.max_source_scan):
            break
        if len(rows) >= int(target_count):
            break
        row, row_diag, row_unknown_predicates = _normalize_example(
            example=example,
            split_name=split_name,
            index=index,
            image_dir=image_dir,
            max_objects=int(args.max_objects),
            min_relationships=int(args.min_relationships),
            save_images=not bool(args.no_images),
            jpeg_quality=int(args.jpeg_quality),
            download_remote_images=not bool(args.no_remote_images),
            image_download_timeout=float(args.image_download_timeout),
            allow_unknown_predicates=bool(args.allow_unknown_predicates),
            map_predicate_aliases=not bool(args.no_predicate_alias_map),
        )
        diag.update(row_diag)
        unknown_predicates.update(row_unknown_predicates)
        if row is None:
            diag["rejected"] += 1
            continue
        rows.append(row)
        diag["accepted"] += 1
        diag["objects"] += len(row["objects"])
        diag["relationships"] += len(row["relationships"])
        pred_counter.update(rel["predicate"] for rel in row["relationships"])
        obj_counter.update(obj["names"][0] for obj in row["objects"])

    return rows, diag, pred_counter, obj_counter, unknown_predicates


def _validate_rows(rows: List[Dict[str, Any]], out_dir: Path, require_images: bool) -> Counter:
    issues: Counter = Counter()
    image_ids = set()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if image_id in image_ids:
            issues["duplicate_image_ids"] += 1
        image_ids.add(image_id)
        boxes = row.get("obj_boxes", [])
        objects = row.get("objects", [])
        rels = row.get("relationships", [])
        if len(boxes) != len(objects):
            issues["box_object_count_mismatch"] += 1
        if require_images and not (out_dir / str(row.get("image", ""))).exists():
            issues["missing_image_files"] += 1
        for rel in rels if isinstance(rels, list) else []:
            subj = _numeric_id(rel.get("subject_id"))
            obj = _numeric_id(rel.get("object_id"))
            if subj is None or obj is None or subj < 0 or obj < 0 or subj >= len(boxes) or obj >= len(boxes) or subj == obj:
                issues["invalid_relationship_indices"] += 1
            pred = str(rel.get("predicate", "")).strip().lower()
            if pred == "":
                issues["empty_predicates_after_write"] += 1
    return issues


def _diagnose_split(
    name: str,
    rows: List[Dict[str, Any]],
    diag: Counter,
    preds: Counter,
    objs: Counter,
    unknown_predicates: Counter,
    validation_issues: Counter,
) -> Dict[str, Any]:
    rel_counts = [len(row.get("relationships", [])) for row in rows]
    obj_counts = [len(row.get("objects", [])) for row in rows]
    return {
        "split": name,
        "rows": len(rows),
        "accepted": int(diag.get("accepted", 0)),
        "rejected": int(diag.get("rejected", 0)),
        "objects_total": int(diag.get("objects", 0)),
        "relationships_total": int(diag.get("relationships", 0)),
        "objects_per_image_avg": float(sum(obj_counts) / max(1, len(obj_counts))),
        "relationships_per_image_avg": float(sum(rel_counts) / max(1, len(rel_counts))),
        "predicate_coverage": len(preds),
        "object_coverage": len(objs),
        "top_predicates": preds.most_common(20),
        "top_objects": objs.most_common(20),
        "unknown_predicates": unknown_predicates.most_common(20),
        "filter_counters": dict(sorted(diag.items())),
        "validation_issues": dict(sorted(validation_issues.items())),
    }


def _fail_if_bad(summary: Dict[str, Any], min_predicate_coverage: int) -> None:
    fatal_messages = []
    for split_name in ("train", "validation"):
        split = summary[split_name]
        if int(split["rows"]) <= 0:
            fatal_messages.append(f"{split_name}: no accepted rows")
        if int(split["predicate_coverage"]) < int(min_predicate_coverage):
            fatal_messages.append(
                f"{split_name}: predicate coverage {split['predicate_coverage']} < {min_predicate_coverage}"
            )
        issues = split.get("validation_issues", {})
        for key in ("box_object_count_mismatch", "missing_image_files", "invalid_relationship_indices", "empty_predicates_after_write"):
            if int(issues.get(key, 0)) > 0:
                fatal_messages.append(f"{split_name}: {key}={issues[key]}")
    if fatal_messages:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(f"- {msg}" for msg in fatal_messages))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate a clean local VG150 JSONL subset from Hugging Face.")
    parser.add_argument("--dataset_id", default="anhkhoa1804/VG150-SGG-Standard")
    parser.add_argument("--out_dir", default="datasets")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--train_images", type=int, default=5000)
    parser.add_argument("--val_images", type=int, default=500)
    parser.add_argument("--max_source_scan", type=int, default=0)
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--min_relationships", type=int, default=1)
    parser.add_argument("--min_predicate_coverage", type=int, default=10)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    parser.add_argument("--no_remote_images", action="store_true")
    parser.add_argument("--image_download_timeout", type=float, default=10.0)
    parser.add_argument("--no_images", action="store_true")
    parser.add_argument("--allow_unknown_predicates", action="store_true")
    parser.add_argument("--no_predicate_alias_map", action="store_true")
    parser.add_argument("--allow_validation_warnings", action="store_true")
    parser.add_argument("--inspect_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if int(args.inspect_examples) > 0:
        _inspect_dataset_schema(args)
        return

    random.seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_diag, train_preds, train_objs, train_unknown = _build_split(args, str(args.train_split), int(args.train_images))
    val_rows, val_diag, val_preds, val_objs, val_unknown = _build_split(args, str(args.val_split), int(args.val_images))

    random.shuffle(train_rows)
    train_count = _write_jsonl(out_dir / "train.jsonl", train_rows)
    val_count = _write_jsonl(out_dir / "validation.jsonl", val_rows)

    train_issues = _validate_rows(train_rows, out_dir, require_images=not bool(args.no_images))
    val_issues = _validate_rows(val_rows, out_dir, require_images=not bool(args.no_images))

    summary = {
        "dataset_id": str(args.dataset_id),
        "out_dir": str(out_dir),
        "train_jsonl": str(out_dir / "train.jsonl"),
        "validation_jsonl": str(out_dir / "validation.jsonl"),
        "image_dir": str(out_dir / "images"),
        "train": _diagnose_split("train", train_rows, train_diag, train_preds, train_objs, train_unknown, train_issues),
        "validation": _diagnose_split("validation", val_rows, val_diag, val_preds, val_objs, val_unknown, val_issues),
    }
    with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {train_count} train rows to {out_dir / 'train.jsonl'}")
    print(f"Wrote {val_count} validation rows to {out_dir / 'validation.jsonl'}")
    print(f"Wrote diagnostics to {out_dir / 'diagnostics.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:8000])

    if not bool(args.allow_validation_warnings):
        _fail_if_bad(summary, int(args.min_predicate_coverage))


if __name__ == "__main__":
    main()