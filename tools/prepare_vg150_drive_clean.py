from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_vg150_subset import (
    STANDARD_VG150_PREDICATES,
    VG150_PREDICATE_ALIASES,
    _first_name,
    _normalize_box,
    _numeric_id,
)

DEFAULT_DRIVE_ID = "1O7IswcnlnzVQ07qWdA9n3gbveA3Oxq14"


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _download_with_gdown(drive_id: str, out_path: Path) -> bool:
    if shutil.which("gdown") is None:
        return False
    commands = [
        ["gdown", str(drive_id), "-O", str(out_path)],
        ["gdown", f"https://drive.google.com/uc?id={drive_id}", "-O", str(out_path)],
    ]
    for cmd in commands:
        try:
            _run(cmd)
        except subprocess.CalledProcessError:
            continue
        if _looks_like_archive(out_path):
            return True
    return False


def _download_with_curl(drive_id: str, out_path: Path) -> bool:
    if shutil.which("curl") is None:
        return False
    cookie_path = out_path.with_suffix(".cookies.txt")
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    try:
        _run(["curl", "-L", "-c", str(cookie_path), url, "-o", str(out_path)])
    except subprocess.CalledProcessError:
        return False
    return _looks_like_archive(out_path)


def _looks_like_archive(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        head = path.read_bytes()[:256].lower()
    except OSError:
        return False
    if b"<html" in head or b"<!doctype html" in head:
        return False
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _extract_archive(archive_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(raw_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(raw_dir)
    else:
        raise SystemExit(f"Downloaded file is not a supported archive: {archive_path}")


def _find_jsonl_root(raw_dir: Path) -> Path:
    candidates = [p.parent for p in raw_dir.rglob("train.jsonl")]
    for candidate in candidates:
        if (candidate / "validation.jsonl").exists() or (candidate / "val.jsonl").exists():
            return candidate
    raise SystemExit(f"Could not find train.jsonl plus validation.jsonl/val.jsonl under {raw_dir}")


def _copy_or_link_images(jsonl_root: Path, out_dir: Path, mode: str) -> None:
    src = jsonl_root / "images"
    dst = out_dir / "images"
    if not src.exists() or dst.exists():
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    if mode == "symlink":
        try:
            dst.symlink_to(os.path.relpath(src, start=out_dir))
            return
        except OSError:
            shutil.copytree(src, dst)
            return
    raise ValueError(f"Unknown image mode: {mode}")


def _canonical_predicate_vocab() -> Dict[str, Any]:
    """Return idx_to_predicate dict derived from STANDARD_VG150_PREDICATES (sorted, 1-indexed).

    The canonical ordering is sorted(STANDARD_VG150_PREDICATES), confirmed by the historical
    frequency_prior.json predicate_vocab field.  This function is the single source of truth
    for the on-disk vocabulary representation; it must never diverge from STANDARD_VG150_PREDICATES.
    """
    ordered = sorted(STANDARD_VG150_PREDICATES)
    assert len(ordered) == 50, f"STANDARD_VG150_PREDICATES must have 50 entries, got {len(ordered)}"
    return {"idx_to_predicate": {str(i + 1): p for i, p in enumerate(ordered)}}


def _copy_vocab(jsonl_root: Path, out_dir: Path, allow_source_mismatch: bool = False) -> None:
    """Write the canonical predicate vocabulary and copy the object vocabulary.

    predicates.json is always derived from STANDARD_VG150_PREDICATES — never blindly
    copied from the source — so a malformed source file cannot silently corrupt
    the predicate-index ↔ predicate-label mapping used by the model.

    If a source predicates.json exists, its predicate set is validated against
    STANDARD_VG150_PREDICATES.  By default (allow_source_mismatch=False) a mismatch
    raises SystemExit.  Pass allow_source_mismatch=True when running against raw VG150
    archives whose vocabulary file is known to differ from the curated 50-class set
    (the JSONL relationship data is still filtered to STANDARD_VG150_PREDICATES).
    """
    dst = out_dir / "vocabulary"
    dst.mkdir(parents=True, exist_ok=True)
    src = jsonl_root / "vocabulary"

    # Validate source predicates.json if present; fail loudly on mismatch (by default).
    src_pred_path = src / "predicates.json"
    if src_pred_path.exists():
        with src_pred_path.open(encoding="utf-8") as fh:
            src_data = json.load(fh)
        src_preds = set(src_data.get("idx_to_predicate", {}).values())
        if src_preds != STANDARD_VG150_PREDICATES:
            extra = src_preds - STANDARD_VG150_PREDICATES
            missing = STANDARD_VG150_PREDICATES - src_preds
            msg = (
                f"Source predicates.json is incompatible with STANDARD_VG150_PREDICATES.\n"
                f"  Extra (in source, not canonical): {sorted(extra)}\n"
                f"  Missing (canonical, not in source): {sorted(missing)}\n"
                f"  The canonical vocabulary will be written regardless.\n"
                f"  Pass --allow_source_vocab_mismatch if the source is raw VG150 data."
            )
            if not allow_source_mismatch:
                raise SystemExit(msg)
            print(f"[WARN] {msg}", flush=True)

    # Always write the canonical vocabulary, never copy from source.
    canonical = _canonical_predicate_vocab()
    with (dst / "predicates.json").open("w", encoding="utf-8") as fh:
        json.dump(canonical, fh, ensure_ascii=False, indent=2)

    # objects.json is copied from source unchanged (no canonical constraint here).
    if src.exists() and (src / "objects.json").exists():
        shutil.copy2(src / "objects.json", dst / "objects.json")


def _raw_relationship_refs(rel: Dict[str, Any]) -> Tuple[Any, Any]:
    subj = rel.get("subject_id", rel.get("subj_id", rel.get("subject", rel.get("subj", -1))))
    obj = rel.get("object_id", rel.get("obj_id", rel.get("object", rel.get("obj", -1))))
    return subj, obj


def _build_objects(row: Dict[str, Any], max_objects: int) -> Tuple[List[List[float]], List[Dict[str, Any]], Dict[int, int], Counter]:
    diag: Counter = Counter()
    raw_boxes = row.get("obj_boxes", row.get("boxes", row.get("bboxes", [])))
    raw_objects = row.get("objects", row.get("obj_labels", row.get("labels", [])))
    candidate_count = max(
        len(raw_boxes) if isinstance(raw_boxes, list) else 0,
        len(raw_objects) if isinstance(raw_objects, list) else 0,
    )
    boxes: List[List[float]] = []
    objects: List[Dict[str, Any]] = []
    lookup: Dict[int, int] = {}
    for raw_idx in range(candidate_count):
        if len(boxes) >= int(max_objects):
            diag["objects_truncated"] += 1
            break
        raw_obj = raw_objects[raw_idx] if isinstance(raw_objects, list) and raw_idx < len(raw_objects) else {}
        raw_box = raw_boxes[raw_idx] if isinstance(raw_boxes, list) and raw_idx < len(raw_boxes) else raw_obj
        box = _normalize_box(raw_box)
        if box is None:
            diag["bad_boxes"] += 1
            continue
        new_idx = len(boxes)
        boxes.append(box)
        objects.append({"object_id": new_idx, "names": [_first_name(raw_obj)]})
        lookup[int(raw_idx)] = new_idx
        if isinstance(raw_obj, dict):
            for key in ("object_id", "obj_id", "id"):
                raw_id = _numeric_id(raw_obj.get(key))
                if raw_id is not None:
                    lookup[int(raw_id)] = new_idx
    return boxes, objects, lookup, diag


def _normalize_predicate(value: Any, map_aliases: bool) -> Tuple[str, bool]:
    predicate = " ".join(str(value).strip().lower().split())
    mapped = False
    if map_aliases and predicate in VG150_PREDICATE_ALIASES:
        predicate = VG150_PREDICATE_ALIASES[predicate]
        mapped = True
    return predicate, mapped


def _build_relationships(
    row: Dict[str, Any],
    lookup: Dict[int, int],
    map_aliases: bool,
) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    diag: Counter = Counter()
    pred_counter: Counter = Counter()
    rels: List[Dict[str, Any]] = []
    seen = set()
    raw_rels = row.get("relationships", row.get("relations", row.get("rels", [])))
    for rel in raw_rels if isinstance(raw_rels, list) else []:
        if not isinstance(rel, dict):
            diag["bad_relationship_records"] += 1
            continue
        subj_raw, obj_raw = _raw_relationship_refs(rel)
        subj = lookup.get(_numeric_id(subj_raw))
        obj = lookup.get(_numeric_id(obj_raw))
        if subj is None or obj is None:
            diag["bad_rel_ref"] += 1
            continue
        if subj == obj:
            diag["self_relationships"] += 1
            continue
        predicate, alias_mapped = _normalize_predicate(rel.get("predicate", rel.get("pred", "")), map_aliases)
        if alias_mapped:
            diag["predicate_aliases_mapped"] += 1
        if predicate not in STANDARD_VG150_PREDICATES:
            diag["unknown_predicates_filtered"] += 1
            continue
        key = (subj, obj, predicate)
        if key in seen:
            diag["duplicate_relationships"] += 1
            continue
        seen.add(key)
        rels.append({"subject_id": int(subj), "object_id": int(obj), "predicate": predicate})
        pred_counter[predicate] += 1
    return rels, diag, pred_counter


def _convert_split(
    jsonl_root: Path,
    out_dir: Path,
    split: str,
    out_name: str,
    max_objects: int,
    min_relationships: int,
    map_aliases: bool,
) -> Tuple[List[Dict[str, Any]], Counter, Counter, Counter, Counter]:
    in_path = jsonl_root / f"{split}.jsonl"
    if not in_path.exists() and split == "validation":
        in_path = jsonl_root / "val.jsonl"
    if not in_path.exists():
        return [], Counter(), Counter(), Counter(), Counter()

    rows: List[Dict[str, Any]] = []
    diag: Counter = Counter()
    pred_counter: Counter = Counter()
    obj_counter: Counter = Counter()
    with in_path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if line == "":
                continue
            row = json.loads(line)
            boxes, objects, lookup, obj_diag = _build_objects(row, int(max_objects))
            diag.update(obj_diag)
            rels, rel_diag, rel_preds = _build_relationships(row, lookup, bool(map_aliases))
            diag.update(rel_diag)
            pred_counter.update(rel_preds)
            if len(boxes) == 0 or len(rels) < int(min_relationships):
                diag["empty_after_filter"] += 1
                continue
            for obj in objects:
                obj_counter[str(obj.get("names", ["object"])[0])] += 1
            rows.append(
                {
                    "image_id": str(row.get("image_id", row.get("img_id", ""))),
                    "image": row.get("image", row.get("image_path", row.get("path", ""))),
                    "obj_boxes": boxes,
                    "objects": objects,
                    "relationships": rels,
                }
            )
    out_path = out_dir / f"{out_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, diag, pred_counter, obj_counter, Counter()


def _validate_clean_rows(rows: List[Dict[str, Any]], out_dir: Path) -> Counter:
    issues: Counter = Counter()
    image_ids = set()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if image_id in image_ids:
            issues["duplicate_image_ids"] += 1
        image_ids.add(image_id)

        raw_image = str(row.get("image", row.get("image_path", row.get("path", "")))).strip()
        candidates: List[Path] = []
        if raw_image:
            candidates.extend([
                out_dir / raw_image,
                out_dir / "images" / raw_image,
                out_dir / "images" / "VG_100K" / os.path.basename(raw_image),
                out_dir / "images" / "VG_100K_2" / os.path.basename(raw_image),
            ])
        if image_id:
            stem = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
            candidates.extend([
                out_dir / "images" / stem,
                out_dir / "images" / "VG_100K" / stem,
                out_dir / "images" / "VG_100K_2" / stem,
            ])
        if not any(path.exists() for path in candidates):
            issues["missing_image_files"] += 1

        boxes = row.get("obj_boxes", row.get("boxes", []))
        objects = row.get("objects", row.get("obj_labels", []))
        if len(boxes) != len(objects):
            issues["box_object_count_mismatch"] += 1
        num_objects = len(objects) if isinstance(objects, list) else 0
        rels = row.get("relationships", row.get("relations", row.get("rels", [])))
        if not isinstance(rels, list) or len(rels) == 0:
            issues["empty_predicates_after_write"] += 1
            continue
        for rel in rels:
            subj = rel.get("subject_id", -1) if isinstance(rel, dict) else -1
            obj = rel.get("object_id", -1) if isinstance(rel, dict) else -1
            if (
                not isinstance(subj, int)
                or not isinstance(obj, int)
                or subj < 0
                or obj < 0
                or subj >= num_objects
                or obj >= num_objects
                or subj == obj
            ):
                issues["invalid_relationship_indices"] += 1
            if not isinstance(rel, dict) or not str(rel.get("predicate", "")).strip():
                issues["empty_predicates_after_write"] += 1
    return issues


def _diagnose_split(
    name: str,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    diag: Counter,
    preds: Counter,
    objs: Counter,
) -> Dict[str, Any]:
    issues = _validate_clean_rows(rows, out_dir)
    return {
        "split": name,
        "rows": len(rows),
        "relationships_total": int(sum(len(r.get("relationships", [])) for r in rows)),
        "predicate_coverage": len(preds),
        "object_coverage": len(objs),
        "top_predicates": preds.most_common(20),
        "top_objects": objs.most_common(20),
        "filter_counters": dict(sorted(diag.items())),
        "validation_issues": dict(sorted(issues.items())),
    }


def _assert_good(summary: Dict[str, Any], min_train_rows: int, min_val_rows: int, min_predicate_coverage: int) -> None:
    failures: List[str] = []
    checks = (("train", min_train_rows), ("validation", min_val_rows))
    for split, min_rows in checks:
        item = summary.get(split, {})
        if int(item.get("rows", 0)) < int(min_rows):
            failures.append(f"{split}: rows {item.get('rows', 0)} < {min_rows}")
        if int(item.get("predicate_coverage", 0)) < int(min_predicate_coverage):
            failures.append(f"{split}: predicate coverage {item.get('predicate_coverage', 0)} < {min_predicate_coverage}")
        issues = item.get("validation_issues", {}) if isinstance(item.get("validation_issues", {}), dict) else {}
        if issues:
            failures.append(f"{split}: validation issues present: {issues}")
    if failures:
        raise SystemExit("VG150 Drive preparation failed:\n" + "\n".join(f"- {x}" for x in failures))


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    archive_path = Path(args.archive_path)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if not bool(args.skip_download) and not archive_path.exists():
        ok = _download_with_gdown(str(args.drive_id), archive_path)
        if not ok:
            ok = _download_with_curl(str(args.drive_id), archive_path)
        if not ok:
            raise SystemExit(
                "Could not download a valid archive from Google Drive. "
                "Ensure the file is shared as 'Anyone with the link' or download it manually to "
                f"{archive_path}."
            )

    if not bool(args.skip_extract):
        if not _looks_like_archive(archive_path):
            raise SystemExit(f"Archive is missing or looks invalid: {archive_path}")
        _extract_archive(archive_path, raw_dir)

    jsonl_root = Path(args.jsonl_root) if args.jsonl_root else _find_jsonl_root(raw_dir)
    print(f"[VG150Drive] jsonl_root={jsonl_root}")
    _copy_vocab(jsonl_root, out_dir, allow_source_mismatch=bool(getattr(args, "allow_source_vocab_mismatch", False)))
    _copy_or_link_images(jsonl_root, out_dir, str(args.images))

    train_rows, train_diag, train_preds, train_objs, _ = _convert_split(
        jsonl_root, out_dir, "train", "train", int(args.max_objects), int(args.min_relationships), not bool(args.no_predicate_alias_map)
    )
    val_rows, val_diag, val_preds, val_objs, _ = _convert_split(
        jsonl_root, out_dir, "validation", "validation", int(args.max_objects), int(args.min_relationships), not bool(args.no_predicate_alias_map)
    )
    summary: Dict[str, Any] = {
        "drive_id": str(args.drive_id),
        "jsonl_root": str(jsonl_root),
        "out_dir": str(out_dir),
        "train_jsonl": str(out_dir / "train.jsonl"),
        "validation_jsonl": str(out_dir / "validation.jsonl"),
        "train": _diagnose_split("train", train_rows, out_dir, train_diag, train_preds, train_objs),
        "validation": _diagnose_split("validation", val_rows, out_dir, val_diag, val_preds, val_objs),
    }
    if (jsonl_root / "test.jsonl").exists():
        test_rows, test_diag, test_preds, test_objs, _ = _convert_split(
            jsonl_root, out_dir, "test", "test", int(args.max_objects), int(args.min_relationships), not bool(args.no_predicate_alias_map)
        )
        summary["test"] = _diagnose_split("test", test_rows, out_dir, test_diag, test_preds, test_objs)
        summary["test_jsonl"] = str(out_dir / "test.jsonl")

    diagnostics_path = out_dir / "diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[VG150Drive] wrote diagnostics to {diagnostics_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:8000])

    if not bool(args.allow_validation_warnings):
        _assert_good(summary, int(args.min_train_rows), int(args.min_val_rows), int(args.min_predicate_coverage))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Google Drive VG JSONL archive and filter it into VG150-clean JSONL.")
    parser.add_argument("--drive_id", default=DEFAULT_DRIVE_ID)
    parser.add_argument("--archive_path", default="downloads/vg_drive_dataset.zip")
    parser.add_argument("--raw_dir", default="datasets/vg_drive_raw")
    parser.add_argument("--jsonl_root", default="", help="Use an already-extracted JSONL root instead of auto-discovery.")
    parser.add_argument("--out_dir", default="datasets_vg150_clean")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--images", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--min_relationships", type=int, default=1)
    parser.add_argument("--min_train_rows", type=int, default=50000)
    parser.add_argument("--min_val_rows", type=int, default=5000)
    parser.add_argument("--min_predicate_coverage", type=int, default=50)
    parser.add_argument("--no_predicate_alias_map", action="store_true")
    parser.add_argument("--allow_validation_warnings", action="store_true")
    parser.add_argument(
        "--allow_source_vocab_mismatch", action="store_true",
        help="Skip strict validation of source predicates.json against STANDARD_VG150_PREDICATES. "
             "Use this when running against raw VG150 archives whose vocabulary file differs from "
             "the curated 50-class set (relationship data is still filtered correctly).",
    )
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()