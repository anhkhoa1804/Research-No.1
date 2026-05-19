#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_predicate_vocab(vg150_root: Path, explicit_vocab: Optional[Path], observed_counts: Counter) -> List[str]:
    if explicit_vocab is not None and explicit_vocab.exists():
        data = json.loads(explicit_vocab.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("predicate_vocab", "predicates", "idx_to_predicate", "predicate_to_idx"):
                value = data.get(key, None)
                if isinstance(value, list):
                    vocab = [_norm(x, "") for x in value]
                    return [x for x in vocab if x != ""]
                if isinstance(value, dict):
                    if key.startswith("idx_to"):
                        items = sorted(value.items(), key=lambda kv: int(kv[0]))
                        vocab = [_norm(name, "") for _idx, name in items]
                    else:
                        items = sorted(value.items(), key=lambda kv: int(kv[1]))
                        vocab = [_norm(name, "") for name, _idx in items]
                    return [x for x in vocab if x != ""]
        if isinstance(data, list):
            vocab = [_norm(x, "") for x in data]
            return [x for x in vocab if x != ""]

    try:
        from openvocab_rel.datasets.vg150_loader import scan_vg150_predicate_vocab

        vocab = [_norm(x, "") for x in scan_vg150_predicate_vocab(str(vg150_root))]
        vocab = [x for x in vocab if x != ""]
        if len(vocab) > 0:
            return vocab
    except Exception:
        pass

    return sorted(str(key) for key in observed_counts.keys())


def _norm(value: Any, fallback: str = "") -> str:
    text = str(value).strip().lower()
    return text if text else fallback


def _object_names(objects: Any) -> Tuple[List[str], Dict[int, int]]:
    names: List[str] = []
    lookup: Dict[int, int] = {}
    for idx, obj in enumerate(objects if isinstance(objects, list) else []):
        if isinstance(obj, dict):
            raw_names = obj.get("names", obj.get("name", obj.get("label", "object")))
            if isinstance(raw_names, list) and len(raw_names) > 0:
                name = raw_names[0]
            else:
                name = raw_names
            for key in ("object_id", "obj_id", "id"):
                raw_id = obj.get(key, None)
                if isinstance(raw_id, (int, float)):
                    lookup[int(raw_id)] = idx
        else:
            name = obj
        names.append(_norm(name, "object"))
        lookup.setdefault(idx, idx)
    return names, lookup




def _name_from_endpoint(endpoint: Any) -> str:
    if isinstance(endpoint, dict):
        raw_names = endpoint.get("names", endpoint.get("name", endpoint.get("label", "")))
        if isinstance(raw_names, list) and len(raw_names) > 0:
            return _norm(raw_names[0], "")
        return _norm(raw_names, "")
    return _norm(endpoint, "")


def _resolve_index(raw: Any, lookup: Dict[int, int], num_objects: int) -> Optional[int]:
    if isinstance(raw, dict):
        for key in ("object_id", "obj_id", "id"):
            if isinstance(raw.get(key, None), (int, float)):
                raw = raw[key]
                break
    if not isinstance(raw, (int, float)):
        return None
    idx = lookup.get(int(raw), int(raw))
    if 0 <= idx < num_objects:
        return int(idx)
    return None


def _relation_indices(rel: Dict[str, Any], lookup: Dict[int, int], num_objects: int) -> Tuple[Optional[int], Optional[int]]:
    subject_raw = rel.get("subject_id", rel.get("subj_id", rel.get("subject", rel.get("subj"))))
    object_raw = rel.get("object_id", rel.get("obj_id", rel.get("object", rel.get("obj"))))
    return _resolve_index(subject_raw, lookup, num_objects), _resolve_index(object_raw, lookup, num_objects)


def _log_probs(counts: Counter, pred_vocab: List[str], smoothing: float) -> List[float]:
    smooth = max(0.0, float(smoothing))
    values = [float(counts.get(pred, 0.0)) + smooth for pred in pred_vocab]
    total = max(sum(values), 1e-12)
    return [math.log(max(value / total, 1e-12)) for value in values]


def build_prior(
    train_jsonl: Path,
    out_path: Path,
    smoothing: float,
    vg150_root: Optional[Path] = None,
    predicate_vocab_path: Optional[Path] = None,
) -> Dict[str, Any]:
    global_counts: Counter = Counter()
    pair_counts: Dict[str, Counter] = defaultdict(Counter)
    subject_counts: Dict[str, Counter] = defaultdict(Counter)
    object_counts: Dict[str, Counter] = defaultdict(Counter)
    object_vocab = set()

    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            objects = row.get("objects", row.get("obj_labels", [])) if isinstance(row, dict) else []
            relationships = row.get("relationships", row.get("relations", row.get("rels", []))) if isinstance(row, dict) else []
            names, lookup = _object_names(objects)
            object_vocab.update(names)
            for rel in relationships if isinstance(relationships, list) else []:
                if not isinstance(rel, dict):
                    continue
                subj_idx, obj_idx = _relation_indices(rel, lookup, len(names))
                if subj_idx is not None and obj_idx is not None and subj_idx != obj_idx:
                    subject = names[subj_idx]
                    obj = names[obj_idx]
                else:
                    subject = _name_from_endpoint(rel.get("subject", rel.get("subj", rel.get("subject_name", ""))))
                    obj = _name_from_endpoint(rel.get("object", rel.get("obj", rel.get("object_name", ""))))
                    if subject == "" or obj == "" or subject == obj:
                        continue
                    object_vocab.update([subject, obj])
                predicate = _norm(rel.get("predicate", rel.get("pred", "relation")), "relation")
                key = f"{subject}||{obj}"
                global_counts[predicate] += 1
                pair_counts[key][predicate] += 1
                subject_counts[subject][predicate] += 1
                object_counts[obj][predicate] += 1

    root = vg150_root if vg150_root is not None else train_jsonl.parent
    pred_vocab = _load_predicate_vocab(root, predicate_vocab_path, global_counts)
    payload = {
        "predicate_vocab": pred_vocab,
        "object_vocab": sorted(object_vocab),
        "smoothing": float(smoothing),
        "default_log_prob": math.log(1.0 / max(1, len(pred_vocab))),
        "global_log_probs": _log_probs(global_counts, pred_vocab, smoothing),
        "pair_log_probs": {key: _log_probs(counts, pred_vocab, smoothing) for key, counts in sorted(pair_counts.items())},
        "subject_log_probs": {key: _log_probs(counts, pred_vocab, smoothing) for key, counts in sorted(subject_counts.items())},
        "object_log_probs": {key: _log_probs(counts, pred_vocab, smoothing) for key, counts in sorted(object_counts.items())},
        "stats": {
            "vocab_source": "fixed_or_scanned" if len(pred_vocab) >= len(global_counts) else "observed_fallback",
            "num_predicates": len(pred_vocab),
            "num_object_labels": len(object_vocab),
            "num_pairs": len(pair_counts),
            "num_relationships": int(sum(global_counts.values())),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VG150 subject-object frequency prior JSON.")
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--vg150_root", type=Path, default=None)
    parser.add_argument("--predicate_vocab", type=Path, default=None)
    args = parser.parse_args()
    payload = build_prior(args.train_jsonl, args.out_path, args.smoothing, args.vg150_root, args.predicate_vocab)
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()