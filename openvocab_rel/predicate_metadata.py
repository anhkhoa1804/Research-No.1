from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

DEFAULT_GROUPS: Dict[str, str] = {
    "above": "spatial",
    "across": "spatial",
    "against": "spatial",
    "along": "spatial",
    "around": "spatial",
    "at": "spatial",
    "behind": "spatial",
    "between": "spatial",
    "in": "spatial",
    "in front of": "spatial",
    "inside": "spatial",
    "near": "spatial",
    "next to": "spatial",
    "on": "spatial",
    "on top of": "spatial",
    "over": "spatial",
    "under": "spatial",
    "below": "spatial",
    "beside": "spatial",
    "by": "spatial",
    "attached to": "contact",
    "carrying": "contact",
    "covering": "contact",
    "covered in": "contact",
    "eating": "action",
    "flying in": "action",
    "growing on": "contact",
    "hanging from": "contact",
    "has": "possession",
    "holding": "contact",
    "laying on": "pose",
    "looking at": "action",
    "mounted on": "contact",
    "painted on": "attribute",
    "parked on": "action",
    "playing": "action",
    "riding": "action",
    "says": "textual",
    "sitting on": "pose",
    "standing on": "pose",
    "to": "spatial",
    "using": "action",
    "walking in": "action",
    "walking on": "action",
    "watching": "action",
    "wearing": "possession",
    "wears": "possession",
    "with": "contact",
}

DEFAULT_SYMMETRIC = {
    "near",
    "next to",
    "beside",
    "by",
    "around",
    "overlap",
    "overlapping",
    "adjacent to",
    "on either side of",
    "looking at each other",
    "surrounding",
}


def normalize_predicate(value: Any) -> str:
    return str(value).strip().lower()


def default_predicate_metadata(predicates: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    keys = set(DEFAULT_GROUPS.keys()) | set(DEFAULT_SYMMETRIC)
    if predicates is not None:
        keys.update(normalize_predicate(p) for p in predicates if normalize_predicate(p))
    out: Dict[str, Dict[str, Any]] = {}
    for pred in sorted(keys):
        out[pred] = {
            "group": DEFAULT_GROUPS.get(pred, "other"),
            "symmetric": pred in DEFAULT_SYMMETRIC,
        }
    return out


def load_predicate_metadata(path: str = "", predicates: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    metadata = default_predicate_metadata(predicates)
    if str(path).strip() == "":
        return metadata
    p = Path(path)
    if not p.exists():
        return metadata
    data = json.loads(p.read_text(encoding="utf-8"))
    raw_items: Mapping[str, Any]
    if isinstance(data, dict) and isinstance(data.get("predicates"), dict):
        raw_items = data["predicates"]
    elif isinstance(data, dict):
        raw_items = data
    else:
        return metadata
    for pred, value in raw_items.items():
        key = normalize_predicate(pred)
        if not key:
            continue
        current = dict(metadata.get(key, {"group": "other", "symmetric": False}))
        if isinstance(value, dict):
            if "group" in value:
                current["group"] = normalize_predicate(value.get("group", "other")) or "other"
            if "symmetric" in value:
                current["symmetric"] = bool(value.get("symmetric", False))
        elif isinstance(value, str):
            current["group"] = normalize_predicate(value) or "other"
        metadata[key] = current
    return metadata


def predicate_group(metadata: Mapping[str, Mapping[str, Any]], predicate: str) -> str:
    item = metadata.get(normalize_predicate(predicate), {})
    return normalize_predicate(item.get("group", "other")) or "other"


def is_symmetric(metadata: Mapping[str, Mapping[str, Any]], predicate: str) -> bool:
    item = metadata.get(normalize_predicate(predicate), None)
    if item is None:
        return normalize_predicate(predicate) in DEFAULT_SYMMETRIC
    return bool(item.get("symmetric", False))