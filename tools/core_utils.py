from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


SCENE_KEYS = ("scene_A", "scene_B")
EXPECTED_CORE_GROUPS = {
    "Occlusion_Depth",
    "Gaze_Attention",
    "Action_Role_Reversal",
    "Spatial_Containment",
    "Attribute_Binding",
    "Extreme_Compositional_OOD",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("pairs", "data", "items", "annotations", "examples", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [data]
    return []


def discover_core_root(root: Path) -> Path:
    root = Path(root)
    if any((root / version).is_dir() for version in ("v1", "v2")):
        return root
    candidates = sorted(root.rglob("metadata.json"))
    for meta in candidates:
        parent = meta.parent.parent
        if parent.name in {"v1", "v2"}:
            return parent.parent
    return root


def iter_metadata_files(root: Path) -> Iterable[Tuple[str, str, Path]]:
    core_root = discover_core_root(root)
    for meta in sorted(core_root.glob("*/*/metadata.json")):
        version = meta.parent.parent.name
        group = meta.parent.name
        yield version, group, meta


def normalize_label(value: Any, fallback: str = "object") -> str:
    if isinstance(value, list) and value:
        return normalize_label(value[0], fallback=fallback)
    text = str(value if value is not None else fallback).strip().lower()
    return text or fallback


def entity_id(entity: Dict[str, Any], fallback: int) -> str:
    for key in ("id", "entity_id", "object_id", "obj_id", "name"):
        value = entity.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return str(fallback)


def entity_label(entity: Dict[str, Any]) -> str:
    for key in ("label", "name", "category", "class", "object", "names"):
        if key in entity:
            return normalize_label(entity.get(key))
    return "object"


def entities_for_scene(item: Dict[str, Any], scene_key: str) -> List[Dict[str, Any]]:
    scene = item.get(scene_key, {}) if isinstance(item.get(scene_key, {}), dict) else {}
    for source in (item.get("shared_entities"), item.get("entities"), scene.get("entities"), scene.get("objects")):
        if isinstance(source, list):
            return [x for x in source if isinstance(x, dict)]
    return []


def scene_image_value(scene: Dict[str, Any]) -> str:
    for key in ("image", "image_path", "filename", "file_name", "path"):
        value = scene.get(key)
        if isinstance(value, str) and value.strip() != "":
            return value.strip()
    return ""


def resolve_image_path(group_dir: Path, scene: Dict[str, Any]) -> Optional[Path]:
    raw = scene_image_value(scene)
    candidates: List[Path] = []
    if raw:
        raw_path = Path(raw)
        candidates.append(raw_path)
        candidates.append(group_dir / raw)
        candidates.append(group_dir / "image" / raw)
        candidates.append(group_dir / "images" / raw)
        candidates.append(group_dir / raw_path.name)
        candidates.append(group_dir / "image" / raw_path.name)
        candidates.append(group_dir / "images" / raw_path.name)
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _image_size_from_header(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                if width > 0 and height > 0:
                    return width, height
            if header.startswith(b"\xff\xd8"):
                handle.seek(2)
                while True:
                    marker_start = handle.read(1)
                    if marker_start != b"\xff":
                        return None
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    length_raw = handle.read(2)
                    if len(length_raw) != 2:
                        return None
                    length = int.from_bytes(length_raw, "big")
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                        data = handle.read(5)
                        if len(data) != 5:
                            return None
                        height = int.from_bytes(data[1:3], "big")
                        width = int.from_bytes(data[3:5], "big")
                        if width > 0 and height > 0:
                            return width, height
                        return None
                    handle.seek(max(0, length - 2), 1)
    except Exception:
        return None
    return None


def get_image_size(path: Optional[Path]) -> Tuple[int, int]:
    if path is None:
        return 336, 336
    header_size = _image_size_from_header(path)
    if header_size is not None:
        return header_size
    if Image is None:
        return 336, 336
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return 336, 336


def normalized_cxcywh_to_xyxy(box: Any, width: int, height: int) -> Optional[List[float]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        cx, cy, bw, bh = [float(x) for x in box]
    except Exception:
        return None
    if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
        x1 = (cx - bw / 2.0) * float(width)
        y1 = (cy - bh / 2.0) * float(height)
        x2 = (cx + bw / 2.0) * float(width)
        y2 = (cy + bh / 2.0) * float(height)
    else:
        x1, y1, x2, y2 = cx, cy, cx + bw, cy + bh
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def relation_predicate(rel: Dict[str, Any]) -> str:
    for key in ("predicate", "pred", "relation", "label", "name"):
        value = rel.get(key)
        if value is not None and str(value).strip() != "":
            return normalize_label(value, fallback="relation")
    return "relation"


def relation_endpoints(rel: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    subject = rel.get("subject_id", rel.get("subj_id", rel.get("subject", rel.get("source"))))
    obj = rel.get("object_id", rel.get("obj_id", rel.get("object", rel.get("target"))))
    subject_id = None if subject is None else str(subject).strip()
    object_id = None if obj is None else str(obj).strip()
    return subject_id, object_id


def relations_for_scene(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("relationships", "relations", "edges", "triplets"):
        value = scene.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    graph = scene.get("knowledge_graph", scene.get("graph"))
    if isinstance(graph, dict):
        for key in ("relationships", "relations", "edges", "triplets"):
            value = graph.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def stable_split_key(*parts: Any) -> float:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(2**64 - 1)


def relpath(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path, root).replace(os.sep, "/")
    except Exception:
        return str(path).replace(os.sep, "/")