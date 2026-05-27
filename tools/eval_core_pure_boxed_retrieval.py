#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from openvocab_rel.clip_utils import configure_clip
from openvocab_rel.config import TrainConfig
from openvocab_rel.datasets.vg150_loader import scan_vg150_predicate_vocab
from openvocab_rel.evals import _relation_predicate_logits
from openvocab_rel.geometry import preprocess_boxes_to_clip224
from openvocab_rel.models.relational_model import RelationalModel
from openvocab_rel.prompts import pred_prompt_roles

from core_utils import (
    SCENE_KEYS,
    discover_core_root,
    entities_for_scene,
    entity_id,
    entity_label,
    get_image_size,
    grounding_box,
    grounding_entity_id,
    iter_metadata_files,
    load_metadata,
    normalized_cxcywh_to_xyxy,
    relation_endpoints,
    relation_predicate,
    relations_for_scene,
    resolve_image_path,
)

GENERIC_OBJECTS = {"", "object", "objects", "thing", "entity", "unknown", "none", "background", "__background__"}
GENERIC_RELATIONS = {"", "relation", "relationships", "unknown", "none", "background", "__background__", "no relation", "no interaction"}


def _clean(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _load_checkpoint(path: Path, device: torch.device, clip_name: str | None) -> tuple[TrainConfig, Any, Any, RelationalModel]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = TrainConfig()
    if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
        for key, value in ckpt["cfg"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    if clip_name:
        cfg.clip_name = clip_name
    cfg.device = str(device)
    clip_model, processor, clip_vision_dim, text_dim = configure_clip(cfg.clip_name, device)
    model = RelationalModel(cfg=cfg, clip_vision_dim=clip_vision_dim, text_dim=text_dim)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = {str(k).replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if isinstance(ckpt, dict) and "clip" in ckpt:
        clip_state = {str(k).replace("module.", ""): v for k, v in ckpt["clip"].items()}
        clip_model.load_state_dict(clip_state, strict=False)
    model.to(device).eval()
    clip_model.eval()
    print(json.dumps({"loaded_checkpoint": str(path), "missing_model_keys": len(missing), "unexpected_model_keys": len(unexpected)}, indent=2))
    return cfg, clip_model, processor, model


def _predicate_vocab(vg150_root: str, extra_predicates: set[str]) -> list[str]:
    vocab = [_clean(x) for x in scan_vg150_predicate_vocab(vg150_root)]
    seen = set(vocab)
    for pred in sorted(extra_predicates):
        if pred and pred not in seen:
            vocab.append(pred)
            seen.add(pred)
    if "relation" not in seen:
        vocab.append("relation")
    return vocab


def _text_features(clip_model: Any, processor: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = clip_model.get_text_features(**inputs)
    return F.normalize(feats.float(), dim=-1)


def _scene_records(version: str, group: str, meta_path: Path, item: dict[str, Any], item_index: int) -> dict[str, dict[str, Any]] | None:
    pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
    item.setdefault("group", group)
    entities = entities_for_scene(item, "scene_A")
    entity_labels = {entity_id(entity, idx): _clean(entity_label(entity)) for idx, entity in enumerate(entities)}
    records: dict[str, dict[str, Any]] = {}
    for scene_key in SCENE_KEYS:
        scene = item.get(scene_key, {}) if isinstance(item.get(scene_key), dict) else {}
        image_path = resolve_image_path(meta_path.parent, scene, pair_id=pair_id, scene_key=scene_key, group=group)
        if image_path is None:
            return None
        width, height = get_image_size(image_path)
        box_by_entity: dict[str, list[float]] = {}
        for grounded in scene.get("grounding", []) if isinstance(scene.get("grounding"), list) else []:
            if not isinstance(grounded, dict):
                continue
            gid = grounding_entity_id(grounded)
            if gid is None:
                continue
            box = normalized_cxcywh_to_xyxy(grounding_box(grounded), width, height)
            if box is not None:
                box_by_entity[gid] = box
        objects = []
        for eid, label in entity_labels.items():
            if label not in GENERIC_OBJECTS and eid in box_by_entity:
                objects.append({"entity_id": eid, "label": label, "box": box_by_entity[eid]})
        if len(objects) < 2:
            return None
        records[scene_key] = {"image": image_path, "objects": objects, "description": scene.get("description", "")}
    return records


def _relation_queries(item: dict[str, Any]) -> list[dict[str, str]]:
    queries = []
    entities = entities_for_scene(item, "scene_A")
    labels = {entity_id(entity, idx): _clean(entity_label(entity)) for idx, entity in enumerate(entities)}
    for scene_key in SCENE_KEYS:
        scene = item.get(scene_key, {}) if isinstance(item.get(scene_key), dict) else {}
        for rel in relations_for_scene(scene):
            sid, oid = relation_endpoints(rel)
            pred = _clean(relation_predicate(rel))
            if not sid or not oid or pred in GENERIC_RELATIONS:
                continue
            subj, obj = labels.get(sid, _clean(sid)), labels.get(oid, _clean(oid))
            if subj in GENERIC_OBJECTS or obj in GENERIC_OBJECTS:
                continue
            queries.append({"target_scene": scene_key, "subject_id": sid, "object_id": oid, "subject": subj, "object": obj, "predicate": pred})
    return queries


def _find_object_index(objects: list[dict[str, Any]], entity_id_value: str, label: str) -> int | None:
    for idx, obj in enumerate(objects):
        if obj["entity_id"] == entity_id_value:
            return idx
    for idx, obj in enumerate(objects):
        if obj["label"] == label:
            return idx
    return None


@torch.no_grad()
def _score_candidate(cfg: TrainConfig, clip_model: Any, processor: Any, model: RelationalModel, pred_emb: torch.Tensor, pred_to_idx: dict[str, int], scene: dict[str, Any], query: dict[str, str], device: torch.device) -> float | None:
    s_idx = _find_object_index(scene["objects"], query["subject_id"], query["subject"])
    o_idx = _find_object_index(scene["objects"], query["object_id"], query["object"])
    if s_idx is None or o_idx is None or s_idx == o_idx:
        return None
    with Image.open(scene["image"]) as image:
        pil = image.convert("RGB")
        pixel = processor(images=[pil], return_tensors="pt")["pixel_values"].to(device)
        boxes = torch.tensor([obj["box"] for obj in scene["objects"]], dtype=torch.float32)
        boxes_224, _ = preprocess_boxes_to_clip224(pil, boxes, out_size=int(cfg.clip_input_res))
    vision_out = clip_model.vision_model(pixel_values=pixel)
    tokens = vision_out.last_hidden_state[:, 1:, :]
    side = int(math.sqrt(int(tokens.shape[1])))
    feat_map = tokens.transpose(1, 2).reshape(1, int(tokens.shape[2]), side, side)
    _, rels, _, _, _, _ = model.forward_from_featmap(
        feat_map,
        obj_boxes_224=[boxes_224.to(device)],
        pairs=[[(int(s_idx), int(o_idx))]],
        return_swapped=False,
        return_kept=True,
        learned_prune_k_override=0,
    )
    if not rels or int(rels[0].shape[0]) == 0:
        return None
    logits = _relation_predicate_logits(cfg, model, rels[0], pred_emb)
    pred_idx = pred_to_idx.get(query["predicate"])
    if pred_idx is None:
        return None
    return float(logits[0, pred_idx].item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CORE retrieval with PURE using metadata boxes and labels.")
    parser.add_argument("--core-root", default="datasets/core")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vg150-root", default="datasets")
    parser.add_argument("--clip-name", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-mode", choices=("random", "both"), default="random")
    parser.add_argument("--score-mode", choices=("classifier", "text", "ensemble", "auto"), default="classifier")
    parser.add_argument("--out", default="runs/core_pure_boxed_retrieval/metrics.json")
    args = parser.parse_args()

    random.seed(int(args.seed))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg, clip_model, processor, model = _load_checkpoint(Path(args.checkpoint), device, args.clip_name or None)
    cfg.eval_sgg_predicate_score_mode = str(args.score_mode)
    cfg.eval_sgg_use_predicate_classifier = str(args.score_mode) != "text"

    core_root = discover_core_root(Path(args.core_root))
    raw_items: list[tuple[str, str, Path, int, dict[str, Any]]] = []
    predicates: set[str] = set()
    for version, group, meta_path in iter_metadata_files(core_root):
        for item_index, item in enumerate(load_metadata(meta_path)):
            item_with_group = {**item, "group": group}
            for query in _relation_queries(item_with_group):
                predicates.add(query["predicate"])
            raw_items.append((version, group, meta_path, item_index, item_with_group))
    pred_vocab = _predicate_vocab(str(args.vg150_root), predicates)
    pred_to_idx = {pred: idx for idx, pred in enumerate(pred_vocab)}
    pred_emb = _text_features(clip_model, processor, [pred_prompt_roles(p, direction="s2o") for p in pred_vocab], device)

    total = correct = skipped = 0
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    query_limit = int(args.max_pairs) * (2 if args.query_mode == "both" else 1) if int(args.max_pairs) > 0 else 0
    for version, group, meta_path, item_index, item in raw_items:
        if query_limit and total >= query_limit:
            break
        pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
        scenes = _scene_records(version, group, meta_path, item, item_index)
        queries = _relation_queries(item)
        if scenes is None or not queries:
            skipped += 1
            continue
        if args.query_mode == "random":
            queries = [random.choice(queries)]
        for query in queries:
            scores = {scene_key: _score_candidate(cfg, clip_model, processor, model, pred_emb, pred_to_idx, scenes[scene_key], query, device) for scene_key in SCENE_KEYS}
            if any(value is None for value in scores.values()):
                skipped += 1
                continue
            pred_scene = max(SCENE_KEYS, key=lambda key: float(scores[key]))
            ok = pred_scene == query["target_scene"]
            total += 1
            correct += int(ok)
            by_group[group]["total"] += 1
            by_group[group]["correct"] += int(ok)
            distractor = "scene_B" if query["target_scene"] == "scene_A" else "scene_A"
            by_group[group]["margin_sum"] += float(scores[query["target_scene"]]) - float(scores[distractor])
            if len(examples) < 50:
                examples.append({"pair_id": pair_id, "group": group, "query": query, "scores": scores, "pred_scene": pred_scene, "descriptions": {k: scenes[k]["description"] for k in SCENE_KEYS}})
            if query_limit and total >= query_limit:
                break

    report = {
        "core_root": str(core_root),
        "checkpoint": str(args.checkpoint),
        "num_queries": total,
        "skipped": skipped,
        "accuracy": correct / total if total else 0.0,
        "score_mode": str(args.score_mode),
        "query_mode": str(args.query_mode),
        "by_group": {
            group: {"accuracy": c["correct"] / c["total"] if c["total"] else 0.0, "mean_margin": c["margin_sum"] / c["total"] if c["total"] else 0.0, **dict(c)}
            for group, c in sorted(by_group.items())
        },
        "examples": examples,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()