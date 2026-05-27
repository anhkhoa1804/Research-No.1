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

PREDICATE_ALIASES = {
    "above": "above",
    "across": "across",
    "against": "against",
    "along": "along",
    "attached": "attached to",
    "attached to": "attached to",
    "behind": "behind",
    "belonging to": "belonging to",
    "between": "between",
    "carrying": "carrying",
    "carries": "carrying",
    "covering": "covering",
    "covers": "covering",
    "covered in": "covered in",
    "eating": "eating",
    "eats": "eating",
    "facing": "looking at",
    "turned towards": "looking at",
    "pointed at": "looking at",
    "pointing towards": "looking at",
    "angled towards": "looking at",
    "aimed at": "looking at",
    "tilted towards": "looking at",
    "directed at": "looking at",
    "staring at": "looking at",
    "watching": "watching",
    "watches": "watching",
    "observes": "looking at",
    "studies": "looking at",
    "scrutinizes": "looking at",
    "tracks": "looking at",
    "monitors": "looking at",
    "inspects": "looking at",
    "gazing at": "looking at",
    "looking at": "looking at",
    "from": "from",
    "hanging from": "hanging from",
    "has": "has",
    "holding": "holding",
    "holds": "holding",
    "in": "in",
    "inside": "in",
    "inside of": "in",
    "in front of": "in front of",
    "front of": "in front of",
    "leaning towards": "near",
    "near": "near",
    "sits near": "near",
    "next to": "next to",
    "beside": "next to",
    "adjacent to": "next to",
    "of": "of",
    "on": "on",
    "resting on": "on",
    "rests on": "on",
    "placed on": "on",
    "balances on": "on",
    "sitting on": "sitting on",
    "standing on": "standing on",
    "lying on": "lying on",
    "laying on": "laying on",
    "on back of": "on back of",
    "over": "over",
    "overlaps": "over",
    "parked on": "parked on",
    "parked next to": "next to",
    "part of": "part of",
    "playing": "playing",
    "riding": "riding",
    "rides": "riding",
    "to": "to",
    "under": "under",
    "placed under": "under",
    "beneath": "under",
    "using": "using",
    "wearing": "wearing",
    "wears": "wears",
    "with": "with",
    "wrapped around": "wrapped around",
}

CORE_RELATION_GROUPS = {"Action_Role_Reversal", "Gaze_Attention", "Occlusion_Depth", "Spatial_Containment", "Extreme_Compositional_OOD"}
CORE_NON_RELATION_GROUPS = {"Attribute_Binding"}


def _clean(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _canonical_predicate(value: Any, *, mode: str = "aliases") -> str:
    pred = _clean(value)
    if mode in {"none", "raw"}:
        return pred
    if pred in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[pred]
    if pred.endswith("ing") and pred[:-3] in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[pred[:-3]]
    if pred.endswith("ed") and pred[:-2] in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[pred[:-2]]
    if pred.endswith("s") and pred[:-1] in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[pred[:-1]]
    return pred


def _feature_tensor(output: Any, *, kind: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in (f"{kind}_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value[:, 0] if value.ndim == 3 else value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        value = output[0]
        return value[:, 0] if value.ndim == 3 else value
    if isinstance(output, dict):
        for key in (f"{kind}_embeds", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value[:, 0] if value.ndim == 3 else value
    raise TypeError(f"Unsupported CLIP {kind} output type: {type(output)}")


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


def _predicate_vocab(vg150_root: str, extra_predicates: set[str], *, mode: str = "union") -> list[str]:
    vocab = [_clean(x) for x in scan_vg150_predicate_vocab(vg150_root)]
    seen = set(vocab)
    if mode == "union":
        for pred in sorted(extra_predicates):
            if pred and pred not in seen:
                vocab.append(pred)
                seen.add(pred)
    if "relation" not in seen:
        vocab.append("relation")
    return vocab


def _text_features(clip_model: Any, processor: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = _feature_tensor(clip_model.get_text_features(**inputs), kind="text")
    return F.normalize(feats.float(), dim=-1)


def _build_oov_predicate_map(
    clip_model: Any,
    processor: Any,
    predicates: set[str],
    pred_vocab: list[str],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    oov_predicates = sorted(pred for pred in predicates if pred and pred not in set(pred_vocab))
    if not oov_predicates:
        return {}
    vocab_emb = _text_features(clip_model, processor, [pred_prompt_roles(p, direction="s2o") for p in pred_vocab], device)
    oov_emb = _text_features(clip_model, processor, [pred_prompt_roles(p, direction="s2o") for p in oov_predicates], device)
    similarities = oov_emb @ vocab_emb.T
    best_scores, best_indices = similarities.max(dim=1)
    return {
        pred: {
            "mapped_predicate": pred_vocab[int(best_indices[idx].item())],
            "similarity": float(best_scores[idx].item()),
        }
        for idx, pred in enumerate(oov_predicates)
    }


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


def _relation_queries(item: dict[str, Any], *, canonicalize: str = "aliases") -> list[dict[str, str]]:
    queries = []
    entities = entities_for_scene(item, "scene_A")
    labels = {entity_id(entity, idx): _clean(entity_label(entity)) for idx, entity in enumerate(entities)}
    for scene_key in SCENE_KEYS:
        scene = item.get(scene_key, {}) if isinstance(item.get(scene_key), dict) else {}
        for rel in relations_for_scene(scene):
            sid, oid = relation_endpoints(rel)
            raw_pred = _clean(relation_predicate(rel))
            pred = _canonical_predicate(raw_pred, mode=canonicalize)
            if not sid or not oid or pred in GENERIC_RELATIONS:
                continue
            subj, obj = labels.get(sid, _clean(sid)), labels.get(oid, _clean(oid))
            if subj in GENERIC_OBJECTS or obj in GENERIC_OBJECTS:
                continue
            queries.append({"target_scene": scene_key, "subject_id": sid, "object_id": oid, "subject": subj, "object": obj, "predicate": pred, "raw_predicate": raw_pred})
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
def _score_candidate(
    cfg: TrainConfig,
    clip_model: Any,
    processor: Any,
    model: RelationalModel,
    pred_emb: torch.Tensor,
    pred_to_idx: dict[str, int],
    scene: dict[str, Any],
    query: dict[str, str],
    device: torch.device,
    *,
    retrieval_score: str = "raw",
) -> tuple[float, dict[str, float]] | None:
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
    pair_list = [(int(s_idx), int(o_idx))]
    if retrieval_score == "directional-margin":
        pair_list.append((int(o_idx), int(s_idx)))
    _, rels, _, _, _, _ = model.forward_from_featmap(
        feat_map,
        obj_boxes_224=[boxes_224.to(device)],
        pairs=[pair_list],
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
    raw = float(logits[0, pred_idx].item())
    swapped = float(logits[1, pred_idx].item()) if retrieval_score == "directional-margin" and int(logits.shape[0]) > 1 else 0.0
    if retrieval_score == "directional-margin":
        return raw - swapped, {"raw": raw, "swapped": swapped, "directional_margin": raw - swapped}
    return raw, {"raw": raw}


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
    parser.add_argument("--retrieval-score", choices=("raw", "directional-margin"), default="raw")
    parser.add_argument("--text-target", choices=("predicate", "triplet"), default="predicate")
    parser.add_argument("--decision-rule", choices=("max", "min"), default="max")
    parser.add_argument("--predicate-vocab-mode", choices=("union", "vg150"), default="union")
    parser.add_argument("--canonicalize-predicates", choices=("aliases", "none"), default="aliases")
    parser.add_argument("--oov-map-mode", choices=("none", "text"), default="none", help="Map predicate OOVs into the active predicate vocab with CLIP text cosine.")
    parser.add_argument("--skip-oov", action="store_true")
    parser.add_argument("--group-filter", default="", help="Comma-separated CORE groups to include, e.g. Action_Role_Reversal,Gaze_Attention.")
    parser.add_argument("--relation-groups-only", action="store_true", help="Skip non-relation diagnostic groups such as Attribute_Binding.")
    parser.add_argument("--out", default="runs/core_pure_boxed_retrieval/metrics.json")
    args = parser.parse_args()

    random.seed(int(args.seed))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg, clip_model, processor, model = _load_checkpoint(Path(args.checkpoint), device, args.clip_name or None)
    cfg.eval_sgg_predicate_score_mode = str(args.score_mode)
    cfg.eval_sgg_use_predicate_classifier = str(args.score_mode) != "text"
    group_filter = {g.strip() for g in str(args.group_filter).split(",") if g.strip()}

    core_root = discover_core_root(Path(args.core_root))
    raw_items: list[tuple[str, str, Path, int, dict[str, Any]]] = []
    predicates: set[str] = set()
    for version, group, meta_path in iter_metadata_files(core_root):
        if group_filter and group not in group_filter:
            continue
        if bool(args.relation_groups_only) and group in CORE_NON_RELATION_GROUPS:
            continue
        for item_index, item in enumerate(load_metadata(meta_path)):
            item_with_group = {**item, "group": group}
            for query in _relation_queries(item_with_group, canonicalize=str(args.canonicalize_predicates)):
                predicates.add(query["predicate"])
            raw_items.append((version, group, meta_path, item_index, item_with_group))
    pred_vocab = _predicate_vocab(str(args.vg150_root), predicates, mode=str(args.predicate_vocab_mode))
    pred_to_idx = {pred: idx for idx, pred in enumerate(pred_vocab)}
    pred_emb = _text_features(clip_model, processor, [pred_prompt_roles(p, direction="s2o") for p in pred_vocab], device)
    oov_predicate_map = (
        _build_oov_predicate_map(clip_model, processor, predicates, pred_vocab, device)
        if str(args.oov_map_mode) == "text"
        else {}
    )
    classifier_classes = int(getattr(model, "predicate_classifier_classes", 0))
    classifier_vocab_mismatch = bool(str(args.score_mode) != "text" and classifier_classes != len(pred_vocab))

    total = correct = skipped = inverted_correct = ties = 0
    candidate_queries = usable_queries = covered_queries = 0
    skip_reasons: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    raw_predicate_counts: Counter[str] = Counter()
    predicate_oov = 0
    mapped_oov = 0
    mapped_oov_counts: Counter[tuple[str, str]] = Counter()
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    query_limit = int(args.max_pairs) * (2 if args.query_mode == "both" else 1) if int(args.max_pairs) > 0 else 0
    for version, group, meta_path, item_index, item in raw_items:
        if query_limit and total >= query_limit:
            break
        pair_id = str(item.get("pair_id", item.get("id", f"idx_{item_index}")))
        scenes = _scene_records(version, group, meta_path, item, item_index)
        queries = _relation_queries(item, canonicalize=str(args.canonicalize_predicates))
        candidate_queries += len(queries)
        if scenes is None:
            skipped += 1
            skip_reasons["missing_scene_records"] += 1
            continue
        if not queries:
            skipped += 1
            skip_reasons["missing_queries"] += 1
            continue
        if args.query_mode == "random":
            queries = [random.choice(queries)]
        for query in queries:
            usable_queries += 1
            predicate_counts[query["predicate"]] += 1
            raw_predicate_counts[query.get("raw_predicate", query["predicate"])] += 1
            scoring_query = query
            local_pred_emb = pred_emb
            local_pred_to_idx = pred_to_idx
            if args.text_target == "triplet":
                local_pred_emb = _text_features(
                    clip_model,
                    processor,
                    [f"a scene where {query['subject']} {query['predicate']} {query['object']}"],
                    device,
                )
                local_pred_to_idx = {query["predicate"]: 0}
            elif query["predicate"] not in pred_to_idx:
                predicate_oov += 1
                mapping = oov_predicate_map.get(query["predicate"])
                if mapping is not None:
                    mapped_predicate = str(mapping["mapped_predicate"])
                    scoring_query = {**query, "predicate": mapped_predicate, "original_predicate": query["predicate"], "oov_mapping_similarity": float(mapping["similarity"])}
                    mapped_oov += 1
                    mapped_oov_counts[(query["predicate"], mapped_predicate)] += 1
                if bool(args.skip_oov) and mapping is None:
                    skipped += 1
                    skip_reasons["predicate_oov"] += 1
                    continue
            if scoring_query["predicate"] in pred_to_idx or args.text_target == "triplet":
                covered_queries += 1
            scored = {
                scene_key: _score_candidate(
                    cfg,
                    clip_model,
                    processor,
                    model,
                    local_pred_emb,
                    local_pred_to_idx,
                    scenes[scene_key],
                    scoring_query,
                    device,
                    retrieval_score=str(args.retrieval_score),
                )
                for scene_key in SCENE_KEYS
            }
            if any(value is None for value in scored.values()):
                skipped += 1
                skip_reasons["unscorable_candidate"] += 1
                continue
            scores = {scene_key: float(scored[scene_key][0]) for scene_key in SCENE_KEYS}
            score_details = {scene_key: scored[scene_key][1] for scene_key in SCENE_KEYS}
            if float(scores["scene_A"]) == float(scores["scene_B"]):
                ties += 1
            pred_scene = (
                max(SCENE_KEYS, key=lambda key: float(scores[key]))
                if args.decision_rule == "max"
                else min(SCENE_KEYS, key=lambda key: float(scores[key]))
            )
            inverted_pred_scene = min(SCENE_KEYS, key=lambda key: float(scores[key])) if args.decision_rule == "max" else max(SCENE_KEYS, key=lambda key: float(scores[key]))
            ok = pred_scene == query["target_scene"]
            inverted_ok = inverted_pred_scene == query["target_scene"]
            total += 1
            correct += int(ok)
            inverted_correct += int(inverted_ok)
            by_group[group]["total"] += 1
            by_group[group]["correct"] += int(ok)
            distractor = "scene_B" if query["target_scene"] == "scene_A" else "scene_A"
            by_group[group]["margin_sum"] += float(scores[query["target_scene"]]) - float(scores[distractor])
            if len(examples) < 50:
                examples.append({"pair_id": pair_id, "group": group, "query": scoring_query, "scores": scores, "score_details": score_details, "pred_scene": pred_scene, "decision_rule": str(args.decision_rule), "descriptions": {k: scenes[k]["description"] for k in SCENE_KEYS}})
            if query_limit and total >= query_limit:
                break

    report = {
        "core_root": str(core_root),
        "checkpoint": str(args.checkpoint),
        "num_queries": total,
        "candidate_queries": candidate_queries,
        "usable_queries": usable_queries,
        "covered_queries": covered_queries,
        "coverage_rate": covered_queries / usable_queries if usable_queries else 0.0,
        "skipped": skipped,
        "skip_reasons": dict(skip_reasons),
        "accuracy": correct / total if total else 0.0,
        "inverted_accuracy": inverted_correct / total if total else 0.0,
        "ties": ties,
        "score_mode": str(args.score_mode),
        "retrieval_score": str(args.retrieval_score),
        "text_target": str(args.text_target),
        "query_mode": str(args.query_mode),
        "decision_rule": str(args.decision_rule),
        "predicate_vocab_mode": str(args.predicate_vocab_mode),
        "canonicalize_predicates": str(args.canonicalize_predicates),
        "oov_map_mode": str(args.oov_map_mode),
        "skip_oov": bool(args.skip_oov),
        "group_filter": sorted(group_filter),
        "relation_groups_only": bool(args.relation_groups_only),
        "classifier_classes": classifier_classes,
        "predicate_vocab_size": len(pred_vocab),
        "classifier_vocab_mismatch": classifier_vocab_mismatch,
        "predicate_oov": predicate_oov,
        "mapped_oov": mapped_oov,
        "top_oov_mappings": [
            {
                "predicate": source,
                "mapped_predicate": target,
                "count": count,
                "similarity": float(oov_predicate_map.get(source, {}).get("similarity", 0.0)),
            }
            for (source, target), count in mapped_oov_counts.most_common(30)
        ],
        "top_predicates": predicate_counts.most_common(30),
        "top_raw_predicates": raw_predicate_counts.most_common(30),
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