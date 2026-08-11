"""
VG150-specific data loader for PURE-SPOA training.
Supports both local file loading and Hugging Face streaming.
Standardized for 150 object classes and 50 predicate classes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import io
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, WeightedRandomSampler, get_worker_info

from ..geometry import geom_feats, preprocess_boxes_to_clip224
from ..prompts import counterfactual_prompt_variants, is_symmetric_predicate, pred_prompt_roles


def _has_local_vg150_files(vg150_root: str) -> bool:
    root = str(vg150_root)
    raw_vg_required = [
        os.path.join(root, "VG-SGG-dicts-with-attri.json"),
        os.path.join(root, "image_data.json"),
        os.path.join(root, "scene_graphs.json"),
        os.path.join(root, "images"),
    ]
    jsonl_ready = (
        os.path.exists(os.path.join(root, "train.jsonl"))
        and os.path.exists(os.path.join(root, "validation.jsonl"))
    )
    return all(os.path.exists(path) for path in raw_vg_required) or jsonl_ready


def _has_local_jsonl_files(vg150_root: str) -> bool:
    root = str(vg150_root)
    return (
        os.path.exists(os.path.join(root, "train.jsonl"))
        and os.path.exists(os.path.join(root, "validation.jsonl"))
    )


def _stable_pair_hash(subject_name: str, object_name: str) -> int:
    import hashlib

    payload = f"{str(subject_name).strip().lower()}||{str(object_name).strip().lower()}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=True)


def _prepare_rel_payload(
    rel_preds: List[str],
    rel_pair_names: List[Tuple[str, str]],
    pred_to_idx: Dict[str, int],
    rel_pos_mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    pred_strs = [
        (pred if pred in pred_to_idx else "relation")
        for pred in [str(p).strip().lower() if str(p).strip() != "" else "relation" for p in rel_preds]
    ]
    pair_names = [
        (
            str(s).strip().lower() if str(s).strip() != "" else "subject",
            str(o).strip().lower() if str(o).strip() != "" else "object",
        )
        for s, o in rel_pair_names
    ]
    
    rel_texts = [pred_prompt_roles(p, direction="s2o") for p in pred_strs]
    rel_role_swap_texts = [pred_prompt_roles(p, direction="o2s") for p in pred_strs]

    rel_cf_attr_texts: List[str] = []
    rel_non_sym_mask: List[bool] = []
    for p, (s_name, o_name) in zip(pred_strs, pair_names):
        rel_cf_attr_texts.append(
            counterfactual_prompt_variants(p, s_name, o_name, direction="s2o").get("with_names_swapped", pred_prompt_roles(p, direction="s2o"))
        )
        rel_non_sym_mask.append(not is_symmetric_predicate(p))

    rel_pred_ids = torch.tensor([int(pred_to_idx.get(p, -1)) for p in pred_strs], dtype=torch.long)
    rel_pair_hashes = torch.tensor(
        [_stable_pair_hash(s_name, o_name) for s_name, o_name in pair_names],
        dtype=torch.long,
    )
    if rel_pos_mask is None:
        rel_pos_mask_t = torch.ones(len(pred_strs), dtype=torch.bool)
    else:
        rel_pos_mask_t = torch.tensor([bool(x) for x in rel_pos_mask], dtype=torch.bool)
    rel_non_sym_mask_t = torch.tensor(rel_non_sym_mask, dtype=torch.bool)

    return {
        "rel_preds": pred_strs,
        "rel_pair_names": pair_names,
        "rel_texts": rel_texts,
        "rel_role_swap_texts": rel_role_swap_texts,
        "rel_cf_attr_texts": rel_cf_attr_texts,
        "rel_pred_ids": rel_pred_ids,
        "rel_pair_hashes": rel_pair_hashes,
        "rel_pos_mask": rel_pos_mask_t,
        "rel_non_sym_mask": rel_non_sym_mask_t,
        "rel_is_pos": rel_pos_mask_t.float(),
    }


def _object_names_from_records(objects: List[Any], max_objects: int) -> List[str]:
    names: List[str] = []
    for obj in objects[: max(0, int(max_objects))]:
        if isinstance(obj, dict):
            names.append(str(obj.get("names", ["object"])[0]).strip().lower())
        else:
            names.append(str(obj).strip().lower())
    return names


def _build_object_id_lookup(objects: List[Any], num_objects: int) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for idx, obj in enumerate(objects[: max(0, int(num_objects))]):
        lookup[int(idx)] = int(idx)
        if not isinstance(obj, dict):
            continue
        for key in ("object_id", "obj_id", "id"):
            raw = obj.get(key, None)
            if isinstance(raw, (int, float)):
                lookup[int(raw)] = int(idx)
    return lookup


def _resolve_rel_object_index(raw_idx: Any, object_id_lookup: Dict[int, int], num_objects: int) -> Optional[int]:
    if not isinstance(raw_idx, (int, float)):
        return None
    raw_int = int(raw_idx)
    mapped = object_id_lookup.get(raw_int, raw_int)
    if 0 <= mapped < int(num_objects):
        return int(mapped)
    return None


def _build_relation_entries(
    obj_boxes_t: torch.Tensor,
    obj_names: List[str],
    relationships: List[Dict[str, Any]],
    pred_to_idx: Dict[str, int],
    use_all_pairs: bool,
    max_pairs: int,
    negative_pair_ratio: float = -1.0,
) -> Tuple[List[Tuple[int, int, None, List[float]]], Dict[str, Any]]:
    num_objects = min(int(obj_boxes_t.shape[0]), len(obj_names))
    if num_objects <= 1:
        return [], _prepare_rel_payload([], [], pred_to_idx)

    object_id_lookup = _build_object_id_lookup([{"names": [n]} for n in obj_names], num_objects)
    positive_entries: List[Tuple[int, int, None, List[float]]] = []
    positive_preds: List[str] = []
    positive_pair_names: List[Tuple[str, str]] = []
    positive_keys = set()

    for rel in relationships:
        subj_idx = _resolve_rel_object_index(rel.get("subject_id", -1), object_id_lookup, num_objects)
        obj_idx = _resolve_rel_object_index(rel.get("object_id", -1), object_id_lookup, num_objects)
        if subj_idx is None or obj_idx is None or subj_idx == obj_idx:
            continue
        pred_name = str(rel.get("predicate", "relation")).strip().lower() or "relation"
        g = geom_feats(obj_boxes_t[subj_idx].tolist(), obj_boxes_t[obj_idx].tolist())
        positive_entries.append((int(subj_idx), int(obj_idx), None, g))
        positive_preds.append(pred_name)
        positive_pair_names.append((obj_names[subj_idx], obj_names[obj_idx]))
        positive_keys.add((int(subj_idx), int(obj_idx)))

    if not bool(use_all_pairs):
        if int(max_pairs) > 0 and len(positive_entries) > int(max_pairs):
            positive_entries = positive_entries[: int(max_pairs)]
            positive_preds = positive_preds[: int(max_pairs)]
            positive_pair_names = positive_pair_names[: int(max_pairs)]
        payload = _prepare_rel_payload(positive_preds, positive_pair_names, pred_to_idx)
        return positive_entries, payload

    pairs = list(positive_entries)
    rel_preds = list(positive_preds)
    rel_pair_names = list(positive_pair_names)
    rel_pos_mask = [True] * len(positive_entries)

    max_negatives = -1
    if float(negative_pair_ratio) >= 0.0:
        max_negatives = int(math.ceil(max(0, len(positive_entries)) * float(negative_pair_ratio)))
    negatives_added = 0

    for subj_idx in range(num_objects):
        for obj_idx in range(num_objects):
            if subj_idx == obj_idx or (subj_idx, obj_idx) in positive_keys:
                continue
            if int(max_pairs) > 0 and len(pairs) >= int(max_pairs):
                break
            if max_negatives >= 0 and negatives_added >= max_negatives:
                break
            g = geom_feats(obj_boxes_t[subj_idx].tolist(), obj_boxes_t[obj_idx].tolist())
            pairs.append((int(subj_idx), int(obj_idx), None, g))
            negatives_added += 1
            rel_preds.append("relation")
            rel_pair_names.append((obj_names[subj_idx], obj_names[obj_idx]))
            rel_pos_mask.append(False)
        if int(max_pairs) > 0 and len(pairs) >= int(max_pairs):
            break
        if max_negatives >= 0 and negatives_added >= max_negatives:
            break

    payload = _prepare_rel_payload(rel_preds, rel_pair_names, pred_to_idx, rel_pos_mask=rel_pos_mask)
    return pairs, payload


def negative_pair_ratio_is_inert(use_all_pairs: bool, negative_pair_ratio: float) -> bool:
    """True iff negative_pair_ratio has zero effect on the pairs
    _build_relation_entries constructs, given these two settings.

    _build_relation_entries's `if not bool(use_all_pairs):` branch (above)
    returns positive-only pairs immediately, before negative_pair_ratio is
    ever read -- so negative_pair_ratio only has any effect when
    use_all_pairs is truthy. config.py's own field comment
    ("0 keeps positives only; positive values sample negatives per
    positive") describes negative_pair_ratio as if IT were the
    positives-only control; in practice use_all_pairs=False silently
    overrides that regardless of negative_pair_ratio's value. See
    docs/known_issues.md's "negative_pair_ratio is a fully inert config
    flag" entry -- this predicate exists so VG150DataLoader can warn once,
    loudly, instead of leaving this silent, without changing which pairs
    actually get constructed (that would be a training-protocol change,
    out of scope for this fix).
    """
    return (not bool(use_all_pairs)) and float(negative_pair_ratio) != 0.0


@dataclass
class VG150LoaderConfig:
    """Configuration for VG150 data loading."""
    # Local data directory
    vg150_root: str = "datasets"
    source: str = "auto"
    
    # Dataset split
    split: str = "train"  # or "val"/"test"
    
    # Streaming and batch config
    streaming: bool = True
    hf_dataset_id: str = ""
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True
    
    # Sampling
    seed: int = 0
    shuffle_buffer_size: int = 5000
    samples_per_epoch: int = 10000
    max_images: int = 0
    use_rfs: bool = True
    rfs_t: float = 0.001
    use_all_pairs: bool = True
    negative_pair_ratio: float = 2.0
    
    # VG150 specific (150 objects, 50 predicates)
    max_objects: int = 30
    max_pairs: int = 64
    predicate_sampler_enabled: bool = False
    predicate_sampler_power: float = 0.75
    predicate_sampler_max_weight: float = 20.0


def _load_vg150_vocab(vg150_root: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load object and predicate vocabularies with silent fallback."""
    dicts_path = os.path.join(vg150_root, "VG-SGG-dicts-with-attri.json")
    clean_obj_path = os.path.join(vg150_root, "vocabulary", "objects.json")
    clean_pred_path = os.path.join(vg150_root, "vocabulary", "predicates.json")
    label_to_idx, pred_to_idx = {}, {}
    
    if os.path.exists(dicts_path):
        with open(dicts_path, "r") as f:
            dicts_data = json.load(f)
        if "idx_to_label" in dicts_data:
            label_to_idx = {str(n).lower(): int(i) for i, n in dicts_data["idx_to_label"].items()}
        if "idx_to_predicate" in dicts_data:
            pred_names = [
                str(n).strip().lower()
                for _, n in sorted(dicts_data["idx_to_predicate"].items(), key=lambda kv: int(kv[0]))
                if str(n).strip() != ""
            ]
            pred_to_idx = {name: idx for idx, name in enumerate(pred_names)}
    elif os.path.exists(clean_obj_path) or os.path.exists(clean_pred_path):
        def _load_clean_vocab(path: str) -> Dict[str, int]:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                names = [str(x).strip().lower() for x in payload if str(x).strip() != ""]
                return {name: idx for idx, name in enumerate(names)}
            if isinstance(payload, dict):
                for nested_key in ("idx_to_predicate", "idx_to_label", "idx_to_object", "idx_to_objects"):
                    nested = payload.get(nested_key)
                    if isinstance(nested, dict):
                        names = [
                            str(v).strip().lower()
                            for _, v in sorted(nested.items(), key=lambda kv: int(kv[0]))
                            if str(v).strip() != ""
                        ]
                        return {name: idx for idx, name in enumerate(names)}
                    if isinstance(nested, list):
                        names = [str(x).strip().lower() for x in nested if str(x).strip() != ""]
                        return {name: idx for idx, name in enumerate(names)}
                if all(str(k).lstrip("-").isdigit() for k in payload.keys()):
                    names = [
                        str(v).strip().lower()
                        for _, v in sorted(payload.items(), key=lambda kv: int(kv[0]))
                        if str(v).strip() != ""
                    ]
                    return {name: idx for idx, name in enumerate(names)}
                out: Dict[str, int] = {}
                for key, value in payload.items():
                    name = str(key).strip().lower()
                    if name == "":
                        continue
                    try:
                        out[name] = int(value)
                    except (TypeError, ValueError):
                        out[name] = len(out)
                return out
            return {}

        label_to_idx = _load_clean_vocab(clean_obj_path)
        pred_to_idx = _load_clean_vocab(clean_pred_path)
    else:
        # Standard 50 predicates for VG150
        preds = [
            'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 
            'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 
            'flying in', 'for', 'from', 'hanging from', 'has', 'holding', 'in', 
            'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 
            'near', 'next to', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 
            'part of', 'playing', 'riding', 'sitting on', 'standing on', 'to', 'under', 
            'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with', 
            'wrapped around'
        ]
        pred_to_idx = {p: i for i, p in enumerate(preds)}
    return label_to_idx, pred_to_idx



def _relationship_predicates_from_row(row: Any) -> List[str]:
    if not isinstance(row, dict):
        return []
    rels = row.get("relationships", row.get("relations", row.get("rels", [])))
    out: List[str] = []
    for rel in rels if isinstance(rels, list) else []:
        if isinstance(rel, dict):
            pred = str(rel.get("predicate", rel.get("pred", ""))).strip().lower()
            if pred:
                out.append(pred)
    return out


def _dataset_rows_for_sampling(dataset: Dataset) -> List[Any]:
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list):
        return rows
    scene_graphs = getattr(dataset, "scene_graphs", None)
    images = getattr(dataset, "images", None)
    valid_indices = getattr(dataset, "valid_indices", None)
    if isinstance(scene_graphs, dict) and isinstance(images, list) and isinstance(valid_indices, list):
        out: List[Any] = []
        for img_idx in valid_indices:
            if int(img_idx) >= len(images):
                continue
            img_id = images[int(img_idx)].get("image_id")
            out.append(scene_graphs.get(img_id, {}))
        return out
    return []


def _predicate_balanced_sampler(dataset: Dataset, cfg: VG150LoaderConfig) -> Optional[WeightedRandomSampler]:
    rows = _dataset_rows_for_sampling(dataset)
    if len(rows) == 0:
        return None
    row_preds = [_relationship_predicates_from_row(row) for row in rows]
    counts: Dict[str, int] = {}
    for preds in row_preds:
        for pred in set(preds):
            counts[pred] = int(counts.get(pred, 0)) + 1
    if len(counts) == 0:
        return None
    mean_count = sum(float(v) for v in counts.values()) / float(max(1, len(counts)))
    power = max(0.0, float(getattr(cfg, "predicate_sampler_power", 0.75)))
    max_weight = max(1.0, float(getattr(cfg, "predicate_sampler_max_weight", 20.0)))
    weights: List[float] = []
    for preds in row_preds:
        if len(preds) == 0:
            weights.append(1.0)
            continue
        rare_weight = max((mean_count / max(1.0, float(counts.get(pred, 1)))) ** power for pred in set(preds))
        weights.append(float(min(max_weight, max(1.0, rare_weight))))
    if len(weights) == 0 or max(weights) <= 1.0:
        return None
    num_samples = int(getattr(cfg, "samples_per_epoch", 0) or 0)
    if num_samples <= 0:
        num_samples = len(weights)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def _collate_identity(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identity collation function (returns batch as-is)."""
    return batch


def _safe_image(image_path: str) -> Image.Image:
    """Load image or return blank image on error."""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (336, 336), color=(128, 128, 128))


class VG150LocalDataset(Dataset):
    """Local file-based VG150 dataset with scene graph annotations."""

    def __init__(
        self,
        cfg: VG150LoaderConfig,
        label_to_idx: Dict[str, int],
        pred_to_idx: Dict[str, int],
        processor=None,
        clip_input_res: int = 336,
    ):
        self.cfg = cfg
        self.label_to_idx = label_to_idx
        self.pred_to_idx = pred_to_idx
        self.processor = processor
        self.clip_input_res = int(clip_input_res)

        # Load image metadata and scene graphs
        self.images = self._load_images()
        self.scene_graphs = self._load_scene_graphs()
        self.valid_indices = self._build_valid_indices()

    def _build_valid_indices(self) -> List[int]:
        valid: List[int] = []
        pred_counts: Dict[str, int] = {}
        total_rels = 0

        for img_idx, img_meta in enumerate(self.images):
            img_id = img_meta.get("image_id")
            sg = self.scene_graphs.get(img_id, {"objects": [], "relationships": []})
            objects = sg.get("objects", [])
            relationships = sg.get("relationships", [])
            
            if len(objects) > 0 and len(relationships) > 0:
                valid.append(img_idx)
                if getattr(self.cfg, "use_rfs", False) and str(self.cfg.split).strip().lower() == "train":
                    for rel in relationships:
                        p = str(rel.get("predicate", "")).strip().lower()
                        if p != "":
                            pred_counts[p] = pred_counts.get(p, 0) + 1
                            total_rels += 1

        max_images = int(getattr(self.cfg, "max_images", 0) or 0)
        if max_images > 0:
            valid = valid[:max_images]

        if getattr(self.cfg, "use_rfs", False) and str(self.cfg.split).strip().lower() == "train" and total_rels > 0:
            t = float(getattr(self.cfg, "rfs_t", 0.001))
            rfs_valid: List[int] = []
            import random
            
            for img_idx in valid:
                img_id = self.images[img_idx].get("image_id")
                sg = self.scene_graphs.get(img_id, {})
                max_r = 1.0
                
                for rel in sg.get("relationships", []):
                    p = str(rel.get("predicate", "")).strip().lower()
                    if p != "":
                        f_c = pred_counts.get(p, 0) / float(total_rels)
                        r_c = math.sqrt(t / f_c) if f_c > 0 else 1.0
                        max_r = max(max_r, r_c)
                
                repeats = int(math.floor(max_r))
                if random.random() < (max_r - repeats):
                    repeats += 1
                
                for _ in range(max(1, repeats)):
                    rfs_valid.append(img_idx)
            return rfs_valid[:max_images] if max_images > 0 else rfs_valid

        return valid

    def _load_images(self) -> List[Dict[str, Any]]:
        """Load image metadata."""
        img_path = os.path.join(self.cfg.vg150_root, "image_data.json")
        with open(img_path, "r") as f:
            images = json.load(f)
        return images if isinstance(images, list) else []

    def _load_scene_graphs(self) -> Dict[int, Dict[str, Any]]:
        """Load scene graph annotations."""
        sg_path = os.path.join(self.cfg.vg150_root, "scene_graphs.json")
        scene_graphs = {}

        if os.path.exists(sg_path):
            with open(sg_path, "r") as f:
                sgs = json.load(f)

            if isinstance(sgs, list):
                for sg in sgs:
                    if isinstance(sg, dict) and "image_id" in sg:
                        scene_graphs[int(sg["image_id"])] = sg

        return scene_graphs

    def _process_example(self, img_idx: int) -> Optional[Dict[str, Any]]:
        if img_idx >= len(self.images):
            return None

        img_meta = self.images[img_idx]
        img_id = img_meta.get("image_id")

        sg = self.scene_graphs.get(img_id, {"objects": [], "relationships": []})
        objects = sg.get("objects", [])
        relationships = sg.get("relationships", [])

        if len(objects) == 0 or len(relationships) == 0:
            return None

        img_path = os.path.join(self.cfg.vg150_root, "images", img_meta.get("url", "").split("/")[-1])
        image = _safe_image(img_path) if os.path.exists(img_path) else None

        obj_boxes = []
        kept_objects = list(objects[: int(self.cfg.max_objects)])
        obj_names = []
        for obj in kept_objects:
            x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 1), obj.get("h", 1)
            obj_boxes.append([x, y, x + w, y + h])
            obj_name = str(obj.get("names", ["object"])[0]).strip().lower()
            obj_names.append(obj_name)

        if len(obj_boxes) == 0:
            return None

        obj_boxes_t = torch.tensor(obj_boxes, dtype=torch.float32)

        if image is not None:
            img_w, img_h = image.size
        else:
            img_w, img_h = float(self.clip_input_res), float(self.clip_input_res)

        obj_boxes_norm = obj_boxes_t.clone()
        obj_boxes_norm[:, [0, 2]] /= float(img_w)
        obj_boxes_norm[:, [1, 3]] /= float(img_h)
        obj_boxes_norm = obj_boxes_norm.clamp(0.0, 1.0)

        MIN_AREA_PX = 100.0
        MAX_CENTER_DIST_RATIO = 1.5
        img_diag = math.sqrt(img_w ** 2 + img_h ** 2)

        object_id_lookup = _build_object_id_lookup(kept_objects, len(obj_boxes))
        filtered_relationships: List[Dict[str, Any]] = []
        for rel in relationships:
            subj_idx = _resolve_rel_object_index(rel.get("subject_id", -1), object_id_lookup, len(obj_boxes))
            obj_idx = _resolve_rel_object_index(rel.get("object_id", -1), object_id_lookup, len(obj_boxes))
            if subj_idx is None or obj_idx is None or subj_idx == obj_idx:
                continue

            s_box = obj_boxes_t[subj_idx]
            o_box = obj_boxes_t[obj_idx]

            s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
            o_area = (o_box[2] - o_box[0]) * (o_box[3] - o_box[1])

            if s_area < MIN_AREA_PX or o_area < MIN_AREA_PX:
                continue

            s_cx, s_cy = (s_box[0] + s_box[2]) * 0.5, (s_box[1] + s_box[3]) * 0.5
            o_cx, o_cy = (o_box[0] + o_box[2]) * 0.5, (o_box[1] + o_box[3]) * 0.5

            dist_px = math.sqrt((s_cx - o_cx) ** 2 + (s_cy - o_cy) ** 2)
            if dist_px > (MAX_CENTER_DIST_RATIO * img_diag):
                continue

            rel_copy = dict(rel)
            rel_copy["subject_id"] = int(subj_idx)
            rel_copy["object_id"] = int(obj_idx)
            filtered_relationships.append(rel_copy)

        pairs, rel_payload = _build_relation_entries(
            obj_boxes_t=obj_boxes_norm,
            obj_names=obj_names,
            relationships=filtered_relationships,
            pred_to_idx=self.pred_to_idx,
            use_all_pairs=bool(getattr(self.cfg, "use_all_pairs", False)),
            max_pairs=int(self.cfg.max_pairs),
            negative_pair_ratio=float(getattr(self.cfg, "negative_pair_ratio", -1.0)),
        )

        if len(pairs) == 0:
            return None

        pixel_values = None
        obj_boxes_224 = None

        try:
            if self.processor is not None and image is not None:
                pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = None

        try:
            if image is not None:
                obj_boxes_224, _ = preprocess_boxes_to_clip224(
                    image,
                    obj_boxes_t,
                    out_size=int(self.clip_input_res),
                )
        except Exception:
            obj_boxes_224 = None

        if pixel_values is None:
            pixel_values = torch.zeros((3, int(self.clip_input_res), int(self.clip_input_res)), dtype=torch.float32)
        if obj_boxes_224 is None:
            obj_boxes_224 = obj_boxes_t.clone()

        out = {
            "image_id": str(img_id),
            "image": image,
            "pixel_values": pixel_values,
            "obj_boxes": obj_boxes_t,
            "obj_boxes_224": obj_boxes_224,
            "obj_labels": obj_names,
            "pairs": pairs,
            "rel_pair_names": rel_payload["rel_pair_names"],
            "rel_preds": rel_payload["rel_preds"],
            "rel_is_pos": rel_payload["rel_is_pos"],
            "rel_pred_ids": rel_payload["rel_pred_ids"],
            "rel_pair_hashes": rel_payload["rel_pair_hashes"],
            "rel_pos_mask": rel_payload["rel_pos_mask"],
            "rel_non_sym_mask": rel_payload["rel_non_sym_mask"],
            "rel_texts": rel_payload["rel_texts"],
            "rel_role_swap_texts": rel_payload["rel_role_swap_texts"],
            "rel_cf_attr_texts": rel_payload["rel_cf_attr_texts"],
            "rel_pred_texts": rel_payload["rel_texts"],
            "rel_pred_texts_swap": rel_payload["rel_role_swap_texts"],
            "task_type": "vg150_sgg",
        }

        return out

    def __len__(self) -> int:
        if len(self.valid_indices) == 0:
            return 0
        target = int(self.cfg.samples_per_epoch)
        if target <= 0:
            return len(self.valid_indices)
        if str(self.cfg.split).strip().lower() == "train":
            return target
        return min(target, len(self.valid_indices))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.valid_indices) == 0:
            raise IndexError("VG150LocalDataset has no valid examples")
        if str(self.cfg.split).strip().lower() == "train":
            src_idx = self.valid_indices[int(idx) % len(self.valid_indices)]
        else:
            src_idx = self.valid_indices[int(idx)]
        example = self._process_example(src_idx)
        if example is None:
            # Fallback to nearest valid entry if this sample became invalid at runtime.
            src_idx = self.valid_indices[int(idx) % len(self.valid_indices)]
            example = self._process_example(src_idx)
        if example is None:
            raise IndexError(f"Failed to materialize VG150 example at index {idx}")
        return example




class VG150JSONLDataset(Dataset):
    """Local JSONL VG150 layout: root/{train,validation,test}.jsonl + root/images."""

    def __init__(
        self,
        cfg: VG150LoaderConfig,
        pred_to_idx: Dict[str, int],
        split: str = "train",
        processor=None,
        clip_input_res: int = 336,
    ):
        self.cfg = cfg
        self.pred_to_idx = pred_to_idx
        self.split = str(split).strip().lower()
        self.processor = processor
        self.clip_input_res = int(clip_input_res)
        split_name = "validation" if self.split in {"val", "valid", "validation"} else self.split
        self.path = os.path.join(str(cfg.vg150_root), f"{split_name}.jsonl")
        if not os.path.exists(self.path) and split_name == "validation":
            self.path = os.path.join(str(cfg.vg150_root), "val.jsonl")
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not os.path.exists(self.path):
            return rows
        max_images = int(getattr(self.cfg, "max_images", 0) or 0)
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                    if max_images > 0 and len(rows) >= max_images:
                        break
        return rows

    def __len__(self) -> int:
        n = int(len(self.rows))
        target = int(getattr(self.cfg, "samples_per_epoch", 0) or 0)
        if target <= 0:
            return n
        if self.split == "train":
            return target
        return min(target, n)

    def _resolve_image(self, ex: Dict[str, Any]) -> Image.Image:
        image_raw = ex.get("image", ex.get("image_path", ex.get("path", ex.get("file_name", ""))))
        candidates: List[str] = []
        if isinstance(image_raw, str) and image_raw.strip() != "":
            raw = image_raw.strip()
            candidates.extend([
                raw,
                os.path.join(str(self.cfg.vg150_root), raw),
                os.path.join(str(self.cfg.vg150_root), "images", raw),
                os.path.join(str(self.cfg.vg150_root), "images", "VG_100K", os.path.basename(raw)),
            ])
        image_id = str(ex.get("image_id", ex.get("img_id", ""))).strip()
        if image_id != "":
            stem = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
            candidates.extend([
                os.path.join(str(self.cfg.vg150_root), "images", stem),
                os.path.join(str(self.cfg.vg150_root), "images", "VG_100K", stem),
                os.path.join(str(self.cfg.vg150_root), "images", "VG_100K_2", stem),
            ])
        for path in candidates:
            try:
                if os.path.exists(path):
                    return Image.open(path).convert("RGB")
            except Exception:
                continue
        return Image.new("RGB", (336, 336), color=(128, 128, 128))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.rows) == 0:
            raise IndexError(f"VG150JSONLDataset is empty: {self.path}")
        ex = self.rows[int(idx) % len(self.rows)] if self.split == "train" else self.rows[int(idx)]
        pil_img = self._resolve_image(ex)
        image_id = ex.get("image_id", ex.get("img_id", ""))
        obj_boxes = ex.get("obj_boxes", ex.get("boxes", []))
        objects = ex.get("objects", ex.get("obj_labels", []))
        relationships = ex.get("relationships", ex.get("relations", ex.get("rels", [])))

        if isinstance(obj_boxes, list) and len(obj_boxes) > 0:
            obj_boxes_t = torch.tensor(obj_boxes, dtype=torch.float32)
        else:
            obj_boxes_t = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

        obj_names = _object_names_from_records(objects, int(obj_boxes_t.shape[0]))
        if len(obj_names) < int(obj_boxes_t.shape[0]):
            obj_names.extend(["object"] * (int(obj_boxes_t.shape[0]) - len(obj_names)))

        object_id_lookup = _build_object_id_lookup(objects if isinstance(objects, list) else [], len(obj_names))
        normalized_relationships: List[Dict[str, Any]] = []
        for rel in relationships if isinstance(relationships, list) else []:
            if not isinstance(rel, dict):
                continue
            subj_idx = _resolve_rel_object_index(
                rel.get("subject_id", rel.get("subj_id", rel.get("subject", -1))), object_id_lookup, len(obj_names)
            )
            obj_idx = _resolve_rel_object_index(
                rel.get("object_id", rel.get("obj_id", rel.get("object", -1))), object_id_lookup, len(obj_names)
            )
            if subj_idx is None or obj_idx is None or subj_idx == obj_idx:
                continue
            rel_copy = dict(rel)
            rel_copy["subject_id"] = int(subj_idx)
            rel_copy["object_id"] = int(obj_idx)
            rel_copy["predicate"] = str(rel.get("predicate", rel.get("pred", "relation"))).strip().lower() or "relation"
            normalized_relationships.append(rel_copy)

        pairs, rel_payload = _build_relation_entries(
            obj_boxes_t=obj_boxes_t,
            obj_names=obj_names,
            relationships=normalized_relationships,
            pred_to_idx=self.pred_to_idx,
            use_all_pairs=bool(getattr(self.cfg, "use_all_pairs", False)),
            max_pairs=int(self.cfg.max_pairs),
            negative_pair_ratio=float(getattr(self.cfg, "negative_pair_ratio", -1.0)),
        )

        pixel_values = None
        obj_boxes_224 = None
        try:
            if self.processor is not None:
                pixel_values = self.processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = None
        try:
            obj_boxes_224, _ = preprocess_boxes_to_clip224(pil_img, obj_boxes_t, out_size=int(self.clip_input_res))
        except Exception:
            obj_boxes_224 = None
        if pixel_values is None:
            pixel_values = torch.zeros((3, int(self.clip_input_res), int(self.clip_input_res)), dtype=torch.float32)
        if obj_boxes_224 is None:
            obj_boxes_224 = obj_boxes_t.clone()

        return {
            "image_id": str(image_id),
            "image": pil_img,
            "pixel_values": pixel_values,
            "obj_boxes": obj_boxes_t,
            "obj_boxes_224": obj_boxes_224,
            "obj_labels": obj_names,
            "pairs": pairs,
            "rel_pair_names": rel_payload["rel_pair_names"],
            "rel_preds": rel_payload["rel_preds"],
            "rel_is_pos": rel_payload["rel_is_pos"],
            "rel_pred_ids": rel_payload["rel_pred_ids"],
            "rel_pair_hashes": rel_payload["rel_pair_hashes"],
            "rel_pos_mask": rel_payload["rel_pos_mask"],
            "rel_non_sym_mask": rel_payload["rel_non_sym_mask"],
            "rel_texts": rel_payload["rel_texts"],
            "rel_role_swap_texts": rel_payload["rel_role_swap_texts"],
            "rel_cf_attr_texts": rel_payload["rel_cf_attr_texts"],
            "rel_pred_texts": rel_payload["rel_texts"],
            "rel_pred_texts_swap": rel_payload["rel_role_swap_texts"],
            "task_type": "vg150_sgg",
        }


class VG150HFMapDataset(Dataset):
    """Map-style HF-backed VG150 dataset for non-streaming mode."""

    def __init__(
        self,
        cfg: VG150LoaderConfig,
        hf_dataset_id: str,
        split: str = "train",
        processor=None,
        clip_input_res: int = 336,
        pred_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.cfg = cfg
        self.hf_dataset_id = str(hf_dataset_id)
        self.split = str(split)
        self.processor = processor
        self.clip_input_res = int(clip_input_res)
        self.pred_to_idx = pred_to_idx or {}
        self.dataset = load_dataset(self.hf_dataset_id, split=self.split, streaming=False)

    def _base_len(self) -> int:
        n = int(len(self.dataset))
        max_images = int(getattr(self.cfg, "max_images", 0) or 0)
        if max_images > 0:
            return min(n, max_images)
        return n

    def __len__(self) -> int:
        n = self._base_len()
        target = int(self.cfg.samples_per_epoch)
        if target <= 0:
            return n
        if self.split.strip().lower() == "train":
            return target
        return min(target, n)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n = self._base_len()
        if n <= 0:
            raise IndexError("VG150HFMapDataset is empty")
        if self.split.strip().lower() == "train":
            ex = self.dataset[int(idx) % n]
        else:
            ex = self.dataset[int(idx)]

        image_raw = ex.get("image", None)
        pil_img = None
        try:
            if isinstance(image_raw, Image.Image):
                pil_img = image_raw.convert("RGB")
            elif isinstance(image_raw, dict):
                if "bytes" in image_raw and image_raw["bytes"]:
                    pil_img = Image.open(io.BytesIO(image_raw["bytes"])).convert("RGB")
                elif "path" in image_raw and image_raw["path"] and os.path.exists(image_raw["path"]):
                    pil_img = Image.open(image_raw["path"]).convert("RGB")
        except Exception:
            pass
        if pil_img is None:
            pil_img = Image.new("RGB", (336, 336), color=(128, 128, 128))

        image_id = ex.get("image_id", ex.get("img_id", ""))
        obj_boxes = ex.get("obj_boxes", [])
        objects = ex.get("objects", [])
        relationships = ex.get("relationships", [])

        if isinstance(obj_boxes, list) and len(obj_boxes) > 0:
            obj_boxes_t = torch.tensor(obj_boxes, dtype=torch.float32)
        else:
            obj_boxes_t = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

        obj_names = _object_names_from_records(objects, int(obj_boxes_t.shape[0]))
        object_id_lookup = _build_object_id_lookup(objects, len(obj_names))
        normalized_relationships: List[Dict[str, Any]] = []
        for rel in relationships:
            subj_idx = _resolve_rel_object_index(rel.get("subject_id", -1), object_id_lookup, len(obj_names))
            obj_idx = _resolve_rel_object_index(rel.get("object_id", -1), object_id_lookup, len(obj_names))
            if subj_idx is None or obj_idx is None or subj_idx == obj_idx:
                continue
            rel_copy = dict(rel)
            rel_copy["subject_id"] = int(subj_idx)
            rel_copy["object_id"] = int(obj_idx)
            normalized_relationships.append(rel_copy)

        pairs, rel_payload = _build_relation_entries(
            obj_boxes_t=obj_boxes_t,
            obj_names=obj_names,
            relationships=normalized_relationships,
            pred_to_idx=self.pred_to_idx,
            use_all_pairs=bool(getattr(self.cfg, "use_all_pairs", False)),
            max_pairs=int(self.cfg.max_pairs),
            negative_pair_ratio=float(getattr(self.cfg, "negative_pair_ratio", -1.0)),
        )

        pixel_values = None
        obj_boxes_224 = None
        try:
            if self.processor is not None:
                pixel_values = self.processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = None
        try:
            obj_boxes_224, _ = preprocess_boxes_to_clip224(pil_img, obj_boxes_t, out_size=int(self.clip_input_res))
        except Exception:
            obj_boxes_224 = None
        if pixel_values is None:
            pixel_values = torch.zeros((3, int(self.clip_input_res), int(self.clip_input_res)), dtype=torch.float32)
        if obj_boxes_224 is None:
            obj_boxes_224 = obj_boxes_t.clone()

        return {
            "image_id": str(image_id),
            "image": pil_img,
            "pixel_values": pixel_values,
            "obj_boxes": obj_boxes_t,
            "obj_boxes_224": obj_boxes_224,
            "obj_labels": obj_names,
            "pairs": pairs,
            "rel_pair_names": rel_payload["rel_pair_names"],
            "rel_preds": rel_payload["rel_preds"],
            "rel_is_pos": rel_payload["rel_is_pos"],
            "rel_pred_ids": rel_payload["rel_pred_ids"],
            "rel_pair_hashes": rel_payload["rel_pair_hashes"],
            "rel_pos_mask": rel_payload["rel_pos_mask"],
            "rel_non_sym_mask": rel_payload["rel_non_sym_mask"],
            "rel_texts": rel_payload["rel_texts"],
            "rel_role_swap_texts": rel_payload["rel_role_swap_texts"],
            "rel_cf_attr_texts": rel_payload["rel_cf_attr_texts"],
            "rel_pred_texts": rel_payload["rel_texts"],
            "rel_pred_texts_swap": rel_payload["rel_role_swap_texts"],
            "task_type": "vg150_sgg",
        }


class VG150DataLoader:
    """High-level VG150 data loader wrapper."""
    
    def __init__(
        self,
        cfg: VG150LoaderConfig,
        split: str = "train",
        shuffle: bool = True,
        processor=None,
        clip_input_res: int = 336,
    ):
        self.cfg = cfg
        self.processor = processor
        self.clip_input_res = int(clip_input_res)
        self.split = str(split).strip().lower()
        self.source_mode = "unknown"
        if negative_pair_ratio_is_inert(bool(getattr(cfg, "use_all_pairs", True)), float(getattr(cfg, "negative_pair_ratio", -1.0))):
            logging.warning(
                "VG150DataLoader(split=%s): negative_pair_ratio=%s has NO EFFECT because "
                "use_all_pairs=False -- _build_relation_entries returns positive-only pairs "
                "before negative_pair_ratio is ever read. If you intend to train/evaluate "
                "with sampled negative pairs, set --use_all_pairs true. If positives-only is "
                "actually what you want, set --negative_pair_ratio 0 to make that explicit "
                "and silence this warning. See docs/known_issues.md.",
                self.split,
                getattr(cfg, "negative_pair_ratio", None),
            )
        # Load vocabularies (prefer local dicts if present). Add the background
        # relation class used for sampled non-relation pairs; predicate ids must
        # match the 51-way classifier head used by training/evaluation.
        self.label_to_idx, self.pred_to_idx = _load_vg150_vocab(cfg.vg150_root)
        if "relation" not in self.pred_to_idx:
            self.pred_to_idx = dict(self.pred_to_idx)
            self.pred_to_idx["relation"] = len(self.pred_to_idx)

        # Create dataset.
        # If a source is explicitly requested, never silently fall back to HF/local.
        # In auto mode, prefer local VG150 files, with HF map-style as a fallback.
        local_ready = _has_local_vg150_files(cfg.vg150_root)
        source = str(getattr(cfg, "source", "auto")).strip().lower().replace("_", "-")
        if source in {"", "default"}:
            source = "auto"
        if source == "local":
            source = "local-jsonl" if _has_local_jsonl_files(cfg.vg150_root) else "local-files"

        if source == "hf-streaming" or (source == "auto" and bool(getattr(cfg, "hf_dataset_id", "")) and bool(cfg.streaming)):
            dataset = VG150HFIterable(cfg, cfg.hf_dataset_id, self.label_to_idx, self.pred_to_idx, split=self.split, processor=getattr(self, 'processor', None), clip_input_res=getattr(self, 'clip_input_res', 336))
            self.source_mode = "hf-streaming"
        elif source == "local-jsonl" or (source == "auto" and _has_local_jsonl_files(cfg.vg150_root)):
            if not _has_local_jsonl_files(cfg.vg150_root):
                raise FileNotFoundError(
                    f"Requested VG150 local-jsonl source, but train.jsonl and validation.jsonl were not found under {cfg.vg150_root}."
                )
            dataset = VG150JSONLDataset(cfg, self.pred_to_idx, split=self.split, processor=getattr(self, 'processor', None), clip_input_res=getattr(self, 'clip_input_res', 336))
            self.source_mode = "local-jsonl"
        elif source == "local-files" or (source == "auto" and local_ready):
            if not local_ready:
                raise FileNotFoundError(
                    f"Requested VG150 local-files source, but required VG150 files were not found under {cfg.vg150_root}. "
                    "Expected either train.jsonl/validation.jsonl or VG-SGG-dicts-with-attri.json, image_data.json, scene_graphs.json, and images/."
                )
            dataset = VG150LocalDataset(cfg, self.label_to_idx, self.pred_to_idx, processor=getattr(self, 'processor', None), clip_input_res=getattr(self, 'clip_input_res', 336))
            self.source_mode = "local-files"
        elif source == "hf-map" or (source == "auto" and bool(getattr(cfg, "hf_dataset_id", ""))):
            dataset = VG150HFMapDataset(cfg, cfg.hf_dataset_id, split=self.split, processor=getattr(self, 'processor', None), clip_input_res=getattr(self, 'clip_input_res', 336), pred_to_idx=self.pred_to_idx)
            self.source_mode = "hf-map"
        else:
            raise ValueError(
                f"Unsupported VG150 source={source!r}. Use auto, local, local-jsonl, local-files, hf-map, or hf-streaming."
            )

        sampler = None
        if (
            bool(getattr(cfg, "predicate_sampler_enabled", False))
            and self.split == "train"
            and not isinstance(dataset, IterableDataset)
        ):
            sampler = _predicate_balanced_sampler(dataset, cfg)
        if sampler is None and not isinstance(dataset, IterableDataset) and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=int(os.environ.get("WORLD_SIZE", "1")),
                rank=int(os.environ.get("RANK", "0")),
                shuffle=bool(shuffle),
                seed=int(getattr(cfg, "seed", 0)),
                drop_last=bool(cfg.drop_last),
            )
        self.sampler = sampler
        self.predicate_sampler_active = sampler is not None and not isinstance(sampler, DistributedSampler)
        use_shuffle = bool(shuffle) and sampler is None and not isinstance(dataset, IterableDataset)

        # Create data loader
        self.loader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=use_shuffle,
            sampler=sampler,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=bool(cfg.persistent_workers) and int(cfg.num_workers) > 0,
            prefetch_factor=(int(cfg.prefetch_factor) if int(cfg.num_workers) > 0 else None),
            drop_last=bool(cfg.drop_last),
            collate_fn=_collate_identity,
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self) -> int:
        if hasattr(self.loader, "__len__"):
            try:
                return int(len(self.loader))
            except TypeError:
                pass
        bs = max(1, int(self.cfg.batch_size))
        steps = int(self.cfg.samples_per_epoch) // bs
        return max(1, steps)


def scan_vg150_predicate_vocab(vg150_root: str = "datasets") -> List[str]:
    """Scan and return VG150 predicate vocabulary in classifier-target index order."""
    _label_to_idx, pred_to_idx = _load_vg150_vocab(vg150_root)
    preds = [
        str(name).strip().lower()
        for name, _idx in sorted(pred_to_idx.items(), key=lambda kv: int(kv[1]))
        if str(name).strip() != ""
    ]
    return preds if len(preds) > 0 else ["relation"]


class VG150HFIterable(IterableDataset):
    """Streaming HF-backed VG150 iterable dataset mapper.

    This maps HF dataset entries into the expected example dict used by training.
    It does not download images by default; it preserves image/URL fields so downstream
    workers can fetch or process them as needed.
    """

    def __init__(self, cfg: VG150LoaderConfig, hf_dataset_id: str, label_to_idx: Dict[str, int], pred_to_idx: Dict[str, int], split: str = "train", processor=None, clip_input_res: int = 336):
        self.cfg = cfg
        self.hf_dataset_id = str(hf_dataset_id)
        self.label_to_idx = label_to_idx
        self.pred_to_idx = pred_to_idx
        self.split = split
        self.processor = processor
        self.clip_input_res = int(clip_input_res)

        self.iterable = load_dataset(self.hf_dataset_id, split=self.split, streaming=True)
        if self.split.strip().lower() == "train":
            self.iterable = self.iterable.shuffle(
                buffer_size=max(1000, int(self.cfg.shuffle_buffer_size)),
                seed=int(self.cfg.seed),
            )

    def __iter__(self):
        import io
        import os

        worker = get_worker_info()
        num_workers = int(worker.num_workers) if worker is not None else 1
        worker_id = int(worker.id) if worker is not None else 0
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        rank = int(os.environ.get("RANK", "0"))
        num_shards = max(1, num_workers * world_size)
        shard_index = (rank * num_workers) + worker_id

        iterable = self.iterable
        if num_shards > 1 and hasattr(iterable, "shard"):
            iterable = iterable.shard(num_shards=num_shards, index=shard_index, contiguous=False)

        for ex in iterable:
            # Expected fields in HF repo: image (may be PIL/Image or dict), image_id, obj_boxes, objects, relationships
            image_raw = ex.get("image", None)
            
            # --- ABSOLUTE SAFEGUARD FOR IMAGE DECODING ---
            pil_img = None
            try:
                if isinstance(image_raw, Image.Image):
                    pil_img = image_raw.convert("RGB")
                elif isinstance(image_raw, dict):
                    if "bytes" in image_raw and image_raw["bytes"]:
                        pil_img = Image.open(io.BytesIO(image_raw["bytes"])).convert("RGB")
                    elif "path" in image_raw and image_raw["path"]:
                        path = image_raw["path"]
                        if os.path.exists(path):
                            pil_img = Image.open(path).convert("RGB")
            except Exception:
                pass # Silently fallback below
                
            # Fallback if decoding failed, or image is missing/corrupted
            if pil_img is None:
                pil_img = Image.new("RGB", (336, 336), color=(128, 128, 128))
            # ---------------------------------------------

            image_id = ex.get("image_id", ex.get("img_id", ""))
            obj_boxes = ex.get("obj_boxes", [])
            objects = ex.get("objects", [])
            relationships = ex.get("relationships", [])

            # Normalize boxes
            if isinstance(obj_boxes, list) and len(obj_boxes) > 0:
                obj_boxes_t = torch.tensor(obj_boxes, dtype=torch.float32)
            else:
                obj_boxes_t = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

            # Build pairs and rels
            obj_names = _object_names_from_records(objects, int(obj_boxes_t.shape[0]))
            object_id_lookup = _build_object_id_lookup(objects, len(obj_names))
            normalized_relationships: List[Dict[str, Any]] = []
            for rel in relationships:
                subj_idx = _resolve_rel_object_index(rel.get("subject_id", -1), object_id_lookup, len(obj_names))
                obj_idx = _resolve_rel_object_index(rel.get("object_id", -1), object_id_lookup, len(obj_names))
                if subj_idx is None or obj_idx is None or subj_idx == obj_idx:
                    continue
                rel_copy = dict(rel)
                rel_copy["subject_id"] = int(subj_idx)
                rel_copy["object_id"] = int(obj_idx)
                normalized_relationships.append(rel_copy)

            pairs, rel_payload = _build_relation_entries(
                obj_boxes_t=obj_boxes_t,
                obj_names=obj_names,
                relationships=normalized_relationships,
                pred_to_idx=self.pred_to_idx,
                use_all_pairs=bool(getattr(self.cfg, "use_all_pairs", False)),
                max_pairs=int(self.cfg.max_pairs),
            )
            pixel_values = None
            obj_boxes_224 = None
            try:
                if self.processor is not None:
                    pixel_values = self.processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)
            except Exception:
                pixel_values = None
            try:
                obj_boxes_224, _ = preprocess_boxes_to_clip224(pil_img, obj_boxes_t, out_size=int(self.clip_input_res))
            except Exception:
                obj_boxes_224 = None
            if pixel_values is None:
                pixel_values = torch.zeros((3, int(self.clip_input_res), int(self.clip_input_res)), dtype=torch.float32)
            if obj_boxes_224 is None:
                obj_boxes_224 = obj_boxes_t.clone()

            yield {
                "image_id": str(image_id),
                "image": pil_img,
                "pixel_values": pixel_values,
                "obj_boxes": obj_boxes_t,
                "obj_boxes_224": obj_boxes_224,
                "obj_labels": obj_names,
                "pairs": pairs,
                "rel_pair_names": rel_payload["rel_pair_names"],
                "rel_preds": rel_payload["rel_preds"],
                "rel_is_pos": rel_payload["rel_is_pos"],
                "rel_pred_ids": rel_payload["rel_pred_ids"],
                "rel_pair_hashes": rel_payload["rel_pair_hashes"],
                "rel_pos_mask": rel_payload["rel_pos_mask"],
                "rel_non_sym_mask": rel_payload["rel_non_sym_mask"],
                "rel_texts": rel_payload["rel_texts"],
                "rel_role_swap_texts": rel_payload["rel_role_swap_texts"],
                "rel_cf_attr_texts": rel_payload["rel_cf_attr_texts"],
                "rel_pred_texts": rel_payload["rel_texts"],
                "rel_pred_texts_swap": rel_payload["rel_role_swap_texts"],
                "task_type": "vg150_sgg",
            }