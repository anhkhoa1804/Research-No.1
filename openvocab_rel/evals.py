from __future__ import annotations

import math
import json
import os
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from PIL import Image

if TYPE_CHECKING:
    from transformers import CLIPProcessor

from .clip_utils import clip_text_features, encode_predicate_vocab
from .ddp_utils import unwrap_ddp
from .geometry import preprocess_boxes_to_clip224
from .prompts import PredicateCanonicalizer, counterfactual_prompt_variants, pred_prompt, pred_prompt_ensemble, pred_prompt_roles
from .config import TrainConfig


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if int(boxes1.numel()) == 0 or int(boxes2.numel()) == 0:
        return torch.zeros((int(boxes1.shape[0]), int(boxes2.shape[0])), dtype=torch.float32)

    b1 = boxes1.float()
    b2 = boxes2.float()
    tl = torch.maximum(b1[:, None, :2], b2[None, :, :2])
    br = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (br - tl).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((b1[:, 2] - b1[:, 0]).clamp_min(0.0) * (b1[:, 3] - b1[:, 1]).clamp_min(0.0)).unsqueeze(1)
    area2 = ((b2[:, 2] - b2[:, 0]).clamp_min(0.0) * (b2[:, 3] - b2[:, 1]).clamp_min(0.0)).unsqueeze(0)
    union = (area1 + area2 - inter).clamp_min(1e-6)
    return inter / union


def _build_all_ordered_pairs(n_obj: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(int(n_obj)) for j in range(int(n_obj)) if i != j]


def _extract_gt_pairs(ex: Dict[str, Any]) -> List[Tuple[int, int]]:
    pairs = ex.get("pairs", [])
    pos_mask = ex.get("rel_pos_mask", None)
    gt_pairs: List[Tuple[int, int]] = []
    for idx, pair in enumerate(pairs):
        if pos_mask is not None and idx < len(pos_mask) and not bool(pos_mask[idx]):
            continue
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        gt_pairs.append((int(pair[0]), int(pair[1])))
    return gt_pairs


def _load_vg150_object_vocab(vg150_root: str) -> List[str]:
    from .datasets.vg150_loader import _load_vg150_vocab  # type: ignore[attr-defined]

    label_to_idx, _ = _load_vg150_vocab(vg150_root)
    if len(label_to_idx) == 0:
        return ["object"]
    return [name for name, _ in sorted(label_to_idx.items(), key=lambda kv: int(kv[1]))]


def _object_prompt(label: str) -> str:
    name = str(label).strip().lower()
    if name == "":
        name = "object"
    article = "an" if name[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"a photo of {article} {name}"


@torch.no_grad()
def _encode_object_vocab(
    clip_model: nn.Module,
    processor: Any,
    obj_vocab: List[str],
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    texts = [_object_prompt(name) for name in obj_vocab]
    feats = clip_text_features(clip_model, processor, texts, device)
    return obj_vocab, feats


def _object_semantic_feats_from_labels(
    labels_per_image: List[List[str]],
    obj_vocab: List[str],
    obj_text_feats: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    vocab_to_idx = {str(label).strip().lower(): i for i, label in enumerate(obj_vocab)}
    out: List[torch.Tensor] = []
    for labels in labels_per_image:
        feats = []
        for label in labels:
            idx = vocab_to_idx.get(str(label).strip().lower())
            if idx is None:
                feats.append(torch.zeros((int(obj_text_feats.shape[-1]),), device=device, dtype=obj_text_feats.dtype))
            else:
                feats.append(obj_text_feats[int(idx)].to(device=device))
        if len(feats) == 0:
            out.append(torch.empty((0, int(obj_text_feats.shape[-1])), device=device, dtype=obj_text_feats.dtype))
        else:
            out.append(torch.stack(feats, dim=0))
    return out


@torch.no_grad()
def _clip_image_features_from_pil(
    clip_model: nn.Module,
    processor: Any,
    images: List[Image.Image],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    def _extract_image_tensor(out_obj: object) -> torch.Tensor:
        if isinstance(out_obj, torch.Tensor):
            return out_obj
        image_embeds = getattr(out_obj, "image_embeds", None)
        if isinstance(image_embeds, torch.Tensor):
            return image_embeds
        pooler_output = getattr(out_obj, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        last_hidden_state = getattr(out_obj, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            if last_hidden_state.ndim >= 3 and int(last_hidden_state.shape[1]) > 0:
                return last_hidden_state[:, 0, :]
            return last_hidden_state
        if isinstance(out_obj, (tuple, list)) and len(out_obj) > 0:
            first = out_obj[0]
            if isinstance(first, torch.Tensor):
                if first.ndim >= 3 and int(first.shape[1]) > 0:
                    return first[:, 0, :]
                return first
        if isinstance(out_obj, dict):
            for key in ("image_embeds", "pooler_output"):
                value = out_obj.get(key, None)
                if isinstance(value, torch.Tensor):
                    return value
            value = out_obj.get("last_hidden_state", None)
            if isinstance(value, torch.Tensor):
                if value.ndim >= 3 and int(value.shape[1]) > 0:
                    return value[:, 0, :]
                return value
        raise TypeError(f"Unsupported CLIP image output type: {type(out_obj)}")

    cm = unwrap_ddp(clip_model)
    outs: List[torch.Tensor] = []
    for start in range(0, len(images), int(batch_size)):
        chunk = images[start : start + int(batch_size)]
        inputs = processor(images=chunk, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out_obj = cm.get_image_features(**inputs)
        feats = _extract_image_tensor(out_obj)
        outs.append(F.normalize(feats.float(), dim=-1).detach())
    if len(outs) == 0:
        proj_dim = int(getattr(getattr(cm, "config", object()), "projection_dim", 512) or 512)
        return torch.empty((0, proj_dim), device=device)
    return torch.cat(outs, dim=0)


def _crop_objects_from_example(ex: Dict[str, Any]) -> List[Image.Image]:
    image = ex.get("image", None)
    boxes = ex.get("obj_boxes", None)
    if not isinstance(image, Image.Image) or not isinstance(boxes, torch.Tensor) or int(boxes.shape[0]) == 0:
        return []

    width, height = image.size
    crops: List[Image.Image] = []
    for box in boxes.tolist():
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        crops.append(image.crop((x1, y1, x2, y2)).convert("RGB"))
    return crops


def _clip_obj_cache_enabled(cfg: TrainConfig) -> bool:
    return bool(getattr(cfg, "eval_sgg_clip_obj_cache_enabled", True))


def _clip_obj_cache_dir(cfg: TrainConfig) -> str:
    return str(getattr(cfg, "eval_sgg_clip_obj_cache_dir", "runs/clip_obj_cache")).strip()


def _clip_obj_cache_key(cfg: TrainConfig, ex: Dict[str, Any], obj_vocab: List[str]) -> str:
    import hashlib

    image_id = str(ex.get("image_id", "")).strip()
    boxes = ex.get("obj_boxes", torch.empty((0, 4), dtype=torch.float32))
    if isinstance(boxes, torch.Tensor):
        boxes_payload = boxes.detach().cpu().float().tolist()
    else:
        boxes_payload = []
    payload = {
        "image_id": image_id,
        "clip_name": str(getattr(cfg, "clip_name", "openai/clip-vit-large-patch14-336")),
        "obj_vocab": [str(x).strip().lower() for x in obj_vocab],
        "boxes": boxes_payload,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _clip_obj_cache_path(cfg: TrainConfig, ex: Dict[str, Any], obj_vocab: List[str]) -> str:
    return os.path.join(_clip_obj_cache_dir(cfg), f"{_clip_obj_cache_key(cfg, ex, obj_vocab)}.json")


def _load_clip_obj_cache(cfg: TrainConfig, ex: Dict[str, Any], obj_vocab: List[str]) -> Optional[Tuple[List[str], torch.Tensor]]:
    if not _clip_obj_cache_enabled(cfg):
        return None
    image_id = str(ex.get("image_id", "")).strip()
    if image_id == "":
        return None
    path = _clip_obj_cache_path(cfg, ex, obj_vocab)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        labels = [str(x).strip().lower() for x in payload.get("labels", [])]
        scores = torch.tensor(payload.get("scores", []), dtype=torch.float32)
        return labels, scores
    except Exception:
        return None


def _save_clip_obj_cache(cfg: TrainConfig, ex: Dict[str, Any], obj_vocab: List[str], labels: List[str], scores: torch.Tensor) -> None:
    if not _clip_obj_cache_enabled(cfg):
        return
    image_id = str(ex.get("image_id", "")).strip()
    if image_id == "":
        return
    path = _clip_obj_cache_path(cfg, ex, obj_vocab)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        payload = {
            "image_id": image_id,
            "labels": [str(x).strip().lower() for x in labels],
            "scores": scores.detach().cpu().float().tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        return


@torch.no_grad()
def _predict_object_labels_from_clip(
    cfg: TrainConfig,
    clip_model: nn.Module,
    processor: Any,
    ex: Dict[str, Any],
    obj_text_feats: torch.Tensor,
    obj_vocab: List[str],
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    boxes = ex.get("obj_boxes", None)
    if not isinstance(boxes, torch.Tensor) or int(boxes.shape[0]) == 0:
        return [], torch.empty((0,), device=device)

    cached = _load_clip_obj_cache(cfg, ex, obj_vocab)
    if cached is not None:
        labels, scores = cached
        return labels, scores.to(device=device)

    crops = _crop_objects_from_example(ex)
    if len(crops) != int(boxes.shape[0]):
        gt_labels = [str(x).strip().lower() for x in ex.get("obj_labels", [])[: int(boxes.shape[0])]]
        return gt_labels, torch.ones((len(gt_labels),), device=device)

    image_feats = _clip_image_features_from_pil(clip_model, processor, crops, device)

    cm = unwrap_ddp(clip_model)
    logit_scale = cm.logit_scale.exp().item() if hasattr(cm, "logit_scale") else 100.0
    logits = (image_feats @ F.normalize(obj_text_feats.float(), dim=-1).t()) * logit_scale

    probs = logits.softmax(dim=-1)
    scores, idx = probs.max(dim=-1)
    labels = [str(obj_vocab[int(i)]).strip().lower() for i in idx.detach().cpu().tolist()]
    _save_clip_obj_cache(cfg, ex, obj_vocab, labels, scores)
    return labels, scores.detach()




def _normalize_vg_label(label: str, aliases: Optional[Dict[str, str]] = None) -> str:
    text = " ".join(str(label).strip().lower().replace("_", " ").split())
    if aliases is not None:
        return aliases.get(text, text)
    return text


def _build_vg_aliases(obj_vocab: List[str], pred_vocab: List[str]) -> Dict[str, str]:
    canonical = {_normalize_vg_label(x): _normalize_vg_label(x) for x in list(obj_vocab) + list(pred_vocab)}
    manual = {
        "man": "person",
        "woman": "person",
        "boy": "person",
        "girl": "person",
        "child": "person",
        "kid": "person",
        "people": "person",
        "puppy": "dog",
        "doggie": "dog",
        "kitty": "cat",
        "kitten": "cat",
        "bike": "bicycle",
        "motorbike": "motorcycle",
        "auto": "car",
        "automobile": "car",
        "plane": "airplane",
        "aeroplane": "airplane",
        "sofa": "couch",
        "tv": "television",
        "television set": "television",
        "cellphone": "phone",
        "mobile phone": "phone",
        "in front of": "in front of",
        "front of": "in front of",
        "beside": "next to",
        "near": "next to",
        "on top of": "on",
        "upon": "on",
        "inside": "in",
        "has": "has",
        "have": "has",
        "wears": "wearing",
        "wear": "wearing",
        "riding": "riding",
        "ride": "riding",
        "sitting on": "sitting on",
        "standing on": "standing on",
    }
    aliases = dict(canonical)
    for src, dst in manual.items():
        src_n = _normalize_vg_label(src)
        dst_n = _normalize_vg_label(dst)
        if dst_n in canonical:
            aliases[src_n] = dst_n
    return aliases


def _normalize_triplet_labels(triplets: Dict[str, Any], aliases: Optional[Dict[str, str]]) -> Dict[str, Any]:
    if aliases is None:
        return triplets
    out = dict(triplets)
    out["subj_labels"] = [_normalize_vg_label(x, aliases) for x in triplets.get("subj_labels", [])]
    out["pred_labels"] = [_normalize_vg_label(x, aliases) for x in triplets.get("pred_labels", [])]
    out["obj_labels"] = [_normalize_vg_label(x, aliases) for x in triplets.get("obj_labels", [])]
    return out


def _print_triplet_debug(task_name: str, triplets: Dict[str, Any], gt: Dict[str, Any], matches: torch.Tensor, budget: Dict[str, int]) -> None:
    if int(budget.get("remaining", 0)) <= 0:
        return
    pred_scores = triplets.get("scores", torch.empty((0,), dtype=torch.float32))
    order = torch.argsort(pred_scores.float(), descending=True).detach().cpu().tolist() if isinstance(pred_scores, torch.Tensor) else []
    max_rows = min(int(budget.get("remaining", 0)), max(1, len(order)))
    for row_idx in range(max_rows):
        pi = int(order[row_idx]) if row_idx < len(order) else row_idx
        if pi >= len(triplets.get("pred_labels", [])):
            break
        hit = bool(matches[pi].any().item()) if matches.ndim == 2 and pi < int(matches.shape[0]) else False
        gt_i = int(torch.nonzero(matches[pi], as_tuple=False)[0].item()) if hit else 0
        pred_t = (
            triplets.get("subj_labels", [""])[pi],
            triplets.get("pred_labels", [""])[pi],
            triplets.get("obj_labels", [""])[pi],
        )
        gt_t = (
            gt.get("subj_labels", [""])[gt_i] if len(gt.get("subj_labels", [])) > 0 else "",
            gt.get("pred_labels", [""])[gt_i] if len(gt.get("pred_labels", [])) > 0 else "",
            gt.get("obj_labels", [""])[gt_i] if len(gt.get("obj_labels", [])) > 0 else "",
        )
        score = float(pred_scores[pi].item()) if isinstance(pred_scores, torch.Tensor) and pi < int(pred_scores.numel()) else 0.0
        print(f"[SGG Debug:{task_name}] pred={pred_t} score={score:.4f} | gt_ref={gt_t} | match={hit}")
        budget["remaining"] = int(budget.get("remaining", 0)) - 1
        if int(budget.get("remaining", 0)) <= 0:
            break

def _collect_gt_triplets(ex: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    pairs = ex.get("pairs", [])
    preds = ex.get("rel_preds", [])
    pos_mask = ex.get("rel_pos_mask", None)
    obj_labels = [str(x).strip().lower() for x in ex.get("obj_labels", [])]
    obj_boxes = ex.get("obj_boxes", torch.empty((0, 4), dtype=torch.float32))

    gt_subj: List[str] = []
    gt_obj: List[str] = []
    gt_pred: List[str] = []
    gt_sb: List[List[float]] = []
    gt_ob: List[List[float]] = []

    n = min(len(pairs), len(preds))
    for idx in range(n):
        if pos_mask is not None and idx < len(pos_mask) and not bool(pos_mask[idx]):
            continue
        pair = pairs[idx]
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        s_idx = int(pair[0])
        o_idx = int(pair[1])
        if not (0 <= s_idx < len(obj_labels) and 0 <= o_idx < len(obj_labels)):
            continue
        if s_idx >= int(obj_boxes.shape[0]) or o_idx >= int(obj_boxes.shape[0]):
            continue
        pred = str(preds[idx]).strip().lower()
        if pred == "":
            continue
        gt_subj.append(obj_labels[s_idx])
        gt_pred.append(pred)
        gt_obj.append(obj_labels[o_idx])
        gt_sb.append(obj_boxes[s_idx].tolist())
        gt_ob.append(obj_boxes[o_idx].tolist())

    return {
        "subj_labels": gt_subj,
        "pred_labels": gt_pred,
        "obj_labels": gt_obj,
        "subj_boxes": torch.tensor(gt_sb, dtype=torch.float32, device=device) if len(gt_sb) > 0 else torch.empty((0, 4), dtype=torch.float32, device=device),
        "obj_boxes": torch.tensor(gt_ob, dtype=torch.float32, device=device) if len(gt_ob) > 0 else torch.empty((0, 4), dtype=torch.float32, device=device),
    }


def _compute_pred_matches(
    pred_triplets: Dict[str, Any],
    gt_triplets: Dict[str, Any],
    iou_thresh: float,
) -> torch.Tensor:
    num_pred = len(pred_triplets["pred_labels"])
    num_gt = len(gt_triplets["pred_labels"])
    matches = torch.zeros((num_pred, num_gt), dtype=torch.bool)
    if num_pred == 0 or num_gt == 0:
        return matches

    pred_subj_boxes = pred_triplets["subj_boxes"].float().cpu()
    pred_obj_boxes = pred_triplets["obj_boxes"].float().cpu()
    gt_subj_boxes = gt_triplets["subj_boxes"].float().cpu()
    gt_obj_boxes = gt_triplets["obj_boxes"].float().cpu()

    subj_iou = _box_iou_xyxy(pred_subj_boxes, gt_subj_boxes)
    obj_iou = _box_iou_xyxy(pred_obj_boxes, gt_obj_boxes)

    for pi in range(num_pred):
        ps = pred_triplets["subj_labels"][pi]
        pp = pred_triplets["pred_labels"][pi]
        po = pred_triplets["obj_labels"][pi]
        for gi in range(num_gt):
            if ps != gt_triplets["subj_labels"][gi]:
                continue
            if pp != gt_triplets["pred_labels"][gi]:
                continue
            if po != gt_triplets["obj_labels"][gi]:
                continue
            if float(subj_iou[pi, gi]) < float(iou_thresh):
                continue
            if float(obj_iou[pi, gi]) < float(iou_thresh):
                continue
            matches[pi, gi] = True
    return matches


def _recall_from_matches(matches: torch.Tensor, pred_scores: torch.Tensor, ks: List[int]) -> Dict[int, float]:
    out = {int(k): 0.0 for k in ks}
    num_gt = int(matches.shape[1]) if matches.ndim == 2 else 0
    if num_gt == 0:
        return out
    order = torch.argsort(pred_scores.float(), descending=True).to(device=matches.device)
    for k in ks:
        topk = order[: min(int(k), int(order.numel()))]
        if int(topk.numel()) == 0:
            out[int(k)] = 0.0
            continue
        hit = matches.index_select(0, topk).any(dim=0).float().sum().item()
        out[int(k)] = float(hit) / float(num_gt)
    return out


def _mean_recall_from_matches(
    matches: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_predicates: List[str],
    ks: List[int],
) -> Dict[int, float]:
    out = {int(k): 0.0 for k in ks}
    if len(gt_predicates) == 0:
        return out
    order = torch.argsort(pred_scores.float(), descending=True).to(device=matches.device)
    uniq = sorted(set([str(x).strip().lower() for x in gt_predicates if str(x).strip() != ""]))
    if len(uniq) == 0:
        return out
    for k in ks:
        topk = order[: min(int(k), int(order.numel()))]
        vals: List[float] = []
        for pred in uniq:
            gt_mask = torch.tensor(
                [1 if str(x).strip().lower() == pred else 0 for x in gt_predicates],
                dtype=torch.bool,
                device=matches.device,
            )
            denom = int(gt_mask.sum().item())
            if denom <= 0:
                continue
            if int(topk.numel()) == 0:
                vals.append(0.0)
                continue
            pred_hit = matches.index_select(0, topk)[:, gt_mask].any(dim=0).float().sum().item()
            vals.append(float(pred_hit) / float(denom))
        out[int(k)] = float(sum(vals) / float(max(1, len(vals))))
    return out


def _make_triplet_predictions(
    ex: Dict[str, Any],
    pair_list: List[Tuple[int, int]],
    pair_scores: torch.Tensor,
    pair_pred_idx: torch.Tensor,
    pair_pred_scores: torch.Tensor,
    object_labels: List[str],
    object_scores: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    boxes = ex.get("obj_boxes", torch.empty((0, 4), dtype=torch.float32)).to(device)
    subj_labels: List[str] = []
    pred_labels: List[str] = []
    obj_labels: List[str] = []
    subj_boxes: List[torch.Tensor] = []
    obj_boxes: List[torch.Tensor] = []
    scores: List[float] = []
    pred_vocab = ex.get("_pred_vocab", [])

    for idx, pair in enumerate(pair_list):
        s_idx, o_idx = int(pair[0]), int(pair[1])
        if s_idx >= len(object_labels) or o_idx >= len(object_labels):
            continue
        if s_idx >= int(boxes.shape[0]) or o_idx >= int(boxes.shape[0]):
            continue
        pred_id = int(pair_pred_idx[idx].item())
        if not (0 <= pred_id < len(pred_vocab)):
            continue
        subj_labels.append(str(object_labels[s_idx]).strip().lower())
        pred_labels.append(str(pred_vocab[pred_id]).strip().lower())
        obj_labels.append(str(object_labels[o_idx]).strip().lower())
        subj_boxes.append(boxes[s_idx])
        obj_boxes.append(boxes[o_idx])
        triplet_score = float(pair_scores[idx].item()) * float(pair_pred_scores[idx].item())
        if int(object_scores.numel()) > 0:
            triplet_score *= float(object_scores[s_idx].item()) * float(object_scores[o_idx].item())
        scores.append(triplet_score)

    if len(scores) == 0:
        return {
            "subj_labels": [],
            "pred_labels": [],
            "obj_labels": [],
            "subj_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
            "obj_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
            "scores": torch.empty((0,), dtype=torch.float32, device=device),
        }

    return {
        "subj_labels": subj_labels,
        "pred_labels": pred_labels,
        "obj_labels": obj_labels,
        "subj_boxes": torch.stack(subj_boxes, dim=0).to(device=device, dtype=torch.float32),
        "obj_boxes": torch.stack(obj_boxes, dim=0).to(device=device, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32, device=device),
    }


def _make_triplet_predictions_nogc(
    ex: Dict[str, Any],
    pair_list: List[Tuple[int, int]],
    rel_probs: torch.Tensor,
    object_labels: List[str],
    object_scores: torch.Tensor,
    pred_vocab: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    boxes = ex.get("obj_boxes", torch.empty((0, 4), dtype=torch.float32)).to(device)
    subj_labels: List[str] = []
    pred_labels: List[str] = []
    obj_labels: List[str] = []
    subj_boxes: List[torch.Tensor] = []
    obj_boxes: List[torch.Tensor] = []
    scores: List[float] = []

    for idx, pair in enumerate(pair_list):
        s_idx, o_idx = int(pair[0]), int(pair[1])
        if s_idx >= len(object_labels) or o_idx >= len(object_labels):
            continue
        if s_idx >= int(boxes.shape[0]) or o_idx >= int(boxes.shape[0]):
            continue
        for pred_id in range(int(rel_probs.shape[1])):
            if not (0 <= pred_id < len(pred_vocab)):
                continue
            subj_labels.append(str(object_labels[s_idx]).strip().lower())
            pred_labels.append(str(pred_vocab[pred_id]).strip().lower())
            obj_labels.append(str(object_labels[o_idx]).strip().lower())
            subj_boxes.append(boxes[s_idx])
            obj_boxes.append(boxes[o_idx])
            triplet_score = float(rel_probs[idx, pred_id].item())
            if int(object_scores.numel()) > 0:
                triplet_score *= float(object_scores[s_idx].item()) * float(object_scores[o_idx].item())
            scores.append(triplet_score)

    if len(scores) == 0:
        return {
            "subj_labels": [],
            "pred_labels": [],
            "obj_labels": [],
            "subj_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
            "obj_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
            "scores": torch.empty((0,), dtype=torch.float32, device=device),
        }

    return {
        "subj_labels": subj_labels,
        "pred_labels": pred_labels,
        "obj_labels": obj_labels,
        "subj_boxes": torch.stack(subj_boxes, dim=0).to(device=device, dtype=torch.float32),
        "obj_boxes": torch.stack(obj_boxes, dim=0).to(device=device, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32, device=device),
    }


def _make_detected_example(
    ex: Dict[str, Any],
    det_boxes: torch.Tensor,
    det_labels: List[str],
    clip_input_res: int,
) -> Optional[Dict[str, Any]]:
    image = ex.get("image", None)
    if not isinstance(image, Image.Image) or int(det_boxes.shape[0]) == 0 or len(det_labels) != int(det_boxes.shape[0]):
        return None
    try:
        boxes_224, _ = preprocess_boxes_to_clip224(image, det_boxes, out_size=int(clip_input_res))
    except Exception:
        boxes_224 = det_boxes.clone()
    out = dict(ex)
    out["obj_boxes"] = det_boxes.clone().float()
    out["obj_boxes_224"] = boxes_224.clone().float()
    out["obj_labels"] = [str(x).strip().lower() for x in det_labels]
    out["pairs"] = _build_all_ordered_pairs(len(det_labels))
    return out


def _empty_sgg_metrics(reason: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "R@20": 0.0,
        "R@50": 0.0,
        "R@100": 0.0,
        "mR@20": 0.0,
        "mR@50": 0.0,
        "mR@100": 0.0,
        "zR@20": 0.0,
        "zR@50": 0.0,
        "zR@100": 0.0,
        "n": 0.0,
        "num_gt": 0.0,
    }
    if str(reason).strip() != "":
        out["reason"] = str(reason)
    return out

def _get_hit_gt_indices(matches: torch.Tensor, pred_scores: torch.Tensor, ks: List[int]) -> Dict[int, torch.Tensor]:
    out = {int(k): torch.zeros(matches.shape[1], dtype=torch.bool, device=matches.device) for k in ks}
    if matches.shape[1] == 0:
        return out
    order = torch.argsort(pred_scores.float(), descending=True).to(device=matches.device)
    for k in ks:
        topk = order[:min(int(k), int(order.numel()))]
        if topk.numel() > 0:
            out[int(k)] = matches.index_select(0, topk).any(dim=0)
    return out


def _background_predicate_indices(pred_vocab: List[str]) -> List[int]:
    bg_names = {"relation", "no relation", "no interaction", "background", "__background__"}
    return [i for i, name in enumerate(pred_vocab) if str(name).strip().lower() in bg_names]

def _ensure_eval_predicate_vocab(pred_vocab: List[str]) -> List[str]:
    out = [str(p).strip().lower() for p in pred_vocab if str(p).strip() != ""]
    out = list(dict.fromkeys(out))
    if "relation" not in out:
        out.append("relation")
    return out


def _mask_background_predicates(probs: torch.Tensor, pred_vocab: List[str]) -> torch.Tensor:
    if probs.numel() == 0:
        return probs
    bg_idx = _background_predicate_indices(pred_vocab)
    if len(bg_idx) == 0:
        return probs
    masked = probs.clone()
    masked[:, torch.tensor(bg_idx, device=masked.device, dtype=torch.long)] = 0.0
    return masked

def _mask_background_logits(logits: torch.Tensor, pred_vocab: List[str]) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    bg_idx = _background_predicate_indices(pred_vocab)
    if len(bg_idx) == 0:
        return logits
    masked = logits.clone()
    masked[:, torch.tensor(bg_idx, device=masked.device, dtype=torch.long)] = -10000.0
    return masked

def _normalize_eval_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.float()
    logits = logits.float()
    finite = torch.isfinite(logits)
    safe = torch.where(finite, logits, torch.zeros_like(logits))
    mean = safe.mean(dim=-1, keepdim=True)
    std = safe.std(dim=-1, keepdim=True).clamp_min(1e-4)
    return torch.where(finite, (safe - mean) / std, logits)

def _predicate_counts_from_rows(rows: List[Any]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        rels = row.get("relationships", row.get("relations", row.get("rels", []))) if isinstance(row, dict) else []
        for rel in rels if isinstance(rels, list) else []:
            if isinstance(rel, dict):
                pred = str(rel.get("predicate", rel.get("pred", ""))).strip().lower()
                if pred:
                    counts[pred] += 1
    return counts


def _predicate_log_prior_for_eval(cfg: Any, loader: DataLoader, pred_vocab: List[str], device: torch.device) -> torch.Tensor:
    counts: Counter = Counter()
    train_jsonl = os.path.join(str(getattr(cfg, "vg150_root", "datasets")), "train.jsonl")
    if os.path.exists(train_jsonl):
        try:
            with open(train_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    counts.update(_predicate_counts_from_rows([json.loads(line)]))
        except Exception:
            counts = Counter()
    dataset = getattr(loader, "dataset", None)
    rows = getattr(dataset, "rows", None)
    if len(counts) == 0 and isinstance(rows, list):
        counts.update(_predicate_counts_from_rows(rows))
    scene_graphs = getattr(dataset, "scene_graphs", None)
    images = getattr(dataset, "images", None)
    valid_indices = getattr(dataset, "valid_indices", None)
    if len(counts) == 0 and isinstance(scene_graphs, dict) and isinstance(images, list) and isinstance(valid_indices, list):
        for img_idx in valid_indices:
            if int(img_idx) >= len(images):
                continue
            img_id = images[int(img_idx)].get("image_id")
            sg = scene_graphs.get(img_id, {})
            for rel in sg.get("relationships", []) if isinstance(sg, dict) else []:
                pred = str(rel.get("predicate", "")).strip().lower()
                if pred:
                    counts[pred] += 1
    freq = torch.tensor([max(1.0, float(counts.get(str(p), 1))) for p in pred_vocab], dtype=torch.float32, device=device)
    return torch.log(freq / freq.sum().clamp_min(1.0)).detach()


def _apply_eval_logit_adjustment(cfg: Any, logits: torch.Tensor, pred_log_prior: Optional[torch.Tensor]) -> torch.Tensor:
    eval_tau = float(getattr(cfg, "eval_logit_adj_tau", -1.0))
    tau = eval_tau if eval_tau >= 0.0 else float(getattr(cfg, "logit_adj_tau", 0.0))
    if tau <= 0.0 or pred_log_prior is None or int(pred_log_prior.numel()) != int(logits.shape[-1]):
        return logits
    return logits.float() - (tau * pred_log_prior.to(device=logits.device, dtype=torch.float32).view(1, -1))


def _freq_bias_key(subject_label: str, object_label: str) -> str:
    return f"{str(subject_label).strip().lower()}||{str(object_label).strip().lower()}"


def _load_frequency_bias(cfg: Any, pred_vocab: List[str], device: torch.device) -> Optional[Dict[str, Any]]:
    if not bool(getattr(cfg, "freq_bias_enabled", False)):
        return None
    path = str(getattr(cfg, "freq_bias_path", "")).strip()
    if path == "" or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None

    source_vocab = [str(x).strip().lower() for x in raw.get("predicate_vocab", [])]
    if len(source_vocab) == 0:
        return None
    source_index = {label: idx for idx, label in enumerate(source_vocab)}
    gather = [source_index.get(str(label).strip().lower(), -1) for label in pred_vocab]

    def remap(values: Any) -> Optional[torch.Tensor]:
        if not isinstance(values, list) or len(values) != len(source_vocab):
            return None
        out_values = []
        for idx in gather:
            out_values.append(float(values[idx]) if idx >= 0 else float(raw.get("default_log_prob", -20.0)))
        return torch.tensor(out_values, dtype=torch.float32, device=device)

    global_log_probs = remap(raw.get("global_log_probs", []))
    if global_log_probs is None:
        return None

    pair_log_probs = {}
    for key, values in raw.get("pair_log_probs", {}).items() if isinstance(raw.get("pair_log_probs", {}), dict) else []:
        mapped = remap(values)
        if mapped is not None:
            pair_log_probs[str(key).strip().lower()] = mapped

    subject_log_probs = {}
    for key, values in raw.get("subject_log_probs", {}).items() if isinstance(raw.get("subject_log_probs", {}), dict) else []:
        mapped = remap(values)
        if mapped is not None:
            subject_log_probs[str(key).strip().lower()] = mapped

    object_log_probs = {}
    for key, values in raw.get("object_log_probs", {}).items() if isinstance(raw.get("object_log_probs", {}), dict) else []:
        mapped = remap(values)
        if mapped is not None:
            object_log_probs[str(key).strip().lower()] = mapped

    return {
        "global": global_log_probs,
        "pairs": pair_log_probs,
        "subjects": subject_log_probs,
        "objects": object_log_probs,
        "smoothing": float(raw.get("smoothing", getattr(cfg, "freq_bias_smoothing", 1.0))),
        "num_source_predicates": int(len(source_vocab)),
    }


def _frequency_bias_for_pairs(
    freq_bias: Optional[Dict[str, Any]],
    pair_list: List[Tuple[int, int]],
    obj_labels: List[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if freq_bias is None or len(pair_list) == 0:
        return None
    global_log_probs = freq_bias.get("global", None)
    if not isinstance(global_log_probs, torch.Tensor):
        return None
    rows = []
    for subj_idx, obj_idx in pair_list:
        if 0 <= int(subj_idx) < len(obj_labels) and 0 <= int(obj_idx) < len(obj_labels):
            subj = str(obj_labels[int(subj_idx)]).strip().lower()
            obj = str(obj_labels[int(obj_idx)]).strip().lower()
            row = freq_bias.get("pairs", {}).get(_freq_bias_key(subj, obj))
            if row is None:
                subj_row = freq_bias.get("subjects", {}).get(subj)
                obj_row = freq_bias.get("objects", {}).get(obj)
                if isinstance(subj_row, torch.Tensor) and isinstance(obj_row, torch.Tensor):
                    row = 0.5 * (subj_row + obj_row)
                elif isinstance(subj_row, torch.Tensor):
                    row = subj_row
                elif isinstance(obj_row, torch.Tensor):
                    row = obj_row
                else:
                    row = global_log_probs
        else:
            row = global_log_probs
        rows.append(row.to(device=device, dtype=torch.float32))
    return torch.stack(rows, dim=0)


def _apply_frequency_bias(
    cfg: Any,
    logits: torch.Tensor,
    freq_bias: Optional[Dict[str, Any]],
    pair_list: List[Tuple[int, int]],
    obj_labels: List[str],
    device: torch.device,
) -> torch.Tensor:
    alpha = float(getattr(cfg, "bayes_calibration_weight", 0.0))
    if alpha == 0.0:
        alpha = float(getattr(cfg, "freq_bias_alpha", 0.0))
    if alpha <= 0.0:
        return logits
    pair_prior = _frequency_bias_for_pairs(freq_bias, pair_list, obj_labels, device)
    if pair_prior is None or tuple(pair_prior.shape) != tuple(logits.shape):
        return logits
    return logits.float() + (alpha * pair_prior)


def _relation_predicate_logits(
    cfg: Any,
    out: Any,
    rel_feat: torch.Tensor,
    pred_emb: torch.Tensor,
    pred_log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    text_logits = out.score(rel_feat, pred_emb).float()
    mode = str(getattr(cfg, "eval_sgg_predicate_score_mode", "ensemble")).strip().lower()
    use_classifier = bool(getattr(cfg, "eval_sgg_use_predicate_classifier", False)) and hasattr(out, "predicate_logits")
    if not use_classifier or mode in {"text", "cosine", "clip"}:
        return text_logits
    if bool(getattr(cfg, "adaptive_calibration_enabled", False)) and hasattr(out, "calibrated_predicate_logits"):
        cls_logits = out.calibrated_predicate_logits(rel_feat, pred_log_prior).float()
    else:
        cls_logits = out.predicate_logits(rel_feat).float()
    if int(cls_logits.shape[-1]) != int(pred_emb.shape[0]):
        return text_logits
    cls_logits = _apply_eval_logit_adjustment(cfg, cls_logits, pred_log_prior)
    if mode in {"classifier", "cls", "ce"}:
        return cls_logits
    cls_norm = _normalize_eval_logits(cls_logits)
    text_norm = _normalize_eval_logits(text_logits)
    cls_temp = max(1e-4, float(getattr(cfg, "eval_sgg_classifier_temperature", 1.0)))
    text_temp = max(1e-4, float(getattr(cfg, "eval_sgg_text_temperature", 1.0)))
    cls_norm = cls_norm / cls_temp
    text_norm = text_norm / text_temp
    if mode == "auto":
        top2 = cls_norm.topk(k=min(2, int(cls_norm.shape[-1])), dim=-1).values
        if int(top2.shape[-1]) == 2:
            alpha = (top2[:, 0] - top2[:, 1]).sigmoid().unsqueeze(1).clamp(0.25, 0.75)
        else:
            alpha = 0.5
    else:
        alpha = max(0.0, min(1.0, float(getattr(cfg, "eval_sgg_predicate_ensemble_alpha", 0.5))))
    return (alpha * cls_norm) + ((1.0 - alpha) * text_norm)


def _update_predicate_diag(diag: Dict[str, Any], pred_vocab: List[str], gt_labels: List[str], pair_pred_idx: torch.Tensor) -> None:
    diag["num_images"] = int(diag.get("num_images", 0)) + 1
    for label in gt_labels:
        key = str(label).strip().lower()
        if key:
            diag["gt_counts"][key] = int(diag["gt_counts"].get(key, 0)) + 1
    for pred_id in pair_pred_idx.detach().cpu().long().tolist():
        if 0 <= int(pred_id) < len(pred_vocab):
            key = str(pred_vocab[int(pred_id)]).strip().lower()
            diag["pred_counts"][key] = int(diag["pred_counts"].get(key, 0)) + 1


def _finalize_predicate_diag(diag: Dict[str, Any], topk: int) -> Dict[str, Any]:
    def top_items(counts: Dict[str, int]) -> List[Dict[str, Any]]:
        items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[: max(1, int(topk))]
        return [{"label": key, "count": int(value)} for key, value in items]
    return {
        "score_mode": str(diag.get("score_mode", "")),
        "ensemble_alpha": float(diag.get("ensemble_alpha", 0.0)),
        "num_images": int(diag.get("num_images", 0)),
        "gt_top": top_items(diag.get("gt_counts", {})),
        "pred_top": top_items(diag.get("pred_counts", {})),
    }


@torch.no_grad()
def eval_sgg_standard(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, Any]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    from .datasets.vg150_loader import scan_vg150_predicate_vocab

    ks = [20, 50, 100]
    iou_thresh = float(getattr(cfg, "eval_sgg_iou_thresh", 0.5))
    pred_vocab = _ensure_eval_predicate_vocab(scan_vg150_predicate_vocab(cfg.vg150_root))
    pred_log_prior = _predicate_log_prior_for_eval(cfg, loader, pred_vocab, device)
    freq_bias = _load_frequency_bias(cfg, pred_vocab, device)
    obj_vocab = _load_vg150_object_vocab(cfg.vg150_root)
    _, pred_emb = encode_predicate_vocab(clip_model, processor, pred_vocab, device, prompt_fn=pred_prompt_roles, direction="s2o")
    use_no_interaction_prior = bool(getattr(cfg, "eval_sgg_use_no_interaction_prior", False))
    no_interaction_emb = None
    if use_no_interaction_prior:
        no_text = str(getattr(cfg, "eval_sgg_no_interaction_text", "no relation or interaction between the objects")).strip()
        if no_text == "":
            no_text = "no relation or interaction between the objects"
        no_interaction_emb = clip_text_features(clip_model, processor, [no_text], device)
    _, obj_text_feats = _encode_object_vocab(clip_model, processor, obj_vocab, device)
    label_aliases = _build_vg_aliases(obj_vocab, pred_vocab) if bool(getattr(cfg, "eval_sgg_use_vg_aliases", True)) else None
    debug_budget = {
        "remaining": int(getattr(cfg, "eval_sgg_debug_triplets_max_samples", 5))
        if bool(getattr(cfg, "eval_sgg_debug_triplets", False))
        else 0
    }
    dino = _GroundingDinoWrapper(cfg, device)

    tasks = ["predcls", "predcls_nogc", "sgcls", "sgcls_nogc", "sgdet", "sgdet_nogc"]
    accum: Dict[str, Dict[str, Any]] = {
        t: {
            "image_r": {k: 0.0 for k in ks},
            "global_hits": {k: 0 for k in ks},
            "global_gt": 0,
            "n": 0,
        }
        for t in tasks
    }
    use_gt_pairs = bool(getattr(cfg, "eval_sgg_use_gt_pairs", False))
    zs_predicates = {
        str(x).strip().lower()
        for x in str(getattr(cfg, "eval_zs_predicates", "")).split(",")
        if str(x).strip() != ""
    }
    
    global_gt_counts = {p: 0 for p in pred_vocab}
    global_hits = {t: {k: {p: 0 for p in pred_vocab} for k in ks} for t in tasks}
    zs_gt_counts = {p: 0 for p in pred_vocab}
    zs_hits = {t: {k: {p: 0 for p in pred_vocab} for k in ks} for t in tasks}
    predicate_diag: Dict[str, Any] = {
        "score_mode": str(getattr(cfg, "eval_sgg_predicate_score_mode", "ensemble")),
        "ensemble_alpha": float(getattr(cfg, "eval_sgg_predicate_ensemble_alpha", 0.5)),
        "num_images": 0,
        "gt_counts": {},
        "pred_counts": {},
    }

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        eval_batch: List[Dict[str, Any]] = []
        pair_lists: List[List[Tuple[int, int]]] = []
        valid_mask: List[bool] = []
        for ex in batch:
            boxes = ex.get("obj_boxes", None)
            n_obj = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else 0
            pair_list = _extract_gt_pairs(ex) if use_gt_pairs else _build_all_ordered_pairs(n_obj)
            pair_lists.append(pair_list)
            valid = n_obj > 1 and len(pair_list) > 0
            valid_mask.append(valid)
            ex_eval = dict(ex)
            ex_eval["pairs"] = pair_list
            ex_eval["_pred_vocab"] = pred_vocab
            eval_batch.append(ex_eval)

        if not any(valid_mask):
            continue

        obj_anchor_feats = None
        anchor_source = "none"
        if bool(getattr(cfg, "object_language_anchor_enabled", False)):
            labels_for_anchor: List[List[str]] = []
            use_clip_labels = bool(getattr(cfg, "eval_sgg_use_clip_obj_classifier", True))
            for ex_eval in eval_batch:
                if use_clip_labels:
                    labels_i, _scores_i = _predict_object_labels_from_clip(cfg, clip_model, processor, ex_eval, obj_text_feats, obj_vocab, device)
                    anchor_source = "clip_pred"
                else:
                    labels_i = [str(x).strip().lower() for x in ex_eval.get("obj_labels", [])]
                    anchor_source = "gt"
                labels_for_anchor.append(labels_i)
            obj_anchor_feats = _object_semantic_feats_from_labels(labels_for_anchor, obj_vocab, obj_text_feats, device)

        _regs, rels, _rel_swaps, _assigns, _gates, _kept_idx = _forward_eval_batch(
            cfg,
            model,
            clip_model,
            processor,
            eval_batch,
            device,
            return_swapped=False,
            learned_prune_k_override=0,
            obj_semantic_feats=obj_anchor_feats,
        )

        for ex, pair_list, rel_feat, valid in zip(eval_batch, pair_lists, rels, valid_mask):
            if not valid:
                continue

            gt = _normalize_triplet_labels(_collect_gt_triplets(ex, device), label_aliases)
            if len(gt["pred_labels"]) == 0:
                continue

            for p in gt["pred_labels"]:
                if p in global_gt_counts:
                    global_gt_counts[p] += 1
                if p in zs_predicates and p in zs_gt_counts:
                    zs_gt_counts[p] += 1

            gt_labels = [str(x).strip().lower() for x in ex.get("obj_labels", [])]
            gt_obj_scores = torch.ones((len(gt_labels),), dtype=torch.float32, device=device)

            rel_logits = _relation_predicate_logits(cfg, out, rel_feat, pred_emb, pred_log_prior)
            rel_logits = _apply_frequency_bias(cfg, rel_logits, freq_bias, pair_list, gt_labels, device)
            rel_probs = _mask_background_logits(rel_logits, pred_vocab).softmax(dim=-1)
            if no_interaction_emb is not None:
                no_logits = out.score(rel_feat, no_interaction_emb).float()
                all_logits = torch.cat([rel_logits, no_logits], dim=-1)
                no_prob = all_logits.softmax(dim=-1)[:, -1].clamp(0.0, 1.0)
                interaction_prior = (1.0 - no_prob).to(dtype=torch.float32)
                rel_probs = _mask_background_predicates(rel_probs * interaction_prior.unsqueeze(1), pred_vocab)
            pair_pred_scores, pair_pred_idx = rel_probs.max(dim=-1)
            _update_predicate_diag(predicate_diag, pred_vocab, gt["pred_labels"], pair_pred_idx)
            pair_base_scores = torch.ones_like(pair_pred_scores)

            def _process_task(task_name: str, triplets: Dict[str, Any]):
                triplets = _normalize_triplet_labels(triplets, label_aliases)
                matches = _compute_pred_matches(triplets, gt, iou_thresh=iou_thresh)
                _print_triplet_debug(task_name, triplets, gt, matches, debug_budget)
                recalls = _recall_from_matches(matches, triplets["scores"], ks)
                hit_dict = _get_hit_gt_indices(matches, triplets["scores"], ks)
                num_gt = len(gt["pred_labels"])
                
                for k in ks:
                    accum[task_name]["image_r"][k] += float(recalls[k])
                    hits_at_k = hit_dict[k].cpu().tolist()
                    hit_count = int(sum(1 for is_hit in hits_at_k if bool(is_hit)))
                    accum[task_name]["global_hits"][k] += hit_count
                    for gi, is_hit in enumerate(hits_at_k):
                        if is_hit:
                            p = gt["pred_labels"][gi]
                            if p in global_hits[task_name][k]:
                                global_hits[task_name][k][p] += 1
                            if p in zs_predicates and p in zs_hits[task_name][k]:
                                zs_hits[task_name][k][p] += 1
                accum[task_name]["global_gt"] += int(num_gt)
                accum[task_name]["n"] += 1

            pred_triplets = _make_triplet_predictions(ex, pair_list, pair_base_scores, pair_pred_idx, pair_pred_scores, gt_labels, gt_obj_scores, device)
            _process_task("predcls", pred_triplets)

            if bool(getattr(cfg, "eval_sgg_report_nograph", True)):
                pred_triplets_ng = _make_triplet_predictions_nogc(ex, pair_list, rel_probs, gt_labels, gt_obj_scores, pred_vocab, device)
                _process_task("predcls_nogc", pred_triplets_ng)

            if bool(getattr(cfg, "eval_sgg_use_clip_obj_classifier", True)):
                pred_obj_labels, pred_obj_scores = _predict_object_labels_from_clip(cfg, clip_model, processor, ex, obj_text_feats, obj_vocab, device)
            else:
                pred_obj_labels, pred_obj_scores = gt_labels, gt_obj_scores

            if len(pred_obj_labels) == len(gt_labels):
                sgcls_triplets = _make_triplet_predictions(ex, pair_list, pair_base_scores, pair_pred_idx, pair_pred_scores, pred_obj_labels, pred_obj_scores, device)
                _process_task("sgcls", sgcls_triplets)

                if bool(getattr(cfg, "eval_sgg_report_nograph", True)):
                    sgcls_triplets_ng = _make_triplet_predictions_nogc(ex, pair_list, rel_probs, pred_obj_labels, pred_obj_scores, pred_vocab, device)
                    _process_task("sgcls_nogc", sgcls_triplets_ng)

            if bool(getattr(cfg, "eval_sgg_grounding_dino_enabled", True)) and isinstance(ex.get("image", None), Image.Image):
                det = dino.detect(ex["image"], obj_vocab, image_id=str(ex.get("image_id", "")))
                det_boxes = det.get("boxes", torch.empty((0, 4), dtype=torch.float32))
                det_labels = det.get("labels", [])
                det_scores = det.get("scores", torch.empty((0,), dtype=torch.float32))
                det_ex = _make_detected_example(ex, det_boxes, det_labels, int(cfg.clip_input_res))
                
                if det_ex is not None and len(det_ex["pairs"]) > 0:
                    _regs_d, rels_d, _sw_d, _as_d, _ga_d, _ki_d = _forward_eval_batch(cfg, model, clip_model, processor, [det_ex], device, return_swapped=False, learned_prune_k_override=0)
                    if len(rels_d) > 0:
                        rel_feat_d = rels_d[0]
                        rel_logits_d = _relation_predicate_logits(cfg, out, rel_feat_d, pred_emb, pred_log_prior)
                        rel_logits_d = _apply_frequency_bias(cfg, rel_logits_d, freq_bias, det_ex["pairs"], det_labels, device)
                        rel_probs_d = _mask_background_logits(rel_logits_d, pred_vocab).softmax(dim=-1)
                        if no_interaction_emb is not None:
                            no_logits_d = out.score(rel_feat_d, no_interaction_emb).float()
                            all_logits_d = torch.cat([rel_logits_d, no_logits_d], dim=-1)
                            no_prob_d = all_logits_d.softmax(dim=-1)[:, -1].clamp(0.0, 1.0)
                            interaction_prior_d = (1.0 - no_prob_d).to(dtype=torch.float32)
                            rel_probs_d = _mask_background_predicates(rel_probs_d * interaction_prior_d.unsqueeze(1), pred_vocab)
                        pair_pred_scores_d, pair_pred_idx_d = rel_probs_d.max(dim=-1)
                        
                        sgdet_triplets = _make_triplet_predictions(det_ex, det_ex["pairs"], torch.ones_like(pair_pred_scores_d), pair_pred_idx_d, pair_pred_scores_d, det_labels, det_scores.to(device=device, dtype=torch.float32), device)
                        _process_task("sgdet", sgdet_triplets)

                        if bool(getattr(cfg, "eval_sgg_report_nograph", True)):
                            sgdet_triplets_ng = _make_triplet_predictions_nogc(det_ex, det_ex["pairs"], rel_probs_d, det_labels, det_scores.to(device=device, dtype=torch.float32), pred_vocab, device)
                            _process_task("sgdet_nogc", sgdet_triplets_ng)

    def _compute_global_mr(hit_dict: Dict[str, int], counts: Dict[str, int]) -> float:
        vals = []
        for p, c in counts.items():
            if c > 0:
                vals.append(float(hit_dict[p]) / float(c))
        return float(sum(vals) / max(1, len(vals)))

    def _compute_global_recall(hit_count: int, gt_count: int) -> float:
        if int(gt_count) <= 0:
            return 0.0
        return float(hit_count) / float(gt_count)

    outm: Dict[str, Any] = {
        "settings": {
            "max_batches": int(max_batches),
            "iou_thresh": float(iou_thresh),
            "graph_constraint": True,
            "recall_aggregation": "dataset_global_hits_over_gt",
            "image_mean_recall_reported_as": "image_mean_R@K",
            "use_gt_pairs": bool(use_gt_pairs),
            "protocols": {
                "predcls": "gt_boxes+gt_labels",
                "sgcls": "gt_boxes+pred_labels",
                "sgdet": "zero_shot_detector_boxes+detector_labels",
            },
            "object_classifier": "clip_top1" if bool(getattr(cfg, "eval_sgg_use_clip_obj_classifier", True)) else "gt",
            "object_language_anchor_enabled": bool(getattr(cfg, "object_language_anchor_enabled", False)),
            "object_language_anchor_source": str(getattr(cfg, "object_language_anchor_source", "auto")),
            "object_language_anchor_eval_source": "clip_pred" if bool(getattr(cfg, "eval_sgg_use_clip_obj_classifier", True)) else "gt",
            "relation_context_layers": int(getattr(cfg, "relation_context_layers", 0)),
            "bayes_calibration_weight": float(getattr(cfg, "bayes_calibration_weight", 0.0)),
            "vg_alias_normalization": bool(label_aliases is not None),
            "debug_triplets": bool(getattr(cfg, "eval_sgg_debug_triplets", False)),
            "clip_obj_cache_enabled": bool(getattr(cfg, "eval_sgg_clip_obj_cache_enabled", True)),
            "clip_obj_cache_dir": str(getattr(cfg, "eval_sgg_clip_obj_cache_dir", "runs/clip_obj_cache")),
            "no_interaction_prior": bool(use_no_interaction_prior),
            "freq_bias_enabled": bool(freq_bias is not None),
            "freq_bias_path": str(getattr(cfg, "freq_bias_path", "")),
            "freq_bias_alpha": float(getattr(cfg, "freq_bias_alpha", 0.0)),
            "freq_bias_smoothing": float(freq_bias.get("smoothing", getattr(cfg, "freq_bias_smoothing", 1.0))) if isinstance(freq_bias, dict) else float(getattr(cfg, "freq_bias_smoothing", 1.0)),
            "freq_bias_source_predicates": int(freq_bias.get("num_source_predicates", 0)) if isinstance(freq_bias, dict) else 0,
            "no_interaction_text": str(getattr(cfg, "eval_sgg_no_interaction_text", "no relation or interaction between the objects")),
            "zero_shot_predicates": sorted(zs_predicates),
            "grounding_dino_enabled": bool(getattr(cfg, "eval_sgg_grounding_dino_enabled", True)),
            "grounding_dino_cache_enabled": bool(getattr(cfg, "eval_sgg_grounding_dino_cache_enabled", True)),
            "grounding_dino_cache_dir": str(getattr(cfg, "eval_sgg_grounding_dino_cache_dir", "runs/grounding_dino_cache")),
        }
    }
    
    for task in tasks:
        n = int(accum[task]["n"])
        if n <= 0:
            reason = "No valid images with GT triplets found."
            if task.startswith("sgdet") and dino.error is not None:
                reason = f"Grounding DINO unavailable: {dino.error}"
            outm[task] = _empty_sgg_metrics(reason=reason)
            continue
            
        gt_count = int(accum[task]["global_gt"])
        outm[task] = {
            "available": True,
            "graph_constraint": not task.endswith("_nogc"),
            "R@20": _compute_global_recall(accum[task]["global_hits"][20], gt_count),
            "R@50": _compute_global_recall(accum[task]["global_hits"][50], gt_count),
            "R@100": _compute_global_recall(accum[task]["global_hits"][100], gt_count),
            "mR@20": _compute_global_mr(global_hits[task][20], global_gt_counts),
            "mR@50": _compute_global_mr(global_hits[task][50], global_gt_counts),
            "mR@100": _compute_global_mr(global_hits[task][100], global_gt_counts),
            "zR@20": _compute_global_mr(zs_hits[task][20], zs_gt_counts) if len(zs_predicates) > 0 else 0.0,
            "zR@50": _compute_global_mr(zs_hits[task][50], zs_gt_counts) if len(zs_predicates) > 0 else 0.0,
            "zR@100": _compute_global_mr(zs_hits[task][100], zs_gt_counts) if len(zs_predicates) > 0 else 0.0,
            "image_mean_R@20": float(accum[task]["image_r"][20]) / float(n),
            "image_mean_R@50": float(accum[task]["image_r"][50]) / float(n),
            "image_mean_R@100": float(accum[task]["image_r"][100]) / float(n),
            "n": float(n),
            "num_gt": float(gt_count),
        }
    outm["predicate_diag"] = _finalize_predicate_diag(
        predicate_diag,
        int(getattr(cfg, "eval_sgg_predicate_diag_topk", 8)),
    )
    return outm

class _EvalTextCache:
    def __init__(self, clip_model: nn.Module, processor: Any, device: torch.device, max_size: int = 4096):
        self.clip_model = clip_model
        self.processor = processor
        self.device = device
        self.max_size = int(max_size)
        self.cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _store(self, key: str, value: torch.Tensor) -> torch.Tensor:
        self.cache[key] = value.detach()
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return self.cache[key]

    def get_many(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        keys = [str(text) for text in texts]
        missing: List[str] = []
        seen = set()
        out: Dict[str, torch.Tensor] = {}

        for key in keys:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                out[key] = value
            elif key not in seen:
                seen.add(key)
                missing.append(key)

        if len(missing) > 0:
            feats = clip_text_features(self.clip_model, self.processor, missing, self.device).detach()
            for key, feat in zip(missing, feats):
                out[key] = self._store(key, feat)

        return out

    def get(self, text: str) -> torch.Tensor:
        key = str(text)
        return self.get_many([key])[key]


class _GroundingDinoWrapper:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.processor = None
        self._error: Optional[str] = None

    def _cache_enabled(self) -> bool:
        return bool(getattr(self.cfg, "eval_sgg_grounding_dino_cache_enabled", True))

    def _cache_dir(self) -> str:
        return str(getattr(self.cfg, "eval_sgg_grounding_dino_cache_dir", "runs/grounding_dino_cache")).strip()

    def _cache_key(self, image_id: str, object_vocab: List[str]) -> str:
        import hashlib

        payload = {
            "image_id": str(image_id),
            "model_id": str(getattr(self.cfg, "eval_sgg_grounding_dino_model_id", "IDEA-Research/grounding-dino-tiny")),
            "box_threshold": float(getattr(self.cfg, "eval_sgg_grounding_dino_box_threshold", 0.25)),
            "text_threshold": float(getattr(self.cfg, "eval_sgg_grounding_dino_text_threshold", 0.25)),
            "max_detections": int(getattr(self.cfg, "eval_sgg_grounding_dino_max_detections", 64)),
            "vocab": [str(x).strip().lower() for x in object_vocab],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.blake2b(raw, digest_size=16).hexdigest()

    def _cache_path(self, image_id: str, object_vocab: List[str]) -> str:
        return os.path.join(self._cache_dir(), f"{self._cache_key(image_id, object_vocab)}.json")

    def _load_cache(self, image_id: str, object_vocab: List[str]) -> Optional[Dict[str, Any]]:
        if not self._cache_enabled() or str(image_id).strip() == "":
            return None
        path = self._cache_path(image_id, object_vocab)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            boxes = torch.tensor(payload.get("boxes", []), dtype=torch.float32)
            scores = torch.tensor(payload.get("scores", []), dtype=torch.float32)
            labels = [str(x).strip().lower() for x in payload.get("labels", [])]
            return {"boxes": boxes, "labels": labels, "scores": scores, "error": None, "cache_hit": True}
        except Exception:
            return None

    def _save_cache(self, image_id: str, object_vocab: List[str], result: Dict[str, Any]) -> None:
        if not self._cache_enabled() or str(image_id).strip() == "":
            return
        path = self._cache_path(image_id, object_vocab)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            payload = {
                "image_id": str(image_id),
                "boxes": result.get("boxes", torch.empty((0, 4), dtype=torch.float32)).tolist(),
                "scores": result.get("scores", torch.empty((0,), dtype=torch.float32)).tolist(),
                "labels": [str(x).strip().lower() for x in result.get("labels", [])],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            return

    @property
    def is_available(self) -> bool:
        return self._error is None and self.model is not None and self.processor is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if self._error is not None:
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            model_id = str(getattr(self.cfg, "eval_sgg_grounding_dino_model_id", "IDEA-Research/grounding-dino-tiny"))
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.model.eval()
        except Exception as exc:
            self._error = str(exc)
            self.model = None
            self.processor = None

    @torch.no_grad()
    def detect(self, image: Image.Image, object_vocab: List[str], image_id: str = "") -> Dict[str, Any]:
        cached = self._load_cache(image_id, object_vocab)
        if cached is not None:
            return cached

        self._ensure_loaded()
        if not self.is_available:
            return {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": [], "scores": torch.empty((0,), dtype=torch.float32), "error": self._error}

        valid_vocab = [str(x).strip().lower() for x in object_vocab if str(x).strip() != ""]
        if not valid_vocab:
            return {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": [], "scores": torch.empty((0,), dtype=torch.float32), "error": "Empty object vocabulary."}

        CHUNK_SIZE = 50
        all_boxes, all_scores, all_labels = [], [], []

        for i in range(0, len(valid_vocab), CHUNK_SIZE):
            chunk = valid_vocab[i:i + CHUNK_SIZE]
            prompt = ". ".join(chunk)
            
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                threshold=float(getattr(self.cfg, "eval_sgg_grounding_dino_box_threshold", 0.25)),
                text_threshold=float(getattr(self.cfg, "eval_sgg_grounding_dino_text_threshold", 0.25)),
                target_sizes=[image.size[::-1]],
            )
            
            if len(results) > 0:
                all_boxes.append(results[0].get("boxes", torch.empty((0, 4), dtype=torch.float32)).detach().cpu().float())
                all_scores.append(results[0].get("scores", torch.empty((0,), dtype=torch.float32)).detach().cpu().float())
                all_labels.extend(results[0].get("labels", []))

        if len(all_boxes) > 0:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = [str(x).strip().lower() for x in all_labels]

            if int(boxes.shape[0]) > 0:
                max_det = max(1, int(getattr(self.cfg, "eval_sgg_grounding_dino_max_detections", 64)))
                order = torch.argsort(scores, descending=True)[:max_det]
                boxes = boxes.index_select(0, order)
                scores = scores.index_select(0, order)
                labels = [labels[int(idx)] for idx in order.tolist()]
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)
            labels = []

        out = {"boxes": boxes, "labels": labels, "scores": scores, "error": None, "cache_hit": False}
        self._save_cache(image_id, object_vocab, out)
        return out

def _resolve_amp_dtype(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    amp_name = str(getattr(cfg, "amp_dtype", "bf16")).strip().lower()
    if amp_name == "fp16":
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _prepare_eval_batch(
    cfg: TrainConfig,
    batch: List[Dict[str, Any]],
    processor: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[Any]]]:
    if len(batch) == 0:
        raise ValueError("Empty batch passed to eval helper.")

    if all(isinstance(ex.get("pixel_values", None), torch.Tensor) for ex in batch):
        pixel = torch.stack([ex["pixel_values"] for ex in batch], dim=0).to(device, non_blocking=True)
    else:
        images = [ex["image"] for ex in batch]
        pixel = processor(images=images, return_tensors="pt")["pixel_values"].to(device, non_blocking=True)

    if device.type == "cuda" and bool(getattr(cfg, "channels_last", False)):
        pixel = pixel.contiguous(memory_format=torch.channels_last)

    if all(isinstance(ex.get("obj_boxes_224", None), torch.Tensor) for ex in batch):
        obj_boxes_224 = [ex["obj_boxes_224"].to(device, non_blocking=True) for ex in batch]
    else:
        obj_boxes_224 = []
        for ex in batch:
            b224, _ = preprocess_boxes_to_clip224(ex["image"], ex["obj_boxes"], out_size=int(cfg.clip_input_res))
            obj_boxes_224.append(b224.to(device, non_blocking=True))

    batch_pairs = [ex["pairs"] for ex in batch]
    return pixel, obj_boxes_224, batch_pairs


def _forward_eval_batch(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    batch: List[Dict[str, Any]],
    device: torch.device,
    return_swapped: bool = True,
    force_keep: Any = None,
    learned_prune_k_override: Optional[int] = None,
    obj_semantic_feats: Optional[List[torch.Tensor]] = None,
):
    out = model.module if isinstance(model, DDP) else model
    pixel, obj_boxes_224, batch_pairs = _prepare_eval_batch(cfg, batch, processor, device)
    amp_dtype = _resolve_amp_dtype(cfg, device)

    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(cfg.amp) and device.type == "cuda"):
        cm = unwrap_ddp(clip_model)
        vision_out = cm.vision_model(pixel_values=pixel)
        tokens = vision_out.last_hidden_state[:, 1:, :]
        batch_size, num_patches, hidden = tokens.shape
        side = int(math.sqrt(num_patches))
        feat_map = tokens.transpose(1, 2).reshape(batch_size, hidden, side, side)
        return out.forward_from_featmap(
            feat_map,
            obj_boxes_224=obj_boxes_224,
            pairs=batch_pairs,
            return_swapped=return_swapped,
            return_kept=True,
            force_keep=force_keep,
            learned_prune_k_override=learned_prune_k_override,
            obj_semantic_feats=obj_semantic_feats,
        )


def split_predicates_head_mid_tail(pred_vocab: List[str], pred_freq: Dict[str, int]) -> Dict[str, set]:
    freqs = sorted([(p, int(pred_freq.get(p, 0))) for p in pred_vocab], key=lambda x: x[1], reverse=True)
    n = len(freqs)
    if n == 0:
        return {"head": set(), "mid": set(), "tail": set()}
    head = set([p for p, _ in freqs[: max(1, int(0.2 * n))]])
    tail = set([p for p, _ in freqs[max(0, int(0.8 * n)) :]])
    mid = set(pred_vocab) - head - tail
    return {"head": head, "mid": mid, "tail": tail}


def _batch_predicate_query_texts(predicates: List[str]) -> List[str]:
    return [pred_prompt(str(p).strip().lower(), direction="s2o") for p in predicates if str(p).strip() != ""]


def _encode_unique_predicate_queries(
    predicates: List[str],
    clip_model: nn.Module,
    processor: "CLIPProcessor",
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    unique_preds: List[str] = []
    seen = set()
    for pred in predicates:
        p = str(pred).strip().lower()
        if p == "" or p in seen:
            continue
        seen.add(p)
        unique_preds.append(p)
    if len(unique_preds) == 0:
        return {}
    texts = _batch_predicate_query_texts(unique_preds)
    feats = clip_text_features(clip_model, processor, texts, device)
    return {p: feats[i] for i, p in enumerate(unique_preds)}


def _eval_topk_hits(sim_logits: torch.Tensor, gt_idx: torch.Tensor, ks: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {int(k): 0.0 for k in ks}
    if int(sim_logits.numel()) == 0 or int(gt_idx.numel()) == 0:
        return out
        
    m = int(sim_logits.shape[0])

    sorted_indices = torch.argsort(sim_logits, descending=True)
    target_idx = int(gt_idx.item())
    
    if target_idx >= m:
        return out
        
    rank = int((sorted_indices == target_idx).nonzero(as_tuple=True)[0].item())
    
    for k in ks:
        if rank < int(k):
            out[int(k)] = 1.0
            
    return out


@torch.no_grad()
def eval_query_grounding(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Evaluates visual grounding and predicate classification using raw Cosine Similarity.
    Logit adjustment tricks are removed to honestly benchmark representation learning.
    """
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    ks = [20, 50, 100]
    r_sum = {k: 0.0 for k in ks}
    n = 0

    per_pred_hits: Dict[str, Dict[int, float]] = {}
    per_pred_cnt: Dict[str, int] = {}

    from .datasets.vg150_loader import scan_vg150_predicate_vocab
    from .clip_utils import encode_predicate_vocab
    from .prompts import pred_prompt_roles

    pred_vocab = _ensure_eval_predicate_vocab(scan_vg150_predicate_vocab(cfg.vg150_root))
    _, pred_emb = encode_predicate_vocab(clip_model, processor, pred_vocab, device, prompt_fn=pred_prompt_roles, direction="s2o")
    pred_to_idx = {p: i for i, p in enumerate(pred_vocab)}

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, kept in zip(batch, rels, kept_idx):
                is_pos_orig = ex.get("rel_is_pos", None)
                preds_orig = ex.get("rel_preds", [])
                
                if is_pos_orig is None or len(preds_orig) == 0:
                    continue
                    
                is_pos_orig = is_pos_orig.to(device)
                m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig))
                orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)
                
                if int(orig_pos_idx.numel()) == 0:
                    continue

                kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
                m_kept = int(r.shape[0])
                if m_kept == 0:
                    continue
                
                orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

                sim_matrix = _relation_predicate_logits(cfg, out, r[:m_kept], pred_emb)
                bg_idx = _background_predicate_indices(pred_vocab)
                if len(bg_idx) > 0:
                    sim_matrix[:, torch.tensor(bg_idx, device=sim_matrix.device, dtype=torch.long)] = -10000.0
                sim_flat = sim_matrix.view(-1) 
                
                max_k = min(100, sim_flat.numel())
                _, global_topk_indices = torch.topk(sim_flat, k=max_k, largest=True)
                
                img_hits = {k: 0.0 for k in ks}
                used_q_total = 0

                for orig_qi in orig_pos_idx:
                    qi_val = int(orig_qi.item())
                    p = str(preds_orig[qi_val]).strip().lower()
                    if p not in pred_to_idx:
                        continue
                        
                    used_q_total += 1
                    per_pred_cnt[p] = int(per_pred_cnt.get(p, 0)) + 1
                    
                    if qi_val in orig_to_kept:
                        k_idx = orig_to_kept[qi_val]
                        gt_pred_id = pred_to_idx[p]
                        target_1d = k_idx * len(pred_vocab) + gt_pred_id
                        
                        rank_tensor = (global_topk_indices == target_1d).nonzero(as_tuple=False)
                        if rank_tensor.numel() > 0:
                            rank = rank_tensor[0][0].item()
                            for k in ks:
                                if rank < k:
                                    img_hits[k] += 1.0
                                    per_pred_hits.setdefault(p, {kk: 0.0 for kk in ks})
                                    per_pred_hits[p][k] += 1.0

                if used_q_total > 0:
                    for k in ks:
                        r_sum[k] += img_hits[k] / float(used_q_total)
                    n += 1

    if n == 0:
        return { "R@20": 0.0, "R@50": 0.0, "R@100": 0.0, "mR@20": 0.0, "mR@50": 0.0, "mR@100": 0.0, "n": 0.0, "n_predicates": 0.0 }

    def _mean_recall_at(k: int) -> float:
        vals: List[float] = []
        for p, cnt in per_pred_cnt.items():
            if int(cnt) <= 0:
                continue
            hits = per_pred_hits.get(p, {}).get(k, 0.0)
            vals.append(float(hits) / float(cnt))
        if len(vals) == 0:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    return {
        "R@20": r_sum[20] / float(n),
        "R@50": r_sum[50] / float(n),
        "R@100": r_sum[100] / float(n),
        "mR@20": _mean_recall_at(20),
        "mR@50": _mean_recall_at(50),
        "mR@100": _mean_recall_at(100),
        "n": float(n),
        "n_predicates": float(len(per_pred_cnt)),
    }

@torch.no_grad()
def eval_query_grounding_vs_k(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    ks: List[int],
    max_batches: int = 25,
) -> Dict[str, Any]:
    ks = [int(k) for k in ks if int(k) > 0]
    if len(ks) == 0:
        return {"ks": [], "vals": {}}

    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    outm: Dict[str, Any] = {"ks": ks, "vals": {}}
    text_cache = _EvalTextCache(clip_model, processor, device)

    for k in ks:
        r1 = 0
        r5 = 0
        r10 = 0
        n = 0

        for bi, batch in enumerate(loader):
            if bi >= int(max_batches):
                break

            regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
                cfg,
                model,
                clip_model,
                processor,
                batch,
                device,
                return_swapped=True,
                learned_prune_k_override=int(k),
            )

            for ex, r, kept in zip(batch, rels, kept_idx):
                is_pos_orig = ex.get("rel_is_pos", None)
                preds_orig = ex.get("rel_preds", [])
                
                if is_pos_orig is None or len(preds_orig) == 0:
                    continue
                    
                is_pos_orig = is_pos_orig.to(device)
                m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig))
                orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)
                
                if int(orig_pos_idx.numel()) == 0:
                    continue

                kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
                m_kept = int(r.shape[0])
                orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

                img_r1 = 0.0
                img_r5 = 0.0
                img_r10 = 0.0
                used_q = 0
                
                for orig_qi in orig_pos_idx:
                    qi_val = int(orig_qi.item())
                    p = str(preds_orig[qi_val]).strip().lower()
                    if p == "":
                        continue

                    used_q += 1
                    if qi_val not in orig_to_kept:
                        continue
                        
                    k_idx = orig_to_kept[qi_val]
                    q = text_cache.get(pred_prompt_roles(p, direction="s2o"))
                    sim = out.score(r[:m_kept], q.unsqueeze(0)).squeeze(1).float()

                    gt_q = torch.tensor([k_idx], device=device, dtype=torch.long)
                    hits = _eval_topk_hits(sim, gt_q, ks)

                    img_r1 += float(hits.get(1, 0.0))
                    img_r5 += float(hits.get(5, 0.0))
                    img_r10 += float(hits.get(10, 0.0))

                if used_q > 0:
                    r1 += img_r1 / float(used_q)
                    r5 += img_r5 / float(used_q)
                    r10 += img_r10 / float(used_q)
                    n += 1

        if n == 0:
            outm["vals"][str(k)] = {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "n": 0.0}
        else:
            outm["vals"][str(k)] = {"R@1": r1 / n, "R@5": r5 / n, "R@10": r10 / n, "n": float(n)}

    return outm

@torch.no_grad()
def eval_prompt_robustness(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    canonicalizer: Optional[PredicateCanonicalizer],
    max_batches: int = 50,
) -> Dict[str, Any]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    def _accum():
        return {"r1": 0, "r5": 0, "r10": 0, "n": 0}

    mets = {
        "base": _accum(),
        "roles": _accum(),
        "ensemble": _accum(),
        "paraphrase": _accum(),
    }
    text_cache = _EvalTextCache(clip_model, processor, device)

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, kept in zip(batch, rels, kept_idx):
            is_pos_orig = ex.get("rel_is_pos", None)
            preds_orig = ex.get("rel_preds", [])
            
            if is_pos_orig is None or len(preds_orig) == 0:
                continue

            is_pos_orig = is_pos_orig.to(device)
            m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig))
            orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)
            
            if orig_pos_idx.numel() == 0:
                continue

            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            m_kept = int(r.shape[0])
            orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

            img_sum = {k: {"r1": 0.0, "r5": 0.0, "r10": 0.0, "n": 0} for k in mets.keys()}

            def _acc_img(sim: torch.Tensor, key: str, k_idx: int):
                topk = sim.topk(k=min(10, sim.numel()), dim=0).indices
                top1 = topk[:1]
                top5 = topk[: min(5, topk.numel())]
                img_sum[key]["r1"] += float((top1 == k_idx).any().item())
                img_sum[key]["r5"] += float((top5 == k_idx).any().item())
                img_sum[key]["r10"] += float((topk == k_idx).any().item())
                img_sum[key]["n"] += 1

            prefetch_texts: List[str] = []
            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = str(preds_orig[qi_val]).strip().lower()
                if p == "":
                    continue
                prefetch_texts.append(pred_prompt(p, direction="s2o"))
                prefetch_texts.append(pred_prompt_roles(p, direction="s2o"))
                prefetch_texts.extend(pred_prompt_ensemble(p, direction="s2o"))
                if canonicalizer is not None and bool(getattr(canonicalizer, "enabled", False)):
                    paras = canonicalizer.generate_paraphrases(p, n=int(cfg.paraphrase_n))
                    if len(paras) > 0:
                        pp = str(paras[0]).strip().lower()
                        prefetch_texts.append(pred_prompt_roles(pp, direction="s2o"))
            prefetched = text_cache.get_many(prefetch_texts)

            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = str(preds_orig[qi_val]).strip().lower()
                if p == "":
                    continue

                if qi_val not in orig_to_kept:
                    for key in mets.keys():
                        img_sum[key]["n"] += 1
                    continue
                    
                k_idx = orig_to_kept[qi_val]
                
                q_base = prefetched[pred_prompt(p, direction="s2o")].unsqueeze(0)
                _acc_img(out.score(r[:m_kept], q_base).squeeze(1).float(), "base", k_idx)

                q_roles = prefetched[pred_prompt_roles(p, direction="s2o")].unsqueeze(0)
                _acc_img(out.score(r[:m_kept], q_roles).squeeze(1).float(), "roles", k_idx)

                ens = pred_prompt_ensemble(p, direction="s2o")
                q_ens = torch.stack([prefetched[t] for t in ens], dim=0)
                q_ens = F.normalize(q_ens.mean(dim=0, keepdim=True), dim=-1)
                _acc_img(out.score(r[:m_kept], q_ens).squeeze(1).float(), "ensemble", k_idx)

                if canonicalizer is not None and bool(getattr(canonicalizer, "enabled", False)):
                    paras = canonicalizer.generate_paraphrases(p, n=int(cfg.paraphrase_n))
                    if len(paras) > 0:
                        pp = str(paras[0]).strip().lower()
                        q_para = prefetched[pred_prompt_roles(pp, direction="s2o")].unsqueeze(0)
                        _acc_img(out.score(r[:m_kept], q_para).squeeze(1).float(), "paraphrase", k_idx)

            for key in mets.keys():
                if img_sum[key]["n"] <= 0:
                    continue
                mets[key]["r1"] += img_sum[key]["r1"] / float(img_sum[key]["n"])
                mets[key]["r5"] += img_sum[key]["r5"] / float(img_sum[key]["n"])
                mets[key]["r10"] += img_sum[key]["r10"] / float(img_sum[key]["n"])
                mets[key]["n"] += 1

    def _finalize(m):
        if m["n"] == 0:
            return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "n": 0.0}
        return {"R@1": m["r1"] / m["n"], "R@5": m["r5"] / m["n"], "R@10": m["r10"] / m["n"], "n": float(m["n"])}

    return {k: _finalize(v) for k, v in mets.items()}

@torch.no_grad()
def eval_prune_reliability(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: CLIPProcessor,
    loader: DataLoader,
    device: torch.device,
    k: Optional[int] = None,
    max_batches: int = 50,
) -> Dict[str, float]:
    k = int(cfg.learned_prune_k) if k is None else int(k)
    if k <= 0:
        return {"n_img": 0.0}

    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    total_pos = 0
    kept_pos = 0
    img_any_drop = 0
    n_img = 0
    kept_sum = 0
    cand_sum = 0

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg,
            model,
            clip_model,
            processor,
            batch,
            device,
            return_swapped=False,
            learned_prune_k_override=int(k),
        )

        for ex, kept in zip(batch, kept_idx):
            is_pos = ex.get("rel_is_pos", None)
            pairs = ex.get("pairs", [])
            if is_pos is None or len(pairs) == 0:
                continue

            y = is_pos.to(device)
            n = int(min(len(pairs), int(y.shape[0])))
            if n == 0:
                continue

            pos_idx = torch.nonzero(y[:n] > 0.5, as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            topk = torch.tensor(kept_l, device=device, dtype=torch.long)
            kk = int(topk.numel())
            if kk == 0:
                continue

            hit = int(torch.isin(pos_idx, topk).sum().item())
            total_pos += int(pos_idx.numel())
            kept_pos += int(hit)

            if hit < int(pos_idx.numel()):
                img_any_drop += 1

            kept_sum += int(kk)
            cand_sum += int(n)
            n_img += 1

    if n_img == 0:
        return {"n_img": 0.0}

    return {
        "pos_recall": float(kept_pos) / float(max(1, total_pos)),
        "img_any_drop_rate": float(img_any_drop) / float(max(1, n_img)),
        "avg_kept": float(kept_sum) / float(max(1, n_img)),
        "avg_cand": float(cand_sum) / float(max(1, n_img)),
        "n_img": float(n_img),
        "total_pos": float(total_pos),
    }

@torch.no_grad()
def eval_global_predicate_classification(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    pred_emb_s2o: torch.Tensor,
    pred_to_idx: Dict[str, int],
    pred_buckets: Optional[Dict[str, set]] = None,
    max_batches: int = 50,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    if pred_emb_s2o.numel() == 0:
        return {"acc@1": 0.0, "acc@5": 0.0, "n": 0.0}

    pred_emb_s2o = F.normalize(pred_emb_s2o.to(device), dim=-1)

    top1 = 0
    top5 = 0
    n = 0

    b_top1 = {"head": 0, "mid": 0, "tail": 0}
    b_n = {"head": 0, "mid": 0, "tail": 0}

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, kept in zip(batch, rels, kept_idx):
            preds_orig = ex.get("rel_preds", [])
            is_pos_orig = ex.get("rel_is_pos", None)
            
            if is_pos_orig is None or len(preds_orig) == 0:
                continue
                
            is_pos_orig = is_pos_orig.to(device)
            m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig))
            orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)
            
            if orig_pos_idx.numel() == 0:
                continue

            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            m_kept = int(r.shape[0])
            orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = preds_orig[qi_val]
                
                if p not in pred_to_idx:
                    continue
                    
                gt_idx = pred_to_idx[p]
                bname = None
                
                if pred_buckets is not None:
                    if p in pred_buckets.get("head", set()): bname = "head"
                    elif p in pred_buckets.get("tail", set()): bname = "tail"
                    else: bname = "mid"
                    
                is_ok1 = False
                is_ok5 = False
                
                if qi_val in orig_to_kept:
                    k_idx = orig_to_kept[qi_val]
                    sim = out.score(r[k_idx].unsqueeze(0), pred_emb_s2o).float()
                    p1 = sim.topk(k=1, dim=1).indices.squeeze(1)
                    p5 = sim.topk(k=min(5, sim.shape[1]), dim=1).indices
                    is_ok1 = bool(p1[0].item() == gt_idx)
                    is_ok5 = bool((p5[0] == gt_idx).any().item())
                    
                top1 += int(is_ok1)
                top5 += int(is_ok5)
                n += 1
                
                if bname is not None:
                    b_top1[bname] += int(is_ok1)
                    b_n[bname] += 1

    if n == 0:
        return {"acc@1": 0.0, "acc@5": 0.0, "n": 0.0}

    outm = {"acc@1": top1 / n, "acc@5": top5 / n, "n": float(n)}
    if pred_buckets is not None:
        outm.update(
            {
                "acc@1_head": float(b_top1["head"]) / max(1, b_n["head"]),
                "acc@1_mid": float(b_top1["mid"]) / max(1, b_n["mid"]),
                "acc@1_tail": float(b_top1["tail"]) / max(1, b_n["tail"]),
                "n_head": float(b_n["head"]),
                "n_mid": float(b_n["mid"]),
                "n_tail": float(b_n["tail"]),
            }
        )
    return outm


@torch.no_grad()
def eval_swap_consistency(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: CLIPProcessor,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    swap_cos = []

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, rs, kept in zip(batch, rels, rel_swaps, kept_idx):
            is_pos = ex.get("rel_is_pos", None)
            
            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            if is_pos is not None and len(kept_l) > 0:
                is_pos = is_pos[kept_l]
            if is_pos is None:
                continue
            is_pos = is_pos.to(device)

            m = min(r.shape[0], rs.shape[0], is_pos.shape[0])
            if m == 0:
                continue

            pos_idx = torch.nonzero(is_pos[:m] > 0.5, as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            rr = F.normalize(r[:m], dim=-1)
            rrs = F.normalize(rs[:m], dim=-1)
            cos = F.cosine_similarity(rr[pos_idx], rrs[pos_idx], dim=-1).detach().cpu().tolist()

            for c in cos:
                swap_cos.append(float(c))

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    mean_cos = _mean(swap_cos)
    total = float(len(swap_cos))
    return {
        "swap_cos": mean_cos,
        "sym_cos": mean_cos,
        "asym_cos": mean_cos,
        "n": total,
        "n_sym": total,
        "n_asym": total,
    }


@torch.no_grad()
def eval_query_grounding_split(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    heldout_pairs: set,
    max_batches: int = 50,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()
    text_cache = _EvalTextCache(clip_model, processor, device)

    def _accum():
        return {"r1": 0, "r5": 0, "r10": 0, "n": 0}

    met_train = _accum()
    met_held = _accum()

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, kept in zip(batch, rels, kept_idx):
            is_pos_orig = ex.get("rel_is_pos", None)
            preds_orig = ex.get("rel_preds", [])
            pair_names_orig = ex.get("rel_pair_names", [])

            if is_pos_orig is None or len(preds_orig) == 0 or len(pair_names_orig) == 0:
                continue

            is_pos_orig = is_pos_orig.to(device)
            m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig), len(pair_names_orig))
            orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)

            if int(orig_pos_idx.numel()) == 0:
                continue

            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            m_kept = int(r.shape[0])
            orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

            train_r1 = 0.0
            train_r5 = 0.0
            train_r10 = 0.0
            train_n = 0
            held_r1 = 0.0
            held_r5 = 0.0
            held_r10 = 0.0
            held_n = 0

            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = str(preds_orig[qi_val]).strip().lower()
                qpair = pair_names_orig[qi_val]
                
                if p == "":
                    continue

                cur_r1 = 0.0
                cur_r5 = 0.0
                cur_r10 = 0.0

                if qi_val in orig_to_kept:
                    k_idx = orig_to_kept[qi_val]
                    q = text_cache.get(pred_prompt(p, direction="s2o"))
                    sim = out.score(r[:m_kept], q.unsqueeze(0)).squeeze(1).float()

                    gt_q = torch.tensor([k_idx], device=device, dtype=torch.long)
                    hits = _eval_topk_hits(sim, gt_q, [1, 5, 10])

                    cur_r1 = float(hits.get(1, 0.0))
                    cur_r5 = float(hits.get(5, 0.0))
                    cur_r10 = float(hits.get(10, 0.0))

                if qpair in heldout_pairs:
                    held_r1 += cur_r1
                    held_r5 += cur_r5
                    held_r10 += cur_r10
                    held_n += 1
                else:
                    train_r1 += cur_r1
                    train_r5 += cur_r5
                    train_r10 += cur_r10
                    train_n += 1

            if train_n > 0:
                met_train["r1"] += train_r1 / float(train_n)
                met_train["r5"] += train_r5 / float(train_n)
                met_train["r10"] += train_r10 / float(train_n)
                met_train["n"] += 1
            if held_n > 0:
                met_held["r1"] += held_r1 / float(held_n)
                met_held["r5"] += held_r5 / float(held_n)
                met_held["r10"] += held_r10 / float(held_n)
                met_held["n"] += 1

    def _finalize(met):
        if met["n"] == 0:
            return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "n": 0.0}
        return {
            "R@1": met["r1"] / met["n"],
            "R@5": met["r5"] / met["n"],
            "R@10": met["r10"] / met["n"],
            "n": float(met["n"]),
        }

    outm: Dict[str, float] = {}
    outm.update({f"train_{k}": v for k, v in _finalize(met_train).items()})
    outm.update({f"heldout_{k}": v for k, v in _finalize(met_held).items()})
    return outm

@torch.no_grad()
def eval_object_leakage_diagnostic(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()
    text_cache = _EvalTextCache(clip_model, processor, device)

    cos_base_names = []
    cos_base_swapped = []
    flip_top1 = 0
    n = 0

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, kept in zip(batch, rels, kept_idx):
            is_pos_orig = ex.get("rel_is_pos", None)
            preds_orig = ex.get("rel_preds", [])
            pair_names_orig = ex.get("rel_pair_names", [])

            if is_pos_orig is None or len(preds_orig) == 0 or len(pair_names_orig) == 0:
                continue

            is_pos_orig = is_pos_orig.to(device)
            m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig), len(pair_names_orig))
            orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)

            if int(orig_pos_idx.numel()) == 0:
                continue

            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            m_kept = int(r.shape[0])
            orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

            img_cos_base_names = 0.0
            img_cos_base_swapped = 0.0
            img_flip_top1 = 0.0
            img_n = 0

            prompt_bank: List[str] = []
            prompt_map: Dict[int, Dict[str, str]] = {}
            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = str(preds_orig[qi_val]).strip().lower()
                if p == "" or qi_val not in orig_to_kept:
                    continue
                s_name, o_name = pair_names_orig[qi_val]
                prompts = counterfactual_prompt_variants(p, s_name=s_name, o_name=o_name, direction="s2o")
                prompt_map[qi_val] = prompts
                prompt_bank.extend([prompts["base"], prompts["with_names"], prompts["with_names_swapped"]])
            prefetched = text_cache.get_many(prompt_bank)

            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = str(preds_orig[qi_val]).strip().lower()
                if p == "":
                    continue

                if qi_val not in orig_to_kept:
                    continue
                    
                k_idx = orig_to_kept[qi_val]
                prompts = prompt_map[qi_val]
                q_base = prefetched[prompts["base"]].unsqueeze(0)
                q_name = prefetched[prompts["with_names"]].unsqueeze(0)
                q_swap = prefetched[prompts["with_names_swapped"]].unsqueeze(0)

                rr = r[k_idx].unsqueeze(0)
                s_base = out.score(rr, q_base).squeeze(1).float()
                s_name_v = out.score(rr, q_name).squeeze(1).float()
                s_swap = out.score(rr, q_swap).squeeze(1).float()

                img_cos_base_names += float(F.cosine_similarity(s_base, s_name_v, dim=0).item())
                img_cos_base_swapped += float(F.cosine_similarity(s_base, s_swap, dim=0).item())
                if int(torch.argmax(s_name_v).item()) != int(torch.argmax(s_swap).item()):
                    img_flip_top1 += 1.0
                img_n += 1

            if img_n <= 0:
                continue

            cos_base_names.append(img_cos_base_names / float(img_n))
            cos_base_swapped.append(img_cos_base_swapped / float(img_n))
            flip_top1 += (img_flip_top1 / float(img_n))
            n += 1

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    return {
        "scorevec_cos_base_vs_names": _mean(cos_base_names),
        "scorevec_cos_base_vs_swapped": _mean(cos_base_swapped),
        "top1_flip_rate_names_swap": float(flip_top1) / max(1, n),
        "n": float(n),
    }

@torch.no_grad()
def eval_latency_throughput(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: CLIPProcessor,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 25,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    times_ms: List[float] = []
    n_img = 0
    n_pairs = 0

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))

        n_img += int(len(batch))
        for ex, kept in zip(batch, kept_idx):
            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            n_pairs += int(len(kept_l))

    if len(times_ms) == 0:
        return {"n_img": float(n_img), "n_pairs": float(n_pairs)}

    mean_ms = float(sum(times_ms) / max(1, len(times_ms)))
    return {
        "ms_per_batch": mean_ms,
        "ms_per_image": mean_ms / max(1.0, float(cfg.batch_size)),
        "pairs_per_image": float(n_pairs) / max(1.0, float(n_img)),
        "n_img": float(n_img),
        "n_pairs": float(n_pairs),
    }

@torch.no_grad()
def eval_prune_tradeoff(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: CLIPProcessor,
    loader: DataLoader,
    device: torch.device,
    ks: Optional[List[int]] = None,
    max_batches: int = 50,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()

    if ks is None:
        ks = [int(cfg.learned_prune_k)]
    ks = [int(k) for k in ks if int(k) > 0]
    if len(ks) == 0:
        return {"n_img": 0.0}

    total_pos = {k: 0 for k in ks}
    kept_pos = {k: 0 for k in ks}
    kept_k_sum = {k: 0 for k in ks}
    n_img = 0
    cand_sum = 0

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        kept_per_k: Dict[int, List[torch.Tensor]] = {}
        for k in ks:
            regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
                cfg,
                model,
                clip_model,
                processor,
                batch,
                device,
                return_swapped=False,
                learned_prune_k_override=int(k),
            )
            kept_per_k[int(k)] = kept_idx

        for bi, ex in enumerate(batch):
            is_pos = ex.get("rel_is_pos", None)
            pairs = ex.get("pairs", [])
            if is_pos is None or len(pairs) == 0:
                continue
            y = is_pos.to(device)
            n = int(min(len(pairs), int(y.shape[0])))
            if n == 0:
                continue
            pos_idx = torch.nonzero(y[:n] > 0.5, as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            for k in ks:
                kept = kept_per_k[int(k)][bi]
                kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
                topk = torch.tensor(kept_l, device=device, dtype=torch.long)
                kk = int(topk.numel())
                if kk == 0:
                    continue
                hit = int(torch.isin(pos_idx, topk).sum().item())
                total_pos[k] += int(pos_idx.numel())
                kept_pos[k] += int(hit)
                kept_k_sum[k] += int(kk)

            n_img += 1
            cand_sum += int(n)

    if n_img == 0:
        return {"n_img": 0.0}

    outm: Dict[str, float] = {"avg_cand": float(cand_sum) / float(n_img), "n_img": float(n_img)}

    for k in ks:
        if total_pos[k] <= 0:
            outm[f"recall@{k}"] = 0.0
        else:
            outm[f"recall@{k}"] = float(kept_pos[k]) / float(total_pos[k])
        outm[f"avg_kept@{k}"] = float(kept_k_sum[k]) / float(n_img)

    k0 = int(ks[0])
    outm["recall@k"] = float(outm.get(f"recall@{k0}", 0.0))
    outm["avg_kept"] = float(outm.get(f"avg_kept@{k0}", 0.0))

    return outm

@torch.no_grad()
def eval_geometry_hard_negative_stress(
    cfg: TrainConfig,
    model: nn.Module,
    clip_model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    hard_k: int = 16,
    max_batches: int = 50,
) -> Dict[str, float]:
    out = model.module if isinstance(model, DDP) else model
    out.eval()
    unwrap_ddp(clip_model).eval()
    text_cache = _EvalTextCache(clip_model, processor, device)

    r1 = 0
    r5 = 0
    n = 0

    for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break

        regs, rels, rel_swaps, assigns, gates, kept_idx = _forward_eval_batch(
            cfg, model, clip_model, processor, batch, device, return_swapped=True
        )

        for ex, r, kept in zip(batch, rels, kept_idx):
            is_pos_orig = ex.get("rel_is_pos", None)
            preds_orig = ex.get("rel_preds", [])
            pairs_orig = ex.get("pairs", [])

            if is_pos_orig is None or len(preds_orig) == 0 or len(pairs_orig) == 0:
                continue

            is_pos_orig = is_pos_orig.to(device)
            m_orig = min(int(is_pos_orig.shape[0]), len(preds_orig), len(pairs_orig))
            orig_pos_idx = torch.nonzero(is_pos_orig[:m_orig] > 0.5, as_tuple=False).squeeze(1)
            orig_neg_idx = torch.nonzero(is_pos_orig[:m_orig] <= 0.5, as_tuple=False).squeeze(1)
            
            if orig_pos_idx.numel() == 0 or orig_neg_idx.numel() == 0:
                continue

            kept_l = kept.detach().cpu().tolist() if isinstance(kept, torch.Tensor) else list(kept)
            m_kept = int(r.shape[0])
            orig_to_kept = {orig_idx: k_idx for k_idx, orig_idx in enumerate(kept_l) if k_idx < m_kept}

            img_r1 = 0.0
            img_r5 = 0.0
            img_n = 0
            
            for orig_qi in orig_pos_idx:
                qi_val = int(orig_qi.item())
                p = str(preds_orig[qi_val]).strip().lower()
                if p == "":
                    continue

                img_n += 1
                if qi_val not in orig_to_kept:
                    continue

                k_idx = orig_to_kept[qi_val]
                
                g_pos = torch.tensor(pairs_orig[qi_val][3], device=device, dtype=torch.float32)
                g_negs = torch.tensor([pairs_orig[i][3] for i in orig_neg_idx.tolist()], device=device, dtype=torch.float32)
                d = (g_negs - g_pos[None, :]).pow(2).sum(dim=1)
                
                take = min(int(hard_k - 1), int(d.numel()))
                hard_neg_orig = orig_neg_idx[d.topk(k=take, largest=False).indices]
                
                cand_idx_kept = [k_idx]
                for hn_orig in hard_neg_orig.tolist():
                    if hn_orig in orig_to_kept:
                        cand_idx_kept.append(orig_to_kept[hn_orig])
                        
                cand_idx_tensor = torch.tensor(cand_idx_kept, device=device, dtype=torch.long)
                rr = r[:m_kept][cand_idx_tensor]
                
                q = text_cache.get(pred_prompt(p, direction="s2o")).unsqueeze(0)
                sim = out.score(rr, q).squeeze(1).float()

                rank = int(torch.argsort(sim, descending=True).tolist().index(0))
                img_r1 += float(rank == 0)
                img_r5 += float(rank < 5)

            if img_n > 0:
                r1 += img_r1 / float(img_n)
                r5 += img_r5 / float(img_n)
                n += 1

    if n == 0:
        return {"hard_R@1": 0.0, "hard_R@5": 0.0, "n": 0.0}
    return {"hard_R@1": r1 / n, "hard_R@5": r5 / n, "n": float(n)}

if __name__ == "__main__":
    import argparse
    from transformers import CLIPModel, CLIPProcessor
    from .config import TrainConfig
    from .models.relational_model import RelationalModel
    from .datasets.vg150_loader import VG150LoaderConfig, VG150DataLoader
    from .ddp_utils import seed_all

    parser = argparse.ArgumentParser(description="Standalone Evaluation for PURE-SPOA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, default="validation")
    parser.add_argument("--vg150_root", type=str, default="datasets")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_streaming", type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.vg150_root = args.vg150_root
    cfg.hf_streaming = args.hf_streaming
    cfg.batch_size = args.batch_size
    cfg.device = args.device

    seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"[System] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    if "cfg" in ckpt and "emb_dim" in ckpt["cfg"]:
        cfg.emb_dim = int(ckpt["cfg"]["emb_dim"])
    else:
        cfg.emb_dim = 768  
        
    from .clip_utils import configure_clip
    print("[System] Loading CLIP backbone...")
    clip_model, processor, clip_vision_dim, text_dim = configure_clip(cfg.clip_name, device)
    clip_model.eval()

    model = RelationalModel(cfg=cfg, clip_vision_dim=clip_vision_dim, text_dim=text_dim)
    state_dict = ckpt.get("model", ckpt)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    if "clip" in ckpt:
        print("[System] Loading fine-tuned CLIP weights...")
        clip_state_dict = {k.replace("module.", ""): v for k, v in ckpt["clip"].items()}
        clip_model.load_state_dict(clip_state_dict, strict=False)

    print(f"[Data] Initializing DataLoader (split: {args.eval_dataset})...")
    loader_cfg = VG150LoaderConfig(
        vg150_root=cfg.vg150_root, 
        split=args.eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        streaming=cfg.hf_streaming,
        hf_dataset_id=cfg.hf_dataset_id
    )
    loader = VG150DataLoader(loader_cfg)
    
    print(f"\n[Evaluate] Running standard Cosine Evaluation...")
    with torch.no_grad():
        metrics = eval_query_grounding(
            cfg=cfg,
            model=model,
            clip_model=clip_model,
            processor=processor,
            loader=loader,
            device=device,
            max_batches=200
        )

    print("\n[Final Metrics]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    import argparse
    from transformers import CLIPModel, CLIPProcessor
    from .config import TrainConfig
    from .models.relational_model import RelationalModel
    from .datasets.vg150_loader import VG150LoaderConfig, VG150DataLoader
    from .ddp_utils import seed_all

    parser = argparse.ArgumentParser(description="Standalone Evaluation for PURE-SPOA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, default="validation")
    parser.add_argument("--vg150_root", type=str, default="datasets")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_streaming", type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.vg150_root = args.vg150_root
    cfg.hf_streaming = args.hf_streaming
    cfg.batch_size = args.batch_size
    cfg.device = args.device

    seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("[System] Loading CLIP backbone...")
    clip_model = CLIPModel.from_pretrained(cfg.clip_name).to(device)
    processor = CLIPProcessor.from_pretrained(cfg.clip_name)
    clip_model.eval()

    print(f"[System] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    if "cfg" in ckpt and "emb_dim" in ckpt["cfg"]:
        cfg.emb_dim = int(ckpt["cfg"]["emb_dim"])
    else:
        cfg.emb_dim = 768  
        
    from .clip_utils import configure_clip
    clip_model, processor, clip_vision_dim, text_dim = configure_clip(cfg.clip_name, device)
    clip_model.eval()

    model = RelationalModel(cfg=cfg, clip_vision_dim=clip_vision_dim, text_dim=text_dim)
    state_dict = ckpt.get("model", ckpt)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    if "clip" in ckpt:
        print("[System] Loading fine-tuned CLIP weights...")
        clip_state_dict = {k.replace("module.", ""): v for k, v in ckpt["clip"].items()}
        clip_model.load_state_dict(clip_state_dict, strict=False)

    print(f"[Data] Initializing DataLoader (split: {args.eval_dataset})...")
    loader_cfg = VG150LoaderConfig(
        vg150_root=cfg.vg150_root, 
        split=args.eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        streaming=cfg.hf_streaming,
        hf_dataset_id=cfg.hf_dataset_id
    )
    loader = VG150DataLoader(loader_cfg)
    
    print(f"\n[Evaluate] Running standard Cosine Evaluation...")
    with torch.no_grad():
        metrics = eval_query_grounding(
            cfg=cfg,
            model=model,
            clip_model=clip_model,
            processor=processor,
            loader=loader,
            device=device,
            max_batches=200
        )

    print("\n[Final Metrics]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")