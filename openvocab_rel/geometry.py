import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def xyxy_from_vg(obj: Dict[str, Any]) -> List[float]:
    x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
    return [x, y, x + w, y + h]


def geom_feats(b1: List[float], b2: List[float]) -> List[float]:
    x1, y1, x2, y2 = b1
    x1p, y1p, x2p, y2p = b2
    w1, h1 = max(1.0, x2 - x1), max(1.0, y2 - y1)
    w2, h2 = max(1.0, x2p - x1p), max(1.0, y2p - y1p)
    c1x, c1y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    c2x, c2y = (x1p + x2p) / 2.0, (y1p + y2p) / 2.0
    dx = (c2x - c1x) / (w1 + 1e-6)
    dy = (c2y - c1y) / (h1 + 1e-6)
    rw = math.log(w2 / (w1 + 1e-6))
    rh = math.log(h2 / (h1 + 1e-6))
    ar1 = math.log((w1 / (h1 + 1e-6)))
    ar2 = math.log((w2 / (h2 + 1e-6)))
    a1 = math.log(w1 * h1)
    a2 = math.log(w2 * h2)
    return [dx, dy, rw, rh, ar1, ar2, a1, a2]


def geom_feats_torch(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    b1 = b1.to(dtype=torch.float32)
    b2 = b2.to(dtype=torch.float32)
    x1, y1, x2, y2 = b1.unbind(dim=1)
    x1p, y1p, x2p, y2p = b2.unbind(dim=1)

    w1 = (x2 - x1).clamp_min(1.0)
    h1 = (y2 - y1).clamp_min(1.0)
    w2 = (x2p - x1p).clamp_min(1.0)
    h2 = (y2p - y1p).clamp_min(1.0)

    c1x = (x1 + x2) / 2.0
    c1y = (y1 + y2) / 2.0
    c2x = (x1p + x2p) / 2.0
    c2y = (y1p + y2p) / 2.0

    dx = ((c2x - c1x) / (w1 + 1e-6)).clamp(-10.0, 10.0)
    dy = ((c2y - c1y) / (h1 + 1e-6)).clamp(-10.0, 10.0)
    rw = torch.log(w2 / (w1 + 1e-6)).clamp(-5.0, 5.0)
    rh = torch.log(h2 / (h1 + 1e-6)).clamp(-5.0, 5.0)
    ar1 = torch.log(w1 / (h1 + 1e-6)).clamp(-5.0, 5.0)
    ar2 = torch.log(w2 / (h2 + 1e-6)).clamp(-5.0, 5.0)
    a1 = torch.log(w1 * h1)
    a2 = torch.log(w2 * h2)

    return torch.stack([dx, dy, rw, rh, ar1, ar2, a1, a2], dim=1)


def centers_wh(boxes_xyxy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp_min(1.0)
    h = (y2 - y1).clamp_min(1.0)
    return torch.stack([cx, cy], dim=1), torch.stack([w, h], dim=1)


def preprocess_boxes_to_clip224(
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    out_size: int = 224,
) -> Tuple[torch.Tensor, Tuple[int, int, float, int, int]]:
    w, h = image.size
    s = out_size / float(min(w, h))
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    left = int(round((new_w - out_size) / 2.0))
    top = int(round((new_h - out_size) / 2.0))

    b = boxes_xyxy.clone().float()
    b = b * s
    b[:, [0, 2]] -= float(left)
    b[:, [1, 3]] -= float(top)
    b = torch.clamp(b, 0.0, float(out_size))
    b[:, 2] = torch.maximum(b[:, 2], b[:, 0] + 1.0)
    b[:, 3] = torch.maximum(b[:, 3], b[:, 1] + 1.0)

    return b, (w, h, s, left, top)


def jitter_boxes_xyxy(boxes: torch.Tensor, frac: float, img_w: float, img_h: float) -> torch.Tensor:
    frac = float(frac)
    if frac <= 0:
        return boxes
    b = boxes.clone().float()
    w = (b[:, 2] - b[:, 0]).clamp_min(1.0)
    h = (b[:, 3] - b[:, 1]).clamp_min(1.0)
    dx = (torch.rand_like(w) * 2 - 1) * (w * frac)
    dy = (torch.rand_like(h) * 2 - 1) * (h * frac)
    dw = (torch.rand_like(w) * 2 - 1) * (w * frac)
    dh = (torch.rand_like(h) * 2 - 1) * (h * frac)

    b[:, 0] = b[:, 0] + dx
    b[:, 1] = b[:, 1] + dy
    b[:, 2] = b[:, 2] + dx + dw
    b[:, 3] = b[:, 3] + dy + dh

    x1 = torch.minimum(b[:, 0], b[:, 2])
    y1 = torch.minimum(b[:, 1], b[:, 3])
    x2 = torch.maximum(b[:, 0], b[:, 2])
    y2 = torch.maximum(b[:, 1], b[:, 3])
    b = torch.stack([x1, y1, x2, y2], dim=1)

    b[:, 0::2] = b[:, 0::2].clamp(0.0, float(img_w))
    b[:, 1::2] = b[:, 1::2].clamp(0.0, float(img_h))
    b[:, 2] = torch.maximum(b[:, 2], b[:, 0] + 1.0)
    b[:, 3] = torch.maximum(b[:, 3], b[:, 1] + 1.0)
    return b


def sample_random_boxes_xyxy(
    n: int,
    img_w: float,
    img_h: float,
    min_scale: float = 0.05,
    max_scale: float = 0.5,
) -> torch.Tensor:
    n = int(n)
    if n <= 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    min_scale = float(min_scale)
    max_scale = float(max_scale)
    img_w = float(img_w)
    img_h = float(img_h)

    ws = (torch.rand(n) * (max_scale - min_scale) + min_scale) * img_w
    hs = (torch.rand(n) * (max_scale - min_scale) + min_scale) * img_h

    x1 = torch.rand(n) * (img_w - ws).clamp(min=1.0)
    y1 = torch.rand(n) * (img_h - hs).clamp(min=1.0)
    x2 = x1 + ws
    y2 = y1 + hs

    b = torch.stack([x1, y1, x2, y2], dim=1)
    b[:, 2] = torch.maximum(b[:, 2], b[:, 0] + 1.0)
    b[:, 3] = torch.maximum(b[:, 3], b[:, 1] + 1.0)
    return b


def build_all_pairs(n_obj: int, max_pairs: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s in range(int(n_obj)):
        for o in range(int(n_obj)):
            if s == o:
                continue
            out.append((s, o))
            if len(out) >= int(max_pairs):
                return out
    return out


def reconstruct_gt_edges_from_example(ex: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    pairs = ex.get("pairs", [])
    preds = ex.get("rel_preds", [])
    is_pos = ex.get("rel_is_pos", None)
    if is_pos is None:
        return {}

    gt: Dict[Tuple[int, int], str] = {}
    n = min(len(pairs), len(preds), int(is_pos.shape[0]))
    for i in range(n):
        if float(is_pos[i].item()) <= 0.5:
            continue
        p = str(preds[i]).strip().lower()
        if p == "":
            continue
        s, o, _, _ = pairs[i]
        gt[(int(s), int(o))] = p
    return gt


def build_candidate_tensors_from_gt(
    obj_boxes: torch.Tensor,
    obj_labels: List[str],
    gt_edges: Dict[Tuple[int, int], str],
    max_pairs: int,
) -> Tuple[List[Tuple], List[str], torch.Tensor, List[Tuple[str, str]]]:
    cand = build_all_pairs(int(obj_boxes.shape[0]), int(max_pairs))

    pairs: List[Tuple] = []
    rel_preds: List[str] = []
    rel_is_pos: List[float] = []
    rel_pair_names: List[Tuple[str, str]] = []

    for (s, o) in cand:
        b1 = obj_boxes[s].tolist()
        b2 = obj_boxes[o].tolist()
        g = geom_feats(b1, b2)
        pairs.append((s, o, None, g))
        rel_pair_names.append((obj_labels[s], obj_labels[o]))

        if (s, o) in gt_edges:
            rel_is_pos.append(1.0)
            rel_preds.append(gt_edges[(s, o)])
        else:
            rel_is_pos.append(0.0)
            rel_preds.append("")

    return pairs, rel_preds, torch.tensor(rel_is_pos, dtype=torch.float32), rel_pair_names


def prune_pairs_by_geometry(obj_boxes: torch.Tensor, cand: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
    k = int(k)
    if k <= 0 or len(cand) <= k:
        return cand
    centers, wh = centers_wh(obj_boxes)
    scores: List[float] = []
    for (s, o) in cand:
        ds = (centers[s] - centers[o]).pow(2).sum().sqrt()
        scale = (wh[s].mean() + wh[o].mean()).clamp_min(1.0)
        scores.append(float((ds / scale).item()))
    idx = np.argsort(np.array(scores))[:k]
    return [cand[int(i)] for i in idx]
