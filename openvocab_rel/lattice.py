from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def build_predicate_similarity_matrix(
    predicate_text_features: torch.Tensor,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    """Build a clipped [0, 1] predicate lattice from normalized text features.

    The matrix is intentionally lightweight: it uses the same predicate text
    embeddings already computed for evaluation/training, so it adds no new
    model dependency and keeps PURE's objective compact.
    """
    if predicate_text_features.numel() == 0:
        return torch.empty((0, 0), device=predicate_text_features.device)
    feats = F.normalize(predicate_text_features.float(), dim=-1)
    sim = feats @ feats.t()
    sim = (sim + 1.0) * 0.5
    sim = sim.clamp(float(min_value), float(max_value))
    eye = torch.eye(int(sim.shape[0]), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, 1.0)
    return sim


def lattice_negative_weights(
    anchor_pred_ids: Optional[torch.Tensor],
    negative_pred_ids: Optional[torch.Tensor],
    pred_sim_matrix: Optional[torch.Tensor],
    min_weight: float = 0.05,
) -> Optional[torch.Tensor]:
    """Return negative denominator weights from predicate similarities.

    A semantically close negative receives a smaller denominator weight instead
    of being treated as an absolute false negative. Invalid ids fall back to
    weight 1.0, preserving standard InfoNCE behavior.
    """
    if anchor_pred_ids is None or negative_pred_ids is None or pred_sim_matrix is None:
        return None
    if pred_sim_matrix.numel() == 0 or anchor_pred_ids.numel() == 0 or negative_pred_ids.numel() == 0:
        return None

    matrix = pred_sim_matrix.to(device=anchor_pred_ids.device, dtype=torch.float32)
    n_pred = int(matrix.shape[0])
    anchor = anchor_pred_ids.view(-1).to(device=matrix.device, dtype=torch.long)
    neg = negative_pred_ids.view(-1).to(device=matrix.device, dtype=torch.long)

    valid_anchor = (anchor >= 0) & (anchor < n_pred)
    valid_neg = (neg >= 0) & (neg < n_pred)
    safe_anchor = anchor.clamp(0, max(0, n_pred - 1))
    safe_neg = neg.clamp(0, max(0, n_pred - 1))

    sim = matrix.index_select(0, safe_anchor).index_select(1, safe_neg)
    valid = valid_anchor.unsqueeze(1) & valid_neg.unsqueeze(0)
    weights = (1.0 - sim).clamp(min=float(min_weight), max=1.0)
    return torch.where(valid, weights, torch.ones_like(weights))