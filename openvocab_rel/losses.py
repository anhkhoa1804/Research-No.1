from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .ddp_utils import ddp_all_gather_tensor, ddp_rank_offset
from .lattice import lattice_negative_weights


def _zero_with_grad(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


class RingQueue:
    def __init__(self, dim: int, size: int, device: torch.device):
        self.size = int(size)
        self.buf = torch.zeros((self.size, dim), device=device)
        self.pred_ids = torch.full((self.size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def add(self, y: torch.Tensor, pred_ids: Optional[torch.Tensor] = None) -> None:
        y = F.normalize(y, dim=-1)
        n = int(y.shape[0])
        if n == 0:
            return
        ids = None
        if pred_ids is not None and int(pred_ids.numel()) == n:
            ids = pred_ids.to(device=y.device, dtype=torch.long).view(-1)
        else:
            ids = torch.full((n,), -1, dtype=torch.long, device=y.device)
        if n >= self.size:
            self.buf[:] = y[-self.size :]
            self.pred_ids[:] = ids[-self.size :]
            self.ptr = 0
            self.full = True
            return
        end = self.ptr + n
        if end <= self.size:
            self.buf[self.ptr : end] = y
            self.pred_ids[self.ptr : end] = ids
        else:
            first = self.size - self.ptr
            self.buf[self.ptr :] = y[:first]
            self.buf[: end - self.size] = y[first:]
            self.pred_ids[self.ptr :] = ids[:first]
            self.pred_ids[: end - self.size] = ids[first:]
        self.ptr = end % self.size
        if self.ptr == 0:
            self.full = True

    def get(self) -> torch.Tensor:
        return self.buf if self.full else self.buf[: self.ptr]

    def get_pred_ids(self) -> torch.Tensor:
        return self.pred_ids if self.full else self.pred_ids[: self.ptr]


def info_nce_inbatch(x: torch.Tensor, y: torch.Tensor, temp: float) -> torch.Tensor:
    temp = max(float(temp), 1e-6)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = (x @ y.t()) / temp
    labels = torch.arange(x.shape[0], device=x.device)
    return F.cross_entropy(logits, labels)


def info_nce_inbatch_global(x: torch.Tensor, y: torch.Tensor, temp: float) -> torch.Tensor:
    temp = max(float(temp), 1e-6)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    y_all = ddp_all_gather_tensor(y)
    logits = (x @ y_all.t()) / temp
    offset = ddp_rank_offset(int(x.shape[0]))
    labels = torch.arange(x.shape[0], device=x.device) + int(offset)
    return F.cross_entropy(logits, labels)


def info_nce_with_queue(
    x: torch.Tensor,
    y: torch.Tensor,
    queue: RingQueue,
    temp: float,
    sim_threshold: float = 0.95,
    anchor_pred_ids: Optional[torch.Tensor] = None,
    pred_sim_matrix: Optional[torch.Tensor] = None,
    lattice_min_weight: float = 0.05,
    min_queue_negatives: int = 1024,
) -> torch.Tensor:
    """Queue InfoNCE with optional lattice-aware soft negatives.

    Standard InfoNCE is preserved by default. When predicate ids and a
    predicate-similarity lattice are provided, semantically close queue
    negatives receive smaller denominator weights instead of being masked or
    treated as definite false negatives.
    """
    temp = max(float(temp), 1e-6)

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    neg = queue.get()

    if neg.numel() == 0 or int(neg.shape[0]) < int(min_queue_negatives):
        return info_nce_inbatch(x, y, temp)

    neg = torch.nan_to_num(neg, nan=0.0)

    pos = torch.sum(x * y, dim=-1, keepdim=True) / temp
    neg_logits = (x @ neg.t()) / temp

    with torch.no_grad():
        neg_norm = F.normalize(neg, dim=-1)
        text_sim = y @ neg_norm.t()
        synonym_mask = text_sim > sim_threshold
        lattice_weights = lattice_negative_weights(
            anchor_pred_ids,
            queue.get_pred_ids(),
            pred_sim_matrix,
            min_weight=float(lattice_min_weight),
        )

    if lattice_weights is None:
        neg_logits = neg_logits.masked_fill(synonym_mask, -10000.0)
    else:
        lattice_weights = lattice_weights.to(device=neg_logits.device, dtype=neg_logits.dtype)
        lattice_weights = lattice_weights.masked_fill(synonym_mask, float(lattice_min_weight))
        neg_logits = neg_logits + torch.log(lattice_weights.clamp_min(1e-6))

    neg_logits = neg_logits.clamp(min=-10000.0, max=10000.0)

    logits = torch.cat([pos, neg_logits], dim=1)
    labels = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)

    return F.cross_entropy(logits, labels)

def info_nce_pos_with_negs(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_negs: torch.Tensor,
    temp: float,
    sim_threshold: float = 0.95,
    neg_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    temp = max(float(temp), 1e-6)
    if y_negs.numel() == 0:
        return x.sum() * 0.0

    x = F.normalize(x, dim=-1)
    y_pos = F.normalize(y_pos, dim=-1)

    if neg_valid_mask is None:
        neg_valid_mask = torch.linalg.vector_norm(y_negs.float(), dim=-1) > 1.0e-6
    else:
        neg_valid_mask = neg_valid_mask.to(device=y_negs.device, dtype=torch.bool)
    if neg_valid_mask.shape != y_negs.shape[:2]:
        return x.sum() * 0.0

    y_negs = F.normalize(y_negs, dim=-1)
    pos = torch.sum(x * y_pos, dim=-1, keepdim=True)
    neg = torch.einsum("nd,nkd->nk", x, y_negs)

    with torch.no_grad():
        text_sim = torch.einsum("nd,nkd->nk", y_pos, y_negs)
        synonym_mask = text_sim > sim_threshold
        valid = neg_valid_mask & (~synonym_mask)

    valid_any = valid.any(dim=1)
    if not bool(valid_any.any()):
        return x.sum() * 0.0

    neg = neg.masked_fill(~valid, -10000.0)
    logits = torch.cat([pos, neg], dim=1) / temp
    labels = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss[valid_any].mean()

def visual_counterfactual_inbatch_loss(
    anchor_x: torch.Tensor,
    anchor_y: torch.Tensor,
    anchor_pair_hashes: torch.Tensor,
    anchor_pred_ids: torch.Tensor,
    pool_r: torch.Tensor,
    pool_pair_hashes: torch.Tensor,
    pool_pred_ids: torch.Tensor,
    temp: float,
    k_vis: int,
) -> Tuple[torch.Tensor, int, int]:
    if anchor_x.numel() == 0 or pool_r.numel() == 0:
        return _zero_with_grad(anchor_x), 0, 0
    if int(anchor_pair_hashes.shape[0]) != int(anchor_x.shape[0]) or int(pool_pair_hashes.shape[0]) != int(pool_r.shape[0]):
        return _zero_with_grad(anchor_x), 0, 0
    if int(anchor_pred_ids.shape[0]) != int(anchor_x.shape[0]) or int(pool_pred_ids.shape[0]) != int(pool_r.shape[0]):
        return _zero_with_grad(anchor_x), 0, 0

    x_n = F.normalize(anchor_x, dim=-1)
    y_n = F.normalize(anchor_y, dim=-1)
    r_n = F.normalize(pool_r, dim=-1)
    anchor_pair_hashes = anchor_pair_hashes.view(-1).to(device=x_n.device, dtype=torch.long)
    anchor_pred_ids = anchor_pred_ids.view(-1).to(device=x_n.device, dtype=torch.long)
    pool_pair_hashes = pool_pair_hashes.view(-1).to(device=x_n.device, dtype=torch.long)
    pool_pred_ids = pool_pred_ids.view(-1).to(device=x_n.device, dtype=torch.long)

    k_take = max(1, int(k_vis))

    cand_mask = (pool_pair_hashes.unsqueeze(0) == anchor_pair_hashes.unsqueeze(1)) & (
        pool_pred_ids.unsqueeze(0) != anchor_pred_ids.unsqueeze(1)
    )
    if not bool(cand_mask.any()):
        return _zero_with_grad(anchor_x), 0, 0

    sim = x_n @ r_n.t()
    sim = sim.masked_fill(~cand_mask, -10000.0)
    take = min(k_take, int(pool_r.shape[0]))
    top_vals, top_idx = torch.topk(sim, k=take, dim=1, largest=True)
    valid = top_vals > -9000.0
    valid_any = valid.any(dim=1)
    if not bool(valid_any.any()):
        return _zero_with_grad(anchor_x), 0, 0

    gathered = r_n.index_select(0, top_idx.reshape(-1)).view(int(x_n.shape[0]), take, -1)
    neg_scores = torch.sum(gathered * y_n.unsqueeze(1), dim=-1)
    pos_scores = torch.sum(x_n * y_n, dim=-1, keepdim=True).expand(-1, take)

    logits = torch.stack([pos_scores, neg_scores], dim=-1) / float(temp)
    labels = torch.zeros((int(x_n.shape[0]) * take,), dtype=torch.long, device=logits.device)
    loss_per_pair = F.cross_entropy(logits.reshape(-1, 2), labels, reduction="none").view(int(x_n.shape[0]), take)
    per_anchor = (loss_per_pair * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1.0)
    mined = int(valid.sum().item())
    used = int(valid_any.sum().item())
    return per_anchor[valid_any].mean(), mined, used


def counterfactual_hard_negative_loss(
    anchor_x: torch.Tensor,
    anchor_y: torch.Tensor,
    role_swap_x: torch.Tensor,
    attr_swap_y: torch.Tensor,
    role_swap_y: torch.Tensor,
    predicate_confusable_y: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    """Compose role-reversal and attribute-swapping hard negatives into one InfoNCE objective."""
    neg_banks: List[torch.Tensor] = []
    valid_banks: List[torch.Tensor] = []

    n_anchor = int(anchor_x.shape[0])

    def _append_bank(bank: Optional[torch.Tensor]) -> None:
        if bank is None or bank.numel() == 0:
            return
        if int(bank.shape[0]) != n_anchor:
            return
        valid = torch.linalg.vector_norm(bank.float(), dim=-1) > 1.0e-6
        if not bool(valid.any()):
            return
        neg_banks.append(bank.unsqueeze(1))
        valid_banks.append(valid.unsqueeze(1))

    _append_bank(role_swap_x)
    _append_bank(attr_swap_y)
    _append_bank(role_swap_y)
    _append_bank(predicate_confusable_y)

    if len(neg_banks) == 0:
        return anchor_x.sum() * 0.0
    y_negs = torch.cat(neg_banks, dim=1)
    neg_valid_mask = torch.cat(valid_banks, dim=1)
    return info_nce_pos_with_negs(anchor_x, anchor_y, y_negs, temp, neg_valid_mask=neg_valid_mask)

