from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TripletRecord:
    image_id: str
    subject: str
    predicate: str
    object: str
    score: float
    subject_box: Optional[List[float]] = None
    object_box: Optional[List[float]] = None
    embedding: Optional[torch.Tensor] = None

    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}".strip()

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "image_id": self.image_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "score": float(self.score),
            "subject_box": self.subject_box,
            "object_box": self.object_box,
        }
        if include_embedding and self.embedding is not None:
            out["embedding"] = self.embedding.detach().cpu().tolist()
        return out


class TripletRetrievalIndex:
    """Small in-memory retrieval index over predicted scene-graph triplets."""

    def __init__(self) -> None:
        self.records: List[TripletRecord] = []

    def add(self, record: TripletRecord) -> None:
        self.records.append(record)

    def extend(self, records: Iterable[TripletRecord]) -> None:
        for record in records:
            self.add(record)

    def __len__(self) -> int:
        return len(self.records)

    def search(
        self,
        query_embedding: torch.Tensor,
        topk: int = 10,
        score_weight: float = 0.15,
    ) -> List[Dict[str, Any]]:
        if len(self.records) == 0:
            return []
        emb_records = [record for record in self.records if record.embedding is not None]
        if len(emb_records) == 0:
            return []
        matrix = torch.stack([record.embedding.detach().float() for record in emb_records], dim=0)
        query = query_embedding.detach().float().view(1, -1)
        sim = F.normalize(query, dim=-1) @ F.normalize(matrix, dim=-1).t()
        base = torch.tensor([float(record.score) for record in emb_records], dtype=torch.float32, device=sim.device)
        final = sim.squeeze(0) + float(score_weight) * base.to(device=sim.device)
        k = min(max(1, int(topk)), int(final.numel()))
        values, indices = torch.topk(final, k=k, largest=True)
        return [
            {**emb_records[int(idx)].to_dict(include_embedding=False), "retrieval_score": float(value)}
            for value, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist())
        ]


def build_triplet_records(
    image_id: str,
    pairs: Sequence[Tuple[int, int]],
    object_labels: Sequence[str],
    predicate_labels: Sequence[str],
    predicate_indices: torch.Tensor,
    predicate_scores: torch.Tensor,
    boxes: Optional[torch.Tensor] = None,
    relation_embeddings: Optional[torch.Tensor] = None,
    topk: int = 100,
) -> List[TripletRecord]:
    if len(pairs) == 0:
        return []
    scores = predicate_scores.detach().float().view(-1)
    keep_k = min(max(1, int(topk)), int(scores.numel()))
    keep = torch.topk(scores, k=keep_k, largest=True).indices.detach().cpu().tolist()
    box_list = boxes.detach().cpu().float().tolist() if isinstance(boxes, torch.Tensor) else None
    records: List[TripletRecord] = []
    for idx in keep:
        pair = pairs[int(idx)]
        subj_idx, obj_idx = int(pair[0]), int(pair[1])
        pred_idx = int(predicate_indices[int(idx)].detach().cpu().item())
        if not (0 <= subj_idx < len(object_labels) and 0 <= obj_idx < len(object_labels) and 0 <= pred_idx < len(predicate_labels)):
            continue
        records.append(
            TripletRecord(
                image_id=str(image_id),
                subject=str(object_labels[subj_idx]),
                predicate=str(predicate_labels[pred_idx]),
                object=str(object_labels[obj_idx]),
                score=float(scores[int(idx)].item()),
                subject_box=box_list[subj_idx] if box_list is not None and subj_idx < len(box_list) else None,
                object_box=box_list[obj_idx] if box_list is not None and obj_idx < len(box_list) else None,
                embedding=relation_embeddings[int(idx)].detach().cpu() if isinstance(relation_embeddings, torch.Tensor) and int(idx) < int(relation_embeddings.shape[0]) else None,
            )
        )
    return records