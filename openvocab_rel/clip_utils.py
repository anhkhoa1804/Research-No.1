"""CLIP helpers for model loading and text feature extraction."""

from __future__ import annotations
import os
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, logging
import transformers.utils.logging as tf_logging

from .ddp_utils import unwrap_ddp

# Silence transformers and hub warnings globally
logging.set_verbosity_error()
tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@torch.no_grad()
def clip_text_features(
    clip_model, 
    processor: "CLIPProcessor", 
    texts: List[str], 
    device: torch.device,
    batch_size: int = 64
) -> torch.Tensor:
    """
    Extracts normalized text features from CLIP.
    Processes texts in mini-batches to prevent CUDA Out-Of-Memory (OOM) errors 
    when encoding large predicate vocabularies on limited VRAM GPUs (e.g., T4).
    """
    def _extract_text_tensor(out_obj: object) -> torch.Tensor:
        if isinstance(out_obj, torch.Tensor):
            return out_obj
        text_embeds = getattr(out_obj, "text_embeds", None)
        if isinstance(text_embeds, torch.Tensor):
            return text_embeds
        pooler_output = getattr(out_obj, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        if isinstance(out_obj, (tuple, list)) and len(out_obj) > 0 and isinstance(out_obj[0], torch.Tensor):
            return out_obj[0]
        if isinstance(out_obj, dict):
            for key in ("text_embeds", "pooler_output", "last_hidden_state"):
                value = out_obj.get(key, None)
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(f"Unsupported CLIP text output type: {type(out_obj)}")

    cm = unwrap_ddp(clip_model)
    was_training = bool(cm.training)
    cm.eval()
    all_features = []

    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = cm.get_text_features(**inputs)
            out_tensor = _extract_text_tensor(out)
            all_features.append(out_tensor.detach())
            del inputs, out
    finally:
        cm.train(was_training)

    # Concatenate all accumulated batches along the batch dimension
    all_features_tensor = torch.cat(all_features, dim=0).to(device, non_blocking=True)

    # Return L2-normalized features
    return F.normalize(all_features_tensor, dim=-1)


def configure_clip(clip_name: str, device: torch.device) -> Tuple[CLIPModel, CLIPProcessor, int, int]:
    """Initializes CLIP model and processor (silent) and returns interface dims.

    Required contract (train.py): returns a 4-tuple
        (model, processor, vision_dim, text_dim)
    """
    # Force silence again during call (Kaggle log overflow protection)
    logging.set_verbosity_error()
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    model = CLIPModel.from_pretrained(str(clip_name))
    processor = CLIPProcessor.from_pretrained(str(clip_name))
    model.to(device)

    cfg = getattr(model, "config", None)
    vision_dim = int(getattr(getattr(cfg, "vision_config", object()), "hidden_size", 0) or 0)
    text_dim = int(getattr(getattr(cfg, "text_config", object()), "hidden_size", 0) or 0)
    if vision_dim <= 0:
        vision_dim = int(getattr(getattr(model, "vision_model", object()), "config", object()).hidden_size)  # type: ignore[attr-defined]
    if text_dim <= 0:
        text_dim = int(getattr(getattr(model, "text_model", object()), "config", object()).hidden_size)  # type: ignore[attr-defined]

    return model, processor, int(vision_dim), int(text_dim)

def clip_text_is_trainable(clip_model: nn.Module) -> bool:
    """Return whether the CLIP text encoder currently has trainable parameters."""
    cm = unwrap_ddp(clip_model)
    tm = getattr(cm, "text_model", None)
    if isinstance(tm, nn.Module):
        return any(bool(p.requires_grad) for p in tm.parameters())
    return any(bool(p.requires_grad) for p in cm.parameters())

def build_confusable_index(text_features: torch.Tensor, topm: int = 32, sim_threshold: float = 0.95) -> torch.Tensor:
    if text_features is None:
        return torch.empty((0, 0), dtype=torch.long)
    if text_features.numel() == 0:
        return torch.empty((0, 0), dtype=torch.long, device=text_features.device)

    feats = F.normalize(text_features.float(), dim=-1)
    sim = feats @ feats.T  
    n = int(sim.shape[0])
    k = int(max(0, min(int(topm), n - 1)))
    if k <= 0:
        return torch.empty((n, 0), dtype=torch.long, device=sim.device)

    sim = sim.clone()
    sim.fill_diagonal_(-10000.0)
    sim[sim > sim_threshold] = -10000.0
    _, idx = torch.topk(sim, k=k, dim=-1, largest=True, sorted=True)
    return idx.to(dtype=torch.long)


@torch.no_grad()
def encode_predicate_vocab(
    clip_model: nn.Module,
    processor: "CLIPProcessor",
    pred_vocab: List[str],
    device: torch.device,
    prompt_fn: Optional[Callable[..., str]] = None,
    direction: str = "s2o",
) -> Tuple[List[str], torch.Tensor]:
    """Encode predicate vocab into CLIP text embedding space.

    train.py expects: (texts, text_features) where text_features is [N, D].
    """
    if prompt_fn is None:
        def _default_prompt(p: str, direction: str = "s2o") -> str:
            _ = direction
            return str(p)

        prompt_fn = lambda p, direction=direction: _default_prompt(p, direction)  # type: ignore[assignment]

    texts: List[str] = []
    for p in pred_vocab:
        try:
            t = prompt_fn(p, direction=direction)  # type: ignore[misc]
        except TypeError:
            t = prompt_fn(p)  # type: ignore[misc]
        texts.append(str(t))
    feats = clip_text_features(clip_model, processor, texts, device)
    return texts, feats
