"""RelationalModel forward-pass smoke test on synthetic tensors.

The constructor only needs (clip_vision_dim, text_dim) as integers -- no
actual CLIP model is loaded or needed anywhere in this file.
"""
import torch

from openvocab_rel.config import TrainConfig
from openvocab_rel.models.relational_model import RelationalModel


def _tiny_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.emb_dim = 32  # divisible by the fixed 8-head attention layers
    cfg.clip_input_res = 64
    cfg.progressive_node_layers = 1
    cfg.progressive_edge_layers = 1
    cfg.progressive_bilinear_layers = 0
    cfg.deformable_num_points = 4
    cfg.relation_context_layers = 0
    cfg.gradient_checkpointing = False
    cfg.learned_prune_k = 0  # no pruning, so pair counts below are exact
    return cfg


def test_forward_from_featmap_shapes():
    cfg = _tiny_cfg()
    model = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
    model.eval()

    batch_size, ch, h, w = 2, 48, 8, 8
    feat_map = torch.randn(batch_size, ch, h, w)
    obj_boxes = [
        torch.tensor([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 40.0, 40.0]], dtype=torch.float32),
        torch.tensor([[5.0, 5.0, 25.0, 25.0]], dtype=torch.float32),
    ]
    pairs = [[(0, 1), (1, 0)], []]

    with torch.no_grad():
        regs, rel_feats, rel_swaps, _assigns, gates, _kept_idx = model.forward_from_featmap(
            feat_map, obj_boxes=obj_boxes, pairs=pairs
        )

    assert len(regs) == batch_size
    assert regs[0].shape == (2, cfg.emb_dim)
    assert regs[1].shape == (1, cfg.emb_dim)
    assert rel_feats[0].shape == (2, cfg.emb_dim)
    assert rel_feats[1].shape == (0, cfg.emb_dim)  # empty image-2 pair list
    assert rel_swaps[0].shape == rel_feats[0].shape
    assert gates[0].shape == (2,)
    assert torch.isfinite(rel_feats[0]).all()


def test_predicate_logits_shape():
    cfg = _tiny_cfg()
    model = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
    rel_feats = torch.randn(5, cfg.emb_dim)
    logits = model.predicate_logits(rel_feats)
    assert logits.shape == (5, cfg.predicate_classifier_classes)
    assert torch.isfinite(logits).all()


def test_relationness_scores_are_valid_probabilities():
    cfg = _tiny_cfg()
    model = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
    rel_feats = torch.randn(4, cfg.emb_dim)
    scores = model.relationness_scores(rel_feats)
    assert scores.shape == (4,)
    assert bool((scores >= 0).all())
    assert bool((scores <= 1).all())


def test_relationness_scores_empty_input():
    cfg = _tiny_cfg()
    model = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
    rel_feats = torch.zeros((0, cfg.emb_dim))
    scores = model.relationness_scores(rel_feats)
    assert scores.shape == (0,)
