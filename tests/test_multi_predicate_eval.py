"""Regression tests for the additive multi-predicate-per-pair ranking.

Background (docs/known_issues.md, P1 "R@K is top-1 predicate accuracy"):
``eval_sgg_standard`` emits exactly one predicate per candidate pair via
``rel_probs.max(-1)``. Under ``eval_sgg_use_gt_pairs=true`` the candidate pool
is ~12 pairs/image, far below K=20, so every candidate is always inside top-K
-- R@20 == R@50 == R@100 identically, K is inert, and the statistic being
reported is top-1 predicate accuracy rather than the recall@K published VG150
papers report (they rank all (pair x predicate) hypotheses).

``_make_multi_triplet_predictions`` builds that second hypothesis set. These
tests pin the two properties that make it safe and correct:

  1. it is ADDITIVE -- the single-predicate path is unchanged, so every
     historical number stays comparable;
  2. it is a genuine superset -- it can only ever find more GT, never less,
     and it actually makes K non-inert.

No model, no CLIP, no GPU, no real data.
"""
import torch

from openvocab_rel.config import TrainConfig
from openvocab_rel.evals import (
    _compute_pred_matches,
    _get_hit_gt_indices,
    _make_multi_triplet_predictions,
    _make_triplet_predictions,
    _recall_from_matches,
)

PRED_VOCAB = ["on", "has", "wearing", "near", "holding", "relation"]
KS = [20, 50, 100]


def _ex(n_obj=3):
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 0.0, 50.0, 10.0]][:n_obj],
        dtype=torch.float32,
    )
    return {"obj_boxes": boxes, "_pred_vocab": PRED_VOCAB, "_object_score_power": 0.0}


def _gt(subj, obj, preds, boxes):
    """GT in the shape _compute_pred_matches expects."""
    return {
        "subj_labels": [f"o{s}" for s in subj],
        "pred_labels": list(preds),
        "obj_labels": [f"o{o}" for o in obj],
        "subj_boxes": torch.stack([boxes[s] for s in subj]) if subj else torch.empty((0, 4)),
        "obj_boxes": torch.stack([boxes[o] for o in obj]) if obj else torch.empty((0, 4)),
        "subj_idx": list(subj),
        "obj_idx": list(obj),
    }


def _labels(n):
    return [f"o{i}" for i in range(n)]


def test_config_flag_exists_and_defaults_on():
    cfg = TrainConfig()
    assert hasattr(cfg, "eval_sgg_multi_predicate_topk")
    assert int(cfg.eval_sgg_multi_predicate_topk) > 0, (
        "the literature-comparable metric should be reported by default; it is "
        "purely additive and cannot change the existing fields"
    )


def test_emits_topk_hypotheses_per_pair():
    ex = _ex()
    pairs = [(0, 1), (1, 2)]
    rel_probs = torch.tensor(
        [[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
         [0.1, 0.6, 0.1, 0.1, 0.05, 0.05]],
        dtype=torch.float32,
    )
    pair_scores = torch.ones(2)
    out = _make_multi_triplet_predictions(
        ex, pairs, pair_scores, rel_probs, _labels(3), torch.empty((0,)), torch.device("cpu"), topk=3
    )
    assert len(out["pred_labels"]) == 2 * 3
    assert int(out["subj_boxes"].shape[0]) == 6
    assert int(out["scores"].numel()) == 6


def test_topk_is_clamped_to_vocabulary_size():
    ex = _ex()
    pairs = [(0, 1)]
    rel_probs = torch.full((1, len(PRED_VOCAB)), 1.0 / len(PRED_VOCAB))
    out = _make_multi_triplet_predictions(
        ex, pairs, torch.ones(1), rel_probs, _labels(3), torch.empty((0,)), torch.device("cpu"), topk=999
    )
    assert len(out["pred_labels"]) == len(PRED_VOCAB)


def test_topk_zero_or_empty_input_yields_no_predictions():
    ex = _ex()
    dev = torch.device("cpu")
    empty = _make_multi_triplet_predictions(
        ex, [(0, 1)], torch.ones(1), torch.rand(1, 6), _labels(3), torch.empty((0,)), dev, topk=0
    )
    assert len(empty["pred_labels"]) == 0
    assert int(empty["scores"].numel()) == 0

    no_pairs = _make_multi_triplet_predictions(
        ex, [], torch.empty((0,)), torch.empty((0, 6)), _labels(3), torch.empty((0,)), dev, topk=5
    )
    assert len(no_pairs["pred_labels"]) == 0


def test_rank_one_hypothesis_matches_the_single_predicate_path():
    """The top-1 slice must reproduce _make_triplet_predictions exactly.

    This is what guarantees the change is additive: the new path is a strict
    extension of the old one, not a different scoring rule.
    """
    ex = _ex()
    pairs = [(0, 1), (1, 2), (0, 2)]
    rel_probs = torch.tensor(
        [[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
         [0.1, 0.6, 0.1, 0.1, 0.05, 0.05],
         [0.1, 0.1, 0.7, 0.05, 0.03, 0.02]],
        dtype=torch.float32,
    )
    pair_scores = torch.tensor([0.9, 0.8, 0.7])
    pair_pred_scores, pair_pred_idx = rel_probs.max(dim=-1)
    dev = torch.device("cpu")

    single = _make_triplet_predictions(
        ex, pairs, pair_scores, pair_pred_idx, pair_pred_scores, _labels(3), torch.empty((0,)), dev
    )
    multi = _make_multi_triplet_predictions(
        ex, pairs, pair_scores, rel_probs, _labels(3), torch.empty((0,)), dev, topk=4
    )
    # rank-0 entry of each pair == the single-predicate prediction for that pair
    for i in range(len(pairs)):
        assert multi["pred_labels"][i * 4] == single["pred_labels"][i]
        assert abs(float(multi["scores"][i * 4]) - float(single["scores"][i])) < 1e-6


def test_multi_recall_is_never_worse_and_can_be_strictly_better():
    """A GT predicate ranked 2nd for its pair is unreachable single-predicate."""
    ex = _ex()
    boxes = ex["obj_boxes"]
    pairs = [(0, 1), (1, 2)]
    # For pair (0,1) the argmax is "on" but GT is "has" (rank 2).
    # For pair (1,2) the argmax is "wearing" and GT is "wearing" (rank 1).
    rel_probs = torch.tensor(
        [[0.5, 0.4, 0.05, 0.03, 0.01, 0.01],
         [0.1, 0.1, 0.7, 0.05, 0.03, 0.02]],
        dtype=torch.float32,
    )
    pair_scores = torch.ones(2)
    gt = _gt([0, 1], [1, 2], ["has", "wearing"], boxes)
    dev = torch.device("cpu")

    pair_pred_scores, pair_pred_idx = rel_probs.max(dim=-1)
    single = _make_triplet_predictions(
        ex, pairs, pair_scores, pair_pred_idx, pair_pred_scores, _labels(3), torch.empty((0,)), dev
    )
    multi = _make_multi_triplet_predictions(
        ex, pairs, pair_scores, rel_probs, _labels(3), torch.empty((0,)), dev, topk=3
    )

    r_single = _recall_from_matches(_compute_pred_matches(single, gt, iou_thresh=0.5), single["scores"], KS)
    r_multi = _recall_from_matches(_compute_pred_matches(multi, gt, iou_thresh=0.5), multi["scores"], KS)

    for k in KS:
        assert r_multi[k] >= r_single[k], f"multi-predicate recall regressed at K={k}"
    assert r_single[50] < 1.0, "fixture should leave one GT unreachable single-predicate"
    assert r_multi[50] > r_single[50], "second-ranked GT predicate should become reachable"


def test_k_becomes_non_inert():
    """With enough hypotheses, R@20 and R@100 can differ -- the whole point.

    Under the single-predicate path they are identically equal whenever the
    candidate pool is smaller than K, which is always true in GT-pairs mode.
    """
    ex = _ex()
    boxes = ex["obj_boxes"]
    pairs = [(0, 1), (1, 2)]
    # GT is the LAST-ranked predicate for both pairs, so it only enters the
    # top-K list once K is large enough.
    rel_probs = torch.tensor(
        [[0.40, 0.30, 0.15, 0.10, 0.04, 0.01],
         [0.40, 0.30, 0.15, 0.10, 0.04, 0.01]],
        dtype=torch.float32,
    )
    gt = _gt([0, 1], [1, 2], ["holding", "holding"], boxes)
    dev = torch.device("cpu")
    multi = _make_multi_triplet_predictions(
        ex, pairs, torch.ones(2), rel_probs, _labels(3), torch.empty((0,)), dev, topk=6
    )
    matches = _compute_pred_matches(multi, gt, iou_thresh=0.5)
    r_small = _recall_from_matches(matches, multi["scores"], [2])
    r_large = _recall_from_matches(matches, multi["scores"], [100])
    assert r_large[100] > r_small[2], "K should now change the result"


def test_hit_indices_align_with_gt_order():
    """mR@K accumulation indexes gt['pred_labels'] by hit position."""
    ex = _ex()
    boxes = ex["obj_boxes"]
    pairs = [(0, 1), (1, 2)]
    rel_probs = torch.tensor(
        [[0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
         [0.05, 0.9, 0.02, 0.01, 0.01, 0.01]],
        dtype=torch.float32,
    )
    gt = _gt([0, 1], [1, 2], ["on", "has"], boxes)
    dev = torch.device("cpu")
    multi = _make_multi_triplet_predictions(
        ex, pairs, torch.ones(2), rel_probs, _labels(3), torch.empty((0,)), dev, topk=2
    )
    matches = _compute_pred_matches(multi, gt, iou_thresh=0.5)
    hits = _get_hit_gt_indices(matches, multi["scores"], KS)
    assert int(hits[50].numel()) == len(gt["pred_labels"])
    assert bool(hits[50].any()), "both GT triplets are rank-1 predictions and should hit"


def test_scores_are_finite_and_ordered_within_a_pair():
    ex = _ex()
    pairs = [(0, 1)]
    rel_probs = torch.tensor([[0.5, 0.25, 0.15, 0.05, 0.03, 0.02]], dtype=torch.float32)
    out = _make_multi_triplet_predictions(
        ex, pairs, torch.ones(1), rel_probs, _labels(3), torch.empty((0,)), torch.device("cpu"), topk=4
    )
    s = out["scores"]
    assert bool(torch.isfinite(s).all())
    assert all(float(s[i]) >= float(s[i + 1]) for i in range(int(s.numel()) - 1))
