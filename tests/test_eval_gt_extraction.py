"""Regression tests for the GT-triplet extraction index-alignment bug.

See docs/GT_EXTRACTION_BUG_TRIAGE.md for the full root-cause analysis. In
short: eval_sgg_standard's per-image prep loop overwrites ex_eval["pairs"]
with an eval-time candidate set (openvocab_rel/evals.py, per-image prep
loop) while leaving "rel_preds"/"rel_pos_mask" in their original,
differently-ordered form. _collect_gt_triplets zips them positionally, so
once the fix's snapshot ("_gt_pairs"/"_gt_preds"/"_gt_pos_mask") is missing
or bypassed, GT gets attributed to the wrong object pair.

All tests build ex dicts using the real _build_relation_entries (the actual
GT producer) and simulate eval_sgg_standard's real per-image prep-loop
overwrite (the actual eval_sgg_use_gt_pairs=False / True branching), rather
than hand-rolling synthetic pairs/preds -- this exercises the same code
paths this repo's evaluation function does, not a reimplementation of it.
"""
import torch

from openvocab_rel.datasets.vg150_loader import _build_relation_entries
from openvocab_rel.evals import (
    _build_all_ordered_pairs,
    _extract_gt_pairs,
    _collect_gt_triplets,
)


def _make_ex(obj_boxes, obj_names, relationships, pred_to_idx, use_all_pairs=True, negative_pair_ratio=-1.0):
    pairs, payload = _build_relation_entries(
        obj_boxes_t=obj_boxes,
        obj_names=obj_names,
        relationships=relationships,
        pred_to_idx=pred_to_idx,
        use_all_pairs=use_all_pairs,
        max_pairs=0,
        negative_pair_ratio=negative_pair_ratio,
    )
    return {
        "obj_boxes": obj_boxes,
        "obj_labels": obj_names,
        "pairs": [(p[0], p[1]) for p in pairs],
        "rel_preds": payload["rel_preds"],
        "rel_pos_mask": payload["rel_pos_mask"],
    }


def _simulate_eval_prep(ex, use_gt_pairs):
    """Mirrors eval_sgg_standard's per-image prep loop exactly (evals.py,
    the `for ex in batch:` block): computes the eval-time candidate pair
    list, snapshots the pre-overwrite GT structures, then overwrites
    "pairs". This is the real overwrite logic the fix touches, not a
    simplified stand-in for it.
    """
    boxes = ex.get("obj_boxes")
    n_obj = int(boxes.shape[0])
    pair_list = _extract_gt_pairs(ex) if use_gt_pairs else _build_all_ordered_pairs(n_obj)
    ex_eval = dict(ex)
    ex_eval["_gt_pairs"] = ex.get("pairs", [])
    ex_eval["_gt_preds"] = ex.get("rel_preds", [])
    ex_eval["_gt_pos_mask"] = ex.get("rel_pos_mask", None)
    ex_eval["pairs"] = pair_list
    return ex_eval


def _gt_triplet_set(gt):
    """Semantic GT triplet set: (subj_label, predicate, obj_label). Deliberately
    NOT index-based and NOT shape/length-based -- this is what "the model saw
    the right ground truth" actually means, independent of internal object
    ordering or tensor shapes.
    """
    return set(zip(gt["subj_labels"], gt["pred_labels"], gt["obj_labels"]))


PRED_TO_IDX = {"wearing": 0, "near": 1, "sitting on": 2, "relation": 3}


def _three_obj_two_rel():
    obj_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]],
        dtype=torch.float32,
    )
    obj_names = ["man", "shirt", "dog"]
    relationships = [
        {"subject_id": 0, "object_id": 1, "predicate": "wearing"},
        {"subject_id": 1, "object_id": 2, "predicate": "near"},
    ]
    return obj_boxes, obj_names, relationships


def test_multi_relation_sparse_gt_recovered_exactly():
    """The triage report's minimal counterexample: 3 objects, 2 relationships
    sharing an endpoint. Fails on the pre-fix code (would recover
    {(man,wearing,shirt), (man,near,dog)} -- the second relation misattributed
    to pair (0,2), which has no real annotation).
    """
    obj_boxes, obj_names, relationships = _three_obj_two_rel()
    ex = _make_ex(obj_boxes, obj_names, relationships, PRED_TO_IDX)
    ex_eval = _simulate_eval_prep(ex, use_gt_pairs=False)

    gt = _collect_gt_triplets(ex_eval, torch.device("cpu"))

    expected = {("man", "wearing", "shirt"), ("shirt", "near", "dog")}
    assert _gt_triplet_set(gt) == expected


def test_single_relation_not_at_row_major_first_position():
    """A lone relationship deliberately NOT at row-major position (0,1) --
    proves the fix doesn't rely on a lucky first-index coincidence.
    """
    obj_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]],
        dtype=torch.float32,
    )
    obj_names = ["man", "shirt", "dog"]
    relationships = [{"subject_id": 1, "object_id": 2, "predicate": "near"}]
    ex = _make_ex(obj_boxes, obj_names, relationships, PRED_TO_IDX)
    ex_eval = _simulate_eval_prep(ex, use_gt_pairs=False)

    gt = _collect_gt_triplets(ex_eval, torch.device("cpu"))

    assert _gt_triplet_set(gt) == {("shirt", "near", "dog")}


def test_empty_relation_image_yields_empty_gt():
    obj_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]],
        dtype=torch.float32,
    )
    obj_names = ["man", "shirt", "dog"]
    ex = _make_ex(obj_boxes, obj_names, [], PRED_TO_IDX)
    ex_eval = _simulate_eval_prep(ex, use_gt_pairs=False)

    gt = _collect_gt_triplets(ex_eval, torch.device("cpu"))

    assert _gt_triplet_set(gt) == set()
    assert len(gt["pred_labels"]) == 0


def test_dense_multi_object_sparse_relations():
    """6 objects, only 3 relationships, several objects with zero incident
    relations -- proves the fix neither invents GT for unrelated pairs nor
    drops legitimate GT once negatives vastly outnumber positives.
    """
    obj_boxes = torch.tensor(
        [[float(i) * 10.0, 0.0, float(i) * 10.0 + 5.0, 5.0] for i in range(6)],
        dtype=torch.float32,
    )
    obj_names = ["a", "b", "c", "d", "e", "f"]
    relationships = [
        {"subject_id": 0, "object_id": 3, "predicate": "wearing"},
        {"subject_id": 2, "object_id": 5, "predicate": "near"},
        {"subject_id": 4, "object_id": 1, "predicate": "sitting on"},
    ]
    ex = _make_ex(obj_boxes, obj_names, relationships, PRED_TO_IDX)
    ex_eval = _simulate_eval_prep(ex, use_gt_pairs=False)

    gt = _collect_gt_triplets(ex_eval, torch.device("cpu"))

    expected = {("a", "wearing", "d"), ("c", "near", "f"), ("e", "sitting on", "b")}
    assert _gt_triplet_set(gt) == expected


def test_use_gt_pairs_true_still_correct():
    """eval_sgg_use_gt_pairs=True must remain unaffected by this fix -- its
    semantics (and the ordering invariant _extract_gt_pairs already
    preserves) are unchanged.
    """
    obj_boxes, obj_names, relationships = _three_obj_two_rel()
    ex = _make_ex(obj_boxes, obj_names, relationships, PRED_TO_IDX)
    ex_eval = _simulate_eval_prep(ex, use_gt_pairs=True)

    gt = _collect_gt_triplets(ex_eval, torch.device("cpu"))

    expected = {("man", "wearing", "shirt"), ("shirt", "near", "dog")}
    assert _gt_triplet_set(gt) == expected


def test_use_gt_pairs_true_and_false_yield_identical_gt():
    """The core invariant this bug violated: for identical annotations, the
    recovered GT triplet set must not depend on eval_sgg_use_gt_pairs --
    only the *candidate* pair set should differ between the two modes.
    """
    obj_boxes, obj_names, relationships = _three_obj_two_rel()
    ex = _make_ex(obj_boxes, obj_names, relationships, PRED_TO_IDX)

    ex_eval_false = _simulate_eval_prep(dict(ex), use_gt_pairs=False)
    ex_eval_true = _simulate_eval_prep(dict(ex), use_gt_pairs=True)

    gt_false = _collect_gt_triplets(ex_eval_false, torch.device("cpu"))
    gt_true = _collect_gt_triplets(ex_eval_true, torch.device("cpu"))

    assert _gt_triplet_set(gt_false) == _gt_triplet_set(gt_true)


def test_shared_endpoint_different_predicates_both_recovered():
    """Two GT relationships with different (subj,obj) pairs but a shared
    endpoint object -- edge case for positional-alignment bugs specifically,
    since shared endpoints are where index-based mistakes are most likely
    to silently produce a plausible-looking (but wrong) pair.
    """
    obj_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0], [60.0, 60.0, 70.0, 70.0]],
        dtype=torch.float32,
    )
    obj_names = ["man", "shirt", "hat", "dog"]
    relationships = [
        {"subject_id": 0, "object_id": 1, "predicate": "wearing"},
        {"subject_id": 0, "object_id": 2, "predicate": "wearing"},
        {"subject_id": 0, "object_id": 3, "predicate": "near"},
    ]
    ex = _make_ex(obj_boxes, obj_names, relationships, PRED_TO_IDX)
    ex_eval = _simulate_eval_prep(ex, use_gt_pairs=False)

    gt = _collect_gt_triplets(ex_eval, torch.device("cpu"))

    expected = {("man", "wearing", "shirt"), ("man", "wearing", "hat"), ("man", "near", "dog")}
    assert _gt_triplet_set(gt) == expected
