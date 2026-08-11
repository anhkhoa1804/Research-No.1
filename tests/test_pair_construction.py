"""Pair/negative construction smoke tests, synthetic in-memory data only."""
import torch

from openvocab_rel.datasets.vg150_loader import _build_relation_entries


def test_build_relation_entries_positives_and_sampled_negatives():
    obj_boxes = torch.tensor(
        [[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.5], [0.6, 0.6, 0.8, 0.8]],
        dtype=torch.float32,
    )
    obj_names = ["person", "chair", "dog"]
    relationships = [{"subject_id": 0, "object_id": 1, "predicate": "sitting on"}]
    pred_to_idx = {"sitting on": 0, "relation": 1}

    pairs, payload = _build_relation_entries(
        obj_boxes_t=obj_boxes,
        obj_names=obj_names,
        relationships=relationships,
        pred_to_idx=pred_to_idx,
        use_all_pairs=True,
        max_pairs=64,
        negative_pair_ratio=2.0,
    )

    assert len(pairs) > 1  # the 1 positive plus some sampled negatives
    assert bool(payload["rel_is_pos"][0].item()) is True
    seen = set()
    for s, o, _extra, _geom in pairs:
        assert s != o
        assert (s, o) not in seen  # no duplicate pairs
        seen.add((s, o))


def test_build_relation_entries_positive_only_mode_has_no_negatives():
    obj_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.5, 0.5]], dtype=torch.float32)
    obj_names = ["cup", "table"]
    relationships = [{"subject_id": 0, "object_id": 1, "predicate": "on"}]
    pred_to_idx = {"on": 0, "relation": 1}

    pairs, payload = _build_relation_entries(
        obj_boxes_t=obj_boxes,
        obj_names=obj_names,
        relationships=relationships,
        pred_to_idx=pred_to_idx,
        use_all_pairs=False,
        max_pairs=64,
    )
    assert len(pairs) == 1
    assert all(bool(x) for x in payload["rel_pos_mask"].tolist())


def test_build_relation_entries_empty_with_single_object():
    obj_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2]], dtype=torch.float32)
    pairs, _payload = _build_relation_entries(
        obj_boxes_t=obj_boxes,
        obj_names=["person"],
        relationships=[],
        pred_to_idx={"relation": 0},
        use_all_pairs=True,
        max_pairs=64,
    )
    assert pairs == []


def test_build_relation_entries_respects_max_pairs():
    n = 6
    obj_boxes = torch.stack(
        [torch.tensor([float(i) * 0.1, 0.0, float(i) * 0.1 + 0.05, 0.05]) for i in range(n)]
    )
    obj_names = [f"obj{i}" for i in range(n)]
    pairs, _payload = _build_relation_entries(
        obj_boxes_t=obj_boxes,
        obj_names=obj_names,
        relationships=[],
        pred_to_idx={"relation": 0},
        use_all_pairs=True,
        max_pairs=5,
        negative_pair_ratio=100.0,  # would sample far more than 5 without the max_pairs cap
    )
    assert len(pairs) <= 5
