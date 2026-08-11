"""Pair/negative construction smoke tests, synthetic in-memory data only."""
import torch

from openvocab_rel.datasets.vg150_loader import _build_relation_entries, negative_pair_ratio_is_inert


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


# ---- negative_pair_ratio inertness: the exact invariant behind the shipped
# training script's config, made loud (VG150DataLoader emits a
# logging.warning) instead of silent. See docs/known_issues.md.


def test_negative_pair_ratio_is_inert_when_use_all_pairs_is_false():
    # This is the shipped scripts/train/train_l4_phase34.sh combination:
    # --use_all_pairs false --negative_pair_ratio 2.0 -- confirmed inert.
    assert negative_pair_ratio_is_inert(use_all_pairs=False, negative_pair_ratio=2.0) is True


def test_negative_pair_ratio_not_inert_when_use_all_pairs_is_true():
    assert negative_pair_ratio_is_inert(use_all_pairs=True, negative_pair_ratio=2.0) is False


def test_negative_pair_ratio_zero_is_not_flagged_as_inert():
    # negative_pair_ratio=0 combined with use_all_pairs=False is the
    # config.py-documented, self-consistent way to say "positives only" --
    # both settings agree, so this is NOT a misleading/inert combination
    # and must not trigger the warning.
    assert negative_pair_ratio_is_inert(use_all_pairs=False, negative_pair_ratio=0.0) is False


def test_negative_pair_ratio_inert_regardless_of_use_all_pairs_when_true():
    # Sanity: once use_all_pairs=True, ANY negative_pair_ratio value
    # (including 0, which _build_relation_entries's `>= 0.0` check treats
    # as "sample zero negatives", still reachable code) is never flagged.
    assert negative_pair_ratio_is_inert(use_all_pairs=True, negative_pair_ratio=0.0) is False
