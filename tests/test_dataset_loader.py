"""VG150JSONLDataset smoke test against a tiny synthetic fixture.

No real images (VG150JSONLDataset._resolve_image's documented fallback --
a solid gray placeholder -- is exercised here rather than worked around),
no CLIP processor (processor=None), no network access.
"""
from pathlib import Path

from openvocab_rel.datasets.vg150_loader import VG150JSONLDataset, VG150LoaderConfig

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tiny_vg150"

# Covers every predicate string used in the fixture JSONL files, plus the
# synthetic background class every real loader appends at runtime.
PRED_TO_IDX = {
    "sitting on": 0,
    "under": 1,
    "on": 2,
    "riding": 3,
    "wearing": 4,
    "holding": 5,
    "near": 6,
    "relation": 7,
}


def _make_dataset(split: str) -> VG150JSONLDataset:
    cfg = VG150LoaderConfig(
        vg150_root=str(FIXTURE_ROOT),
        split=split,
        max_objects=16,
        max_pairs=32,
        use_all_pairs=True,
        negative_pair_ratio=2.0,
    )
    return VG150JSONLDataset(cfg, PRED_TO_IDX, split=split, processor=None, clip_input_res=224)


def test_train_split_loads_all_fixture_rows():
    ds = _make_dataset("train")
    assert len(ds.rows) == 3


def test_validation_split_loads_all_fixture_rows():
    ds = _make_dataset("validation")
    assert len(ds.rows) == 2


def test_getitem_shapes_without_clip_processor():
    ds = _make_dataset("train")
    example = ds[0]
    assert example["pixel_values"].shape == (3, 224, 224)  # zero-tensor fallback, processor=None
    assert example["obj_boxes"].shape[0] == example["obj_boxes_224"].shape[0]
    assert len(example["obj_labels"]) == example["obj_boxes"].shape[0]
    n_pairs = len(example["pairs"])
    assert n_pairs == len(example["rel_preds"]) == int(example["rel_is_pos"].shape[0])
    assert n_pairs > 0
    assert example["task_type"] == "vg150_sgg"


def test_getitem_uses_placeholder_image_when_file_missing():
    ds = _make_dataset("train")
    example = ds[0]
    # no real image file exists anywhere for this fixture's fake image_ids
    assert example["image"].size == (336, 336)


def test_getitem_includes_at_least_one_positive_pair():
    ds = _make_dataset("train")
    example = ds[0]  # tiny_train_0001 has 2 annotated relationships
    assert bool(example["rel_is_pos"].bool().any())
