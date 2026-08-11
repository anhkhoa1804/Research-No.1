"""Geometry helper smoke tests. Pure math + one synthetic PIL image -- no
dataset, no model, no GPU needed."""
import math

import torch
from PIL import Image

from openvocab_rel.geometry import (
    build_all_pairs,
    centers_wh,
    geom_feats,
    geom_feats_torch,
    preprocess_boxes_to_clip224,
    prune_pairs_by_geometry,
)


def test_geom_feats_identity_box_pair_is_near_zero_offset():
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [0.0, 0.0, 10.0, 10.0]
    dx, dy, rw, rh, ar1, ar2, a1, a2 = geom_feats(b1, b2)
    assert abs(dx) < 1e-6
    assert abs(dy) < 1e-6
    assert abs(rw) < 1e-6
    assert abs(rh) < 1e-6
    assert math.isclose(ar1, ar2, rel_tol=1e-6)
    assert math.isclose(a1, a2, rel_tol=1e-6)


def test_geom_feats_torch_matches_geom_feats_scalar_version():
    b1 = [10.0, 20.0, 50.0, 80.0]
    b2 = [30.0, 10.0, 70.0, 60.0]
    py_feats = geom_feats(b1, b2)
    t_feats = geom_feats_torch(
        torch.tensor([b1], dtype=torch.float32),
        torch.tensor([b2], dtype=torch.float32),
    )[0].tolist()
    for a, b in zip(py_feats, t_feats):
        assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)


def test_centers_wh():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]], dtype=torch.float32)
    centers, wh = centers_wh(boxes)
    assert torch.allclose(centers, torch.tensor([[5.0, 10.0]]))
    assert torch.allclose(wh, torch.tensor([[10.0, 20.0]]))


def test_preprocess_boxes_to_clip224_stays_in_bounds():
    image = Image.new("RGB", (400, 300), color=(128, 128, 128))
    boxes = torch.tensor([[10.0, 10.0, 100.0, 100.0]], dtype=torch.float32)
    out_boxes, _meta = preprocess_boxes_to_clip224(image, boxes, out_size=224)
    assert out_boxes.shape == (1, 4)
    assert float(out_boxes.min()) >= 0.0
    assert float(out_boxes.max()) <= 224.0


def test_build_all_pairs_excludes_self_pairs():
    pairs = build_all_pairs(3, max_pairs=100)
    assert len(pairs) == 6  # 3 * 2 ordered pairs
    assert (0, 0) not in pairs
    assert (0, 1) in pairs and (1, 0) in pairs


def test_build_all_pairs_respects_max_pairs():
    pairs = build_all_pairs(5, max_pairs=3)
    assert len(pairs) == 3


def test_prune_pairs_by_geometry_keeps_nearest():
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [200.0, 200.0, 210.0, 210.0]],
        dtype=torch.float32,
    )
    cand = [(0, 1), (0, 2), (1, 2)]
    pruned = prune_pairs_by_geometry(boxes, cand, k=1)
    assert pruned == [(0, 1)]
