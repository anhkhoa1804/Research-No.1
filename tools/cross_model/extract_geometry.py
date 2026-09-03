"""
Extracts GT subject/object boxes for the IMP+ cross-model WPRD decomposition,
in EXACTLY the same (image, relationship) iteration order as
extract_wprd_pairs.py, so the output aligns 1:1 with wprd_pairs_full.pt
without needing an explicit join key.

No model, no CUDA, no DataLoader/collate -- this only reads the plain
Python/numpy attributes VG.__init__ already loaded (gt_boxes, gt_classes,
relationships), the same attributes extract_wprd_pairs.py reads directly
without going through __getitem__. Boxes are stored RAW (pre-rescale); this
is fine because tools/wprd_geometry_control.py's `_geom` normalizes by each
image's own box extent (max-min over ALL of that image's object boxes, not
just the pair), so no absolute image size is needed -- to reproduce that
exactly (not an approximation using only the pair's own extent), this saves
every image's FULL object box array plus the pair's local indices into it,
matching the `B.meta["obj_boxes"][i]` / `B.meta["pairs"][i]` shape
`tools/wprd_geometry_control.py::geometry_features_raw` already consumes.

Alignment is verified, not assumed: gt_y computed here is asserted
IDENTICAL, row for row, to wprd_pairs_full.pt's gt_y before anything is
written.

CPU only. No GPU. No model forward pass.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import vg_loader_patch  # noqa: E402


def run(sgg_repo, data_dir, n_images, out_path, check_against):
    sys.path.insert(0, sgg_repo)
    vg_loader_patch.apply()
    from dataloaders.visual_genome import VG

    VG.split = "stanford"
    test_data = VG(mode="test", data_dir=data_dir, num_val_im=0,
                   filter_empty_rels=True, torch_detector=True)
    if n_images > 0:
        test_data.filenames = test_data.filenames[:n_images]
        test_data.gt_boxes = test_data.gt_boxes[:n_images]
        test_data.gt_classes = test_data.gt_classes[:n_images]
        test_data.relationships = test_data.relationships[:n_images]
    print(f"extracting geometry for {len(test_data)} images")

    obj_boxes, pairs_local, gt_y, image_id = [], [], [], []
    t0 = time.time()
    for img_idx in range(len(test_data)):
        boxes = torch.as_tensor(test_data.gt_boxes[img_idx], dtype=torch.float32)
        obj_boxes.append(boxes)
        img_pairs = []
        for (subj_local, obj_local, pred_idx) in test_data.relationships[img_idx]:
            img_pairs.append((int(subj_local), int(obj_local)))
            gt_y.append(int(pred_idx) - 1)
            image_id.append(img_idx)
        pairs_local.append(torch.tensor(img_pairs, dtype=torch.long)
                           if img_pairs else torch.zeros(0, 2, dtype=torch.long))
        if (img_idx + 1) % 5000 == 0:
            print(f"{img_idx+1}/{len(test_data)} images, elapsed={time.time()-t0:.1f}s")

    out = {
        "obj_boxes": obj_boxes,       # list[Tensor(n_obj_i,4)], per image, ALL objects
        "pairs_local": pairs_local,   # list[Tensor(n_pairs_i,2)], local (subj,obj) indices
        "gt_y": torch.tensor(gt_y, dtype=torch.long),
        "image_id": torch.tensor(image_id, dtype=torch.long),
    }
    print(f"DONE: {len(gt_y)} rows, {time.time()-t0:.1f}s")

    if check_against:
        ref = torch.load(check_against, map_location="cpu", weights_only=False)
        assert ref["gt_y"].shape[0] == out["gt_y"].shape[0], (
            f"row count mismatch: geometry {out['gt_y'].shape[0]} vs "
            f"{check_against} {ref['gt_y'].shape[0]}")
        assert torch.equal(ref["gt_y"], out["gt_y"]), (
            "gt_y sequence mismatch -- geometry rows are NOT in the same "
            "order as the WPRD pairs file; do not use this output")
        print(f"ALIGNMENT CHECK vs {check_against}: PASS "
              f"({out['gt_y'].shape[0]} rows, gt_y identical row-for-row)")

    torch.save(out, out_path)
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgg-repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-images", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-against", default=None,
                    help="wprd_pairs_full.pt (or _300.pt) to verify row alignment against")
    args = ap.parse_args()
    run(args.sgg_repo, args.data_dir, args.n_images, args.out, args.check_against)
