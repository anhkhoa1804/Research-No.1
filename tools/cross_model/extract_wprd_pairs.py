"""
Extracts, for every GT relation row in the (possibly restricted) VG150 test
split, the IMP+ model's predicted 50-way predicate distribution evaluated on
that EXACT (subject, object) pair -- not the model's own ranked/thresholded
output, the raw softmax over all 51 classes (index 0 = background) for the
GT-labeled pair specifically.

This is possible without any extra forward passes because PredCls scores
every ordered (i, j) object pair in the image (get_rel_inds has no overlap
requirement outside sgdet mode), so every GT pair is already among the
model's candidate relations; we just need to look it up.

Output: a torch .pt file matching the project's own cross-model interface
note (docs/CROSS_MODEL_FEASIBILITY.md / tools/sgg_evaluation_table.py):
    {"model_term": Tensor(n_gt_rows, 50), "gt_y": Tensor(n_gt_rows,) in 0..49,
     "subj_label": [str]*n_gt_rows, "obj_label": [str]*n_gt_rows}
"model_term" holds the 50 non-background predicate probabilities in the
model's own (canonical VG150 alphabetical) predicate order, index 0 = "above".
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import vg_loader_patch  # noqa: E402
from run_predcls_eval import build_model  # noqa: E402


def run(sgg_repo, data_dir, ckpt_path, n_images, device_str, out_path):
    sys.path.insert(0, sgg_repo)
    vg_loader_patch.apply()
    from dataloaders.visual_genome import VG, vg_collate
    from lib.pytorch_misc import set_mode

    VG.split = "stanford"
    device = torch.device(device_str)

    test_data = VG(mode="test", data_dir=data_dir, num_val_im=0,
                   filter_empty_rels=True, torch_detector=True)
    if n_images > 0:
        test_data.filenames = test_data.filenames[:n_images]
        test_data.gt_boxes = test_data.gt_boxes[:n_images]
        test_data.gt_classes = test_data.gt_classes[:n_images]
        test_data.relationships = test_data.relationships[:n_images]
    print(f"extracting WPRD pairs for {len(test_data)} images")

    model = build_model(test_data, ckpt_path, device)
    set_mode(model, mode="predcls", is_train=False, verbose=False)

    loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=lambda x: vg_collate(x, mode="rel", num_gpus=1, is_train=False,
                                        torch_detector=True, is_cuda=(device.type == "cuda")),
        drop_last=False,
    )

    model_term_rows, gt_y_rows, subj_labels, obj_labels = [], [], [], []
    n_gt_total, n_gt_missing = 0, 0

    t0 = time.time()
    with torch.no_grad():
        for img_idx, batch in enumerate(loader):
            boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = model(batch.scatter())
            rels_i = rels_i  # (n_pairs, 2) local indices, numpy or tensor
            if not torch.is_tensor(rels_i):
                rels_i = torch.as_tensor(rels_i)
            if not torch.is_tensor(pred_scores_i):
                pred_scores_i = torch.as_tensor(pred_scores_i)

            lookup = {}
            for row in range(rels_i.shape[0]):
                a, b = int(rels_i[row, 0]), int(rels_i[row, 1])
                lookup[(a, b)] = row

            gt_classes_img = test_data.gt_classes[img_idx]  # (n_obj,) class idx 1..150
            for (subj_local, obj_local, pred_idx) in test_data.relationships[img_idx]:
                n_gt_total += 1
                key = (int(subj_local), int(obj_local))
                if key not in lookup:
                    n_gt_missing += 1
                    continue
                row = lookup[key]
                probs = pred_scores_i[row].detach().float().cpu()
                model_term_rows.append(probs[1:51])  # drop background, keep 50-way
                gt_y_rows.append(int(pred_idx) - 1)  # 1..50 -> 0..49
                subj_labels.append(test_data.ind_to_classes[int(gt_classes_img[subj_local])])
                obj_labels.append(test_data.ind_to_classes[int(gt_classes_img[obj_local])])

            if (img_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"{img_idx+1}/{len(test_data)} images, elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"DONE: {len(test_data)} images, {elapsed:.1f}s, "
          f"{n_gt_total} GT rows, {n_gt_missing} missing lookups "
          f"({100*n_gt_missing/max(1,n_gt_total):.3f}%)")

    out = {
        "model_term": torch.stack(model_term_rows) if model_term_rows else torch.zeros(0, 50),
        "gt_y": torch.tensor(gt_y_rows, dtype=torch.long),
        "subj_label": subj_labels,
        "obj_label": obj_labels,
        "n_gt_total": n_gt_total,
        "n_gt_missing": n_gt_missing,
    }
    torch.save(out, out_path)
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgg-repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-images", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.sgg_repo, args.data_dir, args.ckpt, args.n_images, args.device, args.out)
