"""
Standalone PredCls evaluation driver for bknyaz/sgg's IMP+ (Neural Motifs /
IMP) checkpoint against the converted VG150 test split.

Deliberately does NOT go through main.py / VG.splits(): that classmethod
builds train + zero/10/100-shot eval splits with hard-coded assert counts
(4519 / 9602 / 16528 / 26446) that depend on triplet-level overlap with the
*exact* original training corpus. We only need `test_alls` -- the standard,
shot-stratification-free PredCls number -- which `VG(mode='test', ...)`
alone provides, with no dependency on train data at all (confirmed: this
checkpoint has no freq_bias.* weights, so no train statistics are needed for
the forward pass either).

Reuses the library's own evaluator (lib/sgg_eval.py, lib/eval.py::val_batch)
verbatim -- this is the standard evaluator used across the SGG literature
for R@K / mR@K, so we get comparable numbers rather than a hand-rolled
reimplementation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import vg_loader_patch  # noqa: E402

REPO = None  # set in main() once we know --sgg-repo


def build_model(test_data, ckpt_path, device):
    from sgg_models.rel_model_stanford import RelModelStanford

    model = RelModelStanford(
        train_data=test_data,
        mode="predcls",
        use_bias=False,
        test_bias=False,
        backbone="vgg16",
        RELS_PER_IMG=1024,
        edge_model="motifs",
        require_overlap_det=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"checkpoint epoch={ckpt.get('epoch')} global_batch_iter={ckpt.get('global_batch_iter')}")
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print("MISSING:", missing)
    if unexpected:
        print("UNEXPECTED:", unexpected)
    assert len(missing) == 0 and len(unexpected) == 0, (
        "state_dict did not load cleanly -- architecture mismatch, do not trust any result"
    )
    model.to(device)
    model.eval()
    return model


def run(data_dir, ckpt_path, n_images, batch_size, device_str, out_json, wprd_out=None):
    vg_loader_patch.apply()
    from dataloaders.visual_genome import VG, VGDataLoader, vg_collate
    from lib.pytorch_misc import set_mode
    from lib.sgg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
    from lib.eval import val_batch

    VG.split = "stanford"
    device = torch.device(device_str)

    test_data = VG(mode="test", data_dir=data_dir, num_val_im=0,
                   filter_empty_rels=True, torch_detector=True)
    print(f"test_data: {len(test_data)} images")

    if n_images > 0:
        test_data.filenames = test_data.filenames[:n_images]
        test_data.gt_boxes = test_data.gt_boxes[:n_images]
        test_data.gt_classes = test_data.gt_classes[:n_images]
        test_data.relationships = test_data.relationships[:n_images]
        print(f"PILOT: restricted to first {len(test_data)} images")

    model = build_model(test_data, ckpt_path, device)
    set_mode(model, mode="predcls", is_train=False, verbose=True)

    loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: vg_collate(x, mode="rel", num_gpus=1, is_train=False,
                                        torch_detector=test_data.torch_detector, is_cuda=(device.type == "cuda")),
        drop_last=False,
    )

    evaluator = {
        "predcls": BasicSceneGraphEvaluator("predcls"),
        "predcls_nogc": BasicSceneGraphEvaluator("predcls", multiple_preds=True),
    }
    evaluator_list = []
    for index, name_s in enumerate(test_data.ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name_s, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list = []
    for index, name_s in enumerate(test_data.ind_to_predicates):
        if index == 0:
            continue
        evaluator_multiple_preds_list.append(
            (index, name_s, BasicSceneGraphEvaluator.all_modes(multiple_preds=True))
        )

    all_wprd_rows = [] if wprd_out else None

    t0 = time.time()
    n_batches_done = 0
    with torch.no_grad():
        for val_b, batch in enumerate(loader):
            pred_entries = val_batch(
                model, val_b * batch_size, batch, evaluator, "predcls",
                test_data, evaluator_list, evaluator_multiple_preds_list,
            )
            if all_wprd_rows is not None:
                for pe in pred_entries:
                    all_wprd_rows.append({
                        "pred_classes": pe["pred_classes"].tolist(),
                        "pred_rel_inds": pe["pred_rel_inds"].tolist(),
                        "rel_scores": pe["rel_scores"].tolist(),
                    })
            n_batches_done += 1
            if n_batches_done % 20 == 0:
                elapsed = time.time() - t0
                print(f"batch {n_batches_done}/{len(loader)} elapsed={elapsed:.1f}s "
                      f"({elapsed/n_batches_done:.2f} s/batch)")

    elapsed = time.time() - t0
    print(f"DONE: {n_batches_done} batches, {len(test_data)} images, {elapsed:.1f}s "
          f"({elapsed/max(1,len(test_data)):.3f} s/image)")

    gc_stats = evaluator["predcls"].print_stats(verbose=True)
    nogc_stats = evaluator["predcls_nogc"].print_stats(verbose=True)
    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, "predcls")
    mean_recall_nogc = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, "predcls", multiple_preds=True)

    result = {
        "n_images": len(test_data),
        "elapsed_s": elapsed,
        "GC": {
            "R@20": gc_stats["R@20"] * 100, "R@50": gc_stats["R@50"] * 100, "R@100": gc_stats["R@100"] * 100,
            "mR@20": float(mean_recall["R@20"]) * 100, "mR@50": float(mean_recall["R@50"]) * 100,
            "mR@100": float(mean_recall["R@100"]) * 100,
        },
        "NoGC": {
            "R@20": nogc_stats["R@20"] * 100, "R@50": nogc_stats["R@50"] * 100, "R@100": nogc_stats["R@100"] * 100,
            "mR@20": float(mean_recall_nogc["R@20"]) * 100, "mR@50": float(mean_recall_nogc["R@50"]) * 100,
            "mR@100": float(mean_recall_nogc["R@100"]) * 100,
        },
        # CORRECTED 2026-09-03: Table 1 of Knyazev et al. BMVC 2020 (arXiv:2005.08230)
        # reports TWO architectures, "MP" (Xu et al., Message Passing) and "NM"
        # (Zellers et al., Neural Motifs). This checkpoint loads with 0 missing/0
        # unexpected keys against edge_model="motifs" -- it is NM, not MP -- and the
        # README's "-loss baseline" flag for this checkpoint matches Table 1's
        # "NM, Baseline" loss label, not "MP, Baseline". The MP row (74.8/20.6) was
        # cited here originally by mistake; see docs/CROSS_MODEL_IMP_PLUS_RESULT.md
        # section 10.1 for the correction, left as history rather than silently
        # dropped.
        "published_target_NoGC_NM_baseline": {"R@50": 80.5, "mR@50": 26.9, "source": "Knyazev et al. BMVC 2020, Table 1, 'NM, Baseline' row"},
        "published_target_NoGC_MP_baseline_WRONG_ARCHITECTURE": {"R@50": 74.8, "mR@50": 20.6, "source": "Knyazev et al. BMVC 2020, Table 1, 'MP, Baseline' row -- Message Passing, NOT this checkpoint's Neural-Motifs architecture; kept only so the original (incorrect) comparison is traceable"},
    }
    print(json.dumps(result, indent=2))
    if out_json:
        Path(out_json).write_text(json.dumps(result, indent=2))

    if wprd_out and all_wprd_rows is not None:
        Path(wprd_out).write_text(json.dumps(all_wprd_rows))
        print(f"wrote {len(all_wprd_rows)} prediction rows to {wprd_out}")

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgg-repo", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-images", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--wprd-out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.sgg_repo)
    run(args.data_dir, args.ckpt, args.n_images, args.batch_size, args.device,
        args.out_json, args.wprd_out)
