#!/usr/bin/env python
"""The appearance probe's P0 baseline and its additive arms break ties differently.

DEFECT
------
`tools/appearance_probe.py` computes its reported baseline as

    R0, m0, ... = mr_of(Pva.argmax(-1), Yva)                  # line ~377

but every scored arm -- the ablation gate and the whole additive lambda sweep --
predicts through the candidate path

    pred = cand_va.gather(1, s.argmax(1, keepdim=True)).squeeze(1)   # `show()`

where `cand_va = torch.topk(Pva, k=5, dim=-1).indices`.

When a prior row has TIED values among its top entries, `torch.argmax` and
`torch.topk` do not necessarily select the same index. So the two paths
disagree on a non-empty set of pairs, and the additive arm at lambda -> 0 does
NOT converge to the reported P0.

CONSEQUENCE
-----------
`headroom_pct = (m - m0) / (oracle - m0)` credits every additive arm against a
baseline computed under a different decision rule than the arm itself. The
like-for-like baseline is the one the arms actually reduce to: the topk-gather
path at lambda = 0.

This tool measures the disagreement and restates the pre-registered additive
sweep against the like-for-like baseline. It MODIFIES NOTHING and re-runs no
fit: it recomputes the baseline from the same cache and re-reads the already
recorded arm metrics from the probe's own result JSON.

Both numbers are reported. The pre-registered result is not overwritten.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

K = 5


def mr_of(pred: torch.Tensor, Y: torch.Tensor, bucket: Dict[int, str]):
    hit = (pred == Y)
    ph, pg = Counter(), Counter()
    for y, h in zip(Y.tolist(), hit.tolist()):
        pg[y] += 1
        ph[y] += int(h)
    mR = sum(ph[k] / pg[k] for k in pg) / max(1, len(pg))
    bk = {}
    for b in ("head", "body", "tail"):
        ks = [k for k in pg if bucket[k] == b]
        bk[b] = sum(ph[k] / pg[k] for k in ks) / max(1, len(ks)) if ks else 0.0
    return float(hit.float().mean()), float(mR), bk, len(pg)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--probe_result", type=Path, required=True,
                    help="the appearance probe's own output JSON, read-only")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    blob = torch.load(args.cache, weights_only=False)
    TR, VA, PV = blob["train"], blob["val"], blob["PV"]
    P = len(PV)
    Pva = VA["prior"] - VA["prior"].mean(-1, keepdim=True)
    Yva = VA["y"]

    cnt = torch.bincount(TR["y"], minlength=P).float()
    order = torch.argsort(cnt, descending=True)
    HEAD, BODY = set(order[:15].tolist()), set(order[15:35].tolist())
    bucket = {i: ("head" if i in HEAD else "body" if i in BODY else "tail") for i in range(P)}

    cand = torch.topk(Pva, k=K, dim=-1).indices
    pred_argmax = Pva.argmax(-1)                                        # probe's P0 path
    pred_gather = cand.gather(1, Pva.gather(1, cand).argmax(1, keepdim=True)).squeeze(1)

    n_diff = int((pred_argmax != pred_gather).sum())
    R_a, m_a, bk_a, ncls = mr_of(pred_argmax, Yva, bucket)
    R_g, m_g, bk_g, _ = mr_of(pred_gather, Yva, bucket)

    # How many rows carry a tie at the top of the prior row?
    top2 = torch.topk(Pva, k=2, dim=-1).values
    n_tied = int(((top2[:, 0] - top2[:, 1]).abs() < 1e-7).sum())

    probe = json.loads(Path(args.probe_result).read_text(encoding="utf-8"))
    oracle_mR = float(probe["oracle_mR"])

    restated = []
    for key, val in sorted(probe.items()):
        if not key.startswith("additive_lam"):
            continue
        lam = float(key.replace("additive_lam", ""))
        m = float(val["mR"])
        restated.append({
            "lambda": lam,
            "arm_mR": m,
            "as_published_baseline_mR": m_a,
            "as_published_headroom_pct": (m - m_a) / max(1e-9, oracle_mR - m_a) * 100.0,
            "like_for_like_baseline_mR": m_g,
            "like_for_like_delta_mR_points": (m - m_g) * 100.0,
            "like_for_like_headroom_pct": (m - m_g) / max(1e-9, oracle_mR - m_g) * 100.0,
        })
    restated.sort(key=lambda r: r["lambda"])

    any_positive = any(r["like_for_like_delta_mR_points"] > 0 for r in restated)

    out: Dict[str, Any] = {
        "defect": "P0 baseline uses Pva.argmax(-1); every scored arm uses topk-then-gather. "
                  "Tied prior rows make the two disagree, so the additive arms are credited "
                  "against a baseline they do not reduce to at lambda=0.",
        "cache": str(args.cache),
        "probe_result": str(args.probe_result),
        "n_val_pairs": int(Yva.numel()),
        "n_classes_in_mR": ncls,
        "n_pairs_where_paths_disagree": n_diff,
        "frac_pairs_disagree": n_diff / max(1, int(Yva.numel())),
        "n_pairs_with_top1_tie": n_tied,
        "baseline_as_published": {"path": "Pva.argmax(-1)", "R": R_a, "mR": m_a,
                                  "tail_mR": bk_a["tail"]},
        "baseline_like_for_like": {"path": "topk(K=5) then gather-argmax", "R": R_g, "mR": m_g,
                                   "tail_mR": bk_g["tail"]},
        "baseline_mR_gap_points": (m_g - m_a) * 100.0,
        "oracle_mR": oracle_mR,
        "additive_sweep_restated": restated,
        "verdict": ("SOME LAMBDA STILL POSITIVE like-for-like" if any_positive else
                    "EVERY LAMBDA IS NEGATIVE like-for-like: the published +0.2% upper bound "
                    "was an artifact of the tie-breaking mismatch, and the pre-registered "
                    "conclusion (H0 supported) is unchanged and strengthened"),
        "note": "Read-only. The pre-registered probe result is preserved verbatim; this is a "
                "superseding restatement, not a rewrite.",
    }

    print("=" * 88)
    print("TIE-BREAKING FORENSICS on the appearance probe's baseline")
    print("=" * 88)
    print(f"  val pairs                     : {out['n_val_pairs']}")
    print(f"  rows with a top-1 tie         : {n_tied}")
    print(f"  rows where the two paths differ: {n_diff}  ({out['frac_pairs_disagree']*100:.2f}%)")
    print(f"  baseline as published (argmax): R {R_a*100:6.2f}  mR {m_a*100:6.2f}")
    print(f"  baseline like-for-like (topk) : R {R_g*100:6.2f}  mR {m_g*100:6.2f}")
    print(f"  baseline mR gap               : {out['baseline_mR_gap_points']:+.2f} points")
    print()
    print(f"  {'lambda':>7} {'arm mR':>8} {'published %':>12} {'like-for-like dmR':>19} {'l4l %':>8}")
    for r in restated:
        print(f"  {r['lambda']:>7} {r['arm_mR']*100:>8.2f} {r['as_published_headroom_pct']:>12.2f}"
              f" {r['like_for_like_delta_mR_points']:>19.2f} {r['like_for_like_headroom_pct']:>8.2f}")
    print()
    print(f"  VERDICT: {out['verdict']}")
    print("=" * 88)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
