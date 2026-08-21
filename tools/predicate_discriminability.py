#!/usr/bin/env python
"""Does visual evidence carry information the co-occurrence prior lacks?

THE QUESTION THIS SETTLES
-------------------------
A flat 50-way linear probe on box geometry reaches 0.0 % recall on `under`,
`above` and `riding` -- which is not credible as an *information* claim, since
vertical offset is literally one of its input features. That number measures
ESTIMATOR CAPACITY under extreme class imbalance (`on` is 36 % of instances),
not information content.

Asked in the form where capacity is not the bottleneck -- balanced binary
discrimination between two predicates the prior actually confuses -- the
answer inverts:

    above       vs on          AUC 0.865   (top feature: dy, vertical offset)
    under       vs on          AUC 0.832   (top feature: dy, opposite sign)
    walking on  vs on          AUC 0.878
    riding      vs on          AUC 0.850   (top feature: subject aspect ratio)
    eating      vs holding     AUC 0.813   (top feature: dy)
    part of     vs of          AUC 0.559   <- annotation style, correctly ~chance
    near        vs next to     AUC 0.580   <- annotation style, correctly ~chance

10 of 22 confusions are strongly separable (AUC >= 0.75) by EIGHT geometry
features and nine parameters, with physically interpretable weights. The two
that are not separable are exactly the annotator-style pairs.

CONCLUSION: the information exists and is trivially extractable. The flat
softmax formulation is what discards it. This is an argument about PREDICTION
FORMULATION, not about visual representation, encoder capacity, or CLIP.

CAVEAT, stated plainly: a high pairwise AUC does not automatically convert
into R@K/mR@K gains. Turning pairwise discriminability into a 50-way decision
needs calibration, and that step is unmeasured here. What this tool
establishes is the existence of the signal, not the size of the gain.

Usage:
    python tools/predicate_discriminability.py
    python tools/predicate_discriminability.py --out runs/analysis/discriminability.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_TRAIN = Path("datasets_vg150_clean/train.jsonl")
DEFAULT_VAL = Path("datasets_vg150_clean/validation.jsonl")
DEFAULT_VOCAB = Path("datasets_vg150_clean/vocabulary/predicates.json")

GEOM_FEATURE_NAMES = [
    "dx", "dy", "log_w_ratio", "log_h_ratio",
    "log_ar_subj", "log_ar_obj", "log_area_subj", "log_area_obj",
]

# Confusions the frequency prior actually makes, taken from
# tools/headroom_analysis.py's measured confusion table.
DEFAULT_CONFUSIONS: List[Tuple[str, str]] = [
    ("under", "on"), ("behind", "on"), ("in front of", "on"), ("above", "on"),
    ("sitting on", "on"), ("standing on", "on"), ("laying on", "on"),
    ("riding", "on"), ("hanging from", "on"), ("parked on", "on"),
    ("walking on", "on"), ("covered in", "on"), ("mounted on", "on"),
    ("carrying", "holding"), ("eating", "holding"), ("watching", "looking at"),
    ("wearing", "has"), ("in", "on"), ("of", "on"), ("with", "has"),
    ("near", "next to"), ("part of", "of"),
]


def load_geometry(path: Path, pred_index: Dict[str, int], cap: Optional[int] = None):
    """Return (geometry [N,8], labels [N]) using the repo's own feature function."""
    sys.path.insert(0, str(Path.cwd()))
    from openvocab_rel.geometry import geom_feats_torch

    geoms: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if cap and n >= cap:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rels = ex.get("relationships") or []
            boxes = ex.get("obj_boxes") or []
            if not rels or not boxes:
                continue
            names = []
            for obj in ex.get("objects") or []:
                nm = obj.get("names") or []
                names.append(str(nm[0]).strip().lower() if nm else "")
            bt = torch.tensor(boxes, dtype=torch.float32)
            if bt.ndim != 2 or bt.shape[1] != 4:
                continue
            si, oi, yy = [], [], []
            for rel in rels:
                s = int(rel.get("subject_id", -1))
                o = int(rel.get("object_id", -1))
                p = str(rel.get("predicate", "")).strip().lower()
                if not (0 <= s < len(names) and 0 <= o < len(names)):
                    continue
                if not (s < bt.shape[0] and o < bt.shape[0]) or p not in pred_index:
                    continue
                si.append(s)
                oi.append(o)
                yy.append(pred_index[p])
            if not si:
                continue
            geoms.append(geom_feats_torch(bt[si], bt[oi]))
            labels.append(torch.tensor(yy, dtype=torch.long))
            n += len(si)
    return torch.cat(geoms), torch.cat(labels)


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores)
    lab = labels[order]
    n_pos = float(lab.sum())
    n_neg = float(len(lab) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.arange(1, len(lab) + 1, dtype=torch.float32)
    return float((ranks[lab == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def probe_pair(
    a: str, b: str, pred_index: Dict[str, int],
    Xtr: torch.Tensor, ytr_all: torch.Tensor,
    Xva: torch.Tensor, yva_all: torch.Tensor,
    epochs: int = 300,
) -> Optional[Dict[str, Any]]:
    """Balanced binary logistic regression on geometry: predicate a vs b."""
    ia, ib = pred_index[a], pred_index[b]
    trm = (ytr_all == ia) | (ytr_all == ib)
    vam = (yva_all == ia) | (yva_all == ib)
    if int(trm.sum()) < 200 or int(vam.sum()) < 60:
        return None
    xtr, ytr = Xtr[trm], (ytr_all[trm] == ia).float()
    xva, yva = Xva[vam], (yva_all[vam] == ia).float()
    pos_weight = torch.tensor([(1.0 - ytr.mean()) / ytr.mean().clamp_min(1e-6)])

    w = torch.zeros(xtr.shape[1], 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, bias], lr=0.1)
    for _ in range(epochs):
        z = (xtr @ w).squeeze(-1) + bias
        loss = F.binary_cross_entropy_with_logits(z, ytr, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        s = (xva @ w).squeeze(-1) + bias
        weights = w.detach().squeeze(-1)
        top = sorted(zip(GEOM_FEATURE_NAMES, weights.tolist()), key=lambda kv: -abs(kv[1]))[:3]
        return {
            "predicate_a": a, "predicate_b": b,
            "n_train": int(trm.sum()), "n_val": int(vam.sum()),
            "auc": roc_auc(s, yva.long()),
            "balanced_acc": float(((s > 0).float() == yva).float().mean()),
            "majority_baseline": float(max(yva.mean(), 1 - yva.mean())),
            "top_features": [{"feature": k, "weight": v} for k, v in top],
        }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--strong_auc", type=float, default=0.75)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    for p in (Path(args.train), Path(args.val), Path(args.vocab)):
        if not p.exists():
            print(f"[FAIL] missing: {p}", file=sys.stderr)
            return 2

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))["idx_to_predicate"]
    pred_vocab = [str(vocab[str(i)]).strip().lower() for i in range(1, len(vocab) + 1)]
    pred_index = {p: i for i, p in enumerate(pred_vocab)}

    print("loading geometry ...", flush=True)
    Xtr, ytr = load_geometry(Path(args.train), pred_index, args.train_cap)
    Xva, yva = load_geometry(Path(args.val), pred_index)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    print(f"  train {Xtr.shape[0]:,}   val {Xva.shape[0]:,}\n")

    print("=" * 84)
    print("BALANCED BINARY GEOMETRY PROBE   (8 features, 9 parameters per probe)")
    print("AUC 0.50 = geometry carries nothing;  1.00 = fully separable")
    print("=" * 84)
    print(f"{'predicate A vs B':<30}{'n_val':>8}{'AUC':>8}{'bal.acc':>9}  top features")
    print("-" * 84)

    results = []
    for a, b in DEFAULT_CONFUSIONS:
        if a not in pred_index or b not in pred_index:
            continue
        r = probe_pair(a, b, pred_index, Xtr, ytr, Xva, yva)
        if r is None:
            continue
        results.append(r)
        feats = ", ".join(
            f"{f['feature']}{'+' if f['weight'] > 0 else '-'}{abs(f['weight']):.2f}"
            for f in r["top_features"]
        )
        print(f"{a + ' vs ' + b:<30}{r['n_val']:>8,}{r['auc']:>8.3f}{r['balanced_acc']*100:>8.1f}%  {feats}")

    strong = [r for r in results if r["auc"] >= args.strong_auc]
    weak = [r for r in results if r["auc"] < 0.60]
    print("\n" + "=" * 84)
    print(f"STRONGLY separable by geometry alone (AUC >= {args.strong_auc}): {len(strong)}/{len(results)}")
    for r in sorted(strong, key=lambda x: -x["auc"]):
        print(f"    {r['predicate_a']:<16} vs {r['predicate_b']:<14} AUC {r['auc']:.3f}")
    print(f"\nNOT separable (AUC < 0.60): {len(weak)}/{len(results)}  -- expected for annotation-style pairs")
    for r in sorted(weak, key=lambda x: x["auc"]):
        print(f"    {r['predicate_a']:<16} vs {r['predicate_b']:<14} AUC {r['auc']:.3f}")
    print("=" * 84)
    print("\nThe signal exists. A flat 50-way softmax under 36% head dominance is")
    print("what discards it. This is a formulation problem, not a representation one.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "n_probes": len(results),
            "n_strong": len(strong),
            "strong_auc_threshold": args.strong_auc,
            "probes": results,
        }, indent=2), encoding="utf-8")
        print(f"[INFO] written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
