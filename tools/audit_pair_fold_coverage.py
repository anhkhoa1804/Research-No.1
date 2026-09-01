#!/usr/bin/env python
"""How much pair information can an OUT-OF-FOLD pair estimator actually carry?

`runs/p27` reads as "pair-conditioned statistics cannot reproduce the model
term". That reading is only valid if the pair-conditioned arms were ABLE to
carry pair information in the first place. They are estimated per fold from
training rows only, and `Distill._fold_pair_mean` documents its own fallback:

    Groups with no training row fall back to the training global mean

A (subject, object) group seen exactly once in the cache CANNOT appear in the
training rows of the fold that holds it out. Such a row therefore receives the
GLOBAL mean -- no pair information at all -- while still being scored as though
the arm were pair-conditioned. The same rows make `G_residual` carry very nearly
the WHOLE model term rather than a within-group residual, because
`md - global_mean` is just a recentred `md`.

This tool measures that dilution instead of assuming it, using the SAME
`Distill` object, the SAME fold rule and the SAME pair ids as `p27`, so the
numbers describe p27 exactly and not a re-implementation of it.

Reports per fold partition:
  fraction of held-out rows whose pair IS estimable (>=1 training row)
  fraction that fall back to the global mean, overall and by head/body/tail
  how much of the real model term survives in G_residual for fallback rows
    (cosine and correlation against the real term, for fallback vs estimable)

Usage:
    python tools/audit_pair_fold_coverage.py --dump <cache> --prior <prior.json> \
        --out runs/<name>/fold_coverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

import pair_prior_distillation as PPD
from pair_prior_distillation import Distill, Mech, N_FOLDS


def _log(m: str = "") -> None:
    print(m, flush=True)


def audit_one(D: Distill, salt: int) -> Dict[str, Any]:
    """Per-fold estimability of the (subject, object) group, for one partition."""
    n = D.n
    G = int(D.pair_id.max()) + 1
    bucket = [D.B.bucket_of[int(c)] for c in D.y.tolist()]

    per_fold: List[Dict[str, Any]] = []
    est_all = torch.zeros(n, dtype=torch.bool)

    for f in range(N_FOLDS):
        te = D.fold == f
        train = ~te
        cnt = torch.zeros(G).index_add_(
            0, D.pair_id[train], torch.ones(int(train.sum())))
        # a held-out row is "estimable" iff its group has >=1 TRAINING row
        estimable = cnt[D.pair_id] > 0
        est_all[te] = estimable[te]
        held = int(te.sum())
        per_fold.append({
            "fold": f,
            "held_out_rows": held,
            "estimable": int((te & estimable).sum()),
            "fallback": int((te & ~estimable).sum()),
            "fallback_frac": float((te & ~estimable).sum() / max(held, 1)),
        })

    # ---- bucket breakdown over the union of held-out decisions (= all rows) ----
    by_bucket: Dict[str, Dict[str, float]] = {}
    for b in ("head", "body", "tail"):
        m = torch.tensor([x == b for x in bucket])
        tot = int(m.sum())
        if tot:
            by_bucket[b] = {
                "rows": tot,
                "fallback_frac": float((m & ~est_all).sum() / tot),
            }

    # ---- what does G_residual actually carry on fallback rows? ----
    # Rebuild the residual exactly as the arm does, fold by fold, then compare
    # it to the REAL model term on fallback vs estimable rows.
    resid = torch.zeros_like(D.md)
    pmean = torch.zeros_like(D.md)
    for f in range(N_FOLDS):
        te = D.fold == f
        train = ~te
        gm = D._fold_pair_mean(D.md, train)
        pmean[te] = gm[te]
        resid[te] = D.md[te] - gm[te]

    def _sim(mask: torch.Tensor) -> Dict[str, float]:
        if int(mask.sum()) == 0:
            return {"rows": 0}
        a = resid[mask]
        b = D.md[mask]
        # centre each row over classes: only relative shape drives the softmax
        ac = a - a.mean(-1, keepdim=True)
        bc = b - b.mean(-1, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(ac, bc, dim=-1)
        return {
            "rows": int(mask.sum()),
            "mean_cosine_residual_vs_real": float(cos.mean()),
            "frac_rows_cosine_gt_0.99": float((cos > 0.99).float().mean()),
        }

    return {
        "salt": salt,
        "per_fold": per_fold,
        "fallback_frac_overall": float((~est_all).float().mean()),
        "by_bucket": by_bucket,
        "residual_identity": {
            "fallback_rows": _sim(~est_all),
            "estimable_rows": _sim(est_all),
        },
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p10_model_recalibration/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("OUT-OF-FOLD PAIR ESTIMABILITY AUDIT -- CPU only, cache read-only, NO GPU")
    _log("=" * 100)
    B = Mech(args.dump, args.prior, "raw50")
    _log(f"  images={B.n_images} GT rows={B.n_gt} classes={B.n_classes}")

    out: List[Dict[str, Any]] = []
    for salt in range(args.repeats):
        D = Distill(B, args.prior, args.tau, args.k, salt)
        r = audit_one(D, salt)
        out.append(r)
        fb = r["fallback_frac_overall"] * 100.0
        rid = r["residual_identity"]
        _log(f"\n  [salt {salt}] held-out rows falling back to the GLOBAL mean: {fb:.1f}%")
        _log("      by bucket: " + "  ".join(
            f"{b}={v['fallback_frac']*100:.1f}%" for b, v in r["by_bucket"].items()))
        for nm in ("fallback_rows", "estimable_rows"):
            s = rid[nm]
            if s.get("rows"):
                _log(f"      G_residual vs REAL model term on {nm:15s} "
                     f"n={s['rows']:6d}  mean cos={s['mean_cosine_residual_vs_real']:+.4f}"
                     f"  frac cos>0.99 = {s['frac_rows_cosine_gt_0.99']*100:5.1f}%")

    mean_fb = sum(r["fallback_frac_overall"] for r in out) / len(out)
    _log("\n" + "-" * 100)
    _log(f"  MEAN over {len(out)} partitions: {mean_fb*100:.1f}% of scored rows receive "
         f"NO pair information in the pair-conditioned arms.")
    _log("-" * 100)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"tool": "audit_pair_fold_coverage", "dump": args.dump,
         "tau": args.tau, "k": args.k,
         "mean_fallback_frac": mean_fb, "per_salt": out}, indent=2), encoding="utf-8")
    _log(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
