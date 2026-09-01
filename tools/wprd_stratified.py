#!/usr/bin/env python
"""WPRD stratified by pair support -- where does relational grounding live?

runs/p33 reported two numbers that do not agree:

    WPRD macro    0.5542   (every (group, predicate-pair) cell counts once)
    WPRD weighted 0.5266   (cells weighted by how many comparisons they carry)

Weighting by size pulls the score toward 0.5. Large cells are frequent
(subject, object) pairs. So the hypothesis this tool tests is:

    the model's image-conditioned relational discrimination is INVERSELY
    related to pair support -- it is weakest exactly where the pair prior is
    strongest, and strongest exactly where the pair prior cannot be estimated.

If true it explains, in one mechanism, why runs/p32's vision-free pair
statistics beat the model term on the ESTIMABLE subset (which is frequent pairs
by construction) while runs/p33 still measures real grounding overall.

Also reported, because both are confounds that would inflate WPRD if ignored:

  * same-instance rows. VG annotates one (subject instance, object instance)
    with several predicates. Those rows share a rel_feat exactly, so they tie
    at AUC 0.5 and DILUTE the score toward chance. Excluding them raises the
    estimate; the difference bounds the dilution.
  * head/body/tail of the two predicates being discriminated.

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CPA = _load("cprime_analysis")
MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
Mech = MECH.Mech


def _log(m: str = "") -> None:
    print(m, flush=True)


def cells(Gs, B, score: torch.Tensor, cap: int, seed: int,
          drop_same_instance: bool) -> List[Dict[str, Any]]:
    """One record per (group, predicate-pair) cell, with its stratifiers."""
    gen = torch.Generator().manual_seed(seed)
    # instance identity of each GT row: (image, subj_idx, obj_idx)
    inst: List[tuple] = []
    for i in range(B.n_images):
        for si, oi in zip(B.meta["gt_subj_idx"][i], B.meta["gt_obj_idx"][i]):
            inst.append((i, int(si), int(oi)))
    inst_id = {}
    iid = torch.tensor([inst_id.setdefault(t, len(inst_id)) for t in inst])

    out: List[Dict[str, Any]] = []
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        rows = torch.tensor(Gs.rows_of[p])
        yy = Gs.y[rows]
        gsize = len(rows)
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                ra, rb = rows[yy == a], rows[yy == b]
                if drop_same_instance:
                    sa_set = set(iid[ra].tolist())
                    keep_b = torch.tensor([int(x) not in sa_set
                                           for x in iid[rb].tolist()],
                                          dtype=torch.bool)
                    rb = rb[keep_b]
                    if len(rb) == 0:
                        continue
                if len(ra) > cap:
                    ra = ra[torch.randperm(len(ra), generator=gen)[:cap]]
                if len(rb) > cap:
                    rb = rb[torch.randperm(len(rb), generator=gen)[:cap]]
                sa = (score[ra, a] - score[ra, b]).double()
                sb = (score[rb, a] - score[rb, b]).double()
                out.append({"auc": WPD.auc(sa, sb), "w": len(ra) * len(rb),
                            "gsize": gsize, "a": a, "b": b,
                            "bucket_a": B.bucket_of[a], "bucket_b": B.bucket_of[b]})
    return out


def summarise(cc: List[Dict[str, Any]], boot: int = 200,
              seed: int = 1) -> Dict[str, Any]:
    if not cc:
        return {"n_cells": 0}
    v = torch.tensor([c["auc"] for c in cc], dtype=torch.float64)
    w = torch.tensor([c["w"] for c in cc], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                      for _ in range(boot)])
    lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                             dtype=torch.float64)).tolist()
    return {"n_cells": len(cc), "n_comparisons": int(w.sum()),
            "macro": float(v.mean()), "weighted": float((v * w).sum() / w.sum()),
            "ci95": [lo, hi]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p35_wprd_stratified/strat.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 104)
    _log("WPRD STRATIFIED BY PAIR SUPPORT -- CPU only, cache read-only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    dev = Gs.prior_is_constant(B.prior)
    _log(f"  images={B.n_images} rows={Gs.n:,} groups={Gs.G:,}   "
         f"[gate W1] prior within-group max dev {dev:.3e} "
         f"{'PASS' if dev < 1e-3 else 'FAIL'}")
    assert dev < 1e-3

    heads = {"text_head": B.fixed_ensemble(0.0),
             "classifier_head": B.fixed_ensemble(1.0),
             "prior_CONTROL": B.prior}
    res: Dict[str, Any] = {"tool": "wprd_stratified", "dump": args.dump,
                           "cap": args.cap, "gate_prior_dev": dev, "by_head": {}}

    BANDS = [(2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10 ** 9)]
    for hname, s in heads.items():
        _log(f"\n{'-'*104}\n  {hname}\n{'-'*104}")
        cc = cells(Gs, B, s, args.cap, 0, drop_same_instance=False)
        cc_ex = cells(Gs, B, s, args.cap, 0, drop_same_instance=True)
        allsum = summarise(cc, args.boot)
        exsum = summarise(cc_ex, args.boot)
        _log(f"  ALL cells                    macro {allsum['macro']:.4f} "
             f"CI [{allsum['ci95'][0]:.4f}, {allsum['ci95'][1]:.4f}]  "
             f"weighted {allsum['weighted']:.4f}  cells {allsum['n_cells']:,}")
        _log(f"  EXCLUDING same-instance rows macro {exsum['macro']:.4f} "
             f"CI [{exsum['ci95'][0]:.4f}, {exsum['ci95'][1]:.4f}]  "
             f"weighted {exsum['weighted']:.4f}  cells {exsum['n_cells']:,}")

        _log(f"\n  {'pair support (group size)':>28} {'cells':>8} {'macro':>8} "
             f"{'weighted':>9} {'95% CI':>20}")
        bands = {}
        for lo, hi in BANDS:
            sub = [c for c in cc if lo <= c["gsize"] <= hi]
            sm = summarise(sub, args.boot)
            lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
            bands[lab] = sm
            if sm["n_cells"]:
                _log(f"  {lab:>28} {sm['n_cells']:>8,} {sm['macro']:>8.4f} "
                     f"{sm['weighted']:>9.4f}   "
                     f"[{sm['ci95'][0]:.4f}, {sm['ci95'][1]:.4f}]")

        bk = {}
        _log(f"\n  {'predicate buckets compared':>28} {'cells':>8} {'macro':>8} {'95% CI':>22}")
        for key in ("head-head", "head-body", "head-tail", "body-body",
                    "body-tail", "tail-tail"):
            x, y2 = key.split("-")
            sub = [c for c in cc
                   if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([x, y2])]
            sm = summarise(sub, args.boot)
            bk[key] = sm
            if sm["n_cells"]:
                _log(f"  {key:>28} {sm['n_cells']:>8,} {sm['macro']:>8.4f}   "
                     f"[{sm['ci95'][0]:.4f}, {sm['ci95'][1]:.4f}]")
        res["by_head"][hname] = {"all": allsum, "excl_same_instance": exsum,
                                 "by_support": bands, "by_bucket": bk}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
