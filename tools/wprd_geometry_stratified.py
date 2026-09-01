#!/usr/bin/env python
"""WHERE does box geometry beat the trained model? Per-bucket and per-predicate.

runs/p39 established that a train-fitted probe on 19 box-geometry features
outscores both of the checkpoint's predicate heads on prior-free within-pair
relational discrimination (0.5961 vs 0.5728 / 0.5542). That is an aggregate.
A successor model needs to know WHERE the loss is, and whether there is any
region at all in which the trained model beats rectangles.

Reports, on the identical cells:
  * per predicate bucket (head/body/tail combinations)
  * the per-predicate-pair cells where the model is FURTHEST BEHIND geometry
  * the cells, if any, where the model is AHEAD

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


MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
GEO = _load("wprd_geometry_control")
STRAT = _load("wprd_stratified")
Mech = MECH.Mech


def _log(m: str = "") -> None:
    print(m, flush=True)


def boot_ci(v: torch.Tensor, n: int = 200, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                      for _ in range(n)])
    return torch.quantile(bs, torch.tensor([0.025, 0.975],
                                           dtype=torch.float64)).tolist()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p40_geometry_stratified/strat.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    _log("=" * 104)
    _log("WHERE DOES GEOMETRY BEAT THE MODEL? -- CPU only, cache read-only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    dev = Gs.prior_is_constant(B.prior)
    _log(f"  rows {Gs.n:,} groups {Gs.G:,}  [gate W1] prior dev {dev:.3e} "
         f"{'PASS' if dev < 1e-3 else 'FAIL'}")
    assert dev < 1e-3

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tstats = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tstats)
    _log(f"  geometry fitted on TRAIN: {Xt.shape[0]:,} rows -> applied to "
         f"{Xv.shape[0]:,} val rows (no val statistic in the fit)")
    W = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=args.epochs, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W, yt)
                + args.l2 * (W * W).sum())
        loss.backward()
        return loss

    opt.step(closure)
    geo = (Xv @ W).detach()

    heads = {"text": B.fixed_ensemble(0.0), "cls": B.fixed_ensemble(1.0),
             "geom": geo}
    # per-cell records with stratifiers. Same Groups object, same cap, same
    # seed -> identical cell order across arms, which is what makes the paired
    # differences below meaningful.
    recs = STRAT.cells(Gs, B, geo, args.cap, 0, drop_same_instance=False)
    rt = STRAT.cells(Gs, B, heads["text"], args.cap, 0, drop_same_instance=False)
    rc = STRAT.cells(Gs, B, heads["cls"], args.cap, 0, drop_same_instance=False)
    assert len(recs) == len(rt) == len(rc)

    res: Dict[str, Any] = {"tool": "wprd_geometry_stratified", "by_bucket": {},
                           "train_rows": int(Xt.shape[0])}
    _log(f"\n{'-'*104}")
    _log(f"  {'buckets':>14} {'cells':>7} {'geom':>8} {'cls':>8} {'text':>8} "
         f"{'geom-cls':>10} {'geom-text':>10}")
    _log(f"{'-'*104}")
    for key in ("head-head", "head-body", "head-tail", "body-body",
                "body-tail", "tail-tail"):
        x, y2 = key.split("-")
        ix = [i for i, c in enumerate(recs)
              if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([x, y2])]
        if not ix:
            continue
        g = torch.tensor([recs[i]["auc"] for i in ix], dtype=torch.float64)
        tt = torch.tensor([rt[i]["auc"] for i in ix], dtype=torch.float64)
        cl = torch.tensor([rc[i]["auc"] for i in ix], dtype=torch.float64)
        dgc, dgt = (g - cl), (g - tt)
        lo1, hi1 = boot_ci(dgc)
        lo2, hi2 = boot_ci(dgt)
        res["by_bucket"][key] = {
            "n_cells": len(ix), "geom": float(g.mean()), "cls": float(cl.mean()),
            "text": float(tt.mean()),
            "geom_minus_cls": [float(dgc.mean()), lo1, hi1],
            "geom_minus_text": [float(dgt.mean()), lo2, hi2]}
        m1 = "*" if (lo1 > 0 or hi1 < 0) else " "
        m2 = "*" if (lo2 > 0 or hi2 < 0) else " "
        _log(f"  {key:>14} {len(ix):>7,} {float(g.mean()):>8.4f} "
             f"{float(cl.mean()):>8.4f} {float(tt.mean()):>8.4f} "
             f"{float(dgc.mean()):>+9.4f}{m1} {float(dgt.mean()):>+9.4f}{m2}")
    _log("  * = bootstrap 95% CI on the paired difference excludes 0")

    # per predicate-pair aggregation
    agg: Dict[str, List[float]] = defaultdict(list)
    for i, c in enumerate(recs):
        agg[f"{B.classes[c['a']]} vs {B.classes[c['b']]}"].append(c["auc"] - rt[i]["auc"])
    rows = [(sum(v) / len(v), len(v), k) for k, v in agg.items() if len(v) >= 30]
    rows.sort()
    _log(f"\n  predicate pairs where GEOMETRY most exceeds the EVALUATED head "
         f"(>=30 cells)")
    for d, n, k in rows[-12:][::-1]:
        _log(f"    {k:>34} {d:>+8.4f}  ({n} cells)")
    _log(f"\n  predicate pairs where the EVALUATED head most exceeds GEOMETRY")
    for d, n, k in rows[:8]:
        _log(f"    {k:>34} {d:>+8.4f}  ({n} cells)")
    res["per_predicate_pair_geom_minus_text"] = {k: [d, n] for d, n, k in rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
