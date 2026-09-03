#!/usr/bin/env python
"""p67 -- auxiliary dx_rel/dy_rel regression loss on the frozen-rel_feat readout.

Pre-registered in docs/AUXILIARY_SPATIAL_LOSS_PREREGISTRATION.md, committed
before this file's first run. Thresholds quoted from it, not recomputed here.

  delta_aux = max(WPRD at lambda in {0.5,1.0,2.0}) - WPRD at lambda=0.0
    AUX-LOSS-WORKS    >= +0.02
    AUX-LOSS-NEUTRAL  |.| < 0.02
    AUX-LOSS-HARMFUL  <= -0.02

A single shared-trunk MLP feeds a classification head (predicate CE, the
existing objective) and a regression head (MSE against dx_rel, dy_rel).
lambda sweeps a small fixed grid; none is selected by WPRD.

CPU only. Reads the p36 cache. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
STRAT = _load("wprd_stratified")
GEO = _load("wprd_geometry_control")
DEC = _load("geometry_decodability")
Mech = MECH.Mech
N_FOLDS = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


class DualHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_classes: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.cls_head = nn.Linear(hidden, n_classes)
        self.reg_head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.trunk(x)
        return self.cls_head(h), self.reg_head(h)


def fit_dualhead(X: torch.Tensor, y: torch.Tensor, spatial: torch.Tensor,
                 fold: torch.Tensor, C: int, hidden: int, epochs: int,
                 lr: float, l2: float, lam: float, seed: int) -> torch.Tensor:
    out = torch.zeros(X.shape[0], C)
    gen = torch.Generator().manual_seed(seed)
    for f in range(N_FOLDS):
        te = fold == f
        tr = torch.nonzero(~te, as_tuple=True)[0]
        torch.manual_seed(seed)
        net = DualHead(X.shape[1], hidden, C)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
        Xtr, ytr, str_ = X[tr], y[tr], spatial[tr]
        n, bs = Xtr.shape[0], 8192
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, bs):
                ix = perm[i:i + bs]
                opt.zero_grad()
                logits, reg = net(Xtr[ix])
                loss = torch.nn.functional.cross_entropy(logits, ytr[ix])
                if lam > 0:
                    loss = loss + lam * torch.nn.functional.mse_loss(reg, str_[ix])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            logits, _ = net(X[te])
            out[te] = logits
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p67_auxiliary_spatial_loss/aux.json")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--lambdas", default="0.0,0.5,1.0,2.0")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    lambdas = [float(x) for x in args.lambdas.split(",")]

    _log("=" * 108)
    _log("p67 AUXILIARY dx_rel/dy_rel LOSS ON THE FROZEN-rel_feat READOUT -- CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    Gs = WPD.Groups(B)
    fold = DEC.folds_of(B, 0)
    y = B.gt_y
    C = B.n_classes
    gates: List[Dict[str, Any]] = []

    sizes = [int((fold == f).sum()) for f in range(N_FOLDS)]
    w1 = sizes == [26483, 26856, 27190, 26586, 25441]
    gates.append({"gate": "W1 folds as registered", "pass": bool(w1), "detail": str(sizes)})
    _log(f"  W1 folds {sizes} {'PASS' if w1 else 'FAIL'}")

    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    X = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], 1)

    Graw = GEO.geometry_features_raw(B)
    Xg_all, _ = GEO._standardise(Graw)
    spatial = Xg_all[:, 4:6]   # standardised dx_rel, dy_rel

    scores: Dict[str, torch.Tensor] = {}
    for lam in lambdas:
        name = f"lambda_{lam}"
        scores[name] = fit_dualhead(X, y, spatial, fold, C, args.hidden,
                                    args.epochs, args.lr, args.l2, lam, seed=0)
        _log(f"    fitted {name}")
    scores["P_prior"] = B.prior
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    scores["N_shuffled"] = fit_dualhead(X, ysh, spatial, fold, C, args.hidden,
                                        args.epochs, args.lr, args.l2, 0.0, seed=0)

    res: Dict[str, Any] = {
        "tool": "auxiliary_spatial_loss", "lambdas": lambdas,
        "estimator": {"hidden": args.hidden, "epochs": args.epochs,
                      "lr": args.lr, "l2": args.l2, "loss": "CE + lambda*MSE(dx_rel,dy_rel)"},
        "gates": gates, "arms": {}}

    _log(f"\n{'-'*104}")
    _log(f"  {'arm':>16} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
         f"{'head':>7} {'body':>7} {'tail':>7}")
    _log(f"{'-'*104}")
    for name, sc in scores.items():
        r = WPD.wprd(Gs, sc, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        bs_ = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                           for _ in range(args.boot)])
        lo, hi = torch.quantile(bs_, torch.tensor([0.025, 0.975],
                                                  dtype=torch.float64)).tolist()
        cc = STRAT.cells(Gs, B, sc, args.cap, 0, drop_same_instance=False)
        bb = {}
        for key in ("head-head", "body-body", "tail-tail"):
            x1, x2 = key.split("-")
            sub = [c["auc"] for c in cc
                   if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([x1, x2])]
            bb[key] = sum(sub) / len(sub) if sub else float("nan")
        res["arms"][name] = {"macro": r["wprd_macro"], "weighted": r["wprd_weighted"],
                             "ci95": [lo, hi], "n_cells": r["n_cells"], "by_bucket": bb}
        _log(f"  {name:>16} {r['wprd_macro']:8.4f} {r['wprd_weighted']:9.4f} "
             f"[{lo:.4f},{hi:.4f}] "
             f"{bb['head-head']:7.4f} {bb['body-body']:7.4f} {bb['tail-tail']:7.4f}")

    base = res["arms"]["lambda_0.0"]["macro"]
    pr = res["arms"]["P_prior"]["macro"]
    sh = res["arms"]["N_shuffled"]["macro"]
    w2 = abs(base - 0.5732) < 0.005
    w3 = abs(pr - 0.5) < 1e-6
    w4 = 0.49 <= sh <= 0.51
    gates.insert(0, {"gate": "W2 lambda=0.0 reproduces p60 A_relfeat 0.5732 +-0.005",
                     "pass": bool(w2), "detail": f"{base:.4f}"})
    gates.insert(1, {"gate": "W3 prior exactly 0.5000", "pass": bool(w3), "detail": f"{pr:.6f}"})
    gates.insert(2, {"gate": "W4 shuffled at chance", "pass": bool(w4), "detail": f"{sh:.4f}"})

    best_lam = max([l for l in lambdas if l > 0],
                   key=lambda l: res["arms"][f"lambda_{l}"]["macro"])
    best_wprd = res["arms"][f"lambda_{best_lam}"]["macro"]
    delta_aux = best_wprd - base
    verdict = ("AUX-LOSS-WORKS" if delta_aux >= 0.020 else
               "AUX-LOSS-HARMFUL" if delta_aux <= -0.020 else "AUX-LOSS-NEUTRAL")
    allpass = all(g["pass"] for g in gates)
    res["verdict"] = {"best_lambda": best_lam, "delta_aux": delta_aux,
                      "verdict": verdict, "gates_all_pass": bool(allpass)}

    _log(f"\n  delta_aux = WPRD(lambda={best_lam}) [{best_wprd:.4f}] - "
         f"WPRD(lambda=0) [{base:.4f}] = {delta_aux:+.4f}  -> {verdict}")
    _log(f"\n  gates: " + "  ".join(f"{g['gate'].split()[0]}={'PASS' if g['pass'] else 'FAIL'}"
                                    for g in gates))
    if not allpass:
        _log("  *** A GATE FAILED. By the registration, NO NUMBER HERE IS REPORTABLE. ***")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
