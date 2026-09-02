#!/usr/bin/env python
"""p60 -- estimator-matched geometry vs rel_feat.

Pre-registered in docs/ESTIMATOR_MATCHED_GEOMETRY_PREREGISTRATION.md, committed
before this file existed. Thresholds are quoted from it and NOT recomputed here.

  PRIMARY   delta_est = B_geometry - A_relfeat
    GEOMETRY-ABOVE  >= +0.020 · COMPARABLE |.| < 0.020 · GEOMETRY-BELOW <= -0.020
  SECONDARY regime_gap = 0.5961 - B_geometry;  REGIME-SENSITIVE if >= 0.020
  TERTIARY  delta_fuse = C_fusion - max(A, B);  COMPLEMENTARY if >= +0.020

THE POINT. G = 0.5961 was fitted on 1,046,427 TRAIN rows with LBFGS softmax CE;
every rel_feat probe it is contrasted against was cross-fitted on 132,556
VALIDATION rows. This run puts geometry through the IDENTICAL estimator and the
IDENTICAL folds as p55's A_ce_all, so the two become comparable for the first
time. The estimator is p55's fit_ce, imported and reused unmodified.

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
CSP = _load("candidate_scorer_probe")
OBJ = _load("objective_ablation_relfeat")      # p55: the estimator lives here
Mech = MECH.Mech
N_FOLDS = 5

# Anchors. DIFFERENT REGIMES. Quoted, never contrasted arm-to-arm.
G_TRAIN_FITTED_LINEAR = 0.5961     # p37/p39 R8, train-fitted LBFGS CE
G_TRAIN_FITTED_MLP    = 0.6149     # p56 geometry_mlp, train-fitted
RIDGE_CV_GEOM         = 0.5655     # p58, gate-failed, not citable
RIDGE_CV_RELFEAT      = 0.5600     # p58, gate-failed, not citable
P55_A_CE_ALL          = 0.5732     # p55 A_ce_all -- gate G4 target


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p60_estimator_matched_geometry/est.json")
    ap.add_argument("--hidden", type=int, default=256)   # p55 defaults, unchanged
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--expect-folds", default="26483,26856,27190,26586,25441")
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p60 ESTIMATOR-MATCHED GEOMETRY vs rel_feat -- CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    Gs = WPD.Groups(B)
    fold = OBJ.folds_of(B, 0)
    y = B.gt_y
    C = B.n_classes
    gates: List[Dict[str, Any]] = []

    sizes = [int((fold == f).sum()) for f in range(N_FOLDS)]
    expect = [int(v) for v in args.expect_folds.split(",")] if args.expect_folds else None
    g3 = (expect is None) or (sizes == expect)
    gates.append({"gate": "G3 folds as registered", "pass": bool(g3),
                  "detail": f"{sizes} vs {expect}"})
    _log(f"  G3 folds {sizes}  {'PASS' if g3 else 'FAIL'}")

    # ---- features, standardised + bias, exactly as p55 does it ----
    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    assert RF.shape[0] == B.n_gt and not bool(torch.isnan(RF).any())
    Xr = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    Xr = torch.cat([Xr, torch.ones(Xr.shape[0], 1)], 1)
    del RF

    Graw = GEO.geometry_features_raw(B)
    Xg, _ = GEO._standardise(Graw)          # already appends the bias column
    Xf = torch.cat([Xr, Xg], 1)
    _log(f"  rows {B.n_gt:,}  relfeat {tuple(Xr.shape)}  geom {tuple(Xg.shape)}  "
         f"fusion {tuple(Xf.shape)}")

    scores: Dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(0)

    def linear(d_in: int, hidden: int, d_out: int, seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)
        return torch.nn.Linear(d_in, d_out)

    def fit(name: str, feats: torch.Tensor, labels: torch.Tensor,
            make=OBJ.mlp) -> None:
        """p55's fit_ce, inlined verbatim in its no-restrict / no-balance form.
        `make` swaps only the architecture, for arm D."""
        out = torch.zeros(feats.shape[0], C)
        for f in range(N_FOLDS):
            te = fold == f
            tr = torch.nonzero(~te, as_tuple=True)[0]
            net = make(feats.shape[1], args.hidden, C, 0)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                    weight_decay=args.l2)
            Xtr, ytr = feats[tr], labels[tr]
            n, bs = Xtr.shape[0], 8192
            for _ in range(args.epochs):
                perm = torch.randperm(n, generator=gen)
                for i in range(0, n, bs):
                    ix = perm[i:i + bs]
                    opt.zero_grad()
                    torch.nn.functional.cross_entropy(net(Xtr[ix]), ytr[ix]).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                out[te] = net(feats[te])
        scores[name] = out
        _log(f"    fitted {name}")

    _log(f"\n  fitting (cross-fitted, {N_FOLDS} folds, hidden={args.hidden}, "
         f"epochs={args.epochs}, lr={args.lr}, l2={args.l2})")
    fit("A_relfeat", Xr, y)
    fit("B_geometry", Xg, y)
    fit("C_fusion", Xf, y)
    fit("D_geometry_linear", Xg, y, make=linear)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    fit("N_shuffled", Xr, ysh)
    scores["P_prior"] = B.prior
    scores["ref_text_head"] = B.fixed_ensemble(0.0)
    scores["ref_classifier_head"] = B.fixed_ensemble(1.0)

    g5 = all(bool(torch.isfinite(v).all()) and v.shape[0] == B.n_gt
             for v in scores.values())
    gates.append({"gate": "G5 every arm scores every row", "pass": bool(g5),
                  "detail": f"{len(scores)} arms x {B.n_gt} rows"})

    res: Dict[str, Any] = {
        "tool": "estimator_matched_geometry",
        "anchors_DIFFERENT_REGIME": {
            "train_fitted_geometry_linear": G_TRAIN_FITTED_LINEAR,
            "train_fitted_geometry_mlp": G_TRAIN_FITTED_MLP,
            "ridge_cv_geometry_p58_GATE_FAILED": RIDGE_CV_GEOM,
            "ridge_cv_relfeat_p58_GATE_FAILED": RIDGE_CV_RELFEAT,
        },
        "estimator": {"hidden": args.hidden, "epochs": args.epochs,
                      "lr": args.lr, "l2": args.l2, "loss": "softmax CE",
                      "opt": "AdamW", "regime": "5-fold CV on validation, salt 0"},
        "gates": gates, "arms": {}}

    _log(f"\n{'-'*104}")
    _log(f"  {'arm':>22} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
         f"{'head':>7} {'body':>7} {'tail':>7}")
    _log(f"{'-'*104}")
    for name, sc in scores.items():
        # identical to p55's reporting path, so G4 compares like with like
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
        _log(f"  {name:>22} {r['wprd_macro']:8.4f} {r['wprd_weighted']:9.4f} "
             f"[{lo:.4f},{hi:.4f}] "
             f"{bb['head-head']:7.4f} {bb['body-body']:7.4f} {bb['tail-tail']:7.4f}")

    A = res["arms"]["A_relfeat"]["macro"]
    Bg = res["arms"]["B_geometry"]["macro"]
    Cf = res["arms"]["C_fusion"]["macro"]
    pr = res["arms"]["P_prior"]["macro"]
    sh = res["arms"]["N_shuffled"]["macro"]

    g1 = abs(pr - 0.5) < 1e-6
    g2 = 0.49 <= sh <= 0.51
    g4 = abs(A - P55_A_CE_ALL) < 0.005
    gates.insert(0, {"gate": "G1 prior exactly 0.5000", "pass": bool(g1),
                     "detail": f"{pr:.6f}"})
    gates.insert(1, {"gate": "G2 shuffled at chance", "pass": bool(g2),
                     "detail": f"{sh:.4f}"})
    gates.insert(2, {"gate": "G4 A_relfeat reproduces p55 A_ce_all 0.5732 +-0.005",
                     "pass": bool(g4), "detail": f"{A:.4f} (delta {A-P55_A_CE_ALL:+.4f})"})

    d_est = Bg - A
    regime_gap = G_TRAIN_FITTED_LINEAR - Bg
    d_fuse = Cf - max(A, Bg)
    primary = ("GEOMETRY-ABOVE" if d_est >= 0.020 else
               "GEOMETRY-BELOW" if d_est <= -0.020 else "COMPARABLE")
    secondary = "REGIME-SENSITIVE" if regime_gap >= 0.020 else "REGIME-STABLE"
    tertiary = "COMPLEMENTARY" if d_fuse >= 0.020 else "REDUNDANT"
    allpass = all(g["pass"] for g in gates)

    res["verdict"] = {"delta_est": d_est, "regime_gap": regime_gap,
                      "delta_fuse": d_fuse, "primary": primary,
                      "secondary": secondary, "tertiary": tertiary,
                      "gates_all_pass": bool(allpass)}

    _log(f"\n  PRIMARY   delta_est  = B_geometry - A_relfeat = {d_est:+.4f}  -> {primary}")
    _log(f"  SECONDARY regime_gap = 0.5961 - B_geometry    = {regime_gap:+.4f}  -> {secondary}")
    _log(f"  TERTIARY  delta_fuse = C - max(A,B)           = {d_fuse:+.4f}  -> {tertiary}")
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
