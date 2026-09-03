#!/usr/bin/env python
"""p65 -- minimal dx_rel/dy_rel fix and cleaned-fusion, under the matched estimator.

Pre-registered in docs/MINIMAL_FIX_AND_CLEAN_FUSION_PREREGISTRATION.md,
committed before this file's first run. Thresholds are quoted from it and NOT
recomputed here. Does not modify tools/estimator_matched_geometry.py (p60);
that artifact stands as reported. This tool reuses its exact estimator,
folds, and reporting path so every number is comparable to p60's by
construction.

  PRIMARY   delta_min        = D2_relfeat_plus_dxdy - A_relfeat
    MINIMAL-FIX-WORKS >= +0.02 · NEUTRAL |.| < 0.02 · HARMFUL <= -0.02
  SECONDARY delta_clean_fuse = E_groupcentered_relfeat_plus_geometry - B_geometry
    CLEAN-FUSION-GAIN >= +0.02 · NEUTRAL |.| < 0.02 · HARMFUL <= -0.02

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
OBJ = _load("objective_ablation_relfeat")
Mech = MECH.Mech
N_FOLDS = 5

P60_A_RELFEAT = 0.5732
P60_B_GEOMETRY = 0.5976


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p65_minimal_fix_clean_fusion/est.json")
    ap.add_argument("--hidden", type=int, default=256)   # p60 defaults, unchanged
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--expect-folds", default="26483,26856,27190,26586,25441")
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p65 MINIMAL FIX + CLEANED FUSION, matched estimator -- CPU only, NO GPU")
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

    # ---- rel_feat, standardised exactly as p60 does it ----
    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    assert RF.shape[0] == B.n_gt and not bool(torch.isnan(RF).any())
    RFmean, RFstd = RF.mean(0, keepdim=True), RF.std(0, keepdim=True).clamp_min(1e-6)
    Xr_nobias = (RF - RFmean) / RFstd
    Xr = torch.cat([Xr_nobias, torch.ones(Xr_nobias.shape[0], 1)], 1)

    # ---- geometry, raw and standardised, exactly as p60/p37/p57 do it ----
    Graw = GEO.geometry_features_raw(B)
    Xg, _ = GEO._standardise(Graw)          # 19 standardised + bias
    assert Xg.shape[1] == 20

    # dx_rel, dy_rel are columns 4, 5 of _geom's construction order
    # (ocx-scx)/(sw+ow), (ocy-scy)/(sh+oh) -- immediately after the four
    # absolute positions. Verified by reading tools/wprd_geometry_control.py.
    dxdy_raw = Graw[:, 4:6]
    dxdy_std, _ = GEO._standardise(dxdy_raw)   # 2 standardised + bias (drop the extra bias col)
    dxdy = dxdy_std[:, :2]

    Xd2 = torch.cat([Xr_nobias, dxdy, torch.ones(Xr_nobias.shape[0], 1)], 1)
    _log(f"  rows {B.n_gt:,}  relfeat {tuple(Xr.shape)}  geom {tuple(Xg.shape)}  "
         f"D2(relfeat+dxdy) {tuple(Xd2.shape)}")

    # ---- group-centred rel_feat, same convention as p37's R5 ----
    gid, G = Gs.pair_id, Gs.G
    cnt = torch.zeros(G).index_add_(0, gid, torch.ones(Xr_nobias.shape[0]))
    s = torch.zeros(G, Xr_nobias.shape[1]).index_add_(0, gid, Xr_nobias)
    gmean = (s / cnt.clamp_min(1).unsqueeze(1))[gid]
    RFc_nobias = Xr_nobias - gmean
    Xe = torch.cat([RFc_nobias, Xg], 1)   # group-centred relfeat + full standardised geometry (has its own bias)
    _log(f"  groupcentered-relfeat {tuple(RFc_nobias.shape)}  "
         f"E(groupcentered_relfeat+geom) {tuple(Xe.shape)}")

    scores: Dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(0)

    def fit(name: str, feats: torch.Tensor, labels: torch.Tensor) -> None:
        out = torch.zeros(feats.shape[0], C)
        for f in range(N_FOLDS):
            te = fold == f
            tr = torch.nonzero(~te, as_tuple=True)[0]
            net = OBJ.mlp(feats.shape[1], args.hidden, C, 0)
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
    fit("D2_relfeat_plus_dxdy", Xd2, y)
    fit("E_groupcentered_relfeat_plus_geometry", Xe, y)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    fit("N_shuffled", Xr, ysh)
    scores["P_prior"] = B.prior

    g6 = all(bool(torch.isfinite(v).all()) and v.shape[0] == B.n_gt
             for v in scores.values())
    gates.append({"gate": "G6 every arm scores every row", "pass": bool(g6),
                  "detail": f"{len(scores)} arms x {B.n_gt} rows"})

    res: Dict[str, Any] = {
        "tool": "estimator_matched_geometry_v2",
        "anchors_p60": {"A_relfeat": P60_A_RELFEAT, "B_geometry": P60_B_GEOMETRY},
        "estimator": {"hidden": args.hidden, "epochs": args.epochs,
                      "lr": args.lr, "l2": args.l2, "loss": "softmax CE",
                      "opt": "AdamW", "regime": "5-fold CV on validation, salt 0"},
        "gates": gates, "arms": {}}

    _log(f"\n{'-'*104}")
    _log(f"  {'arm':>38} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
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
        _log(f"  {name:>38} {r['wprd_macro']:8.4f} {r['wprd_weighted']:9.4f} "
             f"[{lo:.4f},{hi:.4f}] "
             f"{bb['head-head']:7.4f} {bb['body-body']:7.4f} {bb['tail-tail']:7.4f}")

    A = res["arms"]["A_relfeat"]["macro"]
    Bg = res["arms"]["B_geometry"]["macro"]
    D2 = res["arms"]["D2_relfeat_plus_dxdy"]["macro"]
    E = res["arms"]["E_groupcentered_relfeat_plus_geometry"]["macro"]
    pr = res["arms"]["P_prior"]["macro"]
    sh = res["arms"]["N_shuffled"]["macro"]

    g1 = abs(pr - 0.5) < 1e-6
    g2 = 0.49 <= sh <= 0.51
    g4 = abs(A - P60_A_RELFEAT) < 0.005
    g5 = abs(Bg - P60_B_GEOMETRY) < 0.005
    gates.insert(0, {"gate": "G1 prior exactly 0.5000", "pass": bool(g1),
                     "detail": f"{pr:.6f}"})
    gates.insert(1, {"gate": "G2 shuffled at chance", "pass": bool(g2),
                     "detail": f"{sh:.4f}"})
    gates.insert(2, {"gate": "G4 A_relfeat reproduces p60 0.5732 +-0.005",
                     "pass": bool(g4), "detail": f"{A:.4f} (delta {A-P60_A_RELFEAT:+.4f})"})
    gates.insert(3, {"gate": "G5 B_geometry reproduces p60 0.5976 +-0.005",
                     "pass": bool(g5), "detail": f"{Bg:.4f} (delta {Bg-P60_B_GEOMETRY:+.4f})"})

    d_min = D2 - A
    d_clean = E - Bg
    primary = ("MINIMAL-FIX-WORKS" if d_min >= 0.020 else
               "MINIMAL-FIX-HARMFUL" if d_min <= -0.020 else "MINIMAL-FIX-NEUTRAL")
    secondary = ("CLEAN-FUSION-GAIN" if d_clean >= 0.020 else
                 "CLEAN-FUSION-HARMFUL" if d_clean <= -0.020 else "CLEAN-FUSION-NEUTRAL")
    allpass = all(g["pass"] for g in gates)

    res["verdict"] = {"delta_min": d_min, "delta_clean_fuse": d_clean,
                      "primary": primary, "secondary": secondary,
                      "gates_all_pass": bool(allpass)}

    _log(f"\n  PRIMARY   delta_min        = D2 - A_relfeat = {d_min:+.4f}  -> {primary}")
    _log(f"  SECONDARY delta_clean_fuse = E - B_geometry  = {d_clean:+.4f}  -> {secondary}")
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
