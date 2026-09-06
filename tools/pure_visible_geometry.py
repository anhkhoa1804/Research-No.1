#!/usr/bin/env python
"""p69 -- is the geometry PURE cannot see worth anything?

Pre-registered in docs/PURE_VISIBLE_GEOMETRY_PREREGISTRATION.md, committed
before this file's first run. Thresholds are quoted from it and NOT
recomputed here. Does not modify tools/estimator_matched_geometry.py (p60) or
tools/estimator_matched_geometry_v2.py (p65); those artifacts stand as
reported. This tool reuses their exact estimator, folds and reporting path so
every number is comparable to p60's by construction.

  PRIMARY delta_missing = B_geometry - P_pure_visible
    INERT (no GPU) < +0.01 . AMBIGUOUS [+0.01,+0.03) . MATERIAL >= +0.03

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
OBJ = _load("objective_ablation_relfeat")
Mech = MECH.Mech
N_FOLDS = 5

P60_A_RELFEAT = 0.5732
P60_B_GEOMETRY = 0.5976

# Column indices into tools/wprd_geometry_control.py::_geom's 19-number stack.
# Verified by reading that function's construction order:
#   0..3  scx/W, scy/H, ocx/W, ocy/H          absolute positions
#   4,5   (ocx-scx)/(sw+ow), (ocy-scy)/(sh+oh)  offset, PAIR-BOX scale  [dx_rel/dy_rel]
#   6,7   (ocx-scx)/W, (ocy-scy)/H             offset, IMAGE scale      [what PURE sees]
#   8..11 sw/W, sh/H, ow/W, oh/H               sizes
#   12    log(sa/oa) . 13,14 aspect . 15 IoU . 16,17 containment . 18 distance
COLS_PURE_VISIBLE = [6, 7]
COLS_SIZES = [8, 9, 10, 11]


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p69_pure_visible_geometry/est.json")
    ap.add_argument("--hidden", type=int, default=256)   # p60 defaults, unchanged
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--expect-folds", default="26483,26856,27190,26586,25441")
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p69 PURE-VISIBLE GEOMETRY, matched estimator -- CPU only, NO GPU")
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

    def subset(cols: List[int]) -> torch.Tensor:
        """Standardise a column subset the same way, keeping exactly one bias."""
        std, _ = GEO._standardise(Graw[:, cols])
        return std   # (n, len(cols)+1), bias already appended

    Xp = subset(COLS_PURE_VISIBLE)
    Xq = subset(COLS_PURE_VISIBLE + COLS_SIZES)
    _log(f"  rows {B.n_gt:,}  relfeat {tuple(Xr.shape)}  geom19 {tuple(Xg.shape)}  "
         f"pure_visible {tuple(Xp.shape)}  visible+sizes {tuple(Xq.shape)}")

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
    fit("P_pure_visible", Xp, y)
    fit("Q_visible_plus_sizes", Xq, y)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    fit("N_shuffled", Xr, ysh)
    scores["P_prior"] = B.prior

    g6 = all(bool(torch.isfinite(v).all()) and v.shape[0] == B.n_gt
             for v in scores.values())
    gates.append({"gate": "G6 every arm scores every row", "pass": bool(g6),
                  "detail": f"{len(scores)} arms x {B.n_gt} rows"})

    res: Dict[str, Any] = {
        "tool": "pure_visible_geometry",
        "prereg": "docs/PURE_VISIBLE_GEOMETRY_PREREGISTRATION.md",
        "anchors_p60": {"A_relfeat": P60_A_RELFEAT, "B_geometry": P60_B_GEOMETRY},
        "columns": {"pure_visible": COLS_PURE_VISIBLE, "sizes": COLS_SIZES},
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
    P = res["arms"]["P_pure_visible"]["macro"]
    Q = res["arms"]["Q_visible_plus_sizes"]["macro"]
    pr = res["arms"]["P_prior"]["macro"]
    sh = res["arms"]["N_shuffled"]["macro"]

    g1 = abs(pr - 0.5) < 1e-6
    g2 = 0.49 <= sh <= 0.51
    g4 = abs(A - P60_A_RELFEAT) < 0.005
    g5 = abs(Bg - P60_B_GEOMETRY) < 0.005
    gates.insert(0, {"gate": "G4 prior exactly 0.5000", "pass": bool(g1),
                     "detail": f"{pr:.6f}"})
    gates.insert(1, {"gate": "G5 shuffled at chance", "pass": bool(g2),
                     "detail": f"{sh:.4f}"})
    gates.insert(2, {"gate": "G1 A_relfeat reproduces p60 0.5732 +-0.005",
                     "pass": bool(g4), "detail": f"{A:.4f} (delta {A-P60_A_RELFEAT:+.4f})"})
    gates.insert(3, {"gate": "G2 B_geometry reproduces p60 0.5976 +-0.005",
                     "pass": bool(g5), "detail": f"{Bg:.4f} (delta {Bg-P60_B_GEOMETRY:+.4f})"})

    d_missing = Bg - P
    d_size = Q - P
    primary = ("MATERIAL" if d_missing >= 0.03 else
               "INERT" if d_missing < 0.01 else "AMBIGUOUS")
    allpass = all(g["pass"] for g in gates)

    res["verdict"] = {"delta_missing": d_missing, "delta_size": d_size,
                      "primary": primary, "gates_all_pass": bool(allpass)}

    _log(f"\n  PRIMARY   delta_missing = B_geometry - P_pure_visible = {d_missing:+.4f}  -> {primary}")
    _log(f"  SECONDARY delta_size     = Q_visible_plus_sizes - P    = {d_size:+.4f}")
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
