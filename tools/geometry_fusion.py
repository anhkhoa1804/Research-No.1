#!/usr/bin/env python
"""p58 -- matched-fitting fusion of rel_feat and box geometry.

Pre-registered in docs/GEOMETRY_FUSION_PREREGISTRATION.md, committed before this
file existed. Thresholds are quoted from it and NOT recomputed.

  PRIMARY   delta_fuse = C_fusion - B_geometry
    FUSION-GAIN     >= +0.02
    FUSION-NEUTRAL  |.| < 0.02
    FUSION-HARMFUL  <= -0.02
  SECONDARY delta_min = D_relfeat_plus_dxdy - A_relfeat   (MINIMAL-FIX at +0.02)
  TERTIARY  delta_dim = F_random_plus_geometry - B_geometry

p37 compared a 788-d probe cross-fitted on 132k validation rows against a 20-d
probe fitted with a DIFFERENT LOSS on 1.05M train rows, and read the difference
as a fact about the features. Here every arm shares one regime: same folds, same
estimator, same nested regularisation search, same preprocessing. Dimensionality
is the only thing that varies, and arm F isolates even that.

CPU only. Reads the p36 cache. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
Mech = MECH.Mech
N_FOLDS = 5
# Attempt 1 saturated against a 1e-4 floor on every arm and every fold
# (runs/p58_VOID_lambda_grid_too_narrow), so gate Y5 voided it. Widened
# downward. Y5 itself and every verdict threshold are unchanged.
LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
G_TRAIN_FITTED_ANCHOR = 0.5961        # p37/p39, DIFFERENT REGIME. Anchor only.

# indices into the 19 raw geometry features (see wprd_geometry_control._geom)
IDX_DX_REL, IDX_DY_REL = 4, 5


def _log(m: str = "") -> None:
    print(m, flush=True)


def folds_of(B: Mech, salt: int = 0) -> torch.Tensor:
    per = [CSP.fold_of_image(str(s), N_FOLDS, salt) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per[i]] * len(B.meta["pairs"][i]))
    return torch.tensor(flat)[B.gt_row]


def _gram(A: torch.Tensor, Y: torch.Tensor):
    """(A^T A, A^T Y, n). Depends only on the fit set, never on lambda, so it is
    computed ONCE per fit set and reused across the whole lambda grid. Without
    this the 788-dimensional arms recompute a 106k x 788 Gram five times per
    fold for no reason."""
    return A.T @ A, A.T @ Y, A.shape[0]


def _solve(AtA: torch.Tensor, AtY: torch.Tensor, n: int,
           lam: float) -> torch.Tensor:
    return torch.linalg.solve(
        AtA + lam * n * torch.eye(AtA.shape[0], dtype=AtA.dtype), AtY)


def _logloss(scores: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cross_entropy(scores, y))


def xfit_nested_ridge(X: torch.Tensor, y: torch.Tensor, fold: torch.Tensor,
                      C: int, seed: int = 0) -> Tuple[torch.Tensor, List[float]]:
    """Out-of-fold scores with the ridge strength chosen INSIDE each training
    fold by held-in multiclass log-loss. WPRD is never consulted."""
    out = torch.zeros(X.shape[0], C)
    chosen: List[float] = []
    g = torch.Generator().manual_seed(seed)
    for f in range(N_FOLDS):
        te = fold == f
        tr_idx = torch.nonzero(~te, as_tuple=True)[0]
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        cut = int(0.8 * len(perm))
        fit_i, val_i = perm[:cut], perm[cut:]
        Yfit = torch.zeros(len(fit_i), C)
        Yfit[torch.arange(len(fit_i)), y[fit_i]] = 1.0
        AtA, AtY, n_fit = _gram(X[fit_i], Yfit)
        Xval = X[val_i]
        best, best_lam = float("inf"), LAMBDAS[0]
        for lam in LAMBDAS:
            ll = _logloss(Xval @ _solve(AtA, AtY, n_fit, lam), y[val_i])
            if ll < best:
                best, best_lam = ll, lam
        Yall = torch.zeros(len(tr_idx), C)
        Yall[torch.arange(len(tr_idx)), y[tr_idx]] = 1.0
        AtA2, AtY2, n_all = _gram(X[tr_idx], Yall)
        out[te] = X[te] @ _solve(AtA2, AtY2, n_all, best_lam)
        chosen.append(best_lam)
    return out, chosen


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p58_geometry_fusion/fusion.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 110)
    _log("p58 MATCHED-FITTING FUSION -- one regime for every arm. CPU only, NO GPU")
    _log("=" * 110)

    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    Gs = WPD.Groups(B)
    fold = folds_of(B, 0)
    y = B.gt_y
    C = B.n_classes
    gates: List[Dict[str, Any]] = []

    sizes = [int((fold == f).sum()) for f in range(N_FOLDS)]
    y3 = sizes == [26483, 26856, 27190, 26586, 25441]
    gates.append({"gate": "Y3 folds identical to p37", "pass": bool(y3),
                  "detail": str(sizes)})
    _log(f"  Y3 folds {sizes} {'PASS' if y3 else 'FAIL'}")

    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    RFs = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    del RF
    ones = torch.ones(RFs.shape[0], 1)

    Xg_raw = GEO.geometry_features_raw(B)
    Xg_all, _ = GEO._standardise(Xg_raw)          # 19 features + appended bias
    keep = (Xg_all.std(0) > 1e-8).nonzero(as_tuple=True)[0]
    Xg = Xg_all[:, keep]                          # the 19 real features
    _log(f"  rel_feat {tuple(RFs.shape)}   geometry {tuple(Xg.shape)} "
         f"(bias column dropped, appended once per arm)")

    g = torch.Generator().manual_seed(21)
    sh_rows = torch.randperm(Xg.shape[0], generator=g)
    RND768 = torch.randn(RFs.shape[0], RFs.shape[1], generator=g)

    designs: Dict[str, torch.Tensor] = {
        "A_relfeat": torch.cat([RFs, ones], 1),
        "B_geometry": torch.cat([Xg, ones], 1),
        "C_fusion": torch.cat([RFs, Xg, ones], 1),
        "D_relfeat_plus_dxdy": torch.cat(
            [RFs, Xg[:, [IDX_DX_REL, IDX_DY_REL]], ones], 1),
        "E_relfeat_plus_shuffled_geom": torch.cat([RFs, Xg[sh_rows], ones], 1),
        "F_random_plus_geometry": torch.cat([RND768, Xg, ones], 1),
    }

    scores: Dict[str, torch.Tensor] = {}
    lambdas: Dict[str, List[float]] = {}
    _log(f"\n  fitting (nested ridge, lambda grid {LAMBDAS}, "
         f"selected by inner-fold log-loss)")
    for name, X in designs.items():
        s, lam = xfit_nested_ridge(X, y, fold, C)
        scores[name] = s
        lambdas[name] = lam
        _log(f"    {name:<30} dims {X.shape[1]:>4}   lambdas {lam}")

    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    s, lam = xfit_nested_ridge(designs["A_relfeat"], ysh, fold, C)
    scores["H_shuffled_labels"] = s
    lambdas["H_shuffled_labels"] = lam
    scores["G_prior"] = B.prior

    y4 = all(v.shape[0] == B.n_gt and bool(torch.isfinite(v).all())
             for v in scores.values())
    gates.append({"gate": "Y4 every arm scores every row", "pass": bool(y4),
                  "detail": f"{len(scores)} arms x {B.n_gt} rows"})
    ends = {LAMBDAS[0], LAMBDAS[-1]}
    pinned = [k for k, v in lambdas.items() if set(v) <= ends and len(set(v)) == 1]
    y5 = len(pinned) == 0
    gates.append({"gate": "Y5 no arm pinned to a grid endpoint on every fold",
                  "pass": bool(y5), "detail": f"pinned={pinned}"})

    res: Dict[str, Any] = {"tool": "geometry_fusion", "gates": gates,
                           "lambdas": lambdas,
                           "anchor_train_fitted_geometry_DIFFERENT_REGIME":
                               G_TRAIN_FITTED_ANCHOR, "arms": {}}
    _log(f"\n{'-'*110}")
    _log(f"  {'arm':>30} {'dims':>5} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
         f"{'head':>7} {'body':>7} {'tail':>7}")
    _log(f"{'-'*110}")
    for name, sc in scores.items():
        r = WPD.wprd(Gs, sc, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        gb = torch.Generator().manual_seed(1)
        bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=gb)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        cc = STRAT.cells(Gs, B, sc, args.cap, 0, drop_same_instance=False)
        bk = {}
        for key in ("head-head", "body-body", "tail-tail"):
            a1, a2 = key.split("-")
            sub = [c["auc"] for c in cc
                   if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([a1, a2])]
            bk[key] = sum(sub) / len(sub) if sub else float("nan")
        d = designs[name].shape[1] if name in designs else 0
        res["arms"][name] = {"macro": r["wprd_macro"], "weighted": r["wprd_weighted"],
                             "ci95": [lo, hi], "dims": d, "by_bucket": bk}
        _log(f"  {name:>30} {d:>5} {r['wprd_macro']:>8.4f} {r['wprd_weighted']:>9.4f} "
             f"[{lo:.4f},{hi:.4f}] {bk['head-head']:>7.4f} {bk['body-body']:>7.4f} "
             f"{bk['tail-tail']:>7.4f}")

    pr = res["arms"]["G_prior"]["macro"]
    shm = res["arms"]["H_shuffled_labels"]["macro"]
    y1 = abs(pr - 0.5) < 1e-6
    y2 = 0.49 <= shm <= 0.51
    gates.append({"gate": "Y1 prior exactly 0.5000", "pass": bool(y1),
                  "detail": f"{pr:.6f}"})
    gates.append({"gate": "Y2 shuffled labels at chance", "pass": bool(y2),
                  "detail": f"{shm:.4f}"})
    _log(f"\n  Y1 G_prior={pr:.6f} {'PASS' if y1 else 'FAIL'}   "
         f"Y2 H_shuffled={shm:.4f} {'PASS' if y2 else 'FAIL'}   "
         f"Y4 {'PASS' if y4 else 'FAIL'}   Y5 {'PASS' if y5 else 'FAIL'} {pinned}")

    m = {k: res["arms"][k]["macro"] for k in res["arms"]}
    d_fuse = m["C_fusion"] - m["B_geometry"]
    d_min = m["D_relfeat_plus_dxdy"] - m["A_relfeat"]
    d_dim = m["F_random_plus_geometry"] - m["B_geometry"]
    d_fuse_matched = m["C_fusion"] - m["F_random_plus_geometry"]
    prim = ("FUSION-GAIN" if d_fuse >= 0.02 else
            "FUSION-HARMFUL" if d_fuse <= -0.02 else "FUSION-NEUTRAL")
    sec = "MINIMAL-FIX-WORKS" if d_min >= 0.02 else "MINIMAL-FIX-FAILS"
    dim_confounded = d_dim <= -0.02
    allg = all(gg["pass"] for gg in gates)
    res["verdict"] = {"delta_fuse": d_fuse, "delta_min": d_min,
                      "delta_dim": d_dim,
                      "delta_fuse_dimension_matched": d_fuse_matched,
                      "dimensionality_confounded": bool(dim_confounded),
                      "primary": prim, "secondary": sec,
                      "gates_all_pass": allg}
    _log(f"\n{'-'*110}\n  PRE-REGISTERED VERDICT\n{'-'*110}")
    _log(f"    PRIMARY   delta_fuse = C - B = {d_fuse:+.4f}   -> {prim}")
    _log(f"    SECONDARY delta_min  = D - A = {d_min:+.4f}   -> {sec}")
    _log(f"    TERTIARY  delta_dim  = F - B = {d_dim:+.4f}   "
         f"dimensionality confounded: {dim_confounded}")
    _log(f"    dimension-matched contrast C - F = {d_fuse_matched:+.4f}")
    _log(f"    anchor (DIFFERENT REGIME, train-fitted CE geometry) "
         f"= {G_TRAIN_FITTED_ANCHOR:.4f}")
    _log(f"    gates all pass: {allg}")
    if not allg:
        _log("    -> VOID: a validity gate failed; no number above is reportable")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
