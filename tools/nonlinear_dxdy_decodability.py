#!/usr/bin/env python
"""Nonlinear decodability of dx_rel/dy_rel from rel_feat.

Pre-registered in docs/NONLINEAR_DXDY_DECODABILITY_PREREGISTRATION.md,
committed before this file's first run. Extends p57 (linear R^2 only) with
an MLP arm and a shuffled-label control, both cross-fitted on the identical
5 folds, with bootstrap 95% CIs.

Question: is relative position genuinely absent from rel_feat, or merely not
LINEARLY decodable? p57 found dx_rel R^2=0.052, dy_rel R^2=0.223 under a
ridge probe. This asks whether a small nonlinear probe recovers more.

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
GEO = _load("wprd_geometry_control")
CSP = _load("candidate_scorer_probe")
DEC = _load("geometry_decodability")
OBJ = _load("objective_ablation_relfeat")
Mech = MECH.Mech
N_FOLDS = 5
TARGETS = {"dx_rel": 4, "dy_rel": 5}


def _log(m: str = "") -> None:
    print(m, flush=True)


def mlp_regressor(d_in: int, hidden: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))


def xfit_mlp_r2(X: torch.Tensor, y: torch.Tensor, fold: torch.Tensor,
                hidden: int, epochs: int, lr: float, l2: float,
                seed: int) -> torch.Tensor:
    """Out-of-fold MLP regression predictions for a single target column."""
    pred = torch.zeros_like(y)
    gen = torch.Generator().manual_seed(seed)
    for f in range(N_FOLDS):
        te = fold == f
        tr = torch.nonzero(~te, as_tuple=True)[0]
        net = mlp_regressor(X.shape[1], hidden, seed)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
        Xtr, ytr = X[tr], y[tr]
        n, bs = Xtr.shape[0], 8192
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, bs):
                ix = perm[i:i + bs]
                opt.zero_grad()
                out = net(Xtr[ix]).squeeze(-1)
                torch.nn.functional.mse_loss(out, ytr[ix]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred[te] = net(X[te]).squeeze(-1)
    return pred


def r2_of(y: torch.Tensor, pred: torch.Tensor) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def bootstrap_ci_r2(y: torch.Tensor, pred: torch.Tensor, boot: int,
                    seed: int) -> List[float]:
    n = y.shape[0]
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(boot):
        idx = torch.randint(n, (n,), generator=g)
        vals.append(r2_of(y[idx], pred[idx]))
    v = torch.tensor(vals)
    lo, hi = torch.quantile(v, torch.tensor([0.025, 0.975])).tolist()
    return [lo, hi]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p66_nonlinear_dxdy/dec.json")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("NONLINEAR dx_rel/dy_rel DECODABILITY -- CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    fold = DEC.folds_of(B, 0)
    gates: List[Dict[str, Any]] = []

    sizes = [int((fold == f).sum()) for f in range(N_FOLDS)]
    z1 = sizes == [26483, 26856, 27190, 26586, 25441]
    gates.append({"gate": "Z1 folds as registered", "pass": bool(z1), "detail": str(sizes)})
    _log(f"  Z1 folds {sizes} {'PASS' if z1 else 'FAIL'}")

    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    X = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], 1)

    Graw = GEO.geometry_features_raw(B)
    Xg_all, _ = GEO._standardise(Graw)   # 19 standardised + bias

    res: Dict[str, Any] = {"tool": "nonlinear_dxdy_decodability", "gates": gates,
                           "estimator": {"hidden": args.hidden, "epochs": args.epochs,
                                        "lr": args.lr, "l2": args.l2},
                           "targets": {}}

    _log(f"\n{'-'*108}")
    _log(f"  {'target':>10} {'arm':>14} {'R2':>8} {'95% CI':>20}")
    _log(f"{'-'*108}")

    for name, col in TARGETS.items():
        y = Xg_all[:, col]

        # linear (reproduces p57's D1 for this column)
        pred_lin = DEC.xfit_r2  # not used directly; recompute OOF linear preds below
        pred_lin_vals = torch.zeros_like(y)
        for f in range(N_FOLDS):
            te = fold == f
            tr = ~te
            A = X[tr]
            AtA = A.T @ A + 1e-2 * A.shape[0] * torch.eye(A.shape[1])
            w = torch.linalg.solve(AtA, A.T @ y[tr])
            pred_lin_vals[te] = X[te] @ w
        r2_lin = r2_of(y, pred_lin_vals)
        ci_lin = bootstrap_ci_r2(y, pred_lin_vals, args.boot, seed=3)

        # MLP
        pred_mlp = xfit_mlp_r2(X, y, fold, args.hidden, args.epochs, args.lr,
                               args.l2, seed=0)
        r2_mlp = r2_of(y, pred_mlp)
        ci_mlp = bootstrap_ci_r2(y, pred_mlp, args.boot, seed=4)

        # shuffled-label control (MLP arch, permuted target)
        ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
        pred_sh = xfit_mlp_r2(X, ysh, fold, args.hidden, args.epochs, args.lr,
                              args.l2, seed=0)
        r2_sh = r2_of(ysh, pred_sh)
        ci_sh = bootstrap_ci_r2(ysh, pred_sh, args.boot, seed=5)

        res["targets"][name] = {
            "linear": {"r2": r2_lin, "ci95": ci_lin},
            "mlp": {"r2": r2_mlp, "ci95": ci_mlp},
            "shuffled_mlp": {"r2": r2_sh, "ci95": ci_sh},
        }
        _log(f"  {name:>10} {'linear (ridge)':>14} {r2_lin:8.4f} [{ci_lin[0]:.4f},{ci_lin[1]:.4f}]")
        _log(f"  {name:>10} {'MLP':>14} {r2_mlp:8.4f} [{ci_mlp[0]:.4f},{ci_mlp[1]:.4f}]")
        _log(f"  {name:>10} {'MLP, shuffled':>14} {r2_sh:8.4f} [{ci_sh[0]:.4f},{ci_sh[1]:.4f}]")

    z2 = all(abs(res["targets"][k]["linear"]["r2"] - ref) < 0.01
             for k, ref in [("dx_rel", 0.052), ("dy_rel", 0.223)])
    z3 = all(-0.02 <= res["targets"][k]["shuffled_mlp"]["r2"] <= 0.02
             for k in TARGETS)
    gates.append({"gate": "Z2 linear reproduces p57 (0.052/0.223) +-0.01",
                  "pass": bool(z2),
                  "detail": {k: res["targets"][k]["linear"]["r2"] for k in TARGETS}})
    gates.append({"gate": "Z3 shuffled MLP control in [-0.02,0.02]",
                  "pass": bool(z3),
                  "detail": {k: res["targets"][k]["shuffled_mlp"]["r2"] for k in TARGETS}})
    _log(f"\n  Z2 {'PASS' if z2 else 'FAIL'}   Z3 {'PASS' if z3 else 'FAIL'}")

    verdicts = {}
    for name in TARGETS:
        mlp_ci = res["targets"][name]["mlp"]["ci95"]
        sh_ci = res["targets"][name]["shuffled_mlp"]["ci95"]
        # "clearly > chance": the MLP CI lower bound exceeds the shuffled
        # control's CI upper bound -- non-overlapping intervals.
        clearly_above = mlp_ci[0] > sh_ci[1]
        verdicts[name] = "NONLINEAR CLEARLY > CHANCE" if clearly_above else "NONLINEAR ~ CHANCE"
        _log(f"\n  {name}: MLP CI {mlp_ci} vs shuffled CI {sh_ci} -> {verdicts[name]}")

    allpass = all(g["pass"] for g in gates)
    res["verdicts"] = verdicts
    res["gates_all_pass"] = bool(allpass)
    if not allpass:
        _log("\n  *** A GATE FAILED. By the registration, NO NUMBER HERE IS REPORTABLE. ***")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
