#!/usr/bin/env python
"""p55 -- the objective ablation ON rel_feat, with p48's supervision confound fixed.

Pre-registered in docs/OBJECTIVE_ABLATION_RELFEAT_PREREGISTRATION.md, committed
before this file existed. Thresholds are quoted from it and NOT recomputed here.

  PRIMARY  d_obj = B_contr_decidable - C_ce_matched
    OBJECTIVE-LIMITED  d_obj >= +0.020        (inherited from p48, unchanged)
    WEAK               +0.005 <= d_obj < +0.020
    REFUTED            d_obj < +0.005

  SECONDARY  d_sup = C_ce_matched - A_ce_all,  d_bal = D_ce_pairbal - A_ce_all
    SUPERVISION-SENSITIVE if max(|d_sup|,|d_bal|) >= 0.020

  TERTIARY   vs G = 0.5961 (train-fitted geometry linear, p37/p39)
    BEATS GEOMETRY >= G+0.02 · REACHES within 0.02 · BELOW otherwise

WHY CROSS-FITTED ON VALIDATION. rel_feat exists only for the validation split
(p36). There is no train-split rel_feat cache and building one is ~30 GPU-hours.
So every arm is cross-fitted over 5 validation folds split by IMAGE -- the
identical construction p37 used. Each row's score comes from a model that never
saw that row.

THE CONFOUND FIX. C_ce_matched is A restricted to exactly the rows the
contrastive arm is allowed to learn from (those in groups with >=2 distinct
predicates among the training-fold rows). B - C therefore isolates the OBJECTIVE
at a matched information budget, which is the comparison p48 could not make.

CPU only. Reads the p36 cache. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
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
G_GEOMETRY = 0.5961          # p37/p39 train-fitted geometry linear. Fixed reference.


def _log(m: str = "") -> None:
    print(m, flush=True)


def folds_of(B: Mech, salt: int = 0) -> torch.Tensor:
    per = [CSP.fold_of_image(str(s), N_FOLDS, salt) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per[i]] * len(B.meta["pairs"][i]))
    return torch.tensor(flat)[B.gt_row]


def mlp(d_in: int, hidden: int, d_out: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out))


def decidable_rows_in(rows: torch.Tensor, pair_id: torch.Tensor,
                      y: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
    """Rows (a subset) that live in groups with >=2 distinct predicates
    AMONG THOSE ROWS. Returns the row subset and the group->rows map."""
    by: Dict[int, List[int]] = defaultdict(list)
    for r in rows.tolist():
        by[int(pair_id[r])].append(r)
    keep: List[int] = []
    groups: Dict[int, List[int]] = {}
    for p, rs in by.items():
        if len(set(y[torch.tensor(rs)].tolist())) >= 2:
            keep.extend(rs)
            groups[p] = rs
    return torch.tensor(sorted(keep), dtype=torch.long), groups


def sample_contrast_pairs(groups: Dict[int, List[int]], y: torch.Tensor,
                          n_target: int, gen: torch.Generator,
                          pair_balanced: bool) -> Tuple[torch.Tensor, ...]:
    """(i, j, a, b) with i,j in one group, y_i = a != b = y_j.

    pair_balanced=False draws groups in proportion to how many contrasts they
    supply (p48's behaviour). pair_balanced=True draws each group equally often,
    which is intervention (A)/(B) from the directive: balanced within-pair
    sampling. Nothing else differs.
    """
    keys = list(groups.keys())
    if not keys:
        return (torch.empty(0, dtype=torch.long),) * 4
    if pair_balanced:
        w = torch.ones(len(keys), dtype=torch.double)
    else:
        w = torch.tensor([float(len(groups[k]) * (len(groups[k]) - 1))
                          for k in keys], dtype=torch.double)
    w = w / w.sum()
    pick = torch.multinomial(w, n_target, replacement=True, generator=gen)
    I, J, A, Bc = [], [], [], []
    for gi in pick.tolist():
        rs = groups[keys[gi]]
        t = torch.randint(len(rs), (2,), generator=gen).tolist()
        i, j = rs[t[0]], rs[t[1]]
        a, b = int(y[i]), int(y[j])
        if a == b:
            continue
        I.append(i); J.append(j); A.append(a); Bc.append(b)
    return (torch.tensor(I, dtype=torch.long), torch.tensor(J, dtype=torch.long),
            torch.tensor(A, dtype=torch.long), torch.tensor(Bc, dtype=torch.long))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p55_objective_relfeat/obj.json")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--pairs-per-epoch", type=int, default=200000)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p55 OBJECTIVE ABLATION ON rel_feat -- supervision-matched. CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    Gs = WPD.Groups(B)
    fold = folds_of(B, 0)
    y = B.gt_y
    gates: List[Dict[str, Any]] = []

    # ---- V3 ----
    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    v3 = (RF.shape == (B.n_gt, 768) and int(raw.get("missing_rel_feat", 0)) == 0
          and not bool(torch.isnan(RF).any()))
    gates.append({"gate": "V3 rel_feat complete & finite", "pass": bool(v3),
                  "detail": f"shape={tuple(RF.shape)} missing={raw.get('missing_rel_feat',0)}"})
    _log(f"  V3 rel_feat {tuple(RF.shape)} missing={raw.get('missing_rel_feat',0)}  "
         f"{'PASS' if v3 else 'FAIL'}")

    X = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
    del RF

    # group-centred variant for arm F
    cnt = torch.zeros(Gs.G).index_add_(0, Gs.pair_id, torch.ones(X.shape[0]))
    s = torch.zeros(Gs.G, X.shape[1]).index_add_(0, Gs.pair_id, X)
    Xc = X - (s / cnt.clamp_min(1).unsqueeze(1))[Gs.pair_id]
    del s

    _log(f"  rows {B.n_gt:,}  groups {Gs.G:,}  folds "
         f"{[int((fold==f).sum()) for f in range(N_FOLDS)]}")
    _log(f"  decidable rows (whole split) {int(Gs.decidable.sum()):,} "
         f"({100*float(Gs.decidable.float().mean()):.1f}%)")

    C = B.n_classes
    gen = torch.Generator().manual_seed(0)
    scores: Dict[str, torch.Tensor] = {}
    budget: Dict[str, Dict[str, int]] = {}

    def fit_ce(name: str, feats: torch.Tensor, labels: torch.Tensor,
               restrict_decidable: bool, pair_balanced: bool) -> None:
        out = torch.zeros(feats.shape[0], C)
        n_rows_tot, n_contr_tot = 0, 0
        for f in range(N_FOLDS):
            te = fold == f
            tr_rows = torch.nonzero(~te, as_tuple=True)[0]
            if restrict_decidable:
                tr_rows, grp = decidable_rows_in(tr_rows, Gs.pair_id, labels)
                n_contr_tot += sum(len(v) * (len(v) - 1) for v in grp.values())
            n_rows_tot += len(tr_rows)
            net = mlp(feats.shape[1], args.hidden, C, 0)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                    weight_decay=args.l2)
            Xtr, ytr = feats[tr_rows], labels[tr_rows]
            if pair_balanced:
                gsz = torch.zeros(Gs.G).index_add_(
                    0, Gs.pair_id[tr_rows], torch.ones(len(tr_rows)))
                w = (1.0 / gsz.clamp_min(1))[Gs.pair_id[tr_rows]].double()
                w = w / w.sum()
            n, bs = Xtr.shape[0], 8192
            for ep in range(args.epochs):
                if pair_balanced:
                    perm = torch.multinomial(w, n, replacement=True, generator=gen)
                else:
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
        budget[name] = {"train_rows": int(n_rows_tot),
                        "contrast_pairs": int(n_contr_tot)}
        _log(f"    {name:<26} train rows {n_rows_tot:>9,}  contrasts "
             f"{n_contr_tot:>11,}")

    def fit_contrastive(name: str, pair_balanced: bool) -> None:
        out = torch.zeros(X.shape[0], C)
        n_rows_tot, n_contr_tot, n_used = 0, 0, 0
        for f in range(N_FOLDS):
            te = fold == f
            tr_rows = torch.nonzero(~te, as_tuple=True)[0]
            tr_rows, grp = decidable_rows_in(tr_rows, Gs.pair_id, y)
            n_rows_tot += len(tr_rows)
            n_contr_tot += sum(len(v) * (len(v) - 1) for v in grp.values())
            net = mlp(X.shape[1], args.hidden, C, 0)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                    weight_decay=args.l2)
            pb = 16384
            for ep in range(args.epochs):
                I, J, A, Bc = sample_contrast_pairs(
                    grp, y, args.pairs_per_epoch, gen, pair_balanced)
                if len(I) == 0:
                    continue
                n_used += len(I)
                perm = torch.randperm(len(I), generator=gen)
                for i in range(0, len(I), pb):
                    ix = perm[i:i + pb]
                    ii, jj, aa, cc = I[ix], J[ix], A[ix], Bc[ix]
                    opt.zero_grad()
                    fi, fj = net(X[ii]), net(X[jj])
                    m = ((fi.gather(1, aa.view(-1, 1)) - fi.gather(1, cc.view(-1, 1)))
                         - (fj.gather(1, aa.view(-1, 1)) - fj.gather(1, cc.view(-1, 1))))
                    torch.nn.functional.softplus(-m).mean().backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                out[te] = net(X[te])
        scores[name] = out
        budget[name] = {"train_rows": int(n_rows_tot),
                        "contrast_pairs": int(n_contr_tot),
                        "contrast_samples_drawn": int(n_used)}
        _log(f"    {name:<26} train rows {n_rows_tot:>9,}  contrasts "
             f"{n_contr_tot:>11,}  drawn {n_used:>10,}")

    _log(f"\n  fitting arms (cross-fitted, {N_FOLDS} folds, hidden={args.hidden}, "
         f"epochs={args.epochs})")
    fit_ce("A_ce_all", X, y, False, False)
    fit_contrastive("B_contr_decidable", False)
    fit_ce("C_ce_matched", X, y, True, False)
    fit_ce("D_ce_pairbal", X, y, False, True)
    fit_contrastive("E_contr_pairbal", True)
    fit_ce("F_ce_groupcentred", Xc, y, False, False)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    fit_ce("N_shuffled", X, ysh, False, False)
    scores["P_prior"] = B.prior
    scores["ref_text_head"] = B.fixed_ensemble(0.0)
    scores["ref_classifier_head"] = B.fixed_ensemble(1.0)

    # ---- V4 ----
    v4 = all(bool(torch.isfinite(v).all()) and v.shape[0] == B.n_gt
             for v in scores.values())
    gates.append({"gate": "V4 every arm scores every row", "pass": bool(v4),
                  "detail": f"{len(scores)} arms x {B.n_gt} rows"})

    res: Dict[str, Any] = {"tool": "objective_ablation_relfeat",
                           "gates": gates, "budget": budget, "arms": {}}
    _log(f"\n{'-'*108}")
    _log(f"  {'arm':>26} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
         f"{'head':>7} {'body':>7} {'tail':>7}")
    _log(f"{'-'*108}")
    for name, sc in scores.items():
        r = WPD.wprd(Gs, sc, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        bs_ = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                           for _ in range(args.boot)])
        lo, hi = torch.quantile(bs_, torch.tensor([0.025, 0.975],
                                                  dtype=torch.float64)).tolist()
        cc = STRAT.cells(Gs, B, sc, args.cap, 0, drop_same_instance=False)
        bk = {}
        for key in ("head-head", "body-body", "tail-tail"):
            x1, x2 = key.split("-")
            sub = [c["auc"] for c in cc
                   if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([x1, x2])]
            bk[key] = sum(sub) / len(sub) if sub else float("nan")
        res["arms"][name] = {"macro": r["wprd_macro"], "weighted": r["wprd_weighted"],
                             "ci95": [lo, hi], "n_cells": r["n_cells"],
                             "by_bucket": bk}
        _log(f"  {name:>26} {r['wprd_macro']:>8.4f} {r['wprd_weighted']:>9.4f} "
             f"[{lo:.4f},{hi:.4f}] {bk['head-head']:>7.4f} {bk['body-body']:>7.4f} "
             f"{bk['tail-tail']:>7.4f}")

    # ---- V1, V2 ----
    pr = res["arms"]["P_prior"]["macro"]
    sh = res["arms"]["N_shuffled"]["macro"]
    v1 = abs(pr - 0.5) < 1e-6
    v2 = 0.49 <= sh <= 0.51
    gates.append({"gate": "V1 prior control exactly 0.5000", "pass": bool(v1),
                  "detail": f"{pr:.6f}"})
    gates.append({"gate": "V2 shuffled control at chance", "pass": bool(v2),
                  "detail": f"{sh:.4f}"})
    _log(f"\n  V1 P_prior={pr:.6f} {'PASS' if v1 else 'FAIL'}   "
         f"V2 N_shuffled={sh:.4f} {'PASS' if v2 else 'FAIL'}")

    # ---- verdicts ----
    m = {k: res["arms"][k]["macro"] for k in res["arms"]}
    d_obj = m["B_contr_decidable"] - m["C_ce_matched"]
    d_sup = m["C_ce_matched"] - m["A_ce_all"]
    d_bal = m["D_ce_pairbal"] - m["A_ce_all"]
    prim = ("OBJECTIVE-LIMITED" if d_obj >= 0.020 else
            "WEAK" if d_obj >= 0.005 else "REFUTED")
    sec = ("SUPERVISION-SENSITIVE" if max(abs(d_sup), abs(d_bal)) >= 0.020
           else "SUPERVISION-INSENSITIVE")
    best = max((m[k], k) for k in
               ("A_ce_all", "B_contr_decidable", "C_ce_matched", "D_ce_pairbal",
                "E_contr_pairbal", "F_ce_groupcentred"))
    ter = ("BEATS GEOMETRY" if best[0] >= G_GEOMETRY + 0.02 else
           "REACHES GEOMETRY" if abs(best[0] - G_GEOMETRY) < 0.02 else
           "BELOW GEOMETRY")
    allg = all(g["pass"] for g in gates)
    res["verdict"] = {"d_obj": d_obj, "d_sup": d_sup, "d_bal": d_bal,
                      "best_arm": best[1], "best_macro": best[0],
                      "G_geometry": G_GEOMETRY, "primary": prim,
                      "secondary": sec, "tertiary": ter, "gates_all_pass": allg}
    _log(f"\n{'-'*108}\n  PRE-REGISTERED VERDICT\n{'-'*108}")
    _log(f"    PRIMARY   d_obj = B_contr_decidable - C_ce_matched = {d_obj:+.4f}"
         f"   -> {prim}")
    _log(f"    SECONDARY d_sup = {d_sup:+.4f}   d_bal = {d_bal:+.4f}   -> {sec}")
    _log(f"    TERTIARY  best fitted arm {best[1]} {best[0]:.4f} vs "
         f"G={G_GEOMETRY:.4f}  -> {ter}")
    _log(f"    gates all pass: {allg}")
    if not allg:
        _log(f"    -> VOID: a validity gate failed; no number above is reportable")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
