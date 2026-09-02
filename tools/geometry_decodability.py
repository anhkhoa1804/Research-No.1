#!/usr/bin/env python
"""p57 -- is the box layout that beats rel_feat even PRESENT in rel_feat?

Pre-registered in docs/GEOMETRY_DECODABILITY_PREREGISTRATION.md, committed
before this file existed. Thresholds are quoted from it and NOT recomputed.

  PRIMARY  R2_geom = mean out-of-fold R^2 of rel_feat -> the 19 geometry features
    LAYOUT-ABSENT   R2_geom <  0.30
    LAYOUT-PARTIAL  0.30 <= R2_geom < 0.70
    LAYOUT-PRESENT  R2_geom >= 0.70

After p55/p56, H6 survives only by elimination. This gives it positive content
or takes it away: a linear probe on 19 box numbers beats every probe on the
768-d rel_feat, and that is either because the layout is ABSENT from rel_feat
(a named, actionable deficit) or because it is PRESENT and unused (a
contradiction, since the readout and objective routes are already closed).

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
Mech = MECH.Mech
N_FOLDS = 5

FEAT_NAMES = ["subj_cx", "subj_cy", "obj_cx", "obj_cy",
              "dx_rel", "dy_rel", "dx_img", "dy_img",
              "subj_w", "subj_h", "obj_w", "obj_h",
              "log_area_ratio", "subj_logaspect", "obj_logaspect",
              "iou", "subj_containment", "obj_containment", "feat19"]


def drop_constant(M: torch.Tensor):
    """Indices of columns with real variance.

    _standardise() appends a constant bias column. A constant target has
    ss_tot ~ 0, so its R^2 blows up to ~-1e13 and destroys any mean taken over
    it. Attempt 1 of this run reported a verdict produced entirely by that
    artifact (runs/p57_FAILED_constant_column). Targets are screened here.
    """
    keep = (M.std(0) > 1e-8).nonzero(as_tuple=True)[0]
    return keep


def _log(m: str = "") -> None:
    print(m, flush=True)


def folds_of(B: Mech, salt: int = 0) -> torch.Tensor:
    per = [CSP.fold_of_image(str(s), N_FOLDS, salt) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per[i]] * len(B.meta["pairs"][i]))
    return torch.tensor(flat)[B.gt_row]


def xfit_r2(X: torch.Tensor, Y: torch.Tensor, fold: torch.Tensor,
            l2: float = 1e-2) -> torch.Tensor:
    """Per-column out-of-fold R^2 of a ridge from X to Y."""
    pred = torch.zeros_like(Y)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        A = X[tr]
        AtA = A.T @ A + l2 * A.shape[0] * torch.eye(A.shape[1])
        Wm = torch.linalg.solve(AtA, A.T @ Y[tr])
        pred[te] = X[te] @ Wm
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p57_geometry_decodability/dec.json")
    args = ap.parse_args(argv)

    _log("=" * 104)
    _log("p57 GEOMETRY DECODABILITY -- is the layout that beats rel_feat inside it? "
         "CPU only, NO GPU")
    _log("=" * 104)

    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    fold = folds_of(B, 0)
    gates: List[Dict[str, Any]] = []

    RF = torch.cat([x.float() for x in raw["rel_feat"]], 0)[B.gt_row]
    x2 = (RF.shape == (B.n_gt, 768)) and bool(torch.isfinite(RF).all())
    sizes = [int((fold == f).sum()) for f in range(N_FOLDS)]
    x3 = sizes == [26483, 26856, 27190, 26586, 25441]
    gates.append({"gate": "X2 rel_feat shape/finite", "pass": bool(x2),
                  "detail": str(tuple(RF.shape))})
    gates.append({"gate": "X3 folds identical to p37", "pass": bool(x3),
                  "detail": str(sizes)})
    _log(f"  X2 rel_feat {tuple(RF.shape)} {'PASS' if x2 else 'FAIL'}")
    _log(f"  X3 folds {sizes} {'PASS' if x3 else 'FAIL'}")

    X = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], 1)

    Xg_raw = GEO.geometry_features_raw(B)
    Xg_all, _ = GEO._standardise(Xg_raw)
    keep = drop_constant(Xg_all)
    Xg = Xg_all[:, keep]
    _log(f"  geometry features {tuple(Xg_all.shape)} -> {tuple(Xg.shape)} after "
         f"dropping {Xg_all.shape[1]-Xg.shape[1]} constant column(s) "
         f"(the appended bias)")

    # ---- D1 ----
    _log(f"\n  D1  rel_feat -> the 19 geometry features (out-of-fold ridge)")
    r2 = xfit_r2(X, Xg, fold)
    for i in range(Xg.shape[1]):
        nm = FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"f{i}"
        _log(f"      {nm:>18}  R2 = {float(r2[i]):+.4f}")
    R2_geom = float(r2.mean())
    _log(f"      {'MEAN':>18}  R2 = {R2_geom:+.4f}")

    # ---- D2: the decision-relevant projection ----
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xg_raw, tst)
    Wg = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    o = torch.optim.LBFGS([Wg], max_iter=200, history_size=10,
                          line_search_fn="strong_wolfe")

    def cl():
        o.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ Wg, yt) + 1e-4 * (Wg * Wg).sum()
        l.backward()
        return l
    o.step(cl)
    geo_logits = (Xv @ Wg).detach()
    r2d2 = float(xfit_r2(X, geo_logits, fold).mean())
    _log(f"\n  D2  rel_feat -> the geometry probe's 50 LOGITS   mean R2 = {r2d2:+.4f}")

    # ---- D3 reverse ----
    Xgb = torch.cat([Xg, torch.ones(Xg.shape[0], 1)], 1)
    r2d3 = float(xfit_r2(Xgb, X[:, :-1], fold).mean())
    _log(f"  D3  geometry -> rel_feat (orientation)            mean R2 = {r2d3:+.4f}")

    # ---- controls ----
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(X.shape[0], generator=g)
    n1 = float(xfit_r2(X[perm], Xg, fold).mean())
    RND = torch.randn(X.shape[0], 768, generator=g)
    RND = torch.cat([RND, torch.ones(RND.shape[0], 1)], 1)
    n2 = float(xfit_r2(RND, Xg, fold).mean())
    x1 = (abs(n1) < 0.02) and (abs(n2) < 0.02)   # two-sided: see attempt 1
    gates.append({"gate": "X1 null controls near zero", "pass": bool(x1),
                  "detail": f"N1={n1:.4f} N2={n2:.4f}"})
    _log(f"\n  N1  shuffled-row control                          mean R2 = {n1:+.4f}")
    _log(f"  N2  random 768-d Gaussian -> geometry              mean R2 = {n2:+.4f}")
    _log(f"  X1 {'PASS' if x1 else 'FAIL'}")

    verdict = ("LAYOUT-ABSENT" if R2_geom < 0.30 else
               "LAYOUT-PARTIAL" if R2_geom < 0.70 else "LAYOUT-PRESENT")
    allg = all(gg["pass"] for gg in gates)
    res = {"tool": "geometry_decodability", "gates": gates,
           "per_feature_r2": {(FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"f{i}"):
                              float(r2[i]) for i in range(Xg.shape[1])},
           "R2_geom": R2_geom, "D2_logits_r2": r2d2, "D3_reverse_r2": r2d3,
           "N1_shuffled": n1, "N2_random": n2,
           "verdict": verdict, "gates_all_pass": allg}
    _log(f"\n{'-'*104}\n  PRE-REGISTERED VERDICT\n{'-'*104}")
    _log(f"    R2_geom (mean out-of-fold R^2, rel_feat -> 19 box features) "
         f"= {R2_geom:.4f}")
    _log(f"    -> {verdict}")
    _log(f"    gates all pass: {allg}")
    if not allg:
        _log("    -> VOID: a validity gate failed; no number above is reportable")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
