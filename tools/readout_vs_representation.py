#!/usr/bin/env python
"""p37 -- is the weak grounding a READOUT failure or a REPRESENTATION failure?

Pre-registered in docs/READOUT_VS_REPRESENTATION_PREREGISTRATION.md, including
the addendum that added the geometry reference arm before any number here
existed. Thresholds below are quoted from it and are not recomputed.

  PRIMARY   P* = max(R3_linear, R4_mlp) vs C = R2_cls
    READOUT-LIMITED        P* >= C + 0.03
    REPRESENTATION-LIMITED P* <  C + 0.01
    INTERMEDIATE           otherwise

  SECONDARY (addendum)  P* vs G = 0.5961, the train-fitted geometry probe
    BEYOND GEOMETRY     P* >= G + 0.02
    GEOMETRY-EQUIVALENT |P* - G| < 0.02
    BELOW GEOMETRY      P* <= G - 0.02

An asymmetry worth stating: the rel_feat probes are CROSS-FITTED ON VALIDATION
while the geometry reference is TRAIN-FITTED. That advantages rel_feat. If
rel_feat still fails to clear geometry despite the advantage, the
representation-limited reading is conservative.

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


CPA = _load("cprime_analysis")
MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
STRAT = _load("wprd_stratified")
GEO = _load("wprd_geometry_control")
CSP = _load("candidate_scorer_probe")
Mech = MECH.Mech
N_FOLDS = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


def folds_of(B: Mech, salt: int = 0) -> torch.Tensor:
    per = [CSP.fold_of_image(str(s), N_FOLDS, salt) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per[i]] * len(B.meta["pairs"][i]))
    return torch.tensor(flat)[B.gt_row]


def xfit_linear(X, y, fold, C, epochs=200, l2=1e-4):
    out = torch.zeros(X.shape[0], C)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        A = X[tr]
        AtA = A.T @ A + l2 * A.shape[0] * torch.eye(A.shape[1])
        Y = torch.zeros(A.shape[0], C)
        Y[torch.arange(A.shape[0]), y[tr]] = 1.0
        Wm = torch.linalg.solve(AtA, A.T @ Y)
        out[te] = X[te] @ Wm
    return out


def xfit_mlp(X, y, fold, C, hidden=512, epochs=25, lr=2e-3, l2=1e-4, seed=0):
    out = torch.zeros(X.shape[0], C)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, C))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
        Xtr, ytr = X[tr], y[tr]
        n, bs = Xtr.shape[0], 4096
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                ix = perm[i:i + bs]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(Xtr[ix]), ytr[ix]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            out[te] = net(X[te])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--ref-dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p37_readout_vs_representation/rvr.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p37 READOUT vs REPRESENTATION -- CPU only, reads the p36 cache, NO GPU")
    _log("=" * 108)
    B = Mech(args.dump, args.prior, "raw50")
    raw = torch.load(args.dump, map_location="cpu", weights_only=False)
    gates: List[Dict[str, Any]] = []

    # ---- V1 ----
    rf_list = raw.get("rel_feat")
    assert rf_list is not None, "cache has no rel_feat -- wrong dump?"
    RF = torch.cat([x.float() for x in rf_list], 0)[B.gt_row]
    v1 = (int(raw.get("missing_rel_feat", 0)) == 0
          and not bool(torch.isnan(RF).any()))
    gates.append({"gate": "V1 rel_feat present & finite", "pass": bool(v1),
                  "detail": f"missing={raw.get('missing_rel_feat', 0)} "
                            f"shape={tuple(RF.shape)}"})
    _log(f"  V1 rel_feat {tuple(RF.shape)}  missing={raw.get('missing_rel_feat',0)}  "
         f"{'PASS' if v1 else 'FAIL'}")

    # ---- V2: the load-bearing one ----
    PE = raw.get("pred_emb")
    v2pass, v2det = False, "pred_emb absent"
    if PE is not None:
        PE = PE.float()
        bg = set(int(i) for i in raw.get("background_predicate_indices", []))
        fg = [i for i in range(len(raw["pred_vocab"])) if i not in bg]
        recon = (torch.nn.functional.normalize(RF, dim=-1)
                 @ torch.nn.functional.normalize(PE, dim=-1).T)[:, fg]
        stored = torch.cat([x.float() for x in raw["text_logits"]], 0)[B.gt_row][:, fg]
        md = float((recon - stored).abs().max())
        v2pass = md < 1e-2
        v2det = f"max|recomputed - stored text_logits| = {md:.3e} (fp16 tol 1e-2)"
    gates.append({"gate": "V2 rel_feat IS the tensor the head read",
                  "pass": bool(v2pass), "detail": v2det})
    _log(f"  V2 {v2det}  {'PASS' if v2pass else 'FAIL'}")

    # ---- V4: reproduces p24 ----
    Bref = Mech(args.ref_dump, args.prior, "raw50")
    same = float((B.model - Bref.model).abs().max())
    v4 = same < 1e-3 and B.n_gt == Bref.n_gt
    gates.append({"gate": "V4 reproduces the p24 cache", "pass": bool(v4),
                  "detail": f"max|model_term diff| = {same:.3e}, rows "
                            f"{B.n_gt} vs {Bref.n_gt}"})
    _log(f"  V4 max|model_term - p24| = {same:.3e}, rows {B.n_gt} vs {Bref.n_gt}  "
         f"{'PASS' if v4 else 'FAIL'}")

    Gs = WPD.Groups(B)
    fold = folds_of(B, 0)
    y = B.gt_y
    _log(f"  folds {[int((fold==f).sum()) for f in range(N_FOLDS)]}")

    # standardise rel_feat, append bias
    RFs = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    RFs = torch.cat([RFs, torch.ones(RFs.shape[0], 1)], 1)

    # geometry reference (train-fitted, as in p39)
    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    Wg = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    o = torch.optim.LBFGS([Wg], max_iter=200, history_size=10,
                          line_search_fn="strong_wolfe")

    def cl():
        o.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ Wg, yt) + 1e-4 * (Wg * Wg).sum()
        l.backward()
        return l

    o.step(cl)
    geo = (Xv @ Wg).detach()

    # group-centred rel_feat for R5
    cnt = torch.zeros(Gs.G).index_add_(0, Gs.pair_id, torch.ones(RFs.shape[0]))
    s = torch.zeros(Gs.G, RFs.shape[1]).index_add_(0, Gs.pair_id, RFs)
    RFc = RFs - (s / cnt.clamp_min(1).unsqueeze(1))[Gs.pair_id]

    _log(f"\n  fitting probes (cross-fitted on validation folds)")
    arms: Dict[str, torch.Tensor] = {}
    arms["R1_text (evaluated head)"] = B.fixed_ensemble(0.0)
    arms["R2_cls (discarded head)"] = B.fixed_ensemble(1.0)
    _log(f"    R3 linear on rel_feat ...")
    arms["R3_linear on rel_feat"] = xfit_linear(RFs, y, fold, B.n_classes)
    _log(f"    R4 MLP on rel_feat ...")
    arms["R4_mlp on rel_feat"] = xfit_mlp(RFs, y, fold, B.n_classes,
                                          args.hidden, args.epochs)
    _log(f"    R5 linear on group-centred rel_feat ...")
    arms["R5_residual (group-centred)"] = xfit_linear(RFc, y, fold, B.n_classes)
    _log(f"    R6 shuffled-label control ...")
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    arms["R6_shuffled (must be 0.5)"] = xfit_linear(RFs, ysh, fold, B.n_classes)
    arms["R7_prior (must be 0.5)"] = B.prior
    arms["R8_geom (train-fitted)"] = geo
    _log(f"    R9 linear on [rel_feat, geometry] ...")
    both = torch.cat([RFs, Xv], 1)
    arms["R9_relfeat+geometry"] = xfit_linear(both, y, fold, B.n_classes)

    res: Dict[str, Any] = {"tool": "readout_vs_representation", "gates": gates,
                           "arms": {}}
    _log(f"\n{'-'*108}")
    _log(f"  {'arm':>32} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
         f"{'head':>7} {'body':>7} {'tail':>7}")
    _log(f"{'-'*108}")
    store = {}
    for name, sc in arms.items():
        r = WPD.wprd(Gs, sc, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        cc = STRAT.cells(Gs, B, sc, args.cap, 0, drop_same_instance=False)
        bk = {}
        for key in ("head-head", "body-body", "tail-tail"):
            x, y2 = key.split("-")
            sub = [c["auc"] for c in cc
                   if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([x, y2])]
            bk[key] = sum(sub) / len(sub) if sub else float("nan")
        store[name] = v
        res["arms"][name] = {"macro": r["wprd_macro"], "weighted": r["wprd_weighted"],
                             "ci95": [lo, hi], "by_bucket": bk}
        _log(f"  {name:>32} {r['wprd_macro']:>8.4f} {r['wprd_weighted']:>9.4f} "
             f"[{lo:.4f},{hi:.4f}] {bk['head-head']:>7.4f} {bk['body-body']:>7.4f} "
             f"{bk['tail-tail']:>7.4f}")

    # ---- V3 ----
    r6 = res["arms"]["R6_shuffled (must be 0.5)"]["macro"]
    r7 = res["arms"]["R7_prior (must be 0.5)"]["macro"]
    v3 = (0.49 <= r6 <= 0.51) and (0.49 <= r7 <= 0.51)
    gates.append({"gate": "V3 null controls at chance", "pass": bool(v3),
                  "detail": f"R6={r6:.4f} R7={r7:.4f}"})
    _log(f"\n  V3 R6_shuffled={r6:.4f}  R7_prior={r7:.4f}  {'PASS' if v3 else 'FAIL'}")

    # ---- collapse measure: how much of rel_feat is (s,o) identity? ----
    # The one-hot is built over the STANDARD VG150 150-object vocabulary plus a
    # single "other" bucket. Building it over the raw labels instead produces a
    # 132,556 x 14,503 matrix (~7.7 GB) because this dataset retains 16,929 raw
    # Visual Genome object names -- that is what OOM-killed the first attempt.
    # See docs/DATASET_IDENTITY_OBJECT_VOCAB.md.
    vj = json.loads((Path(args.train_jsonl).parent / "vocabulary" /
                     "objects.json").read_text(encoding="utf-8"))
    cats = sorted(set(str(x).strip().lower() for x in vj["idx_to_label"].values()))
    ci = {v: i for i, v in enumerate(cats)}
    K = len(cats) + 1                       # + "other"
    n_rows = RFs.shape[0]
    OH = torch.zeros(n_rows, 2 * K)
    ar = torch.arange(n_rows)
    OH[ar, torch.tensor([ci.get(v, len(cats)) for v in Gs.subj])] = 1
    OH[ar, K + torch.tensor([ci.get(v, len(cats)) for v in Gs.obj])] = 1
    OH = torch.cat([OH, torch.ones(n_rows, 1)], 1)
    _log(f"  collapse one-hot width {OH.shape[1]} "
         f"(150 categories + other, per role)")
    r2s = []
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        A = OH[tr]
        AtA = A.T @ A + 1e-2 * A.shape[0] * torch.eye(A.shape[1])
        Wm = torch.linalg.solve(AtA, A.T @ RFs[tr])
        pred = OH[te] @ Wm
        ss = float(((RFs[te] - RFs[tr].mean(0, keepdim=True)) ** 2).sum())
        r2s.append(1.0 - float(((RFs[te] - pred) ** 2).sum()) / ss)
    r2 = sum(r2s) / len(r2s)
    res["collapse_r2_subject_object_onehot_to_relfeat"] = r2
    _log(f"\n  COLLAPSE: out-of-fold R^2 of (subject,object) one-hots predicting "
         f"rel_feat = {r2*100:.2f}%")

    # ---- verdict ----
    C = res["arms"]["R2_cls (discarded head)"]["macro"]
    P = max(res["arms"]["R3_linear on rel_feat"]["macro"],
            res["arms"]["R4_mlp on rel_feat"]["macro"])
    G = res["arms"]["R8_geom (train-fitted)"]["macro"]
    prim = ("READOUT-LIMITED" if P >= C + 0.03 else
            "REPRESENTATION-LIMITED" if P < C + 0.01 else "INTERMEDIATE")
    sec = ("BEYOND GEOMETRY" if P >= G + 0.02 else
           "BELOW GEOMETRY" if P <= G - 0.02 else "GEOMETRY-EQUIVALENT")
    allg = all(g["pass"] for g in gates)
    res["verdict"] = {"P_star": P, "C_classifier": C, "G_geometry": G,
                      "primary": prim, "secondary": sec, "gates_all_pass": allg}
    _log(f"\n{'-'*108}\n  PRE-REGISTERED VERDICT\n{'-'*108}")
    _log(f"    P* (best rel_feat probe) {P:.4f}   C (classifier head) {C:.4f}   "
         f"G (geometry) {G:.4f}")
    _log(f"    P* - C = {P-C:+.4f}    P* - G = {P-G:+.4f}")
    _log(f"    PRIMARY   : {prim}")
    _log(f"    SECONDARY : {sec}")
    _log(f"    gates all pass: {allg}")
    if not allg:
        _log(f"    -> VOID: a validity gate failed; no number above is reportable")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
