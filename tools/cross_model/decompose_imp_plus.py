#!/usr/bin/env python
"""IMP+ WPRD decomposition: is 0.6205 grounding, pair identity, geometry, or
calibration/score construction?

Reuses this project's own WPRD machinery throughout -- it does NOT reimplement
AUC or the WPRD cell/weighting definition. Every arm below is scored by
`tools/cross_model/compute_wprd.py::wprd_generic`, which itself imports
`auc()` from `tools/within_pair_discrimination.py` (the canonical
implementation, byte-identical tie-handling to every WPRD number in this
project, including PURE's). The geometry probe reuses
`tools/wprd_geometry_control.py`'s exact `_geom` feature function, standardiser
and cross-fit LBFGS classifier verbatim. The layout/residual ridge split
reuses `tools/within_group_decomposition.py`'s exact closed-form ridge
approach. Only the bucket-stratified cell aggregation and the pair-matched
permutation null are new code, both small, direct ports of the same technique
`tools/wprd_stratified.py` and `runs/p26` already used for PURE.

Inputs (already on disk, produced by prior runs -- no GPU, no model forward
here):
  --pairs     wprd_pairs_full.pt   (model_term, gt_y, subj_label, obj_label)
  --geometry  geometry_pairs_full.pt (obj_boxes, pairs_local, gt_y, image_id)

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

REPO_ROOT = Path("/home/leanhkhoa150204/Research-No.1")


def _load(name: str, subdir: str = "tools"):
    spec = importlib.util.spec_from_file_location(
        name, str(REPO_ROOT / subdir / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


WPD = _load("within_pair_discrimination")           # auc()
GEO = _load("wprd_geometry_control")                # _geom, _standardise, cross_fit_logits, CSP
CW = _load("compute_wprd", "tools/cross_model")      # SimpleGroups, wprd_generic, empirical_group_prior

AUC = WPD.auc
CAP, SEED = 64, 0


def _log(m: str = "") -> None:
    print(m, flush=True)


def group_mean(score: torch.Tensor, pair_id: torch.Tensor, G: int) -> torch.Tensor:
    cnt = torch.zeros(G).index_add_(0, pair_id, torch.ones(score.shape[0]))
    s = torch.zeros(G, score.shape[1]).index_add_(0, pair_id, score)
    return (s / cnt.clamp_min(1).unsqueeze(1))[pair_id]


def pair_matched_null(Gs, score: torch.Tensor, seed: int = SEED) -> torch.Tensor:
    """Permute the model's score vector among rows of the SAME (s,o) group --
    destroys row-to-image correspondence, preserves each row's own GT label
    and the group's pair identity. Direct port of runs/p26's construction."""
    gen = torch.Generator().manual_seed(seed)
    out = score.clone()
    for p, rows in Gs.rows_of.items():
        if len(rows) < 2:
            continue
        rows_t = torch.tensor(rows)
        perm = rows_t[torch.randperm(len(rows_t), generator=gen)]
        out[rows_t] = score[perm]
    return out


def bootstrap_ci(vals: List[float], boot: int = 200, seed: int = 1):
    if not vals:
        return [float("nan"), float("nan")]
    v = torch.tensor(vals, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                      for _ in range(boot)])
    return torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()


def geometry_features_raw(obj_boxes: List[torch.Tensor],
                          pairs_local: List[torch.Tensor]) -> torch.Tensor:
    """Byte-identical to GEO.geometry_features_raw(B), adapted from the
    Mech-cache's B.meta["obj_boxes"]/B.meta["pairs"] shape to this project's
    plain list-of-tensors extraction (extract_geometry.py)."""
    feats = []
    for boxes, pairs in zip(obj_boxes, pairs_local):
        if boxes.numel() == 0 or pairs.numel() == 0:
            continue
        W = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
        H = float(boxes[:, 3].max() - boxes[:, 1].min()) or 1.0
        feats.append(GEO._geom(boxes[pairs[:, 0].long()], boxes[pairs[:, 1].long()], W, H))
    return torch.cat(feats, 0)


def wprd_stratified_by_bucket(Gs, score: torch.Tensor, bucket_of: Dict[int, str],
                              cap: int = CAP, seed: int = SEED) -> Dict[str, Any]:
    """Same cell loop/cap/seed/AUC as CW.wprd_generic, additionally tagged by
    the (bucket_a, bucket_b) of each cell -- the same extension technique
    tools/wprd_stratified.py applies to within_pair_discrimination.wprd()."""
    gen = torch.Generator().manual_seed(seed)
    buckets: Dict[str, Dict[str, list]] = defaultdict(lambda: {"vals": [], "wts": []})
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        rows = torch.tensor(Gs.rows_of[p])
        yy = Gs.y[rows]
        byc = {c: rows[yy == c] for c in cls}
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                ra, rb = byc[a], byc[b]
                if len(ra) > cap:
                    ra = ra[torch.randperm(len(ra), generator=gen)[:cap]]
                if len(rb) > cap:
                    rb = rb[torch.randperm(len(rb), generator=gen)[:cap]]
                sa = score[ra, a] - score[ra, b]
                sb = score[rb, a] - score[rb, b]
                v = AUC(sa.double(), sb.double())
                w = len(ra) * len(rb)
                key = "-".join(sorted([bucket_of[a], bucket_of[b]]))
                buckets[key]["vals"].append(v)
                buckets[key]["wts"].append(w)
    out = {}
    for key, d in buckets.items():
        v = torch.tensor(d["vals"], dtype=torch.float64)
        w = torch.tensor(d["wts"], dtype=torch.float64)
        lo, hi = bootstrap_ci(d["vals"])
        out[key] = {"wprd_macro": float(v.mean()), "wprd_weighted": float((v * w).sum() / w.sum()),
                    "n_cells": len(d["vals"]), "n_comparisons": int(w.sum()), "ci95": [lo, hi]}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(REPO_ROOT.parent / "external_models/runs/wprd_pairs_full.pt"))
    ap.add_argument("--geometry", default=str(REPO_ROOT.parent / "external_models/runs/geometry_pairs_full.pt"))
    ap.add_argument("--out", default="runs/cross_model_imp_plus/decomposition.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2-geom", type=float, default=1e-4)
    ap.add_argument("--l2-ridge", type=float, default=1e-3)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("IMP+ WPRD DECOMPOSITION -- CPU only, cache read-only, NO GPU")
    _log("=" * 100)

    d = torch.load(args.pairs, map_location="cpu", weights_only=False)
    model_term, gt_y = d["model_term"], d["gt_y"]
    subj_label, obj_label = d["subj_label"], d["obj_label"]
    n = model_term.shape[0]
    _log(f"loaded {n:,} GT-aligned pairs from {args.pairs}")

    Gs = CW.SimpleGroups(subj_label, obj_label, gt_y)
    pair_id = torch.empty(n, dtype=torch.long)
    for p, rows in Gs.rows_of.items():
        pair_id[torch.tensor(rows)] = p
    n_decidable_groups = sum(1 for p in range(Gs.G) if len(Gs.classes_of[p]) >= 2)
    n_decidable_rows = sum(len(Gs.rows_of[p]) for p in range(Gs.G) if len(Gs.classes_of[p]) >= 2)
    n_singleton = sum(1 for rows in Gs.rows_of.values() if len(rows) < 2)
    _log(f"  n_groups={Gs.G:,}  n_singleton={n_singleton:,}  "
         f"n_decidable_groups={n_decidable_groups:,} ({100*n_decidable_groups/Gs.G:.1f}%)  "
         f"n_decidable_rows={n_decidable_rows:,} ({100*n_decidable_rows/n:.1f}%)")

    res: Dict[str, Any] = {
        "tool": "decompose_imp_plus", "pairs": args.pairs, "geometry": args.geometry,
        "n_rows": n, "n_groups": Gs.G, "n_singleton_groups": n_singleton,
        "n_decidable_groups": n_decidable_groups, "n_decidable_rows": n_decidable_rows,
        "arms": {},
    }

    # ---------- A/B/D: model, prior control, pair-matched null, random null ----------
    _log(f"\n{'-'*100}\n  ARMS -- model / prior control / pair-matched null / random null\n{'-'*100}")
    prior = CW.empirical_group_prior(Gs, n, 50)
    gen = torch.Generator().manual_seed(SEED)
    random_null = torch.randn(model_term.shape, generator=gen)
    pmn = pair_matched_null(Gs, model_term)

    arms = {
        "model (IMP+ rel_fc/motifs head)": model_term,
        "prior_control (empirical within-group frequency, must be exactly 0.5)": prior,
        "pair_matched_null (model score permuted WITHIN group, image destroyed)": pmn,
        "random_null (iid gaussian, must be ~0.5)": random_null,
    }
    for name, s in arms.items():
        r = CW.wprd_generic(Gs, s, CAP, SEED)
        res["arms"][name] = {k: v for k, v in r.items()}
        _log(f"  {name:<62} macro={r['wprd_macro']:.4f} weighted={r['wprd_weighted']:.4f} "
             f"n_cells={r['n_cells']:,}")

    model_macro = res["arms"]["model (IMP+ rel_fc/motifs head)"]["wprd_macro"]
    assert abs(model_macro - 0.620539) < 1e-3, (
        f"model WPRD {model_macro:.6f} does not match the registered "
        f"wprd_result_full.json value 0.620539 -- grouping mismatch, do not trust the rest")
    _log(f"  [gate] model WPRD reproduces the registered 0.620539 to 1e-3: PASS")
    prior_macro = res["arms"]["prior_control (empirical within-group frequency, must be exactly 0.5)"]["wprd_macro"]
    assert abs(prior_macro - 0.5) < 1e-9, f"prior control {prior_macro} != 0.5 exactly"
    _log(f"  [gate] prior control reads exactly 0.500000: PASS")

    # ---------- variance split + invariance gate ----------
    _log(f"\n{'-'*100}\n  VARIANCE SPLIT (between-group = pair identity, within-group = the only "
         f"place image conditioning can live)\n{'-'*100}")
    gm = group_mean(model_term, pair_id, Gs.G)
    model_centered = model_term - gm
    tot = float(model_term.var(unbiased=False))
    wit = float(model_centered.var(unbiased=False))
    between_share = 1 - wit / tot
    r_centered = CW.wprd_generic(Gs, model_centered, CAP, SEED)
    invariance_ok = abs(r_centered["wprd_macro"] - model_macro) < 1e-9
    res["variance_split"] = {"between_group_share": between_share, "within_group_share": wit / tot,
                             "invariance_gate_pass": bool(invariance_ok),
                             "wprd_raw": model_macro, "wprd_within_group_centered": r_centered["wprd_macro"]}
    _log(f"  between-group (pair identity) {100*between_share:>6.2f}%   within-group {100*(wit/tot):>6.2f}%")
    _log(f"  [gate] WPRD(raw) == WPRD(within-group-centered): "
         f"{model_macro:.6f} vs {r_centered['wprd_macro']:.6f}  "
         f"{'PASS' if invariance_ok else 'FAIL'}")

    # ---------- C: geometry ----------
    _log(f"\n{'-'*100}\n  GEOMETRY -- cross-fitted probe (5-fold, image-level; NO train split "
         f"is converted for this model, see caveat in the result doc)\n{'-'*100}")
    g = torch.load(args.geometry, map_location="cpu", weights_only=False)
    assert torch.equal(g["gt_y"], gt_y), "geometry file gt_y does not match pairs file -- alignment broken"
    Xr = geometry_features_raw(g["obj_boxes"], g["pairs_local"])
    assert Xr.shape[0] == n, f"geometry row count {Xr.shape[0]} != pairs row count {n}"
    X, _ = GEO._standardise(Xr)
    fold = torch.tensor([GEO.CSP.fold_of_image(str(int(iid)), 5, 0) for iid in g["image_id"]])
    _log(f"  geometry features: {X.shape[1]-1} + bias, fold sizes {[int((fold==f).sum()) for f in range(5)]}")

    geo_logits = GEO.cross_fit_logits(X, gt_y, fold, 50, args.epochs, args.l2_geom)
    ysh = gt_y[torch.randperm(n, generator=torch.Generator().manual_seed(7))]
    geo_sh_logits = GEO.cross_fit_logits(X, ysh, fold, 50, args.epochs, args.l2_geom)
    for name, s in {
        "geometry_crossfit (19 box numbers, cross-fitted, no train split available)": geo_logits,
        "geometry_crossfit_SHUFFLED_labels (must be ~0.5)": geo_sh_logits,
    }.items():
        r = CW.wprd_generic(Gs, s, CAP, SEED)
        res["arms"][name] = r
        _log(f"  {name:<62} macro={r['wprd_macro']:.4f} weighted={r['wprd_weighted']:.4f}")

    # ---------- within-group layout vs residual (ridge, p42-style) ----------
    _log(f"\n{'-'*100}\n  WITHIN-GROUP LAYOUT/RESIDUAL SPLIT (does the within-pair signal REDUCE "
         f"to geometry?)\n{'-'*100}")
    Xw = X - group_mean(X, pair_id, Gs.G)
    Mw = model_centered
    pred = torch.zeros_like(Mw)
    for f in range(5):
        te = fold == f
        tr = ~te
        A = Xw[tr]
        AtA = A.T @ A + args.l2_ridge * A.shape[0] * torch.eye(A.shape[1])
        Wm = torch.linalg.solve(AtA, A.T @ Mw[tr])
        pred[te] = Xw[te] @ Wm
    resid = Mw - pred
    ss_tot = float((Mw ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
    w_lay = CW.wprd_generic(Gs, pred, CAP, SEED)
    w_res = CW.wprd_generic(Gs, resid, CAP, SEED)
    res["layout_residual_split"] = {
        "oof_r2_of_geometry_on_within_group_model": r2,
        "wprd_within_group_total": model_macro,
        "wprd_layout_predictable_part": w_lay["wprd_macro"],
        "wprd_residual_nonlayout_part": w_res["wprd_macro"],
    }
    _log(f"  out-of-fold R^2 of box geometry predicting the within-group model term: {r2*100:.2f}%")
    _log(f"  WPRD of the whole within-group term    {model_macro:.4f}")
    _log(f"  WPRD of its LAYOUT-predictable part     {w_lay['wprd_macro']:.4f}")
    _log(f"  WPRD of the RESIDUAL (non-layout) part  {w_res['wprd_macro']:.4f}")

    # ---------- head/body/tail ----------
    _log(f"\n{'-'*100}\n  HEAD/BODY/TAIL (buckets from THIS split's own GT counts: top15/next20/last15, "
         f"same rule as tools/cprime_analysis.py)\n{'-'*100}")
    cnt = torch.bincount(gt_y, minlength=50)
    order = torch.argsort(cnt, descending=True)
    head_set, body_set = set(order[:15].tolist()), set(order[15:35].tolist())
    bucket_of = {i: ("head" if i in head_set else "body" if i in body_set else "tail") for i in range(50)}
    res["bucket_of"] = bucket_of
    for name, s in {"model": model_term, "geometry_crossfit": geo_logits,
                    "random_null": random_null}.items():
        strat = wprd_stratified_by_bucket(Gs, s, bucket_of)
        res.setdefault("head_body_tail", {})[name] = strat
        _log(f"\n  {name}")
        for key in ("head-head", "body-head", "body-body", "head-tail", "body-tail", "tail-tail"):
            if key in strat:
                v = strat[key]
                _log(f"    {key:<12} macro={v['wprd_macro']:.4f}  n_cells={v['n_cells']:>5}  "
                     f"n_comparisons={v['n_comparisons']:>8,}  95% CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")

    # ---------- prior-conflict / override: N/A for this checkpoint ----------
    res["prior_override"] = {
        "available": False,
        "reason": ("this checkpoint has use_bias=False, test_bias=False, and no freq_bias.* "
                   "keys in its state_dict (confirmed at extraction time, docs/CROSS_MODEL_"
                   "IMP_PLUS_RESULT.md); its score is the motifs/message-passing head alone, "
                   "not a blend with any learned or frequency prior, so there is no prior term "
                   "for the model to conflict with or override -- runs/p63's prior-override "
                   "diagnostic has no analogue here."),
    }
    _log(f"\n  PRIOR-CONFLICT / OVERRIDE: not available -- {res['prior_override']['reason']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
