#!/usr/bin/env python
"""What is the within-group signal MADE OF? Layout, or something else?

A fact that reframes everything, and follows from the definition rather than
from a measurement:

    WPRD is invariant to any quantity that is CONSTANT WITHIN A GROUP.

It scores (score[i,a]-score[i,b]) against (score[j,a]-score[j,b]) for i, j in
the same (subject,object) group, so subtracting that group's mean from every
row changes nothing. Therefore WPRD(model term) == WPRD(within-group component)
exactly, and the 82.6% of the model term's variance that lives BETWEEN groups
contributes precisely zero to it.

That is the clean separation the p32 interpretation needs:

  between-group variance (82.6%)  -> pair identity. Drives R@50 through the
                                     additive composition. Invisible to WPRD.
  within-group variance  (17.4%)  -> the ONLY place image conditioning can
                                     live. This is all WPRD measures.

But within-group is not the same as "visual reasoning". Bounding boxes vary
within a group too, and they are image-derived. So this tool splits the
within-group component again:

  within-group = (part linearly predictable from box geometry) + residual

cross-fitted so no row helps predict itself, and then measures WPRD of each
part separately. If the residual's WPRD falls to chance, the checkpoint's
entire within-pair discrimination is layout. If it does not, there is
non-layout image-conditioned evidence, and its size is measured here.

CPU only. Cache read-only. No GPU.
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
GEO = _load("wprd_geometry_control")
CSP = _load("candidate_scorer_probe")
Mech = MECH.Mech
N_FOLDS = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


def group_center(X: torch.Tensor, gid: torch.Tensor, G: int):
    cnt = torch.zeros(G).index_add_(0, gid, torch.ones(X.shape[0]))
    s = torch.zeros(G, X.shape[1]).index_add_(0, gid, X)
    gm = (s / cnt.clamp_min(1).unsqueeze(1))[gid]
    return X - gm, gm


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p42_within_group_decomp/decomp.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args(argv)

    _log("=" * 104)
    _log("WITHIN-GROUP DECOMPOSITION -- layout vs the rest. CPU only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    gid, G, n = Gs.pair_id, Gs.G, Gs.n
    _log(f"  rows {n:,}  groups {G:,}")

    # ---------- 1. variance split of each head ----------
    _log(f"\n{'-'*104}\n  1. VARIANCE SPLIT (this is what p32's 'pair prior' "
         f"vs 'within-group' means)\n{'-'*104}")
    res: Dict[str, Any] = {"tool": "within_group_decomposition", "variance": {}}
    heads = {"text_head": B.fixed_ensemble(0.0), "classifier_head": B.fixed_ensemble(1.0)}
    for name, M in heads.items():
        W, _ = group_center(M, gid, G)
        tot = float(M.var(unbiased=False))
        wit = float(W.var(unbiased=False))
        res["variance"][name] = {"between_share": 1 - wit / tot,
                                 "within_share": wit / tot}
        _log(f"  {name:>18}  between-group (pair identity) {100*(1-wit/tot):>6.2f}%   "
             f"within-group {100*wit/tot:>6.2f}%")

    # ---------- 2. WPRD is invariant to the between-group part ----------
    _log(f"\n{'-'*104}\n  2. GATE -- WPRD must be IDENTICAL on the raw term and "
         f"its within-group part\n{'-'*104}")
    for name, M in heads.items():
        Wc, _ = group_center(M, gid, G)
        a = WPD.wprd(Gs, M, args.cap)["wprd_macro"]
        b = WPD.wprd(Gs, Wc, args.cap)["wprd_macro"]
        ok = abs(a - b) < 1e-9
        res.setdefault("invariance_gate", {})[name] = {"raw": a, "centred": b,
                                                       "pass": bool(ok)}
        _log(f"  {name:>18}  raw {a:.6f}   within-group-centred {b:.6f}   "
             f"|diff| {abs(a-b):.2e}  {'PASS' if ok else 'FAIL'}")
    _log("  => the 82.6% between-group share contributes EXACTLY ZERO to WPRD.")
    _log("     WPRD and R@50 are therefore reading different halves of the term.")

    # ---------- 3. split within-group into layout + residual ----------
    _log(f"\n{'-'*104}\n  3. IS THE WITHIN-GROUP PART LAYOUT? cross-fitted, "
         f"out-of-fold\n{'-'*104}")
    Xr = GEO.geometry_features_raw(B)
    Xs, _ = GEO._standardise(Xr)
    Xw, _ = group_center(Xs, gid, G)          # within-group geometry variation
    per_image = [CSP.fold_of_image(str(s), N_FOLDS, 0) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per_image[i]] * len(B.meta["pairs"][i]))
    fold = torch.tensor(flat)[B.gt_row]
    _log(f"  fold sizes {[int((fold == f).sum()) for f in range(N_FOLDS)]}")

    for name, M in heads.items():
        Mw, _ = group_center(M, gid, G)
        pred = torch.zeros_like(Mw)
        for f in range(N_FOLDS):
            te = fold == f
            tr = ~te
            A = Xw[tr]
            # ridge closed form: (A'A + lI)^-1 A' y
            AtA = A.T @ A + args.l2 * A.shape[0] * torch.eye(A.shape[1])
            Wm = torch.linalg.solve(AtA, A.T @ Mw[tr])
            pred[te] = Xw[te] @ Wm
        resid = Mw - pred
        ss_tot = float((Mw ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
        w_all = WPD.wprd(Gs, Mw, args.cap)
        w_lay = WPD.wprd(Gs, pred, args.cap)
        w_res = WPD.wprd(Gs, resid, args.cap)

        def ci(vals):
            v = torch.tensor(vals, dtype=torch.float64)
            g = torch.Generator().manual_seed(1)
            bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                              for _ in range(args.boot)])
            return torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                   dtype=torch.float64)).tolist()

        c_all, c_lay, c_res = ci(w_all["_vals"]), ci(w_lay["_vals"]), ci(w_res["_vals"])
        res.setdefault("layout_split", {})[name] = {
            "oof_r2_of_geometry_on_within": r2,
            "wprd_within": [w_all["wprd_macro"], c_all],
            "wprd_layout_part": [w_lay["wprd_macro"], c_lay],
            "wprd_residual_part": [w_res["wprd_macro"], c_res]}
        _log(f"\n  {name}")
        _log(f"    out-of-fold R^2 of box geometry predicting the within-group "
             f"term: {r2*100:>6.2f}%")
        _log(f"    WPRD of the whole within-group term      {w_all['wprd_macro']:.4f} "
             f"[{c_all[0]:.4f}, {c_all[1]:.4f}]")
        _log(f"    WPRD of its LAYOUT-predictable part      {w_lay['wprd_macro']:.4f} "
             f"[{c_lay[0]:.4f}, {c_lay[1]:.4f}]")
        _log(f"    WPRD of the RESIDUAL (non-layout) part   {w_res['wprd_macro']:.4f} "
             f"[{c_res[0]:.4f}, {c_res[1]:.4f}]"
             f"   {'<-- contains chance' if c_res[0] <= 0.5 <= c_res[1] else '<-- excludes chance'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
