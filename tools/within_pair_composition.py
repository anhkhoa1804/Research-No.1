#!/usr/bin/env python
"""Amplify only the WITHIN-group part of the model term. Does it recover rows?

Pre-registered in docs/WITHIN_PAIR_COMPOSITION_PREREGISTRATION.md.

    term(lambda) = between_group(model) + lambda * within_group(model)
    score        = 3.75 * (prior - tau*logP) + term(lambda)

lambda = 1 is exactly the deployed system. No labels enter the construction; the
group mean is a label-free function of the model term. It IS transductive -- it
reads other validation rows' model terms -- so this is a HEADROOM DIAGNOSTIC,
not a deployable method, and a fold-restricted non-transductive variant is
reported alongside.

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
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST
N_FOLDS = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


def split(X, gid, G, train=None):
    """between = group mean (from `train` rows if given), within = X - between."""
    src = torch.ones(X.shape[0], dtype=torch.bool) if train is None else train
    cnt = torch.zeros(G).index_add_(0, gid[src], torch.ones(int(src.sum())))
    s = torch.zeros(G, X.shape[1]).index_add_(0, gid[src], X[src])
    gm = torch.where(cnt.unsqueeze(1) > 0, s / cnt.clamp_min(1).unsqueeze(1),
                     X[src].mean(0, keepdim=True))[gid]
    return gm, X - gm


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p45_within_pair_composition/comp.json")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--lambdas", default="1.0,1.5,2.0,3.0,5.0,8.0,12.0")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args(argv)

    lams = [float(x) for x in args.lambdas.split(",")]
    _log("=" * 108)
    _log("WITHIN-PAIR COMPOSITION -- headroom diagnostic. CPU only, NO GPU")
    _log("=" * 108)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    gid, G, y = Gs.pair_id, Gs.G, B.gt_y
    n = int(y.numel())
    FLOOR = float(B.metrics(B.score(args.tau, ALPHA_HIST, None))["R"])
    pred_prior = B.predict(B.score(args.tau, ALPHA_HIST, None))
    adv, ok = pred_prior != y, pred_prior == y
    na, no = int(adv.sum()), int(ok.sum())
    _log(f"  rows {n:,}  groups {G:,}   tau={args.tau}")
    _log(f"  prior-adversarial {na:,} ({na/n*100:.1f}%)   floor = prior R@50 "
         f"{FLOOR*100:.3f}")

    g = torch.Generator().manual_seed(0)
    perm_all = torch.randperm(n, generator=g)
    pm = torch.arange(n)
    for q in gid.unique().tolist():
        ix = (gid == q).nonzero().squeeze(1)
        if ix.numel() > 1:
            pm[ix] = ix[torch.randperm(ix.numel(), generator=g)]

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    Wg = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([Wg], max_iter=args.epochs, history_size=10,
                            line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ Wg, yt) + 1e-4 * (Wg * Wg).sum()
        l.backward()
        return l

    opt.step(cl)
    geo = CPA.Bench._norm((Xv @ Wg).detach())

    arms = {"real": B.model, "null_shuffled": B.model[perm_all],
            "null_pair_matched": B.model[pm], "geometry": geo}
    res: Dict[str, Any] = {"tool": "within_pair_composition", "tau": args.tau,
                           "floor": FLOOR, "n_adversarial": na, "arms": {}}
    _log(f"\n{'-'*108}")
    _log(f"  {'arm':>18} {'lambda':>7} {'adv fixed':>18} {'kept':>18} "
         f"{'net':>8} {'R@50':>9} {'floor':>6}")
    _log(f"{'-'*108}")
    for name, M in arms.items():
        bet, wit = split(M, gid, G)
        rows = []
        for lam in lams:
            p = B.predict(B.score(args.tau, ALPHA_HIST, bet + lam * wit))
            fx = int(((p == y) & adv).sum())
            kp = int(((p == y) & ok).sum())
            R = float((p == y).float().mean())
            rows.append({"lambda": lam, "fixed": fx, "fixed_pct": fx / na * 100,
                         "kept": kp, "net": (fx + kp) - no, "R": R,
                         "floor_ok": bool(R >= FLOOR)})
            _log(f"  {name:>18} {lam:>7.1f} {fx:>8,} ({fx/na*100:>5.2f}%) "
                 f"{kp:>10,} ({kp/no*100:>5.2f}%) {(fx+kp)-no:>+8,} {R*100:>9.3f} "
                 f"{'ok' if R >= FLOOR else 'FAIL':>6}")
        res["arms"][name] = rows
        _log("")

    # ---- verdict against the registered criterion ----
    def best_gain(name):
        r = res["arms"][name]
        base = r[0]["fixed_pct"]
        elig = [x for x in r if x["floor_ok"] and x["lambda"] > 1.0]
        if not elig:
            return None, base, None
        b = max(elig, key=lambda x: x["fixed_pct"])
        return b["fixed_pct"] - base, base, b

    gr, base_r, br = best_gain("real")
    gs, _, _ = best_gain("null_shuffled")
    gp, _, _ = best_gain("null_pair_matched")
    _log(f"{'-'*108}\n  PRE-REGISTERED VERDICT\n{'-'*108}")
    _log(f"    lambda=1 (deployed) adversarial-fixed  {base_r:.2f}%")
    if gr is None:
        verdict = "REFUTED (no lambda>1 holds the floor)"
        _log(f"    no lambda>1 holds the R@50 floor")
    else:
        _log(f"    best floor-holding lambda={br['lambda']}  fixed "
             f"{br['fixed_pct']:.2f}%   gain {gr:+.2f} pts   R@50 {br['R']*100:.3f}")
        _log(f"    null_shuffled best gain     {('%+.2f' % gs) if gs is not None else 'n/a'} pts")
        _log(f"    null_pair_matched best gain {('%+.2f' % gp) if gp is not None else 'n/a'} pts")
        margin = gr - (gs if gs is not None else -99)
        if gr >= 2.0 and margin >= 1.0:
            verdict = "SUPPORTED"
        elif gr >= 0.5 and margin >= 1.0:
            verdict = "WEAK"
        else:
            verdict = "REFUTED"
        if gp is not None and gp >= gr - 0.5:
            verdict += " -- but null_pair_matched matches: recovered rows are PAIR IDENTITY"
    res["verdict"] = {"verdict": verdict, "real_gain": gr,
                      "null_shuffled_gain": gs, "null_pair_matched_gain": gp}
    _log(f"\n    VERDICT: {verdict}")
    _log(f"\n    NOTE: transductive (group means read other validation rows). "
         f"Headroom diagnostic, not a deployable method.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
