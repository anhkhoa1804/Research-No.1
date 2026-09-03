#!/usr/bin/env python
"""Corrected nested-selection arm for `tools/geometry_sgg_baseline.py`.

`geometry_sgg_baseline.py`'s nested arm selects the geometry mixing weight `w`
with `key = (m["mR"],)` -- mR alone. Geometry always lowers R@50 relative to
`prior+MODEL` at low tau, so that key picks `w = 0.0` on every fold at every
tau (confirmed identically on both validation, `runs/p41`, and test,
`runs/p64_test_geometry_sgg`). This was flagged as broken, not re-run, in
`docs/GEOMETRY_SGG_BASELINE_RESULT.md` limitation 1: "The rest of this project
selects on `(R >= floor, mR)`."

This script is that correction, run separately so the original (bug-faithful)
number is never silently replaced. Floor = prior-only R@50 AT THAT TAU (the
arm must not cost recall relative to the prior alone, per
`docs/RESEARCH_STATE.md` section 9's stated invariant), not the fixed 0.665
literal used in `tools/candidate_scorer_probe.py`'s single-tau nested design
-- a fixed floor is meaningless once prior-only R@50 itself falls below it at
higher tau (61.8% at tau=0.2 in this cache).

CPU only. Cache read-only. No GPU. No retraining of the geometry probe: same
fit as the base tool, on `--train-jsonl` only.
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
GEO = _load("wprd_geometry_control")
CSP = _load("candidate_scorer_probe")
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST
N_FOLDS = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", default="0.0,0.05,0.1,0.2")
    ap.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    taus = [float(t) for t in args.taus.split(",")]
    ws = [float(w) for w in args.weights.split(",")]

    _log("=" * 108)
    _log("CORRECTED NESTED GEOMETRY -- key = (R >= prior_only R at this tau, mR). CPU only, NO GPU")
    _log("=" * 108)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    _log(f"  images={B.n_images} GT rows={B.n_gt}  [gate] model-term identity {e:.3e} OK")

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tstats = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tstats)
    _log(f"  geometry fitted on TRAIN {Xt.shape[0]:,} rows -> eval {Xv.shape[0]:,} rows"
         f"   ({Xt.shape[1] - 1} features + bias; no eval-split statistic in the fit)")
    W = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=args.epochs, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W, yt)
                + args.l2 * (W * W).sum())
        loss.backward()
        return loss

    opt.step(closure)
    geo = CPA.Bench._norm((Xv @ W).detach())
    model = B.model

    per_image = [CSP.fold_of_image(str(s), N_FOLDS, 0) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per_image[i]] * len(B.meta["pairs"][i]))
    fold = torch.tensor(flat)[B.gt_row]

    res: Dict[str, Any] = {"tool": "geometry_sgg_nested_corrected", "taus": taus,
                           "weights": ws, "train_rows": int(Xt.shape[0]), "by_tau": {}}
    for tau in taus:
        curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
                 for t in CPA.TAUS]
        prior_only = B.metrics(B.score(tau, ALPHA_HIST, None))
        floor = prior_only["R"]
        mm = B.metrics(B.score(tau, ALPHA_HIST, model))
        gm0 = CPA.pareto_gap(curve, mm["R"], mm["mR"])

        picks, chosen = [], torch.zeros(B.n_gt, B.n_classes)
        for f in range(N_FOLDS):
            te = fold == f
            tr = ~te
            best_w, best_key = ws[0], None
            for w in ws:
                m = B.metrics(B.score(tau, ALPHA_HIST, w * geo), tr)
                key = (m["R"] >= floor, m["mR"])  # corrected: floor first, then mR
                if best_key is None or key > best_key:
                    best_key, best_w = key, w
            picks.append(best_w)
            chosen[te] = (best_w * geo)[te]
        gm_n = B.metrics(B.score(tau, ALPHA_HIST, chosen))
        p_n = CPA.pareto_gap(curve, gm_n["R"], gm_n["mR"])
        meets_floor = bool(gm_n["R"] >= floor)

        res["by_tau"][str(tau)] = {
            "prior_only_R": prior_only["R"], "floor": floor,
            "model_pareto": gm0,
            "nested_geometry_corrected": {
                "R": gm_n["R"], "mR": gm_n["mR"], "pareto": p_n,
                "meets_floor": meets_floor, "w_per_fold": picks},
        }
        _log(f"  tau={tau:<5} floor(R>=prior)={floor*100:.3f}  "
             f"nested prior+GEOMETRY(corrected) R@50 {gm_n['R']*100:.3f} "
             f"mR {gm_n['mR']*100:.3f} pareto {p_n:+.3f} floor {'ok' if meets_floor else 'FAIL'} "
             f"w={picks}   |  prior+MODEL pareto {gm0:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
