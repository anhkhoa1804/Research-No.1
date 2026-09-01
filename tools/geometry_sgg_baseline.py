#!/usr/bin/env python
"""Can (frequency prior + two bounding boxes) match PURE on VG150 PredCls?

Every result in this cycle so far is measured with WPRD, a metric this project
invented. That is the right instrument for the question, and it is also a fair
objection: a new metric can always be built to make a model look bad. So this
run drops the new metric entirely and uses the FIELD'S OWN one -- R@50 / mR@50
under the exact composition the evaluator uses:

    score = freq_bias_alpha * (prior - tau * log P(p)) + model_term

and swaps `model_term` for a train-fitted linear probe on 19 box-geometry
features, normalised exactly the way the evaluator normalises its own heads
(`_normalize_eval_logits`, per-row standardisation).

Arms:
  prior_only          the frequency prior alone
  prior + MODEL       the checkpoint, i.e. what runs/p24 reports
  prior + GEOMETRY    the same prior, with rectangles instead of the network
  prior + MODEL + GEO does the network add anything ON TOP of geometry?

The geometry probe is fitted on datasets_vg150_clean/train.jsonl and never sees
a validation statistic, so it is under exactly the constraint the checkpoint was
under. The mixing weight w IS selected on validation, which favours geometry;
that is stated with the result and a nested cross-fitted number is also
reported, in which w is chosen inside training folds only.

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
GEO = _load("wprd_geometry_control")
CSP = _load("candidate_scorer_probe")
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST
N_FOLDS = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p41_geometry_sgg/base.json")
    ap.add_argument("--taus", default="0.0,0.05,0.1,0.2")
    ap.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    taus = [float(t) for t in args.taus.split(",")]
    ws = [float(w) for w in args.weights.split(",")]

    _log("=" * 108)
    _log("GEOMETRY AS AN SGG BASELINE -- the field's own metric. CPU only, NO GPU")
    _log("=" * 108)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    _log(f"  images={B.n_images} GT rows={B.n_gt}  [gate] model-term identity {e:.3e} OK")

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tstats = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tstats)
    _log(f"  geometry fitted on TRAIN {Xt.shape[0]:,} rows -> val {Xv.shape[0]:,} rows"
         f"   ({Xt.shape[1] - 1} features + bias; no validation statistic in the fit)")
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
    # normalise EXACTLY as the evaluator normalises its own heads
    geo = CPA.Bench._norm((Xv @ W).detach())
    model = B.model

    res: Dict[str, Any] = {"tool": "geometry_sgg_baseline", "taus": taus,
                           "weights": ws, "train_rows": int(Xt.shape[0]),
                           "by_tau": {}}
    best_overall = None
    for tau in taus:
        curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
                 for t in CPA.TAUS]
        base = B.metrics(B.score(tau, ALPHA_HIST, None))
        mm = B.metrics(B.score(tau, ALPHA_HIST, model))
        gm = CPA.pareto_gap(curve, mm["R"], mm["mR"])
        _log(f"\n{'-'*108}")
        _log(f"  tau={tau}   prior-only R@50 {base['R']*100:.3f} mR {base['mR']*100:.3f}")
        _log(f"  prior + MODEL (this is what runs/p24 reports): "
             f"R@50 {mm['R']*100:.3f}  mR {mm['mR']*100:.3f}  pareto {gm:+.3f}")
        _log(f"{'-'*108}")
        _log(f"  {'w':>6} {'prior+GEOMETRY':>32} {'':>4} {'prior+MODEL+GEOMETRY':>34}")
        _log(f"  {'':>6} {'R@50':>8} {'mR@50':>8} {'pareto':>8} {'tail':>6} "
             f"{'':>2} {'R@50':>8} {'mR@50':>8} {'pareto':>8} {'tail':>6}")
        rows = []
        for w in ws:
            g = B.metrics(B.score(tau, ALPHA_HIST, w * geo))
            gg = CPA.pareto_gap(curve, g["R"], g["mR"])
            mg = B.metrics(B.score(tau, ALPHA_HIST, model + w * geo))
            mgg = CPA.pareto_gap(curve, mg["R"], mg["mR"])
            rows.append({"w": w, "geo": {"R": g["R"], "mR": g["mR"], "pareto": gg,
                                         "tail_mR": g["tail_mR"]},
                         "model_geo": {"R": mg["R"], "mR": mg["mR"], "pareto": mgg,
                                       "tail_mR": mg["tail_mR"]}})
            _log(f"  {w:>6.2f} {g['R']*100:>8.3f} {g['mR']*100:>8.3f} {gg:>+8.3f} "
                 f"{g['tail_mR']*100:>6.2f}   {mg['R']*100:>8.3f} {mg['mR']*100:>8.3f} "
                 f"{mgg:>+8.3f} {mg['tail_mR']*100:>6.2f}")
        res["by_tau"][str(tau)] = {"prior": base, "model": {**mm, "pareto": gm},
                                   "rows": rows}
        bg = max(rows, key=lambda r: r["geo"]["pareto"] if r["geo"]["pareto"] else -9e9)
        bmg = max(rows, key=lambda r: r["model_geo"]["pareto"] if r["model_geo"]["pareto"] else -9e9)
        _log(f"\n    best prior+GEOMETRY       w={bg['w']:.2f}  pareto {bg['geo']['pareto']:+.3f}"
             f"   vs prior+MODEL {gm:+.3f}   -> geometry "
             f"{'WINS' if bg['geo']['pareto'] > gm else 'loses'} by "
             f"{bg['geo']['pareto'] - gm:+.3f}")
        _log(f"    best prior+MODEL+GEOMETRY w={bmg['w']:.2f}  pareto "
             f"{bmg['model_geo']['pareto']:+.3f}   -> the model adds "
             f"{bmg['model_geo']['pareto'] - bg['geo']['pareto']:+.3f} on top of geometry")

    # ---- nested: w chosen inside training folds only ----
    _log(f"\n{'-'*108}\n  NESTED (w chosen inside training folds; no validation "
         f"selection)\n{'-'*108}")
    per_image = [CSP.fold_of_image(str(s), N_FOLDS, 0) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per_image[i]] * len(B.meta["pairs"][i]))
    fold = torch.tensor(flat)[B.gt_row]
    for tau in taus:
        curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
                 for t in CPA.TAUS]
        picks, sel = [], torch.zeros(B.n_gt, dtype=torch.bool)
        chosen = torch.zeros(B.n_gt, B.n_classes)
        for f in range(N_FOLDS):
            te = fold == f
            tr = ~te
            best_w, best_key = ws[0], None
            for w in ws:
                m = B.metrics(B.score(tau, ALPHA_HIST, w * geo), tr)
                key = (m["mR"],)
                if best_key is None or key > best_key:
                    best_key, best_w = key, w
            picks.append(best_w)
            chosen[te] = (best_w * geo)[te]
        gm_n = B.metrics(B.score(tau, ALPHA_HIST, chosen))
        p_n = CPA.pareto_gap(curve, gm_n["R"], gm_n["mR"])
        mm = B.metrics(B.score(tau, ALPHA_HIST, model))
        gm0 = CPA.pareto_gap(curve, mm["R"], mm["mR"])
        res["by_tau"][str(tau)]["nested_geometry"] = {
            "R": gm_n["R"], "mR": gm_n["mR"], "pareto": p_n, "w_per_fold": picks}
        _log(f"  tau={tau:<5} nested prior+GEOMETRY R@50 {gm_n['R']*100:.3f} "
             f"mR {gm_n['mR']*100:.3f} pareto {p_n:+.3f}  w={picks}"
             f"   |  prior+MODEL pareto {gm0:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
