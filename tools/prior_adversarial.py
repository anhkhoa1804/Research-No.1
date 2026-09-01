#!/usr/bin/env python
"""C7 -- can the model override the prior when the image contradicts it?

WPRD asks whether a score can SEPARATE two relations inside one (s,o) group.
This asks the decision-level question that a user of an SGG system actually
cares about: on the rows where following P(p|s,o) is WRONG, does the model fix
them, and at what cost on the rows where the prior was already right?

Population, defined without touching validation labels: the prior is
train-derived, so `argmax_p prior[row]` is a pure function of (subject, object)
and of the training split. A row is PRIOR-ADVERSARIAL when its GT differs from
that argmax. On those rows the prior scores 0% by construction.

Arms are the exact evaluator composition, `alpha*(prior - tau*logP) + term`:

  prior only          0% on adversarial rows by construction
  + MODEL             the checkpoint
  + GEOMETRY          train-fitted 19-feature box probe (SpatialSense's 2D-only
                      baseline, reconstructed)
  + MODEL + GEOMETRY

and the cost side is reported with the benefit side, because a method that
fixes adversarial rows by breaking prior-correct ones has not improved anything.

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
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p43_prior_adversarial/adv.json")
    ap.add_argument("--taus", default="0.0,0.05,0.1")
    ap.add_argument("--weights", default="0.25,0.5,1.0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    taus = [float(t) for t in args.taus.split(",")]
    ws = [float(w) for w in args.weights.split(",")]
    _log("=" * 104)
    _log("C7 PRIOR-ADVERSARIAL SUBSET -- CPU only, cache read-only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4
    y = B.gt_y
    n = int(y.numel())

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tstats = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tstats)
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

    res: Dict[str, Any] = {"tool": "prior_adversarial", "by_tau": {}}
    for tau in taus:
        pred_prior = B.predict(B.score(tau, ALPHA_HIST, None))
        adv = pred_prior != y                 # prior is WRONG here, by definition
        ok = ~adv
        na, no = int(adv.sum()), int(ok.sum())
        _log(f"\n{'-'*104}")
        _log(f"  tau={tau}   PRIOR-ADVERSARIAL rows {na:,} ({na/n*100:.1f}%)   "
             f"prior-correct rows {no:,} ({no/n*100:.1f}%)")
        _log(f"  the prior scores 0.00% on the adversarial rows by construction")
        _log(f"{'-'*104}")
        _log(f"  {'arm':>34} {'ADVERSARIAL fixed':>19} {'prior-correct kept':>20} "
             f"{'net rows':>10} {'overall R@50':>13}")
        rows = []

        def report(name: str, term):
            p = B.predict(B.score(tau, ALPHA_HIST, term))
            fixed = int(((p == y) & adv).sum())
            kept = int(((p == y) & ok).sum())
            net = (fixed + kept) - no
            R = float((p == y).float().mean())
            rows.append({"arm": name, "fixed": fixed, "fixed_frac": fixed / max(1, na),
                         "kept": kept, "kept_frac": kept / max(1, no),
                         "net": net, "R": R})
            _log(f"  {name:>34} {fixed:>8,} ({fixed/max(1,na)*100:>5.2f}%) "
                 f"{kept:>10,} ({kept/max(1,no)*100:>6.2f}%) {net:>+10,} {R*100:>13.3f}")

        report("prior only", None)
        report("+ MODEL", model)
        for w in ws:
            report(f"+ GEOMETRY (w={w})", w * geo)
        for w in ws:
            report(f"+ MODEL + GEOMETRY (w={w})", model + w * geo)
        res["by_tau"][str(tau)] = {"n_adversarial": na, "n_prior_correct": no,
                                   "arms": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
