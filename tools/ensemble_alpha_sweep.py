#!/usr/bin/env python
"""Does the predicate head the checkpoint DISCARDS actually help?

The evaluated checkpoint runs at ensemble_alpha = 0.0:

    model_term = ea * norm(cls_logits)/cls_temp + (1-ea) * norm(text_logits)/text_temp

so at ea=0 the learned predicate classifier contributes exactly nothing and the
text head carries the whole term. Both heads are functions of the SAME
image-derived rel_feat -- text_logits = normalize(rel_feat) @ normalize(pred_emb).T
and cls_logits = predicate_classifier(rel_feat) -- so ea is a choice of readout,
not a choice of whether to use vision.

runs/p33 measured the two heads' prior-free within-pair relational
discrimination and found the DISCARDED head is the better one
(WPRD 0.5728 vs 0.5542, non-overlapping 95% CIs). This asks the operating-point
question that follows: does mixing it back in move R@50/mR@50, and does the
Pareto gap against the tau frontier track WPRD?

Both quantities are computed on the same p24 full-validation cache. CPU only.
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
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p34_ensemble_alpha_sweep/sweep.json")
    ap.add_argument("--alphas", default="0.0,0.1,0.25,0.5,0.75,0.9,1.0")
    ap.add_argument("--taus", default="0.0,0.05,0.1")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--wprd", action="store_true",
                    help="also compute WPRD per alpha (slower)")
    args = ap.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",")]
    taus = [float(t) for t in args.taus.split(",")]

    _log("=" * 104)
    _log("ENSEMBLE-ALPHA SWEEP -- is the discarded classifier head better? CPU only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    _log(f"  images={B.n_images} GT rows={B.n_gt} classes={B.n_classes}  "
         f"[gate] model-term identity {e:.3e} OK")
    _log(f"  checkpoint ran at ensemble_alpha={B.meta.get('ensemble_alpha')}  "
         f"cls_temp={B.meta.get('classifier_temperature')}  "
         f"text_temp={B.meta.get('text_temperature')}")

    Gs = WPD.Groups(B) if args.wprd else None
    res: Dict[str, Any] = {"tool": "ensemble_alpha_sweep", "dump": args.dump,
                           "alphas": alphas, "taus": taus, "by_tau": {}}

    for tau in taus:
        curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
                 for t in CPA.TAUS]
        base = B.metrics(B.score(tau, ALPHA_HIST, None))
        _log(f"\n{'-'*104}")
        _log(f"  tau={tau}   prior-only R@50 {base['R']*100:.3f} mR {base['mR']*100:.3f}")
        _log(f"{'-'*104}")
        hdr = (f"  {'ens_alpha':>10} {'R@50':>8} {'mR@50':>8} {'dR':>8} {'dmR':>8} "
               f"{'pareto':>8} {'head':>7} {'body':>7} {'tail':>7}")
        if args.wprd:
            hdr += f" {'WPRD':>8}"
        _log(hdr)
        rows = []
        for ea in alphas:
            mt = B.fixed_ensemble(ea)
            m = B.metrics(B.score(tau, ALPHA_HIST, mt))
            g = CPA.pareto_gap(curve, m["R"], m["mR"])
            r = {"ensemble_alpha": ea, "R": m["R"], "mR": m["mR"],
                 "dR_points": (m["R"] - base["R"]) * 100.0,
                 "dmR_points": (m["mR"] - base["mR"]) * 100.0,
                 "pareto": g, "head_mR": m["head_mR"], "body_mR": m["body_mR"],
                 "tail_mR": m["tail_mR"]}
            line = (f"  {ea:>10.2f} {m['R']*100:>8.3f} {m['mR']*100:>8.3f} "
                    f"{r['dR_points']:>+8.3f} {r['dmR_points']:>+8.3f} "
                    f"{(g if g is not None else float('nan')):>+8.3f} "
                    f"{m['head_mR']*100:>7.2f} {m['body_mR']*100:>7.2f} {m['tail_mR']*100:>7.2f}")
            if args.wprd and tau == taus[0]:
                w = WPD.wprd(Gs, mt, args.cap)
                r["wprd_macro"] = w["wprd_macro"]
                r["wprd_weighted"] = w["wprd_weighted"]
                line += f" {w['wprd_macro']:>8.4f}"
            elif args.wprd:
                line += f" {'':>8}"
            rows.append(r)
            _log(line)
        res["by_tau"][str(tau)] = {"prior": base, "rows": rows}
        best = max(rows, key=lambda x: (x["pareto"] if x["pareto"] is not None else -9e9))
        bestR = max(rows, key=lambda x: x["R"])
        _log(f"\n    best Pareto : ens_alpha={best['ensemble_alpha']:.2f} "
             f"gap {best['pareto']:+.3f}   (checkpoint runs at 0.00, gap "
             f"{rows[0]['pareto']:+.3f})")
        _log(f"    best R@50   : ens_alpha={bestR['ensemble_alpha']:.2f} "
             f"R {bestR['R']*100:.3f}   (checkpoint runs at 0.00, R "
             f"{rows[0]['R']*100:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
