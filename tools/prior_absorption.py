#!/usr/bin/env python
"""Is the model MORE pair-determined than reality is?

A single principle would explain every result in this cycle at once:

    Training on VG150 with a recall objective under extreme predicate imbalance
    makes reproducing P(p | s,o) the dominant gradient signal. The visual
    pathway is therefore optimised to PREDICT the prior rather than to DEVIATE
    from it -- and deviation is penalised on average, because the prior is right
    two thirds of the time, even though deviation is exactly what is required on
    the third where it is wrong.

Call it PRIOR ABSORPTION. It makes a sharp, falsifiable, cheap prediction:

    the model's OUTPUT should be MORE determined by (subject, object) than the
    GROUND TRUTH is.

If the model merely inherited the prior it would be *as* pair-determined as the
data. If it over-absorbed it, it is *more* so -- it has thrown away
image-conditioned variation the data actually contains.

Measured two ways, on the same (s,o) groups:

  1. within-group entropy of the label distribution -- GT vs each arm's argmax
  2. accuracy of the best pair-CONSTANT predictor at predicting GT, vs at
     predicting each arm's own output. A predictor that is a pure function of
     (s,o) reproducing an arm's output well means that arm is nearly a pure
     function of (s,o).

Both are computed on the decidable rows, where the question is meaningful.

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
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
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def pair_stats(labels: torch.Tensor, gid: torch.Tensor, sel: torch.Tensor):
    """Within-group entropy (nats) and best pair-constant accuracy for `labels`."""
    by = defaultdict(list)
    for g, v in zip(gid[sel].tolist(), labels[sel].tolist()):
        by[g].append(v)
    ent, wts, hit, tot = [], [], 0, 0
    for g, vs in by.items():
        c = Counter(vs)
        n = len(vs)
        e = -sum((k / n) * math.log(k / n) for k in c.values())
        ent.append(e)
        wts.append(n)
        hit += c.most_common(1)[0][1]
        tot += n
    w = torch.tensor(wts, dtype=torch.float64)
    e = torch.tensor(ent, dtype=torch.float64)
    return {"entropy_macro": float(e.mean()),
            "entropy_weighted": float((e * w).sum() / w.sum()),
            "pair_constant_acc": hit / max(1, tot),
            "n_groups": len(by), "n_rows": tot}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p44_prior_absorption/absorb.json")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("PRIOR ABSORPTION -- is the model more pair-determined than reality? "
         "CPU only, NO GPU")
    _log("=" * 100)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    dec = Gs.decidable
    _log(f"  rows {Gs.n:,}  groups {Gs.G:,}  decidable rows {int(dec.sum()):,}")

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
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

    arms = {
        "GROUND TRUTH (what reality does)": B.gt_y,
        "prior only": B.predict(B.score(args.tau, ALPHA_HIST, None)),
        "prior + MODEL (the checkpoint)": B.predict(B.score(args.tau, ALPHA_HIST, B.model)),
        "prior + GEOMETRY (w=1)": B.predict(B.score(args.tau, ALPHA_HIST, geo)),
        "prior + MODEL + GEOMETRY (w=1)": B.predict(B.score(args.tau, ALPHA_HIST, B.model + geo)),
        "MODEL term alone (argmax)": B.col_to_class[B.model.argmax(-1)],
    }
    res: Dict[str, Any] = {"tool": "prior_absorption", "tau": args.tau, "arms": {}}
    _log(f"\n{'-'*100}")
    _log(f"  on DECIDABLE rows (group has >=2 distinct GT predicates)")
    _log(f"  {'arm':>36} {'within-grp entropy':>19} {'pair-constant acc':>19}")
    _log(f"  {'':>36} {'macro':>9} {'wtd':>9} {'':>19}")
    _log(f"{'-'*100}")
    gt_ent = None
    for name, lab in arms.items():
        s = pair_stats(lab, Gs.pair_id, dec)
        res["arms"][name] = s
        if gt_ent is None:
            gt_ent = s["entropy_macro"]
        _log(f"  {name:>36} {s['entropy_macro']:>9.4f} {s['entropy_weighted']:>9.4f} "
             f"{s['pair_constant_acc']*100:>18.2f}%")

    _log(f"\n  READING")
    g = res["arms"]["GROUND TRUTH (what reality does)"]
    for name in list(arms)[1:]:
        a = res["arms"][name]
        de = a["entropy_macro"] - g["entropy_macro"]
        da = a["pair_constant_acc"] - g["pair_constant_acc"]
        verdict = ("MORE pair-determined than reality" if da > 0.01
                   else "less pair-determined than reality" if da < -0.01
                   else "about as pair-determined as reality")
        res["arms"][name]["vs_gt_entropy"] = de
        res["arms"][name]["vs_gt_pair_constant_acc"] = da
        _log(f"    {name:>36}  entropy {de:>+7.4f}  pair-const acc {da*100:>+6.2f} pts"
             f"   -> {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
