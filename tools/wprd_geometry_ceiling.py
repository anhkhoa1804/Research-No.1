#!/usr/bin/env python
"""How much relational discrimination do the BOXES ALONE permit?

Every WPRD number so far is uncalibrated: 0.5542 for the evaluated head is
clearly above the 0.5 floor, but is it far below what VG150 permits, or close
to it? Without a ceiling the number cannot be called low.

A ceiling is measurable for one input channel exactly, and it is the important
one. WPRD conditions on the (subject, object) group, and WITHIN a group the
subject label, the object label and the prior are all CONSTANT. So the only
inputs that vary within a group are the PIXELS and the BOXES. The box channel
can therefore be pushed to its ceiling on CPU, right now, with no model:

    linear probe        already measured, 0.5961 train-fitted
    MLP probe           this run -- how much does non-linearity add?

Whatever the box channel reaches is a LOWER bound on the achievable WPRD for
any model that can see boxes, which every SGG model can. If the MLP reaches
0.65, the evaluated head at 0.5542 is far below an available ceiling. If it
stalls near 0.60, the linear number was already close to the box ceiling and
the remaining headroom must come from pixels -- which is what runs/p36 measures.

Fitted on the TRAIN split, applied to validation. No validation statistic
enters the fit.

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
Mech = MECH.Mech


def _log(m: str = "") -> None:
    print(m, flush=True)


def fit_mlp(X, y, C, hidden, epochs, lr, l2, seed, log_every=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, C))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    n = X.shape[0]
    bs = 8192
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(X[ix]), y[ix])
            loss.backward()
            opt.step()
            tot += float(loss) * len(ix)
        if log_every and (ep + 1) % log_every == 0:
            _log(f"      epoch {ep+1:>3}/{epochs}  train CE {tot/n:.4f}")
    net.eval()
    return net


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p46_geometry_ceiling/ceiling.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("BOX-CHANNEL WPRD CEILING -- CPU only, cache read-only, NO GPU")
    _log("=" * 100)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    dev = Gs.prior_is_constant(B.prior)
    _log(f"  rows {Gs.n:,} groups {Gs.G:,}   [gate W1] prior dev {dev:.3e} "
         f"{'PASS' if dev < 1e-3 else 'FAIL'}")
    assert dev < 1e-3
    _log("  within a group the subject label, object label and prior are CONSTANT,")
    _log("  so the only inputs that can vary are the PIXELS and the BOXES.")

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    _log(f"\n  train rows {Xt.shape[0]:,}  val rows {Xv.shape[0]:,}  "
         f"features {Xt.shape[1]-1}+bias")

    arms: Dict[str, torch.Tensor] = {
        "text_head (evaluated)": B.fixed_ensemble(0.0),
        "classifier_head (discarded)": B.fixed_ensemble(1.0),
    }

    # linear reference, same recipe as p39
    W = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=200, history_size=10,
                            line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ W, yt) + 1e-4 * (W * W).sum()
        l.backward()
        return l

    opt.step(cl)
    arms["geometry LINEAR (train-fitted)"] = (Xv @ W).detach()

    _log(f"\n  fitting MLP {Xt.shape[1]}-{args.hidden}-{args.hidden}-{B.n_classes} "
         f"on the TRAIN split")
    net = fit_mlp(Xt, yt, B.n_classes, args.hidden, args.epochs, args.lr,
                  args.l2, 0, log_every=10)
    with torch.no_grad():
        arms["geometry MLP (train-fitted)"] = net(Xv)

    # a null: MLP on shuffled train labels
    _log(f"  fitting MLP on SHUFFLED train labels (control)")
    ysh = yt[torch.randperm(len(yt), generator=torch.Generator().manual_seed(7))]
    net0 = fit_mlp(Xt, ysh, B.n_classes, args.hidden, args.epochs, args.lr,
                   args.l2, 0)
    with torch.no_grad():
        arms["geometry MLP, SHUFFLED labels"] = net0(Xv)
    arms["prior (must be 0.5)"] = B.prior

    res: Dict[str, Any] = {"tool": "wprd_geometry_ceiling", "arms": {}}
    _log(f"\n{'-'*100}")
    _log(f"  {'arm':>36} {'macro':>8} {'weighted':>9} {'95% CI':>22}")
    _log(f"{'-'*100}")
    store = {}
    for name, s in arms.items():
        r = WPD.wprd(Gs, s, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        store[name] = v
        res["arms"][name] = {"macro": r["wprd_macro"], "weighted": r["wprd_weighted"],
                             "ci95": [lo, hi]}
        _log(f"  {name:>36} {r['wprd_macro']:>8.4f} {r['wprd_weighted']:>9.4f}  "
             f"[{lo:.4f}, {hi:.4f}]")

    _log(f"\n  PAIRED contrasts")
    for a, b in (("geometry MLP (train-fitted)", "geometry LINEAR (train-fitted)"),
                 ("geometry MLP (train-fitted)", "classifier_head (discarded)"),
                 ("geometry MLP (train-fitted)", "text_head (evaluated)")):
        d = store[a] - store[b]
        g = torch.Generator().manual_seed(2)
        bs = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        res.setdefault("contrasts", {})[f"{a} - {b}"] = [float(d.mean()), lo, hi]
        _log(f"    {a[:32]:>32} - {b[:32]:<32} {float(d.mean()):>+8.4f} "
             f"[{lo:+.4f}, {hi:+.4f}]  "
             f"{'excludes 0' if (lo > 0 or hi < 0) else 'INCLUDES 0'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
