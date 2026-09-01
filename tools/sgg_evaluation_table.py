#!/usr/bin/env python
"""The Paper-1 evaluation table: what each metric actually measures.

The diagnostic claim is that the standard SGG protocol conflates four things:

  1. the object-pair prior P(p | s,o)
  2. calibration (the tau / class-reweighting axis)
  3. candidate selection
  4. relational discrimination

R@50 and mR@50 mix all four. WPRD isolates (4) by conditioning on the (s,o)
group, which makes (1) exactly non-informative and cancels (2) in the same
double difference. This tool emits every column side by side so the conflation
is visible in one table rather than argued for in prose.

MODEL-AGNOSTIC BY DESIGN. `--extra` takes `name=path.pt`, where the file is a
dict with either

    {"model_term": Tensor(n_gt_rows, n_classes)}          # already GT-row aligned
    {"per_image_logits": [Tensor(n_pairs_i, n_classes)]}  # cache-order per image

in the SAME row order and predicate vocabulary as the reference dump. Any SGG
model that can emit per-pair predicate logits on VG150 validation can therefore
be added without touching this file. That is the interface a cross-model study
needs, and it is the reason the study is not blocked on any one codebase.

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
STRAT = _load("wprd_stratified")
GEO = _load("wprd_geometry_control")
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def load_extra(path: str, B: Mech) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "model_term" in d:
        t = d["model_term"].float()
    elif "per_image_logits" in d:
        t = torch.cat([x.float() for x in d["per_image_logits"]], 0)[B.gt_row]
    else:
        raise KeyError(f"{path}: need 'model_term' or 'per_image_logits'")
    assert t.shape == (B.n_gt, B.n_classes), \
        f"{path}: got {tuple(t.shape)}, expected {(B.n_gt, B.n_classes)}"
    return t


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p47_evaluation_table/table.json")
    ap.add_argument("--extra", action="append", default=[],
                    help="name=path.pt, repeatable")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 118)
    _log("SGG EVALUATION TABLE -- what each metric measures. CPU only, NO GPU")
    _log("=" * 118)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    dev = Gs.prior_is_constant(B.prior)
    assert dev < 1e-3
    _log(f"  images {B.n_images}  GT rows {B.n_gt:,}  groups {Gs.G:,}  "
         f"decidable {int(Gs.decidable.sum()):,} ({float(Gs.decidable.float().mean())*100:.1f}%)")
    _log(f"  [gate W1] prior within-group max dev {dev:.3e} PASS  "
         f"-> WPRD is prior-free on this data")

    # geometry references
    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    W = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=200, history_size=10,
                            line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ W, yt) + 1e-4 * (W * W).sum()
        l.backward()
        return l

    opt.step(cl)
    geo_lin = CPA.Bench._norm((Xv @ W).detach())

    terms: Dict[str, Optional[torch.Tensor]] = {
        "PURE text head (evaluated)": B.fixed_ensemble(0.0),
        "PURE classifier head (discarded)": B.fixed_ensemble(1.0),
        "geometry linear (train-fitted)": geo_lin,
        "pair prior only": None,
        "random null": torch.randn(B.n_gt, B.n_classes,
                                   generator=torch.Generator().manual_seed(0)),
    }
    for e in args.extra:
        name, _, path = e.partition("=")
        terms[name] = CPA.Bench._norm(load_extra(path, B))
        _log(f"  loaded extra model '{name}' from {path}")

    curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
             for t in CPA.TAUS]
    base = B.metrics(B.score(args.tau, ALPHA_HIST, None))
    rows: List[Dict[str, Any]] = []
    _log(f"\n{'-'*118}")
    _log(f"  tau={args.tau}   composed score = {ALPHA_HIST}*(prior - tau*logP) + term")
    _log(f"{'-'*118}")
    _log(f"  {'model':>34} {'R@50':>7} {'mR@50':>7} {'pareto':>7} | "
         f"{'WPRD':>7} {'CI':>16} | {'head':>6} {'body':>6} {'tail':>6}")
    _log(f"  {'':>34} {'---- conflated ----':>23} | "
         f"{'--- isolated #4 ---':>24} | {'-- WPRD by bucket --':>20}")
    for name, term in terms.items():
        m = B.metrics(B.score(args.tau, ALPHA_HIST, term))
        pg = CPA.pareto_gap(curve, m["R"], m["mR"])
        sc = B.prior if term is None else term
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
        rows.append({"model": name, "R": m["R"], "mR": m["mR"], "pareto": pg,
                     "wprd": r["wprd_macro"], "wprd_ci": [lo, hi],
                     "wprd_by_bucket": bk})
        _log(f"  {name:>34} {m['R']*100:>7.3f} {m['mR']*100:>7.3f} "
             f"{(pg if pg is not None else float('nan')):>+7.3f} | "
             f"{r['wprd_macro']:>7.4f} [{lo:.4f},{hi:.4f}] | "
             f"{bk['head-head']:>6.4f} {bk['body-body']:>6.4f} {bk['tail-tail']:>6.4f}")

    _log(f"\n  READING THE TABLE")
    _log(f"    R@50/mR@50/pareto mix the pair prior, calibration, candidate")
    _log(f"    selection and relational discrimination. WPRD isolates the last:")
    _log(f"    the pair-prior row MUST read 0.5000 and the random row ~0.5.")
    res = {"tool": "sgg_evaluation_table", "tau": args.tau,
           "prior_baseline": base, "rows": rows,
           "interface": "extra models: {'model_term': (n_gt, n_classes)} or "
                        "{'per_image_logits': [ (n_pairs_i, n_classes) ]}"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
