#!/usr/bin/env python
"""Do the field's metrics rank scoring functions the way grounding does?

runs/p47 found Spearman(R@50, WPRD) = +1.000 and Spearman(mR@50, WPRD) = -0.400
on FOUR arms. Four points cannot establish a correlation, and that was said in
the write-up. This widens the family to a dozen scoring functions -- every one
composed and evaluated identically -- so the rank correlation has some power.

These are NOT published SGG models, and the study that needs published models is
still the gate on any general claim. What this CAN establish is whether the
inversion is systematic across a diverse family of scoring functions built from
the same data, or an accident of which four arms happened to be in the table.

Every arm is scored the same way:
    composed:  R@50 / mR@50 / Pareto under 3.75*(prior - tau*logP) + term
    isolated:  WPRD on the identical cells

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
OBJ = _load("objective_ablation")
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def spearman(a: List[float], b: List[float]):
    n = len(a)

    def rank(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    rho = num / den if den else float("nan")
    # permutation p-value, exact enough at 2000 draws
    g = torch.Generator().manual_seed(0)
    cnt = 0
    for _ in range(2000):
        perm = torch.randperm(n, generator=g).tolist()
        rb2 = [rb[i] for i in perm]
        m2 = sum(rb2) / n
        nu = sum((x - ma) * (y - m2) for x, y in zip(ra, rb2))
        de = (sum((x - ma) ** 2 for x in ra)
              * sum((y - m2) ** 2 for y in rb2)) ** 0.5
        if de and abs(nu / de) >= abs(rho):
            cnt += 1
    return rho, (cnt + 1) / 2001


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p49_metric_grounding/corr.json")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args(argv)

    _log("=" * 112)
    _log("METRIC vs GROUNDING RANK CORRELATION -- CPU only, NO GPU")
    _log("=" * 112)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    N = CPA.Bench._norm

    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    Wg = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    o = torch.optim.LBFGS([Wg], max_iter=200, history_size=10,
                          line_search_fn="strong_wolfe")

    def cl():
        o.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ Wg, yt) + 1e-4 * (Wg * Wg).sum()
        l.backward()
        return l

    o.step(cl)
    geo_lin = N((Xv @ Wg).detach())
    _log("  fitting geometry MLP on train ...")
    netm = OBJ.mlp(Xt.shape[1], 256, B.n_classes, 0)
    opt = torch.optim.AdamW(netm.parameters(), lr=3e-3, weight_decay=1e-4)
    n, bs = Xt.shape[0], 8192
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(netm(Xt[ix]), yt[ix]).backward()
            opt.step()
    netm.eval()
    with torch.no_grad():
        geo_mlp = N(netm(Xv))

    txt, cls = B.fixed_ensemble(0.0), B.fixed_ensemble(1.0)
    arms: Dict[str, Optional[torch.Tensor]] = {
        "pair prior only": None,
        "random null": N(torch.randn(B.n_gt, B.n_classes,
                                     generator=torch.Generator().manual_seed(0))),
        "PURE text (a=0)": txt,
        "PURE ens a=0.25": B.fixed_ensemble(0.25),
        "PURE ens a=0.5": B.fixed_ensemble(0.5),
        "PURE ens a=0.75": B.fixed_ensemble(0.75),
        "PURE classifier (a=1)": cls,
        "geometry linear": geo_lin,
        "geometry MLP": geo_mlp,
        "text + geometry": txt + geo_lin,
        "classifier + geometry": cls + geo_lin,
        "classifier + geoMLP": cls + geo_mlp,
    }
    curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
             for t in CPA.TAUS]
    rows = []
    _log(f"\n  {'arm':>24} {'R@50':>8} {'mR@50':>8} {'pareto':>8} {'WPRD':>8}")
    _log(f"  {'-'*62}")
    for name, term in arms.items():
        m = B.metrics(B.score(args.tau, ALPHA_HIST, term))
        pg = CPA.pareto_gap(curve, m["R"], m["mR"])
        sc = B.prior if term is None else term
        w = WPD.wprd(Gs, sc, args.cap)["wprd_macro"]
        rows.append({"arm": name, "R": m["R"] * 100, "mR": m["mR"] * 100,
                     "pareto": pg, "wprd": w})
        _log(f"  {name:>24} {m['R']*100:>8.3f} {m['mR']*100:>8.3f} "
             f"{(pg if pg is not None else float('nan')):>+8.3f} {w:>8.4f}")

    wp = [r["wprd"] for r in rows]
    res: Dict[str, Any] = {"tool": "metric_grounding_correlation", "n_arms": len(rows),
                           "rows": rows, "spearman": {}}
    _log(f"\n  SPEARMAN vs WPRD  (n = {len(rows)} arms, permutation p, 2000 draws)")
    for key, lab in (("R", "R@50"), ("mR", "mR@50"), ("pareto", "Pareto gap")):
        v = [r[key] if r[key] is not None else float("nan") for r in rows]
        rho, p = spearman(v, wp)
        res["spearman"][lab] = {"rho": rho, "p_perm": p}
        _log(f"    {lab:>12}  rho = {rho:+.3f}   p = {p:.4f}"
             f"   {'significant' if p < 0.05 else 'not significant'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
