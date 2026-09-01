#!/usr/bin/env python
"""Is the text head's failure READOUT GEOMETRY rather than supervision?

runs/p50 found a dissociation: the evaluated TEXT head's within-pair
discrimination tracks how much within-pair supervision each predicate-bucket
received (Spearman +0.657), while the DISCARDED CLASSIFIER head does not
(-0.086) and reaches 0.6167 on body-body, a bucket holding 0.53% of all
supervision. Scarcity therefore cannot be the whole story.

The heads differ structurally in exactly one way:

    text_logits = normalize(rel_feat) . normalize(pred_emb)^T   FIXED geometry
    cls_logits  = predicate_classifier(rel_feat)                LEARNED map

The text head's ability to separate predicates a and b is bounded by the angle
between pred_emb[a] and pred_emb[b]. Those embeddings are CLIP text features and
are never trained. If two predicates are near-collinear there, the cosine readout
cannot separate them no matter what rel_feat contains.

pred_emb itself lives only in the p36 cache. But the hypothesis is testable NOW
from p24, because for a fixed embedding set

    cov( logits[:,a], logits[:,b] ) = n(pe_a)^T . Cov(rel_feat) . n(pe_b)

so the COLUMN CORRELATION MATRIX of the stored logits is a proxy for the
embedding Gram matrix, and the variance of the contrast column

    s(a,b) = logits[:,a] - logits[:,b]

is the discriminative BUDGET the readout has for that predicate pair. A pair
with near-zero budget cannot be discriminated by that readout at any WPRD.

Prediction if the hypothesis holds:
  * text head: high column correlation / low contrast variance -> low WPRD
  * classifier head: weaker or absent association

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
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


MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
STRAT = _load("wprd_stratified")
Mech = MECH.Mech


def _log(m: str = "") -> None:
    print(m, flush=True)


def spearman(a, b):
    n = len(a)

    def rk(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = rk(a), rk(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    rho = num / den if den else float("nan")
    g = torch.Generator().manual_seed(0)
    cnt = 0
    for _ in range(2000):
        p = torch.randperm(n, generator=g).tolist()
        rb2 = [rb[i] for i in p]
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
    ap.add_argument("--out", default="runs/p51_readout_geometry/geom.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--min-cells", type=int, default=20)
    args = ap.parse_args(argv)

    _log("=" * 104)
    _log("READOUT GEOMETRY -- is the text head bounded by its embedding angles? NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    heads = {"text_head": B.fixed_ensemble(0.0),
             "classifier_head": B.fixed_ensemble(1.0)}
    res: Dict[str, Any] = {"tool": "readout_geometry", "heads": {}}

    for hname, M in heads.items():
        # column correlation matrix + contrast budget
        Mc = M - M.mean(0, keepdim=True)
        sd = Mc.std(0).clamp_min(1e-8)
        Corr = (Mc.T @ Mc) / (Mc.shape[0] * sd.unsqueeze(1) * sd.unsqueeze(0))

        # per-predicate-pair WPRD, on the same cells the other tools use
        cells = STRAT.cells(Gs, B, M, args.cap, 0, drop_same_instance=False)
        per = defaultdict(list)
        for c in cells:
            per[(c["a"], c["b"])].append(c["auc"])
        rows = []
        for (a, b), v in per.items():
            if len(v) < args.min_cells:
                continue
            budget = float((M[:, a] - M[:, b]).var())
            rows.append({"a": B.classes[a], "b": B.classes[b],
                         "n_cells": len(v),
                         "wprd": sum(v) / len(v),
                         "col_corr": float(Corr[a, b]),
                         "contrast_var": budget,
                         "log_contrast_var": math.log10(max(budget, 1e-12)),
                         "bucket": "-".join(sorted([B.bucket_of[a], B.bucket_of[b]]))})
        w = [r["wprd"] for r in rows]
        cc = [r["col_corr"] for r in rows]
        bv = [r["log_contrast_var"] for r in rows]
        r1, p1 = spearman(cc, w)
        r2, p2 = spearman(bv, w)
        res["heads"][hname] = {"n_predicate_pairs": len(rows),
                               "spearman_colcorr_vs_wprd": [r1, p1],
                               "spearman_logcontrastvar_vs_wprd": [r2, p2],
                               "rows": rows}
        _log(f"\n{'-'*104}\n  {hname}   ({len(rows)} predicate pairs with "
             f">= {args.min_cells} cells)\n{'-'*104}")
        _log(f"    Spearman(column correlation, WPRD)   = {r1:+.3f}  p = {p1:.4f}"
             f"   {'significant' if p1 < 0.05 else 'ns'}")
        _log(f"    Spearman(log contrast variance, WPRD)= {r2:+.3f}  p = {p2:.4f}"
             f"   {'significant' if p2 < 0.05 else 'ns'}")
        rows.sort(key=lambda r: r["contrast_var"])
        _log(f"\n    LOWEST discriminative budget (readout can barely separate these)")
        _log(f"      {'a':>16} {'b':>16} {'corr':>7} {'var':>10} {'WPRD':>7} {'bucket':>11}")
        for r in rows[:8]:
            _log(f"      {r['a']:>16} {r['b']:>16} {r['col_corr']:>+7.3f} "
                 f"{r['contrast_var']:>10.4f} {r['wprd']:>7.4f} {r['bucket']:>11}")
        _log(f"\n    HIGHEST discriminative budget")
        for r in rows[-5:]:
            _log(f"      {r['a']:>16} {r['b']:>16} {r['col_corr']:>+7.3f} "
                 f"{r['contrast_var']:>10.4f} {r['wprd']:>7.4f} {r['bucket']:>11}")

    # do the two heads differ in budget on the SAME pairs?
    t = {(r["a"], r["b"]): r for r in res["heads"]["text_head"]["rows"]}
    c = {(r["a"], r["b"]): r for r in res["heads"]["classifier_head"]["rows"]}
    common = sorted(set(t) & set(c))
    if common:
        dt = [t[k]["log_contrast_var"] for k in common]
        dc = [c[k]["log_contrast_var"] for k in common]
        diff = [y - x for x, y in zip(dt, dc)]
        res["budget_shift_classifier_minus_text_log10"] = sum(diff) / len(diff)
        _log(f"\n{'-'*104}")
        _log(f"  On the {len(common)} shared predicate pairs, the CLASSIFIER head's "
             f"log10 contrast variance is")
        _log(f"  {sum(diff)/len(diff):+.3f} vs the text head "
             f"({'more' if sum(diff) > 0 else 'less'} discriminative budget)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
