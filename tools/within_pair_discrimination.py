#!/usr/bin/env python
"""Within-Pair Relational Discrimination (WPRD) -- a PRIOR-FREE grounding metric.

Every SGG number this project has produced is contaminated by the frequency
prior, and the standard defences (tau sweeps, shuffled nulls, pair-matched
nulls) control it only on average. This measures relational grounding on a
population where the prior is *arithmetically incapable* of contributing.

THE CONSTRUCTION
----------------
Within one (subject, object) category group the train-derived prior row is
CONSTANT -- measured at max deviation 9.4e-05 over the p24 cache, i.e. exact to
float noise. So for two rows i, j of the same group and two predicates a, b:

    (prior[i,a] - prior[i,b]) - (prior[j,a] - prior[j,b]) == 0

The prior cancels EXACTLY. So does any per-class additive calibration, any tau,
any logit adjustment, and any global temperature: they are all functions of the
class alone and cancel in the same double difference. What survives is only
what varies with the IMAGE.

Define, for a group g and an ordered predicate pair (a, b) both present in g:

    s_i(a,b) = score[i,a] - score[i,b]

    WPRD(g,a,b) = AUC( {s_i : y_i = a}  vs  {s_j : y_j = b} )

WPRD = 0.5 exactly when the score carries no image-conditioned information about
which of a or b is the true relation. It is 1.0 for a perfect within-pair
discriminator. It has no free parameters, no operating point and no baseline to
argue about.

WHY THIS IS THE RIGHT POPULATION
--------------------------------
Of 132,556 GT rows in the p24 full-validation cache:
  23.3% are in singleton (s,o) groups          -- no within-pair contrast exists
  19.8% are in multi-row groups with CONSTANT GT -- contrast exists, answer is trivial
  56.9% (75,366 rows) are DECIDABLE            -- >= 2 distinct GT predicates

A pair-matched null permutes the model term inside a group, so on the first two
populations it is a NO-OP by construction: 43.1% of the rows in `runs/p26` and
`runs/p29` could not have shown a difference no matter what the model did.
Those runs' "+0.031 +- 0.188 Pareto points" is therefore an average over a
population that is 43% structurally inert. This tool measures only where the
question is answerable.

WHAT IS SCORED
--------------
The dump stores BOTH predicate heads, and the evaluated checkpoint runs at
ensemble_alpha = 0.0, which selects the text head and discards the classifier
head. Both are functions of the same image-derived `rel_feat`:

    text_logits = normalize(rel_feat) @ normalize(pred_emb).T
    cls_logits  = predicate_classifier(rel_feat)

so "the text branch" is NOT a lookup table -- it is a cosine between the trained
relational encoder's output on THIS image and the predicate embeddings. Both
heads are scored here, which also answers whether the discarded head would have
been better.

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(pos > neg) + 0.5 P(pos == neg). Exact, not sampled."""
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(allv), dtype=torch.float64)
    # average ranks over ties
    sortv = allv[order]
    i = 0
    while i < len(sortv):
        j = i
        while j + 1 < len(sortv) and sortv[j + 1] == sortv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    np_, nn = len(pos), len(neg)
    rsum = ranks[:np_].sum().item()
    return (rsum - np_ * (np_ - 1) / 2.0) / (np_ * nn)


class Groups:
    """(subject, object) grouping of the GT rows, plus the decidable subset."""

    def __init__(self, B: Mech):
        d = B.meta
        sl: List[str] = []
        ol: List[str] = []
        for i in range(B.n_images):
            sl.extend(d["subj_label"][i])
            ol.extend(d["obj_label"][i])
        self.subj = [sl[int(r)] for r in B.gt_row.tolist()]
        self.obj = [ol[int(r)] for r in B.gt_row.tolist()]
        keys = [f"{a}||{b}" for a, b in zip(self.subj, self.obj)]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.key_of = {v: k for k, v in uniq.items()}
        self.pair_id = torch.tensor([uniq[k] for k in keys])
        self.G = len(uniq)
        self.y = B.gt_y
        self.n = int(self.y.numel())
        self.image_of = B.gt_row.clone()

        self.rows_of: Dict[int, List[int]] = defaultdict(list)
        for r, p in enumerate(self.pair_id.tolist()):
            self.rows_of[p].append(r)
        self.classes_of = {p: set(self.y[torch.tensor(rs)].tolist())
                           for p, rs in self.rows_of.items()}
        nd = torch.tensor([len(self.classes_of[p]) for p in range(self.G)])
        gs = torch.tensor([len(self.rows_of[p]) for p in range(self.G)])
        self.ndist, self.gsize = nd[self.pair_id], gs[self.pair_id]
        self.decidable = self.ndist >= 2

    def prior_is_constant(self, prior: torch.Tensor) -> float:
        cnt = torch.zeros(self.G).index_add_(
            0, self.pair_id, torch.ones(self.n))
        s = torch.zeros(self.G, prior.shape[1]).index_add_(
            0, self.pair_id, prior)
        gm = (s / cnt.clamp_min(1).unsqueeze(1))[self.pair_id]
        multi = self.gsize > 1
        dev = (prior - gm).abs().max(dim=1).values
        return float(dev[multi].max()) if bool(multi.any()) else 0.0


def wprd(Gs: Groups, score: torch.Tensor, cap: int = 64,
         seed: int = 0) -> Dict[str, Any]:
    """WPRD over every (group, predicate-pair) with rows on both sides."""
    gen = torch.Generator().manual_seed(seed)
    vals: List[float] = []
    wts: List[int] = []
    per_pair: Dict[str, List[float]] = defaultdict(list)
    n_cells = 0
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        rows = torch.tensor(Gs.rows_of[p])
        yy = Gs.y[rows]
        byc = {c: rows[yy == c] for c in cls}
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                ra, rb = byc[a], byc[b]
                if len(ra) > cap:
                    ra = ra[torch.randperm(len(ra), generator=gen)[:cap]]
                if len(rb) > cap:
                    rb = rb[torch.randperm(len(rb), generator=gen)[:cap]]
                sa = score[ra, a] - score[ra, b]
                sb = score[rb, a] - score[rb, b]
                v = auc(sa.double(), sb.double())
                w = len(ra) * len(rb)
                vals.append(v)
                wts.append(w)
                per_pair[f"{a}|{b}"].append(v)
                n_cells += 1
    v = torch.tensor(vals, dtype=torch.float64)
    w = torch.tensor(wts, dtype=torch.float64)
    return {"wprd_macro": float(v.mean()),
            "wprd_weighted": float((v * w).sum() / w.sum()),
            "n_cells": n_cells,
            "n_comparisons": int(w.sum()),
            "frac_cells_above_half": float((v > 0.5).double().mean()),
            "_vals": vals, "_wts": wts, "_per_pair": per_pair}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p33_within_pair_discrimination/wprd.json")
    ap.add_argument("--cap", type=int, default=64,
                    help="max rows per (group, class) cell")
    ap.add_argument("--boot", type=int, default=200,
                    help="cluster-bootstrap resamples over IMAGES")
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("WITHIN-PAIR RELATIONAL DISCRIMINATION -- prior-free. CPU only, cache read-only, NO GPU")
    _log("=" * 100)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    Gs = Groups(B)
    _log(f"  images={B.n_images}  GT rows={Gs.n:,}  groups={Gs.G:,}  "
         f"[gate] model-term identity {e:.3e} OK")

    dev = Gs.prior_is_constant(B.prior)
    _log(f"\n  GATE W1 -- the prior must be CONSTANT within a group for this metric to")
    _log(f"  be prior-free. max |prior - group mean| over multi-row rows = {dev:.3e}"
         f"   {'PASS' if dev < 1e-3 else 'FAIL'}")
    assert dev < 1e-3, "prior is not constant within (s,o) groups; metric is not prior-free"

    dec = Gs.decidable
    _log(f"\n  POPULATION")
    _log(f"    decidable rows (group has >=2 distinct GT predicates) : "
         f"{int(dec.sum()):,} ({float(dec.float().mean())*100:.1f}%)")
    _log(f"    singleton-group rows (no contrast possible)           : "
         f"{int((Gs.gsize==1).sum()):,} ({float((Gs.gsize==1).float().mean())*100:.1f}%)")
    _log(f"    multi-row groups with CONSTANT GT (contrast trivial)  : "
         f"{int(((Gs.gsize>1)&(Gs.ndist==1)).sum()):,} "
         f"({float(((Gs.gsize>1)&(Gs.ndist==1)).float().mean())*100:.1f}%)")
    _log(f"    -> {float(((Gs.gsize==1)|(Gs.ndist==1)).float().mean())*100:.1f}% of rows are "
         f"structurally INERT to a pair-matched null (runs/p26, runs/p29)")

    text = B.fixed_ensemble(0.0)
    cls = B.fixed_ensemble(1.0)
    gen = torch.Generator().manual_seed(0)
    arms = {
        "text_head (the evaluated model term, alpha=0)": text,
        "classifier_head (stored but DISCARDED at alpha=0)": cls,
        "prior (must be exactly 0.5 -- it is constant in group)": B.prior,
        "random null": torch.randn(text.shape, generator=gen),
    }
    res: Dict[str, Any] = {
        "tool": "within_pair_discrimination", "dump": args.dump,
        "cap": args.cap, "gate_prior_constant_max_dev": dev,
        "n_rows": Gs.n, "n_groups": Gs.G,
        "n_decidable_rows": int(dec.sum()),
        "frac_decidable": float(dec.float().mean()),
        "frac_inert_to_pair_matched_null":
            float(((Gs.gsize == 1) | (Gs.ndist == 1)).float().mean()),
        "arms": {}}

    _log(f"\n{'-'*100}")
    _log(f"  WPRD -- 0.5 = no image-conditioned relational information. cap={args.cap} rows/cell")
    _log(f"{'-'*100}")
    _log(f"  {'arm':>54} {'macro':>8} {'weighted':>9} {'cells>.5':>9} {'cells':>8}")
    store = {}
    for name, s in arms.items():
        r = wprd(Gs, s, args.cap)
        store[name] = r
        res["arms"][name] = {k: v for k, v in r.items() if not k.startswith("_")}
        _log(f"  {name:>54} {r['wprd_macro']:>8.4f} {r['wprd_weighted']:>9.4f} "
             f"{r['frac_cells_above_half']*100:>8.1f}% {r['n_cells']:>8,}")

    # cluster bootstrap over the (group, class-pair) cells
    _log(f"\n  bootstrap over cells ({args.boot} resamples)")
    for name in ("text_head (the evaluated model term, alpha=0)",
                 "classifier_head (stored but DISCARDED at alpha=0)"):
        v = torch.tensor(store[name]["_vals"], dtype=torch.float64)
        g2 = torch.Generator().manual_seed(1)
        bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g2)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        res["arms"][name]["boot_ci95"] = [lo, hi]
        _log(f"    {name:>54}  95% CI [{lo:.4f}, {hi:.4f}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
