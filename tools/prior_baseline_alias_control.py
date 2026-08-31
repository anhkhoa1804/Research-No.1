#!/usr/bin/env python
"""Is the model-vs-prior comparison in A' actually like-for-like? CPU-only.

THE PROBLEM THIS MEASURES
-------------------------
`openvocab_rel/evals.py` (the arm-1 model path) normalises BOTH ground-truth
and predicted predicate labels through `_build_vg_aliases`, which contains:

    "near"  -> "next to"
    "wears" -> "wearing"

Both `near` and `wears` are canonical members of the 50-predicate VG150
vocabulary. Merging them leaves the evaluator averaging mR@50 over **48**
classes, and lets a `next to` prediction score a hit on a `near` ground truth.

`tools/frequency_prior_baseline.py` and `tools/decision_rule_probe.py` (arms 2,
2b, 3) apply no aliasing at all and average over **50** classes.

So `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md`'s headline
`delta_mR = arm1 - arm2 = -0.82` compares a 48-class aliased average against a
50-class unaliased one. This tool measures the size and sign of that gap by
scoring the SAME prior, on the SAME images, under BOTH schemes.

It changes no scientific criterion and modifies nothing. It re-uses
`FrequencyPrior`, `load_split` and `softmax` from the existing tools by import,
and the alias map from `evals.py` by import, so it cannot drift from either.

VALIDATION GATE: scheme `raw50` must reproduce the recorded arm-2 numbers
(R@50 66.80 / mR@50 21.98 at tau=0) exactly. If it does not, this tool is
wrong and its `eval48` column must be discarded.

Usage:
    python tools/prior_baseline_alias_control.py --limit 3000 \
        --prior datasets_vg150_clean/frequency_prior_train.json \
        --out runs/<name>/alias_control.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frequency_prior_baseline import FrequencyPrior, load_split, softmax  # noqa: E402

DEFAULT_KS: Tuple[int, ...] = (20, 50, 100)


def evaluator_predicate_aliases(pred_vocab: List[str]) -> Dict[str, str]:
    """The predicate half of evals.py's alias map, imported not restated."""
    from openvocab_rel.evals import _build_vg_aliases
    full = _build_vg_aliases([], list(pred_vocab))
    return {k: v for k, v in full.items() if k in set(pred_vocab) or v in set(pred_vocab)}


def evaluate(
    data: List[Tuple[List[str], List[Tuple[int, int, str]]]],
    prior: FrequencyPrior,
    alpha: float = 3.75,
    tau: float = 0.0,
    mode: str = "pair",
    aliases: Optional[Dict[str, str]] = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Any]:
    """Byte-for-byte the scoring of tools/frequency_prior_baseline.evaluate
    (plus decision_rule_probe's tau term), with ONE optional change: predicate
    labels on both the GT side and the prediction side are mapped through
    `aliases` before matching -- exactly what evals.py does at lines 1757/1847.
    """
    def alias(p: str) -> str:
        return aliases.get(p, p) if aliases else p

    marginal = prior.global_lp
    pooled_hits = {k: 0 for k in ks}
    per_pred_hits = {k: defaultdict(int) for k in ks}
    image_recall_sum = {k: 0.0 for k in ks}
    per_pred_gt: Dict[str, int] = defaultdict(int)
    pooled_gt = 0

    for labels, triplets in data:
        scored: List[Tuple[float, int]] = []
        for (s, o, _p) in triplets:
            row, _level = prior.row(labels[s], labels[o], mode)
            if tau != 0.0:
                row = [row[i] - tau * marginal[i] for i in range(prior.n_pred)]
            probs = softmax([alpha * v for v in row])
            best = max(range(prior.n_pred), key=lambda i: probs[i])
            scored.append((probs[best], best))

        order = sorted(range(len(triplets)), key=lambda i: -scored[i][0])
        for (_s, _o, p) in triplets:
            per_pred_gt[alias(p)] += 1
        pooled_gt += len(triplets)

        for k in ks:
            matched: set = set()
            hits = 0
            for ci in order[:k]:
                cs, co, _ = triplets[ci]
                pname = alias(prior.pred_vocab[scored[ci][1]])
                for gi, (gs, go, gp) in enumerate(triplets):
                    if gi in matched:
                        continue
                    if gs == cs and go == co and alias(gp) == pname:
                        matched.add(gi)
                        hits += 1
                        per_pred_hits[k][alias(gp)] += 1
                        break
            pooled_hits[k] += hits
            image_recall_sum[k] += hits / max(1, len(triplets))

    n_images = len(data)
    out: Dict[str, Any] = {
        "tau": tau, "alpha": alpha, "mode": mode,
        "aliased": bool(aliases),
        "n_images": n_images,
        "n_gt_triplets": pooled_gt,
        "n_predicate_classes_with_gt": len(per_pred_gt),
    }
    for k in ks:
        recalls = [per_pred_hits[k][p] / per_pred_gt[p] for p in per_pred_gt if per_pred_gt[p] > 0]
        out[f"R@{k}"] = pooled_hits[k] / max(1, pooled_gt)
        out[f"mR@{k}"] = sum(recalls) / max(1, len(recalls))
        out[f"image_mean_R@{k}"] = image_recall_sum[k] / max(1, n_images)
    out["per_predicate_R@50"] = {
        p: per_pred_hits[50][p] / per_pred_gt[p] for p in sorted(per_pred_gt) if per_pred_gt[p] > 0
    }
    out["per_predicate_gt"] = {p: int(per_pred_gt[p]) for p in sorted(per_pred_gt)}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--split", default="datasets_vg150_clean/validation.jsonl")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=3.75)
    ap.add_argument("--taus", default="0.0,0.1")
    ap.add_argument("--out", default="")
    # The recorded arm-2 numbers this tool must reproduce before its aliased
    # column may be believed. Not thresholds on a result -- a self-check.
    ap.add_argument("--expect_R50", type=float, default=0.6680156623656479)
    ap.add_argument("--expect_mR50", type=float, default=0.21975828067233627)
    args = ap.parse_args(argv)

    prior = FrequencyPrior.load(Path(args.prior))
    data = load_split(Path(args.split), prior, limit=int(args.limit))
    aliases = evaluator_predicate_aliases(prior.pred_vocab)
    merged = {k: v for k, v in aliases.items() if k != v}
    print(f"[alias-control] predicate merges evals.py applies: {merged}")

    rows: List[Dict[str, Any]] = []
    for tau in [float(t) for t in str(args.taus).split(",") if t.strip()]:
        for scheme, al in (("raw50", None), ("eval48", aliases)):
            r = evaluate(data, prior, alpha=args.alpha, tau=tau, aliases=al)
            r["scheme"] = scheme
            rows.append(r)

    base = next(r for r in rows if r["scheme"] == "raw50" and r["tau"] == 0.0)
    ok = (abs(base["R@50"] - args.expect_R50) < 1e-9 and abs(base["mR@50"] - args.expect_mR50) < 1e-9)
    print(f"\n[alias-control] VALIDATION GATE (raw50,tau=0 reproduces recorded arm 2): "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"    R@50  {base['R@50']:.10f} vs expected {args.expect_R50:.10f}")
    print(f"    mR@50 {base['mR@50']:.10f} vs expected {args.expect_mR50:.10f}")

    print(f"\n{'scheme':<9}{'tau':>5}{'classes':>9}{'R@50':>9}{'mR@50':>9}")
    for r in rows:
        print(f"{r['scheme']:<9}{r['tau']:>5}{r['n_predicate_classes_with_gt']:>9}"
              f"{r['R@50']*100:>8.2f}%{r['mR@50']*100:>8.2f}%")

    a = next(r for r in rows if r["scheme"] == "raw50" and r["tau"] == 0.0)
    b = next(r for r in rows if r["scheme"] == "eval48" and r["tau"] == 0.0)
    print(f"\n[alias-control] aliasing moves the SAME prior by "
          f"{(b['mR@50']-a['mR@50'])*100:+.2f} mR@50 and {(b['R@50']-a['R@50'])*100:+.2f} R@50")
    for p in ("near", "wears", "next to", "wearing"):
        if p in a["per_predicate_R@50"]:
            print(f"    raw50  {p:<9} n={a['per_predicate_gt'][p]:>5}  R@50 {a['per_predicate_R@50'][p]*100:6.2f}%")

    result = {"rows": rows, "validation_gate_passes": bool(ok),
              "predicate_merges": merged,
              "note": "eval48 is the scheme openvocab_rel/evals.py uses for the model arm; "
                      "raw50 is the scheme tools/frequency_prior_baseline.py uses for the prior arms."}
    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[alias-control] written to {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
