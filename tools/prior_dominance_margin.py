#!/usr/bin/env python
"""How large a visual swing would be NEEDED to change the prior's decision?
CPU-only, no GPU, no model.

WHY
---
`openvocab_rel/evals.py` composes the model and the prior like this (verified
by reading the code at HEAD, for the exact config arm 1 ran with:
score_mode=ensemble, ensemble_alpha=0.0, freq_bias_alpha=3.75):

    text_logits = model.score(...)            raw cosine, range [-1, +1]
    visual      = _normalize_eval_logits(text_logits)    per-row z-score:
                                              mean 0, std 1 ACROSS predicates
    final       = visual + 3.75 * log P(p | s, o)
    pred        = argmax(final)

So the visual term is standardised to unit spread, while the prior term is
scaled by 3.75 and spans the full dynamic range of a log-probability. The
model can only change a decision when two predicates' prior scores are close
enough for a unit-scale perturbation to bridge them.

This tool measures that bridge width on real data. For every GT pair it
computes, in the SAME units the argmax sees:

  margin_12  = 3.75 * (logP[top1] - logP[top2])
               how much swing is needed to change the answer AT ALL

  margin_gt  = 3.75 * (logP[top1] - logP[gt])
               how much swing is needed to make the answer CORRECT
               (0 when the prior is already right)

The visual term is a difference of two z-scores, so its swing between any two
predicates has std sqrt(2) ~ 1.41. A margin of 4 is ~2.8 sigma; a margin of 10
is unreachable. The reported percentiles therefore bound, from data, how much
of the prior's error is even ADDRESSABLE by a term of this scale -- separating
"the encoder sees nothing" from "the composition cannot express it".

This measures the COMPOSITION, not any model. It runs without a checkpoint.

Usage:
    python tools/prior_dominance_margin.py --limit 3000 \
        --prior datasets_vg150_clean/frequency_prior_train.json \
        --out runs/<name>/prior_margin.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frequency_prior_baseline import FrequencyPrior, load_split  # noqa: E402

# Swing budgets, in z-score units, for a term with per-row std 1 across
# predicates. The difference of two such entries has std sqrt(2).
BUDGETS = (1.0, 2.0, 3.0, 4.0, 6.0, 10.0)


def pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--split", default="datasets_vg150_clean/validation.jsonl")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=3.75, help="freq_bias_alpha, as the evaluator applies it")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    prior = FrequencyPrior.load(Path(args.prior))
    data = load_split(Path(args.split), prior, limit=int(args.limit))
    alpha = float(args.alpha)

    m12: List[float] = []
    mgt_err: List[float] = []
    n = n_correct = 0
    addressable = {b: 0 for b in BUDGETS}
    flippable = {b: 0 for b in BUDGETS}

    for labels, triplets in data:
        for (s, o, p) in triplets:
            row, _ = prior.row(labels[s], labels[o], "pair")
            order = sorted(range(prior.n_pred), key=lambda i: -row[i])
            top1, top2 = order[0], order[1]
            gi = prior.pred_index[p]
            g12 = alpha * (row[top1] - row[top2])
            m12.append(g12)
            n += 1
            for b in BUDGETS:
                if g12 <= b:
                    flippable[b] += 1
            if top1 == gi:
                n_correct += 1
            else:
                ggt = alpha * (row[top1] - row[gi])
                mgt_err.append(ggt)
                for b in BUDGETS:
                    if ggt <= b:
                        addressable[b] += 1

    m12.sort(); mgt_err.sort()
    n_err = len(mgt_err)
    res: Dict[str, Any] = {
        "prior": str(args.prior), "split": str(args.split),
        "n_images": len(data), "n_pairs": n, "alpha": alpha,
        "prior_top1_correct": n_correct,
        "prior_top1_accuracy": n_correct / max(1, n),
        "n_prior_errors": n_err,
        "margin_top1_top2_percentiles": {q: pct(m12, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "margin_to_gt_on_errors_percentiles": {q: pct(mgt_err, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "fraction_flippable_at_budget": {b: flippable[b] / max(1, n) for b in BUDGETS},
        "fraction_of_errors_addressable_at_budget": {b: addressable[b] / max(1, n_err) for b in BUDGETS},
    }

    print(f"[margin] {len(data):,} images  {n:,} GT pairs  alpha={alpha}")
    print(f"[margin] prior top-1 accuracy {res['prior_top1_accuracy']*100:.2f}%  "
          f"({n_err:,} errors)")
    print(f"\nmargin_12 = {alpha} * (logP[top1] - logP[top2])   -- swing needed to change the answer at all")
    for q, v in res["margin_top1_top2_percentiles"].items():
        print(f"   p{int(q*100):<3} {v:8.2f}")
    print(f"\nmargin_gt = {alpha} * (logP[top1] - logP[gt])     -- swing needed to make it CORRECT, on errors only")
    for q, v in res["margin_to_gt_on_errors_percentiles"].items():
        print(f"   p{int(q*100):<3} {v:8.2f}")
    print(f"\n{'budget':>8}{'pairs flippable':>18}{'prior errors addressable':>28}")
    print(f"{'(z units)':>8}{'':>18}{'':>28}")
    for b in BUDGETS:
        print(f"{b:>8.1f}{res['fraction_flippable_at_budget'][b]*100:>17.2f}%"
              f"{res['fraction_of_errors_addressable_at_budget'][b]*100:>27.2f}%")
    print("\nreference: the visual term is a difference of two unit-variance z-scores,")
    print("           so its swing has std ~1.41; budget 3 is ~2.1 sigma, budget 6 is ~4.2 sigma.")

    if args.out:
        pth = Path(args.out); pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[margin] written to {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
