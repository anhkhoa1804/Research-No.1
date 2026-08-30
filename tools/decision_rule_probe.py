#!/usr/bin/env python
"""Is the bottleneck the decision rule rather than the representation?

Hypothesis and pre-registered criteria: docs/DECISION_RULE_HYPOTHESIS.md.

THE CONTRADICTION
-----------------
tools/predicate_discriminability.py measures AUC 0.83-0.88 for the predicate
confusions that matter, using 8 box-geometry features. tools/
candidate_reranking_analysis.py reports that 200 dedicated pairwise probes on
the same features capture 0.0 % of the oracle headroom. Both cannot be simply
true.

The AUC is measured BALANCED. Ranking happens under the NATURAL distribution,
where the prior puts 45 % of its argmax mass on `on`. Flipping `on` -> `above`
needs ~2.6 nats of evidence; an AUC-0.865 linear probe rarely supplies that.
A scorer trained by cross-entropy on that distribution minimises its loss by
AGREEING with the prior.

WHAT THIS TESTS
---------------
Whether an explicit class-balancing adjustment to the DECISION converts
headroom that argmax-of-score discards. No training, no vision, no GPU:

    score(p | s,o) = log P(p | s,o)  -  tau * log P(p)

tau = 0 is exactly tools/frequency_prior_baseline.py, so any movement is
attributable to the adjustment alone. log P(p) is the TRAIN marginal
(the prior's own global_log_probs), never the evaluation split's.

This is also a necessary CONTROL. If recalibrating the prior -- with no visual
input at all -- reproduces the size of gain attributed to a model, that gain is
not evidence of visual understanding.

Both R@50 and mR@50 are reported at every tau. Raising a class-averaged metric
by sacrificing head recall is a well-known and often vacuous manoeuvre; the
PARETO-ONLY verdict exists to name it when that is all that happened.

Usage:
    python tools/decision_rule_probe.py
    python tools/decision_rule_probe.py --out runs/<run>/decision_rule.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.frequency_prior_baseline import (  # noqa: E402
    DEFAULT_KS,
    FrequencyPrior,
    load_split,
    softmax,
)

DEFAULT_PRIOR = Path("datasets_vg150_clean/frequency_prior_train.json")
DEFAULT_SPLIT = Path("datasets_vg150_clean/validation.jsonl")

# Fixed in advance in docs/DECISION_RULE_HYPOTHESIS.md. Not tuned after seeing
# any result.
DEFAULT_TAUS: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

# Pre-registered decision thresholds (points of R@50 / mR@50).
MIN_MR_GAIN_SUPPORT = 3.0
MAX_R_LOSS_SUPPORT = 3.0
MIN_MR_GAIN_ANY = 1.0


def head_body_tail(prior: FrequencyPrior, gt_counts: Dict[str, int]) -> Dict[str, str]:
    """Bucket predicates by GT frequency: 15 head, next 20 body, rest tail."""
    ordered = sorted(gt_counts, key=lambda p: -gt_counts[p])
    bucket: Dict[str, str] = {}
    for i, p in enumerate(ordered):
        bucket[p] = "head" if i < 15 else ("body" if i < 35 else "tail")
    return bucket


def evaluate_tau(
    data: List[Tuple[List[str], List[Tuple[int, int, str]]]],
    prior: FrequencyPrior,
    tau: float,
    mode: str = "pair",
    ks: Sequence[int] = DEFAULT_KS,
    alpha: float = 3.75,
) -> Dict[str, Any]:
    """Rank pairs by the class-balanced score. Mirrors frequency_prior_baseline.

    The only difference from the tau=0 baseline is the subtraction of
    tau * global_log_prob from each predicate's score before the softmax.

    The cross-pair ranking key must be the SOFTMAX PROBABILITY of the chosen
    predicate, exactly as the baseline computes it -- not the raw log-prob.
    Both give the same per-pair argmax (softmax is monotonic), but ranking
    pairs against each other by an unnormalised log-prob reorders them, which
    changes R@20 on images with more than 20 candidate pairs. Getting this
    wrong would break the claim that tau=0 IS the baseline.
    """
    pooled_hits = {k: 0 for k in ks}
    per_pred_hits = {k: defaultdict(int) for k in ks}
    per_pred_gt: Dict[str, int] = defaultdict(int)
    image_recall_sum = {k: 0.0 for k in ks}
    pooled_gt = 0
    argmax_counts: Dict[str, int] = defaultdict(int)

    marginal = prior.global_lp

    for labels, triplets in data:
        scored: List[Tuple[float, int]] = []
        for (s, o, _p) in triplets:
            row, _level = prior.row(labels[s], labels[o], mode)
            adjusted = [row[i] - tau * marginal[i] for i in range(prior.n_pred)]
            probs = softmax([alpha * v for v in adjusted])
            best = max(range(prior.n_pred), key=lambda i: probs[i])
            scored.append((probs[best], best))
            argmax_counts[prior.pred_vocab[best]] += 1

        order = sorted(range(len(triplets)), key=lambda i: -scored[i][0])
        for (_s, _o, p) in triplets:
            per_pred_gt[p] += 1
        pooled_gt += len(triplets)

        for k in ks:
            matched: set = set()
            hits = 0
            for ci in order[:k]:
                cs, co, _ = triplets[ci]
                pname = prior.pred_vocab[scored[ci][1]]
                for gi, (gs, go, gp) in enumerate(triplets):
                    if gi in matched:
                        continue
                    if gs == cs and go == co and gp == pname:
                        matched.add(gi)
                        hits += 1
                        per_pred_hits[k][gp] += 1
                        break
            pooled_hits[k] += hits
            image_recall_sum[k] += hits / max(1, len(triplets))

    bucket = head_body_tail(prior, per_pred_gt)
    metrics: Dict[str, Any] = {"tau": tau, "alpha": alpha, "n_images": len(data), "n_gt_triplets": pooled_gt}
    for k in ks:
        recalls = {p: per_pred_hits[k][p] / per_pred_gt[p] for p in per_pred_gt if per_pred_gt[p] > 0}
        metrics[f"R@{k}"] = pooled_hits[k] / max(1, pooled_gt)
        metrics[f"mR@{k}"] = sum(recalls.values()) / max(1, len(recalls))
        metrics[f"image_mean_R@{k}"] = image_recall_sum[k] / max(1, len(data))
        if k == 50:
            for b in ("head", "body", "tail"):
                vals = [v for p, v in recalls.items() if bucket.get(p) == b]
                metrics[f"{b}_mR@50"] = sum(vals) / max(1, len(vals))
    metrics["n_distinct_predicates_predicted"] = len(argmax_counts)
    metrics["argmax_top5"] = [
        {"predicate": n, "count": c}
        for n, c in sorted(argmax_counts.items(), key=lambda kv: -kv[1])[:5]
    ]
    return metrics


def verdict(rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Apply the criteria fixed in docs/DECISION_RULE_HYPOTHESIS.md.

    The registered H2 SUPPORTED criterion is "mR@50 gain >= +3.0 points AT SOME
    tau, with R@50 loss <= 3.0 points". It is therefore a search over taus that
    satisfy the R@50 constraint -- NOT the mR-argmax tau. Taking the mR-argmax
    first picks the most aggressive tau on the sweep, which necessarily blows
    the R@50 budget, and reports PARETO-ONLY even when a cheap tau satisfies
    both halves of the criterion.
    """
    base = next(r for r in rows if r["tau"] == 0.0)
    r0, m0 = base["R@50"] * 100, base["mR@50"] * 100

    def deltas(row: Dict[str, Any]) -> Tuple[float, float]:
        return row["mR@50"] * 100 - m0, row["R@50"] * 100 - r0

    # Taus that stay inside the pre-registered R@50 budget.
    affordable = [r for r in rows if r["tau"] != 0.0 and deltas(r)[1] >= -MAX_R_LOSS_SUPPORT]
    if affordable:
        best_aff = max(affordable, key=lambda r: r["mR@50"])
        d_mr, d_r = deltas(best_aff)
        if d_mr >= MIN_MR_GAIN_SUPPORT:
            return ("H2 SUPPORTED",
                    f"tau={best_aff['tau']} gains {d_mr:+.2f} mR@50 for {d_r:+.2f} R@50 "
                    f"(within the {MAX_R_LOSS_SUPPORT:.1f}-point R@50 budget): the decision rule "
                    f"was discarding recoverable signal")

    best_any = max(rows, key=lambda r: r["mR@50"])
    d_mr, d_r = deltas(best_any)
    if d_mr < MIN_MR_GAIN_ANY:
        return ("H2 REJECTED",
                f"best tau gains only {d_mr:+.2f} mR@50; recalibration is not the missing piece")
    if d_mr > 0 and d_r <= -d_mr:
        return ("PARETO-ONLY",
                f"tau={best_any['tau']} gains {d_mr:+.2f} mR@50 but loses {d_r:+.2f} R@50, and no "
                f"tau achieves {MIN_MR_GAIN_SUPPORT:+.1f} mR@50 within the R@50 budget: this only "
                f"moves along a known trade-off and is NOT a contribution")
    return ("INCONCLUSIVE",
            f"best tau gains {d_mr:+.2f} mR@50 for {d_r:+.2f} R@50, between the fixed thresholds")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default=str(DEFAULT_PRIOR))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--mode", choices=("pair", "global"), default="pair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--taus", default=",".join(str(t) for t in DEFAULT_TAUS))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    prior_path, split_path = Path(args.prior), Path(args.split)
    for p in (prior_path, split_path):
        if not p.exists():
            print(f"[FAIL] missing: {p}", file=sys.stderr)
            return 2

    prior = FrequencyPrior.load(prior_path)
    data = load_split(split_path, prior, limit=args.limit or None)
    taus = [float(t) for t in str(args.taus).split(",") if t.strip() != ""]

    print("=" * 78)
    print("  DECISION-RULE PROBE   score = log P(p|s,o) - tau * log P(p)")
    print("=" * 78)
    print(f"  prior : {prior_path}")
    print(f"  split : {split_path}   images {len(data):,}")
    print(f"  tau=0 is exactly the frequency-prior baseline; gains are attributable")
    print(f"  to the adjustment alone. log P(p) is the TRAIN marginal.\n")
    print(f"{'tau':>6}{'R@50':>10}{'mR@50':>10}{'head':>9}{'body':>9}{'tail':>9}{'#preds':>8}")

    rows: List[Dict[str, Any]] = []
    for tau in taus:
        m = evaluate_tau(data, prior, tau, mode=args.mode)
        rows.append(m)
        print(f"{tau:>6.2f}{m['R@50']*100:>9.2f}%{m['mR@50']*100:>9.2f}%"
              f"{m['head_mR@50']*100:>8.1f}%{m['body_mR@50']*100:>8.1f}%"
              f"{m['tail_mR@50']*100:>8.1f}%{m['n_distinct_predicates_predicted']:>8}")

    base = next(r for r in rows if r["tau"] == 0.0)
    print(f"\n  deltas vs tau=0 ({base['R@50']*100:.2f} / {base['mR@50']*100:.2f}):")
    for m in rows:
        if m["tau"] == 0.0:
            continue
        print(f"    tau={m['tau']:<5} dR@50 {(m['R@50']-base['R@50'])*100:+6.2f}   "
              f"dmR@50 {(m['mR@50']-base['mR@50'])*100:+6.2f}   "
              f"dtail {(m['tail_mR@50']-base['tail_mR@50'])*100:+6.2f}")

    v, explanation = verdict(rows)
    print("\n" + "=" * 78)
    print(f"  VERDICT: {v}")
    print(f"  {explanation}")
    print("=" * 78)

    out = {"prior": str(prior_path), "split": str(split_path), "mode": args.mode,
           "n_images": len(data), "rows": rows, "verdict": v, "verdict_explanation": explanation}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[INFO] written to {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
