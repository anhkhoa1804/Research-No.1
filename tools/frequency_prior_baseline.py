#!/usr/bin/env python
"""Frequency-prior-only PredCls control — the baseline every model number must beat.

WHY THIS EXISTS
---------------
The pair-conditioned frequency prior (`frequency_prior.json`) encodes
P(predicate | subject_label, object_label): pure label co-occurrence, with no
visual information whatsoever. Under this repository's GT-pairs PredCls
protocol it scores R@50 = 64.37% / mR@50 = 20.30% *by itself*, against a
historical claim of 67.09% / 22.64% for the full model+prior system.

That makes it the control: **the only meaningful measure of what the model
contributes is its delta over this number.** A calibrated result reported
without it is indistinguishable from a lookup table.

WHAT THIS REPRODUCES
--------------------
The scoring path of `openvocab_rel/evals.py:eval_sgg_standard` under
`eval_sgg_use_gt_pairs=true`, with the model term set to zero:

    candidates   = _extract_gt_pairs(ex)          one per GT relationship
    rel_logits   = 0 + alpha * pair_log_prior     model contributes nothing
    rel_probs    = softmax(rel_logits)
    pred, score  = rel_probs.max(-1)              ONE predicate per pair
    rank by score, take top-K, match against GT

Metrics match the evaluator's definitions: pooled `R@K` = sum(hits)/sum(GT),
`mR@K` = unweighted mean of per-predicate recall, `image_mean_R@K` = mean over
images of per-image recall.

No torch, no CLIP, no GPU, no model. Runs on CPU in well under a minute.

Usage:
    python tools/frequency_prior_baseline.py
    python tools/frequency_prior_baseline.py --mode global      # ablate conditioning
    python tools/frequency_prior_baseline.py --alpha 1.0        # alpha is inert here
    python tools/frequency_prior_baseline.py --out runs/<run>/frequency_prior_control.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PRIOR = Path("checkpoints/demo_best/frequency_prior.json")
DEFAULT_SPLIT = Path("datasets_vg150_clean/validation.jsonl")
DEFAULT_KS: Tuple[int, ...] = (20, 50, 100)


def freq_key(subject_label: str, object_label: str) -> str:
    """Must match openvocab_rel/evals.py:_freq_bias_key exactly."""
    return f"{str(subject_label).strip().lower()}||{str(object_label).strip().lower()}"


def softmax(values: Sequence[float]) -> List[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class FrequencyPrior:
    def __init__(self, raw: Dict[str, Any]):
        self.pred_vocab = [str(p).strip().lower() for p in raw["predicate_vocab"]]
        self.n_pred = len(self.pred_vocab)
        self.pred_index = {p: i for i, p in enumerate(self.pred_vocab)}
        self.global_lp: List[float] = raw["global_log_probs"]
        self.pair_lp: Dict[str, List[float]] = raw.get("pair_log_probs", {})
        self.subject_lp: Dict[str, List[float]] = raw.get("subject_log_probs", {})
        self.object_lp: Dict[str, List[float]] = raw.get("object_log_probs", {})

    @classmethod
    def load(cls, path: Path) -> "FrequencyPrior":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def row(self, subject_label: str, object_label: str, mode: str = "pair") -> Tuple[List[float], str]:
        """Return (log-prob row, which backoff level was used).

        Backoff order mirrors evals.py:_frequency_bias_for_pairs.
        """
        if mode == "global":
            return self.global_lp, "global"
        row = self.pair_lp.get(freq_key(subject_label, object_label))
        if row is not None:
            return row, "pair"
        subj = self.subject_lp.get(str(subject_label).strip().lower())
        obj = self.object_lp.get(str(object_label).strip().lower())
        if subj is not None and obj is not None:
            return [0.5 * (a + b) for a, b in zip(subj, obj)], "subject_object"
        if subj is not None:
            return subj, "single"
        if obj is not None:
            return obj, "single"
        return self.global_lp, "global"


def load_split(path: Path, prior: FrequencyPrior, limit: Optional[int] = None) -> List[Tuple[List[str], List[Tuple[int, int, str]]]]:
    """Yield (object_labels, [(subj_idx, obj_idx, predicate), ...]) per image."""
    out: List[Tuple[List[str], List[Tuple[int, int, str]]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rels = ex.get("relationships") or []
            if not rels:
                continue
            labels: List[str] = []
            for obj in ex.get("objects") or []:
                names = obj.get("names") or []
                labels.append(str(names[0]).strip().lower() if names else "")
            triplets: List[Tuple[int, int, str]] = []
            for rel in rels:
                s = int(rel.get("subject_id", -1))
                o = int(rel.get("object_id", -1))
                p = str(rel.get("predicate", "")).strip().lower()
                if 0 <= s < len(labels) and 0 <= o < len(labels) and p in prior.pred_index:
                    triplets.append((s, o, p))
            if triplets:
                out.append((labels, triplets))
    return out


def evaluate(
    data: List[Tuple[List[str], List[Tuple[int, int, str]]]],
    prior: FrequencyPrior,
    alpha: float = 3.75,
    mode: str = "pair",
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Any]:
    pooled_hits = {k: 0 for k in ks}
    per_pred_hits = {k: defaultdict(int) for k in ks}
    image_recall_sum = {k: 0.0 for k in ks}
    per_pred_gt: Dict[str, int] = defaultdict(int)
    pooled_gt = 0
    n_candidates = 0
    backoff: Dict[str, int] = defaultdict(int)
    argmax_counts: Dict[str, int] = defaultdict(int)

    for labels, triplets in data:
        scored: List[Tuple[float, int]] = []
        for (s, o, _p) in triplets:
            row, level = prior.row(labels[s], labels[o], mode)
            backoff[level] += 1
            probs = softmax([alpha * v for v in row])
            best = max(range(prior.n_pred), key=lambda i: probs[i])
            scored.append((probs[best], best))
            argmax_counts[prior.pred_vocab[best]] += 1
        n_candidates += len(triplets)

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

    n_images = len(data)
    metrics: Dict[str, Any] = {
        "n_images": n_images,
        "n_gt_triplets": pooled_gt,
        "n_candidates": n_candidates,
        "avg_candidates_per_image": n_candidates / max(1, n_images),
        "alpha": alpha,
        "mode": mode,
        "n_predicate_classes_with_gt": len(per_pred_gt),
    }
    for k in ks:
        recalls = [per_pred_hits[k][p] / per_pred_gt[p] for p in per_pred_gt if per_pred_gt[p] > 0]
        metrics[f"R@{k}"] = pooled_hits[k] / max(1, pooled_gt)
        metrics[f"mR@{k}"] = sum(recalls) / max(1, len(recalls))
        metrics[f"image_mean_R@{k}"] = image_recall_sum[k] / max(1, n_images)
    total_backoff = max(1, sum(backoff.values()))
    metrics["prior_lookup_fractions"] = {
        level: count / total_backoff for level, count in sorted(backoff.items())
    }
    metrics["argmax_distribution_top8"] = [
        {"predicate": name, "count": count, "fraction": count / max(1, n_candidates)}
        for name, count in sorted(argmax_counts.items(), key=lambda kv: -kv[1])[:8]
    ]
    metrics["n_distinct_predicates_predicted"] = len(argmax_counts)
    return metrics


def protocol_ceiling(data: List[Tuple[List[str], List[Tuple[int, int, str]]]], k: int = 50) -> float:
    """Max achievable R@k when only one predicate may be emitted per pair."""
    hits = 0
    total = 0
    for _labels, triplets in data:
        by_pair: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for (s, o, p) in triplets:
            by_pair[(s, o)].append(p)
        hits += min(k, len(by_pair))
        total += len(triplets)
    return hits / max(1, total)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default=str(DEFAULT_PRIOR))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--alpha", type=float, default=3.75)
    ap.add_argument("--mode", choices=("pair", "global"), default="pair",
                    help="'global' ablates subject/object conditioning entirely")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    ap.add_argument("--ceiling", action="store_true", help="also report the one-predicate-per-pair ceiling")
    ap.add_argument("--out", default="", help="write metrics JSON here")
    args = ap.parse_args(argv)

    prior_path, split_path = Path(args.prior), Path(args.split)
    for path, label in ((prior_path, "frequency prior"), (split_path, "evaluation split")):
        if not path.exists():
            print(f"[FAIL] {label} not found: {path}", file=sys.stderr)
            print("       This artifact is gitignored and must be transferred out of band —", file=sys.stderr)
            print("       see docs/HISTORICAL_CHECKPOINT_MANIFEST.md.", file=sys.stderr)
            return 2

    print("=" * 70)
    print("  FREQUENCY-PRIOR-ONLY CONTROL  (model contributes exactly zero)")
    print("=" * 70)
    prior = FrequencyPrior.load(prior_path)
    print(f"  prior : {prior_path}  ({prior.n_pred} predicates, {len(prior.pair_lp):,} pair rows)")
    data = load_split(split_path, prior, args.limit or None)
    print(f"  split : {split_path}")

    metrics = evaluate(data, prior, alpha=args.alpha, mode=args.mode)
    print(f"\n  images {metrics['n_images']:,}   GT triplets {metrics['n_gt_triplets']:,}"
          f"   avg candidates/image {metrics['avg_candidates_per_image']:.2f}")
    print(f"  mode={metrics['mode']}  alpha={metrics['alpha']}")
    print("\n  lookup level: " + "  ".join(
        f"{lvl} {frac * 100:.1f}%" for lvl, frac in metrics["prior_lookup_fractions"].items()))

    print(f"\n  {'':>8}{'R@K':>12}{'mR@K':>12}{'image_mean_R@K':>18}")
    for k in DEFAULT_KS:
        print(f"  {'K=' + str(k):>8}{metrics[f'R@{k}'] * 100:>11.2f}%"
              f"{metrics[f'mR@{k}'] * 100:>11.2f}%{metrics[f'image_mean_R@{k}'] * 100:>17.2f}%")

    print(f"\n  argmax distribution (top 8 of {metrics['n_distinct_predicates_predicted']} predicted):")
    for row in metrics["argmax_distribution_top8"]:
        print(f"      {row['predicate']:<20}{row['count']:>8,}  ({row['fraction'] * 100:.1f}%)")

    if args.ceiling:
        ceil = protocol_ceiling(data)
        metrics["protocol_ceiling_R@50"] = ceil
        print(f"\n  protocol ceiling (1 predicate/pair, K=50): {ceil * 100:.2f}%")

    print("\n" + "-" * 70)
    print("  This is the number any model result must be compared against.")
    print("  Report the DELTA over this control, never the calibrated score alone.")
    print("-" * 70)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[INFO] metrics written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
