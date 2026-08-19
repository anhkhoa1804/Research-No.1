#!/usr/bin/env python
"""How much information is left for vision, given the object-label pair?

The frequency-prior control (tools/frequency_prior_baseline.py) shows a
pair-conditioned lookup table reaches R@50 = 64.37 % / mR@50 = 20.30 % with
the model contributing nothing. This tool asks the follow-up question that
determines whether the project has a viable research direction at all:

    of the remaining 35.63 points, how much is REAL, VISUALLY RESOLVABLE
    signal, and how much is annotation style that no model can win?

It reports four things, all from data alone -- no model, no CLIP, no GPU:

  1. ORACLE-RERANK CURVE -- what a perfect visual reranker over the prior's
     top-K candidates would score. Tells us whether the task is "rerank a
     handful of plausible predicates" or "find a needle in 50".

  2. RANK of the true predicate under the prior.

  3. GENERIC vs DECIDABLE split. VG's vocabulary mixes interchangeable
     attachment/possession terms (on, in, of, has, with, wearing, near --
     "wheel of bus" vs "wheel on bus" is annotator choice) with predicates
     whose truth is genuinely determined by the image given the same object
     pair (under, behind, riding, carrying, hanging from...). Only the
     latter is recoverable by vision.

  4. IRREDUCIBLE CEILING -- the best accuracy any label-only model can reach
     on repeated pair types, which bounds the prior from above.

Usage:
    python tools/headroom_analysis.py
    python tools/headroom_analysis.py --out runs/analysis/headroom.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PRIOR = Path("checkpoints/demo_best/frequency_prior.json")
DEFAULT_SPLIT = Path("datasets_vg150_clean/validation.jsonl")

# Predicates VG annotators use interchangeably for attachment, possession and
# proximity. Confusion *among* these is largely irreducible label style, not a
# visual failure -- "wheel of bus" / "wheel on bus" describe the same pixels.
GENERIC_PREDICATES = {
    "on", "in", "of", "has", "with", "wearing", "wears", "near", "next to",
    "at", "and", "to", "for", "from", "part of", "belonging to", "attached to",
    "made of", "says", "over", "along", "against", "across", "between",
}


def load_prior(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab = [str(p).strip().lower() for p in raw["predicate_vocab"]]
    return {
        "vocab": vocab,
        "index": {p: i for i, p in enumerate(vocab)},
        "global": raw["global_log_probs"],
        "pair": raw.get("pair_log_probs", {}),
        "subject": raw.get("subject_log_probs", {}),
        "object": raw.get("object_log_probs", {}),
    }


def prior_row(prior: Dict[str, Any], subject: str, object_: str) -> Tuple[List[float], bool]:
    row = prior["pair"].get(f"{subject}||{object_}")
    if row is not None:
        return row, True
    s = prior["subject"].get(subject)
    o = prior["object"].get(object_)
    if s is not None and o is not None:
        return [0.5 * (a + b) for a, b in zip(s, o)], False
    return (s or o or prior["global"]), False


def load_split(path: Path, prior: Dict[str, Any], limit: Optional[int] = None):
    out = []
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
            labels = []
            for obj in ex.get("objects") or []:
                names = obj.get("names") or []
                labels.append(str(names[0]).strip().lower() if names else "")
            triplets = []
            for rel in rels:
                s = int(rel.get("subject_id", -1))
                o = int(rel.get("object_id", -1))
                p = str(rel.get("predicate", "")).strip().lower()
                if 0 <= s < len(labels) and 0 <= o < len(labels) and p in prior["index"]:
                    triplets.append((s, o, p))
            if triplets:
                out.append((labels, triplets))
    return out


def analyse(data, prior: Dict[str, Any]) -> Dict[str, Any]:
    vocab, index = prior["vocab"], prior["index"]
    n_pred = len(vocab)
    decidable = {p for p in vocab if p not in GENERIC_PREDICATES}

    rank_hist: Counter = Counter()
    dec_rank: Counter = Counter()
    entropies: List[float] = []
    pairtype: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    n_total = n_dec_gt = n_err = n_err_dec = n_err_generic_pair = 0
    multilabel_pairs = total_pairs = 0

    for labels, triplets in data:
        by_pair = defaultdict(list)
        for (s, o, p) in triplets:
            by_pair[(s, o)].append(p)
        total_pairs += len(by_pair)
        multilabel_pairs += sum(1 for v in by_pair.values() if len(v) > 1)

        for (s, o, gp) in triplets:
            n_total += 1
            sl, ol = labels[s], labels[o]
            pairtype[(sl, ol)][gp] += 1
            row, _exact = prior_row(prior, sl, ol)
            order = sorted(range(n_pred), key=lambda i: -row[i])
            rank = order.index(index[gp]) + 1
            rank_hist[rank] += 1

            m = max(row)
            exps = [math.exp(v - m) for v in row]
            z = sum(exps)
            entropies.append(-sum((e / z) * math.log(e / z + 1e-12) for e in exps))

            is_dec = gp in decidable
            if is_dec:
                n_dec_gt += 1
                dec_rank[rank] += 1
            if rank > 1:
                n_err += 1
                top1 = vocab[order[0]]
                confusion[(top1, gp)] += 1
                if is_dec:
                    n_err_dec += 1
                elif top1 in GENERIC_PREDICATES:
                    n_err_generic_pair += 1

    def cum(hist: Counter, k: int) -> int:
        return sum(c for r, c in hist.items() if r <= k)

    noisy = noisy_total = 0
    for counts in pairtype.values():
        n = sum(counts.values())
        if n < 5:
            continue
        noisy_total += n
        noisy += n - counts.most_common(1)[0][1]

    return {
        "n_images": len(data),
        "n_triplets": n_total,
        "n_pairs": total_pairs,
        "multilabel_pair_fraction": multilabel_pairs / max(1, total_pairs),
        "oracle_rerank": {str(k): cum(rank_hist, k) / max(1, n_total) for k in (1, 2, 3, 5, 8, 10, 20, 50)},
        "rank_histogram": {
            "1": rank_hist[1],
            "2": rank_hist[2],
            "3": rank_hist[3],
            "4-5": sum(rank_hist[r] for r in (4, 5)),
            "6-10": sum(rank_hist[r] for r in range(6, 11)),
            "11-20": sum(rank_hist[r] for r in range(11, 21)),
            "21-50": sum(rank_hist[r] for r in range(21, 51)),
        },
        "mean_conditional_entropy_nats": sum(entropies) / max(1, len(entropies)),
        "max_entropy_nats": math.log(n_pred),
        "n_decidable_predicates": len(decidable),
        "decidable_gt_fraction": n_dec_gt / max(1, n_total),
        "decidable_oracle_rerank": {str(k): cum(dec_rank, k) / max(1, n_dec_gt) for k in (1, 2, 3, 5, 10)},
        "n_errors": n_err,
        "error_fraction": n_err / max(1, n_total),
        "errors_truth_decidable": n_err_dec,
        "errors_generic_to_generic": n_err_generic_pair,
        "recoverable_headroom_points": n_err_dec / max(1, n_total) * 100.0,
        "total_headroom_points": n_err / max(1, n_total) * 100.0,
        "irreducible_label_noise_fraction": noisy / max(1, noisy_total),
        "label_only_ceiling": (noisy_total - noisy) / max(1, noisy_total),
        "top_confusions": [
            {"prior_says": a, "truth_is": b, "count": c}
            for (a, b), c in confusion.most_common(15)
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default=str(DEFAULT_PRIOR))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    prior_path, split_path = Path(args.prior), Path(args.split)
    for path, label in ((prior_path, "frequency prior"), (split_path, "evaluation split")):
        if not path.exists():
            print(f"[FAIL] {label} not found: {path}", file=sys.stderr)
            return 2

    prior = load_prior(prior_path)
    data = load_split(split_path, prior, args.limit or None)
    r = analyse(data, prior)

    print("=" * 72)
    print("  VISUAL HEADROOM ANALYSIS")
    print("=" * 72)
    print(f"  {r['n_images']:,} images   {r['n_triplets']:,} GT triplets\n")

    print("1. ORACLE-RERANK CURVE (perfect visual reordering of the prior's top-K)")
    for k, v in r["oracle_rerank"].items():
        marker = "   <- the prior itself" if k == "1" else ""
        print(f"     oracle@{k:<3} = {v * 100:6.2f}%{marker}")

    print("\n2. RANK OF THE TRUE PREDICATE UNDER THE PRIOR")
    for label, count in r["rank_histogram"].items():
        print(f"     rank {label:<7} {count:>8,}  ({count / r['n_triplets'] * 100:5.2f}%)")

    h, hmax = r["mean_conditional_entropy_nats"], r["max_entropy_nats"]
    print(f"\n3. RESIDUAL UNCERTAINTY GIVEN THE LABEL PAIR")
    print(f"     H(pred | subj,obj) = {h:.3f} nats ({h / math.log(2):.2f} bits) of {hmax:.3f} max")
    print(f"     the label pair removes only {(1 - h / hmax) * 100:.1f}% of predicate uncertainty")

    print(f"\n4. GENERIC vs DECIDABLE  ({r['n_decidable_predicates']} of 50 predicates are decidable)")
    print(f"     decidable share of GT triplets : {r['decidable_gt_fraction'] * 100:.1f}%")
    print(f"     prior errors, truth decidable  : {r['errors_truth_decidable']:,} "
          f"({r['errors_truth_decidable'] / max(1, r['n_errors']) * 100:.1f}% of errors)")
    print(f"     prior errors, generic->generic : {r['errors_generic_to_generic']:,} "
          f"({r['errors_generic_to_generic'] / max(1, r['n_errors']) * 100:.1f}% of errors)")
    print(f"\n     TOTAL headroom       = {r['total_headroom_points']:.2f} R@50 points")
    print(f"     RECOVERABLE headroom = {r['recoverable_headroom_points']:.2f} R@50 points")
    print(f"     (the difference is largely annotation style, not visual failure)")

    print(f"\n5. THE PRIOR IS WEAKEST EXACTLY WHERE VISION SHOULD WIN")
    for k, v in r["decidable_oracle_rerank"].items():
        tag = "   <- prior alone on decidable predicates" if k == "1" else ""
        print(f"     decidable oracle@{k:<3} = {v * 100:6.2f}%{tag}")

    print(f"\n6. IRREDUCIBLE CEILING (pair types seen >=5x)")
    print(f"     best label-only accuracy = {r['label_only_ceiling'] * 100:.1f}%")
    print(f"     {r['irreducible_label_noise_fraction'] * 100:.1f}% cannot be fixed without vision")

    print("\n  top confusions (prior says -> truth is):")
    for c in r["top_confusions"][:8]:
        print(f"     {c['prior_says']:<16} -> {c['truth_is']:<16} {c['count']:>7,}")
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"[INFO] written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
