#!/usr/bin/env python
"""How much of the SAME oracle@5 headroom does tau convert? CPU-only.

The appearance probe's pre-registered primary metric is

    captured = (mR_arm - mR_prior) / (mR_oracle@5 - mR_prior)

measured on the prior's top-5 candidate set. Experiment B scored -1.2 % on it.
This tool computes the identical quantity for the tau recalibration on the
A' 3,000-image subset, so the two interventions are compared in the same
units on the same denominator rather than by eyeballing two different tables.

It also reports, per predicate, the prior's top-1 recall against its top-5
COVERAGE. A class with high coverage and near-zero top-1 recall is one where
the prior already contains the right answer and only the decision rule hides
it -- which is exactly the headroom tau converts, and exactly the headroom
appearance failed to convert.

oracle@5 is defined as in tools/appearance_probe.py:score --
predict the GT when it is inside the prior's top-5, else the prior's argmax.

Usage:
    python tools/prior_temperature_vs_oracle.py --limit 3000 --taus 0.0,0.05,0.1,0.2 \
        --out runs/<name>/tau_vs_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.frequency_prior_baseline import FrequencyPrior, load_split  # noqa: E402
from tools.decision_rule_probe import head_body_tail  # noqa: E402

K = 5


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--split", default="datasets_vg150_clean/validation.jsonl")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--taus", default="0.0,0.05,0.1,0.2")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    prior = FrequencyPrior.load(Path(args.prior))
    data = load_split(Path(args.split), prior, limit=int(args.limit))
    taus = [float(t) for t in str(args.taus).split(",") if t.strip()]
    marg = prior.global_lp
    PV = prior.pred_vocab

    gt_n: Dict[str, int] = defaultdict(int)
    cover_n: Dict[str, int] = defaultdict(int)     # GT inside prior top-5
    hits: Dict[float, Dict[str, int]] = {t: defaultdict(int) for t in taus}
    oracle_hits: Dict[str, int] = defaultdict(int)
    n = in5 = 0

    for labels, triplets in data:
        for (s, o, gp) in triplets:
            row, _ = prior.row(labels[s], labels[o], "pair")
            top5 = sorted(range(prior.n_pred), key=lambda i: -row[i])[:K]
            gi = prior.pred_index[gp]
            gt_n[gp] += 1
            n += 1
            covered = gi in top5
            if covered:
                cover_n[gp] += 1
                in5 += 1
                oracle_hits[gp] += 1           # oracle picks GT when covered
            else:
                # oracle falls back to the prior's argmax (tau=0 argmax)
                if PV[max(range(prior.n_pred), key=lambda i: row[i])] == gp:
                    oracle_hits[gp] += 1
            for t in taus:
                adj_best = max(range(prior.n_pred), key=lambda i: row[i] - t * marg[i])
                if PV[adj_best] == gp:
                    hits[t][gp] += 1

    classes = [p for p in gt_n if gt_n[p] > 0]
    def mR(h: Dict[str, int]) -> float:
        return sum(h[p] / gt_n[p] for p in classes) / max(1, len(classes))

    mr_oracle = mR(oracle_hits)
    mr_by_tau = {t: mR(hits[t]) for t in taus}
    base = mr_by_tau[0.0]
    head_n = head_body_tail(prior, gt_n)

    print("=" * 84)
    print("TAU vs the appearance probe's OWN pre-registered denominator")
    print("=" * 84)
    print(f"  images {len(data):,}   GT pairs {n:,}   classes with GT {len(classes)}")
    print(f"  coverage@5           : {in5/n*100:.2f}%")
    print(f"  prior mR@50 (tau=0)  : {base*100:.2f}%")
    print(f"  ORACLE@5 mR@50       : {mr_oracle*100:.2f}%")
    print(f"  headroom             : {(mr_oracle-base)*100:+.2f} mR points\n")
    print(f"{'tau':>7}{'mR@50':>10}{'d vs tau=0':>13}{'captured headroom':>20}")
    for t in taus:
        d = (mr_by_tau[t] - base) * 100
        cap = (mr_by_tau[t] - base) / max(1e-9, (mr_oracle - base)) * 100
        print(f"{t:>7}{mr_by_tau[t]*100:>9.2f}%{d:>+12.2f}{cap:>19.1f}%")
    print("\n  reference: experiment B (frozen ViT-L/14-336 appearance, selected lambda)")
    print("             captured -1.2 % of this same quantity on its own subsample.")

    print("\n" + "=" * 84)
    print("WHERE THE PRIOR ALREADY KNOWS: top-5 coverage vs top-1 recall (tau=0)")
    print("=" * 84)
    print(f"  {'predicate':<16}{'bucket':>7}{'n':>7}{'cover@5':>10}{'top1 tau=0':>12}{'top1 tau=0.1':>14}{'unused':>9}")
    rows = []
    for p in sorted(classes, key=lambda q: -(cover_n[q] / gt_n[q] - hits[0.0][q] / gt_n[q])):
        cov = cover_n[p] / gt_n[p]
        r0 = hits[0.0][p] / gt_n[p]
        r1 = hits.get(0.1, {}).get(p, 0) / gt_n[p] if 0.1 in hits else float("nan")
        rows.append({"predicate": p, "bucket": head_n.get(p, "?"), "n": gt_n[p],
                     "coverage@5": cov, "top1_tau0": r0, "top1_tau0.1": r1,
                     "unused_coverage": cov - r0})
    for r in rows[:22]:
        print(f"  {r['predicate']:<16}{r['bucket']:>7}{r['n']:>7,}{r['coverage@5']*100:>9.1f}%"
              f"{r['top1_tau0']*100:>11.1f}%{r['top1_tau0.1']*100:>13.1f}%"
              f"{r['unused_coverage']*100:>8.1f}")
    tot_unused = sum((cover_n[p] - hits[0.0][p]) for p in classes)
    print(f"\n  GT pairs where the truth is inside the prior's top-5 but NOT its top-1:"
          f" {tot_unused:,} ({tot_unused/n*100:.2f}% of all pairs)")

    res: Dict[str, Any] = {
        "n_images": len(data), "n_pairs": n, "n_classes": len(classes),
        "coverage@5": in5 / n, "prior_mR": base, "oracle5_mR": mr_oracle,
        "headroom_points": (mr_oracle - base) * 100,
        "taus": [{"tau": t, "mR@50": mr_by_tau[t],
                  "delta_mR_points": (mr_by_tau[t] - base) * 100,
                  "captured_headroom_pct": (mr_by_tau[t] - base) / max(1e-9, (mr_oracle - base)) * 100}
                 for t in taus],
        "per_predicate": rows,
        "pairs_truth_in_top5_not_top1": tot_unused,
    }
    if args.out:
        pth = Path(args.out); pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\n[tau-vs-oracle] written to {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
