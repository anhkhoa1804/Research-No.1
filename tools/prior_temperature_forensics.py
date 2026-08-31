#!/usr/bin/env python
"""WHY does score = log P(p|s,o) - tau*log P(p) buy +4.03 mR@50 for free?
CPU-only, read-only, no GPU, no model, no training.

`tools/decision_rule_probe.py` MEASURED the effect and applied its
pre-registered verdict. This tool does not re-measure it and does not touch
its thresholds -- it decomposes the mechanism:

  1. TOP-K INERTNESS. Under GT pairs the candidate set IS the GT relationship
     set, so tau cannot change candidate INCLUSION. It can only change (a) the
     predicate emitted per pair and (b) the cross-pair ranking key. If almost
     every image has fewer than K candidates, (b) is inert too and mR@50 is
     exactly class-averaged TOP-1 ACCURACY. Measured, not assumed.

  2. FLIP ACCOUNTING. Every pair whose argmax changes between two taus, keyed
     by (from-class -> to-class) and by whether the flip was
     wrong->right, right->wrong, or wrong->wrong.

  3. LEVERAGE. mR is an UNWEIGHTED mean over classes, so one instance moved
     out of a class with n=13,900 costs 1/13,900 of a class-recall while the
     same instance moved into a class with n=22 buys 1/22. This tool computes
     the realised leverage ratio rather than asserting it.

  4. ENTROPY. Mean Shannon entropy of the per-pair softmax at each tau, and
     the support of the prediction distribution (how many classes are ever
     emitted). Reported at the protocol's alpha=3.75 and at alpha=1.0, to
     separate what alpha does from what tau does.

VALIDATION GATE: R@50 and mR@50 at tau=0 and tau=0.1 must equal the values
recorded by tools/decision_rule_probe.py on the same subset
(66.80/21.98 and 66.16/26.00). If they do not, this tool is wrong and every
decomposition below must be discarded.

Usage:
    python tools/prior_temperature_forensics.py --limit 3000 \
        --taus 0.0,0.1,0.2 --out runs/<name>/forensics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.frequency_prior_baseline import FrequencyPrior, load_split, softmax  # noqa: E402
from tools.decision_rule_probe import head_body_tail  # noqa: E402

KS = (20, 50, 100)


def score_pairs(prior: FrequencyPrior, labels, triplets, tau: float, alpha: float):
    """Exactly tools/decision_rule_probe.evaluate_tau's per-pair computation."""
    marginal = prior.global_lp
    out = []
    for (s, o, _p) in triplets:
        row, _lvl = prior.row(labels[s], labels[o], "pair")
        adjusted = [row[i] - tau * marginal[i] for i in range(prior.n_pred)]
        probs = softmax([alpha * v for v in adjusted])
        best = max(range(prior.n_pred), key=lambda i: probs[i])
        out.append((probs[best], best, probs))
    return out


def entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def run_tau(data, prior: FrequencyPrior, tau: float, alpha: float) -> Dict[str, Any]:
    pooled_hits = {k: 0 for k in KS}
    per_pred_hits = {k: defaultdict(int) for k in KS}
    per_pred_gt: Dict[str, int] = defaultdict(int)
    pooled_gt = 0
    argmax_counts: Dict[str, int] = defaultdict(int)
    ent_sum = 0.0
    n_pairs = 0
    top1_correct = 0
    # per-pair argmax, for flip accounting across taus
    argmax_trace: List[Tuple[str, str]] = []   # (predicted, truth)

    for labels, triplets in data:
        scored = score_pairs(prior, labels, triplets, tau, alpha)
        for (pmax, best, probs), (_s, _o, gp) in zip(scored, triplets):
            ent_sum += entropy(probs)
            n_pairs += 1
            pname = prior.pred_vocab[best]
            argmax_counts[pname] += 1
            argmax_trace.append((pname, gp))
            if pname == gp:
                top1_correct += 1

        order = sorted(range(len(triplets)), key=lambda i: -scored[i][0])
        for (_s, _o, p) in triplets:
            per_pred_gt[p] += 1
        pooled_gt += len(triplets)
        for k in KS:
            matched: set = set()
            hits = 0
            for ci in order[:k]:
                cs, co, _ = triplets[ci]
                pname = prior.pred_vocab[scored[ci][1]]
                for gi, (gs, go, gp) in enumerate(triplets):
                    if gi in matched:
                        continue
                    if gs == cs and go == co and gp == pname:
                        matched.add(gi); hits += 1
                        per_pred_hits[k][gp] += 1
                        break
            pooled_hits[k] += hits

    bucket = head_body_tail(prior, per_pred_gt)
    recalls = {p: per_pred_hits[50][p] / per_pred_gt[p] for p in per_pred_gt if per_pred_gt[p] > 0}
    res: Dict[str, Any] = {
        "tau": tau, "alpha": alpha,
        "R@50": pooled_hits[50] / max(1, pooled_gt),
        "R@20": pooled_hits[20] / max(1, pooled_gt),
        "mR@50": sum(recalls.values()) / max(1, len(recalls)),
        "top1_accuracy": top1_correct / max(1, n_pairs),
        "mean_entropy_nats": ent_sum / max(1, n_pairs),
        "n_classes_in_mR": len(recalls),
        "n_classes_ever_predicted": len(argmax_counts),
        "per_predicate_recall": recalls,
        "per_predicate_gt": dict(per_pred_gt),
        "argmax_counts": dict(argmax_counts),
        "buckets": bucket,
    }
    for b in ("head", "body", "tail"):
        vals = [v for p, v in recalls.items() if bucket.get(p) == b]
        res[f"{b}_mR@50"] = sum(vals) / max(1, len(vals))
    res["_argmax_trace"] = argmax_trace
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--split", default="datasets_vg150_clean/validation.jsonl")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=3.75)
    ap.add_argument("--taus", default="0.0,0.1,0.2")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    prior = FrequencyPrior.load(Path(args.prior))
    data = load_split(Path(args.split), prior, limit=int(args.limit))
    taus = [float(t) for t in str(args.taus).split(",") if t.strip()]

    # ---- 1. top-K inertness -------------------------------------------------
    cand = [len(t) for _l, t in data]
    cand.sort()
    n_img = len(cand)
    over = {k: sum(1 for c in cand if c > k) for k in KS}
    print("=" * 78)
    print("1. TOP-K INERTNESS  (GT-pairs: candidate set == GT relationship set)")
    print("=" * 78)
    print(f"  images {n_img:,}   candidates/image: mean {sum(cand)/n_img:.2f}  "
          f"median {cand[n_img//2]}  max {cand[-1]}")
    for k in KS:
        print(f"  images with more than {k:>3} candidates : {over[k]:>5,}  "
              f"({over[k]/n_img*100:.2f}%)")
    print("  => where an image has <= K candidates, top-K keeps ALL of them and the")
    print("     cross-pair ranking key cannot change R@K or mR@K.")

    rows = [run_tau(data, prior, t, args.alpha) for t in taus]
    by_tau = {r["tau"]: r for r in rows}

    # ---- validation gate ----------------------------------------------------
    gate_ok = True
    expect = {0.0: (0.6680156623656479, 0.21975828067233627),
              0.1: (0.6615772409534071, 0.2600402480414711)}
    print("\n" + "=" * 78)
    print("VALIDATION GATE vs tools/decision_rule_probe.py on this subset")
    print("=" * 78)
    for t, (er, em) in expect.items():
        if t in by_tau:
            r, m = by_tau[t]["R@50"], by_tau[t]["mR@50"]
            ok = abs(r - er) < 5e-4 and abs(m - em) < 5e-4
            gate_ok &= ok
            print(f"  tau={t}: R@50 {r*100:.2f}% (exp {er*100:.2f}%)  "
                  f"mR@50 {m*100:.2f}% (exp {em*100:.2f}%)  {'PASS' if ok else 'FAIL'}")
    print(f"  overall: {'PASS' if gate_ok else 'FAIL'}")

    # ---- 4. entropy / support ----------------------------------------------
    print("\n" + "=" * 78)
    print("4. ENTROPY AND SUPPORT")
    print("=" * 78)
    print(f"{'tau':>6}{'mean H (nats)':>16}{'top-1 acc':>12}{'classes emitted':>18}{'mR@50':>9}")
    for r in rows:
        print(f"{r['tau']:>6}{r['mean_entropy_nats']:>16.4f}{r['top1_accuracy']*100:>11.2f}%"
              f"{r['n_classes_ever_predicted']:>18}{r['mR@50']*100:>8.2f}%")
    a1 = run_tau(data, prior, 0.0, 1.0)
    a1b = run_tau(data, prior, 0.1, 1.0)
    print(f"\n  alpha control (alpha=1.0 instead of {args.alpha}):")
    print(f"    tau=0.0  H {a1['mean_entropy_nats']:.4f}  R@50 {a1['R@50']*100:.2f}%  mR@50 {a1['mR@50']*100:.2f}%")
    print(f"    tau=0.1  H {a1b['mean_entropy_nats']:.4f}  R@50 {a1b['R@50']*100:.2f}%  mR@50 {a1b['mR@50']*100:.2f}%")
    print("    (softmax is monotonic in alpha, so alpha cannot change a per-pair argmax;")
    print("     it can only change the cross-pair ranking key.)")

    # ---- 2. flip accounting -------------------------------------------------
    base_t = taus[0]
    out_flips: Dict[str, Any] = {}
    for t in taus[1:]:
        A, B = by_tau[base_t]["_argmax_trace"], by_tau[t]["_argmax_trace"]
        flips = defaultdict(int)
        w2r = r2w = w2w = 0
        gained = defaultdict(int); lost = defaultdict(int)
        for (pa, gt), (pb, _gt) in zip(A, B):
            if pa == pb:
                continue
            flips[(pa, pb)] += 1
            lost[pa] += 1; gained[pb] += 1
            if pa != gt and pb == gt: w2r += 1
            elif pa == gt and pb != gt: r2w += 1
            else: w2w += 1
        n_flip = sum(flips.values())
        print("\n" + "=" * 78)
        print(f"2. FLIP ACCOUNTING  tau={base_t} -> tau={t}")
        print("=" * 78)
        print(f"  pairs whose argmax changed : {n_flip:,} of {len(A):,} ({n_flip/len(A)*100:.2f}%)")
        print(f"    wrong -> right : {w2r:>6,}")
        print(f"    right -> wrong : {r2w:>6,}")
        print(f"    wrong -> wrong : {w2w:>6,}")
        print(f"    net top-1      : {w2r-r2w:+,}  "
              f"(pooled R@50 moves {(w2r-r2w)/len(A)*100:+.2f} points)")
        print(f"\n  largest flows (from -> to, count):")
        for (pa, pb), c in sorted(flips.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {pa:<14} -> {pb:<16}{c:>7,}")
        out_flips[str(t)] = {
            "n_flips": n_flip, "wrong_to_right": w2r, "right_to_wrong": r2w,
            "wrong_to_wrong": w2w,
            "top_flows": [{"from": a, "to": b, "n": c}
                          for (a, b), c in sorted(flips.items(), key=lambda kv: -kv[1])[:25]],
        }

        # ---- 3. leverage ---------------------------------------------------
        base, cur = by_tau[base_t], by_tau[t]
        gt = base["per_predicate_gt"]
        deltas = []
        for p in gt:
            if gt[p] <= 0:
                continue
            d = cur["per_predicate_recall"].get(p, 0.0) - base["per_predicate_recall"].get(p, 0.0)
            deltas.append((p, gt[p], base["per_predicate_recall"].get(p, 0.0),
                           cur["per_predicate_recall"].get(p, 0.0), d, base["buckets"].get(p, "?")))
        gainers = [d for d in deltas if d[4] > 0]
        losers = [d for d in deltas if d[4] < 0]
        print("\n" + "=" * 78)
        print(f"3. LEVERAGE  tau={base_t} -> tau={t}   (mR is an UNWEIGHTED class mean)")
        print("=" * 78)
        print(f"  classes improved : {len(gainers):>3}   summed dRecall {sum(d[4] for d in gainers):+.4f}")
        print(f"  classes degraded : {len(losers):>3}   summed dRecall {sum(d[4] for d in losers):+.4f}")
        print(f"  net / 50         : {sum(d[4] for d in deltas)/max(1,len(deltas))*100:+.2f} mR points")
        gn = sum(d[1] for d in gainers); ln = sum(d[1] for d in losers)
        print(f"  GT instances held by improving classes : {gn:>7,}")
        print(f"  GT instances held by degrading classes : {ln:>7,}")
        if gn and ln:
            print(f"  mean class size, improving : {gn/len(gainers):>9.1f}")
            print(f"  mean class size, degrading : {ln/len(losers):>9.1f}")
            print(f"  LEVERAGE RATIO (degrading/improving mean size) : "
                  f"{(ln/len(losers))/(gn/len(gainers)):.1f}x")
        out_flips[str(t)]["leverage"] = {
            "n_classes_improved": len(gainers), "n_classes_degraded": len(losers),
            "sum_d_recall_improved": sum(d[4] for d in gainers),
            "sum_d_recall_degraded": sum(d[4] for d in losers),
            "gt_instances_improving_classes": gn, "gt_instances_degrading_classes": ln,
        }

        print(f"\n  per-predicate, tau={base_t} -> tau={t}  (sorted by delta)")
        print(f"  {'predicate':<16}{'bucket':>7}{'n':>8}{'raw':>9}{'calib':>9}{'delta':>9}")
        for p, n, a, b, d, bk in sorted(deltas, key=lambda x: -x[4]):
            flag = "  <- n<25, noisy" if n < 25 else ""
            print(f"  {p:<16}{bk:>7}{n:>8,}{a*100:>8.1f}%{b*100:>8.1f}%{d*100:>+8.1f}{flag}")
        out_flips[str(t)]["per_predicate"] = [
            {"predicate": p, "bucket": bk, "n": n, "raw_recall": a,
             "calibrated_recall": b, "delta": d}
            for p, n, a, b, d, bk in sorted(deltas, key=lambda x: -x[4])]

    result = {
        "validation_gate_passes": bool(gate_ok),
        "n_images": n_img, "alpha": args.alpha,
        "candidates_per_image": {"mean": sum(cand)/n_img, "median": cand[n_img//2],
                                 "max": cand[-1],
                                 "images_over_K": {str(k): over[k] for k in KS},
                                 "frac_images_over_K": {str(k): over[k]/n_img for k in KS}},
        "taus": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "alpha_control": {"alpha": 1.0,
                          "tau0": {k: a1[k] for k in ("R@50", "mR@50", "mean_entropy_nats")},
                          "tau01": {k: a1b[k] for k in ("R@50", "mR@50", "mean_entropy_nats")}},
        "flips": out_flips,
    }
    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[forensics] written to {p}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
