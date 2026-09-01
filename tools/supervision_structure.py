#!/usr/bin/env python
"""How much WITHIN-PAIR supervision does VG150 actually contain?

runs/p48 found that only ~19% of TRAIN (subject,object) groups hold >=2 distinct
predicates, and that an objective restricted to those groups underperforms plain
cross-entropy. That raises a dataset-level question this tool answers exactly,
on all three splits:

    a model can only learn to distinguish predicate a from predicate b WITHIN a
    fixed (s,o) group if the training data contains rows of BOTH a and b for the
    SAME (s,o) group. How many such rows exist, and for which predicate pairs?

Reported per split:
  * groups, rows, singleton fraction
  * P(group has >=2 rows), P(>=2 distinct predicates), P(>=3 distinct)
  * rows living in decidable groups
  * within-group predicate entropy
  * CONTRASTIVE PAIR COUNT per bucket combination -- sum over (group, {a,b})
    cells of n_a * n_b. This is the number of within-pair training comparisons
    available, and it is the quantity that determines whether within-pair
    discrimination is learnable at all for a given predicate pair type.

NO CAUSAL CLAIM is made here. This measures what supervision exists; it does not
establish that its scarcity causes any model's behaviour.

CPU only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


MECH = _load("cprime_mechanism")
Mech = MECH.Mech


def _log(m: str = "") -> None:
    print(m, flush=True)


def read_split(path: str, classes: List[str]):
    """(group key -> Counter(predicate)) for one split."""
    idx = {c: i for i, c in enumerate(classes)}
    groups: Dict[str, Counter] = defaultdict(Counter)
    n_rows = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            names = [str((o.get("names") or [""])[0]).strip().lower()
                     for o in (d.get("objects") or [])]
            for r in (d.get("relationships") or []):
                c = idx.get(str(r.get("predicate", "")).strip().lower())
                a, b = int(r.get("subject_id", -1)), int(r.get("object_id", -1))
                if c is None or not (0 <= a < len(names) and 0 <= b < len(names)):
                    continue
                groups[f"{names[a]}||{names[b]}"][c] += 1
                n_rows += 1
    return groups, n_rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--root", default="datasets_vg150_clean")
    ap.add_argument("--out", default="runs/p50_supervision_structure/sup.json")
    args = ap.parse_args(argv)

    _log("=" * 106)
    _log("VG150 WITHIN-PAIR SUPERVISION STRUCTURE -- CPU only, NO GPU")
    _log("=" * 106)
    B = Mech(args.dump, args.prior, "raw50")
    classes = list(B.classes)
    bucket = {c: B.bucket_of[c] for c in range(B.n_classes)}
    _log(f"  predicate vocabulary {len(classes)}  "
         f"head {sum(1 for c in bucket.values() if c=='head')} / "
         f"body {sum(1 for c in bucket.values() if c=='body')} / "
         f"tail {sum(1 for c in bucket.values() if c=='tail')}")

    res: Dict[str, Any] = {"tool": "supervision_structure", "splits": {}}
    for split in ("train", "validation", "test"):
        path = str(Path(args.root) / f"{split}.jsonl")
        if not Path(path).exists():
            _log(f"  [skip] {path} absent")
            continue
        G, n_rows = read_split(path, classes)
        n_groups = len(G)
        sizes = {k: sum(v.values()) for k, v in G.items()}
        ndist = {k: len(v) for k, v in G.items()}
        singleton_groups = sum(1 for k in G if sizes[k] == 1)
        rows_singleton = singleton_groups
        ge2rows = sum(1 for k in G if sizes[k] >= 2)
        ge2pred = sum(1 for k in G if ndist[k] >= 2)
        ge3pred = sum(1 for k in G if ndist[k] >= 3)
        rows_decidable = sum(sizes[k] for k in G if ndist[k] >= 2)
        ent = []
        for k, v in G.items():
            n = sizes[k]
            if n > 1:
                ent.append(-sum((c / n) * math.log(c / n) for c in v.values()))
        # contrastive pair counts per bucket combination
        cells = Counter()
        pairs = Counter()
        for k, v in G.items():
            cs = sorted(v)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    key = "-".join(sorted([bucket[cs[i]], bucket[cs[j]]]))
                    cells[key] += 1
                    pairs[key] += v[cs[i]] * v[cs[j]]
        s = {
            "n_rows": n_rows, "n_groups": n_groups,
            "singleton_group_frac": singleton_groups / n_groups,
            "rows_in_singleton_groups_frac": rows_singleton / n_rows,
            "P_group_ge2_rows": ge2rows / n_groups,
            "P_group_ge2_distinct_predicates": ge2pred / n_groups,
            "P_group_ge3_distinct_predicates": ge3pred / n_groups,
            "rows_in_decidable_groups_frac": rows_decidable / n_rows,
            "mean_group_size": n_rows / n_groups,
            "mean_within_group_entropy_multirow": (sum(ent) / len(ent)) if ent else 0.0,
            "contrastive_cells": dict(cells),
            "contrastive_pairs": dict(pairs),
        }
        res["splits"][split] = s
        _log(f"\n{'-'*106}")
        _log(f"  {split.upper()}   rows {n_rows:,}   groups {n_groups:,}   "
             f"mean group size {n_rows/n_groups:.2f}")
        _log(f"{'-'*106}")
        _log(f"    singleton GROUPS                    {singleton_groups:>9,} "
             f"({singleton_groups/n_groups*100:>5.1f}% of groups)")
        _log(f"    rows in singleton groups            {rows_singleton:>9,} "
             f"({rows_singleton/n_rows*100:>5.1f}% of rows)")
        _log(f"    P(group has >=2 rows)                        {ge2rows/n_groups*100:>5.1f}%")
        _log(f"    P(group has >=2 DISTINCT predicates)         {ge2pred/n_groups*100:>5.1f}%"
             f"   <- the only groups that teach within-pair discrimination")
        _log(f"    P(group has >=3 DISTINCT predicates)         {ge3pred/n_groups*100:>5.1f}%")
        _log(f"    rows living in decidable groups              {rows_decidable/n_rows*100:>5.1f}%")
        _log(f"    mean within-group entropy (multi-row)        "
             f"{(sum(ent)/len(ent)) if ent else 0:.4f} nats")
        _log(f"\n    WITHIN-PAIR CONTRASTIVE SUPPLY by predicate-bucket pair")
        _log(f"      {'buckets':>12} {'cells':>10} {'comparisons':>14} {'% of all':>9}")
        tot = sum(pairs.values()) or 1
        for key in ("head-head", "body-head", "head-tail", "body-body",
                    "body-tail", "tail-tail"):
            lab = key if key in pairs else key
            _log(f"      {lab:>12} {cells.get(key,0):>10,} {pairs.get(key,0):>14,} "
                 f"{pairs.get(key,0)/tot*100:>8.2f}%")

    # train vs val/test comparability
    if "train" in res["splits"] and "validation" in res["splits"]:
        t, v = res["splits"]["train"], res["splits"]["validation"]
        _log(f"\n{'-'*106}\n  TRAIN vs VALIDATION comparability\n{'-'*106}")
        for k in ("P_group_ge2_distinct_predicates", "rows_in_decidable_groups_frac",
                  "mean_within_group_entropy_multirow", "mean_group_size"):
            _log(f"    {k:>42}  train {t[k]:>8.4f}   val {v[k]:>8.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
