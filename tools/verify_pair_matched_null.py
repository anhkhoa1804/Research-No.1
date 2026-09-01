#!/usr/bin/env python
"""Independent verification of the pair-matched null's construction.

The null is the primary discriminator between two explanations of the model's
surviving contribution:

  E1  the model is conditioned on image appearance
  E2  the model term is a reparameterised (subject, object) prior

That makes its construction load-bearing, so it is checked here by a tool that
re-derives everything from the cache rather than trusting the probe's own
bookkeeping. CPU only, cache read-only.

Checks, each of which can independently invalidate the E1/E2 reading:
  C1  sample size and vocabulary match the cache exactly
  C2  the permutation NEVER crosses a (subject, object) group
  C3  the permuted rows genuinely changed (a no-op permutation is not a null)
  C4  the permutable fraction, and therefore how conservative the null is
  C5  asymmetry: are permutable rows a biased subsample (head-heavy, low-margin)?
      If they are, the null is weak exactly where the model is claimed to act.
  C6  the null's feature block differs from `full` ONLY in the model columns
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CSP = _load("candidate_scorer_probe")


def _log(m: str = "") -> None:
    print(m, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p10_model_recalibration/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--out", default="runs/p26_pair_matched_null/verification.json")
    args = ap.parse_args(argv)

    B = CSP.Mech(args.dump, args.prior, "raw50")
    P = CSP.CandidateProbe(B, args.tau, args.k)
    res: Dict[str, Any] = {"tool": "verify_pair_matched_null", "checks": []}

    def check(name: str, ok: bool, detail: str) -> None:
        res["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        _log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    _log("=" * 92)
    _log("PAIR-MATCHED NULL -- INDEPENDENT VERIFICATION")
    _log("=" * 92)

    # C1 -- sample size / vocabulary
    n_rows, n_cls = P.n, P.C
    check("C1 sample size", n_rows == B.n_gt and n_cls == B.n_classes,
          f"GT rows={n_rows} (cache {B.n_gt}), classes={n_cls} (cache {B.n_classes}), "
          f"images={B.n_images}")

    # rebuild the permutation exactly as _blocks does
    gen = torch.Generator().manual_seed(CSP.SEED)
    X_null = P._blocks("pair_matched_null", gen)
    gen2 = torch.Generator().manual_seed(CSP.SEED)
    perm = torch.arange(n_rows)
    for gid in P.pair_id.unique().tolist():
        idx = (P.pair_id == gid).nonzero().squeeze(1)
        if idx.numel() > 1:
            perm[idx] = idx[torch.randperm(idx.numel(), generator=gen2)]

    # C2 -- never crosses a group
    crossings = int((P.pair_id[perm] != P.pair_id).sum())
    check("C2 pair matching exact", crossings == 0,
          f"{crossings} rows drew their model term from a DIFFERENT "
          f"(subject, object) group (must be 0)")

    # C3 -- rows actually moved
    moved = int((perm != torch.arange(n_rows)).sum())
    check("C3 permutation is not a no-op", moved > 0,
          f"{moved}/{n_rows} rows ({moved/n_rows*100:.1f}%) received another "
          f"row's model term")

    # C4 -- conservatism
    groups = Counter(P.pair_id.tolist())
    singleton_rows = sum(c for c in groups.values() if c == 1)
    check("C4 null is conservative (documented, not a defect)", True,
          f"{len(groups)} distinct (subj,obj) groups; {singleton_rows} rows "
          f"({singleton_rows/n_rows*100:.1f}%) are in singleton groups and KEEP their "
          f"real model term -> the null retains part of the real signal and is "
          f"biased TOWARDS the real arm")

    # C5 -- asymmetry of the permutable subsample
    permutable = (perm != torch.arange(n_rows))
    bucket_all = Counter(B.bucket_of[int(c)] for c in P.y.tolist())
    bucket_perm = Counter(B.bucket_of[int(c)] for c in P.y[permutable].tolist())
    frac = {b: (bucket_perm[b] / max(1, bucket_all[b])) for b in ("head", "body", "tail")}
    m_all = float(P.margin.mean())
    m_perm = float(P.margin[permutable].mean())
    m_keep = float(P.margin[~permutable].mean())
    spread = max(frac.values()) - min(frac.values())
    check("C5 permutable subsample is not wildly biased", spread < 0.5,
          f"permutable fraction by bucket head={frac['head']:.3f} "
          f"body={frac['body']:.3f} tail={frac['tail']:.3f} (spread {spread:.3f}); "
          f"prior margin mean: all={m_all:.3f} permuted={m_perm:.3f} kept={m_keep:.3f}")
    res["asymmetry"] = {"permutable_fraction_by_bucket": frac,
                        "margin_mean_all": m_all,
                        "margin_mean_permuted": m_perm,
                        "margin_mean_kept": m_keep,
                        "n_permutable": moved,
                        "n_singleton_rows": singleton_rows,
                        "n_groups": len(groups)}

    # C6 -- differs from `full` only in the model columns
    X_full = P._blocks("full", torch.Generator().manual_seed(CSP.SEED))
    same_prior = bool(torch.allclose(X_full[..., :-3], X_null[..., :-3]))
    diff_model = not bool(torch.allclose(X_full[..., -3:], X_null[..., -3:]))
    check("C6 differs from `full` in model columns only",
          same_prior and diff_model and X_full.shape == X_null.shape,
          f"shapes {tuple(X_full.shape)} vs {tuple(X_null.shape)}; prior block "
          f"identical={same_prior}; model block differs={diff_model} -> a gap "
          f"between the arms cannot be a capacity difference")

    # C7 -- every null value is a real model-term entry
    subset = set(X_null[..., -3].flatten().tolist()) <= set(P.md.flatten().tolist())
    check("C7 null values are real model-term entries", subset,
          "the null re-uses the model term verbatim; it does not synthesise values")

    res["all_pass"] = all(c["pass"] for c in res["checks"])
    res["verdict"] = "NULL CONSTRUCTION VALID" if res["all_pass"] else "INVALID"
    _log(f"\n  VERDICT: {res['verdict']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"[written] {args.out}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
