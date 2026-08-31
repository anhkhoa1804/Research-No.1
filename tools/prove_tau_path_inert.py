#!/usr/bin/env python
"""PROOF (CPU, no checkpoint): --eval_logit_adj_tau is a no-op at ensemble_alpha=0.

Runs openvocab_rel.evals._relation_predicate_logits -- the real function, not a
reimplementation -- against stub model objects, and compares its output with and
without the tau flag under the historical protocol's configuration.

THE PATH, read at HEAD:

  evals.py:1256   cls_logits = _apply_eval_logit_adjustment(cfg, cls_logits, pred_log_prior)
  evals.py:1259   cls_norm   = _normalize_eval_logits(cls_logits)
  evals.py:1272   alpha      = max(0.0, min(1.0, cfg.eval_sgg_predicate_ensemble_alpha))
  evals.py:1273   return (alpha * cls_norm) + ((1.0 - alpha) * text_norm)

`_apply_eval_logit_adjustment` only ever touches `cls_logits`. The historical
protocol sets eval_sgg_predicate_ensemble_alpha = 0.0
(runs/p5_model_vs_leakfree_prior/, canary PASS), so line 1273 multiplies the
adjusted tensor by exactly zero. Independently, evals.py:1075-1077 falls back
from eval_logit_adj_tau = -1.0 to logit_adj_tau = 0.0 and returns early.

Two independent reasons the flag does nothing. This script demonstrates both.

Usage:
    python tools/prove_tau_path_inert.py --out runs/p8_tau_path_bug/proof.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openvocab_rel.evals import (  # noqa: E402
    _apply_eval_logit_adjustment,
    _relation_predicate_logits,
)

P = 51          # runtime vocabulary: 50 predicates + 1 synthetic background
N = 7           # pairs
D = 16          # feature width


class StubOut:
    """Minimal stand-in for the model. Deterministic, no checkpoint needed."""

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self._text = torch.randn(N, P, generator=g)
        self._cls = torch.randn(N, P, generator=g) * 3.0 + 1.0

    def text_predicate_logits(self, rel_feat, pred_emb):
        return self._text

    def predicate_logits(self, rel_feat):
        return self._cls

    def calibrated_predicate_logits(self, rel_feat, pred_log_prior):
        return self._cls


class Cfg:
    def __init__(self, **kw):
        self.eval_sgg_predicate_score_mode = "ensemble"
        self.eval_sgg_use_predicate_classifier = True
        self.eval_sgg_predicate_ensemble_alpha = 0.0
        self.adaptive_calibration_enabled = True
        self.eval_sgg_classifier_temperature = 1.0
        self.eval_sgg_text_temperature = 1.0
        self.logit_adj_tau = 0.0
        self.eval_logit_adj_tau = -1.0
        for k, v in kw.items():
            setattr(self, k, v)


def run(cfg, out, prior) -> torch.Tensor:
    rel_feat = torch.zeros(N, D)
    pred_emb = torch.zeros(P, D)
    return _relation_predicate_logits(cfg, out, rel_feat, pred_emb, prior)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    out = StubOut()
    g = torch.Generator().manual_seed(1)
    prior = torch.log(torch.rand(P, generator=g).clamp_min(1e-6))

    checks: List[Dict[str, Any]] = []

    def check(name: str, a: torch.Tensor, b: torch.Tensor, expect_equal: bool, note: str):
        same = bool(torch.equal(a, b))
        maxdiff = float((a - b).abs().max()) if a.shape == b.shape else float("nan")
        ok = (same == expect_equal)
        checks.append({"check": name, "identical": same, "max_abs_diff": maxdiff,
                       "expected_identical": expect_equal, "pass": ok, "note": note})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         identical={same}  max|diff|={maxdiff:.3e}  (expected identical={expect_equal})")
        print(f"         {note}")
        return ok

    print("=" * 82)
    print("1. HISTORICAL PROTOCOL (ensemble_alpha=0.0) -- does --eval_logit_adj_tau do anything?")
    print("=" * 82)
    base = run(Cfg(), out, prior)                                   # no tau flag
    tau01 = run(Cfg(eval_logit_adj_tau=0.1), out, prior)            # --eval_logit_adj_tau 0.1
    tau10 = run(Cfg(eval_logit_adj_tau=1.0), out, prior)            # an extreme tau
    ok = True
    ok &= check("alpha=0.0: tau=0.1 vs no flag", base, tau01, True,
                "evals.py:1273 multiplies the tau-adjusted cls_norm by alpha=0.0")
    ok &= check("alpha=0.0: tau=1.0 vs no flag", base, tau10, True,
                "even an extreme tau is annihilated by the same multiplication")

    print("\n" + "=" * 82)
    print("2. THE SECOND, INDEPENDENT REASON: the -1.0 -> 0.0 fallback early-returns")
    print("=" * 82)
    cls = out.predicate_logits(None)
    a = _apply_eval_logit_adjustment(Cfg(), cls, prior)
    ok &= check("eval_logit_adj_tau=-1.0 falls back to logit_adj_tau=0.0", cls, a, True,
                "evals.py:1075-1077: tau<=0.0 returns logits unchanged")

    print("\n" + "=" * 82)
    print("3. CONTROL: the adjustment IS live when alpha>0 -- so the code is not dead,")
    print("   it is unreachable *under this protocol*")
    print("=" * 82)
    b0 = run(Cfg(eval_sgg_predicate_ensemble_alpha=0.5), out, prior)
    b1 = run(Cfg(eval_sgg_predicate_ensemble_alpha=0.5, eval_logit_adj_tau=0.1), out, prior)
    ok &= check("alpha=0.5: tau=0.1 vs no flag", b0, b1, False,
                "with alpha>0 the adjusted classifier logits reach the output")
    c0 = run(Cfg(eval_sgg_predicate_score_mode="classifier"), out, prior)
    c1 = run(Cfg(eval_sgg_predicate_score_mode="classifier", eval_logit_adj_tau=0.1), out, prior)
    ok &= check("score_mode=classifier: tau=0.1 vs no flag", c0, c1, False,
                "the classifier-only path returns cls_logits directly (evals.py:1257-1258)")

    print("\n" + "=" * 82)
    print("4. WHAT C' ACTUALLY NEEDS: tau on the FREQUENCY-PRIOR term, which this")
    print("   function never touches")
    print("=" * 82)
    print("   _relation_predicate_logits returns the MODEL term only. The prior is added")
    print("   afterwards by _apply_frequency_bias (evals.py:1178-1193), and no tau of any")
    print("   kind is applied there at HEAD. That is the pathway C' requires.")

    verdict = ("CONFIRMED: --eval_logit_adj_tau cannot affect the historical protocol's output"
               if ok else "UNEXPECTED: at least one check failed -- do not rely on this analysis")
    print("\n" + "=" * 82)
    print(f"VERDICT: {verdict}")
    print("=" * 82)

    res = {"verdict": verdict, "all_checks_pass": bool(ok), "checks": checks,
           "protocol": {"eval_sgg_predicate_score_mode": "ensemble",
                        "eval_sgg_predicate_ensemble_alpha": 0.0,
                        "adaptive_calibration_enabled": True,
                        "source": "runs/p5_model_vs_leakfree_prior/latest_metrics.json"},
           "source_path": ["evals.py:1074-1080 _apply_eval_logit_adjustment",
                           "evals.py:1256 call site (cls_logits only)",
                           "evals.py:1272-1273 alpha=0.0 annihilates cls_norm",
                           "evals.py:1178-1193 _apply_frequency_bias (no tau at HEAD)"]}
    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[proof] written to {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
