#!/usr/bin/env python
"""C7-extended -- prior-override diagnostics, named per the directive.

Extends p43 (prior_adversarial.py) with the specific named quantities and
stratifications a benchmark reader needs, rather than only fixed/kept counts:

  Prior Override Rate      : on ADVERSARIAL rows, P(model argmax != prior argmax)
                              -- did the model even try to move off the prior?
  Successful Override Rate : on ADVERSARIAL rows, P(model argmax == GT)
                              -- p43's "fixed" rate, renamed to match this run
  Wrong Override Rate      : on ADVERSARIAL rows, P(model argmax != prior argmax
                              AND model argmax != GT) -- moved, but to the wrong
                              place. Override Rate = Successful + Wrong (+ ties).
  Tail Override Rate       : Successful Override Rate restricted to GT in the
                              tail predicate bucket.

Also reports, per row, joined against:
  prior margin    : prior score gap between its top-1 and top-2 class
  model margin    : model-term score gap between its top-1 and top-2 class
  confidence bin  : quartile bin of the prior's top-1 probability
  predicate freq  : train-count bucket of the GT predicate (head/body/tail)
  prior-correct vs prior-adversarial rows (p43's population split, reused)

Population and prior are IDENTICAL to p43: train-derived, so
argmax_p prior[row] is a pure function of (subject, object) and the training
split -- no validation label is touched to define the population.

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


CPA = _load("cprime_analysis")
MECH = _load("cprime_mechanism")
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST


def _log(m: str = "") -> None:
    print(m, flush=True)


def top1_top2_margin(s: torch.Tensor) -> torch.Tensor:
    """Per-row gap between the top-1 and top-2 column scores."""
    top2 = torch.topk(s, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p63_prior_override_diagnostics/diag.json")
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args(argv)

    _log("=" * 104)
    _log("C7-EXTENDED PRIOR-OVERRIDE DIAGNOSTICS -- CPU only, cache read-only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4
    y = B.gt_y
    n = int(y.numel())

    prior_score = B.score(args.tau, ALPHA_HIST, None)
    model_score = B.score(args.tau, ALPHA_HIST, B.model)
    prior_pred = B.predict(prior_score)
    model_pred = B.predict(model_score)

    adv = prior_pred != y      # prior-adversarial: following the prior is WRONG
    ok = ~adv
    na, no = int(adv.sum()), int(ok.sum())
    _log(f"\npopulation: {na:,} prior-adversarial ({na/n*100:.1f}%), "
         f"{no:,} prior-correct ({no/n*100:.1f}%)   tau={args.tau}")

    overridden = model_pred != prior_pred           # model moved off the prior's choice
    successful = model_pred == y                    # model landed on GT
    wrong_override = overridden & ~successful        # moved, but not to GT

    por = float(overridden[adv].float().mean()) if na else 0.0
    sor = float(successful[adv].float().mean()) if na else 0.0
    wor = float(wrong_override[adv].float().mean()) if na else 0.0

    _log(f"\n  Prior Override Rate      (adv, moved off prior)      : {por*100:6.2f}%")
    _log(f"  Successful Override Rate (adv, landed on GT)          : {sor*100:6.2f}%")
    _log(f"  Wrong Override Rate      (adv, moved but missed GT)   : {wor*100:6.2f}%")
    _log(f"  (sanity: moved-and-hit + moved-and-missed = override) "
         f"{sor + wor:.4f} vs {por:.4f}")

    tail_mask = adv & torch.tensor([B.bucket_of[int(c)] == "tail" for c in y])
    ntail = int(tail_mask.sum())
    tor = float(successful[tail_mask].float().mean()) if ntail else float("nan")
    _log(f"\n  Tail Override Rate (successful, GT in tail bucket, n={ntail}) : "
         f"{tor*100:6.2f}%")

    # ---- per-bucket successful/wrong/prior override, head/body/tail ----
    by_bucket: Dict[str, Any] = {}
    for buck in ("head", "body", "tail"):
        m = adv & torch.tensor([B.bucket_of[int(c)] == buck for c in y])
        nb = int(m.sum())
        by_bucket[buck] = {
            "n_adversarial": nb,
            "prior_override_rate": float(overridden[m].float().mean()) if nb else float("nan"),
            "successful_override_rate": float(successful[m].float().mean()) if nb else float("nan"),
            "wrong_override_rate": float(wrong_override[m].float().mean()) if nb else float("nan"),
        }
        _log(f"    {buck:>5}  n={nb:>6,}  override={by_bucket[buck]['prior_override_rate']*100:6.2f}%  "
             f"success={by_bucket[buck]['successful_override_rate']*100:6.2f}%  "
             f"wrong={by_bucket[buck]['wrong_override_rate']*100:6.2f}%")

    # ---- margins ----
    pm = top1_top2_margin(prior_score[B.gt_row])
    mm = top1_top2_margin(model_score[B.gt_row])
    _log(f"\n  prior margin  mean={float(pm.mean()):.4f}  "
         f"adv={float(pm[adv].mean()):.4f}  ok={float(pm[ok].mean()):.4f}")
    _log(f"  model margin  mean={float(mm.mean()):.4f}  "
         f"adv={float(mm[adv].mean()):.4f}  ok={float(mm[ok].mean()):.4f}")

    # ---- confidence bins (quartiles of the prior's top-1 probability) ----
    prior_top1_score = prior_score[B.gt_row].max(-1).values
    q = torch.quantile(prior_top1_score, torch.tensor([0.25, 0.5, 0.75]))
    conf_bin = torch.bucketize(prior_top1_score, q)  # 0..3, low..high confidence
    by_confbin: Dict[str, Any] = {}
    for cb in range(4):
        m = adv & (conf_bin == cb)
        nb = int(m.sum())
        by_confbin[str(cb)] = {
            "n_adversarial": nb,
            "successful_override_rate": float(successful[m].float().mean()) if nb else float("nan"),
            "prior_override_rate": float(overridden[m].float().mean()) if nb else float("nan"),
        }
        _log(f"    confidence bin {cb} (q{cb*25}-{(cb+1)*25}%)  n={nb:>6,}  "
             f"success={by_confbin[str(cb)]['successful_override_rate']*100:6.2f}%")

    # ---- predicate-frequency-bucket cross of correct-prior vs wrong-prior ----
    by_predicate: List[Dict[str, Any]] = []
    for c in range(B.n_classes):
        m_adv = adv & (y == c)
        m_ok = ok & (y == c)
        na_c, no_c = int(m_adv.sum()), int(m_ok.sum())
        if na_c + no_c == 0:
            continue
        by_predicate.append({
            "class": B.classes[c], "bucket": B.bucket_of[c],
            "n_adversarial": na_c, "n_prior_correct": no_c,
            "successful_override_rate": float(successful[m_adv].float().mean()) if na_c else float("nan"),
        })

    res = {
        "tool": "prior_override_diagnostics", "tau": args.tau,
        "n_total": n, "n_adversarial": na, "n_prior_correct": no,
        "headline": {
            "prior_override_rate": por, "successful_override_rate": sor,
            "wrong_override_rate": wor, "tail_override_rate": tor, "n_tail_adversarial": ntail,
        },
        "by_bucket": by_bucket,
        "margins": {"prior_mean": float(pm.mean()), "prior_adv": float(pm[adv].mean()),
                   "prior_ok": float(pm[ok].mean()), "model_mean": float(mm.mean()),
                   "model_adv": float(mm[adv].mean()), "model_ok": float(mm[ok].mean())},
        "by_confidence_bin": by_confbin,
        "by_predicate": by_predicate,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
