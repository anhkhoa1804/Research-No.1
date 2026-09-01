#!/usr/bin/env python
"""C' oracle ceiling -- how much is in the tie-breaker channel AT ALL?

Reads runs/p10_model_recalibration/pair_logits.pt read-only. CPU only. No GPU,
no training, no modification of any C' artifact.

WHY THIS AND NOT A RERANKER
---------------------------
docs/CPRIME_MECHANISM_REPORT.md measures that the model's influence is confined
to rows where the prior's top1-top2 margin is small (0.00% of rows above the
5th margin decile change their argmax) and that restricting it to the prior's
top-3 is strictly BETTER than letting it act globally. The open question is
therefore not "can a better reranker be built" but "how much is in that channel
at all". An oracle bounds every possible tie-breaker at this alpha, so it
answers that before any GPU time is spent.

This is NOT the same measurement as runs/p7_tau_vs_oracle_headroom/, which is
mR-primary, uses top-5 only and applies no margin budget. This one is R-primary
with an explicit R@50 floor and an explicit reachability budget, because the
mechanism report shows the mR axis is carried by two predicates and is not a
sound primary criterion here.

THE LADDER
----------
Four arms at each (k, budget, tau), on the SAME rows and the SAME denominator:

  prior          the prior-only argmax                       -- the baseline
  combined       prior + the real model term at alpha=3.75   -- what we ACHIEVE
  model_rerank   argmax of the MODEL score inside the prior's top-k, on rows
                 inside the budget                           -- REALIZABLE, and
                 needs no calibration: it is the alpha -> inf limit of the
                 restricted arm, i.e. the best the CURRENT representation can
                 do if composition were solved
  oracle         GT itself whenever GT is in the prior's top-k, on rows inside
                 the budget                                  -- UPPER BOUND on
                 every possible tie-breaker in this channel

The three gaps are what decide the next experiment:
  oracle - model_rerank  -> headroom that needs BETTER FEATURES (GPU)
  model_rerank - combined-> headroom that needs BETTER COMPOSITION (CPU)
  combined - prior       -> what C' already banked

BUDGET
------
A bounded scorer can only move an argmax on a row whose prior margin is below
the score differential it can supply. The budgets are read off the measured
margin distribution at tau=0 (mechanism report section 3): decile-1 boundary
0.93 and the median 5.08, plus unrestricted. Rows outside the budget keep the
prior's top-1 in every arm, including the oracle -- that is what makes the
oracle a ceiling for a TIE-BREAKER rather than a ceiling for omniscience.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


MECH = _load("cprime_mechanism")
CPA = sys.modules["cprime_analysis"]
Mech, ALPHA_HIST, TAUS = MECH.Mech, MECH.ALPHA_HIST, MECH.TAUS

KS = [2, 3, 5]
BUDGETS = [("unrestricted", float("inf")), ("median_5.08", 5.08), ("decile1_0.93", 0.93)]
R_FLOOR = 0.665          # the R@50 floor the C' criterion lacked
SUCCESS_PTS = 2.0
INCONCLUSIVE_PTS = 1.5


def _log(m: str = "") -> None:
    print(m, flush=True)


class Oracle:
    """All four arms share one row set, one denominator and one argmax helper."""

    def __init__(self, B: Mech, tau: float):
        self.B, self.tau = B, tau
        self.pr = B.rows(B.score(tau, ALPHA_HIST, None))        # prior term, GT rows
        self.md = B.rows(B.score(tau, 0.0, B.model))            # model term, GT rows
        self.cb = self.pr + self.md
        self.y = B.gt_y
        self.gt_col = B.gt_col_of()
        t2 = self.pr.topk(2, dim=-1)
        self.margin = t2.values[:, 0] - t2.values[:, 1]
        # argmax, NOT topk(2).indices[:, 0]: 522 GT rows tie for the prior's
        # maximum and the two calls break that tie differently. topk there
        # yields prior R@50 66.7543 against the evaluator's -- and C's --
        # 66.8016, which would have shifted every baseline, every achieved dR
        # and therefore every oracle gap in this table. `margin` comes from the
        # topk VALUES and is tie-order invariant.
        self.prior_top1_col = self.pr.argmax(-1)

    def _metrics(self, pred_col: torch.Tensor) -> Dict[str, float]:
        pred = self.B.col_to_class[pred_col]
        hit = (pred == self.y).float()
        K = self.B.n_classes
        num = torch.zeros(K).index_add_(0, self.y, hit)
        den = torch.zeros(K).index_add_(0, self.y, torch.ones_like(hit))
        pres = den > 0
        rec = num[pres] / den[pres]
        out = {"R": float(hit.mean()),
               "mR": float(rec.mean()),
               "n_classes": int(pres.sum())}
        # head/body/tail on the SAME bucket definition Bench uses, so these are
        # comparable with every other C' table. Buckets come from GT counts and
        # GT is fixed across arms, so cross-arm comparison is sound.
        cls_ids = pres.nonzero().squeeze(1).tolist()
        per = {c: float(num[c] / den[c]) for c in cls_ids}
        for b in ("head", "body", "tail"):
            ks = [c for c in cls_ids if self.B.bucket_of[c] == b]
            out[f"{b}_mR"] = float(sum(per[c] for c in ks) / len(ks)) if ks else 0.0
        out["_per_class_recall"] = per
        return out

    def coverage(self, k: int, budget: float) -> Dict[str, float]:
        """P(GT in the prior top-k). This is what the oracle arm actually
        measures, and it is NOT evidence about any realizable scorer."""
        cand = self.candidates(k)
        gt_in = cand.gather(1, self.gt_col.unsqueeze(1)).squeeze(1)
        el = self.eligible(budget)
        return {"coverage_all_rows": float(gt_in.float().mean()),
                "coverage_eligible_rows": float(gt_in[el].float().mean()) if int(el.sum()) else 0.0,
                "n_eligible": int(el.sum())}

    def eligible(self, budget: float) -> torch.Tensor:
        return self.margin < budget

    def candidates(self, k: int) -> torch.Tensor:
        """Boolean mask of the prior's top-k columns, per row."""
        m = torch.zeros_like(self.pr, dtype=torch.bool)
        m.scatter_(1, self.pr.topk(min(k, self.pr.shape[1]), dim=-1).indices, True)
        return m

    def arm(self, kind: str, k: int, budget: float) -> Dict[str, Any]:
        el = self.eligible(budget)
        cand = self.candidates(k)
        base = self.prior_top1_col.clone()
        if kind == "prior":
            pred = base
        elif kind == "combined":
            pred = torch.where(el, self.cb.argmax(-1), base)
        elif kind == "model_rerank":
            masked = torch.where(cand, self.md, torch.full_like(self.md, -1e30))
            pred = torch.where(el, masked.argmax(-1), base)
        elif kind == "oracle":
            gt_in = cand.gather(1, self.gt_col.unsqueeze(1)).squeeze(1)
            pred = torch.where(el & gt_in, self.gt_col, base)
        else:
            raise ValueError(kind)
        out = self._metrics(pred)
        out.update({"arm": kind, "k": k, "budget": budget,
                    "n_eligible": int(el.sum()),
                    "frac_eligible": float(el.float().mean())})
        return out


def region_decomposition(B: Mech, tau: float) -> Dict[str, Any]:
    """Where do the model's ACTUAL net flips live, in (k, budget) space?

    The tie-breaker hypothesis predicts they are all inside the prior's top few
    candidates and inside the margin budget. This measures it directly instead
    of inferring it.
    """
    O = Oracle(B, tau)
    pred_p = B.col_to_class[O.prior_top1_col]
    pred_c = B.col_to_class[O.cb.argmax(-1)]
    okp, okc = (pred_p == O.y), (pred_c == O.y)
    resc, dest = (~okp) & okc, okp & (~okc)
    net_total = int(resc.sum()) - int(dest.sum())
    rows = []
    for k in KS + [B.n_classes]:
        cand = O.candidates(k)
        landed_in = cand.gather(1, O.cb.argmax(-1).unsqueeze(1)).squeeze(1)
        for name, b in BUDGETS:
            el = O.eligible(b)
            m = el & landed_in
            rows.append({"k": k, "budget": name,
                         "net_flips_inside": int((resc & m).sum()) - int((dest & m).sum()),
                         "rescued_inside": int((resc & m).sum()),
                         "destroyed_inside": int((dest & m).sum()),
                         "frac_of_total_net": (int((resc & m).sum()) - int((dest & m).sum())) / net_total
                         if net_total else None})
    return {"tau": tau, "net_flips_total": net_total, "regions": rows}


def tau_independence(B: Mech) -> Dict[str, Any]:
    """Falsification: is the model 'restoring recall that tau spent'?

    If it were, its dR would GROW with tau -- there would be more to restore.
    If dR is roughly flat in tau, the model adds a tau-independent increment
    that merely happens to pay for tau, and the causal 'restores' framing in
    docs/CPRIME_MECHANISM_REPORT.md is too strong.
    """
    out = []
    for t in TAUS:
        mp = B.metrics(B.score(t, ALPHA_HIST, None))
        mc = B.metrics(B.score(t, ALPHA_HIST, B.model))
        out.append({"tau": t, "prior_R": mp["R"], "combined_R": mc["R"],
                    "dR_points": (mc["R"] - mp["R"]) * 100.0,
                    "R_spent_by_tau_points": (out[0]["prior_R"] - mp["R"]) * 100.0 if out else 0.0})
    return {"rows": out}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p10_model_recalibration/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--scheme", default="raw50", choices=["raw50", "eval48"])
    ap.add_argument("--out", default="runs/p13_oracle_ceiling/oracle.json")
    ap.add_argument("--taus", default="0.0,0.05,0.1")
    args = ap.parse_args(argv)

    taus = [float(x) for x in args.taus.split(",")]
    _log("=" * 92)
    _log("C' ORACLE CEILING -- CPU only, cache read-only, NO GPU")
    _log("=" * 92)
    B = Mech(args.dump, args.prior, args.scheme)
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    _log(f"  images={B.n_images} GT rows={B.n_gt} classes={B.n_classes}  "
         f"[gate] model-term identity {e:.3e} OK")
    _log(f"  R@50 floor = {R_FLOOR*100:.1f}   SUCCESS > +{SUCCESS_PTS} pts   "
         f"EXHAUSTED < +{INCONCLUSIVE_PTS} pts")

    res: Dict[str, Any] = {"tool": "cprime_oracle_ceiling", "dump": args.dump,
                           "prior": args.prior, "scheme": args.scheme,
                           "alpha": ALPHA_HIST, "R_floor": R_FLOOR,
                           "success_points": SUCCESS_PTS,
                           "inconclusive_points": INCONCLUSIVE_PTS,
                           "ks": KS, "budgets": {n: b for n, b in BUDGETS},
                           "by_tau": {}}

    for tau in taus:
        O = Oracle(B, tau)
        prior_m = O.arm("prior", B.n_classes, float("inf"))
        achieved = O.arm("combined", B.n_classes, float("inf"))
        achieved_dR = (achieved["R"] - prior_m["R"]) * 100.0
        _log(f"\n{'-'*92}\n  tau={tau}  prior R@50 {prior_m['R']*100:.3f} mR {prior_m['mR']*100:.3f}"
             f"   |  ACHIEVED (real model, alpha=3.75) R@50 {achieved['R']*100:.3f}"
             f" mR {achieved['mR']*100:.3f}  dR {achieved_dR:+.3f}\n{'-'*92}")
        _log(f"  {'k':>3} {'budget':>14} {'elig%':>7} {'cover%':>7} | {'rerank R':>9} {'rerank dR':>10}"
             f" {'r.floor':>8} | {'oracle R':>9} {'oracle mR':>10} {'oracle dR':>10} {'gap vs ach':>11}"
             f" | {'PREREG':>7} {'REALIZABLE':>11}")
        rows = []
        for k in KS:
            for bname, b in BUDGETS:
                rr = O.arm("model_rerank", k, b)
                orc = O.arm("oracle", k, b)
                o_dR = (orc["R"] - prior_m["R"]) * 100.0
                r_dR = (rr["R"] - prior_m["R"]) * 100.0
                gap = o_dR - achieved_dR
                floor_ok = orc["R"] >= R_FLOOR
                verdict = ("SUCCESS" if (gap > SUCCESS_PTS and floor_ok) else
                           "INCONCLUSIVE" if (gap >= INCONCLUSIVE_PTS and floor_ok) else
                           "EXHAUSTED")
                cov = O.coverage(k, b)
                # CORRECTED GATE (documented in docs/ORACLE_CEILING_RESULT.md).
                # The pre-registered gate above cannot fail: the oracle never
                # makes a wrong decision, so oracle_R >= prior_R >= floor holds
                # by construction and `gap` is mechanically large -- it measures
                # candidate COVERAGE, not tie-breaker headroom. The floor was
                # meant to stop a degenerate operating point masquerading as
                # headroom, so it must bind on the arm that can actually BE
                # degenerate: model_rerank, the only realizable arm here.
                rerank_floor_ok = rr["R"] >= R_FLOOR
                realizable_verdict = ("SUCCESS" if (rerank_floor_ok and r_dR > achieved_dR + SUCCESS_PTS)
                                      else "INCONCLUSIVE" if (rerank_floor_ok and r_dR > achieved_dR + INCONCLUSIVE_PTS)
                                      else "EXHAUSTED")
                row = {"k": k, "budget_name": bname, "budget": b,
                       "frac_eligible": orc["frac_eligible"],
                       "prior_R": prior_m["R"], "prior_mR": prior_m["mR"],
                       "prior_head_mR": prior_m["head_mR"], "prior_body_mR": prior_m["body_mR"],
                       "prior_tail_mR": prior_m["tail_mR"],
                       "achieved_R": achieved["R"], "achieved_mR": achieved["mR"],
                       "achieved_dR_points": achieved_dR,
                       "model_rerank_R": rr["R"], "model_rerank_mR": rr["mR"],
                       "model_rerank_dR_points": r_dR,
                       "model_rerank_head_mR": rr["head_mR"], "model_rerank_body_mR": rr["body_mR"],
                       "model_rerank_tail_mR": rr["tail_mR"],
                       "oracle_R": orc["R"], "oracle_mR": orc["mR"],
                       "oracle_dR_points": o_dR,
                       "oracle_head_mR": orc["head_mR"], "oracle_body_mR": orc["body_mR"],
                       "oracle_tail_mR": orc["tail_mR"],
                       "gap_oracle_minus_achieved_points": gap,
                       "gap_oracle_minus_rerank_points": o_dR - r_dR,
                       "gap_rerank_minus_achieved_points": r_dR - achieved_dR,
                       "candidate_coverage_all_rows": cov["coverage_all_rows"],
                       "candidate_coverage_eligible_rows": cov["coverage_eligible_rows"],
                       "meets_R_floor": bool(floor_ok),
                       "model_rerank_meets_R_floor": bool(rerank_floor_ok),
                       "verdict": verdict,
                       "realizable_verdict": realizable_verdict}
                # per-predicate oracle contribution: which classes the ceiling
                # is actually built out of, in recall points.
                pr_per, or_per = prior_m["_per_class_recall"], orc["_per_class_recall"]
                contrib = sorted(((B.classes[c], (or_per[c] - pr_per[c]) * 100.0,
                                   B.bucket_of[c]) for c in or_per),
                                 key=lambda t: -t[1])
                row["per_predicate_oracle_gain_points"] = [
                    {"predicate": n, "d_recall_points": d, "bucket": bk}
                    for n, d, bk in contrib[:10]]
                rows.append(row)
                _log(f"  {k:>3} {bname:>14} {orc['frac_eligible']*100:>6.1f}%"
                     f" {cov['coverage_eligible_rows']*100:>6.1f}% |"
                     f" {rr['R']*100:>9.3f} {r_dR:>+10.3f}"
                     f" {'ok' if rerank_floor_ok else 'FAIL':>8} |"
                     f" {orc['R']*100:>9.3f} {orc['mR']*100:>10.3f} {o_dR:>+10.3f}"
                     f" {gap:>+11.3f} | {verdict:>7} {realizable_verdict:>11}")
        rd = region_decomposition(B, tau)
        _log(f"\n  model's actual net flips = {rd['net_flips_total']:+d}; share inside each region:")
        for r in rd["regions"]:
            if r["frac_of_total_net"] is not None:
                _log(f"    k={r['k']:<3} budget={r['budget']:<14} net_inside={r['net_flips_inside']:+5d} "
                     f"({r['frac_of_total_net']*100:5.1f}% of total)")
        for _m in (prior_m, achieved):
            _m.pop("_per_class_recall", None)
        res["by_tau"][str(tau)] = {"prior": prior_m, "achieved": achieved,
                                   "achieved_dR_points": achieved_dR,
                                   "ladder": rows, "region_decomposition": rd}

    _log(f"\n{'-'*92}\n  FALSIFICATION: does dR grow with tau (i.e. is the model RESTORING what tau spent)?\n{'-'*92}")
    ti = tau_independence(B)
    _log(f"  {'tau':>6} {'prior R':>9} {'combined R':>11} {'model dR':>9} {'R spent by tau':>15}")
    for r in ti["rows"]:
        _log(f"  {r['tau']:>6} {r['prior_R']*100:>9.3f} {r['combined_R']*100:>11.3f} "
             f"{r['dR_points']:>+9.3f} {r['R_spent_by_tau_points']:>+15.3f}")
    res["tau_independence"] = ti

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
