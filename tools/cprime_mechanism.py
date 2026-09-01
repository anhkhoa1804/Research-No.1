#!/usr/bin/env python
"""C' follow-up -- the MECHANISM of the model's contribution, from the same cache.

Reads runs/p10_model_recalibration/pair_logits.pt read-only. CPU only. No GPU,
no training, no modification of any C' artifact.

The question this answers
-------------------------
C' measured that model + prior beats the prior-only tau frontier on both axes,
while the model *worsens* the mean GT rank (1.83 -> 2.78) and worsens more rows
(4,675) than it improves (2,013), yet still yields +256 net beneficial top-1
flips. Those three facts are only paradoxical if the model is assumed to act as
a *ranker*. This tool tests the alternative: it acts as a bounded TIE-BREAKER on
an argmax the prior has already almost decided.

Four claims that C' did not separate, and that this tool separates:
  1. information EXISTS in the model            (arm C, vs a majority baseline)
  2. information CHANGES scores                 (argmax churn rate)
  3. information improves GLOBAL RANKING        (MRR / mean rank / recall@k)
  4. information improves the FINAL TOP-1       (net flips, R@50, mR@50)

A NORMALISATION DEFECT IN cprime_analysis.py
--------------------------------------------
`Bench.ensemble_term` slices to the 50 foreground columns and *then*
standardises. `evals.py::_normalize_eval_logits` standardises over the FULL 51
columns (background included) and the background is suppressed afterwards.
Order matters: max|difference| = 1.61e-01 over the cache.

This tool implements `fixed_ensemble`, which normalises over the full width and
then slices. `fixed_ensemble(0.0)` reproduces the cache's stored `model_logits`
to 3.6e-06, which the buggy path does not -- that identity is the proof of which
order is correct, and it is asserted at startup (`--assert-fix`).

Every arm below uses the STORED model term (`Bench.model`), which was never
affected by the defect. The defect's blast radius is reported separately by
`--reconcile` and is NOT silently corrected in any C' artifact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load_cpa():
    """Import tools/cprime_analysis.py by path -- reuse its VALIDATED Bench.

    Reusing it (rather than reimplementing) guarantees the marginal loading,
    the alias/scheme handling and the Pareto gap are bit-identical to C'.
    """
    p = _ROOT / "tools" / "cprime_analysis.py"
    spec = importlib.util.spec_from_file_location("cprime_analysis", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cprime_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


CPA = _load_cpa()
ALPHA_HIST = CPA.ALPHA_HIST
TAUS = CPA.TAUS
OP_TAUS = [0.0, 0.05]          # the two operating points C' actually quotes
SCALES = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
TOPK_RESTRICT = [2, 3, 5, 10, 50]


def _log(m: str = "") -> None:
    print(m, flush=True)


def _sd(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


class Mech(CPA.Bench):
    """Bench plus the correctly-ordered branch normalisation and row views."""

    def __init__(self, dump: str, prior: str, scheme: str = "raw50"):
        super().__init__(Path(dump), scheme, Path(prior))
        d = self.meta
        bg = set(int(i) for i in d.get("background_predicate_indices", []))
        fgc = [i for i in range(len(d["pred_vocab"])) if i not in bg]
        tn, cn = [], []
        for i in range(self.n_images):
            tn.append(self._norm(d["text_logits"][i])[:, fgc])
            cn.append(self._norm(d["cls_logits"][i])[:, fgc])
        self.text_norm51 = torch.cat(tn, 0)
        self.cls_norm51 = torch.cat(cn, 0)
        # GT-row views: every analysis below is per GT row, not per pair row.
        self.gt_img = self.img_of_row[self.gt_row]
        self.n_gt = int(self.gt_row.numel())

    def fixed_ensemble(self, ea: float) -> torch.Tensor:
        """The model term the evaluator ACTUALLY builds (normalise 51, slice)."""
        ct = float(self.meta.get("classifier_temperature", 1.0)) or 1.0
        tt = float(self.meta.get("text_temperature", 1.0)) or 1.0
        return ea * (self.cls_norm51 / ct) + (1.0 - ea) * (self.text_norm51 / tt)

    # --------------------------------------------------------------- helpers
    def rows(self, s: torch.Tensor) -> torch.Tensor:
        """Score matrix restricted to GT rows: (n_gt, n_classes)."""
        return s[self.gt_row]

    def gt_col_of(self) -> torch.Tensor:
        """Column index of the GT class (raw50: col_to_class is a bijection)."""
        inv = torch.zeros(self.n_classes, dtype=torch.long)
        for c in range(self.n_classes):
            inv[c] = int((self.col_to_class == c).nonzero()[0])
        return inv[self.gt_y]

    def per_class_recall_vec(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, y = self.predict(s), self.gt_y
        hit = (pred == y).float()
        num = torch.zeros(self.n_classes).index_add_(0, y, hit)
        den = torch.zeros(self.n_classes).index_add_(0, y, torch.ones_like(hit))
        return num, den


# ------------------------------------------------------------------ analyses
def analysis_A_flips(B: Mech, tau: float) -> Dict[str, Any]:
    """A. Exact top-1 flip decomposition at one operating point."""
    sp = B.score(tau, ALPHA_HIST, None)
    sc = B.score(tau, ALPHA_HIST, B.model)
    pp, pc, y = B.predict(sp), B.predict(sc), B.gt_y
    ok_p, ok_c = (pp == y), (pc == y)
    changed = (pp != pc)
    resc = (~ok_p) & ok_c
    dest = ok_p & (~ok_c)
    both_w_ch = (~ok_p) & (~ok_c) & changed
    return {
        "tau": tau,
        "n_gt": B.n_gt,
        "prior_R": float(ok_p.float().mean()),
        "combined_R": float(ok_c.float().mean()),
        "both_right": int((ok_p & ok_c).sum()),
        "rescued_wrong_to_right": int(resc.sum()),
        "destroyed_right_to_wrong": int(dest.sum()),
        "both_wrong_unchanged": int(((~ok_p) & (~ok_c) & ~changed).sum()),
        "both_wrong_changed": int(both_w_ch.sum()),
        "net_flips": int(resc.sum()) - int(dest.sum()),
        "argmax_changed": int(changed.sum()),
        "argmax_changed_frac": float(changed.float().mean()),
        "churn_efficiency_net_per_change": (int(resc.sum()) - int(dest.sum())) / max(1, int(changed.sum())),
        "rescue_rate_of_prior_errors": float(resc.sum() / max(1, int((~ok_p).sum()))),
        "destruction_rate_of_prior_hits": float(dest.sum() / max(1, int(ok_p.sum()))),
    }


def analysis_B_ranks(B: Mech, tau: float) -> Dict[str, Any]:
    """B. GT-rank distribution before/after, plus proper ranking metrics."""
    sp = B.score(tau, ALPHA_HIST, None)
    sc = B.score(tau, ALPHA_HIST, B.model)
    sm = B.score(tau, 0.0, B.model)
    rp, rc, rm = B.gt_rank(sp), B.gt_rank(sc), B.gt_rank(sm)

    def hist(r: torch.Tensor) -> Dict[str, int]:
        b = {"1": int((r == 1).sum()), "2": int((r == 2).sum()), "3": int((r == 3).sum()),
             "4": int((r == 4).sum()), "5": int((r == 5).sum()),
             "6-10": int(((r >= 6) & (r <= 10)).sum()),
             "11-20": int(((r >= 11) & (r <= 20)).sum()),
             "21+": int((r >= 21).sum())}
        return b

    def summ(r: torch.Tensor) -> Dict[str, float]:
        rf = r.float()
        return {"mean": float(rf.mean()), "median": float(rf.median()),
                "MRR": float((1.0 / rf).mean()),
                **{f"recall@{k}": float((r <= k).float().mean()) for k in (1, 2, 3, 5, 10)}}

    top5 = (rp <= 5)
    return {
        "tau": tau,
        "prior": {"hist": hist(rp), **summ(rp)},
        "combined": {"hist": hist(rc), **summ(rc)},
        "model_only": {"hist": hist(rm), **summ(rm)},
        "rank_improved": int((rc < rp).sum()),
        "rank_worsened": int((rc > rp).sum()),
        "rank_unchanged": int((rc == rp).sum()),
        "mean_rank_within_prior_top5": {
            "n": int(top5.sum()),
            "prior": float(rp[top5].float().mean()),
            "combined": float(rc[top5].float().mean()),
        },
        "mean_rank_outside_prior_top5": {
            "n": int((~top5).sum()),
            "prior": float(rp[~top5].float().mean()),
            "combined": float(rc[~top5].float().mean()),
        },
        "mean_rank_delta_by_prior_rank": {
            str(k): float((rc[rp == k].float() - k).mean()) if int((rp == k).sum()) else None
            for k in (1, 2, 3, 4, 5)
        },
    }


def analysis_C_margins(B: Mech, tau: float) -> Dict[str, Any]:
    """C. Confidence and top1-top2 margin: the tie-breaker budget test.

    The combined argmax can differ from the prior argmax only where the prior's
    own top1-top2 margin is smaller than the model's differential across those
    two classes. The model term is per-row standardised (unit sd, 50 classes),
    so its differential is bounded in practice. This measures the budget.
    """
    pr = B.rows(B.score(tau, ALPHA_HIST, None))
    md = B.rows(B.score(tau, 0.0, B.model))
    sc = pr + md
    t2 = pr.topk(2, dim=-1)
    margin = (t2.values[:, 0] - t2.values[:, 1])
    # The prior's top-1 must be argmax, NOT topk(2).indices[:, 0]. 522 GT rows
    # have an exact tie for the maximum, and torch breaks that tie differently
    # in the two calls -- topk there gives prior R@50 66.7543 where the
    # evaluator (and therefore C') gives 66.8016. `margin` is read off topk
    # VALUES, which are tie-order invariant, so only the index moves.
    ptop = pr.argmax(-1)
    ctop = sc.argmax(-1)
    changed = (ctop != ptop)
    # model differential across the actually-competing pair
    supplied = md.gather(1, ctop.unsqueeze(1)).squeeze(1) - md.gather(1, ptop.unsqueeze(1)).squeeze(1)
    q = torch.tensor([0.1 * i for i in range(1, 10)])
    dec = torch.quantile(margin, q)
    edges = [float("-inf")] + dec.tolist() + [float("inf")]
    by_dec = []
    for i in range(10):
        m = (margin >= edges[i]) & (margin < edges[i + 1])
        if not bool(m.any()):
            continue
        by_dec.append({"decile": i + 1, "n": int(m.sum()),
                       "margin_lo": edges[i] if i else float(margin.min()),
                       "margin_hi": edges[i + 1] if i < 9 else float(margin.max()),
                       "changed_frac": float(changed[m].float().mean())})
    md_spread = (md.max(-1).values - md.min(-1).values)
    return {
        "tau": tau,
        "prior_margin": {"mean": float(margin.mean()), "median": float(margin.median()),
                         "p10": float(torch.quantile(margin, 0.10)),
                         "p25": float(torch.quantile(margin, 0.25))},
        "model_row_spread_max_minus_min": {"mean": float(md_spread.mean()),
                                           "p95": float(torch.quantile(md_spread, 0.95))},
        "frac_rows_margin_below_model_spread": float((margin < md_spread).float().mean()),
        "changed_frac_overall": float(changed.float().mean()),
        "changed_by_prior_margin_decile": by_dec,
        "mean_prior_margin_changed": float(margin[changed].mean()) if bool(changed.any()) else None,
        "mean_prior_margin_unchanged": float(margin[~changed].mean()),
        "mean_supplied_when_changed": float(supplied[changed].mean()) if bool(changed.any()) else None,
    }


def analysis_DE_buckets_predicates(B: Mech, tau: float) -> Dict[str, Any]:
    """D. head/body/tail breakdown.  E. per-predicate contribution to mR."""
    sp = B.score(tau, ALPHA_HIST, None)
    sc = B.score(tau, ALPHA_HIST, B.model)
    np_, dp = B.per_class_recall_vec(sp)
    nc, dc = B.per_class_recall_vec(sc)
    present = (dp > 0)
    rec_p = torch.where(present, np_ / dp.clamp_min(1), torch.zeros_like(np_))
    rec_c = torch.where(present, nc / dc.clamp_min(1), torch.zeros_like(nc))
    K = int(present.sum())
    per_class = []
    for c in range(B.n_classes):
        if not bool(present[c]):
            continue
        per_class.append({
            "class": B.classes[c], "bucket": B.bucket_of[c], "n_gt": int(dp[c]),
            "recall_prior": float(rec_p[c]), "recall_combined": float(rec_c[c]),
            "delta_recall": float(rec_c[c] - rec_p[c]),
            "contribution_mR_points": float((rec_c[c] - rec_p[c]) / K * 100.0),
        })
    per_class.sort(key=lambda r: -abs(r["contribution_mR_points"]))

    pp, pc, y = B.predict(sp), B.predict(sc), B.gt_y
    ok_p, ok_c = (pp == y), (pc == y)
    buckets = {}
    for b in ("head", "body", "tail"):
        cls = [c for c in range(B.n_classes) if B.bucket_of[c] == b and bool(present[c])]
        sel = torch.zeros(B.n_gt, dtype=torch.bool)
        for c in cls:
            sel |= (y == c)
        d_mr = float(sum(float(rec_c[c] - rec_p[c]) for c in cls) / K * 100.0)
        buckets[b] = {
            "n_classes": len(cls), "n_gt_rows": int(sel.sum()),
            "mR_prior": float(sum(float(rec_p[c]) for c in cls) / max(1, len(cls))),
            "mR_combined": float(sum(float(rec_c[c]) for c in cls) / max(1, len(cls))),
            "contribution_to_global_dmR_points": d_mr,
            "rescued": int(((~ok_p) & ok_c & sel).sum()),
            "destroyed": int((ok_p & (~ok_c) & sel).sum()),
            "net_flips": int(((~ok_p) & ok_c & sel).sum()) - int((ok_p & (~ok_c) & sel).sum()),
        }
    # where do the flips LAND (predicted class), not only where they come from
    land = {"rescued_pred_bucket": Counter(), "destroyed_pred_bucket": Counter()}
    for i in ((~ok_p) & ok_c).nonzero().squeeze(1).tolist():
        land["rescued_pred_bucket"][B.bucket_of[int(pc[i])]] += 1
    for i in (ok_p & (~ok_c)).nonzero().squeeze(1).tolist():
        land["destroyed_pred_bucket"][B.bucket_of[int(pc[i])]] += 1
    return {"tau": tau, "n_classes_present": K, "buckets": buckets,
            "flip_landing_bucket": {k: dict(v) for k, v in land.items()},
            "per_class_top20": per_class[:20],
            "per_class_all": per_class}


def analysis_F_entropy(B: Mech, tau: float) -> Dict[str, Any]:
    """F. prior-confidence / prior-entropy stratification."""
    sp = B.score(tau, ALPHA_HIST, None)
    sc = B.score(tau, ALPHA_HIST, B.model)
    pp, pc, y = B.predict(sp), B.predict(sc), B.gt_y
    ok_p, ok_c = (pp == y), (pc == y)
    out = {}
    for si, nm in ((0, "low_entropy"), (1, "mid_entropy"), (2, "high_entropy")):
        m = (B.ent_stratum == si)
        mp, mc = B.metrics(sp, mask=m), B.metrics(sc, mask=m)
        out[nm] = {
            "n": int(m.sum()),
            "prior_entropy_mean": float(B.prior_entropy[m].mean()),
            "R_prior": mp["R"], "R_combined": mc["R"], "dR_points": (mc["R"] - mp["R"]) * 100,
            "mR_prior": mp["mR"], "mR_combined": mc["mR"], "dmR_points": (mc["mR"] - mp["mR"]) * 100,
            "rescued": int(((~ok_p) & ok_c & m).sum()),
            "destroyed": int((ok_p & (~ok_c) & m).sum()),
            "net_flips": int(((~ok_p) & ok_c & m).sum()) - int((ok_p & (~ok_c) & m).sum()),
            "argmax_changed": int(((pp != pc) & m).sum()),
        }
    return {"tau": tau, "strata": out}


def analysis_G_rescue(B: Mech, tau: float) -> Dict[str, Any]:
    """G. model-vs-prior GT rescue: where was GT, and who supplied the rescue?"""
    sp = B.score(tau, ALPHA_HIST, None)
    sc = B.score(tau, ALPHA_HIST, B.model)
    sm = B.score(tau, 0.0, B.model)
    rp, rm = B.gt_rank(sp), B.gt_rank(sm)
    pp, pc, y = B.predict(sp), B.predict(sc), B.gt_y
    ok_p, ok_c = (pp == y), (pc == y)
    resc = (~ok_p) & ok_c
    err = ~ok_p
    by_prior_rank = []
    for k in (2, 3, 4, 5):
        m = err & (rp == k)
        by_prior_rank.append({"prior_gt_rank": k, "n_prior_errors": int(m.sum()),
                              "rescued": int((m & resc).sum()),
                              "rescue_rate": float((m & resc).float().sum() / max(1, int(m.sum())))})
    m6 = err & (rp >= 6)
    by_prior_rank.append({"prior_gt_rank": "6+", "n_prior_errors": int(m6.sum()),
                          "rescued": int((m6 & resc).sum()),
                          "rescue_rate": float((m6 & resc).float().sum() / max(1, int(m6.sum())))})
    rmf = rm.float()
    return {
        "tau": tau,
        "n_prior_errors": int(err.sum()),
        "rescued": int(resc.sum()),
        "rescue_by_prior_gt_rank": by_prior_rank,
        "model_only_gt_rank_mean_rescued": float(rmf[resc].mean()) if bool(resc.any()) else None,
        "model_only_gt_rank_mean_prior_errors": float(rmf[err].mean()),
        "model_only_gt_rank_mean_all": float(rmf.mean()),
        "model_ranks_gt_top1_frac_rescued": float((rm[resc] == 1).float().mean()) if bool(resc.any()) else None,
        "model_ranks_gt_top1_frac_all": float((rm == 1).float().mean()),
        "model_ranks_gt_top5_frac_rescued": float((rm[resc] <= 5).float().mean()) if bool(resc.any()) else None,
        "model_ranks_gt_top5_frac_all": float((rm <= 5).float().mean()),
    }


def analysis_H_beneficial_vs_harmful(B: Mech, tau: float) -> Dict[str, Any]:
    """H. compare the rescued rows against the destroyed rows, side by side."""
    pr = B.rows(B.score(tau, ALPHA_HIST, None))
    md = B.rows(B.score(tau, 0.0, B.model))
    sc = pr + md
    y = B.gt_y
    pp = B.col_to_class[pr.argmax(-1)]
    pc = B.col_to_class[sc.argmax(-1)]
    ok_p, ok_c = (pp == y), (pc == y)
    resc, dest = (~ok_p) & ok_c, ok_p & (~ok_c)
    t2 = pr.topk(2, dim=-1)
    margin = t2.values[:, 0] - t2.values[:, 1]
    gtcol = B.gt_col_of()
    model_gt_adv = md.gather(1, gtcol.unsqueeze(1)).squeeze(1) - md.max(-1).values

    def prof(m: torch.Tensor, name: str) -> Dict[str, Any]:
        if not bool(m.any()):
            return {"group": name, "n": 0}
        cls_from = Counter(B.classes[int(c)] for c in y[m].tolist())
        return {
            "group": name, "n": int(m.sum()),
            "mean_prior_margin": float(margin[m].mean()),
            "median_prior_margin": float(margin[m].median()),
            "mean_prior_entropy": float(B.prior_entropy[m].mean()),
            "gt_bucket_counts": dict(Counter(B.bucket_of[int(c)] for c in y[m].tolist())),
            "top5_gt_classes": cls_from.most_common(5),
            "mean_model_gt_advantage": float(model_gt_adv[m].mean()),
        }

    return {"tau": tau,
            "rescued": prof(resc, "rescued"),
            "destroyed": prof(dest, "destroyed"),
            "unchanged": prof(pp == pc, "unchanged"),
            "all_rows": prof(torch.ones_like(resc), "all")}


# ---------------------------------------------------------------- controls
def controls(B: Mech, tau: float, seeds: int, curve: List[Dict[str, float]]) -> Dict[str, Any]:
    """I. falsification battery: is the effect information, or is it noise?"""
    y = B.gt_y
    sp = B.score(tau, ALPHA_HIST, None)
    pp = B.predict(sp)
    ok_p = (pp == y)
    base = B.metrics(sp)

    def report(mt: Optional[torch.Tensor], rows: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        s = B.score(tau, ALPHA_HIST, mt, model_rows=rows)
        m = B.metrics(s)
        pc = B.predict(s)
        ok_c = (pc == y)
        return {"R": m["R"], "mR": m["mR"],
                "dR_points": (m["R"] - base["R"]) * 100, "dmR_points": (m["mR"] - base["mR"]) * 100,
                "rescued": int(((~ok_p) & ok_c).sum()), "destroyed": int((ok_p & (~ok_c)).sum()),
                "net_flips": int(((~ok_p) & ok_c).sum()) - int((ok_p & (~ok_c)).sum()),
                "argmax_changed": int((pp != pc).sum()),
                "pareto_dmR_points": CPA.pareto_gap(curve, m["R"], m["mR"])}

    out: Dict[str, Any] = {"tau": tau, "prior_only": {"R": base["R"], "mR": base["mR"]},
                           "real": report(B.model)}

    # C1 -- shuffled model term (the C' nulls, re-read at the FLIP level)
    for kind in ("N1", "N2"):
        rs = [report(B.model, CPA.null_rows(B, kind, 7000 + 137 * s)) for s in range(seeds)]
        out[f"null_{kind}"] = {
            "seeds": seeds,
            "mean_net_flips": sum(r["net_flips"] for r in rs) / seeds,
            "mean_dR_points": sum(r["dR_points"] for r in rs) / seeds,
            "mean_dmR_points": sum(r["dmR_points"] for r in rs) / seeds,
            "sd_dmR_points": _sd([r["dmR_points"] for r in rs]),
            "mean_argmax_changed": sum(r["argmax_changed"] for r in rs) / seeds,
            "mean_pareto_dmR_points": sum(r["pareto_dmR_points"] for r in rs) / seeds,
        }

    # C2 -- Gaussian noise with the model term's own per-row scale
    sd_row = B.model.std(-1, keepdim=True)
    rs = []
    for s in range(seeds):
        g = torch.Generator().manual_seed(90210 + 31 * s)
        rs.append(report(torch.randn(B.model.shape, generator=g) * sd_row))
    out["null_gaussian_matched_scale"] = {
        "seeds": seeds,
        "mean_net_flips": sum(r["net_flips"] for r in rs) / seeds,
        "mean_dR_points": sum(r["dR_points"] for r in rs) / seeds,
        "mean_dmR_points": sum(r["dmR_points"] for r in rs) / seeds,
        "sd_dmR_points": _sd([r["dmR_points"] for r in rs]),
        "mean_argmax_changed": sum(r["argmax_changed"] for r in rs) / seeds,
        "mean_pareto_dmR_points": sum(r["pareto_dmR_points"] for r in rs) / seeds,
    }

    # C3 -- scale sweep: tie-breaker regime is PEAKED, a true ranker is monotone
    out["scale_sweep"] = [dict(scale=c, **report(B.model * c)) for c in SCALES]

    # C4 -- rank transform: destroys magnitude, keeps ordering
    order = B.model.argsort(-1).argsort(-1).float()
    rk = (order - order.mean(-1, keepdim=True)) / order.std(-1, keepdim=True).clamp_min(1e-4)
    out["rank_transform"] = report(rk)

    # C5 -- restrict the model to reorder only inside the prior's top-k
    pr = B.prior - tau * B.log_marginal.view(1, -1)
    for k in TOPK_RESTRICT:
        keep = torch.zeros_like(B.model, dtype=torch.bool)
        keep.scatter_(1, pr.topk(min(k, B.n_classes), dim=-1).indices, True)
        out[f"restrict_prior_top{k}"] = report(torch.where(keep, B.model, torch.zeros_like(B.model)))

    # C6 -- sparse: the model votes only for its own argmax, magnitude discarded
    onehot = torch.zeros_like(B.model)
    onehot.scatter_(1, B.model.argmax(-1, keepdim=True), 1.0)
    out["sparse_argmax_vote"] = {str(d): report(onehot * d) for d in (0.5, 1.0, 2.0, 4.0)}

    # C7 -- majority-class baseline, for claim 1
    cnt = torch.bincount(y, minlength=B.n_classes)
    out["majority_class_baseline_R"] = float(cnt.max() / y.numel())
    out["majority_class"] = B.classes[int(cnt.argmax())]
    return out


def heldout(B: Mech, seeds: int, select_seed: int) -> Dict[str, Any]:
    """Held-out selection of the model-term scale, with no validation leakage.

    Images (not rows) are split 50/50, so no image contributes rows to both
    halves. The scale c and tau are chosen on half A by Pareto gap against
    half A's OWN prior frontier, and read once on half B against half B's own
    frontier. The null is rescored on half B at the SELECTED (c, tau).
    """
    g = torch.Generator().manual_seed(select_seed)
    perm = torch.randperm(B.n_images, generator=g)
    halfA = set(perm[: B.n_images // 2].tolist())
    inA = torch.tensor([int(i) in halfA for i in B.gt_img.tolist()])
    curveA = [dict(B.metrics(B.score(t, ALPHA_HIST, None), mask=inA), tau=t) for t in TAUS]
    curveB = [dict(B.metrics(B.score(t, ALPHA_HIST, None), mask=~inA), tau=t) for t in TAUS]
    best, bestgap = None, -1e9
    grid = []
    for c in SCALES:
        for t in TAUS:
            mA = B.metrics(B.score(t, ALPHA_HIST, B.model * c), mask=inA)
            gA = CPA.pareto_gap(curveA, mA["R"], mA["mR"])
            grid.append({"scale": c, "tau": t, "halfA_pareto": gA})
            if gA is not None and gA > bestgap:
                bestgap, best = gA, (c, t)
    c, t = best
    mB = B.metrics(B.score(t, ALPHA_HIST, B.model * c), mask=~inA)
    gB = CPA.pareto_gap(curveB, mB["R"], mB["mR"])
    nl = []
    for s in range(seeds):
        idx = CPA.null_rows(B, "N1", 7000 + 137 * s)
        mn = B.metrics(B.score(t, ALPHA_HIST, B.model * c, model_rows=idx), mask=~inA)
        gg = CPA.pareto_gap(curveB, mn["R"], mn["mR"])
        if gg is not None:
            nl.append(gg)
    return {"select_seed": select_seed, "n_images_A": len(halfA),
            "n_rows_A": int(inA.sum()), "n_rows_B": int((~inA).sum()),
            "selected_scale": c, "selected_tau": t, "halfA_pareto_dmR_points": bestgap,
            "heldout_B_R": mB["R"], "heldout_B_mR": mB["mR"],
            "heldout_B_pareto_dmR_points": gB,
            "heldout_B_null_mean": (sum(nl) / len(nl)) if nl else None,
            "heldout_B_null_sd": _sd(nl),
            "heldout_B_margin_over_null": (gB - sum(nl) / len(nl)) if (nl and gB is not None) else None,
            "grid": grid}


def analysis_J_stability(B: Mech, tau: float, n_boot: int, seed: int) -> Dict[str, Any]:
    """J. Is the effect carried by enough rows to survive resampling?

    E shows the mR delta is concentrated on a few low-count predicates, and mR
    weights a 56-row class exactly as heavily as a 13,902-row class. So the
    headline deltas need a sampling distribution, and the resampling unit must
    be the IMAGE -- rows inside an image are not independent.

    Implemented as a multinomial reweighting of per-image, per-class hit and
    count matrices, which is algebraically identical to resampling images with
    replacement but costs one matmul per draw instead of a gather.
    """
    sp = B.score(tau, ALPHA_HIST, None)
    sc = B.score(tau, ALPHA_HIST, B.model)
    y = B.gt_y
    okp = (B.predict(sp) == y).float()
    okc = (B.predict(sc) == y).float()
    I, K = B.n_images, B.n_classes
    idx = B.gt_img * K + y
    Hp = torch.zeros(I * K).index_add_(0, idx, okp).view(I, K)
    Hc = torch.zeros(I * K).index_add_(0, idx, okc).view(I, K)
    N = torch.zeros(I * K).index_add_(0, idx, torch.ones_like(okp)).view(I, K)

    def agg(w: torch.Tensor) -> Tuple[float, float, float, float]:
        np_, nc_, nn = w @ Hp, w @ Hc, w @ N
        pres = nn > 0
        R_p = float(np_.sum() / nn.sum()); R_c = float(nc_.sum() / nn.sum())
        mR_p = float((np_[pres] / nn[pres]).mean()); mR_c = float((nc_[pres] / nn[pres]).mean())
        return R_p, R_c, mR_p, mR_c

    g = torch.Generator().manual_seed(seed)
    probs = torch.full((I,), 1.0 / I)
    dR, dmR = [], []
    for _ in range(n_boot):
        w = torch.multinomial(probs, I, replacement=True, generator=g).bincount(minlength=I).float()
        Rp, Rc, mp, mc = agg(w)
        dR.append((Rc - Rp) * 100.0); dmR.append((mc - mp) * 100.0)
    dR_t, dmR_t = torch.tensor(dR), torch.tensor(dmR)

    # leave-one-class-out: how much of dmR survives dropping its best class?
    npc_p, den = B.per_class_recall_vec(sp)
    npc_c, _ = B.per_class_recall_vec(sc)
    pres = den > 0
    rp = torch.where(pres, npc_p / den.clamp_min(1), torch.zeros_like(den))
    rc = torch.where(pres, npc_c / den.clamp_min(1), torch.zeros_like(den))
    delta = (rc - rp)[pres]
    names = [B.classes[c] for c in range(K) if bool(pres[c])]
    Kp = int(pres.sum())
    full = float(delta.sum() / Kp * 100.0)
    order = torch.argsort(delta, descending=True)
    drop1 = float((delta.sum() - delta[order[0]]) / (Kp - 1) * 100.0)
    drop2 = float((delta.sum() - delta[order[:2]].sum()) / (Kp - 2) * 100.0)
    return {
        "tau": tau, "n_boot": n_boot, "seed": seed, "resample_unit": "image",
        "dR_points": {"point": float(dR_t.mean()), "sd": float(dR_t.std()),
                      "ci2.5": float(torch.quantile(dR_t, 0.025)),
                      "ci97.5": float(torch.quantile(dR_t, 0.975)),
                      "frac_draws_positive": float((dR_t > 0).float().mean())},
        "dmR_points": {"point": float(dmR_t.mean()), "sd": float(dmR_t.std()),
                       "ci2.5": float(torch.quantile(dmR_t, 0.025)),
                       "ci97.5": float(torch.quantile(dmR_t, 0.975)),
                       "frac_draws_positive": float((dmR_t > 0).float().mean())},
        "leave_best_classes_out": {
            "dmR_full": full,
            "best_class": names[int(order[0])], "dmR_without_best": drop1,
            "second_class": names[int(order[1])], "dmR_without_best_two": drop2},
    }

def reconcile(B: Mech) -> Dict[str, Any]:
    """The normalisation defect in cprime_analysis.Bench.ensemble_term."""
    buggy = B.ensemble_term(0.0)
    fixed = B.fixed_ensemble(0.0)
    stored = B.model
    curve = [dict(B.metrics(B.score(t, ALPHA_HIST, None)), tau=t) for t in TAUS]
    rows = []
    for ea in (0.0, 0.25, 0.5, 0.75, 1.0):
        for t in TAUS:
            mb = B.metrics(B.score(t, ALPHA_HIST, B.ensemble_term(ea)))
            mf = B.metrics(B.score(t, ALPHA_HIST, B.fixed_ensemble(ea)))
            rows.append({"ensemble_alpha": ea, "tau": t,
                         "buggy_R": mb["R"], "buggy_mR": mb["mR"],
                         "buggy_pareto": CPA.pareto_gap(curve, mb["R"], mb["mR"]),
                         "fixed_R": mf["R"], "fixed_mR": mf["mR"],
                         "fixed_pareto": CPA.pareto_gap(curve, mf["R"], mf["mR"])})
    return {
        "defect": "Bench.ensemble_term standardises AFTER slicing to 50 fg columns; "
                  "evals.py::_normalize_eval_logits standardises over all 51 columns "
                  "(background included) and slices afterwards.",
        "max_abs_diff_buggy_vs_stored": float((buggy - stored).abs().max()),
        "mean_abs_diff_buggy_vs_stored": float((buggy - stored).abs().mean()),
        "max_abs_diff_fixed_vs_stored": float((fixed - stored).abs().max()),
        "identity_proof": "fixed_ensemble(0.0) reproduces the cache's stored model_logits; "
                          "ensemble_term(0.0) does not.",
        "affected_arms": ["arm_E_ensemble_alpha", "criterion5_heldout",
                          "criterion3_predicate_decomposition"],
        "unaffected_arms": ["arm_A_prior_only", "arm_B_model", "arm_D_alpha_sweep",
                            "nulls", "complementarity", "Q4_entropy_strata",
                            "p7_reproduction_gate"],
        "table": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p10_model_recalibration/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--scheme", default="raw50", choices=["raw50", "eval48"])
    ap.add_argument("--out", default="runs/p12_cprime_mechanism/mechanism.json")
    ap.add_argument("--null_seeds", type=int, default=5)
    ap.add_argument("--select_seed", type=int, default=20260901)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=4242)
    ap.add_argument("--assert-fix", dest="assert_fix", action="store_true", default=True)
    args = ap.parse_args(argv)

    _log("=" * 88)
    _log("C' MECHANISM ANALYSIS -- CPU only, cache read-only")
    _log("=" * 88)
    B = Mech(args.dump, args.prior, args.scheme)
    _log(f"  scheme={args.scheme}  images={B.n_images}  pair rows={B.model.shape[0]}  "
         f"GT rows={B.n_gt}  classes={B.n_classes}")

    if args.assert_fix:
        e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
        assert e < 1e-4, f"fixed_ensemble(0.0) must reproduce stored model_logits, got {e:.3e}"
        _log(f"  [gate] fixed_ensemble(0.0) vs stored model term: max|diff|={e:.3e}  OK")

    curve = [dict(B.metrics(B.score(t, ALPHA_HIST, None)), tau=t) for t in TAUS]
    res: Dict[str, Any] = {
        "tool": "cprime_mechanism", "dump": args.dump, "prior": args.prior,
        "scheme": args.scheme, "alpha": ALPHA_HIST, "operating_taus": OP_TAUS,
        "n_images": B.n_images, "n_gt_rows": B.n_gt, "n_classes": B.n_classes,
        "reconciliation": reconcile(B),
        "prior_frontier": curve,
        "by_tau": {},
    }
    _log(f"\n  [reconcile] max|buggy-stored|={res['reconciliation']['max_abs_diff_buggy_vs_stored']:.3e}  "
         f"max|fixed-stored|={res['reconciliation']['max_abs_diff_fixed_vs_stored']:.3e}")

    for tau in OP_TAUS:
        _log(f"\n{'-'*88}\n  OPERATING POINT tau={tau}, alpha={ALPHA_HIST}\n{'-'*88}")
        A = analysis_A_flips(B, tau)
        _log(f"  A  flips: rescued {A['rescued_wrong_to_right']}  destroyed "
             f"{A['destroyed_right_to_wrong']}  net {A['net_flips']:+d}  "
             f"changed {A['argmax_changed']} ({A['argmax_changed_frac']*100:.2f}%)")
        Bk = analysis_B_ranks(B, tau)
        _log(f"  B  mean GT rank {Bk['prior']['mean']:.3f} -> {Bk['combined']['mean']:.3f}   "
             f"MRR {Bk['prior']['MRR']:.4f} -> {Bk['combined']['MRR']:.4f}   "
             f"R@5 {Bk['prior']['recall@5']*100:.2f} -> {Bk['combined']['recall@5']*100:.2f}")
        C = analysis_C_margins(B, tau)
        _log(f"  C  prior margin: changed rows {C['mean_prior_margin_changed']:.3f} vs "
             f"unchanged {C['mean_prior_margin_unchanged']:.3f}")
        DE = analysis_DE_buckets_predicates(B, tau)
        _log("  D  net flips by GT bucket: " + "  ".join(
            f"{b}={DE['buckets'][b]['net_flips']:+d}" for b in ("head", "body", "tail")))
        F = analysis_F_entropy(B, tau)
        G = analysis_G_rescue(B, tau)
        H = analysis_H_beneficial_vs_harmful(B, tau)
        _log(f"  G  rescue rate by prior GT rank: " + "  ".join(
            f"{r['prior_gt_rank']}:{r['rescue_rate']*100:.1f}%" for r in G["rescue_by_prior_gt_rank"]))
        J = analysis_J_stability(B, tau, args.n_boot, args.boot_seed)
        _log(f"  J  bootstrap over images (n={args.n_boot}): "
             f"dR {J['dR_points']['point']:+.3f} [{J['dR_points']['ci2.5']:+.3f},{J['dR_points']['ci97.5']:+.3f}] "
             f"pos {J['dR_points']['frac_draws_positive']*100:.1f}%  |  "
             f"dmR {J['dmR_points']['point']:+.3f} [{J['dmR_points']['ci2.5']:+.3f},{J['dmR_points']['ci97.5']:+.3f}] "
             f"pos {J['dmR_points']['frac_draws_positive']*100:.1f}%")
        _log(f"     dmR {J['leave_best_classes_out']['dmR_full']:+.3f} -> "
             f"{J['leave_best_classes_out']['dmR_without_best']:+.3f} without "
             f"'{J['leave_best_classes_out']['best_class']}' -> "
             f"{J['leave_best_classes_out']['dmR_without_best_two']:+.3f} without it and "
             f"'{J['leave_best_classes_out']['second_class']}'")
        I = controls(B, tau, args.null_seeds, curve)
        _log(f"  I  real net {I['real']['net_flips']:+d} / dmR {I['real']['dmR_points']:+.3f}  |  "
             f"N1 net {I['null_N1']['mean_net_flips']:+.1f} / dmR {I['null_N1']['mean_dmR_points']:+.3f}  |  "
             f"gauss net {I['null_gaussian_matched_scale']['mean_net_flips']:+.1f} / dmR "
             f"{I['null_gaussian_matched_scale']['mean_dmR_points']:+.3f}")
        res["by_tau"][str(tau)] = {"A_flips": A, "B_ranks": Bk, "C_margins": C,
                                   "DE_buckets_predicates": DE, "F_entropy": F,
                                   "G_rescue": G, "H_beneficial_vs_harmful": H,
                                   "I_controls": I, "J_stability": J}

    _log(f"\n{'-'*88}\n  HELD-OUT SELECTION (image-level 50/50)\n{'-'*88}")
    ho = heldout(B, args.null_seeds, args.select_seed)
    _log(f"  selected on A: scale={ho['selected_scale']} tau={ho['selected_tau']} "
         f"(gap {ho['halfA_pareto_dmR_points']:+.2f})")
    _log(f"  read on B    : R@50 {ho['heldout_B_R']*100:.2f}  mR@50 {ho['heldout_B_mR']*100:.2f}  "
         f"gap {ho['heldout_B_pareto_dmR_points']:+.2f}  null {ho['heldout_B_null_mean']:+.2f}  "
         f"margin {ho['heldout_B_margin_over_null']:+.2f}")
    res["heldout"] = ho

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
