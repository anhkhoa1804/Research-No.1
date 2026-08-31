#!/usr/bin/env python
"""Experiment C' -- the whole CPU analysis family, from one GPU cache.

Pre-registration: docs/MODEL_RECALIBRATION_C_PREREGISTRATION.md
Cache schema:     runs/p10_model_recalibration/cache_schema.md

Question: does the trained model contain complementary predicate information
beyond the long-tail decision frontier the leak-free frequency prior already
reaches on its own?

The primary instrument is NOT a metric delta. tau -- a zero-information scalar
-- moves mR@50 by +4.03 points (runs/p7_prior_temperature_sweep/), so any
criterion satisfiable by a metric delta is satisfiable without information. The
instrument is the PARETO GAP: an arm's mR@50 minus the prior-only tau
frontier's mR@50 interpolated at the SAME R@50, referenced to a matched null
that goes through the identical pipeline.

Arms (registration section 3):
  A   prior only, over the tau grid                     -- the frontier
  B   model + prior, over the tau grid, alpha=3.75      -- the C question
  B'  classifier branch + prior                         -- the branch the
                                                           historical protocol
                                                           multiplies by zero
  C   model only (alpha=0), text and classifier separately
  D   model + prior over an alpha grid

Nulls (registration section 5): N1 pair-shuffled within image, N2 shuffled
across the split. 5 seeds each, rescored under every tau and alpha the real arm
uses. A fitted real arm is never compared against an unfitted null.

Denominator (registration section 6): ONE predicate scheme for every arm.
Primary raw50 (literature-standard VG150); eval48 reported as a robustness
column. Possible only because the cache stores RAW GT plus the alias map.

CPU only. Reads the cache read-only.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

TAUS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.75, 7.5]
ALPHA_HIST = 3.75
K_TOPK = 5


def _log(m: str = "") -> None:
    print(m, flush=True)


# ---------------------------------------------------------------- data model
class Bench:
    """Flattened view of the cache, with GT triplets mapped onto pair rows."""

    def __init__(self, dump_path: Path, scheme: str = "raw50"):
        d = torch.load(dump_path, weights_only=False)
        self.meta = d
        self.scheme = scheme
        pv = [str(x).strip().lower() for x in d["pred_vocab"]]
        bg = set(int(i) for i in d.get("background_predicate_indices", []))
        self.fg_cols = [i for i in range(len(pv)) if i not in bg]
        self.fg_names_raw = [pv[i] for i in self.fg_cols]

        amap = {str(k): str(v) for k, v in d.get("predicate_alias_map", {}).items()}
        if scheme == "eval48":
            self.col_label = [amap.get(n, n) for n in self.fg_names_raw]
        elif scheme == "raw50":
            self.col_label = list(self.fg_names_raw)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        self.classes = sorted(set(self.col_label))

        model, prior, text, cls = [], [], [], []
        gt_row, gt_lab, img_of_row = [], [], []
        self.n_images = len(d["image_id"])
        off = 0
        self.pairs_per_image: List[int] = []
        for i in range(self.n_images):
            n = int(d["pairs"][i].shape[0])
            self.pairs_per_image.append(n)
            model.append(d["model_logits"][i][:, self.fg_cols])
            prior.append(d["prior_rows"][i][:, self.fg_cols])
            text.append(d["text_logits"][i][:, self.fg_cols])
            cls.append(d["cls_logits"][i][:, self.fg_cols])
            img_of_row.extend([i] * n)
            index = {(int(a), int(b)): j for j, (a, b) in enumerate(d["pairs"][i].tolist())}
            for si, oi, p in zip(d["gt_subj_idx"][i], d["gt_obj_idx"][i], d["gt_pred"][i]):
                j = index.get((int(si), int(oi)))
                if j is None:
                    continue
                lab = str(p).strip().lower()
                lab = amap.get(lab, lab) if scheme == "eval48" else lab
                if lab not in set(self.classes):
                    continue
                gt_row.append(off + j)
                gt_lab.append(lab)
            off += n

        self.model = torch.cat(model, 0)
        self.prior = torch.cat(prior, 0)
        self.text = torch.cat(text, 0)
        self.cls = torch.cat(cls, 0)
        self.img_of_row = torch.tensor(img_of_row, dtype=torch.long)
        self.gt_row = torch.tensor(gt_row, dtype=torch.long)
        lab_to_id = {c: i for i, c in enumerate(self.classes)}
        self.gt_y = torch.tensor([lab_to_id[l] for l in gt_lab], dtype=torch.long)
        self.col_to_class = torch.tensor([lab_to_id[c] for c in self.col_label], dtype=torch.long)
        self.n_classes = len(self.classes)

        # log P(p): the prior file's own class marginal, recovered from the
        # cache as the row every unseen pair falls back to. Column-space.
        self.log_marginal = self._recover_marginal()

        # buckets from the evaluated split's GT counts. Identical for every
        # arm because GT is fixed, so cross-arm comparisons are unaffected;
        # they are NOT comparable across different N.
        cnt = torch.bincount(self.gt_y, minlength=self.n_classes)
        order = torch.argsort(cnt, descending=True)
        self.head = set(order[:15].tolist())
        self.body = set(order[15:35].tolist())
        self.bucket_of = {i: ("head" if i in self.head else "body" if i in self.body else "tail")
                          for i in range(self.n_classes)}

        # prior entropy per GT row, for the uncertainty strata
        pp = torch.softmax(self.prior[self.gt_row], dim=-1)
        self.prior_entropy = -(pp * torch.log(pp.clamp_min(1e-12))).sum(-1)
        q = torch.quantile(self.prior_entropy, torch.tensor([1 / 3, 2 / 3]))
        self.ent_stratum = torch.bucketize(self.prior_entropy, q)  # 0 low, 1 mid, 2 high

    def _recover_marginal(self) -> torch.Tensor:
        """log P(p), in column space.

        _frequency_bias_for_pairs falls back to the prior file's `global` row
        for any pair it has no statistics for, so the modal prior row across
        the split IS that marginal. Recovered rather than re-read so the CPU
        tau is provably the same vector the in-evaluator tau would use.
        """
        rounded = (self.prior * 1e4).round().to(torch.int64)
        keys = Counter(tuple(r) for r in rounded.tolist())
        modal, count = keys.most_common(1)[0]
        self.marginal_support = count
        return torch.tensor(modal, dtype=torch.float32) / 1e4

    # ------------------------------------------------------------ scoring
    def score(self, tau: float, alpha: float, model_term: Optional[torch.Tensor],
              model_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """score = model_term + alpha * (prior - tau * log P(p)).

        `model_rows` optionally reindexes the model term (used by the nulls),
        leaving the prior term attached to its own pair.
        """
        s = alpha * (self.prior - tau * self.log_marginal.view(1, -1)) if alpha != 0.0 \
            else torch.zeros_like(self.prior)
        if model_term is not None:
            m = model_term if model_rows is None else model_term[model_rows]
            s = s + m
        return s

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        """Predicted CLASS id per GT row (column argmax mapped through scheme)."""
        return self.col_to_class[s[self.gt_row].argmax(-1)]

    def metrics(self, s: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        pred = self.predict(s)
        y = self.gt_y
        if mask is not None:
            pred, y = pred[mask], y[mask]
        hit = (pred == y)
        ph, pg = Counter(), Counter()
        for yy, hh in zip(y.tolist(), hit.tolist()):
            pg[yy] += 1
            ph[yy] += int(hh)
        rec = {k: ph[k] / pg[k] for k in pg}
        mR = sum(rec.values()) / max(1, len(rec))
        out = {"R": float(hit.float().mean()) if hit.numel() else 0.0,
               "mR": float(mR), "n": int(y.numel()), "n_classes": len(pg)}
        for b in ("head", "body", "tail"):
            ks = [k for k in pg if self.bucket_of[k] == b]
            out[f"{b}_mR"] = float(sum(rec[k] for k in ks) / max(1, len(ks))) if ks else 0.0
        return out

    def gt_rank(self, s: torch.Tensor) -> torch.Tensor:
        """1-based rank of the GT class in the descending column order."""
        rows = s[self.gt_row]
        gt_col_score = torch.full((rows.shape[0],), -1e30)
        for c in range(self.n_classes):
            cols = (self.col_to_class == c).nonzero().squeeze(1)
            sel = (self.gt_y == c)
            if sel.any() and cols.numel():
                gt_col_score[sel] = rows[sel][:, cols].max(-1).values
        return (rows > gt_col_score.unsqueeze(1)).sum(-1) + 1


# ------------------------------------------------------------------ pareto
def pareto_gap(curve: List[Dict[str, float]], R: float, mR: float) -> Optional[float]:
    """mR minus the prior-only frontier's mR at the SAME R@50, in mR points.

    Returns None only for a genuine extrapolation (see below).

    The frontier is traced by tau and is monotone: raising tau trades R@50 away
    for mR@50. Three regimes:

    * R inside the frontier's R range -> linear interpolation between the two
      bracketing tau points. The ordinary case.
    * R ABOVE the frontier's maximum R (which is the tau=0 point) -> the
      frontier cannot reach that R at all. If the arm also beats the frontier's
      mR there, it strictly dominates every point on the curve, and the gap
      against the tau=0 point is a CONSERVATIVE LOWER BOUND on its advantage.
      Returning None here would record a two-axis win as "indeterminate", which
      is exactly backwards.
    * R BELOW the frontier's minimum R -> the frontier is not defined that far
      out and extending it would be extrapolation. None.
    """
    pts = sorted((c["R"], c["mR"]) for c in curve)
    if not pts:
        return None
    if R > pts[-1][0] + 1e-12:
        return (mR - pts[-1][1]) * 100.0        # conservative lower bound
    if R < pts[0][0] - 1e-12:
        return None                             # true extrapolation
    for (r0, m0), (r1, m1) in zip(pts, pts[1:]):
        if r0 - 1e-12 <= R <= r1 + 1e-12:
            if abs(r1 - r0) < 1e-12:
                return (mR - max(m0, m1)) * 100.0
            w = (R - r0) / (r1 - r0)
            return (mR - (m0 + w * (m1 - m0))) * 100.0
    return None


def _sd(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ------------------------------------------------------------------- nulls
def null_rows(B: Bench, kind: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(B.model.shape[0])
    if kind == "N1":                      # permute within each image
        out = idx.clone()
        off = 0
        for n in B.pairs_per_image:
            if n > 1:
                out[off:off + n] = idx[off:off + n][torch.randperm(n, generator=g)]
            off += n
        return out
    if kind == "N2":                      # permute across the whole split
        return idx[torch.randperm(idx.numel(), generator=g)]
    raise ValueError(kind)


# -------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--schemes", type=str, default="raw50,eval48")
    ap.add_argument("--null_seeds", type=int, default=5)
    ap.add_argument("--select_seed", type=int, default=0,
                    help="image-level 50/50 split seed for the held-out point estimate")
    args = ap.parse_args(argv)

    res: Dict[str, Any] = {
        "tool": "cprime_analysis",
        "dump": str(args.dump),
        "preregistration": "docs/MODEL_RECALIBRATION_C_PREREGISTRATION.md",
        "taus": TAUS, "alphas": ALPHAS, "alpha_historical": ALPHA_HIST,
        "null_seeds": args.null_seeds,
        "per_scheme": {},
    }

    for scheme in [s for s in args.schemes.split(",") if s.strip()]:
        _log("\n" + "=" * 96)
        _log(f"SCHEME = {scheme}")
        _log("=" * 96)
        B = Bench(args.dump, scheme)
        S: Dict[str, Any] = {
            "n_classes": B.n_classes, "n_gt": int(B.gt_y.numel()),
            "n_images": B.n_images, "n_pairs": int(B.model.shape[0]),
            "marginal_support_rows": B.marginal_support,
        }
        _log(f"  classes={B.n_classes}  GT triplets={int(B.gt_y.numel())}  "
             f"pairs={int(B.model.shape[0])}  images={B.n_images}")
        _log(f"  log P(p) recovered from {B.marginal_support} identical fallback rows")

        # ---- ARM A: the frontier
        _log(f"\n  ARM A  prior only")
        _log(f"  {'tau':>7} {'R@50':>8} {'mR@50':>8} {'head':>7} {'body':>7} {'tail':>7}")
        curve = []
        for t in TAUS:
            m = B.metrics(B.score(t, ALPHA_HIST, None))
            m["tau"] = t
            curve.append(m)
            _log(f"  {t:>7} {m['R']*100:>8.2f} {m['mR']*100:>8.2f} {m['head_mR']*100:>7.2f}"
                 f" {m['body_mR']*100:>7.2f} {m['tail_mR']*100:>7.2f}")
        S["arm_A_prior_only"] = curve

        # ---- ARMS B / B'
        model_terms = {"B_model": B.model, "Bp_classifier": B.cls, "B_text": B.text}
        for name, mt in model_terms.items():
            _log(f"\n  ARM {name}  (+ prior, alpha={ALPHA_HIST})")
            _log(f"  {'tau':>7} {'R@50':>8} {'mR@50':>8} {'tail':>7} {'dPareto':>9}")
            rows = []
            for t in TAUS:
                m = B.metrics(B.score(t, ALPHA_HIST, mt))
                m["tau"] = t
                m["pareto_dmR_points"] = pareto_gap(curve, m["R"], m["mR"])
                rows.append(m)
                g = m["pareto_dmR_points"]
                _log(f"  {t:>7} {m['R']*100:>8.2f} {m['mR']*100:>8.2f} {m['tail_mR']*100:>7.2f}"
                     f" {'   n/a' if g is None else format(g, '+9.2f')}")
            S[f"arm_{name}"] = rows

        # ---- ARM C: model only
        _log(f"\n  ARM C  model only (alpha=0)")
        conly = {}
        for name, mt in model_terms.items():
            m = B.metrics(B.score(0.0, 0.0, mt))
            conly[name] = m
            _log(f"    {name:<16} R@50 {m['R']*100:6.2f}  mR@50 {m['mR']*100:6.2f}  "
                 f"tail {m['tail_mR']*100:5.2f}")
        S["arm_C_model_only"] = conly

        # ---- ARM D: alpha sweep
        _log(f"\n  ARM D  model + prior over alpha (tau=0 and tau=0.1)")
        _log(f"  {'alpha':>7} {'tau':>6} {'R@50':>8} {'mR@50':>8} {'dPareto':>9}")
        drows = []
        for a in ALPHAS:
            for t in (0.0, 0.1):
                m = B.metrics(B.score(t, a, B.model))
                m["alpha"], m["tau"] = a, t
                m["pareto_dmR_points"] = pareto_gap(curve, m["R"], m["mR"])
                drows.append(m)
                g = m["pareto_dmR_points"]
                _log(f"  {a:>7} {t:>6} {m['R']*100:>8.2f} {m['mR']*100:>8.2f}"
                     f" {'   n/a' if g is None else format(g, '+9.2f')}")
        S["arm_D_alpha_sweep"] = drows

        # ---- NULLS
        _log(f"\n  NULLS  ({args.null_seeds} seeds each, rescored at every tau)")
        nulls: Dict[str, Any] = {}
        for kind in ("N1", "N2"):
            per_tau: Dict[str, Any] = {}
            for t in TAUS:
                gaps, Rs, mRs = [], [], []
                for s in range(args.null_seeds):
                    idx = null_rows(B, kind, 7000 + 137 * s)
                    m = B.metrics(B.score(t, ALPHA_HIST, B.model, model_rows=idx))
                    g = pareto_gap(curve, m["R"], m["mR"])
                    if g is not None:
                        gaps.append(g)
                    Rs.append(m["R"]); mRs.append(m["mR"])
                per_tau[str(t)] = {
                    "mean_R": sum(Rs) / len(Rs), "mean_mR": sum(mRs) / len(mRs),
                    "gaps": gaps,
                    "mean_pareto_dmR_points": (sum(gaps) / len(gaps)) if gaps else None,
                    "sd_pareto_dmR_points": _sd(gaps),
                }
            nulls[kind] = per_tau
            gg = [v["mean_pareto_dmR_points"] for v in per_tau.values()
                  if v["mean_pareto_dmR_points"] is not None]
            _log(f"    {kind}: mean Pareto gap over tau grid "
                 f"{(sum(gg)/len(gg)) if gg else float('nan'):+.2f} points")
        S["nulls"] = nulls

        # ---- COMPLEMENTARITY / rescue / ranks / strata
        _log(f"\n  COMPLEMENTARITY  (tau=0, alpha={ALPHA_HIST})")
        s_prior = B.score(0.0, ALPHA_HIST, None)
        s_comb = B.score(0.0, ALPHA_HIST, B.model)
        s_modl = B.score(0.0, 0.0, B.model)
        pr_pred, cb_pred = B.predict(s_prior), B.predict(s_comb)
        y = B.gt_y
        pr_ok, cb_ok = (pr_pred == y), (cb_pred == y)
        rank_prior, rank_model, rank_comb = B.gt_rank(s_prior), B.gt_rank(s_modl), B.gt_rank(s_comb)

        top5 = s_prior[B.gt_row].topk(K_TOPK, dim=-1).indices
        in_top5 = torch.zeros_like(y, dtype=torch.bool)
        for j in range(K_TOPK):
            in_top5 |= (B.col_to_class[top5[:, j]] == y)

        comp = {
            "prior_top5_coverage": float(in_top5.float().mean()),
            "prior_top1_accuracy": float(pr_ok.float().mean()),
            "combined_top1_accuracy": float(cb_ok.float().mean()),
            "Q2_rescued_wrong_to_right": int((~pr_ok & cb_ok).sum()),
            "Q1_destroyed_right_to_wrong": int((pr_ok & ~cb_ok).sum()),
            "net_flips": int((~pr_ok & cb_ok).sum()) - int((pr_ok & ~cb_ok).sum()),
            "rescue_rate_of_prior_errors": float((~pr_ok & cb_ok).sum() / max(1, int((~pr_ok).sum()))),
            "destruction_rate_of_prior_hits": float((pr_ok & ~cb_ok).sum() / max(1, int(pr_ok.sum()))),
            "mean_gt_rank_prior": float(rank_prior.float().mean()),
            "mean_gt_rank_model_only": float(rank_model.float().mean()),
            "mean_gt_rank_combined": float(rank_comb.float().mean()),
            "median_gt_rank_prior": float(rank_prior.float().median()),
            "median_gt_rank_combined": float(rank_comb.float().median()),
            "Q3_rank_improved": int((rank_comb < rank_prior).sum()),
            "Q3_rank_worsened": int((rank_comb > rank_prior).sum()),
            "Q3_within_prior_top5_rank_improved": int(((rank_comb < rank_prior) & in_top5).sum()),
            "Q3_within_prior_top5_rank_worsened": int(((rank_comb > rank_prior) & in_top5).sum()),
        }
        for k, v in comp.items():
            _log(f"    {k:<38} {v}")
        S["complementarity"] = comp

        # Q4: entropy strata; Q5: buckets -- Pareto gap computed within stratum
        _log(f"\n  Q4  prior-entropy strata (Pareto gap of arm B at tau=0.1, within stratum)")
        strata = {}
        for si, nm in ((0, "low_entropy"), (1, "mid_entropy"), (2, "high_entropy")):
            mask = (B.ent_stratum == si)
            sub_curve = []
            for t in TAUS:
                m = B.metrics(B.score(t, ALPHA_HIST, None), mask=mask)
                m["tau"] = t
                sub_curve.append(m)
            mB = B.metrics(B.score(0.1, ALPHA_HIST, B.model), mask=mask)
            g = pareto_gap(sub_curve, mB["R"], mB["mR"])
            strata[nm] = {"n": int(mask.sum()), "prior_tau0_R": sub_curve[0]["R"],
                          "prior_tau0_mR": sub_curve[0]["mR"], "modelB_tau0.1_R": mB["R"],
                          "modelB_tau0.1_mR": mB["mR"], "pareto_dmR_points": g,
                          "rescue_rate": float((~pr_ok & cb_ok)[mask].sum()
                                               / max(1, int((~pr_ok)[mask].sum())))}
            _log(f"    {nm:<14} n={int(mask.sum()):>6}  prior mR {sub_curve[0]['mR']*100:6.2f}"
                 f"  B mR {mB['mR']*100:6.2f}  dPareto "
                 f"{'n/a' if g is None else format(g, '+.2f')}")
        S["Q4_entropy_strata"] = strata
        res["per_scheme"][scheme] = S

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
