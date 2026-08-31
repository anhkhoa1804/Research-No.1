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

    def __init__(self, dump_path: Path, scheme: str = "raw50",
                 prior_path: Optional[Path] = None):
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

        # log P(p): the prior file's OWN `global_log_probs`, remapped onto the
        # cache's column order. Read from the file, never inferred.
        self.log_marginal = self._load_marginal(prior_path)

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

    def _load_marginal(self, prior_path: Optional[Path]) -> torch.Tensor:
        """log P(p) in column space, from the prior file's `global_log_probs`.

        An earlier revision INFERRED this as the modal prior row, on the theory
        that unseen pairs fall back to the global marginal. That is false: they
        fall back to `default_log_prob`, a UNIFORM row (measured: -3.932 in
        every column, 616 rows). Subtracting a constant vector cannot move an
        argmax, so the inferred tau was a silent no-op and the prior-only
        frontier came out almost flat -- tau=0.5 gave mR@50 23.39 where
        runs/p7_prior_temperature_sweep/ measures 38.15. Caught by that
        cross-check, which is why it is now a hard gate in this tool.
        """
        if prior_path is None:
            raise ValueError("--prior is required: log P(p) is read, never inferred")
        raw = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        src = [str(x).strip().lower() for x in raw.get("predicate_vocab", [])]
        g = raw.get("global_log_probs")
        if not isinstance(g, list) or len(g) != len(src):
            raise ValueError(f"{prior_path}: unusable global_log_probs")
        default = float(raw.get("default_log_prob", -20.0))
        idx = {p: i for i, p in enumerate(src)}
        self.marginal_source = str(prior_path)
        self.marginal_missing = [n for n in self.fg_names_raw if n not in idx]
        return torch.tensor([float(g[idx[n]]) if n in idx else default
                             for n in self.fg_names_raw], dtype=torch.float32)

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        """`_normalize_eval_logits` from evals.py: per-row standardisation."""
        m = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True).clamp_min(1e-4)
        return (x - m) / sd

    def ensemble_term(self, ea: float) -> torch.Tensor:
        """The model term the evaluator WOULD build at this ensemble_alpha.

        The historical protocol pins ea = 0.0, which makes this the text branch
        alone. Sweeping it is the counterfactual that separates "the model has
        no information" from "the protocol discards the model's information".
        Both branches are normalised, exactly as _relation_predicate_logits
        does -- comparing a normalised branch against a RAW one would confound
        the branch with its scale, and arm D shows scale matters a great deal.
        """
        ct = float(self.meta.get("classifier_temperature", 1.0)) or 1.0
        tt = float(self.meta.get("text_temperature", 1.0)) or 1.0
        return ea * (self._norm(self.cls) / ct) + (1.0 - ea) * (self._norm(self.text) / tt)

    def per_predicate_recall(self, s: torch.Tensor) -> Dict[int, Tuple[int, int]]:
        pred, y = self.predict(s), self.gt_y
        hit = (pred == y)
        ph, pg = Counter(), Counter()
        for yy, hh in zip(y.tolist(), hit.tolist()):
            pg[yy] += 1
            ph[yy] += int(hh)
        return {k: (ph[k], pg[k]) for k in pg}

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
    ap.add_argument("--prior", type=Path, required=True,
                    help="the SAME frequency prior file the GPU pass used; "
                         "log P(p) is read from its global_log_probs")
    ap.add_argument("--schemes", type=str, default="raw50,eval48")
    ap.add_argument("--null_seeds", type=int, default=5)
    ap.add_argument("--select_seed", type=int, default=0,
                    help="image-level 50/50 split seed for the held-out point estimate")
    args = ap.parse_args(argv)

    res: Dict[str, Any] = {
        "tool": "cprime_analysis",
        "dump": str(args.dump), "prior": str(args.prior),
        "preregistration": "docs/MODEL_RECALIBRATION_C_PREREGISTRATION.md",
        "taus": TAUS, "alphas": ALPHAS, "alpha_historical": ALPHA_HIST,
        "null_seeds": args.null_seeds,
        "per_scheme": {},
    }

    for scheme in [s for s in args.schemes.split(",") if s.strip()]:
        _log("\n" + "=" * 96)
        _log(f"SCHEME = {scheme}")
        _log("=" * 96)
        B = Bench(args.dump, scheme, prior_path=args.prior)
        S: Dict[str, Any] = {
            "n_classes": B.n_classes, "n_gt": int(B.gt_y.numel()),
            "n_images": B.n_images, "n_pairs": int(B.model.shape[0]),
            "marginal_source": B.marginal_source,
            "marginal_missing_predicates": B.marginal_missing,
        }
        _log(f"  classes={B.n_classes}  GT triplets={int(B.gt_y.numel())}  "
             f"pairs={int(B.model.shape[0])}  images={B.n_images}")
        _log(f"  log P(p) read from {B.marginal_source}"
             f"{'' if not B.marginal_missing else '  MISSING: ' + str(B.marginal_missing)}")

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

        # VALIDATION GATE. The prior-only frontier is computed here from the
        # cached prior rows; runs/p7_prior_temperature_sweep/ computed it
        # independently, from the prior FILE, with a different tool. They must
        # agree. This gate is what caught the inferred-marginal defect: a
        # silent no-op tau reproduced tau=0 exactly and diverged everywhere
        # else, which a tau=0-only check would have passed.
        if scheme == "raw50":
            ref = {0.0: (66.80, 21.98), 0.05: (66.47, 24.22), 0.1: (66.16, 26.00),
                   0.2: (61.95, 28.42), 0.5: (44.64, 38.15)}
            devs = []
            for row in curve:
                if row["tau"] in ref:
                    rR, rM = ref[row["tau"]]
                    devs.append((row["tau"], row["R"] * 100 - rR, row["mR"] * 100 - rM))
            worst = max((max(abs(a), abs(b)) for _, a, b in devs), default=0.0)
            S["p7_reproduction_gate"] = {
                "reference": "runs/p7_prior_temperature_sweep/decision_rule_sweep.json",
                "deviations_points": [{"tau": t, "dR": a, "dmR": b} for t, a, b in devs],
                "worst_abs_deviation_points": worst,
                "passes": bool(worst < 1.0),
            }
            _log(f"\n  GATE  prior-only frontier vs p7 (independent tool): "
                 f"worst |deviation| {worst:.2f} points -> "
                 f"{'PASS' if worst < 1.0 else 'FAIL'}")
            for t, a, b in devs:
                _log(f"        tau={t:<6} dR {a:+6.2f}  dmR {b:+6.2f}")

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

        # ---- ARM E: the ensemble_alpha counterfactual (Q7 / decision B)
        EAS = [0.0, 0.25, 0.5, 0.75, 1.0]
        _log(f"\n  ARM E  ensemble_alpha counterfactual  (both branches normalised, "
             f"+ prior alpha={ALPHA_HIST})")
        _log(f"  {'ens_a':>6} {'tau':>6} {'R@50':>8} {'mR@50':>8} {'tail':>7} {'dPareto':>9} {'nullN1':>8} {'margin':>8}")
        erows = []
        for ea in EAS:
            mt = B.ensemble_term(ea)
            for t in TAUS:
                m = B.metrics(B.score(t, ALPHA_HIST, mt))
                m["ensemble_alpha"], m["tau"] = ea, t
                m["pareto_dmR_points"] = pareto_gap(curve, m["R"], m["mR"])
                gaps = []
                for sd_ in range(args.null_seeds):
                    idx = null_rows(B, "N1", 7000 + 137 * sd_)
                    mn = B.metrics(B.score(t, ALPHA_HIST, mt, model_rows=idx))
                    gg = pareto_gap(curve, mn["R"], mn["mR"])
                    if gg is not None:
                        gaps.append(gg)
                m["null_N1_mean"] = (sum(gaps) / len(gaps)) if gaps else None
                m["null_N1_sd"] = _sd(gaps)
                if m["pareto_dmR_points"] is not None and m["null_N1_mean"] is not None:
                    m["margin_over_null_points"] = m["pareto_dmR_points"] - m["null_N1_mean"]
                    m["null_2sd"] = 2.0 * m["null_N1_sd"] if m["null_N1_sd"] else 0.0
                    m["passes_null_gate"] = bool(m["pareto_dmR_points"] > 0
                                                 and m["margin_over_null_points"] >= m["null_2sd"])
                else:
                    m["margin_over_null_points"] = None
                    m["passes_null_gate"] = False
                erows.append(m)
                fmt = lambda x: "     n/a" if x is None else f"{x:+8.2f}"
                _log(f"  {ea:>6} {t:>6} {m['R']*100:>8.2f} {m['mR']*100:>8.2f} "
                     f"{m['tail_mR']*100:>7.2f} {fmt(m['pareto_dmR_points'])} "
                     f"{fmt(m['null_N1_mean'])} {fmt(m['margin_over_null_points'])}")
        S["arm_E_ensemble_alpha"] = erows

        # ---- CRITERION 5: held-out selection, image-level 50/50 split.
        # tau and ensemble_alpha are chosen on half A and READ on half B, so no
        # reported number is selected on the data it is reported from.
        g = torch.Generator().manual_seed(args.select_seed)
        perm = torch.randperm(B.n_images, generator=g)
        halfA = set(perm[: B.n_images // 2].tolist())
        row_img = B.img_of_row[B.gt_row]
        inA = torch.tensor([int(i) in halfA for i in row_img.tolist()])
        sel_out = {}
        for armname, termfn in (("E_ensemble", B.ensemble_term),):
            best, bestgap = None, -1e9
            curveA = [dict(B.metrics(B.score(t, ALPHA_HIST, None), mask=inA), tau=t) for t in TAUS]
            curveB = [dict(B.metrics(B.score(t, ALPHA_HIST, None), mask=~inA), tau=t) for t in TAUS]
            for ea in EAS:
                mt = termfn(ea)
                for t in TAUS:
                    mA = B.metrics(B.score(t, ALPHA_HIST, mt), mask=inA)
                    gA = pareto_gap(curveA, mA["R"], mA["mR"])
                    if gA is not None and gA > bestgap:
                        bestgap, best = gA, (ea, t)
            ea, t = best
            mt = termfn(ea)
            mB_ = B.metrics(B.score(t, ALPHA_HIST, mt), mask=~inA)
            gB = pareto_gap(curveB, mB_["R"], mB_["mR"])
            nulls_b = []
            for sd_ in range(args.null_seeds):
                idx = null_rows(B, "N1", 7000 + 137 * sd_)
                mn = B.metrics(B.score(t, ALPHA_HIST, mt, model_rows=idx), mask=~inA)
                gg = pareto_gap(curveB, mn["R"], mn["mR"])
                if gg is not None:
                    nulls_b.append(gg)
            sel_out[armname] = {
                "selected_on": "half A (image-level 50/50, seed %d)" % args.select_seed,
                "selected_ensemble_alpha": ea, "selected_tau": t,
                "halfA_pareto_dmR_points": bestgap,
                "heldout_halfB_R": mB_["R"], "heldout_halfB_mR": mB_["mR"],
                "heldout_halfB_pareto_dmR_points": gB,
                "heldout_null_mean": (sum(nulls_b) / len(nulls_b)) if nulls_b else None,
                "heldout_null_sd": _sd(nulls_b),
            }
            _log(f"\n  CRITERION 5  held-out selection ({armname})")
            _log(f"    selected on half A: ensemble_alpha={ea}, tau={t} (gap {bestgap:+.2f})")
            _log(f"    read on half B    : R@50 {mB_['R']*100:.2f}  mR@50 {mB_['mR']*100:.2f}  "
                 f"gap {'n/a' if gB is None else format(gB, '+.2f')}  "
                 f"null {(sum(nulls_b)/len(nulls_b)) if nulls_b else float('nan'):+.2f}")
        S["criterion5_heldout"] = sel_out

        # ---- CRITERION 3: is the gain carried by one predicate?
        best_e = max((r for r in erows if r["pareto_dmR_points"] is not None),
                     key=lambda r: r["pareto_dmR_points"])
        mt = B.ensemble_term(best_e["ensemble_alpha"])
        base = B.per_predicate_recall(B.score(best_e["tau"], ALPHA_HIST, None))
        arm = B.per_predicate_recall(B.score(best_e["tau"], ALPHA_HIST, mt))
        contrib = []
        for k in base:
            d0 = base[k][0] / base[k][1]
            d1 = arm.get(k, (0, base[k][1]))[0] / base[k][1]
            contrib.append({"class": B.classes[k], "bucket": B.bucket_of[k],
                            "n": base[k][1], "delta_recall": d1 - d0,
                            "contribution_mR_points": (d1 - d0) / len(base) * 100.0})
        contrib.sort(key=lambda r: -abs(r["contribution_mR_points"]))
        pos = sum(c["contribution_mR_points"] for c in contrib if c["contribution_mR_points"] > 0)
        total = sum(c["contribution_mR_points"] for c in contrib)
        top = contrib[0]
        S["criterion3_predicate_decomposition"] = {
            "at": {"ensemble_alpha": best_e["ensemble_alpha"], "tau": best_e["tau"]},
            "total_mR_delta_points": total,
            "largest_single_predicate": top,
            "largest_share_of_positive_gain": (top["contribution_mR_points"] / pos) if pos > 0 else None,
            "top10": contrib[:10],
        }
        _log(f"\n  CRITERION 3  predicate decomposition at ensemble_alpha="
             f"{best_e['ensemble_alpha']}, tau={best_e['tau']}")
        _log(f"    total mR delta {total:+.2f} pts; largest single predicate "
             f"'{top['class']}' ({top['bucket']}, n={top['n']}) "
             f"{top['contribution_mR_points']:+.2f} pts = "
             f"{(top['contribution_mR_points']/pos*100) if pos>0 else float('nan'):.1f}% of positive gain")

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
