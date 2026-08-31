#!/usr/bin/env python
"""Does appearance add anything ON TOP of a calibrated prior? (CPU-only)

WHY THIS TOOL EXISTS
--------------------
`tools/appearance_probe.py` composed appearance with the frequency prior as

    s = log P(p | s, o)  +  lambda * appearance

and swept `lambda` alone. Two facts discovered afterwards make that sweep an
inconclusive test of "does appearance carry convertible signal":

1. `runs/p7_prior_temperature_sweep/` shows a ZERO-information scalar tau,
       log P(p | s, o)  ->  log P(p | s, o) - tau * log P(p),
   moves mR@50 by +4.03 points for -0.64 R@50. The lambda=0 baseline the probe
   measured against is therefore NOT on the achievable frontier -- it is a
   miscalibrated point well inside it.

2. In the lambda-only family, the appearance term is the ONLY thing that can
   flatten the prior's very peaked top-5 ranking (mean entropy 0.557 nats).
   So lambda has to pay for calibration out of the same budget it uses to
   express appearance. The family does not contain "calibrate AND add
   appearance" as a reachable point.

A negative result in a family that cannot express the calibrated operating
point does not license the conclusion "appearance does not convert". This tool
sweeps the 2-D family (tau, lambda) instead, and asks the PARETO question that
`POST_B_PRIOR_CALIBRATION_ANALYSIS.md` section 6.2 requires:

    does the best (tau, lambda>0) point lie strictly OUTSIDE the
    (tau, lambda=0) curve on the (R@50, mR@50) plane, by more than the
    shuffled-appearance null band?

It also varies PCA_DIM, because PCA-48 retains only ~56% of each block's
variance (measured), making "the representation was truncated" a live
alternative to "the representation is uninformative".

WHAT IS HELD FIXED FROM THE PRE-REGISTERED PROBE
------------------------------------------------
Same cache, same image-level 80/20 held-out split, same top-5 candidate set
taken from the RAW (tau=0) prior, same optimiser/epochs, same mR definition
and head/body/tail bucketing. The candidate set is deliberately NOT re-derived
per tau, so that tau and lambda are the only variables and every arm is scored
over an identical set of decisions.

DISCIPLINE
----------
- Validation is NEVER used to select tau, lambda or PCA dim. Selection is on
  the held-out-from-train split, exactly as the probe did.
- The shuffled-appearance arm is refit at every (tau, lambda) it is run at, so
  the null accounts for "adding any scalar-weighted noise term perturbs the
  ranking", not merely for "appearance is absent".
- `tools/appearance_probe.py` is NOT modified. This is a new, additional
  measurement; the pre-registered one stands as recorded.

Pre-registration: docs/APPEARANCE_TAU_INTERACTION_PREREGISTRATION.md
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

K = 5           # candidate depth, unchanged from the probe
EPOCHS = 30     # unchanged from the probe
LR = 0.02       # unchanged
WD = 3e-3       # unchanged


def _log(msg: str = "") -> None:
    print(msg, flush=True)


class Bench:
    """Everything derived from the cache that every arm shares."""

    def __init__(self, cache: Path, pca_dim: int, fix_val_scaling: bool, seed: int = 0):
        torch.manual_seed(seed)
        blob = torch.load(cache, weights_only=False)
        self.TR, self.VA, self.META, self.PV = blob["train"], blob["val"], blob["meta"], blob["PV"]
        self.P = len(self.PV)
        self.pca_dim = int(pca_dim)
        self.fix_val_scaling = bool(fix_val_scaling)

        # Prior rows, mean-centred exactly as the probe does.
        self.Ptr = self.TR["prior"] - self.TR["prior"].mean(-1, keepdim=True)
        self.Pva = self.VA["prior"] - self.VA["prior"].mean(-1, keepdim=True)
        self.Ytr, self.Yva = self.TR["y"], self.VA["y"]

        # Image-level held-out split. Same construction as the probe: the
        # first randperm drawn after manual_seed(seed).
        uimg = torch.unique(self.TR["img"])
        perm = uimg[torch.randperm(len(uimg))]
        hold = set(perm[: max(1, int(0.2 * len(perm)))].tolist())
        self.is_hold = torch.tensor([int(i) in hold for i in self.TR["img"].tolist()])

        # Head/body/tail buckets from TRAIN counts (never from val).
        cnt = torch.bincount(self.Ytr, minlength=self.P).float()
        order = torch.argsort(cnt, descending=True)
        HEAD, BODY = set(order[:15].tolist()), set(order[15:35].tolist())
        self.bucket = {i: ("head" if i in HEAD else "body" if i in BODY else "tail")
                       for i in range(self.P)}

        # log P(p): the class marginal, estimated on the TRAIN-FIT rows only.
        # This is the tau term. Estimating it on train (not val) is what makes
        # tau a leak-free, zero-information transformation here.
        fit_cnt = torch.bincount(self.Ytr[~self.is_hold], minlength=self.P).float()
        self.log_marginal = torch.log((fit_cnt + 1.0) / (fit_cnt.sum() + self.P))

        # Candidate sets: top-K of the RAW prior, identical for every arm.
        self.cand_tr = torch.topk(self.Ptr, k=K, dim=-1).indices
        self.cand_va = torch.topk(self.Pva, k=K, dim=-1).indices
        self.in_tr = (self.cand_tr == self.Ytr.unsqueeze(1)).any(1)
        self.in_va = (self.cand_va == self.Yva.unsqueeze(1)).any(1)

        self._pc: Dict[str, Any] = {}
        self._evr: Dict[str, float] = {}

    # -- features -----------------------------------------------------------
    def blk(self, split: str, nm: str) -> torch.Tensor:
        if nm not in self._pc:
            X = self.TR[nm][~self.is_hold]
            mu = X.mean(0, keepdim=True)
            q = min(self.pca_dim, X.shape[1])
            _, S, V = torch.pca_lowrank(X - mu, q=q, center=False, niter=4)
            Ztr_fit = (X - mu) @ V
            # Train-fit standard deviation. The pre-registered probe instead
            # rescaled each split by its OWN std, which (a) is transductive on
            # val and (b) applies W to differently-scaled features than it was
            # fit on. --fix_val_scaling uses this train-fit sd for both splits.
            self._pc[nm] = (mu, V, Ztr_fit.std(0, keepdim=True).clamp_min(1e-6))
        mu, V, sd_tr = self._pc[nm]
        Z = ((self.TR[nm] if split == "tr" else self.VA[nm]) - mu) @ V
        if self.fix_val_scaling:
            return Z / sd_tr
        return Z / Z.std(0, keepdim=True).clamp_min(1e-6)

    def feats(self, split: str, shuffle: bool = False, zero: bool = False,
              gen: Optional[torch.Generator] = None) -> torch.Tensor:
        X = torch.cat([self.blk(split, n) for n in ("glob", "subj", "obj", "union")], -1)
        if zero:
            return torch.zeros_like(X)
        if shuffle:
            idx = torch.randperm(X.shape[0], generator=gen) if gen is not None \
                else torch.randperm(X.shape[0])
            return X[idx]
        return X

    # -- metric -------------------------------------------------------------
    def mr_of(self, pred: torch.Tensor, Y: torch.Tensor) -> Tuple[float, float, Dict[str, float]]:
        hit = (pred == Y)
        ph, pg = Counter(), Counter()
        for y, h in zip(Y.tolist(), hit.tolist()):
            pg[y] += 1
            ph[y] += int(h)
        mR = sum(ph[k] / pg[k] for k in pg) / max(1, len(pg))
        bk = {}
        for bname in ("head", "body", "tail"):
            ks = [k for k in pg if self.bucket[k] == bname]
            bk[bname] = sum(ph[k] / pg[k] for k in ks) / max(1, len(ks)) if ks else 0.0
        return float(hit.float().mean()), float(mR), bk

    # -- tau-adjusted prior -------------------------------------------------
    def prior_tau(self, split: str, tau: float) -> torch.Tensor:
        Pm = self.Ptr if split == "tr" else self.Pva
        if tau == 0.0:
            return Pm
        return Pm - tau * self.log_marginal.view(1, -1)

    # -- arms ---------------------------------------------------------------
    def tau_only(self, tau: float, split: str = "va") -> Dict[str, Any]:
        """lambda = 0: the achievable frontier the appearance arm must beat."""
        cand = self.cand_va if split == "va" else self.cand_tr
        Y = self.Yva if split == "va" else self.Ytr
        Pt = self.prior_tau(split, tau)
        s = Pt.gather(1, cand)
        pred = cand.gather(1, s.argmax(1, keepdim=True)).squeeze(1)
        R, m, bk = self.mr_of(pred, Y)
        return {"R": R, "mR": m, "tail_mR": bk["tail"], "head_mR": bk["head"],
                "body_mR": bk["body"]}

    def fit(self, tau: float, lam: float, Xtr: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Fit the appearance head inside the (tau, lambda) composition.

        Model selection (which epoch) is on the held-out-from-train rows.
        Validation is untouched.
        """
        D = Xtr.shape[1]
        W = torch.zeros(self.P, D, requires_grad=True)
        b = torch.zeros(self.P, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=LR, weight_decay=WD)
        Pt = self.prior_tau("tr", tau)
        fi = torch.nonzero((~self.is_hold) & self.in_tr).squeeze(1)
        hm_mask = self.is_hold & self.in_tr
        best = (-1.0, W.detach().clone(), b.detach().clone())
        for _ in range(EPOCHS):
            pm = fi[torch.randperm(int(fi.numel()))]
            for i in range(0, len(pm), 4096):
                sel = pm[i:i + 4096]
                c = self.cand_tr[sel]
                s = (Xtr[sel].unsqueeze(1) * W[c]).sum(-1) + b[c]
                s = Pt[sel].gather(1, c) + lam * s
                tgt = (c == self.Ytr[sel].unsqueeze(1)).float().argmax(1)
                loss = F.cross_entropy(s, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                c = self.cand_tr[hm_mask]
                sh = (Xtr[hm_mask].unsqueeze(1) * W[c]).sum(-1) + b[c]
                sh = Pt[hm_mask].gather(1, c) + lam * sh
                _r, hm, _bk = self.mr_of(c.gather(1, sh.argmax(1, keepdim=True)).squeeze(1),
                                         self.Ytr[hm_mask])
            if hm > best[0]:
                best = (hm, W.detach().clone(), b.detach().clone())
        return best

    def apply(self, W: torch.Tensor, b: torch.Tensor, tau: float, lam: float,
              Xva: torch.Tensor) -> Dict[str, Any]:
        Pt = self.prior_tau("va", tau)
        s = (Xva.unsqueeze(1) * W[self.cand_va]).sum(-1) + b[self.cand_va]
        s = Pt.gather(1, self.cand_va) + lam * s
        pred = self.cand_va.gather(1, s.argmax(1, keepdim=True)).squeeze(1)
        R, m, bk = self.mr_of(pred, self.Yva)
        return {"R": R, "mR": m, "tail_mR": bk["tail"], "head_mR": bk["head"],
                "body_mR": bk["body"]}


def pareto_gap(curve: List[Dict[str, float]], R: float, mR: float) -> Optional[float]:
    """mR of this point minus the tau-only curve's mR at the SAME R@50.

    The curve is monotone decreasing in R as tau rises, so we interpolate
    linearly between the two bracketing tau points. Returns None when R falls
    outside the curve's range (the comparison would be an extrapolation).
    """
    pts = sorted(((c["R"], c["mR"]) for c in curve))
    if not pts or R < pts[0][0] or R > pts[-1][0]:
        return None
    for (r0, m0), (r1, m1) in zip(pts, pts[1:]):
        if r0 <= R <= r1:
            if r1 == r0:
                return mR - max(m0, m1)
            w = (R - r0) / (r1 - r0)
            return mR - (m0 + w * (m1 - m0))
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--taus", type=str, default="0.0,0.05,0.1,0.15,0.2,0.3,0.5")
    ap.add_argument("--lambdas", type=str, default="0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--pca_dims", type=str, default="48",
                    help="comma-separated; 48 reproduces the pre-registered probe")
    ap.add_argument("--null_seeds", type=int, default=5,
                    help="shuffled-appearance refits per (tau,lambda) on the null grid")
    ap.add_argument("--fix_val_scaling", action="store_true",
                    help="normalise val by TRAIN-fit sd (the probe used val's own sd)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    lams = [float(x) for x in args.lambdas.split(",") if x.strip()]
    pdims = [int(x) for x in args.pca_dims.split(",") if x.strip()]

    t_start = time.time()
    out: Dict[str, Any] = {
        "tool": "appearance_tau_interaction",
        "cache": str(args.cache),
        "taus": taus, "lambdas": lams, "pca_dims": pdims,
        "null_seeds": args.null_seeds,
        "fix_val_scaling": bool(args.fix_val_scaling),
        "seed": args.seed,
        "selection": "held-out-from-train; validation never used to select tau, lambda or pca_dim",
        "candidate_set": "top-5 of the RAW (tau=0) prior, identical for every arm",
        "blocks": ["glob", "subj", "obj", "union"],
    }

    per_dim: Dict[str, Any] = {}
    for pd in pdims:
        _log("\n" + "=" * 92)
        _log(f"PCA_DIM = {pd}   fix_val_scaling = {args.fix_val_scaling}")
        _log("=" * 92)
        B = Bench(args.cache, pd, args.fix_val_scaling, seed=args.seed)
        out["meta"] = {k: B.META.get(k) for k in
                       ("clip_model", "n_train_instances", "n_val_instances",
                        "n_crops_encoded", "fp16_autocast", "git_commit")}
        out["coverage@5"] = float(B.in_va.float().mean())

        # Oracle within the same candidate set, for the headroom denominator.
        _Ro, oracle_mR, _bk = B.mr_of(torch.where(B.in_va, B.Yva, B.Pva.argmax(-1)), B.Yva)

        Xtr = B.feats("tr")
        Xva = B.feats("va")
        _log(f"  feature dim: {Xtr.shape[1]}   train rows {Xtr.shape[0]}   val rows {Xva.shape[0]}")

        # --- the lambda=0 frontier -----------------------------------------
        curve = []
        _log(f"\n  {'tau':>6} {'R@50':>8} {'mR@50':>8} {'head':>7} {'body':>7} {'tail':>7}   (lambda=0)")
        for t in taus:
            r = B.tau_only(t)
            r["tau"] = t
            curve.append(r)
            _log(f"  {t:>6} {r['R']*100:>8.2f} {r['mR']*100:>8.2f} {r['head_mR']*100:>7.2f}"
                 f" {r['body_mR']*100:>7.2f} {r['tail_mR']*100:>7.2f}")

        # --- the (tau, lambda) surface, real appearance ---------------------
        _log(f"\n  {'tau':>6} {'lam':>6} {'R@50':>8} {'mR@50':>8} {'tail':>7} {'heldout':>8}"
             f" {'dParetoMR':>10}   (real appearance)")
        grid: List[Dict[str, Any]] = []
        for t in taus:
            for lm in lams:
                hm, W, b = B.fit(t, lm, Xtr)
                r = B.apply(W, b, t, lm, Xva)
                r.update({"tau": t, "lam": lm, "heldout_mR": hm, "arm": "real"})
                r["pareto_dmR_points"] = None
                g = pareto_gap(curve, r["R"], r["mR"])
                if g is not None:
                    r["pareto_dmR_points"] = g * 100.0
                grid.append(r)
                gs = "  n/a" if r["pareto_dmR_points"] is None else f"{r['pareto_dmR_points']:+10.2f}"
                _log(f"  {t:>6} {lm:>6} {r['R']*100:>8.2f} {r['mR']*100:>8.2f}"
                     f" {r['tail_mR']*100:>7.2f} {hm*100:>8.2f} {gs}")

        # --- selection on held-out, then read val ---------------------------
        sel = max(grid, key=lambda r: r["heldout_mR"])

        # --- shuffled null at the SELECTED point ----------------------------
        nulls = []
        _log(f"\n  shuffled-appearance null at selected (tau={sel['tau']}, lam={sel['lam']}), "
             f"{args.null_seeds} refits")
        for s in range(args.null_seeds):
            g = torch.Generator().manual_seed(1000 + s)
            Xtr_s = B.feats("tr", shuffle=True, gen=g)
            g2 = torch.Generator().manual_seed(2000 + s)
            Xva_s = B.feats("va", shuffle=True, gen=g2)
            hm, W, b = B.fit(sel["tau"], sel["lam"], Xtr_s)
            r = B.apply(W, b, sel["tau"], sel["lam"], Xva_s)
            r.update({"seed": s, "heldout_mR": hm, "arm": "shuffled"})
            r["pareto_dmR_points"] = None
            gp = pareto_gap(curve, r["R"], r["mR"])
            if gp is not None:
                r["pareto_dmR_points"] = gp * 100.0
            nulls.append(r)
            gs = "  n/a" if r["pareto_dmR_points"] is None else f"{r['pareto_dmR_points']:+10.2f}"
            _log(f"    seed {s}  R {r['R']*100:>6.2f}  mR {r['mR']*100:>6.2f}  dPareto {gs}")

        nz = [n["pareto_dmR_points"] for n in nulls if n["pareto_dmR_points"] is not None]
        null_mean = sum(nz) / len(nz) if nz else None
        null_sd = (math.sqrt(sum((x - null_mean) ** 2 for x in nz) / max(1, len(nz) - 1))
                   if null_mean is not None and len(nz) > 1 else None)

        per_dim[str(pd)] = {
            "feature_dim": int(Xtr.shape[1]),
            "oracle5_mR": oracle_mR,
            "tau_only_curve": curve,
            "grid": grid,
            "selected": sel,
            "null_shuffled": nulls,
            "null_pareto_dmR_mean_points": null_mean,
            "null_pareto_dmR_sd_points": null_sd,
        }

        _log(f"\n  SELECTED (held-out) tau={sel['tau']} lambda={sel['lam']}")
        _log(f"    val R@50 {sel['R']*100:.2f}   val mR@50 {sel['mR']*100:.2f}")
        if sel["pareto_dmR_points"] is not None:
            _log(f"    Pareto gap vs lambda=0 curve at matched R@50: "
                 f"{sel['pareto_dmR_points']:+.2f} mR points")
        if null_mean is not None:
            sd_s = "n/a" if null_sd is None else f"{null_sd:.2f}"
            _log(f"    shuffled null Pareto gap: {null_mean:+.2f} +/- {sd_s} points")

    out["per_pca_dim"] = per_dim
    out["runtime_seconds"] = round(time.time() - t_start, 2)

    # ---- verdict -----------------------------------------------------------
    # Pre-registered: real appearance converts signal beyond calibration iff
    # its Pareto gap exceeds the shuffled null's mean by >= 2 SD AND is
    # positive in absolute terms.
    verdicts = {}
    for pd, d in per_dim.items():
        sel = d["selected"]
        gap = sel.get("pareto_dmR_points")
        nm_, nsd = d["null_pareto_dmR_mean_points"], d["null_pareto_dmR_sd_points"]
        if gap is None or nm_ is None:
            verdicts[pd] = "INDETERMINATE (selected point outside the curve's R range)"
            continue
        margin = gap - nm_
        thresh = 2.0 * nsd if nsd else 0.0
        verdicts[pd] = ("APPEARANCE ADDS BEYOND CALIBRATION"
                        if (gap > 0 and margin >= thresh)
                        else "NULL: appearance adds nothing the tau curve does not already give")
        d["pareto_margin_over_null_points"] = margin
        d["null_2sd_threshold_points"] = thresh
    out["verdict_per_pca_dim"] = verdicts

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _log("\n" + "=" * 92)
    for pd, v in verdicts.items():
        _log(f"VERDICT  pca_dim={pd}: {v}")
    _log("=" * 92)
    _log(f"[written] {args.out}   ({out['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
