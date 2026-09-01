#!/usr/bin/env python
"""Learned candidate-restricted scorer -- the gate on any GPU spend.

Pre-registered in docs/CANDIDATE_SCORER_PREREGISTRATION.md. CPU only. Reads
runs/p10_model_recalibration/pair_logits.pt read-only.

WHY THIS IS A FAIRER TEST THAN model_rerank
-------------------------------------------
cprime_oracle_ceiling's `model_rerank` arm is the alpha -> infinity limit: it
takes the argmax of the RAW model score inside the prior's top-k and therefore
throws the prior away inside the candidate set. It bounds the raw score as a
ranker and nothing else.

Here the scorer sees the prior term as a feature, and the prior's own top-1 is
always a candidate (it is rank 1 of the top-k by construction). So every arm
strictly GENERALIZES the prior baseline -- weights that read off the prior logit
alone reproduce it exactly -- and can only fall below it through estimation
error. That is what makes the comparison against the achieved additive arm a
statement about learnable capacity rather than about one particular composition.

LEAKAGE
-------
Folds are split by IMAGE, never by row: GT rows from one image share objects,
boxes and a prior context, so a row-wise split would leak. Every reported number
is out-of-fold. Fold assignment is a deterministic hash of the image id, so it
does not depend on iteration order or on the seed used for initialisation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MECH = _load("cprime_mechanism")
Mech, ALPHA_HIST = MECH.Mech, MECH.ALPHA_HIST

R_FLOOR = 0.665
SUCCESS_PTS = 2.0
INCONCLUSIVE_PTS = 1.5
CALIBRATION_EPS = 0.25      # "full adds nothing beyond a decision rule" threshold
NULL_MARGIN_PTS = 1.0
N_FOLDS = 5
SEED = 0
ARMS = ["prior_only", "model_only", "full", "shuffled_model"]


def _log(m: str = "") -> None:
    print(m, flush=True)


def fold_of_image(image_id: str, n_folds: int = N_FOLDS) -> int:
    """Deterministic, order-independent, seed-independent fold assignment."""
    h = hashlib.sha256(str(image_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_folds


class CandidateProbe:
    def __init__(self, B: Mech, tau: float, k: int, seed: int = SEED):
        self.B, self.tau, self.k, self.seed = B, tau, k, seed
        self.pr = B.rows(B.score(tau, ALPHA_HIST, None))     # (n_gt, C) prior term
        self.md = B.rows(B.score(tau, 0.0, B.model))         # (n_gt, C) model term
        self.y = B.gt_y
        self.gt_col = B.gt_col_of()
        self.n, self.C = self.pr.shape

        t2 = self.pr.topk(2, dim=-1)
        self.margin = t2.values[:, 0] - t2.values[:, 1]
        self.prior_top1_col = self.pr.argmax(-1)

        # Candidate set = prior top-k, ordered by prior score (rank 0 = top-1).
        # Rank 0 must be the ARGMAX, not topk's tie-break, so that the "prior's
        # own answer" the scorer can always fall back to is the evaluator's.
        # canonical_topk gives that for free and is shared with the oracle tool,
        # so the two agree on the candidate set by construction rather than by
        # coincidence -- they did not before (4 rows at k=2, 2 rows at k=3).
        self.cand = B.canonical_topk(self.pr, k)                         # (n, k)
        assert torch.equal(self.cand[:, 0], self.prior_top1_col), \
            "candidate 0 must be the evaluator's argmax"
        self.gt_in = (self.cand == self.gt_col.unsqueeze(1)).any(-1)   # (n,)
        # position of GT inside the candidate list, -1 if absent
        eq = (self.cand == self.gt_col.unsqueeze(1))
        self.gt_pos = torch.where(self.gt_in, eq.float().argmax(-1), torch.full((self.n,), -1)).long()

        self.fold = torch.tensor([fold_of_image(B.meta["image_id"][int(i)])
                                  for i in B.gt_img], dtype=torch.long)

    # ------------------------------------------------------------- features
    def _blocks(self, arm: str, gen: torch.Generator) -> torch.Tensor:
        """(n, k, D) candidate features for one arm."""
        n, k, C = self.n, self.k, self.C
        ar = torch.arange(n).unsqueeze(1).expand(n, k)
        pr_c = self.pr[ar, self.cand]                                  # (n,k)
        md_c = self.md[ar, self.cand]

        if arm == "shuffled_model":
            # Permute the MODEL term across rows, preserving each row's internal
            # candidate structure. Destroys the row<->model correspondence while
            # keeping the marginal distribution of the feature identical.
            perm = torch.randperm(n, generator=gen)
            md_c = self.md[perm.unsqueeze(1).expand(n, k), self.cand]

        # row-normalised versions: the scorer sees SHAPE, not scale
        pr_rel = pr_c - pr_c[:, :1]
        md_rel = md_c - md_c.mean(-1, keepdim=True)
        rank1h = torch.eye(k).unsqueeze(0).expand(n, k, k)
        marg = self.B.log_marginal[self.cand]                          # (n,k)
        cls1h = torch.zeros(n, k, C)
        cls1h.scatter_(2, self.cand.unsqueeze(-1), 1.0)
        row_margin = self.margin.view(n, 1, 1).expand(n, k, 1)
        row_ent = self.B.prior_entropy.view(n, 1, 1).expand(n, k, 1)

        prior_blk = [pr_c.unsqueeze(-1), pr_rel.unsqueeze(-1), marg.unsqueeze(-1),
                     rank1h, cls1h, row_margin, row_ent]
        # the third block is the margin interaction: the mechanism report says
        # the model only ever acts on low-margin rows, so give the scorer the
        # ability to gate on that explicitly rather than hoping it infers it.
        model_blk = [md_c.unsqueeze(-1), md_rel.unsqueeze(-1),
                     (md_rel * row_margin.squeeze(-1)).unsqueeze(-1)]
        if arm == "prior_only":
            blk = prior_blk
        elif arm == "model_only":
            blk = model_blk + [rank1h]      # rank is structural, not prior info
        else:
            blk = prior_blk + model_blk
        return torch.cat(blk, dim=-1)

    # ------------------------------------------------------------------ fit
    def _fit_fold(self, X: torch.Tensor, tr: torch.Tensor,
                  epochs: int, l2: float, beta: float = 0.0) -> torch.Tensor:
        """Listwise softmax over candidates, cross-entropy against GT position.

        Only rows whose GT is inside the candidate set carry a gradient: rows
        without GT are unlearnable here and would only add a constant.

        `beta` weights each row by count(GT class)^-beta. beta=0 is plain
        cross-entropy, which maximises accuracy and therefore -- on VG150's label
        distribution -- maximises HEAD recall; beta=1 is fully class-balanced,
        which maximises mR. Sweeping beta traces the scorer's own R/mR frontier,
        which is the only fair thing to compare against the prior-only tau
        frontier: both are one-parameter families trading the same two axes.

        The counts come from the TRAINING rows of this fold only. Using
        full-split counts would leak the held-out label distribution into the
        fit, which at beta=1 is most of the signal.
        """
        use = tr & self.gt_in
        Xtr, ytr = X[use], self.gt_pos[use]
        if beta > 0.0:
            ycls = self.y[use]
            cnt = torch.bincount(ycls, minlength=self.C).float()
            rw = cnt[ycls].clamp_min(1.0) ** (-beta)
            rw = rw / rw.mean()
        else:
            rw = None
        D = X.shape[-1]
        w = torch.zeros(D, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=epochs, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = Xtr @ w
            ce = torch.nn.functional.cross_entropy(logits, ytr, reduction="none")
            loss = (ce * rw).mean() if rw is not None else ce.mean()
            loss = loss + l2 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach()

    def run_arm(self, arm: str, epochs: int, l2: float,
                beta: float = 0.0) -> Dict[str, Any]:
        gen = torch.Generator().manual_seed(self.seed)
        X = self._blocks(arm, gen)
        pred_col = self.prior_top1_col.clone()
        for f in range(N_FOLDS):
            te = self.fold == f
            w = self._fit_fold(X, ~te, epochs, l2, beta)
            sel = (X[te] @ w).argmax(-1)                    # (n_te,) index into cand
            pred_col[te] = self.cand[te].gather(1, sel.unsqueeze(1)).squeeze(1)
        m = self._metrics(pred_col, arm)
        m["beta"] = float(beta)
        return m

    # -------------------------------------------------------------- metrics
    def _metrics_on(self, pred_col: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """R/mR on a row SUBSET -- used only inside nested selection."""
        pred = self.B.col_to_class[pred_col]
        hit = (pred == y).float()
        num = torch.zeros(self.C).index_add_(0, y, hit)
        den = torch.zeros(self.C).index_add_(0, y, torch.ones_like(hit))
        pres = den > 0
        return {"R": float(hit.mean()), "mR": float((num[pres] / den[pres]).mean())}

    def _metrics(self, pred_col: torch.Tensor, arm: str) -> Dict[str, Any]:
        pred = self.B.col_to_class[pred_col]
        hit = (pred == self.y).float()
        num = torch.zeros(self.C).index_add_(0, self.y, hit)
        den = torch.zeros(self.C).index_add_(0, self.y, torch.ones_like(hit))
        pres = den > 0
        ids = pres.nonzero().squeeze(1).tolist()
        per = {c: float(num[c] / den[c]) for c in ids}
        out = {"arm": arm, "R": float(hit.mean()),
               "mR": float((num[pres] / den[pres]).mean()),
               "n_classes": int(pres.sum()),
               "frac_argmax_changed": float((pred_col != self.prior_top1_col).float().mean())}
        for b in ("head", "body", "tail"):
            ks = [c for c in ids if self.B.bucket_of[c] == b]
            out[f"{b}_mR"] = float(sum(per[c] for c in ks) / len(ks)) if ks else 0.0
        out["_per"] = per
        return out

    def baselines(self) -> Dict[str, Any]:
        pm = self._metrics(self.prior_top1_col, "prior")
        cb = self._metrics((self.pr + self.md).argmax(-1), "achieved_additive")
        return {"prior": pm, "achieved": cb,
                "achieved_dR_points": (cb["R"] - pm["R"]) * 100.0,
                "candidate_coverage": float(self.gt_in.float().mean())}


def frontier(B: Mech, tau: float, k: int, betas: List[float],
             epochs: int, l2: float) -> Dict[str, Any]:
    """Trace the learned scorer's R/mR frontier and compare it with tau's.

    THE question this project has not yet asked in a falsifiable form: the tau
    frontier is a one-parameter family trading R@50 for mR@50 with NO visual
    input. If the learned scorer's own one-parameter family (beta) merely lands
    ON that frontier, everything measured so far is calibration. If it lies
    ABOVE it, the model buys something calibration cannot.

    `pareto_gap` is cprime_analysis's, so the frontier and the gap are the same
    ones every other C' table uses.
    """
    CPAm = sys.modules["cprime_analysis"]
    curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
             for t in CPAm.TAUS]
    P = CandidateProbe(B, tau, k)
    bl = P.baselines()
    rows = []
    for arm in ("prior_only", "full", "shuffled_model"):
        for beta in betas:
            m = P.run_arm(arm, epochs, l2, beta)
            m["dR_points"] = (m["R"] - bl["prior"]["R"]) * 100.0
            m["dmR_points"] = (m["mR"] - bl["prior"]["mR"]) * 100.0
            # pareto_gap ALREADY returns mR points, not a fraction. Scaling it
            # by 100 again reported the achieved arm at +86.079 instead of its
            # true +0.861 -- and +0.861 is exactly its dmR, which is the
            # tau=0 lower-bound branch of the frontier, i.e. the cross-check
            # that caught it.
            m["pareto_dmR_points"] = CPAm.pareto_gap(curve, m["R"], m["mR"])
            m["meets_R_floor"] = bool(m["R"] >= R_FLOOR)
            m.pop("_per", None)
            rows.append(m)
    ach = bl["achieved"]
    return {"tau": tau, "k": k, "betas": betas,
            "tau_frontier": [{"tau": t, **c} for t, c in zip(CPAm.TAUS, curve)],
            "achieved_pareto_dmR_points": CPAm.pareto_gap(curve, ach["R"], ach["mR"]),
            "rows": rows, "baselines": bl}


def nested(B: Mech, tau: float, k: int, betas: List[float],
           epochs: int, l2: float, floor: float = R_FLOOR) -> Dict[str, Any]:
    """Nested cross-fitting: beta is SELECTED inside the training folds only.

    The frontier table shows `full` clearing the R@50 floor at beta=0.20 with a
    Pareto gap of +2.86 against the tau frontier, where prior_only and the null
    both fail the floor. But beta=0.20 was read off that table, so as it stands
    it is a validation-optimised operating point quoted as if it were
    pre-registered -- exactly what the directive forbids.

    This removes the operating point from the reported number entirely. For each
    outer fold: hold it out, split the remaining folds into an inner fit set and
    an inner selection set, fit every beta on the inner fit set, choose the beta
    that maximises mR@50 subject to R@50 >= floor on the inner SELECTION set,
    refit at that beta on all non-outer folds, and predict the outer fold. No
    label from an outer fold ever touches the fit or the choice, so the pooled
    out-of-fold metrics are honest with no chosen constant anywhere in them.

    The selection objective is the project's actual target -- maximise mR at the
    R floor -- and is fixed before the run, not the metric that happened to look
    best.
    """
    chosen, pred_col = [], None
    P = CandidateProbe(B, tau, k)
    bl = P.baselines()
    Xc = {a: P._blocks(a, torch.Generator().manual_seed(SEED))
          for a in ("prior_only", "full", "shuffled_model")}
    out: Dict[str, Any] = {"tau": tau, "k": k, "betas": betas, "floor": floor,
                           "baselines": bl, "arms": {}}
    for arm, X in Xc.items():
        pred_col = P.prior_top1_col.clone()
        picks = []
        for f in range(N_FOLDS):
            te = P.fold == f
            inner_sel = P.fold == ((f + 1) % N_FOLDS)      # one inner fold selects
            inner_fit = (~te) & (~inner_sel)
            best_b, best_key = betas[0], None
            for b in betas:
                w = P._fit_fold(X, inner_fit, epochs, l2, b)
                sel = (X[inner_sel] @ w).argmax(-1)
                pc = P.cand[inner_sel].gather(1, sel.unsqueeze(1)).squeeze(1)
                m = P._metrics_on(pc, P.y[inner_sel])
                key = (m["R"] >= floor, m["mR"])           # floor first, then mR
                if best_key is None or key > best_key:
                    best_key, best_b = key, b
            picks.append(best_b)
            w = P._fit_fold(X, ~te, epochs, l2, best_b)
            sel = (X[te] @ w).argmax(-1)
            pred_col[te] = P.cand[te].gather(1, sel.unsqueeze(1)).squeeze(1)
        m = P._metrics(pred_col, arm)
        m.pop("_per", None)
        m["dR_points"] = (m["R"] - bl["prior"]["R"]) * 100.0
        m["dmR_points"] = (m["mR"] - bl["prior"]["mR"]) * 100.0
        m["meets_R_floor"] = bool(m["R"] >= floor)
        m["betas_chosen_per_outer_fold"] = picks
        CPAm = sys.modules["cprime_analysis"]
        curve = [{"R": (q := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": q["mR"]}
                 for t in CPAm.TAUS]
        m["pareto_dmR_points"] = CPAm.pareto_gap(curve, m["R"], m["mR"])
        out["arms"][arm] = m
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p10_model_recalibration/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--scheme", default="raw50", choices=["raw50", "eval48"])
    ap.add_argument("--out", default="runs/p16_candidate_scorer/probe.json")
    ap.add_argument("--taus", default="0.0,0.05")
    ap.add_argument("--ks", default="3,5")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--nested", action="store_true",
                    help="select beta inside the training folds only (honest OOF)")
    ap.add_argument("--frontier", action="store_true",
                    help="sweep beta and compare the learned R/mR frontier with tau's")
    ap.add_argument("--betas", default="0.0,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--balanced", action="store_true",
                    help="class-balanced listwise loss; weights from TRAIN rows only")
    args = ap.parse_args(argv)

    torch.manual_seed(SEED)
    _log("=" * 104)
    _log("LEARNED CANDIDATE-RESTRICTED SCORER -- CPU only, cache read-only, NO GPU")
    _log("=" * 104)
    B = Mech(args.dump, args.prior, args.scheme)
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    _log(f"  images={B.n_images} GT rows={B.n_gt} classes={B.n_classes}  "
         f"[gate] model-term identity {e:.3e} OK")
    _log(f"  folds={N_FOLDS} (split by IMAGE) seed={SEED} l2={args.l2} "
         f"floor R@50={R_FLOOR*100:.1f} loss={'class-balanced' if args.balanced else 'plain CE'}")

    if args.nested:
        betas = [float(b) for b in args.betas.split(",")]
        out: Dict[str, Any] = {"tool": "candidate_scorer_probe_nested", "seed": SEED,
                               "n_folds": N_FOLDS, "R_floor": R_FLOOR,
                               "betas": betas, "by": {}}
        for tau in [float(t) for t in args.taus.split(",")]:
            for k in [int(x) for x in args.ks.split(",")]:
                nr = nested(B, tau, k, betas, args.epochs, args.l2)
                bl = nr["baselines"]
                _log(f"\n{'-'*104}\n  NESTED (beta chosen inside training folds)  tau={tau} k={k}"
                     f"\n  prior R@50 {bl['prior']['R']*100:.3f} mR {bl['prior']['mR']*100:.3f}"
                     f"   |  achieved additive R@50 {bl['achieved']['R']*100:.3f}"
                     f" mR {bl['achieved']['mR']*100:.3f}\n{'-'*104}")
                _log(f"  {'arm':>16} {'R@50':>8} {'mR@50':>8} {'dR':>8} {'dmR':>8}"
                     f" {'paretoGap':>10} {'floor':>6} {'head':>7} {'body':>7} {'tail':>7}  betas")
                for a, m in nr["arms"].items():
                    pg = m["pareto_dmR_points"]
                    _log(f"  {a:>16} {m['R']*100:>8.3f} {m['mR']*100:>8.3f}"
                         f" {m['dR_points']:>+8.3f} {m['dmR_points']:>+8.3f}"
                         f" {('n/a' if pg is None else f'{pg:+.3f}'):>10}"
                         f" {'ok' if m['meets_R_floor'] else 'FAIL':>6}"
                         f" {m['head_mR']*100:>7.2f} {m['body_mR']*100:>7.2f}"
                         f" {m['tail_mR']*100:>7.2f}  {m['betas_chosen_per_outer_fold']}")
                F, Pq = nr["arms"]["full"], nr["arms"]["prior_only"]
                Nq = nr["arms"]["shuffled_model"]
                _log(f"\n    full - prior_only : dR {F['dR_points']-Pq['dR_points']:+.3f}"
                     f"   dmR {F['dmR_points']-Pq['dmR_points']:+.3f}")
                _log(f"    full - null       : dR {F['dR_points']-Nq['dR_points']:+.3f}"
                     f"   dmR {F['dmR_points']-Nq['dmR_points']:+.3f}")
                nr["baselines"]["prior"].pop("_per", None)
                nr["baselines"]["achieved"].pop("_per", None)
                out["by"][f"tau{tau}_k{k}"] = nr
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        _log(f"\n[written] {args.out}")
        return 0

    if args.frontier:
        betas = [float(b) for b in args.betas.split(",")]
        out: Dict[str, Any] = {"tool": "candidate_scorer_probe_frontier",
                               "seed": SEED, "n_folds": N_FOLDS,
                               "R_floor": R_FLOOR, "betas": betas, "by": {}}
        for tau in [float(t) for t in args.taus.split(",")]:
            for k in [int(x) for x in args.ks.split(",")]:
                fr = frontier(B, tau, k, betas, args.epochs, args.l2)
                bl = fr["baselines"]
                _log(f"\n{'-'*104}\n  FRONTIER  tau={tau} k={k}   prior R@50 "
                     f"{bl['prior']['R']*100:.3f} mR {bl['prior']['mR']*100:.3f}   |  achieved "
                     f"R@50 {bl['achieved']['R']*100:.3f} mR {bl['achieved']['mR']*100:.3f}"
                     f"  paretoGap {fr['achieved_pareto_dmR_points']:+.3f}\n{'-'*104}")
                _log(f"  {'arm':>12} {'beta':>6} {'R@50':>8} {'mR@50':>8} {'dR':>8} {'dmR':>8}"
                     f" {'paretoGap':>10} {'floor':>6} {'head':>7} {'body':>7} {'tail':>7}")
                for r in fr["rows"]:
                    pg = r["pareto_dmR_points"]
                    _log(f"  {r['arm']:>12} {r['beta']:>6.2f} {r['R']*100:>8.3f} {r['mR']*100:>8.3f}"
                         f" {r['dR_points']:>+8.3f} {r['dmR_points']:>+8.3f}"
                         f" {('n/a' if pg is None else f'{pg:+.3f}'):>10}"
                         f" {'ok' if r['meets_R_floor'] else 'FAIL':>6}"
                         f" {r['head_mR']*100:>7.2f} {r['body_mR']*100:>7.2f} {r['tail_mR']*100:>7.2f}")
                fr["baselines"]["prior"].pop("_per", None)
                fr["baselines"]["achieved"].pop("_per", None)
                out["by"][f"tau{tau}_k{k}"] = fr
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        _log(f"\n[written] {args.out}")
        return 0

    res: Dict[str, Any] = {"tool": "candidate_scorer_probe", "seed": SEED,
                           "n_folds": N_FOLDS, "R_floor": R_FLOOR,
                           "l2": args.l2, "epochs": args.epochs,
                           "balanced": bool(args.balanced), "by": {}}

    for tau in [float(t) for t in args.taus.split(",")]:
        for k in [int(x) for x in args.ks.split(",")]:
            P = CandidateProbe(B, tau, k)
            bl = P.baselines()
            A = bl["achieved_dR_points"]
            pR = bl["prior"]["R"]
            _log(f"\n{'-'*104}\n  tau={tau}  k={k}   prior R@50 {pR*100:.3f} mR {bl['prior']['mR']*100:.3f}"
                 f"   |  ACHIEVED additive R@50 {bl['achieved']['R']*100:.3f}"
                 f" mR {bl['achieved']['mR']*100:.3f}  dR {A:+.3f}"
                 f"   |  GT-in-top-{k} {bl['candidate_coverage']*100:.1f}%\n{'-'*104}")
            _log(f"  {'arm':>16} {'R@50':>8} {'dR':>8} {'floor':>6} {'mR@50':>8} {'dmR':>8}"
                 f" {'head':>7} {'body':>7} {'tail':>7} {'chg%':>6}")
            _log(f"  {'prior':>16} {pR*100:>8.3f} {0.0:>+8.3f} {'ok':>6}"
                 f" {bl['prior']['mR']*100:>8.3f} {0.0:>+8.3f}"
                 f" {bl['prior']['head_mR']*100:>7.2f} {bl['prior']['body_mR']*100:>7.2f}"
                 f" {bl['prior']['tail_mR']*100:>7.2f} {0.0:>6.2f}")
            _log(f"  {'achieved(add)':>16} {bl['achieved']['R']*100:>8.3f} {A:>+8.3f}"
                 f" {'ok' if bl['achieved']['R']>=R_FLOOR else 'FAIL':>6}"
                 f" {bl['achieved']['mR']*100:>8.3f}"
                 f" {(bl['achieved']['mR']-bl['prior']['mR'])*100:>+8.3f}"
                 f" {bl['achieved']['head_mR']*100:>7.2f} {bl['achieved']['body_mR']*100:>7.2f}"
                 f" {bl['achieved']['tail_mR']*100:>7.2f}"
                 f" {bl['achieved']['frac_argmax_changed']*100:>6.2f}")

            arms: Dict[str, Any] = {}
            for arm in ARMS:
                m = P.run_arm(arm, args.epochs, args.l2, 1.0 if args.balanced else 0.0)
                m["dR_points"] = (m["R"] - pR) * 100.0
                m["dmR_points"] = (m["mR"] - bl["prior"]["mR"]) * 100.0
                m["meets_R_floor"] = bool(m["R"] >= R_FLOOR)
                arms[arm] = m
                _log(f"  {arm:>16} {m['R']*100:>8.3f} {m['dR_points']:>+8.3f}"
                     f" {'ok' if m['meets_R_floor'] else 'FAIL':>6}"
                     f" {m['mR']*100:>8.3f} {m['dmR_points']:>+8.3f}"
                     f" {m['head_mR']*100:>7.2f} {m['body_mR']*100:>7.2f}"
                     f" {m['tail_mR']*100:>7.2f} {m['frac_argmax_changed']*100:>6.2f}")

            F = arms["full"]["dR_points"]
            Pq = arms["prior_only"]["dR_points"]
            Nq = arms["shuffled_model"]["dR_points"]
            floor_ok = arms["full"]["meets_R_floor"]
            beats_null = (F - Nq) > NULL_MARGIN_PTS
            verdict = ("SUCCESS" if (floor_ok and F > A + SUCCESS_PTS and beats_null)
                       else "INCONCLUSIVE" if (floor_ok and F > A + INCONCLUSIVE_PTS)
                       else "EXHAUSTED")
            calib = (F - Pq) <= CALIBRATION_EPS
            # The null must be compared with prior_only, NOT with the achieved
            # additive arm. A correct null carries no model information, so the
            # fit falls back on the prior features and REPRODUCES prior_only --
            # which is what happens here (|null - prior_only| <= 0.032 in every
            # cell). The pre-registration wrote this condition as `null > A`,
            # which fires whenever a learned decision rule beats C', i.e. on a
            # real finding rather than on pipeline pathology. Documented in
            # docs/CANDIDATE_SCORER_RESULT.md; the mis-specification is recorded
            # rather than quietly dropped.
            null_breach = (Nq - Pq) > NULL_MARGIN_PTS
            null_breach_as_prereg = Nq > A
            _log(f"\n    full-achieved {F-A:+.3f}   full-prior_only {F-Pq:+.3f}"
                 f"   full-null {F-Nq:+.3f}   ->  {verdict}")
            if calib:
                _log(f"    CALIBRATION FINDING: model features add {F-Pq:+.3f} pts beyond a "
                     f"learned per-class decision rule (threshold {CALIBRATION_EPS}).")
            if Pq > A:
                _log(f"    NOTE: prior_only ({Pq:+.3f}) exceeds the achieved additive arm "
                     f"({A:+.3f}) -- a rule with NO visual input reproduces the C' gain.")
            if null_breach:
                _log(f"    *** NULL BREACH: shuffled_model ({Nq:+.3f}) exceeds prior_only "
                     f"({Pq:+.3f}) by more than {NULL_MARGIN_PTS}. The pipeline is "
                     f"manufacturing gain; no arm may be reported. ***")
            else:
                _log(f"    null check ok: shuffled_model - prior_only = {Nq-Pq:+.3f} "
                     f"(a correct null reproduces prior_only)")
            if null_breach_as_prereg:
                _log(f"    [as pre-registered, null({Nq:+.3f}) > achieved({A:+.3f}): that "
                     f"condition tests a learned rule against C', not the pipeline -- see report]")

            for m in list(arms.values()) + [bl["prior"], bl["achieved"]]:
                m.pop("_per", None)
            res["by"][f"tau{tau}_k{k}"] = {
                "tau": tau, "k": k, "baselines": bl, "arms": arms,
                "full_minus_achieved_points": F - A,
                "full_minus_prior_only_points": F - Pq,
                "full_minus_null_points": F - Nq,
                "verdict": verdict,
                "calibration_finding": bool(calib),
                "prior_only_exceeds_achieved": bool(Pq > A),
                "null_breach": bool(null_breach),
                "null_breach_as_prereg_written": bool(null_breach_as_prereg),
                "null_minus_prior_only_points": Nq - Pq,
                "balanced": bool(args.balanced)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
