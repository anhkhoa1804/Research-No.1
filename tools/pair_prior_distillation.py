#!/usr/bin/env python
"""How much of the checkpoint's contribution is just P(predicate | subject, object)?

MECHANISTIC DECOMPOSITION, not a confirmatory model claim. CPU only. Reads
runs/p10_model_recalibration/pair_logits.pt and the train-derived prior
read-only. No GPU, no training of any encoder, no modification of any C'
artifact or pre-registration.

WHY
---
runs/p26 measured that destroying image content while preserving (subject,
object) identity costs -0.114 +- 0.265 Pareto points, while destroying pair
identity costs +3.163. So the checkpoint's usable contribution is pair-
conditioned. This asks the follow-up that decides whether the checkpoint is
needed at all: can pair-conditioned STATISTICS -- no vision, no checkpoint --
reproduce it?

THE LADDER
----------
Every arm shares the candidate set (the prior's canonical top-k), the
denominator (all GT rows, 50 classes), tau, alpha, the folds, the nested beta
selection, the R@50 floor and the evaluation semantics. Arms differ ONLY in
which score columns they receive.

  A global        log P(p)                       -- the marginal
  B subject       log P(p | subject)
  C object        log P(p | object)
  D pair          log P(p | subject, object)     -- the existing prior term
  E backoff       A + B + C + D, weights learned -- hierarchical smoothing
  F pair_foldfit  E + P(p|s,o) re-estimated from the TRAINING FOLDS of this
                  cache, so it carries validation-distribution pair statistics
                  the train-derived prior cannot have
  G model         D + the C' model term          -- the checkpoint arm

Plus the decomposition that makes the answer mechanical rather than inferential:

  G_pairmean      D + the model term replaced by its TRAINING-FOLD (subject,
                  object) group mean -- the pair-conditioned COMPONENT of the
                  model term, with the within-group part deleted
  G_residual      D + the model term MINUS that group mean -- the within-group
                  component alone, which is the only place image conditioning
                  can live

If G_pairmean reproduces G and G_residual reproduces D, the model term IS a pair
prior, measured rather than argued.

Nulls carried through the identical path: shuffled_model (identity destroyed)
and pair_matched_null (image destroyed, identity kept).

LEAKAGE
-------
A-E come from the TRAIN split (83,249 images, disjoint from validation --
pinned by tests/test_split_separation.py), so they cannot leak regardless of
folds. F, G_pairmean and G_residual are estimated per fold from TRAINING ROWS
ONLY; groups unseen in a fold's training rows fall back to that fold's global
training mean. Folds split by image. Every reported number is out-of-fold.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
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


CSP = _load("candidate_scorer_probe")
CPA = sys.modules["cprime_analysis"]
Mech, ALPHA_HIST = CSP.Mech, CSP.ALPHA_HIST
N_FOLDS, SEED, R_FLOOR = CSP.N_FOLDS, CSP.SEED, CSP.R_FLOOR

ARMS = ["A_global", "B_subject", "C_object", "D_pair", "E_backoff",
        "F_pair_foldfit", "G_model", "G_pairmean", "G_residual",
        "null_shuffled", "null_pair_matched"]
# which conditional log-prob sources each arm receives
SOURCES: Dict[str, Tuple[str, ...]] = {
    "A_global": ("g",), "B_subject": ("s",), "C_object": ("o",),
    "D_pair": ("p",), "E_backoff": ("g", "s", "o", "p"),
    "F_pair_foldfit": ("g", "s", "o", "p", "f"),
    "G_model": ("p",), "G_pairmean": ("p",), "G_residual": ("p",),
    "null_shuffled": ("p",), "null_pair_matched": ("p",),
}
MODEL_ARMS = {"G_model": "real", "G_pairmean": "pairmean", "G_residual": "residual",
              "null_shuffled": "shuffled", "null_pair_matched": "pair_matched"}


def _log(m: str = "") -> None:
    print(m, flush=True)


class Distill:
    """One cache, one fold structure, many conditional-prior arms."""

    def __init__(self, B: Mech, prior_path: str, tau: float, k: int,
                 fold_salt: int = 0):
        self.B, self.tau, self.k = B, tau, k
        self.P = CSP.CandidateProbe(B, tau, k, SEED, fold_salt)
        P = self.P
        self.n, self.C = P.n, P.C
        self.y, self.fold, self.pair_id = P.y, P.fold, P.pair_id
        self.md = P.md
        self.prior_term = P.pr                       # alpha*(prior - tau*marg)
        self.gt_col = P.gt_col_of() if hasattr(P, "gt_col_of") else P.gt_col

        # ---- row-level subject / object label strings, cache order ----
        d = B.meta
        sl, ol = [], []
        for i in range(B.n_images):
            sl.extend(d["subj_label"][i])
            ol.extend(d["obj_label"][i])
        self.subj = [sl[int(r)] for r in B.gt_row.tolist()]
        self.obj = [ol[int(r)] for r in B.gt_row.tolist()]

        # ---- conditional log-probs from the TRAIN-derived prior file ----
        raw = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        src = [str(x).strip().lower() for x in raw["predicate_vocab"]]
        idx = {p: i for i, p in enumerate(src)}
        self.default = float(raw.get("default_log_prob", -20.0))
        cols = [idx.get(nm) for nm in B.fg_names_raw]
        missing = sum(1 for c in cols if c is None)
        assert missing == 0, f"{missing} cache predicates absent from the prior vocab"
        self.cols = cols

        g = raw["global_log_probs"]
        self.v_g = torch.tensor([float(g[c]) for c in cols]).unsqueeze(0).expand(self.n, self.C)
        self.v_s = self._lookup(raw.get("subject_log_probs", {}), self.subj)
        self.v_o = self._lookup(raw.get("object_log_probs", {}), self.obj)
        self.v_p = self._lookup(raw.get("pair_log_probs", {}),
                                [f"{a}||{b}" for a, b in zip(self.subj, self.obj)])
        self.coverage = {
            "subject": float(self._seen(raw.get("subject_log_probs", {}), self.subj)),
            "object": float(self._seen(raw.get("object_log_probs", {}), self.obj)),
            "pair": float(self._seen(raw.get("pair_log_probs", {}),
                                     [f"{a}||{b}" for a, b in zip(self.subj, self.obj)])),
        }

        # rank bucket over the FULL vocabulary (k+1 buckets; last = "beyond top-k")
        order = B.canonical_topk(self.prior_term, self.C)
        rank = torch.empty(self.n, self.C, dtype=torch.long)
        ar = torch.arange(self.n).unsqueeze(1).expand(self.n, self.C)
        rank[ar, order] = torch.arange(self.C).unsqueeze(0).expand(self.n, self.C)
        self.rank_bucket = rank.clamp(max=k)
        self.cand_mask = torch.zeros(self.n, self.C, dtype=torch.bool)
        self.cand_mask.scatter_(1, P.cand, True)
        self.prior_top1_col = P.prior_top1_col
        self.margin, self.entropy = P.margin, B.prior_entropy

    # --------------------------------------------------- estimable subset
    def estimable_mask(self) -> torch.Tensor:
        """Rows whose (subject, object) group HAS a training row out of fold.

        `_fold_pair_mean` falls back to the training GLOBAL mean for a group
        with no training row, so those rows are scored by a "pair-conditioned"
        arm that carries no pair information at all. A singleton group is always
        in that state. `runs/p27` was scored on them anyway and was withdrawn
        for it (docs/PAIR_PRIOR_DISTILLATION_RESULT.md).

        This mask is the analysis population of the corrected experiment. It
        reads `pair_id` and the fold assignment only -- never `y`, never the
        model term -- so conditioning on it is not label leakage. It is NOT a
        random subsample: it over-represents frequent pairs by construction, so
        every baseline and the tau frontier must be recomputed under it.
        """
        G = int(self.pair_id.max()) + 1
        total = torch.zeros(G).index_add_(
            0, self.pair_id, torch.ones(self.n))
        est = torch.zeros(self.n, dtype=torch.bool)
        for f in range(N_FOLDS):
            te = self.fold == f
            in_fold = torch.zeros(G).index_add_(
                0, self.pair_id[te], torch.ones(int(te.sum())))
            has_train = (total - in_fold) > 0            # >=1 row outside fold f
            est |= te & has_train[self.pair_id]
        return est

    def fallback_frac(self, train: torch.Tensor,
                      rows: Optional[torch.Tensor] = None) -> float:
        """Fraction of `rows` whose group has NO training row in `train`."""
        G = int(self.pair_id.max()) + 1
        cnt = torch.zeros(G).index_add_(
            0, self.pair_id[train], torch.ones(int(train.sum())))
        fb = cnt[self.pair_id] == 0
        sel = fb if rows is None else fb[rows]
        return float(sel.float().mean()) if sel.numel() else 0.0

    def residual_cosine(self, rows: torch.Tensor) -> float:
        """Class-centred cosine of G_residual against the REAL model term.

        Gate G2. On fallback rows the group mean IS the global mean, so the
        "residual" is a recentred copy of the whole term and this reads ~0.94
        (runs/p30). On genuinely estimable rows p30 measured ~0.46.
        """
        out = []
        for f in range(N_FOLDS):
            te = (self.fold == f) & rows
            if not bool(te.any()):
                continue
            gm = self._fold_pair_mean(self.md, ~(self.fold == f))
            res = (self.md - gm)[te]
            real = self.md[te]
            res = res - res.mean(-1, keepdim=True)
            real = real - real.mean(-1, keepdim=True)
            cs = torch.nn.functional.cosine_similarity(res, real, dim=-1)
            out.append(cs)
        return float(torch.cat(out).mean()) if out else float("nan")

    def _lookup(self, table: Dict[str, Any], keys: List[str]) -> torch.Tensor:
        out = torch.full((len(keys), self.C), self.default)
        for r, kk in enumerate(keys):
            v = table.get(kk)
            if v is not None:
                out[r] = torch.tensor([float(v[c]) for c in self.cols])
        return out

    @staticmethod
    def _seen(table: Dict[str, Any], keys: List[str]) -> float:
        return sum(1 for kk in keys if kk in table) / max(1, len(keys))

    # ------------------------------------------------------------ features
    def _fold_pair_mean(self, X: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
        """Per-(subject,object) mean of X over TRAINING rows only.

        Groups with no training row fall back to the training global mean, so no
        held-out row ever contributes to the statistic used to score it.
        """
        G = int(self.pair_id.max()) + 1
        cnt = torch.zeros(G).index_add_(0, self.pair_id[train], torch.ones(int(train.sum())))
        s = torch.zeros(G, X.shape[1]).index_add_(0, self.pair_id[train], X[train])
        gmean = torch.where(cnt.unsqueeze(1) > 0, s / cnt.clamp_min(1).unsqueeze(1),
                            X[train].mean(0, keepdim=True))
        return gmean[self.pair_id]

    def blocks(self, arm: str, train: torch.Tensor,
               gen: torch.Generator) -> torch.Tensor:
        """(n, C, D) features over the FULL vocabulary for one arm."""
        n, C = self.n, self.C
        parts: List[torch.Tensor] = []
        srcs = SOURCES[arm]
        table = {"g": self.v_g, "s": self.v_s, "o": self.v_o, "p": self.v_p}
        for key in srcs:
            if key == "f":
                parts.append(self._fold_pair_mean(self.prior_term, train).unsqueeze(-1))
            else:
                v = table[key]
                parts.append(v.unsqueeze(-1))
                parts.append((v - v.mean(-1, keepdim=True)).unsqueeze(-1))

        if arm in MODEL_ARMS:
            kind = MODEL_ARMS[arm]
            md = self.md
            if kind == "shuffled":
                md = self.md[torch.randperm(n, generator=gen)]
            elif kind == "pair_matched":
                perm = torch.arange(n)
                for gid in self.pair_id.unique().tolist():
                    ix = (self.pair_id == gid).nonzero().squeeze(1)
                    if ix.numel() > 1:
                        perm[ix] = ix[torch.randperm(ix.numel(), generator=gen)]
                md = self.md[perm]
            elif kind == "pairmean":
                md = self._fold_pair_mean(self.md, train)
            elif kind == "residual":
                md = self.md - self._fold_pair_mean(self.md, train)
            parts.append(md.unsqueeze(-1))
            parts.append((md - md.mean(-1, keepdim=True)).unsqueeze(-1))
            parts.append(((md - md.mean(-1, keepdim=True))
                          * self.margin.view(n, 1)).unsqueeze(-1))

        # structural features, identical in every arm
        parts.append(torch.nn.functional.one_hot(self.rank_bucket, self.k + 1).float())
        parts.append(torch.eye(C).unsqueeze(0).expand(n, C, C))
        parts.append(self.margin.view(n, 1, 1).expand(n, C, 1))
        parts.append(self.entropy.view(n, 1, 1).expand(n, C, 1))
        return torch.cat(parts, dim=-1)

    # ----------------------------------------------------------------- fit
    def _fit(self, X: torch.Tensor, train: torch.Tensor, beta: float,
             epochs: int, l2: float) -> torch.Tensor:
        gt_in = self.cand_mask.gather(1, self.gt_col.unsqueeze(1)).squeeze(1)
        use = train & gt_in
        # restrict the listwise softmax to the candidate columns
        cand = self.P.cand[use]
        Xtr = X[use].gather(1, cand.unsqueeze(-1).expand(-1, -1, X.shape[-1]))
        ytr = (cand == self.gt_col[use].unsqueeze(1)).float().argmax(-1)
        if beta > 0.0:
            yc = self.y[use]
            cnt = torch.bincount(yc, minlength=self.C).float()
            rw = cnt[yc].clamp_min(1.0) ** (-beta)
            rw = rw / rw.mean()
        else:
            rw = None
        w = torch.zeros(X.shape[-1], requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=epochs, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(Xtr @ w, ytr, reduction="none")
            loss = ((ce * rw).mean() if rw is not None else ce.mean()) + l2 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach()

    # ------------------------------------------------------------- metrics
    def metrics(self, pred_col: torch.Tensor, full_score: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if mask is not None:
            pred_col, full_score = pred_col[mask], full_score[mask]
        y = self.y if mask is None else self.y[mask]
        gt_col = self.gt_col if mask is None else self.gt_col[mask]
        prior_top1 = (self.prior_top1_col if mask is None
                      else self.prior_top1_col[mask])
        pred = self.B.col_to_class[pred_col]
        hit = (pred == y).float()
        num = torch.zeros(self.C).index_add_(0, y, hit)
        den = torch.zeros(self.C).index_add_(0, y, torch.ones_like(hit))
        pres = den > 0
        ids = pres.nonzero().squeeze(1).tolist()
        per = {c: float(num[c] / den[c]) for c in ids}
        gt_s = full_score.gather(1, gt_col.unsqueeze(1))
        rank = (full_score > gt_s).sum(-1) + 1
        out = {"R": float(hit.mean()), "mR": float((num[pres] / den[pres]).mean()),
               "n_classes": int(pres.sum()), "n_rows": int(y.numel()),
               "top1": float(hit.mean()),
               "mean_gt_rank": float(rank.float().mean()),
               "MRR": float((1.0 / rank.float()).mean()),
               "R_at_5": float((rank <= 5).float().mean()),
               "frac_argmax_changed": float((pred_col != prior_top1).float().mean())}
        for b in ("head", "body", "tail"):
            ks = [c for c in ids if self.B.bucket_of[c] == b]
            out[f"{b}_mR"] = float(sum(per[c] for c in ks) / len(ks)) if ks else 0.0
        return out

    def run_arm(self, arm: str, betas: List[float], epochs: int,
                l2: float, floor: float,
                mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        pred_col = self.prior_top1_col.clone()
        full_score = torch.zeros(self.n, self.C)
        picks = []
        for f in range(N_FOLDS):
            te = self.fold == f
            inner_sel = self.fold == ((f + 1) % N_FOLDS)
            inner_fit = (~te) & (~inner_sel)
            X_in = self.blocks(arm, inner_fit, torch.Generator().manual_seed(SEED))
            best_b, best_key = betas[0], None
            for b in betas:
                w = self._fit(X_in, inner_fit, b, epochs, l2)
                sc = X_in[inner_sel] @ w
                sc = sc.masked_fill(~self.cand_mask[inner_sel], -1e30)
                pc = sc.argmax(-1)
                # beta is selected on the SAME population the arm is scored on,
                # otherwise the inner criterion optimises rows the outer read
                # never sees.
                keep = (torch.ones(int(inner_sel.sum()), dtype=torch.bool)
                        if mask is None else mask[inner_sel])
                m = (self.P._metrics_on(pc[keep], self.y[inner_sel][keep])
                     if bool(keep.any())
                     else self.P._metrics_on(pc, self.y[inner_sel]))
                key = (m["R"] >= floor, m["mR"])
                if best_key is None or key > best_key:
                    best_key, best_b = key, b
            picks.append(best_b)
            X_out = self.blocks(arm, ~te, torch.Generator().manual_seed(SEED))
            w = self._fit(X_out, ~te, best_b, epochs, l2)
            sc = X_out[te] @ w
            full_score[te] = sc
            pred_col[te] = sc.masked_fill(~self.cand_mask[te], -1e30).argmax(-1)
        m = self.metrics(pred_col, full_score, mask)
        m["arm"] = arm
        m["betas_chosen"] = picks
        return m


# ------------------------------------------------------- pair decomposition
def pair_support(D: Distill) -> Dict[str, Any]:
    """Structure of the (subject, object) partition the whole result rests on."""
    cnt = Counter(D.pair_id.tolist())
    sizes = sorted(cnt.values(), reverse=True)
    n = D.n
    cum, tops = 0, {}
    for frac in (0.01, 0.05, 0.10, 0.25):
        take = max(1, int(len(sizes) * frac))
        tops[f"rows_in_top_{int(frac*100)}pct_groups"] = sum(sizes[:take]) / n
    ent = []
    for gid, c in cnt.items():
        if c > 1:
            ix = (D.pair_id == gid).nonzero().squeeze(1)
            p = torch.bincount(D.y[ix], minlength=D.C).float()
            p = p / p.sum()
            nz = p[p > 0]
            ent.append(float(-(nz * nz.log()).sum()))
    bucket = Counter(D.B.bucket_of[int(c)] for c in D.y.tolist())
    return {"n_rows": n, "n_unique_pairs": len(cnt),
            "singleton_groups": sum(1 for c in cnt.values() if c == 1),
            "singleton_rows": sum(c for c in cnt.values() if c == 1),
            "singleton_row_rate": sum(c for c in cnt.values() if c == 1) / n,
            "max_group_size": sizes[0], "median_group_size": sizes[len(sizes) // 2],
            "mean_group_size": n / len(cnt),
            **tops,
            "mean_within_group_gt_entropy_nats": statistics.mean(ent) if ent else 0.0,
            "median_within_group_gt_entropy_nats": statistics.median(ent) if ent else 0.0,
            "frac_multirow_groups_with_zero_entropy":
                (sum(1 for e in ent if e < 1e-9) / len(ent)) if ent else 0.0,
            "gt_rows_by_bucket": dict(bucket),
            "prior_coverage": D.coverage}


def within_group(D: Distill, min_n: int = 5) -> Dict[str, Any]:
    """The 13.13% the model term does NOT explain by pair identity.

    Reported as an unexplained residual. This tool CANNOT tell whether it is
    visual signal, contextual signal, or annotation noise -- it can only bound
    how much predicate information it carries, and it does that by measuring
    whether the GT label varies within a group at all.
    """
    cnt = Counter(D.pair_id.tolist())
    gids = [g for g, c in cnt.items() if c >= min_n]
    md_v, gt_ent, top1_var, rank_var, n_rows = [], [], [], [], 0
    agree = 0
    for gid in gids:
        ix = (D.pair_id == gid).nonzero().squeeze(1)
        n_rows += int(ix.numel())
        sub = D.md[ix]
        md_v.append(float(sub.var(0, unbiased=False).mean()))
        p = torch.bincount(D.y[ix], minlength=D.C).float()
        p = p / p.sum()
        nz = p[p > 0]
        gt_ent.append(float(-(nz * nz.log()).sum()))
        t1 = sub.argmax(-1)
        top1_var.append(float(len(set(t1.tolist())) / ix.numel()))
        gs = sub.gather(1, D.gt_col[ix].unsqueeze(1))
        rk = (sub > gs).sum(-1) + 1
        rank_var.append(float(rk.float().std(unbiased=False)) if ix.numel() > 1 else 0.0)
        agree += int((D.y[ix] == D.y[ix][0]).all())
    # global variance decomposition of the model term
    G = int(D.pair_id.max()) + 1
    c = torch.zeros(G).index_add_(0, D.pair_id, torch.ones(D.n))
    s = torch.zeros(G, D.C).index_add_(0, D.pair_id, D.md)
    gmean = (s / c.clamp_min(1).unsqueeze(1))[D.pair_id]
    grand = D.md.mean(0, keepdim=True)
    ss = lambda x: float((x ** 2).sum())
    tot, btw, wth = ss(D.md - grand), ss(gmean - grand), ss(D.md - gmean)
    return {"min_group_size": min_n, "n_groups_analysed": len(gids),
            "n_rows_in_those_groups": n_rows,
            "variance_between_pairs_share": btw / tot,
            "variance_within_pairs_share": wth / tot,
            "mean_within_group_model_variance": statistics.mean(md_v) if md_v else 0.0,
            "mean_within_group_gt_entropy_nats": statistics.mean(gt_ent) if gt_ent else 0.0,
            "frac_groups_with_constant_gt": agree / max(1, len(gids)),
            "mean_distinct_model_top1_rate": statistics.mean(top1_var) if top1_var else 0.0,
            "mean_within_group_gt_rank_sd": statistics.mean(rank_var) if rank_var else 0.0}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p10_model_recalibration/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p27_pair_prior_distillation/distill.json")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--betas", default="0.0,0.1,0.2,0.3")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--estimable-only", action="store_true",
                    help="restrict every arm to rows whose (s,o) group is "
                         "estimable out of fold (docs/PAIR_PRIOR_DISTILLATION_"
                         "PREREGISTRATION.md). Recomputes the baseline, the tau "
                         "frontier and the R@50 floor on that subset.")
    ap.add_argument("--floor-delta", type=float, default=0.30,
                    help="registered floor = prior R@50(subset) - this, in "
                         "points. 0.30 reproduces R_FLOOR's own construction "
                         "(3k baseline 66.802 - 0.302 = 66.5).")
    ap.add_argument("--min-rows", type=int, default=80000,
                    help="gate G4: minimum estimable rows per partition.")
    args = ap.parse_args(argv)

    betas = [float(b) for b in args.betas.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    _log("=" * 116)
    _log("PAIR-PRIOR DISTILLATION -- CPU only, cache read-only, NO GPU")
    _log("=" * 116)
    B = Mech(args.dump, args.prior, "raw50")
    e = float((B.fixed_ensemble(0.0) - B.model).abs().max())
    assert e < 1e-4, f"model-term identity gate failed: {e:.3e}"
    _log(f"  images={B.n_images} GT rows={B.n_gt} classes={B.n_classes}  "
         f"[gate] model-term identity {e:.3e} OK")

    D0 = Distill(B, args.prior, args.tau, args.k, 0)
    curve = [{"R": (m := B.metrics(B.score(t, ALPHA_HIST, None)))["R"], "mR": m["mR"]}
             for t in CPA.TAUS]
    base = B.metrics(B.score(args.tau, ALPHA_HIST, None))
    ach = B.metrics(B.score(args.tau, ALPHA_HIST, B.model))
    _log(f"  prior-only tau={args.tau}: R@50 {base['R']*100:.3f} mR {base['mR']*100:.3f}"
         f"   |  achieved additive C': R@50 {ach['R']*100:.3f} mR {ach['mR']*100:.3f}"
         f"  pareto {CPA.pareto_gap(curve, ach['R'], ach['mR']):+.3f}")
    _log(f"  prior file coverage of these rows: subject {D0.coverage['subject']*100:.2f}%  "
         f"object {D0.coverage['object']*100:.2f}%  pair {D0.coverage['pair']*100:.2f}%")

    ps, wg = pair_support(D0), within_group(D0)
    _log(f"\n  PAIR SUPPORT: {ps['n_unique_pairs']:,} unique pairs over {ps['n_rows']:,} rows; "
         f"singleton rows {ps['singleton_row_rate']*100:.1f}%; mean group {ps['mean_group_size']:.2f}; "
         f"max {ps['max_group_size']}")
    _log(f"    rows in top 1%/5%/25% of groups: {ps['rows_in_top_1pct_groups']*100:.1f}% / "
         f"{ps['rows_in_top_5pct_groups']*100:.1f}% / {ps['rows_in_top_25pct_groups']*100:.1f}%")
    _log(f"    within-group GT entropy: mean {ps['mean_within_group_gt_entropy_nats']:.3f} nats; "
         f"{ps['frac_multirow_groups_with_zero_entropy']*100:.1f}% of multi-row groups have a CONSTANT GT")
    _log(f"  MODEL-TERM VARIANCE: between pairs {wg['variance_between_pairs_share']*100:.2f}%  "
         f"within pairs {wg['variance_within_pairs_share']*100:.2f}%")

    res: Dict[str, Any] = {"tool": "pair_prior_distillation", "tau": args.tau,
                           "k": args.k, "repeats": args.repeats, "betas": betas,
                           "R_floor": R_FLOOR, "seed": SEED,
                           "baseline_prior": base, "achieved_additive": ach,
                           "achieved_pareto": CPA.pareto_gap(curve, ach["R"], ach["mR"]),
                           "pair_support": ps, "within_group": wg,
                           "per_salt": [], "arms": {}}

    res["estimable_only"] = bool(args.estimable_only)
    if args.estimable_only:
        _log(f"\n{'-'*116}\n  ESTIMABLE-SUBSET MODE "
             f"(docs/PAIR_PRIOR_DISTILLATION_PREREGISTRATION.md)\n{'-'*116}")
        res["gates"] = []

    for salt in range(args.repeats):
        D = D0 if salt == 0 else Distill(B, args.prior, args.tau, args.k, salt)
        row: Dict[str, Any] = {"salt": salt, "arms": {}}

        if args.estimable_only:
            mask = D.estimable_mask()
            n_est = int(mask.sum())
            # Baseline, tau frontier and floor are ALL recomputed on the subset.
            s_base = B.metrics(B.score(args.tau, ALPHA_HIST, None), mask)
            s_ach = B.metrics(B.score(args.tau, ALPHA_HIST, B.model), mask)
            s_curve = [{"R": (mm := B.metrics(B.score(tt, ALPHA_HIST, None), mask))["R"],
                        "mR": mm["mR"]} for tt in CPA.TAUS]
            floor = s_base["R"] - args.floor_delta / 100.0

            # ---- validity gates, prereg §8 ----
            fb_out = max(D.fallback_frac(~(D.fold == f), mask & (D.fold == f))
                         for f in range(N_FOLDS))
            fb_in = max(D.fallback_frac(
                (~(D.fold == f)) & (~(D.fold == ((f + 1) % N_FOLDS))),
                mask & (D.fold == f)) for f in range(N_FOLDS))
            cos = D.residual_cosine(mask)
            g = {"salt": salt,
                 "G1_outer_fallback_frac": fb_out, "G1_pass": fb_out == 0.0,
                 "G2_residual_cosine": cos, "G2_pass": bool(cos < 0.70),
                 "G4_rows": n_est, "G4_pass": bool(n_est >= args.min_rows),
                 "inner_fallback_frac_reported_not_gated": fb_in}
            res["gates"].append(g)
            row.update({"n_rows": n_est,
                        "subset_prior": s_base, "subset_achieved": s_ach,
                        "subset_floor": floor,
                        "subset_achieved_pareto":
                            CPA.pareto_gap(s_curve, s_ach["R"], s_ach["mR"])})
            _log(f"\n  [salt {salt}] estimable rows {n_est:,} "
                 f"({n_est/D.n*100:.1f}% of {D.n:,})  "
                 f"subset prior R@50 {s_base['R']*100:.3f} mR {s_base['mR']*100:.3f}  "
                 f"floor {floor*100:.3f}")
            _log(f"    GATES  G1 outer fallback {fb_out*100:.3f}% "
                 f"{'PASS' if g['G1_pass'] else 'FAIL'}   "
                 f"G2 residual cosine {cos:.3f} {'PASS' if g['G2_pass'] else 'FAIL'}   "
                 f"G4 rows {'PASS' if g['G4_pass'] else 'FAIL'}   "
                 f"(inner fallback {fb_in*100:.2f}%, reported not gated)")
        else:
            mask, floor, s_base, s_curve = None, R_FLOOR, base, curve

        for arm in arms:
            m = D.run_arm(arm, betas, args.epochs, args.l2, floor, mask)
            m["pareto"] = CPA.pareto_gap(s_curve, m["R"], m["mR"])
            m["dR_points"] = (m["R"] - s_base["R"]) * 100.0
            m["dmR_points"] = (m["mR"] - s_base["mR"]) * 100.0
            m["meets_R_floor"] = bool(m["R"] >= floor)
            row["arms"][arm] = m
        res["per_salt"].append(row)
        _log(f"\n  [salt {salt}] " + "  ".join(
            f"{a}:{row['arms'][a]['pareto']:+.2f}" for a in arms))

    if args.estimable_only:
        ref = {"R": statistics.mean(r["subset_prior"]["R"] for r in res["per_salt"]),
               "mR": statistics.mean(r["subset_prior"]["mR"] for r in res["per_salt"])}
        res["subset_baseline_mean"] = ref
        res["subset_rows_mean"] = statistics.mean(r["n_rows"] for r in res["per_salt"])
        scope = (f"ESTIMABLE SUBSET, {res['subset_rows_mean']:,.0f} rows mean; "
                 f"floor = subset prior - {args.floor_delta:.2f} pts")
    else:
        ref, scope = base, f"FULL population; floor {R_FLOOR*100:.1f}"
    _log(f"\n{'-'*116}\n  ARMS, mean over {args.repeats} fold partitions "
         f"(prior-only baseline R@50 {ref['R']*100:.3f} mR {ref['mR']*100:.3f})"
         f"\n  scope: {scope}\n{'-'*116}")
    _log(f"  {'arm':>18} {'R@50':>8} {'mR@50':>8} {'pareto':>8} {'floor':>7} {'head':>6} "
         f"{'body':>6} {'tail':>6} {'meanRank':>9} {'MRR':>6} {'R@5':>6} {'chg%':>6}")
    for arm in arms:
        vals = [r["arms"][arm] for r in res["per_salt"]]
        agg = {f: statistics.mean([v[f] for v in vals]) for f in
               ("R", "mR", "pareto", "head_mR", "body_mR", "tail_mR",
                "mean_gt_rank", "MRR", "R_at_5", "frac_argmax_changed")}
        agg["pareto_sd"] = statistics.stdev([v["pareto"] for v in vals]) if len(vals) > 1 else 0.0
        agg["floor_held"] = sum(1 for v in vals if v["meets_R_floor"])
        agg["dR_points"] = (agg["R"] - ref["R"]) * 100.0
        agg["dmR_points"] = (agg["mR"] - ref["mR"]) * 100.0
        res["arms"][arm] = agg
        _log(f"  {arm:>18} {agg['R']*100:>8.3f} {agg['mR']*100:>8.3f} {agg['pareto']:>+8.3f}"
             f" {agg['floor_held']:>4}/{args.repeats} {agg['head_mR']*100:>6.2f}"
             f" {agg['body_mR']*100:>6.2f} {agg['tail_mR']*100:>6.2f}"
             f" {agg['mean_gt_rank']:>9.3f} {agg['MRR']:>6.3f} {agg['R_at_5']*100:>6.2f}"
             f" {agg['frac_argmax_changed']*100:>6.2f}")

    def gap(a: str, b: str) -> Optional[float]:
        if a in res["arms"] and b in res["arms"]:
            return res["arms"][a]["pareto"] - res["arms"][b]["pareto"]
        return None

    _log(f"\n  KEY CONTRASTS (pareto points, mean over partitions)")
    for a, b, why in (("G_model", "D_pair", "checkpoint over the existing pair prior"),
                      ("G_model", "E_backoff", "checkpoint over hierarchical smoothing"),
                      ("G_model", "F_pair_foldfit", "checkpoint over fold-fitted pair stats"),
                      ("G_model", "G_pairmean", "the WITHIN-group part of the model term"),
                      ("G_pairmean", "D_pair", "the PAIR part of the model term"),
                      ("G_residual", "D_pair", "residual alone over the pair prior"),
                      ("G_model", "null_pair_matched", "image content (p26 replication)"),
                      ("G_model", "null_shuffled", "pair identity in the model term")):
        g = gap(a, b)
        if g is not None:
            res.setdefault("contrasts", {})[f"{a}_minus_{b}"] = g
            _log(f"    {a:>16} - {b:<18} {g:>+8.3f}   {why}")

    if args.estimable_only:
        VISION_FREE = ["A_global", "B_subject", "C_object",
                       "D_pair", "E_backoff", "F_pair_foldfit"]
        gates_pass = all(g["G1_pass"] and g["G2_pass"] and g["G4_pass"]
                         for g in res["gates"])
        # G5: the identity-destroying null must sit at or below the frontier.
        g5 = ("null_shuffled" not in res["arms"]
              or res["arms"]["null_shuffled"]["pareto"] <= 0.0)
        for g in res["gates"]:
            g["G5_pass"] = bool(g5)
        P_G = res["arms"].get("G_model", {}).get("pareto")
        elig, inelig = [], []
        for a in VISION_FREE:
            if a not in res["arms"]:
                continue
            # prereg §7: floor held on ">= 4 of 5" partitions. Expressed as a
            # fraction so a smoke run with --repeats 1 does not silently make
            # every arm ineligible; at the registered --repeats 5 this is 4.
            need = max(1, math.ceil(0.8 * args.repeats))
            (elig if res["arms"][a]["floor_held"] >= need else inelig).append(
                (res["arms"][a]["pareto"], a))
        best_e = max(elig) if elig else None
        best_any = max(elig + inelig) if (elig or inelig) else None
        if P_G is None or best_any is None:
            verdict, P_V, who = "VOID (missing arms)", None, None
        elif best_e is None:
            verdict, (P_V, who) = "NOT EXPLAINED", best_any
        else:
            P_V, who = best_e
            verdict = ("EXPLAINED" if P_V >= P_G - 0.50
                       else "MOSTLY EXPLAINED" if P_V >= P_G - 1.50
                       else "NOT EXPLAINED")
        if not (gates_pass and g5):
            verdict = "VOID (validity gate failed)"
        res["verdict"] = {
            "floor_partitions_required": max(1, math.ceil(0.8 * args.repeats)),
            "verdict": verdict, "P_G_model": P_G, "P_V_best_vision_free": P_V,
            "best_vision_free_arm": who,
            "best_arm_floor_eligible": bool(best_e is not None),
            "gap_P_G_minus_P_V": (None if (P_G is None or P_V is None) else P_G - P_V),
            "gates_all_pass": bool(gates_pass and g5),
            "thresholds": {"EXPLAINED": "P_V >= P_G - 0.50",
                           "MOSTLY": "P_V in [P_G - 1.50, P_G - 0.50)",
                           "NOT_EXPLAINED": "P_V < P_G - 1.50"}}
        _log(f"\n{'-'*116}\n  PRE-REGISTERED VERDICT\n{'-'*116}")
        _log(f"    P_G  (G_model)                 {P_G:+.3f}" if P_G is not None else "    P_G  n/a")
        _log(f"    P_V  (best floor-eligible      {P_V:+.3f}   [{who}]"
             if P_V is not None else "    P_V  n/a")
        _log(f"          vision-free pair arm)")
        if P_G is not None and P_V is not None:
            _log(f"    P_G - P_V                      {P_G - P_V:+.3f}")
        _log(f"    gates all pass                 {gates_pass and g5}")
        _log(f"\n    VERDICT: {verdict}")
        _log(f"\n    Scope: frequent-pair rows only. The ~{100 - res['subset_rows_mean']/D0.n*100:.0f}% "
             f"singleton-pair rows are unaddressable by ANY pair-conditioned\n"
             f"    estimator and are excluded here; that is a property of VG150's "
             f"pair distribution, not an estimator detail.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
