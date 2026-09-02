#!/usr/bin/env python
"""p56 -- does within-pair discrimination scale with SUPERVISION SUPPLY?

Pre-registered in docs/SUPERVISION_SCALING_PREREGISTRATION.md, committed before
this file existed. Thresholds are quoted from it and NOT recomputed here.

  PRIMARY   rho_model = Spearman(log1p(n_train_contrasts), cell WPRD), text head
    H8-SUPPORTED  rho >= +0.15, p < 0.05, AND top support bin >= 0.5961
    H8-WEAK       rho >= +0.05, p < 0.05, top bin below 0.5961
    H8-REFUTED    rho < +0.05 or p >= 0.05

  SECONDARY rho_perpair from Part C
    LEVER / NOT A LEVER at +0.15, p < 0.05

  TERTIARY  rho_model - rho_geometry

THE LOGIC. p37 says the representation is the limit, but measures it as it
exists. If H8 (supervision scarcity) is why, discrimination should be better for
exactly those (s,o) pairs the TRAIN split taught most. Geometry is the control:
it is fitted globally over 19 shared features and accumulates no per-pair
capacity, so it cannot manufacture that signature from the population alone.

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
GEO = _load("wprd_geometry_control")
Mech = MECH.Mech

BINS = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10000),
        (10000, float("inf"))]
BIN_NAMES = ["0", "1-9", "10-99", "100-999", "1k-10k", ">=10k"]
G_GEOMETRY = 0.5961


def _log(m: str = "") -> None:
    print(m, flush=True)


def _rank(v: torch.Tensor) -> torch.Tensor:
    """Average ranks, ties shared. Pulled out so it can be computed ONCE."""
    o = v.argsort()
    r = torch.empty_like(o, dtype=torch.float64)
    r[o] = torch.arange(len(v), dtype=torch.float64)
    s = v[o]
    d = torch.ones(len(s), dtype=torch.bool)
    d[1:] = s[1:] != s[:-1]
    grp = torch.cumsum(d.long(), 0) - 1
    cnt = torch.zeros(int(grp[-1]) + 1, dtype=torch.float64).index_add_(
        0, grp, torch.ones(len(s), dtype=torch.float64))
    tot = torch.zeros(int(grp[-1]) + 1, dtype=torch.float64).index_add_(
        0, grp, r[o])
    r[o] = (tot / cnt)[grp]
    return r


def _rho_from_ranks(rx: torch.Tensor, ry: torch.Tensor) -> float:
    a = rx - rx.mean()
    b = ry - ry.mean()
    d = float((a.norm() * b.norm()).clamp_min(1e-12))
    return float((a * b).sum() / d)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _rho_from_ranks(_rank(x.double()), _rank(y.double()))


def perm_p(x: torch.Tensor, y: torch.Tensor, obs: float, draws: int,
           seed: int = 0) -> float:
    """Permutation p on Spearman.

    The ranks of a permutation of y ARE the permutation of y's ranks, so both
    rank vectors are computed once and only the index shuffle is repeated.
    Without this the tie-averaging pass runs 2000 times per arm.
    """
    g = torch.Generator().manual_seed(seed)
    rx, ry = _rank(x.double()), _rank(y.double())
    hits = 0
    for _ in range(draws):
        if abs(_rho_from_ranks(rx, ry[torch.randperm(len(ry), generator=g)])) >= abs(obs):
            hits += 1
    return (hits + 1) / (draws + 1)


def train_supervision(path: str, classes: List[str]
                      ) -> Tuple[Dict[str, Dict[str, int]], torch.Tensor,
                                 torch.Tensor, List[str]]:
    """Per-(s,o) training supply, plus train geometry rows keyed by pair.

    Returns (supply_by_key, X_train_geom, y_train, key_of_train_row).
    The key convention is `f"{subj}||{obj}"` on raw lowercased names, which is
    exactly what within_pair_discrimination.Groups builds (gate W4).
    """
    idx = {c: i for i, c in enumerate(classes)}
    per: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    XS: List[torch.Tensor] = []
    YS: List[int] = []
    KS: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            boxes = torch.tensor(d.get("obj_boxes") or [], dtype=torch.float32)
            rels = d.get("relationships") or []
            objs = d.get("objects") or []
            if boxes.numel() == 0 or not rels:
                continue
            names: Dict[int, str] = {}
            for k, o in enumerate(objs):
                nm = (o.get("names") or [""])[0]
                names[int(o.get("object_id", k))] = str(nm).strip().lower()
            W = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
            H = float(boxes[:, 3].max() - boxes[:, 1].min()) or 1.0
            si, oi, yy, kk = [], [], [], []
            nb = boxes.shape[0]
            for r in rels:
                c = idx.get(str(r.get("predicate", "")).strip().lower())
                a, b = int(r.get("subject_id", -1)), int(r.get("object_id", -1))
                if c is None or not (0 <= a < nb and 0 <= b < nb):
                    continue
                key = f"{names.get(a,'')}||{names.get(b,'')}"
                si.append(a); oi.append(b); yy.append(c); kk.append(key)
                per[key][c] += 1
            if not yy:
                continue
            XS.append(GEO._geom(boxes[si], boxes[oi], W, H))
            YS.extend(yy)
            KS.extend(kk)
    supply: Dict[str, Dict[str, int]] = {}
    for k, cc in per.items():
        n = sum(cc.values())
        vals = list(cc.values())
        contr = sum(vals[i] * vals[j]
                    for i in range(len(vals)) for j in range(i + 1, len(vals)))
        supply[k] = {"n_train_rows": n, "n_train_predicates": len(cc),
                     "n_train_contrasts": contr}
    return supply, torch.cat(XS, 0), torch.tensor(YS), KS


def cells_with_group(Gs, score: torch.Tensor, cap: int, seed: int
                     ) -> List[Dict[str, Any]]:
    """One record per (group, predicate-pair) cell, carrying its group id.

    wprd_stratified.cells() does not return the group, and the join to training
    supply needs it, so the enumeration is repeated here with the SAME cap,
    ordering and RNG stream as within_pair_discrimination.wprd().
    """
    gen = torch.Generator().manual_seed(seed)
    out: List[Dict[str, Any]] = []
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        rows = torch.tensor(Gs.rows_of[p])
        yy = Gs.y[rows]
        byc = {c: rows[yy == c] for c in cls}
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                ra, rb = byc[a], byc[b]
                if len(ra) > cap:
                    ra = ra[torch.randperm(len(ra), generator=gen)[:cap]]
                if len(rb) > cap:
                    rb = rb[torch.randperm(len(rb), generator=gen)[:cap]]
                sa = (score[ra, a] - score[ra, b]).double()
                sb = (score[rb, a] - score[rb, b]).double()
                out.append({"auc": WPD.auc(sa, sb), "w": len(ra) * len(rb),
                            "gid": p, "a": a, "b": b})
    return out


def bin_of(n: int) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= n < hi:
            return i
    return len(BINS) - 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p56_supervision_scaling/scal.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--min-train-rows", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args(argv)

    _log("=" * 112)
    _log("p56 SUPERVISION SCALING -- does discrimination track supervision supply? "
         "CPU only, NO GPU")
    _log("=" * 112)

    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    gates: List[Dict[str, Any]] = []
    _log(f"  rows {Gs.n:,}  groups {Gs.G:,}  decidable "
         f"{int(Gs.decidable.sum()):,}")

    _log(f"  reading train supervision supply from {args.train_jsonl} ...")
    supply, Xt_raw, yt, KS = train_supervision(args.train_jsonl, list(B.classes))
    _log(f"  train rows {len(yt):,}  distinct (s,o) keys {len(supply):,}")

    # ---- W4: the join actually joins ----
    val_keys = [Gs.key_of[p] for p in range(Gs.G)]
    hit = sum(1 for k in val_keys if k in supply)
    w4 = hit > 0.5 * len(val_keys)
    gates.append({"gate": "W4 train->val group-key join", "pass": bool(w4),
                  "detail": f"{hit}/{len(val_keys)} validation groups found in train "
                            f"({100*hit/max(1,len(val_keys)):.1f}%)"})
    _log(f"  W4 join {hit:,}/{len(val_keys):,} validation groups present in train "
         f"({100*hit/max(1,len(val_keys)):.1f}%)  {'PASS' if w4 else 'FAIL'}")

    contr_of_gid = torch.tensor(
        [float(supply.get(Gs.key_of[p], {}).get("n_train_contrasts", 0))
         for p in range(Gs.G)], dtype=torch.float64)
    rows_of_gid = torch.tensor(
        [float(supply.get(Gs.key_of[p], {}).get("n_train_rows", 0))
         for p in range(Gs.G)], dtype=torch.float64)

    # ---- arms ----
    _log(f"\n  fitting geometry references (train-fitted) ...")
    Xr = GEO.geometry_features_raw(B)
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    Wg = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
    o = torch.optim.LBFGS([Wg], max_iter=200, history_size=10,
                          line_search_fn="strong_wolfe")

    def cl():
        o.zero_grad()
        l = torch.nn.functional.cross_entropy(Xt @ Wg, yt) + 1e-4 * (Wg * Wg).sum()
        l.backward()
        return l
    o.step(cl)
    geo_lin = (Xv @ Wg).detach()

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, B.n_classes))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    g0 = torch.Generator().manual_seed(0)
    n, bs = Xt.shape[0], 8192
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=g0)
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Xt[ix]), yt[ix]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        geo_mlp = net(Xv)

    gr = torch.Generator().manual_seed(3)
    arms = {
        "text_head (evaluated)": B.fixed_ensemble(0.0),
        "classifier_head (discarded)": B.fixed_ensemble(1.0),
        "geometry_linear": geo_lin,
        "geometry_mlp": geo_mlp,
        "prior (must be 0.5)": B.prior,
        "random (must be ~0.5)": torch.randn(B.n_gt, B.n_classes, generator=gr),
    }

    res: Dict[str, Any] = {"tool": "supervision_scaling", "gates": gates,
                           "bins": BIN_NAMES, "arms": {}}
    cell_cache: Dict[str, List[Dict[str, Any]]] = {}
    _log(f"\n{'-'*112}")
    _log(f"  WPRD BY TRAINING SUPERVISION SUPPLY of the (s,o) pair "
         f"(n_train_contrasts)")
    _log(f"{'-'*112}")
    _log(f"  {'arm':>30} " + " ".join(f"{b:>9}" for b in BIN_NAMES) + f" {'all':>9}")
    for name, sc in arms.items():
        cc = cells_with_group(Gs, sc, args.cap, 0)
        cell_cache[name] = cc
        per_bin: List[List[float]] = [[] for _ in BINS]
        for c in cc:
            per_bin[bin_of(int(contr_of_gid[c["gid"]]))].append(c["auc"])
        means = [sum(v) / len(v) if v else float("nan") for v in per_bin]
        allm = sum(c["auc"] for c in cc) / len(cc)
        res["arms"][name] = {"by_bin": means,
                             "n_cells_by_bin": [len(v) for v in per_bin],
                             "macro_all": allm}
        _log(f"  {name:>30} " + " ".join(f"{m:>9.4f}" for m in means)
             + f" {allm:>9.4f}")
    _log(f"  {'n cells':>30} " + " ".join(
        f"{len(v):>9,}" for v in
        [[c for c in cell_cache['prior (must be 0.5)']
          if bin_of(int(contr_of_gid[c['gid']])) == i] for i in range(len(BINS))]))

    # ---- W1, W2, W3 ----
    pb = res["arms"]["prior (must be 0.5)"]["by_bin"]
    w1 = all((math.isnan(v) or abs(v - 0.5) < 1e-6) for v in pb)
    rnd = res["arms"]["random (must be ~0.5)"]["macro_all"]
    w2 = 0.49 <= rnd <= 0.51
    ncells = len(cell_cache["prior (must be 0.5)"])
    w3 = sum(res["arms"]["prior (must be 0.5)"]["n_cells_by_bin"]) == ncells
    for g_, p_, d_ in (("W1 prior 0.5000 in every bin", w1, str([round(v, 6) for v in pb])),
                       ("W2 random at chance", w2, f"{rnd:.4f}"),
                       ("W3 bins partition cells", w3, f"{ncells} cells")):
        gates.append({"gate": g_, "pass": bool(p_), "detail": d_})
        _log(f"  {g_}: {'PASS' if p_ else 'FAIL'}  {d_}")

    # ---- PRIMARY / TERTIARY ----
    _log(f"\n{'-'*112}\n  SPEARMAN(log1p(n_train_contrasts), cell WPRD)\n{'-'*112}")
    rhos: Dict[str, Dict[str, float]] = {}
    for name in arms:
        cc = cell_cache[name]
        x = torch.log1p(torch.tensor([float(contr_of_gid[c["gid"]]) for c in cc],
                                     dtype=torch.float64))
        yv = torch.tensor([c["auc"] for c in cc], dtype=torch.float64)
        r = spearman(x, yv)
        p = perm_p(x, yv, r, args.draws)
        rhos[name] = {"rho": r, "p": p, "n": len(cc)}
        _log(f"  {name:>30}  rho = {r:+.4f}   p = {p:.4f}   n = {len(cc):,}"
             f"   {'significant' if p < 0.05 else 'ns'}")
    res["spearman_vs_train_contrasts"] = rhos

    # ---- EXPLORATORY, NOT A CRITERION ----------------------------------
    # Added AFTER seeing the pilot, and labelled so it can never be read as a
    # registered result. The pilot showed the classifier head's BIN MEANS rising
    # monotonically across support bins while the registered cell-level Spearman
    # read ~0. Cell AUCs are computed from very few rows and are correspondingly
    # noisy, so a cell-level rank correlation can miss a real trend in the means.
    # This reports the cell-count-weighted trend across bin means. It changes no
    # threshold and decides no verdict.
    _log(f"\n{'-'*112}")
    _log("  EXPLORATORY (added after the pilot; NOT a criterion): "
         "cell-weighted trend across bin means")
    _log(f"{'-'*112}")
    expl: Dict[str, Any] = {}
    for name in arms:
        bm = res["arms"][name]["by_bin"]
        nb = res["arms"][name]["n_cells_by_bin"]
        pts = [(i, v, n) for i, (v, n) in enumerate(zip(bm, nb))
               if n > 0 and not math.isnan(v)]
        if len(pts) < 3:
            continue
        xs = torch.tensor([float(i) for i, _, _ in pts], dtype=torch.float64)
        ys = torch.tensor([v for _, v, _ in pts], dtype=torch.float64)
        ws = torch.tensor([float(n) for _, _, n in pts], dtype=torch.float64)
        ws = ws / ws.sum()
        mx = float((ws * xs).sum())
        my = float((ws * ys).sum())
        cov = float((ws * (xs - mx) * (ys - my)).sum())
        var = float((ws * (xs - mx) ** 2).sum())
        slope = cov / max(var, 1e-12)
        expl[name] = {"slope_per_bin": slope,
                      "first_last_delta": float(ys[-1] - ys[0])}
        _log(f"  {name:>30}  slope {slope:+.5f} WPRD per bin   "
             f"top-minus-bottom {float(ys[-1]-ys[0]):+.4f}")
    res["exploratory_bin_trend_not_a_criterion"] = expl

    # ---- PART C: what per-pair supervision actually buys ----
    _log(f"\n{'-'*112}\n  PART C -- per-pair discriminator fitted on THAT PAIR's "
         f"train rows only (geometry)\n{'-'*112}")
    kid: Dict[str, int] = {}
    tk = torch.tensor([kid.setdefault(k, len(kid)) for k in KS])
    rows_by_key: Dict[int, List[int]] = defaultdict(list)
    for i, k in enumerate(tk.tolist()):
        rows_by_key[k].append(i)
    percell: List[Tuple[float, float]] = []
    n_fit = 0
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        key = Gs.key_of[p]
        ki = kid.get(key)
        if ki is None:
            continue
        tr = rows_by_key[ki]
        if len(tr) < args.min_train_rows:
            continue
        tri = torch.tensor(tr)
        ytr = yt[tri]
        if len(set(ytr.tolist())) < 2:
            continue
        A = Xt[tri]
        AtA = A.T @ A + 1e-2 * A.shape[0] * torch.eye(A.shape[1])
        Y = torch.zeros(A.shape[0], B.n_classes)
        Y[torch.arange(A.shape[0]), ytr] = 1.0
        Wm = torch.linalg.solve(AtA, A.T @ Y)
        rows = torch.tensor(Gs.rows_of[p])
        sc = Xv[rows] @ Wm
        yy = Gs.y[rows]
        n_fit += 1
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                ra = (yy == a).nonzero(as_tuple=True)[0]
                rb = (yy == b).nonzero(as_tuple=True)[0]
                if len(ra) == 0 or len(rb) == 0:
                    continue
                sa = (sc[ra, a] - sc[ra, b]).double()
                sb = (sc[rb, a] - sc[rb, b]).double()
                percell.append((WPD.auc(sa, sb), float(rows_of_gid[p])))
    if percell:
        pv = torch.tensor([v for v, _ in percell], dtype=torch.float64)
        ps = torch.log1p(torch.tensor([s for _, s in percell], dtype=torch.float64))
        rpp = spearman(ps, pv)
        ppp = perm_p(ps, pv, rpp, args.draws)
        _log(f"  pairs fitted {n_fit:,}   cells {len(percell):,}   "
             f"mean per-pair WPRD {float(pv.mean()):.4f}")
        _log(f"  rho(log1p(n_train_rows of pair), per-pair WPRD) = {rpp:+.4f}"
             f"   p = {ppp:.4f}   {'significant' if ppp < 0.05 else 'ns'}")
        res["part_c"] = {"n_pairs_fitted": n_fit, "n_cells": len(percell),
                         "mean_wprd": float(pv.mean()), "rho": rpp, "p": ppp}
    else:
        res["part_c"] = {"n_pairs_fitted": 0}
        rpp, ppp = float("nan"), 1.0
        _log("  no pair met the minimum training-row threshold")

    # ---- verdicts ----
    rm, pm = rhos["text_head (evaluated)"]["rho"], rhos["text_head (evaluated)"]["p"]
    rg = rhos["geometry_linear"]["rho"]
    topbin = res["arms"]["text_head (evaluated)"]["by_bin"][-1]
    prim = ("H8-SUPPORTED" if (rm >= 0.15 and pm < 0.05
                               and not math.isnan(topbin) and topbin >= G_GEOMETRY)
            else "H8-WEAK" if (rm >= 0.05 and pm < 0.05)
            else "H8-REFUTED")
    sec = ("LEVER" if (not math.isnan(rpp) and rpp >= 0.15 and ppp < 0.05)
           else "NOT A LEVER")
    allg = all(g["pass"] for g in gates)
    res["verdict"] = {"rho_model": rm, "p_model": pm, "rho_geometry": rg,
                      "differential": rm - rg, "top_bin_model": topbin,
                      "rho_perpair": rpp, "p_perpair": ppp,
                      "primary": prim, "secondary": sec, "gates_all_pass": allg}
    _log(f"\n{'-'*112}\n  PRE-REGISTERED VERDICT\n{'-'*112}")
    _log(f"    PRIMARY   rho_model {rm:+.4f} (p {pm:.4f}), top bin "
         f"{topbin:.4f} vs G {G_GEOMETRY:.4f}   -> {prim}")
    _log(f"    SECONDARY rho_perpair {rpp:+.4f} (p {ppp:.4f})   -> {sec}")
    _log(f"    TERTIARY  rho_model - rho_geometry = {rm - rg:+.4f}")
    _log(f"    gates all pass: {allg}")
    if not allg:
        _log(f"    -> VOID: a validity gate failed; no number above is reportable")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
