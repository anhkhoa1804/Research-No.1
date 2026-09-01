#!/usr/bin/env python
"""H2 -- does the OBJECTIVE, not the features, limit within-pair discrimination?

PRE-REGISTERED INLINE, before the run:
  SUPPORTED  contrastive WPRD >= CE WPRD + 0.02
  WEAK       gain in [0.005, 0.02)
  REFUTED    gain < 0.005
0.02 is the classifier-vs-text head gap (0.0186) rounded up -- a difference
already large enough to move the operating point in runs/p34 -- and is about
four CI half-widths on these estimates.

THE DESIGN. Everything is held fixed except the loss:

  features   the same 19 box-geometry numbers
  capacity   the same 2-layer 256-unit MLP
  data       the same TRAIN split rows
  eval       the same WPRD on the same validation cells

  objective A   plain cross-entropy      <- what SGG training actually does
  objective B   within-group contrastive <- directly optimises WPRD's quantity

B's loss is the AUC surrogate WPRD measures. For two TRAIN rows i, j sharing an
(s,o) group with y_i != y_j:

    margin = (f_i[y_i] - f_i[y_j]) - (f_j[y_i] - f_j[y_j])
    loss   = softplus(-margin)

The prior cancels in that double difference exactly as it does in WPRD, so the
objective cannot be satisfied by learning P(p|s,o) -- which is the whole point.

If B beats A on the SAME features, the limitation is the objective, not the
representation. This is run on geometry because those features exist now; the
identical comparison transfers to rel_feat once runs/p36 lands.

CPU only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
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
WPD = _load("within_pair_discrimination")
GEO = _load("wprd_geometry_control")
Mech = MECH.Mech


def _log(m: str = "") -> None:
    print(m, flush=True)


def train_groups(path: str, classes: List[str]):
    """(s,o) group id per TRAIN relationship row, aligned with train_split_geometry."""
    idx = {c: i for i, c in enumerate(classes)}
    gids: List[int] = []
    key: Dict[str, int] = {}
    import json as _json
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = _json.loads(line)
            boxes = d.get("obj_boxes") or []
            rels = d.get("relationships") or []
            if not boxes or not rels:
                continue
            names = [str((o.get("names") or [""])[0]).strip().lower()
                     for o in (d.get("objects") or [])]
            nb = len(boxes)
            for r in rels:
                c = idx.get(str(r.get("predicate", "")).strip().lower())
                a, b = int(r.get("subject_id", -1)), int(r.get("object_id", -1))
                if c is None or not (0 <= a < nb and 0 <= b < nb):
                    continue
                sa = names[a] if a < len(names) else ""
                ob = names[b] if b < len(names) else ""
                k = f"{sa}||{ob}"
                gids.append(key.setdefault(k, len(key)))
    return torch.tensor(gids), len(key)


def mlp(d_in, hidden, C, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, C))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", default="runs/p48_objective_ablation/obj.json")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pairs-per-epoch", type=int, default=400000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("H2 OBJECTIVE ABLATION -- same features, same capacity, different loss. NO GPU")
    _log("=" * 100)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    Xr = GEO.geometry_features_raw(B)
    Xt_raw, yt = GEO.train_split_geometry(args.train_jsonl, list(B.classes))
    Xt, tst = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tst)
    gt, G = train_groups(args.train_jsonl, list(B.classes))
    assert gt.numel() == yt.numel(), f"group/label misalignment {gt.numel()} vs {yt.numel()}"
    _log(f"  train rows {Xt.shape[0]:,}  groups {G:,}   val rows {Xv.shape[0]:,}")

    # index: group -> {class -> rows}
    byg: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for r, (g, c) in enumerate(zip(gt.tolist(), yt.tolist())):
        byg[g][c].append(r)
    usable = [(g, list(d.keys())) for g, d in byg.items() if len(d) >= 2]
    _log(f"  TRAIN groups with >=2 distinct predicates (usable for contrastive): "
         f"{len(usable):,} of {G:,}")

    gen = torch.Generator().manual_seed(0)

    def sample_pairs(n):
        gi = torch.randint(len(usable), (n,), generator=gen).tolist()
        I, J, A, Bc = [], [], [], []
        for k in gi:
            g, cs = usable[k]
            ci = torch.randint(len(cs), (2,), generator=gen).tolist()
            if ci[0] == ci[1]:
                continue
            a, b = cs[ci[0]], cs[ci[1]]
            ra, rb = byg[g][a], byg[g][b]
            I.append(ra[torch.randint(len(ra), (1,), generator=gen).item()])
            J.append(rb[torch.randint(len(rb), (1,), generator=gen).item()])
            A.append(a)
            Bc.append(b)
        return (torch.tensor(I), torch.tensor(J), torch.tensor(A), torch.tensor(Bc))

    results: Dict[str, Any] = {"tool": "objective_ablation", "arms": {}}
    scores: Dict[str, torch.Tensor] = {}

    # ---------- A: plain CE ----------
    _log(f"\n  [A] plain cross-entropy")
    netA = mlp(Xt.shape[1], args.hidden, B.n_classes, 0)
    opt = torch.optim.AdamW(netA.parameters(), lr=args.lr, weight_decay=args.l2)
    n, bs = Xt.shape[0], 8192
    for ep in range(args.epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(netA(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ix)
        if (ep + 1) % 10 == 0:
            _log(f"      epoch {ep+1:>3}/{args.epochs}  CE {tot/n:.4f}")
    netA.eval()
    with torch.no_grad():
        scores["A_cross_entropy"] = netA(Xv)

    # ---------- B: within-group contrastive ----------
    _log(f"\n  [B] within-group contrastive (the WPRD surrogate)")
    netB = mlp(Xt.shape[1], args.hidden, B.n_classes, 0)
    opt = torch.optim.AdamW(netB.parameters(), lr=args.lr, weight_decay=args.l2)
    pb = 16384
    for ep in range(args.epochs):
        I, J, A, C2 = sample_pairs(args.pairs_per_epoch)
        perm = torch.randperm(len(I), generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, len(I), pb):
            ix = perm[i:i + pb]
            ii, jj, aa, cc = I[ix], J[ix], A[ix], C2[ix]
            opt.zero_grad()
            fi, fj = netB(Xt[ii]), netB(Xt[jj])
            m = ((fi.gather(1, aa.view(-1, 1)) - fi.gather(1, cc.view(-1, 1)))
                 - (fj.gather(1, aa.view(-1, 1)) - fj.gather(1, cc.view(-1, 1))))
            loss = torch.nn.functional.softplus(-m).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ix)
            cnt += len(ix)
        if (ep + 1) % 10 == 0:
            _log(f"      epoch {ep+1:>3}/{args.epochs}  contrastive {tot/cnt:.4f}"
                 f"   ({cnt:,} pairs)")
    netB.eval()
    with torch.no_grad():
        scores["B_within_group_contrastive"] = netB(Xv)

    scores["prior (must be 0.5)"] = B.prior
    _log(f"\n{'-'*100}")
    _log(f"  {'arm':>34} {'WPRD':>8} {'weighted':>9} {'95% CI':>22}")
    _log(f"{'-'*100}")
    keep = {}
    for name, s in scores.items():
        r = WPD.wprd(Gs, s, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        b = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                         for _ in range(args.boot)])
        lo, hi = torch.quantile(b, torch.tensor([0.025, 0.975],
                                                dtype=torch.float64)).tolist()
        keep[name] = v
        results["arms"][name] = {"macro": r["wprd_macro"],
                                 "weighted": r["wprd_weighted"], "ci95": [lo, hi]}
        _log(f"  {name:>34} {r['wprd_macro']:>8.4f} {r['wprd_weighted']:>9.4f}  "
             f"[{lo:.4f}, {hi:.4f}]")

    d = keep["B_within_group_contrastive"] - keep["A_cross_entropy"]
    g = torch.Generator().manual_seed(2)
    b = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean()
                     for _ in range(args.boot)])
    lo, hi = torch.quantile(b, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    gain = float(d.mean())
    verdict = ("SUPPORTED" if gain >= 0.02 else
               "WEAK" if gain >= 0.005 else "REFUTED")
    results["contrast"] = {"B_minus_A": gain, "ci95": [lo, hi], "verdict": verdict}
    _log(f"\n  PAIRED  B(contrastive) - A(cross-entropy) = {gain:+.4f} "
         f"[{lo:+.4f}, {hi:+.4f}]  "
         f"{'excludes 0' if (lo > 0 or hi < 0) else 'INCLUDES 0'}")
    _log(f"\n  PRE-REGISTERED VERDICT: {verdict}   "
         f"(SUPPORTED >= +0.020, WEAK >= +0.005)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
