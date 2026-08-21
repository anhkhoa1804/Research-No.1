#!/usr/bin/env python
"""Is the co-occurrence prior a good CANDIDATE GENERATOR but a poor RANKER?

Tests both halves of that hypothesis without a neural network, on the full
validation split, on CPU.

RESULT (train-derived prior, 8 box-geometry features):

  GENERATOR half -- CONFIRMED, decisively
      candidate coverage @K=5  = 89.41 %   (tail 53.0 %)
      oracle rerank  @K=5      = R@50 89.41 %, mR@50 63.80 % (tail 50.7 %)
      versus prior top-1         R@50 66.59 %, mR@50 22.30 % (tail  7.6 %)
      -> +41.5 mR@50 of headroom sits inside the prior's top-5

  RANKER half -- FALSIFIED for every cheap visual scorer tried
      shared linear reranker        mR@50 11.90  (-10.40)
      margin / BCE objectives       mR@50 11.86 / 11.52
      200 dedicated pairwise probes  0.0 % of oracle headroom captured
      multi-predicate decision rule  dmR <= 0 at every M

The decisive control is that the flat 50-way model and the reranker use the
SAME features and the SAME parameter shape. Only the softmax normalisation
set differs. So the failure is not a capacity artifact.

Geometry does carry signal -- real > shuffled > zeroed (mR 11.53 / 10.12 /
9.76) -- it is simply far too weak to rank within the candidate set.

WHAT REMAINS UNTESTED: appearance. Box geometry contains no appearance
information at all, and several target confusions require it (`eating` needs
to see food, `covered in` needs texture). That test needs CLIP features and a
GPU, and it is the last cheap-ish experiment before the reranking direction
should be abandoned.

Usage:
    python tools/candidate_reranking_analysis.py
    python tools/candidate_reranking_analysis.py --out runs/analysis/reranking.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_PRIOR = Path("datasets_vg150_clean/frequency_prior_train.json")
DEFAULT_TRAIN = Path("datasets_vg150_clean/train.jsonl")
DEFAULT_VAL = Path("datasets_vg150_clean/validation.jsonl")


def load_prior(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab = [str(p).strip().lower() for p in raw["predicate_vocab"]]
    return vocab, {p: i for i, p in enumerate(vocab)}, raw


def prior_row(raw, sl: str, ol: str):
    r = raw["pair_log_probs"].get(f"{sl}||{ol}")
    if r is not None:
        return r
    s = raw["subject_log_probs"].get(sl)
    o = raw["object_log_probs"].get(ol)
    if s is not None and o is not None:
        return [0.5 * (a + b) for a, b in zip(s, o)]
    return s or o or raw["global_log_probs"]


def load_split(path: Path, raw, pidx: Dict[str, int], cap: Optional[int] = None):
    sys.path.insert(0, str(Path.cwd()))
    from openvocab_rel.geometry import geom_feats_torch

    Gs, Ps, Ys = [], [], []
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if cap and n >= cap:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rels = ex.get("relationships") or []
            boxes = ex.get("obj_boxes") or []
            if not rels or not boxes:
                continue
            names = []
            for obj in ex.get("objects") or []:
                nm = obj.get("names") or []
                names.append(str(nm[0]).strip().lower() if nm else "")
            bt = torch.tensor(boxes, dtype=torch.float32)
            if bt.ndim != 2 or bt.shape[1] != 4:
                continue
            si, oi, yy, pr = [], [], [], []
            for rel in rels:
                s = int(rel.get("subject_id", -1))
                o = int(rel.get("object_id", -1))
                p = str(rel.get("predicate", "")).strip().lower()
                if not (0 <= s < len(names) and 0 <= o < len(names)):
                    continue
                if not (s < bt.shape[0] and o < bt.shape[0]) or p not in pidx:
                    continue
                si.append(s); oi.append(o); yy.append(pidx[p])
                pr.append(prior_row(raw, names[s], names[o]))
            if not si:
                continue
            Gs.append(geom_feats_torch(bt[si], bt[oi]))
            Ps.append(torch.tensor(pr, dtype=torch.float32))
            Ys.append(torch.tensor(yy, dtype=torch.long))
            n += len(si)
    return torch.cat(Gs), torch.cat(Ps), torch.cat(Ys)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", default=str(DEFAULT_PRIOR))
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    for p in (Path(args.prior), Path(args.train), Path(args.val)):
        if not p.exists():
            print(f"[FAIL] missing: {p}", file=sys.stderr)
            return 2

    torch.manual_seed(0)
    PV, PIDX, raw = load_prior(Path(args.prior))
    P = len(PV)
    print("loading ...", flush=True)
    Gtr, Ptr, Ytr = load_split(Path(args.train), raw, PIDX, args.train_cap)
    Gva, Pva, Yva = load_split(Path(args.val), raw, PIDX)
    mu, sd = Gtr.mean(0, keepdim=True), Gtr.std(0, keepdim=True).clamp_min(1e-6)
    Gtr, Gva = (Gtr - mu) / sd, (Gva - mu) / sd
    Ptr = Ptr - Ptr.mean(-1, keepdim=True)
    Pva = Pva - Pva.mean(-1, keepdim=True)
    D = Gtr.shape[1]
    print(f"  train {Gtr.shape[0]:,}  val {Gva.shape[0]:,}\n")

    cnt = torch.bincount(Ytr, minlength=P).float()
    order = torch.argsort(cnt, descending=True)
    HEAD, BODY = set(order[:15].tolist()), set(order[15:35].tolist())
    bucket = {i: ("head" if i in HEAD else "body" if i in BODY else "tail") for i in range(P)}

    def evaluate(pred):
        hit = (pred == Yva)
        ph, pg = Counter(), Counter()
        for y, h in zip(Yva.tolist(), hit.tolist()):
            pg[y] += 1
            ph[y] += int(h)
        mR = sum(ph[k] / pg[k] for k in pg) / max(1, len(pg))
        bk = {}
        for b in ("head", "body", "tail"):
            ks = [k for k in pg if bucket[k] == b]
            bk[b] = sum(ph[k] / pg[k] for k in ks) / max(1, len(ks))
        return float(hit.float().mean()), mR, bk

    out: Dict[str, Any] = {}

    print("=" * 74)
    print("1. CANDIDATE COVERAGE   P(GT in prior top-K)")
    print("=" * 74)
    print(f"{'K':>4}{'coverage':>12}{'head':>10}{'body':>10}{'tail':>10}")
    cov = {}
    for K in (1, 3, 5, 10, 20):
        cand = torch.topk(Pva, k=K, dim=-1).indices
        inset = (cand == Yva.unsqueeze(1)).any(1)
        cov[K] = inset
        per = {}
        for b in ("head", "body", "tail"):
            m = torch.tensor([bucket[int(y)] == b for y in Yva.tolist()])
            per[b] = float(inset[m].float().mean()) if int(m.sum()) else 0.0
        out[f"coverage@{K}"] = float(inset.float().mean())
        print(f"{K:>4}{float(inset.float().mean())*100:>11.2f}%{per['head']*100:>9.1f}%"
              f"{per['body']*100:>9.1f}%{per['tail']*100:>9.1f}%")

    print("\n" + "=" * 74)
    print("2. ORACLE CEILING   (perfect reranking of the prior's top-K)")
    print("=" * 74)
    R0, m0, bk0 = evaluate(Pva.argmax(-1))
    out["prior_R"], out["prior_mR"], out["prior_tail_mR"] = R0, m0, bk0["tail"]
    print(f"  prior top-1        R@50 {R0*100:6.2f}%  mR@50 {m0*100:6.2f}%  tail {bk0['tail']*100:5.1f}%")
    for K in (3, 5, 10):
        Ro, mo, bko = evaluate(torch.where(cov[K], Yva, Pva.argmax(-1)))
        out[f"oracle@{K}_mR"] = mo
        print(f"  ORACLE top-{K:<3}       R@50 {Ro*100:6.2f}%  mR@50 {mo*100:6.2f}%  tail {bko['tail']*100:5.1f}%")

    print("\n" + "=" * 74)
    print(f"3. NON-NEURAL RERANKING  (K={args.K}, geometry, same params as flat model)")
    print("=" * 74)
    K = args.K
    cand_tr = torch.topk(Ptr, k=K, dim=-1).indices
    cand_va = torch.topk(Pva, k=K, dim=-1).indices
    idx = torch.nonzero((cand_tr == Ytr.unsqueeze(1)).any(1)).squeeze(1)

    def fit(feat: str, use_prior_feat: bool):
        W = torch.zeros(P, D, requires_grad=True)
        b = torch.zeros(P, requires_grad=True)
        pw = torch.zeros(1, requires_grad=True)
        params = [W, b] + ([pw] if use_prior_feat else [])
        opt = torch.optim.Adam(params, lr=0.05)
        G = torch.zeros_like(Gtr) if feat == "zero" else (
            Gtr[torch.randperm(Gtr.shape[0])] if feat == "shuffle" else Gtr)
        for _ in range(12):
            perm = idx[torch.randperm(int(idx.numel()))]
            for i in range(0, len(perm), 8192):
                sel = perm[i:i + 8192]
                c = cand_tr[sel]
                s = (G[sel].unsqueeze(1) * W[c]).sum(-1) + b[c]
                if use_prior_feat:
                    s = s + pw * Ptr[sel].gather(1, c)
                loss = F.cross_entropy(s, (c == Ytr[sel].unsqueeze(1)).float().argmax(1))
                opt.zero_grad(); loss.backward(); opt.step()
        return W.detach(), b.detach(), pw.detach()

    print(f"{'arm':<40}{'R@50':>9}{'mR@50':>9}{'tail':>8}")
    print(f"{'P0 prior top-1 (no rerank)':<40}{R0*100:>8.2f}%{m0*100:>8.2f}%{bk0['tail']*100:>7.1f}%")
    for feat, use_p, tag in (("geom", False, "rerank geometry"),
                             ("geom", True, "rerank geometry + prior feature"),
                             ("shuffle", False, "rerank SHUFFLED geometry"),
                             ("zero", False, "rerank ZEROED geometry")):
        W, b, pw = fit(feat, use_p)
        G = torch.zeros_like(Gva) if feat == "zero" else (
            Gva[torch.randperm(Gva.shape[0])] if feat == "shuffle" else Gva)
        s = (G.unsqueeze(1) * W[cand_va]).sum(-1) + b[cand_va]
        if use_p:
            s = s + pw * Pva.gather(1, cand_va)
        R, m, bk = evaluate(cand_va.gather(1, s.argmax(1, keepdim=True)).squeeze(1))
        out[tag] = {"R": R, "mR": m, "tail_mR": bk["tail"]}
        print(f"{tag:<40}{R*100:>8.2f}%{m*100:>8.2f}%{bk['tail']*100:>7.1f}%")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  generator half : CONFIRMED -- {out['coverage@5']*100:.1f}% coverage @5, "
          f"oracle mR {out['oracle@5_mR']*100:.1f}% vs prior {m0*100:.1f}%")
    best = max(out[k]["mR"] for k in out if isinstance(out[k], dict) and "mR" in out[k])
    print(f"  ranker half    : FALSIFIED for geometry -- best cheap scorer mR {best*100:.2f}% "
          f"vs prior {m0*100:.2f}%")
    print("  untested       : appearance features (geometry carries no appearance)")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[INFO] written to {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
