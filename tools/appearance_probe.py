#!/usr/bin/env python
"""Can frozen CLIP appearance rank predicates inside the prior's top-K?

The terminal experiment of the candidate-reranking investigation. Prior phases
established that the pair-conditioned prior is an excellent CANDIDATE
GENERATOR (89.4 % coverage @5, oracle mR@50 63.80 vs prior 22.30) but that no
geometry-based scorer captures any of that headroom. Box geometry carries no
appearance information at all, so appearance was the last untested variable.

RESULT: appearance signal is REAL but far too WEAK.

  additive composition   score = log P(p|s,o) + lambda * f(appearance)
    lambda = 0.25 (best)   mR@50 21.34 vs prior 20.98   -> 0.8 % of headroom

  visual ablation gate    PASSES
    real 16.75  >  shuffled 14.54  >  zero 8.83   (mR@50, replacement arm)

  per-predicate, best additive arm
    eating       40.9 -> 54.5  (+13.6, n=22)
    walking on    0.0 -> 13.0  (+13.0, n=23)
    riding       10.2 -> 16.3  ( +6.1, n=49)
    standing on   1.2 ->  4.9  ( +3.7, n=81)
    holding      64.9 -> 67.2  ( +2.3, n=308)

Appearance helps exactly the action/pose predicates theory predicts it should,
and hurts nothing systematically -- but it converts <1 % of the available
headroom. That is not enough to justify a neural reranker.

CAVEATS, none of which are hidden:
  * ViT-B/32, not the repo's ViT-L/14-336. Forced: L/14-336 runs at 0.3
    crops/s on a CPU-only machine (3.1 s/crop), infeasible for ~70k crops.
    B/32 is weaker, so this is a LOWER BOUND and a negative result is
    correspondingly weaker evidence than a positive one would have been.
  * 1,200 train + 1,200 validation images, not the full 10,401 split. Tail
    predicates are thin (eating n=22), so per-predicate deltas are noisy.
  * PCA-48 per feature block may discard signal.
  * Held-out mR (~49 %) far exceeds val mR (~21 %). Part of that gap is that
    the train-derived prior was built from the full train split and has
    therefore seen the held-out train images -- so held-out-from-train
    numbers are optimistically biased. The validation numbers are the honest
    ones.

Usage:
    python tools/appearance_probe.py --extract        # ~60 min CPU, writes cache
    python tools/appearance_probe.py --score          # scores an existing cache
    python tools/appearance_probe.py --extract --score --out runs/analysis/appearance.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_CACHE = Path("runs/analysis/clip_appearance_cache.pt")
DEFAULT_PRIOR = Path("datasets_vg150_clean/frequency_prior_train.json")
DEFAULT_TRAIN = Path("datasets_vg150_clean/train.jsonl")
DEFAULT_VAL = Path("datasets_vg150_clean/validation.jsonl")
# Default kept at B/32 so an existing CPU-only invocation reproduces the
# original measurement byte-for-byte. --clip_name selects the encoder; the
# repo's own model is openai/clip-vit-large-patch14-336, which was infeasible
# on the CPU-only machine this probe was first written on (0.3 crops/s) and is
# the encoder the negative result must be re-tested against on a GPU.
CLIP_ID = "openai/clip-vit-base-patch32"
K = 5
PCA_DIM = 48
EPOCHS = 30
ENCODE_BATCH = 64

# Decision thresholds, pre-registered in
# docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md before the L/14-336 run.
# The 5% figure is inherited verbatim from the earlier B/32 analysis's own
# stated criterion, not chosen after seeing any result. Between the two bounds
# the outcome is reported as inconclusive rather than rounded to a verdict.
THRESHOLD_LOW_PCT = 4.0
THRESHOLD_HIGH_PCT = 6.0


# ===========================================================================
# extraction
# ===========================================================================

def _image_features(model, px):
    """Version-stable CLIP image features.

    transformers 5.x returns BaseModelOutputWithPooling from
    get_image_features rather than a tensor, so go through the vision tower
    and the projection explicitly.
    """
    out = model.vision_model(pixel_values=px)
    pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
    return model.visual_projection(pooled)


@torch.no_grad()
def extract(prior_path: Path, train_path: Path, val_path: Path, cache: Path,
            n_train: int, n_val: int, images_root: Path,
            clip_name: str = CLIP_ID, device: str = "cpu",
            encode_batch: int = ENCODE_BATCH, amp: bool = True) -> None:
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel

    sys.path.insert(0, str(Path.cwd()))
    from openvocab_rel.geometry import geom_feats_torch

    # Grad is disabled by the @torch.no_grad() decorator above, which scopes it
    # to this function. The previous `torch.set_grad_enabled(False)` here was a
    # PROCESS-WIDE switch that was never restored, so the documented combined
    # invocation (`--extract --score`) always died in score()'s first
    # loss.backward() with "element 0 of tensors does not require grad".
    dev = torch.device(device)
    # fp16 only on CUDA: the encoder is frozen and its output is L2-normalized
    # immediately, so reduced precision cannot leak into the fitted scorer as
    # anything but a tiny feature perturbation. On CPU fp16 is both slower and
    # less accurate, so it is never used there.
    use_amp = bool(amp) and dev.type == "cuda"
    print(f"loading {clip_name} (FROZEN -- never fine-tuned) on {dev}"
          f"{' [fp16 autocast]' if use_amp else ''} ...", flush=True)
    model = CLIPModel.from_pretrained(clip_name).eval().to(dev)
    proc = CLIPImageProcessor.from_pretrained(clip_name)
    dim = model.config.projection_dim

    raw = json.loads(prior_path.read_text(encoding="utf-8"))
    PV = [str(p).strip().lower() for p in raw["predicate_vocab"]]
    PIDX = {p: i for i, p in enumerate(PV)}

    def prior_row(sl, ol):
        r = raw["pair_log_probs"].get(f"{sl}||{ol}")
        if r is not None:
            return r
        s, o = raw["subject_log_probs"].get(sl), raw["object_log_probs"].get(ol)
        if s is not None and o is not None:
            return [0.5 * (a + b) for a, b in zip(s, o)]
        return s or o or raw["global_log_probs"]

    index = {}
    for sub in ("VG_100K", "VG_100K_2"):
        d = images_root / sub
        if d.is_dir():
            for f in d.iterdir():
                index[f.stem] = f

    def crop(img, box, pad=0.0):
        w, h = img.size
        x1, y1, x2, y2 = box
        if pad:
            bw, bh = x2 - x1, y2 - y1
            x1 -= bw * pad; x2 += bw * pad; y1 -= bh * pad; y2 += bh * pad
        x1 = max(0, min(w - 1, int(x1))); y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2))); y2 = max(y1 + 1, min(h, int(y2)))
        return img.crop((x1, y1, x2, y2))

    def encode(pils):
        out = []
        for i in range(0, len(pils), encode_batch):
            px = proc(images=pils[i:i + encode_batch], return_tensors="pt")["pixel_values"].to(dev)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f = _image_features(model, px)
            else:
                f = _image_features(model, px)
            # Normalize and store in fp32 on CPU: the cache is consumed by a
            # CPU scorer, and keeping features on the GPU would pin memory for
            # the whole extraction for no benefit.
            out.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, dim)

    def run(path, cap, tag):
        rows = {k: [] for k in ("glob", "subj", "obj", "union", "geom", "prior", "y", "img")}
        n_img = n_crops = 0
        t0 = time.perf_counter()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if n_img >= cap:
                    break
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                rels, boxes = ex.get("relationships") or [], ex.get("obj_boxes") or []
                if not rels or not boxes:
                    continue
                fp = index.get(str(ex.get("image_id", "")))
                if fp is None:
                    continue
                try:
                    img = Image.open(fp).convert("RGB")
                except Exception:
                    continue
                labels = []
                for o in ex.get("objects") or []:
                    nm = o.get("names") or []
                    labels.append(str(nm[0]).strip().lower() if nm else "")
                bt = torch.tensor(boxes, dtype=torch.float32)
                if bt.ndim != 2 or bt.shape[1] != 4:
                    continue
                valid = [(s, o, p) for s, o, p in
                         ((int(r.get("subject_id", -1)), int(r.get("object_id", -1)),
                           str(r.get("predicate", "")).strip().lower()) for r in rels)
                         if 0 <= s < min(len(labels), bt.shape[0])
                         and 0 <= o < min(len(labels), bt.shape[0]) and p in PIDX]
                if not valid:
                    continue
                oids = sorted({i for s, o, _ in valid for i in (s, o)})
                opos = {i: k for k, i in enumerate(oids)}
                prs = sorted({(s, o) for s, o, _ in valid})
                ppos = {p: k for k, p in enumerate(prs)}
                crops = [img] + [crop(img, bt[i].tolist(), 0.1) for i in oids]
                for (s, o) in prs:
                    b1, b2 = bt[s].tolist(), bt[o].tolist()
                    crops.append(crop(img, [min(b1[0], b2[0]), min(b1[1], b2[1]),
                                            max(b1[2], b2[2]), max(b1[3], b2[3])], 0.05))
                f = encode(crops)
                n_crops += len(crops)
                gf, of, uf = f[0], f[1:1 + len(oids)], f[1 + len(oids):]
                for (s, o, p) in valid:
                    rows["glob"].append(gf); rows["subj"].append(of[opos[s]])
                    rows["obj"].append(of[opos[o]]); rows["union"].append(uf[ppos[(s, o)]])
                    rows["geom"].append(geom_feats_torch(bt[s:s + 1], bt[o:o + 1])[0])
                    rows["prior"].append(torch.tensor(prior_row(labels[s], labels[o]), dtype=torch.float32))
                    rows["y"].append(PIDX[p]); rows["img"].append(n_img)
                n_img += 1
                if n_img % 100 == 0:
                    el = time.perf_counter() - t0
                    print(f"  [{tag}] {n_img}/{cap} img  {n_crops:,} crops  "
                          f"{len(rows['y']):,} inst  {el:.0f}s  eta {el/n_img*(cap-n_img):.0f}s", flush=True)
        out = {k: (torch.stack(v) if k not in ("y", "img") else torch.tensor(v)) for k, v in rows.items()}
        out["n_images"], out["n_crops"] = n_img, n_crops
        return out

    tr = run(train_path, n_train, "train")
    va = run(val_path, n_val, "val")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    res = proc.crop_size if isinstance(getattr(proc, "crop_size", None), dict) else None
    meta = {
        "clip_model": clip_name, "clip_frozen": True, "projection_dim": dim,
        "preprocess_resolution": int(res["height"]) if res else None,
        "l2_normalised": True,
        "device": str(dev), "fp16_autocast": bool(use_amp), "encode_batch": int(encode_batch),
        "subject_object_crop_pad": 0.1, "union_crop_pad": 0.05,
        "git_commit": commit,
        "n_train_images": int(tr["n_images"]), "n_train_instances": int(tr["y"].numel()),
        "n_val_images": int(va["n_images"]), "n_val_instances": int(va["y"].numel()),
        "n_crops_encoded": int(tr["n_crops"] + va["n_crops"]),
        "note": (
            "Encoder recorded above. A weaker encoder than the repo's own "
            "openai/clip-vit-large-patch14-336 makes a NEGATIVE result a LOWER "
            "BOUND only."
        ),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": tr, "val": va, "meta": meta, "PV": PV}, cache)
    print(json.dumps(meta, indent=2))
    print(f"[INFO] cache -> {cache}")


# ===========================================================================
# scoring
# ===========================================================================

def score(cache: Path, out_path: str) -> Dict[str, Any]:
    torch.manual_seed(0)
    blob = torch.load(cache, weights_only=False)
    TR, VA, META, PV = blob["train"], blob["val"], blob["meta"], blob["PV"]
    P = len(PV)
    print(json.dumps({k: META.get(k) for k in
                      ("clip_model", "n_train_instances", "n_val_instances",
                       "n_crops_encoded", "git_commit")}, indent=2))

    Ptr = TR["prior"] - TR["prior"].mean(-1, keepdim=True)
    Pva = VA["prior"] - VA["prior"].mean(-1, keepdim=True)
    Ytr, Yva = TR["y"], VA["y"]
    gmu = TR["geom"].mean(0, keepdim=True)
    gsd = TR["geom"].std(0, keepdim=True).clamp_min(1e-6)
    Gtr, Gva = (TR["geom"] - gmu) / gsd, (VA["geom"] - gmu) / gsd

    uimg = torch.unique(TR["img"])
    perm = uimg[torch.randperm(len(uimg))]
    hold = set(perm[: max(1, int(0.2 * len(perm)))].tolist())
    is_hold = torch.tensor([int(i) in hold for i in TR["img"].tolist()])

    cnt = torch.bincount(Ytr, minlength=P).float()
    order = torch.argsort(cnt, descending=True)
    HEAD, BODY = set(order[:15].tolist()), set(order[15:35].tolist())
    bucket = {i: ("head" if i in HEAD else "body" if i in BODY else "tail") for i in range(P)}

    _pc: Dict[str, Any] = {}

    def blk(split, nm):
        if nm not in _pc:
            X = TR[nm][~is_hold]
            mu = X.mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(X - mu, q=min(PCA_DIM, X.shape[1]), center=False)
            _pc[nm] = (mu, V)
        mu, V = _pc[nm]
        Z = ((TR[nm] if split == "tr" else VA[nm]) - mu) @ V
        return Z / Z.std(0, keepdim=True).clamp_min(1e-6)

    def feats(split, kind, shuffle=False, zero=False):
        parts = [blk(split, n) for n in ("glob", "subj", "obj", "union") if n in kind]
        X = torch.cat(parts, -1) if parts else torch.zeros((Ytr if split == "tr" else Yva).numel(), 0)
        if zero:
            X = torch.zeros_like(X)
        elif shuffle and X.numel():
            X = X[torch.randperm(X.shape[0])]
        if "geom" in kind:
            X = torch.cat([X, Gtr if split == "tr" else Gva], -1)
        return X

    cand_tr = torch.topk(Ptr, k=K, dim=-1).indices
    cand_va = torch.topk(Pva, k=K, dim=-1).indices
    in_tr = (cand_tr == Ytr.unsqueeze(1)).any(1)
    in_va = (cand_va == Yva.unsqueeze(1)).any(1)

    def mr_of(pred, Y):
        hit = (pred == Y)
        ph, pg = Counter(), Counter()
        for y, h in zip(Y.tolist(), hit.tolist()):
            pg[y] += 1
            ph[y] += int(h)
        mR = sum(ph[k] / pg[k] for k in pg) / max(1, len(pg))
        bk = {}
        for b in ("head", "body", "tail"):
            ks = [k for k in pg if bucket[k] == b]
            bk[b] = sum(ph[k] / pg[k] for k in ks) / max(1, len(ks)) if ks else 0.0
        return float(hit.float().mean()), mR, bk, ph, pg

    def fit(kind, lam=None, shuffle=False, zero=False, wd=3e-3, lr=0.02):
        """lam=None -> replacement scorer; lam set -> additive on top of the prior."""
        X = feats("tr", kind, shuffle, zero)
        D = X.shape[1]
        W = torch.zeros(P, D, requires_grad=True)
        b = torch.zeros(P, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
        fi = torch.nonzero((~is_hold) & in_tr).squeeze(1)
        hm_mask = is_hold & in_tr
        best = (-1.0, None, None)
        for _ in range(EPOCHS):
            pm = fi[torch.randperm(int(fi.numel()))]
            for i in range(0, len(pm), 4096):
                sel = pm[i:i + 4096]
                c = cand_tr[sel]
                s = (X[sel].unsqueeze(1) * W[c]).sum(-1) + b[c]
                if lam is not None:
                    s = Ptr[sel].gather(1, c) + lam * s
                tgt = (c == Ytr[sel].unsqueeze(1)).float().argmax(1)
                loss = F.cross_entropy(s, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                c = cand_tr[hm_mask]
                sh = (X[hm_mask].unsqueeze(1) * W[c]).sum(-1) + b[c]
                if lam is not None:
                    sh = Ptr[hm_mask].gather(1, c) + lam * sh
                _r, hm, _b, _p, _g = mr_of(c.gather(1, sh.argmax(1, keepdim=True)).squeeze(1), Ytr[hm_mask])
                if hm > best[0]:
                    best = (hm, W.detach().clone(), b.detach().clone())
        return best

    def apply(W, b, kind, lam=None, shuffle=False, zero=False):
        X = feats("va", kind, shuffle, zero)
        s = (X.unsqueeze(1) * W[cand_va]).sum(-1) + b[cand_va]
        if lam is not None:
            s = Pva.gather(1, cand_va) + lam * s
        return s

    R0, m0, bk0, ph0, pg0 = mr_of(Pva.argmax(-1), Yva)
    Ro, mo, bko, _, _ = mr_of(torch.where(in_va, Yva, Pva.argmax(-1)), Yva)
    res: Dict[str, Any] = {"prior_R": R0, "prior_mR": m0, "oracle_R": Ro, "oracle_mR": mo,
                           "coverage@5": float(in_va.float().mean())}

    print("\n" + "=" * 86)
    print("BASELINES on this validation subsample (NOT the full split)")
    print("=" * 86)
    print(f"  coverage @5   : {res['coverage@5']*100:.2f}%")
    print(f"  P0 prior      : R {R0*100:6.2f}%  mR {m0*100:6.2f}%  tail {bk0['tail']*100:5.1f}%")
    print(f"  ORACLE @5     : R {Ro*100:6.2f}%  mR {mo*100:6.2f}%  tail {bko['tail']*100:5.1f}%")
    print(f"  headroom      : dmR {(mo-m0)*100:+.2f}")

    def show(tag, s, store_key=None):
        pred = cand_va.gather(1, s.argmax(1, keepdim=True)).squeeze(1)
        R, m, bk, ph, pg = mr_of(pred, Yva)
        cap = (m - m0) / max(1e-9, (mo - m0)) * 100
        print(f"{tag:<36}{R*100:>8.2f}%{m*100:>8.2f}%{bk['tail']*100:>7.1f}%{cap:>10.1f}%")
        if store_key:
            res[store_key] = {"R": R, "mR": m, "tail_mR": bk["tail"], "headroom_pct": cap}
        return m, ph, pg

    print("\n" + "=" * 86)
    print("VISUAL ABLATION GATE   required: real > shuffled ~= zero")
    print("=" * 86)
    print(f"{'arm':<36}{'val R':>8}{'val mR':>8}{'tail':>7}{'headroom':>10}")
    abl = {}
    for tag, sh, ze, key in (("REAL appearance", False, False, "abl_real"),
                             ("SHUFFLED appearance", True, False, "abl_shuffled"),
                             ("ZERO appearance", False, True, "abl_zero")):
        hm, W, b = fit("subj+obj+union", shuffle=sh, zero=ze)
        abl[tag] = show(tag, apply(W, b, "subj+obj+union", shuffle=sh, zero=ze), key)[0]
    gate = abl["REAL appearance"] > abl["SHUFFLED appearance"] > abl["ZERO appearance"]
    res["visual_gate_passes"] = bool(gate)
    print(f"\n  real > shuffled > zero ?  {'YES' if gate else 'NO'}")

    print("\n" + "=" * 86)
    print("ADDITIVE COMPOSITION   score = log P(p|s,o) + lambda * appearance")
    print("lambda = 0 is exactly P0, so any gain is attributable to appearance alone")
    print("=" * 86)
    print(f"{'lambda':<36}{'val R':>8}{'val mR':>8}{'tail':>7}{'headroom':>10}")
    print(f"{'0.00  (P0)':<36}{R0*100:>8.2f}%{m0*100:>8.2f}%{bk0['tail']*100:>7.1f}%{0.0:>10.1f}%")

    # lambda is selected on the HELD-OUT-FROM-TRAIN split (`hm`, returned by
    # fit), never on validation. Selecting it by validation score would make
    # the headline the maximum of six correlated draws on the same split it is
    # reported on -- an optimistic bias, not a measurement. The
    # validation-argmax is still computed and reported below, explicitly
    # labelled as the cherry-picked upper bound, so the size of that bias is
    # visible rather than hidden.
    selected = (-1.0, m0, None, None, None)   # (heldout_mR, val_mR, lam, ph, pg)
    val_argmax = (m0, None)                   # (val_mR, lam) -- optimistic, not the headline
    for lam in (0.1, 0.25, 0.5, 1.0, 2.0):
        hm, W, b = fit("subj+obj+union+geom", lam=lam)
        m, ph, pg = show(f"{lam:<5} subj+obj+union+geom", apply(W, b, "subj+obj+union+geom", lam=lam),
                         f"additive_lam{lam}")
        res[f"additive_lam{lam}"]["heldout_mR"] = float(hm)
        if hm > selected[0]:
            selected = (hm, m, lam, ph, pg)
        if m > val_argmax[0]:
            val_argmax = (m, lam)

    best_mR, best_lam, ph_best, pg_best = selected[1], selected[2], selected[3], selected[4]
    res["lambda_selection"] = "held-out-from-train (validation never used to select)"
    res["best_additive_mR"] = best_mR
    res["best_additive_lambda"] = best_lam
    res["best_additive_headroom_pct"] = (best_mR - m0) / max(1e-9, (mo - m0)) * 100
    res["val_argmax_mR"] = val_argmax[0]
    res["val_argmax_lambda"] = val_argmax[1]
    res["val_argmax_headroom_pct"] = (val_argmax[0] - m0) / max(1e-9, (mo - m0)) * 100
    print("\n" + "=" * 86)
    print(f"SELECTED (lambda chosen on held-out train): lambda={best_lam}  "
          f"mR {best_mR*100:.2f}%  vs P0 {m0*100:.2f}%  delta {(best_mR-m0)*100:+.2f}   "
          f"headroom captured {res['best_additive_headroom_pct']:.1f}%")
    print(f"  [reference only, NOT the headline] validation-argmax lambda={val_argmax[1]}  "
          f"mR {val_argmax[0]*100:.2f}%  headroom {res['val_argmax_headroom_pct']:.1f}%  "
          f"<- optimistic: selected on the split it is reported on")
    print("=" * 86)

    if ph_best is not None:
        ph, pg = ph_best, pg_best
        rows = [(PV[k], pg0[k], ph0[k] / pg0[k], (ph[k] / pg[k]) if pg.get(k) else 0.0)
                for k in pg0 if pg0[k] >= 15]
        rows = [(n, c, a, b, b - a) for n, c, a, b in rows]
        rows.sort(key=lambda r: -r[4])
        res["per_predicate_top_gains"] = [
            {"predicate": n, "n": c, "prior_R1": a, "additive_R1": b, "delta": d}
            for n, c, a, b, d in rows[:10]]
        print("\nper-predicate at best lambda (n>=15), largest gains:")
        print(f"{'predicate':<16}{'n':>7}{'P0 R1':>9}{'additive':>10}{'delta':>8}")
        for n, c, a, b, dd in rows[:10]:
            print(f"{n:<16}{c:>7,}{a*100:>8.1f}%{b*100:>9.1f}%{dd*100:>+8.1f}")

    # The verdict is DERIVED from the measured numbers against the threshold
    # pre-registered in docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md. It was
    # previously a hardcoded sentence that printed "signal is REAL but far too
    # WEAK" regardless of what was measured -- including on runs where the
    # ablation gate FAILED and the measurement was therefore invalid.
    captured = res["best_additive_headroom_pct"]
    if not gate:
        verdict = "INVALID"
        explanation = ("the visual ablation gate failed (real > shuffled > zero does not hold), "
                       "so this run measures nothing about appearance either way")
    elif captured >= THRESHOLD_HIGH_PCT:
        verdict = "H1 SUPPORTED"
        explanation = (f"captured {captured:.1f}% >= {THRESHOLD_HIGH_PCT:.0f}%: appearance converts "
                       "a material share of the headroom")
    elif captured < THRESHOLD_LOW_PCT:
        verdict = "H0 SUPPORTED"
        explanation = (f"captured {captured:.1f}% < {THRESHOLD_LOW_PCT:.0f}%: appearance does not "
                       "convert the headroom at this encoder")
    else:
        verdict = "INCONCLUSIVE"
        explanation = (f"captured {captured:.1f}% sits in the pre-registered "
                       f"{THRESHOLD_LOW_PCT:.0f}-{THRESHOLD_HIGH_PCT:.0f}% inconclusive band")
    res["verdict"] = verdict
    res["verdict_explanation"] = explanation
    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    print(f"  visual ablation gate      : {'PASS' if gate else 'FAIL'}")
    print(f"  oracle headroom captured  : {captured:.1f}%   (lambda selected on held-out train)")
    print(f"  -> {verdict}: {explanation}")

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[INFO] written to {p}")
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--prior", default=str(DEFAULT_PRIOR))
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL))
    ap.add_argument("--images", default="datasets_vg150_clean/images")
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_val", type=int, default=1200)
    ap.add_argument("--out", default="")
    ap.add_argument("--clip_name", default=CLIP_ID,
                    help="CLIP encoder to extract with. The repo's own model is "
                         "openai/clip-vit-large-patch14-336; the B/32 default preserves "
                         "the original CPU-only measurement.")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--encode_batch", type=int, default=ENCODE_BATCH)
    ap.add_argument("--no_amp", action="store_true",
                    help="disable fp16 autocast on CUDA (fp32 control arm)")
    args = ap.parse_args(argv)

    if not args.extract and not args.score:
        ap.error("pass --extract and/or --score")
    cache = Path(args.cache)
    if args.extract:
        for p in (Path(args.prior), Path(args.train), Path(args.val)):
            if not p.exists():
                print(f"[FAIL] missing: {p}", file=sys.stderr)
                return 2
        extract(Path(args.prior), Path(args.train), Path(args.val), cache,
                args.n_train, args.n_val, Path(args.images),
                clip_name=str(args.clip_name), device=str(args.device),
                encode_batch=int(args.encode_batch), amp=not bool(args.no_amp))
    if args.score:
        if not cache.exists():
            print(f"[FAIL] cache not found: {cache}. Run with --extract first.", file=sys.stderr)
            return 2
        score(cache, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
