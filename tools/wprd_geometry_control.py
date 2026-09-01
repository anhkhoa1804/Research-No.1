#!/usr/bin/env python
"""Is the model's within-pair grounding anything more than BOX GEOMETRY?

runs/p33 established that the checkpoint carries real image-conditioned
relational information (WPRD 0.5542 text / 0.5728 classifier against an exactly
0.5000 prior control). "Image-conditioned" is not the same as "understands the
relation": the subject and object BOXES are read off the image too, and for a
large part of VG150's predicate vocabulary -- on, under, behind, in, near --
the answer is close to a function of relative position and size alone.

So this fits a probe on GEOMETRY ONLY, cross-fitted over the same 5 image-level
folds, and scores it with the same WPRD. The geometry probe never sees pixels,
never sees rel_feat and never sees the predicate text embeddings. It sees 19
numbers per pair derived from two rectangles.

Reading:

  geometry WPRD ~ model WPRD  ->  the model's relational grounding is spatial
                                  layout. It has learned WHERE, not WHAT.
  geometry WPRD << model WPRD ->  the model carries appearance/semantic
                                  relational evidence beyond layout.
  model WPRD - geometry WPRD  ->  the part that is not layout.

Both readings are informative and the experiment is cheap, which is why it is
worth running before any successor model is designed.

CPU only. Cache read-only. No GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
CSP = _load("candidate_scorer_probe")
Mech = MECH.Mech
N_FOLDS, SEED = 5, 0


def _log(m: str = "") -> None:
    print(m, flush=True)


def _geom(s: torch.Tensor, o: torch.Tensor, W: float, H: float) -> torch.Tensor:
    """19 scale-invariant numbers from two aligned stacks of rectangles.

    Shared verbatim by the validation path and the train-fitted path, so the
    two cannot drift into measuring different things.
    """
    if True:
        sw, sh = (s[:, 2] - s[:, 0]).clamp_min(1e-3), (s[:, 3] - s[:, 1]).clamp_min(1e-3)
        ow, oh = (o[:, 2] - o[:, 0]).clamp_min(1e-3), (o[:, 3] - o[:, 1]).clamp_min(1e-3)
        scx, scy = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2
        ocx, ocy = (o[:, 0] + o[:, 2]) / 2, (o[:, 1] + o[:, 3]) / 2
        sa, oa = sw * sh, ow * oh
        ix = (torch.min(s[:, 2], o[:, 2]) - torch.max(s[:, 0], o[:, 0])).clamp_min(0)
        iy = (torch.min(s[:, 3], o[:, 3]) - torch.max(s[:, 1], o[:, 1])).clamp_min(0)
        inter = ix * iy
        union = (sa + oa - inter).clamp_min(1e-3)
        f = torch.stack([
            scx / W, scy / H, ocx / W, ocy / H,                    # absolute positions
            (ocx - scx) / (sw + ow), (ocy - scy) / (sh + oh),      # relative offset
            (ocx - scx) / W, (ocy - scy) / H,                      # offset, image scale
            sw / W, sh / H, ow / W, oh / H,                        # sizes
            torch.log(sa / oa),                                    # log area ratio
            torch.log(sw / sh), torch.log(ow / oh),                # aspect ratios
            inter / union,                                         # IoU
            inter / sa.clamp_min(1e-3),                            # subject containment
            inter / oa.clamp_min(1e-3),                            # object containment
            torch.sqrt(((ocx - scx) / W) ** 2 + ((ocy - scy) / H) ** 2),  # distance
        ], dim=-1)
        return f


def _standardise(X: torch.Tensor, stats=None):
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if stats is None:
        stats = (X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6))
    X = (X - stats[0]) / stats[1]
    return torch.cat([X, torch.ones(X.shape[0], 1)], dim=-1), stats


def geometry_features_raw(B: Mech) -> torch.Tensor:
    """Un-standardised geometry rows for the validation GT rows."""
    feats: List[torch.Tensor] = []
    for i in range(B.n_images):
        boxes = B.meta["obj_boxes"][i].float()
        pairs = B.meta["pairs"][i]
        if boxes.numel() == 0 or len(pairs) == 0:
            continue
        W = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
        H = float(boxes[:, 3].max() - boxes[:, 1].min()) or 1.0
        feats.append(_geom(boxes[pairs[:, 0].long()], boxes[pairs[:, 1].long()], W, H))
    return torch.cat(feats, 0)[B.gt_row]


def train_split_geometry(path: str, classes: List[str]):
    """Geometry rows + predicate labels from the TRAIN split.

    Fitting on train and evaluating on validation removes the only advantage
    the cross-fitted validation probe had over the model: the model was trained
    on train and evaluated on val, and now so is the probe.
    """
    idx = {c: i for i, c in enumerate(classes)}
    XS: List[torch.Tensor] = []
    YS: List[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            boxes = torch.tensor(d.get("obj_boxes") or [], dtype=torch.float32)
            rels = d.get("relationships") or []
            if boxes.numel() == 0 or not rels:
                continue
            W = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
            H = float(boxes[:, 3].max() - boxes[:, 1].min()) or 1.0
            si, oi, yy = [], [], []
            nb = boxes.shape[0]
            for r in rels:
                c = idx.get(str(r.get("predicate", "")).strip().lower())
                a, b = int(r.get("subject_id", -1)), int(r.get("object_id", -1))
                if c is None or not (0 <= a < nb and 0 <= b < nb):
                    continue
                si.append(a); oi.append(b); yy.append(c)
            if not yy:
                continue
            XS.append(_geom(boxes[si], boxes[oi], W, H))
            YS.extend(yy)
    return torch.cat(XS, 0), torch.tensor(YS)


def fold_of_image(B: Mech, salt: int = 0) -> torch.Tensor:
    """The SAME image-level fold rule every other analysis in this repo uses.

    Delegates to tools/candidate_scorer_probe.fold_of_image rather than
    reimplementing it, so this probe cannot silently drift onto a different
    partition from p22/p25/p26/p29/p32.
    """
    per_image = [CSP.fold_of_image(str(s), N_FOLDS, salt)
                 for s in B.meta["image_id"]]
    # B.gt_row indexes the FLATTENED pair list, not the image list, so the fold
    # of each image has to be broadcast over that image's pairs first.
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per_image[i]] * len(B.meta["pairs"][i]))
    return torch.tensor(flat)[B.gt_row]


def cross_fit_logits(X: torch.Tensor, y: torch.Tensor, fold: torch.Tensor,
                     C: int, epochs: int, l2: float) -> torch.Tensor:
    """Out-of-fold class logits. Held-out rows never influence their own score."""
    out = torch.zeros(X.shape[0], C)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        W = torch.zeros(X.shape[1], C, requires_grad=True)
        opt = torch.optim.LBFGS([W], max_iter=epochs, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (torch.nn.functional.cross_entropy(X[tr] @ W, y[tr])
                    + l2 * (W * W).sum())
            loss.backward()
            return loss

        opt.step(closure)
        out[te] = (X[te] @ W).detach()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p24_full_val_cache/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default="runs/p38_wprd_geometry/geom.json")
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--train-jsonl", default="",
                    help="fit the geometry probe on the TRAIN split instead of "
                         "cross-fitting on validation. Removes the probe's only "
                         "advantage over the model.")
    args = ap.parse_args(argv)

    _log("=" * 100)
    _log("WPRD GEOMETRY CONTROL -- is the grounding just box layout? CPU only, NO GPU")
    _log("=" * 100)
    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    dev = Gs.prior_is_constant(B.prior)
    _log(f"  images={B.n_images} rows={Gs.n:,} groups={Gs.G:,}  "
         f"[gate W1] prior within-group max dev {dev:.3e} "
         f"{'PASS' if dev < 1e-3 else 'FAIL'}")
    assert dev < 1e-3

    Xr = geometry_features_raw(B)
    X, stats = _standardise(Xr)
    fold = fold_of_image(B, 0)
    _log(f"  geometry features: {X.shape[1] - 1} + bias, rows {X.shape[0]:,}; "
         f"cross-fitted over {N_FOLDS} image-level folds (seed {SEED})")
    _log(f"  fold sizes: {[int((fold == f).sum()) for f in range(N_FOLDS)]}")

    y = B.gt_y
    geo = cross_fit_logits(X, y, fold, B.n_classes, args.epochs, args.l2)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    geo_sh = cross_fit_logits(X, ysh, fold, B.n_classes, args.epochs, args.l2)

    arms = {
        "text_head (evaluated)": B.fixed_ensemble(0.0),
        "classifier_head (discarded)": B.fixed_ensemble(1.0),
        "GEOMETRY probe (cross-fitted, no pixels)": geo,
        "geometry, SHUFFLED labels (must be 0.5)": geo_sh,
        "prior (must be 0.5)": B.prior,
    }

    if args.train_jsonl:
        _log(f"\n  fitting geometry on the TRAIN split: {args.train_jsonl}")
        Xt_raw, yt = train_split_geometry(args.train_jsonl, list(B.classes))
        # standardise train by ITS OWN statistics, then apply the same map to
        # validation -- no validation statistic touches the fit.
        Xt, tstats = _standardise(Xt_raw)
        Xv, _ = _standardise(Xr, tstats)
        _log(f"  train rows {Xt.shape[0]:,} over {int(yt.max()) + 1} classes seen; "
             f"val rows {Xv.shape[0]:,}")
        W = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
        opt = torch.optim.LBFGS([W], max_iter=args.epochs, history_size=10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (torch.nn.functional.cross_entropy(Xt @ W, yt)
                    + args.l2 * (W * W).sum())
            loss.backward()
            return loss

        opt.step(closure)
        arms["GEOMETRY probe TRAIN-FITTED (no val fit at all)"] = (Xv @ W).detach()
        train_fit_rows = int(Xt.shape[0])
    res: Dict[str, Any] = {"tool": "wprd_geometry_control", "dump": args.dump,
                           "n_geom_features": X.shape[1] - 1, "arms": {}}
    if args.train_jsonl:
        res["train_jsonl"] = args.train_jsonl
        res["train_fit_rows"] = train_fit_rows
    _log(f"\n{'-'*100}")
    _log(f"  {'arm':>44} {'macro':>8} {'weighted':>9} {'95% CI':>22}")
    _log(f"{'-'*100}")
    store = {}
    for name, s in arms.items():
        cc = WPD.wprd(Gs, s, args.cap)
        v = torch.tensor(cc["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        bs = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        store[name] = v
        res["arms"][name] = {"macro": cc["wprd_macro"], "weighted": cc["wprd_weighted"],
                             "ci95": [lo, hi], "n_cells": cc["n_cells"]}
        _log(f"  {name:>44} {cc['wprd_macro']:>8.4f} {cc['wprd_weighted']:>9.4f}  "
             f"[{lo:.4f}, {hi:.4f}]")

    _log(f"\n  PAIRED contrasts (same cells, bootstrap over cells)")
    contrast_pairs = [("text_head (evaluated)", "GEOMETRY probe (cross-fitted, no pixels)"),
                      ("classifier_head (discarded)", "GEOMETRY probe (cross-fitted, no pixels)"),
                      ("classifier_head (discarded)", "text_head (evaluated)")]
    TF = "GEOMETRY probe TRAIN-FITTED (no val fit at all)"
    if TF in store:
        contrast_pairs += [("text_head (evaluated)", TF),
                           ("classifier_head (discarded)", TF)]
    for a, b in contrast_pairs:
        d = store[a] - store[b]
        g = torch.Generator().manual_seed(2)
        bs = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean()
                          for _ in range(args.boot)])
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        sig = "excludes 0" if (lo > 0 or hi < 0) else "INCLUDES 0"
        res.setdefault("contrasts", {})[f"{a} - {b}"] = {
            "mean": float(d.mean()), "ci95": [lo, hi]}
        _log(f"    {a[:34]:>34} - {b[:34]:<34} {float(d.mean()):>+8.4f} "
             f"[{lo:+.4f}, {hi:+.4f}]  {sig}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
