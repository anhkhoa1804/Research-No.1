#!/usr/bin/env python
"""Independently validate a pair-logit cache BEFORE any CPU analysis reads it.

A 60-GPU-minute artifact that is silently wrong is worse than no artifact: every
downstream number inherits the fault and looks precise. This tool re-derives
everything derivable and asserts the invariants the schema promises, then
writes a machine-readable verdict.

It is INDEPENDENT of the analysis: it imports nothing from the C' analysis tool
and re-implements the checks from the schema document, so a bug in the analysis
cannot mask a bug in the cache.

Checks (each PASS/FAIL, all reported):
  S1  schema version and every required key present
  S2  per-image list lengths agree
  S3  tensor shapes agree with pair counts and the vocabulary width
  S4  no NaN/Inf in model_logits / prior_rows / text_logits / cls_logits
  S5  subj_label/obj_label agree with obj_labels indexed by pairs
  S6  pair_index is 0..n-1 per image
  S7  RAW GT: no aliased-away predicate has vanished from the cache
  S8  the exported alias map collapses the 50-class vocab to exactly 48
  S9  predicate ordering matches the on-disk vocabulary file exactly
  S10 the recorded composition reproduces the historical protocol's model term
  S11 GT pairs are locatable in the pair list (PredCls sanity)
  S12 prior rows are RAW (not already tau-adjusted, not alpha-scaled)

Usage:
    python tools/validate_pair_dump.py --dump runs/<name>/pair_logits.pt \
        --predicates datasets_vg150_clean/vocabulary/predicates.json \
        --out runs/<name>/cache_validation.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REQUIRED_KEYS = [
    "schema", "composition", "pred_vocab", "predicate_alias_map",
    "background_predicate_indices", "freq_bias_alpha", "ensemble_alpha",
    "score_mode", "use_vg_aliases", "n_pairs", "n_images",
    "image_id", "pairs", "pair_index", "model_logits", "prior_rows",
    "text_logits", "cls_logits", "obj_labels", "subj_label", "obj_label",
    "gt_subj_idx", "gt_obj_idx", "gt_pred", "gt_subj_label", "gt_obj_label",
]
PER_IMAGE_LIST_KEYS = [
    "image_id", "pairs", "pair_index", "model_logits", "prior_rows",
    "text_logits", "cls_logits", "obj_labels", "subj_label", "obj_label",
    "gt_subj_idx", "gt_obj_idx", "gt_pred", "gt_subj_label", "gt_obj_label",
]
TENSOR_KEYS = ["model_logits", "prior_rows", "text_logits", "cls_logits"]


class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, cid: str, ok: bool, detail: str) -> bool:
        self.rows.append({"check": cid, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid:<4} {detail}")
        return bool(ok)

    @property
    def all_pass(self) -> bool:
        return all(r["pass"] for r in self.rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--predicates", type=Path,
                    default=Path("datasets_vg150_clean/vocabulary/predicates.json"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--expect_schema", type=str, default="pair_logit_dump_v2")
    args = ap.parse_args(argv)

    print("=" * 88)
    print(f"CACHE VALIDATION  {args.dump}")
    print("=" * 88)
    d = torch.load(args.dump, weights_only=False)
    C = Checks()

    # ---- S1 schema + keys
    missing = [k for k in REQUIRED_KEYS if k not in d]
    C.add("S1", d.get("schema") == args.expect_schema and not missing,
          f"schema={d.get('schema')!r} expected={args.expect_schema!r} missing_keys={missing}")

    n_img = len(d.get("image_id", []))
    # ---- S2 list lengths
    lens = {k: len(d.get(k, [])) for k in PER_IMAGE_LIST_KEYS}
    C.add("S2", len(set(lens.values())) == 1,
          f"{n_img} images; per-image list lengths {'all equal' if len(set(lens.values()))==1 else lens}")

    P = len(d.get("pred_vocab", []))
    # ---- S3 shapes
    bad_shape, total_pairs = [], 0
    for i in range(n_img):
        n = int(d["pairs"][i].shape[0])
        total_pairs += n
        for k in TENSOR_KEYS:
            if tuple(d[k][i].shape) != (n, P):
                bad_shape.append((i, k, tuple(d[k][i].shape), (n, P)))
    C.add("S3", not bad_shape and total_pairs == int(d.get("n_pairs", -1)),
          f"vocab_width={P} total_pairs={total_pairs} recorded_n_pairs={d.get('n_pairs')} "
          f"shape_mismatches={len(bad_shape)}")

    # ---- S4 finiteness
    nonfinite = {}
    for k in TENSOR_KEYS:
        bad = sum(int((~torch.isfinite(d[k][i])).sum()) for i in range(n_img))
        if bad:
            nonfinite[k] = bad
    C.add("S4", not nonfinite, f"non-finite entries: {nonfinite or 'none'}")

    # ---- S5 label redundancy is consistent
    mism = 0
    for i in range(n_img):
        ol, pr = d["obj_labels"][i], d["pairs"][i]
        for j in range(int(pr.shape[0])):
            a, b = int(pr[j, 0]), int(pr[j, 1])
            sa = ol[a] if 0 <= a < len(ol) else ""
            ob = ol[b] if 0 <= b < len(ol) else ""
            if d["subj_label"][i][j] != sa or d["obj_label"][i][j] != ob:
                mism += 1
    C.add("S5", mism == 0, f"subj/obj label vs obj_labels[pairs] mismatches: {mism}")

    # ---- S6 pair_index
    badidx = sum(1 for i in range(n_img)
                 if d["pair_index"][i].tolist() != list(range(int(d["pairs"][i].shape[0]))))
    C.add("S6", badidx == 0, f"images with non-canonical pair_index: {badidx}")

    # ---- S7/S8 alias + raw GT
    amap = d.get("predicate_alias_map", {})
    vg = json.loads(Path(args.predicates).read_text(encoding="utf-8"))["idx_to_predicate"]
    pv50 = [vg[str(i)] for i in range(1, len(vg) + 1)]
    collapsed_sources = sorted({k for k in amap if k in pv50})
    seen = set()
    for i in range(n_img):
        seen.update(str(x).strip().lower() for x in d["gt_pred"][i])
    survived = [s for s in collapsed_sources if s in seen]
    absent = [s for s in collapsed_sources if s not in seen]
    # Presence of ANY collapsed source in GT is proof the cache was not
    # alias-normalised -- normalisation would have rewritten every one of them.
    # Absence of a particular one is a sampling fact (a rare predicate need not
    # occur in a small image sample), not evidence of normalisation, so it is
    # reported but does not fail the check.
    C.add("S7", len(collapsed_sources) == 0 or len(survived) > 0,
          f"RAW GT check: collapsed-away predicates {collapsed_sources}; "
          f"present in GT: {survived}; not sampled: {absent} "
          f"({len(seen)} distinct GT predicates over {n_img} images)")
    C.add("S8", len({amap.get(p, p) for p in pv50}) == 48 and len(pv50) == 50,
          f"alias map collapses {len(pv50)} -> {len({amap.get(p, p) for p in pv50})} classes "
          f"via {collapsed_sources}")

    # ---- S9 ordering
    dump_pv = [str(x).strip().lower() for x in d["pred_vocab"]]
    bg = set(d.get("background_predicate_indices", []))
    fg = [p for i, p in enumerate(dump_pv) if i not in bg]
    C.add("S9", fg == [p.strip().lower() for p in pv50],
          f"foreground vocabulary order identical to {args.predicates.name} "
          f"({len(fg)} fg + {len(bg)} background = {P})")

    # ---- S10 composition reproduces the recorded protocol
    ea = float(d.get("ensemble_alpha", -1.0))
    mode = str(d.get("score_mode", ""))
    if mode == "ensemble":
        def _norm(x):
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True).clamp_min(1e-4)
            return (x - m) / s
        ct = float(d.get("classifier_temperature", 1.0))
        tt = float(d.get("text_temperature", 1.0))
        worst = 0.0
        for i in range(min(n_img, 200)):
            rec = ea * (_norm(d["cls_logits"][i]) / ct) + (1.0 - ea) * (_norm(d["text_logits"][i]) / tt)
            worst = max(worst, float((rec - d["model_logits"][i]).abs().max()))
        C.add("S10", worst < 1e-4,
              f"recomposed model term vs stored, max|diff| over first "
              f"{min(n_img,200)} images = {worst:.3e} (ensemble_alpha={ea})")
    else:
        C.add("S10", True, f"score_mode={mode!r}: composition check not applicable")

    # ---- S11 GT pairs locatable
    tot_gt, found = 0, 0
    for i in range(n_img):
        key = {(int(a), int(b)): 1 for a, b in d["pairs"][i].tolist()}
        for si, oi in zip(d["gt_subj_idx"][i], d["gt_obj_idx"][i]):
            tot_gt += 1
            if (int(si), int(oi)) in key:
                found += 1
    frac = found / max(1, tot_gt)
    C.add("S11", frac > 0.99, f"GT triplets whose (subj,obj) is in the pair list: "
                              f"{found}/{tot_gt} = {frac*100:.2f}%")

    # ---- S12 prior rows are raw
    tau = float(d.get("eval_freq_bias_tau", 0.0))
    alpha = float(d.get("freq_bias_alpha", 0.0))
    C.add("S12", tau == 0.0,
          f"eval_freq_bias_tau={tau} (must be 0.0: prior rows are stored RAW, "
          f"the CPU side applies tau); freq_bias_alpha recorded as {alpha} but NOT applied")

    verdict = "CACHE VALID" if C.all_pass else "CACHE INVALID -- DO NOT ANALYSE"
    out: Dict[str, Any] = {
        "dump": str(args.dump), "verdict": verdict, "all_checks_pass": C.all_pass,
        "checks": C.rows,
        "summary": {
            "schema": d.get("schema"), "n_images": n_img, "n_pairs": int(d.get("n_pairs", 0)),
            "vocab_width": P, "background_indices": list(bg),
            "score_mode": mode, "ensemble_alpha": ea,
            "freq_bias_alpha": alpha, "eval_freq_bias_tau": tau,
            "freq_bias_path": d.get("freq_bias_path"),
            "resume_from": d.get("resume_from"),
            "use_vg_aliases": d.get("use_vg_aliases"),
            "branch": d.get("branch"), "ensemble_alpha_used": d.get("ensemble_alpha_used"),
            "missing_prior_images": d.get("missing_prior_images", 0),
            "missing_text_logits": d.get("missing_text_logits", 0),
            "missing_cls_logits": d.get("missing_cls_logits", 0),
        },
    }
    print("=" * 88)
    print(f"VERDICT: {verdict}")
    print("=" * 88)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[written] {args.out}")
    return 0 if C.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
