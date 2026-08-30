#!/usr/bin/env python
"""Write a class-balanced copy of a frequency prior, for the model to consume.

WHY THIS EXISTS
---------------
docs/PHASE4_SCIENTIFIC_REASSESSMENT.md section 9 names the one experiment that
would actually settle whether the model contributes anything beyond
recalibration:

    does  model + tau-adjusted prior  beat  tau-adjusted prior alone?

Measuring that needs the model's per-pair scores composed with an ADJUSTED
prior. Rather than change how the evaluator applies the prior -- which would
alter evaluation semantics and make the result incomparable to every earlier
run -- this tool applies the adjustment to the prior FILE:

    log P'(p | s,o) = log P(p | s,o) - tau * log P(p)

The evaluator then loads it through the unmodified `--freq_bias_path` code
path and cannot tell the difference. Nothing in openvocab_rel/ changes.

log P(p) is the prior's own `global_log_probs`, i.e. the TRAIN marginal. The
evaluation split is never touched, so this cannot leak.

Rows are re-normalised to log-probabilities after adjustment (log-sum-exp),
because _load_frequency_bias consumes them as log-probs and downstream
smoothing/default handling assumes that scale. Re-normalisation is monotone
within a row, so it changes no per-pair argmax -- only the cross-pair
comparability that R@K ranking depends on.

SAFETY: writes a NEW file and refuses to overwrite an existing one. The
historical prior at checkpoints/demo_best/frequency_prior.json is immutable
evidence and is only ever read.

Usage:
    python tools/make_recalibrated_prior.py --tau 0.1 \
        --in datasets_vg150_clean/frequency_prior_train.json \
        --out datasets_vg150_clean/frequency_prior_train_tau0.1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROW_KEYS = ("pair_log_probs", "subject_log_probs", "object_log_probs")


def _normalise(row: List[float]) -> List[float]:
    m = max(row)
    total = sum(math.exp(v - m) for v in row)
    log_z = m + math.log(total)
    return [v - log_z for v in row]


def recalibrate(raw: Dict[str, Any], tau: float) -> Dict[str, Any]:
    marginal = list(raw["global_log_probs"])
    n = len(marginal)

    def adjust(row: List[float]) -> List[float]:
        if len(row) != n:
            return list(row)
        return _normalise([row[i] - tau * marginal[i] for i in range(n)])

    out: Dict[str, Any] = dict(raw)
    for key in ROW_KEYS:
        table = raw.get(key)
        if isinstance(table, dict):
            out[key] = {k: adjust(v) for k, v in table.items()}
    # The global row is adjusted too, so a backed-off lookup is treated
    # consistently with a conditioned one.
    out["global_log_probs"] = adjust(marginal)

    prov = dict(raw.get("provenance", {})) if isinstance(raw.get("provenance"), dict) else {}
    prov.update({
        "recalibrated": True,
        "recalibration_tau": tau,
        "recalibration_formula": "log P'(p|s,o) = normalise(log P(p|s,o) - tau * log P_train(p))",
        "recalibration_note": (
            "Class-balancing adjustment applied to the PRIOR FILE so the evaluator's "
            "prior-loading path is unchanged. tau=0 reproduces the source file's "
            "decisions exactly. The marginal used is the source prior's own "
            "global_log_probs (train split); the evaluation split is never read."
        ),
    })
    out["provenance"] = prov
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--force", action="store_true", help="allow overwriting the output file")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        print(f"[FAIL] missing input prior: {src}", file=sys.stderr)
        return 2
    if dst.exists() and not args.force:
        print(f"[FAIL] {dst} exists; refusing to overwrite (pass --force)", file=sys.stderr)
        return 2
    if dst.resolve() == src.resolve():
        print("[FAIL] output would overwrite the input prior", file=sys.stderr)
        return 2

    raw = json.loads(src.read_text(encoding="utf-8"))
    out = recalibrate(raw, float(args.tau))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out), encoding="utf-8")

    print(f"[recalibrate] tau={args.tau}")
    print(f"[recalibrate] {src}  ->  {dst}")
    print(f"[recalibrate] pair rows      : {len(out.get('pair_log_probs', {})):,}")
    print(f"[recalibrate] predicate vocab: {len(out.get('predicate_vocab', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
