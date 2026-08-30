#!/usr/bin/env python
"""Verify a run's RESOLVED protocol against the frozen historical configuration.

The point of this tool is the distinction between *intended* and *resolved*
settings. ``scripts/eval/eval_historical_checkpoint.sh`` passes every
compatibility flag explicitly, but ``openvocab_rel/train.py`` uses
``parse_known_args``, which silently discards unrecognized flags -- so a
mistyped flag would leave a stage-3 default in place with no runtime signal
whatsoever. Reading the script tells you what was *asked for*. Only
``metrics.jsonl`` tells you what the evaluator *actually used*.

Every check below therefore reads the ``config`` / ``experiment`` blocks that
``_attach_experiment_snapshot`` (``train.py``) embeds into each metrics row,
which are dumps of the live ``TrainConfig`` after all preset and CLI merging.

This tool computes nothing about model quality and asserts nothing about
R@K/mR@K values. It checks configuration identity only. A canary that PASSES
means "this run used the protocol we think it did", not "the numbers are
good".

Usage:
    python tools/verify_canary.py runs/<run>/metrics.jsonl [--out <verdict.txt>]

Exit codes:
    0  PASS -- resolved protocol matches the frozen configuration
    1  FAIL -- at least one setting differs, or the run is pathological
    2  could not read/parse the metrics file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_PATH = Path("data/manifests/historical_checkpoint_v1.yaml")

# (config key, expected value, why this matters if it drifts)
EXPECTED: List[Tuple[str, Any, str]] = [
    ("eval_sgg_predicate_score_mode", "ensemble",
     "demo_config.env EVAL_SCORE_MODE; the historical scoring path"),
    ("eval_sgg_predicate_ensemble_alpha", 0.0,
     "alpha>0 mixes in this checkpoint's UNTRAINED classifier head; "
     "0.0 means 100% CLIP text-cosine"),
    ("eval_sgg_use_gt_pairs", True,
     "demo_config.env PROTOCOL=PredCls_GT_pair"),
    ("explicit_spoa_enabled", False,
     "the checkpoint predates the SPOA branches entirely"),
    ("text_conditioned_projection_enabled", False,
     "forced true by stage 3 (config.py:742); no trained weights exist"),
    ("relationness_enabled", False,
     "forced true by stage 3 (config.py:744); head would be random-init"),
    ("eval_sgg_use_relationness", False,
     "forced true by stage 3 (config.py:746); would prune pairs on random scores"),
    ("adaptive_calibration_enabled", True,
     "the checkpoint's own embedded config has this true"),
    ("bayes_calibration_weight", 0.0,
     "held at 0 so the frequency prior is the ONLY calibration applied"),
    ("freq_bias_enabled", True,
     "required for the alpha=3.75 prior to have any effect at all"),
    ("freq_bias_alpha", 3.75,
     "demo_config.env FREQ_BIAS_ALPHA"),
    ("clip_input_res", 336,
     "checkpoint is clip-vit-large-patch14-336; stage 3 would give 448"),
    ("eval_sgg_grounding_dino_enabled", False,
     "PredCls over GT boxes needs no detector"),
    ("vg150_source", "local-jsonl",
     "the maintained loader backend"),
]

# The RUNTIME predicate vocabulary is the 50 canonical VG150 predicates plus one
# synthetic "relation" background class appended at index 50 by
# VG150DataLoader (vg150_loader.py, "predicate ids must match the 51-way
# classifier head"), and documented in data/manifests/vg150.yaml:
#     num_predicates: 50   # +1 synthetic "relation" background class at runtime
#
# This check previously compared that runtime count against the ON-DISK count
# of 50 and therefore failed on every correctly-configured run. Comparing the
# count alone was also the weaker check: a vocabulary of the right SIZE can
# still carry the wrong ORDER, which silently misaligns every predicate index
# against the checkpoint and the frequency prior. The size and the exact
# ordering are both verified below, against a hash derived from
# STANDARD_VG150_PREDICATES rather than a hardcoded digest.
EXPECTED_NUM_PREDICATES = 51
BACKGROUND_PREDICATE = "relation"


def expected_predicate_vocab() -> Optional[List[str]]:
    """The 51-entry runtime vocabulary this protocol requires, in index order."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES  # type: ignore
    except Exception:
        return None
    return sorted(STANDARD_VG150_PREDICATES) + [BACKGROUND_PREDICATE]


def expected_predicate_vocab_hash() -> Optional[str]:
    """Recompute train.py's predicate_vocab_hash for the expected ordering.

    Mirrors _stable_hash_json in openvocab_rel/train.py. Derived, never
    hardcoded, so it cannot drift away from the canonical vocabulary.
    """
    vocab = expected_predicate_vocab()
    if vocab is None:
        return None
    import hashlib
    raw = json.dumps(vocab, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class Verdict:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.lines: List[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {label}"
        if not ok and detail:
            line += f"\n         {detail}"
        self.lines.append(line)
        print(line)
        if not ok:
            self.failures.append(label)
        return ok

    def note(self, text: str) -> None:
        self.lines.append(text)
        print(text)


def _num_eq(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def load_last_row(path: Path) -> Optional[Dict[str, Any]]:
    """Return the last non-empty JSON object in a .jsonl file."""
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def expected_checkpoint_sha() -> Optional[str]:
    if not MANIFEST_PATH.exists():
        return None
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        return str(data["artifacts"]["checkpoint"]["sha256"])
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", help="path to runs/<run>/metrics.jsonl")
    ap.add_argument("--out", default="", help="write the verdict to this file")
    args = ap.parse_args(argv)

    metrics_path = Path(args.metrics)
    print("=" * 70)
    print("  CANARY VERIFICATION -- resolved protocol vs. frozen configuration")
    print("=" * 70)
    print(f"  metrics: {metrics_path}")

    if not metrics_path.exists():
        print(f"\n[ERROR] metrics file not found: {metrics_path}")
        print("        The evaluation did not produce output. Nothing to verify.")
        return 2

    row = load_last_row(metrics_path)
    if row is None:
        print(f"\n[ERROR] no parseable JSON row in {metrics_path}")
        return 2

    cfg = row.get("config") or (row.get("experiment") or {}).get("train_config") or {}
    exp = row.get("experiment") or {}
    if not cfg:
        print("\n[ERROR] metrics row has no 'config' block -- cannot verify resolved settings.")
        print("        Expected _attach_experiment_snapshot to embed it (train.py).")
        return 2

    v = Verdict()

    v.note("\n--- resolved runtime settings ------------------------------------")
    for key, want, why in EXPECTED:
        if key not in cfg:
            v.check(False, f"{key} present in resolved config", f"key absent from metrics 'config' block")
            continue
        got = cfg[key]
        if isinstance(want, bool):
            ok = bool(got) is want
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            ok = _num_eq(got, want)
        else:
            ok = str(got) == str(want)
        v.check(ok, f"{key} == {want!r}", f"resolved to {got!r}. Why it matters: {why}")

    v.note("\n--- predicate vocabulary -----------------------------------------")
    n_pred = exp.get("num_predicates")
    if n_pred is None:
        v.check(False, f"predicate vocabulary size == {EXPECTED_NUM_PREDICATES}",
                "experiment.num_predicates absent")
    else:
        v.check(int(n_pred) == EXPECTED_NUM_PREDICATES,
                f"predicate vocabulary size == {EXPECTED_NUM_PREDICATES}",
                f"resolved to {n_pred}; a different size means a different index mapping "
                f"than the checkpoint and the frequency prior were built against")
    # Size alone does not establish index alignment; the ORDER is what the
    # checkpoint's classifier head and the frequency prior are indexed against.
    recorded_hash = exp.get("predicate_vocab_hash")
    expected_hash = expected_predicate_vocab_hash()
    if recorded_hash and expected_hash:
        v.check(
            str(recorded_hash) == str(expected_hash),
            "predicate vocabulary ORDER == sorted(STANDARD_VG150_PREDICATES) + ['relation']",
            f"recorded predicate_vocab_hash {recorded_hash!r} != expected {expected_hash!r}; "
            "the vocabulary is the right size but a different ordering, which misaligns "
            "every predicate index against the checkpoint and the frequency prior",
        )
    elif recorded_hash:
        v.note(f"  [WARN] could not derive the expected vocabulary hash; "
               f"recorded {recorded_hash!r} left unchecked")
    if recorded_hash:
        v.note(f"  [INFO] predicate_vocab_hash = {recorded_hash}")

    v.note("\n--- checkpoint identity ------------------------------------------")
    resumed = str(cfg.get("resume_from", ""))
    v.check("demo_best" in resumed and resumed.endswith(".pt"),
            "resumed from the historical checkpoint",
            f"resume_from = {resumed!r}")
    sha = expected_checkpoint_sha()
    if sha:
        v.note(f"  [INFO] manifest checkpoint sha256 = {sha}")
        v.note("  [INFO] hash itself is enforced by tools/gcp_preflight.py, not here")

    v.note("\n--- frequency prior actually configured --------------------------")
    fbp = str(cfg.get("freq_bias_path", ""))
    v.check(bool(fbp) and Path(fbp).name == "frequency_prior.json",
            "freq_bias_path points at frequency_prior.json",
            f"freq_bias_path = {fbp!r}")
    if fbp:
        exists = Path(fbp).exists()
        v.check(exists, "frequency prior file exists at the configured path",
                f"{fbp} not found. _load_frequency_bias returns None SILENTLY, so the run "
                f"would have been UNCALIBRATED while reporting normally")

    v.note("\n--- run sanity ---------------------------------------------------")
    n_images = row.get("n_images_evaluated", row.get("num_images"))
    if n_images is not None:
        v.check(float(n_images) > 0, "evaluated at least one image", f"n_images = {n_images}")
    num_gt = row.get("num_gt")
    if num_gt is not None:
        v.check(float(num_gt) > 0, "ground-truth triplets were found",
                f"num_gt = {num_gt}; zero GT means R@K/mR@K are meaningless")

    for k in ("R@50", "mR@50"):
        if k in row:
            val = row[k]
            try:
                bad = val != val  # NaN
            except Exception:
                bad = True
            v.check(not bad, f"{k} is not NaN", f"{k} = {val}")

    pd = row.get("predicate_diag") or {}
    if pd:
        v.note(f"  [INFO] predicate_diag.score_mode    = {pd.get('score_mode')}")
        v.note(f"  [INFO] predicate_diag.ensemble_alpha = {pd.get('ensemble_alpha')}")
        if "score_mode" in pd:
            v.check(str(pd["score_mode"]) == "ensemble",
                    "predicate_diag confirms score_mode=ensemble",
                    f"diagnostic reports {pd['score_mode']!r}")
        if "ensemble_alpha" in pd:
            v.check(_num_eq(pd["ensemble_alpha"], 0.0),
                    "predicate_diag confirms ensemble_alpha=0.0",
                    f"diagnostic reports {pd['ensemble_alpha']!r}")

    pp = row.get("pair_proposal") or {}
    if pp:
        v.note(f"  [INFO] pair_proposal.gt_pair_recall@32 = {pp.get('gt_pair_recall@32')}")
        v.note(f"  [INFO] avg candidate pairs/image       = {pp.get('avg_candidate_pairs_per_image')}")

    v.note("\n--- observed metrics (recorded, NOT asserted) --------------------")
    v.note("  These are reported for the log. This tool does not judge them:")
    v.note("  a canary PASS means the protocol is right, not that the result is good.")
    for k in ("R@20", "R@50", "R@100", "mR@20", "mR@50", "mR@100",
              "image_mean_R@50", "head_mR@50", "body_mR@50", "tail_mR@50"):
        if k in row:
            try:
                v.note(f"  [INFO] {k:<18} = {float(row[k]) * 100:.2f} %")
            except (TypeError, ValueError):
                v.note(f"  [INFO] {k:<18} = {row[k]}")
    if exp.get("git_commit"):
        v.note(f"  [INFO] git_commit         = {exp['git_commit']}")

    print("\n" + "=" * 70)
    if v.failures:
        header = f"FAIL -- {len(v.failures)} protocol check(s) failed"
        print(f"  CANARY {header}")
        print("=" * 70)
        for f in v.failures:
            print(f"    - {f}")
        print("\n  DO NOT PROCEED TO THE FULL RUN. These numbers were produced")
        print("  under a different protocol than the historical configuration.")
        status = "FAIL"
    else:
        header = "PASS -- resolved protocol matches the frozen configuration"
        print(f"  CANARY {header}")
        print("=" * 70)
        print("  This confirms configuration identity only, not result quality.")
        status = "PASS"
    print("=" * 70)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"{status}\n")
            f.write(f"metrics: {metrics_path}\n")
            f.write(f"failures: {len(v.failures)}\n")
            for fail in v.failures:
                f.write(f"  - {fail}\n")
            f.write("\n")
            f.write("\n".join(v.lines))
            f.write("\n")
        print(f"[INFO] verdict written to {out_path}")

    return 1 if v.failures else 0


if __name__ == "__main__":
    sys.exit(main())
