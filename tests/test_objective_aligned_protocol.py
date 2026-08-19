"""Regression tests for scripts/train/train_objective_aligned.sh.

Same hazard as tests/test_historical_eval_protocol.py: train.py uses
``parse_known_args``, so a flag the parser does not define is **silently
discarded** and the corresponding stage-3 default stays in force with no
runtime signal. These tests caught exactly that during development --
``--eval_sgg_multi_predicate_topk`` was added to ``TrainConfig`` but not to
``build_argparser``, so it was being dropped while appearing to work (the
dataclass default happened to equal the requested value).

They additionally pin the property the whole run exists to establish: that
exactly ONE long-tail correction is active per arm. The repository previously
stacked three at training time (inverse-frequency class weights, focal loss,
logit adjustment) while evaluation applied a fourth in the opposite direction
(freq_bias alpha=3.75 pushes predictions back toward the prior).

No GPU, no data, no model.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train" / "train_objective_aligned.sh"

ARMS = {
    "baseline": {"CE_LOSS": "focal", "CE_WEIGHT_POWER": "0.5", "TAIL_ADJ": "false", "TAIL_TAU": "0.0"},
    "logit_adj": {"CE_LOSS": "ce", "CE_WEIGHT_POWER": "0.0", "TAIL_ADJ": "true", "TAIL_TAU": "1.0"},
    "none": {"CE_LOSS": "ce", "CE_WEIGHT_POWER": "0.0", "TAIL_ADJ": "false", "TAIL_TAU": "0.0"},
}

BASE_SUBS = {
    "${GPU_PRESET:-l4_24gb}": "l4_24gb", "${RUN_NAME}": "r", "${OUT_DIR}": "runs/r",
    "${OUT_DIR}/metrics.jsonl": "runs/r/metrics.jsonl", "${DATA_ROOT}": "datasets_vg150_clean",
    "${SEED:-0}": "0", "${EPOCHS:-6}": "6", "${BATCH_SIZE:-12}": "12", "${NUM_WORKERS:-4}": "4",
    "${CLIP_INPUT_RES:-336}": "336", "${MAX_IMAGES:-0}": "0",
    "${SAMPLES_PER_EPOCH:-20000}": "20000", "${LR:-2e-5}": "2e-5",
    "${NEGATIVE_PAIR_RATIO:-2.0}": "2.0", "${LAMBDA_PREDICATE_CE:-2.0}": "2.0",
    "${LAMBDA_SPOA:-0.75}": "0.75", "${LAMBDA_GROUND:-0.25}": "0.25",
    "${EVAL_BATCHES:-300}": "300",
}


def _tokens() -> list[str]:
    assert SCRIPT.exists(), f"missing training entrypoint: {SCRIPT}"
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"^PROTOCOL=\(\s*$(.*?)^\)\s*$", text, re.M | re.S)
    assert m, "PROTOCOL=( ... ) array not found; update this test if the script was restructured"
    toks: list[str] = []
    for line in m.group(1).splitlines():
        line = line.split("#")[0].strip()
        if line:
            toks.extend(shlex.split(line))
    return toks


def _argv(arm: str) -> list[str]:
    subs = dict(BASE_SUBS)
    subs.update({"${" + k + "}": v for k, v in ARMS[arm].items()})
    return [subs.get(t, t) for t in _tokens()]


def _flags() -> list[str]:
    return [t[2:] for t in _tokens() if t.startswith("--")]


def _resolve(arm: str):
    """Replicate train.py main()'s merge order exactly (see train.py:1250-1326)."""
    from openvocab_rel.config import TrainConfig, apply_gpu_preset, apply_stage_config
    from openvocab_rel.train import build_argparser

    argv = _argv(arm)
    args, _unknown = build_argparser().parse_known_args(argv)
    cfg = TrainConfig()
    cfg.stage = args.stage
    cfg = apply_stage_config(cfg)
    called = [a[2:].split("=")[0] for a in argv if a.startswith("--")]
    for key, value in vars(args).items():
        if key in called:
            setattr(cfg, key, value)
        elif not hasattr(cfg, key) or getattr(cfg, key) is None:
            setattr(cfg, key, value)
    if str(getattr(cfg, "gpu_preset", "")).strip():
        cfg = apply_gpu_preset(cfg, str(cfg.gpu_preset))
    for key in called:
        if hasattr(cfg, key) and hasattr(args, key):
            setattr(cfg, key, getattr(args, key))
    return cfg


def test_every_flag_is_recognised_by_the_argparser():
    from openvocab_rel.train import build_argparser

    known = set(vars(build_argparser().parse_known_args([])[0]).keys())
    unknown = sorted(set(_flags()) - known)
    assert not unknown, (
        f"{SCRIPT.name} passes flags the argparser does not define: {unknown}. "
        "parse_known_args DISCARDS these silently -- the stage-3 default stays "
        "in force with no error. Add the flag to build_argparser()."
    )


def test_every_flag_is_a_trainconfig_field():
    from openvocab_rel.config import TrainConfig

    fields = set(TrainConfig().__dict__.keys())
    orphans = sorted(set(_flags()) - fields - {"eval_only"})
    assert not orphans, f"flags that are not TrainConfig fields: {orphans}"


def test_no_duplicate_flags():
    flags = _flags()
    dupes = sorted({f for f in flags if flags.count(f) > 1})
    assert not dupes, f"passed more than once (later value silently wins): {dupes}"


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_argparse_consumes_every_token(arm):
    from openvocab_rel.train import build_argparser

    _args, unknown = build_argparser().parse_known_args(_argv(arm))
    assert not unknown, f"arm={arm}: tokens silently ignored: {unknown}"


# Which long-tail corrections each arm is SUPPOSED to activate. `baseline`
# deliberately keeps the repository's existing stacked pair (inverse-frequency
# weights AND focal) -- that is precisely what makes it the control for "we
# changed nothing about the correction, we just trained the head properly".
# The comparison baseline -> logit_adj is therefore "three-way-stacked, ad hoc"
# versus "one principled correction", which is the question worth answering.
EXPECTED_CORRECTIONS = {
    "baseline": {"inverse_frequency_weights", "focal"},
    "logit_adj": {"logit_adjustment"},
    "none": set(),
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_long_tail_corrections_match_the_arm_specification(arm):
    """The central design property of this run.

    The repository stacks up to three corrections for the same imbalance at
    training time (inverse-frequency class weights, focal loss, logit
    adjustment) while the evaluator applies a fourth in the *opposite*
    direction (freq_bias alpha=3.75 pushes predictions back toward the prior).
    Each arm must therefore declare exactly which mechanisms it activates, so
    a change in mR@50 is attributable to a named mechanism rather than to an
    unlabelled combination.
    """
    cfg = _resolve(arm)
    active = {
        name
        for name, on in {
            "inverse_frequency_weights": float(cfg.predicate_ce_weight_power) > 0.0,
            "focal": str(cfg.predicate_ce_loss) == "focal",
            "logit_adjustment": bool(cfg.tail_logit_adjustment_enabled),
        }.items()
        if on
    }
    assert active == EXPECTED_CORRECTIONS[arm], (
        f"arm '{arm}' activated {sorted(active) or 'nothing'}, "
        f"expected {sorted(EXPECTED_CORRECTIONS[arm]) or 'nothing'}"
    )


def test_the_arms_actually_differ_in_their_correction():
    """Guard against all three arms silently collapsing to the same run."""
    sets = [frozenset(EXPECTED_CORRECTIONS[a]) for a in ARMS]
    assert len(set(sets)) == len(sets), "arms must be mutually distinguishable"


def test_only_one_arm_stacks_corrections():
    stacked = [a for a, s in EXPECTED_CORRECTIONS.items() if len(s) > 1]
    assert stacked == ["baseline"], (
        "only the 'baseline' control may stack corrections; every other arm "
        f"must apply at most one. Stacked arms: {stacked}"
    )


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_model_learns_the_background_class(arm):
    """use_all_pairs=false meant CE never saw a negative pair, so the
    background class (predicate_classifier_classes=51) was never trained."""
    cfg = _resolve(arm)
    assert bool(cfg.use_all_pairs) is True
    assert float(cfg.negative_pair_ratio) > 0.0


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_head_the_evaluator_scores_is_actually_trained(arm):
    cfg = _resolve(arm)
    assert bool(cfg.predicate_classifier_enabled) is True
    assert float(cfg.lambda_predicate_ce) > 0.0
    assert float(cfg.lr) > 1e-6, (
        "the historical run used lr=2e-6, which left the predicate classifier "
        "at random initialisation while the evaluator scored it"
    )


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_in_training_eval_is_raw_not_calibrated(arm):
    """Raw and system-level debiased scores must never be conflated.

    Calibrated numbers are produced separately, so a checkpoint is never
    selected on a score the frequency prior is doing the work for.
    """
    cfg = _resolve(arm)
    assert str(cfg.eval_sgg_predicate_score_mode) == "classifier"
    assert bool(cfg.freq_bias_enabled) is False
    assert bool(cfg.adaptive_calibration_enabled) is False
    assert float(cfg.bayes_calibration_weight) == 0.0


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_literature_comparable_metric_is_enabled(arm):
    cfg = _resolve(arm)
    assert int(cfg.eval_sgg_multi_predicate_topk) > 0


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_architecture_is_identical_to_the_frozen_baseline(arm):
    """This run changes the OBJECTIVE, never the model.

    If any of these drift, the comparison against the frozen baseline on main
    stops being an objective ablation and becomes an architecture change.
    """
    cfg = _resolve(arm)
    assert bool(cfg.explicit_spoa_enabled) is True
    assert bool(cfg.asymmetric_pair_fusion_enabled) is False
    assert int(cfg.clip_input_res) == 336
    assert int(cfg.relation_context_layers) == 0


def test_script_refuses_to_overwrite_and_checks_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "ALLOW_OVERWRITE" in text
    assert "refusing to overwrite" in text.lower()
    assert "MISSING REQUIRED ARTIFACT" in text
    assert "RUN_STAMP" in text and "date -u" in text


def test_script_prints_the_control_to_beat():
    """A run whose banner does not state the frequency-prior control invites
    reporting an absolute score as if it were evidence."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "64.37" in text and "20.30" in text
    assert "frequency_prior_baseline" in text


def test_unknown_arm_is_rejected():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "unknown ARM" in text
    for arm in ARMS:
        assert arm in text
