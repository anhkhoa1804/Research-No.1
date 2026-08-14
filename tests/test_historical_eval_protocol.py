"""Regression tests for the historical-checkpoint evaluation protocol.

These tests exist because of one specific hazard, documented in
``docs/known_issues.md``: ``openvocab_rel/train.py`` uses
``parse_known_args``, which **silently discards unrecognized flags**. A typo
in ``scripts/eval/eval_historical_checkpoint.sh`` -- say
``--explicit_spoa_enable`` for ``--explicit_spoa_enabled`` -- would leave
``--stage 3``'s default of ``True`` in place, load the recovered checkpoint
into an architecture it was never trained for, and emit **no runtime signal
at all**. The run would complete and report plausible numbers.

That failure cannot be caught at runtime, so it is caught here instead: a
typo in the script becomes a test failure at commit time rather than a wasted
GPU run producing a silently-wrong research result.

Nothing here needs the real checkpoint, dataset, or GPU -- the script is
parsed as text and the config is resolved in-process.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval" / "eval_historical_checkpoint.sh"

# The frozen historical protocol. Source of truth for these values is
# docs/HISTORICAL_CHECKPOINT_MANIFEST.md section 4, which in turn derives them
# from checkpoints/demo_best/demo_config.env and the checkpoint's embedded cfg.
FROZEN_PROTOCOL = {
    "eval_sgg_predicate_score_mode": "ensemble",
    "eval_sgg_predicate_ensemble_alpha": 0.0,
    "eval_sgg_use_gt_pairs": True,
    "explicit_spoa_enabled": False,
    "text_conditioned_projection_enabled": False,
    "relationness_enabled": False,
    "eval_sgg_use_relationness": False,
    "adaptive_calibration_enabled": True,
    "bayes_calibration_weight": 0.0,
    "freq_bias_enabled": True,
    "freq_bias_alpha": 3.75,
    "freq_bias_smoothing": 1.0,
    "clip_input_res": 336,
    "eval_sgg_grounding_dino_enabled": False,
    "vg150_source": "local-jsonl",
    "vg150_root": "datasets_vg150_clean",
    "eval_fast_mode": False,
    "epochs": 0,
    "resume": True,
}

# Shell variables the script sets before building PROTOCOL, with the values it
# sets them to. Kept explicit rather than shelling out, so these tests run
# identically on a machine without bash.
SHELL_SUBSTITUTIONS = {
    "${STAGE}": "3",
    "${GPU_PRESET}": "l4_24gb",
    "${CKPT}": "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
    "${FREQ_PRIOR}": "checkpoints/demo_best/frequency_prior.json",
    "${DATA_ROOT}": "datasets_vg150_clean",
    "${DEVICE}": "cuda",
    "${BATCH_SIZE}": "12",
    "${NUM_WORKERS}": "4",
    "${CLIP_INPUT_RES}": "336",
    "${EVAL_BATCHES}": "2",
    "${ENSEMBLE_ALPHA}": "0.0",
    "${FREQ_BIAS_ALPHA}": "3.75",
    "${FREQ_BIAS_SMOOTHING}": "1.0",
    "${RUN_NAME}": "test_run",
    "${OUT_DIR}": "runs/test_run",
    "${OUT_DIR}/metrics.jsonl": "runs/test_run/metrics.jsonl",
}


def _script_text() -> str:
    assert SCRIPT_PATH.exists(), f"missing evaluation entrypoint: {SCRIPT_PATH}"
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _protocol_tokens() -> list[str]:
    """Extract the PROTOCOL=( ... ) array from the shell script, as tokens."""
    text = _script_text()
    match = re.search(r"^PROTOCOL=\(\s*$(.*?)^\)\s*$", text, re.M | re.S)
    assert match, (
        "could not find a `PROTOCOL=( ... )` array in "
        f"{SCRIPT_PATH.name}. These tests parse that array; if the script's "
        "structure changed, update this test rather than deleting it."
    )
    tokens: list[str] = []
    for line in match.group(1).splitlines():
        line = line.split("#")[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _resolved_argv() -> list[str]:
    return [SHELL_SUBSTITUTIONS.get(tok, tok) for tok in _protocol_tokens()]


def _flag_names() -> list[str]:
    return [tok[2:] for tok in _protocol_tokens() if tok.startswith("--")]


# ---------------------------------------------------------------------------
# 1. Every flag the script passes must actually exist.
# ---------------------------------------------------------------------------

def test_script_exists_and_declares_a_protocol_array():
    tokens = _protocol_tokens()
    assert tokens, "PROTOCOL array is empty"
    assert "--stage" in tokens


def test_every_flag_is_recognised_by_the_argparser():
    """A flag the argparser does not know is silently discarded at runtime."""
    from openvocab_rel.train import build_argparser

    parser = build_argparser()
    known = set(vars(parser.parse_known_args([])[0]).keys())
    unknown = sorted(set(_flag_names()) - known)
    assert not unknown, (
        f"{SCRIPT_PATH.name} passes flags the argparser does not define: {unknown}. "
        "train.py uses parse_known_args, so these are SILENTLY DISCARDED -- the "
        "corresponding stage-3 default would stay in effect with no error. "
        "Fix the flag name in the script."
    )


def test_every_flag_maps_to_a_real_trainconfig_field():
    """A flag that is not a TrainConfig field cannot influence the run."""
    from openvocab_rel.config import TrainConfig

    fields = set(TrainConfig().__dict__.keys())
    # `eval_only` is an argparse-only switch consumed by main()'s merge loop
    # rather than a dataclass field; it is set dynamically on the config.
    argparse_only = {"eval_only"}
    orphans = sorted(set(_flag_names()) - fields - argparse_only)
    assert not orphans, (
        f"{SCRIPT_PATH.name} passes flags that are not TrainConfig fields: {orphans}"
    )


def test_no_flag_is_passed_twice():
    """A duplicated flag means the later value silently wins -- ambiguous intent."""
    names = _flag_names()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"flags passed more than once: {duplicates}"


def test_argv_is_well_formed_no_unconsumed_tokens():
    from openvocab_rel.train import build_argparser

    _args, unknown = build_argparser().parse_known_args(_resolved_argv())
    assert not unknown, (
        f"argparse could not consume these tokens: {unknown}. "
        "They would be silently ignored at runtime."
    )


# ---------------------------------------------------------------------------
# 2. The flags must RESOLVE to the frozen protocol after all preset merging.
# ---------------------------------------------------------------------------

def _resolve_config():
    """Replicate openvocab_rel/train.py main()'s config merge order exactly.

    See train.py lines ~1250-1326:
        apply_stage_config -> explicit-CLI merge -> apply_gpu_preset
        -> _reapply_explicit_cli_args
    """
    from openvocab_rel.config import TrainConfig, apply_gpu_preset, apply_stage_config
    from openvocab_rel.train import build_argparser

    argv = _resolved_argv()
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

    for key in called:  # _reapply_explicit_cli_args
        if hasattr(cfg, key) and hasattr(args, key):
            setattr(cfg, key, getattr(args, key))
    return cfg


@pytest.mark.parametrize("field,expected", sorted(FROZEN_PROTOCOL.items(), key=lambda kv: kv[0]))
def test_resolved_config_matches_frozen_protocol(field, expected):
    """The value that survives every preset must be the historical one."""
    cfg = _resolve_config()
    actual = getattr(cfg, field)
    if isinstance(expected, bool):
        assert bool(actual) is expected, f"{field} resolved to {actual!r}, expected {expected!r}"
    elif isinstance(expected, (int, float)):
        assert abs(float(actual) - float(expected)) < 1e-9, (
            f"{field} resolved to {actual!r}, expected {expected!r}"
        )
    else:
        assert str(actual) == str(expected), f"{field} resolved to {actual!r}, expected {expected!r}"


def test_stage_three_defaults_really_are_dangerous_for_this_checkpoint():
    """Pin the reason this whole file exists.

    If a future change makes stage 3 safe by default, this test fails and
    someone must consciously decide whether the dedicated entrypoint is still
    needed -- rather than the justification quietly becoming false while the
    documentation still claims it.
    """
    from openvocab_rel.config import TrainConfig, apply_stage_config

    cfg = TrainConfig()
    cfg.stage = 3
    cfg = apply_stage_config(cfg)

    assert cfg.explicit_spoa_enabled is True
    assert cfg.text_conditioned_projection_enabled is True
    assert cfg.relationness_enabled is True
    assert cfg.eval_sgg_use_relationness is True
    assert int(cfg.clip_input_res) == 448
    assert abs(float(cfg.eval_sgg_predicate_ensemble_alpha) - 0.45) < 1e-9
    assert cfg.eval_sgg_use_gt_pairs is False
    assert cfg.eval_sgg_grounding_dino_enabled is True


def test_explicit_cli_flags_win_over_stage_and_gpu_presets():
    """The override mechanism the entire protocol depends on."""
    cfg = _resolve_config()
    stage3_would_force = {
        "explicit_spoa_enabled": True,
        "text_conditioned_projection_enabled": True,
        "relationness_enabled": True,
        "eval_sgg_use_relationness": True,
    }
    for field, forced in stage3_would_force.items():
        assert bool(getattr(cfg, field)) is not forced, (
            f"{field} was NOT overridden -- it kept stage 3's {forced!r}. "
            "The explicit CLI flag failed to take effect."
        )


# ---------------------------------------------------------------------------
# 3. The script's own safety rails.
# ---------------------------------------------------------------------------

def test_script_requires_an_explicit_mode():
    text = _script_text()
    assert "--canary" in text and "--full" in text
    assert "specify --canary or --full" in text


def test_script_refuses_to_overwrite_an_existing_run_directory():
    text = _script_text()
    assert "ALLOW_OVERWRITE" in text
    assert "Refusing to overwrite" in text


def test_full_run_is_gated_on_a_passing_canary():
    text = _script_text()
    assert "ALLOW_UNGATED_FULL" in text
    assert "canary_verdict.txt" in text
    assert "no passing canary found" in text


def test_script_prints_resolved_protocol_before_executing():
    text = _script_text()
    banner = text.index("RESOLVED PROTOCOL")
    launch = text.index("launching evaluation")
    assert banner < launch, "the protocol banner must print before the run starts"


def test_script_runs_preflight_before_the_model():
    text = _script_text()
    assert "tools/gcp_preflight.py" in text
    assert text.index("gcp_preflight.py") < text.index("launching evaluation")
    assert "--strict" in text


def test_script_verifies_the_run_afterwards():
    text = _script_text()
    assert "tools/verify_canary.py" in text
    assert "PROTOCOL VERIFICATION FAILED" in text


def test_script_captures_required_artifacts():
    text = _script_text()
    for artifact in ("command.txt", "git_commit.txt", "environment.txt",
                     "run.log", "metrics.jsonl", "manifest.yaml"):
        assert artifact in text, f"{artifact} is not captured by the script"


def test_script_checks_required_artifacts_exist_before_running():
    text = _script_text()
    assert "MISSING REQUIRED ARTIFACT" in text
    assert text.index("MISSING REQUIRED ARTIFACT") < text.index("launching evaluation")


def test_run_directory_name_is_unique_by_default():
    """Two runs must not silently share an output directory."""
    text = _script_text()
    assert "RUN_STAMP" in text
    assert "date -u" in text
    assert "${RUN_PREFIX}_${RUN_STAMP}" in text
