"""Regression tests for tools/verify_canary.py.

The canary verifier is the last gate before a full GPU reproduction run, and
the only check that reads what the evaluator *actually resolved* rather than
what the launch script *intended*. Every failure mode it is supposed to catch
is pinned here against a synthetic ``metrics.jsonl``.

No real checkpoint, dataset, GPU or CLIP weights are needed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIER = REPO_ROOT / "tools" / "verify_canary.py"

# A metrics row whose resolved config matches the frozen historical protocol.
GOOD_CONFIG = {
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
    "clip_input_res": 336,
    "eval_sgg_grounding_dino_enabled": False,
    "vg150_source": "local-jsonl",
    "resume_from": "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
    "freq_bias_path": "checkpoints/demo_best/frequency_prior.json",
}


def make_row(config_overrides: dict | None = None, row_overrides: dict | None = None) -> dict:
    config = dict(GOOD_CONFIG)
    config.update(config_overrides or {})
    row = {
        "config": config,
        "experiment": {
            "num_predicates": 50,
            "predicate_vocab_hash": "0123456789abcdef",
            "git_commit": "abcdef1",
            "train_config": config,
        },
        "n_images_evaluated": 24.0,
        "num_gt": 271.0,
        "R@50": 0.61,
        "mR@50": 0.20,
        "predicate_diag": {"score_mode": "ensemble", "ensemble_alpha": 0.0},
        "pair_proposal": {"gt_pair_recall@32": 1.0, "avg_candidate_pairs_per_image": 11.3},
    }
    row.update(row_overrides or {})
    return row


def write_metrics(tmp_path: Path, row: dict, name: str = "metrics.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def run_verifier(metrics_path: Path, *extra: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(VERIFIER), str(metrics_path), *extra],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def test_verifier_exists():
    assert VERIFIER.exists()


def test_correct_protocol_passes(tmp_path):
    code, out = run_verifier(write_metrics(tmp_path, make_row()))
    assert code == 0, out
    assert "CANARY PASS" in out


def test_pass_does_not_claim_the_result_is_good(tmp_path):
    """A protocol check must never be mistaken for a quality judgement."""
    _code, out = run_verifier(write_metrics(tmp_path, make_row()))
    assert "not that the result is good" in out


def test_metrics_are_reported_but_not_asserted(tmp_path):
    """An implausibly bad metric must still PASS if the protocol is right."""
    row = make_row(row_overrides={"R@50": 0.0001, "mR@50": 0.0001})
    code, out = run_verifier(write_metrics(tmp_path, row))
    assert code == 0, "the verifier must not judge metric values"
    assert "NOT asserted" in out


@pytest.mark.parametrize(
    "overrides,expected_fragment",
    [
        ({"eval_sgg_predicate_ensemble_alpha": 0.45}, "eval_sgg_predicate_ensemble_alpha == 0.0"),
        ({"eval_sgg_predicate_score_mode": "classifier"}, "eval_sgg_predicate_score_mode == 'ensemble'"),
        ({"eval_sgg_use_gt_pairs": False}, "eval_sgg_use_gt_pairs == True"),
        ({"explicit_spoa_enabled": True}, "explicit_spoa_enabled == False"),
        ({"text_conditioned_projection_enabled": True}, "text_conditioned_projection_enabled == False"),
        ({"relationness_enabled": True}, "relationness_enabled == False"),
        ({"eval_sgg_use_relationness": True}, "eval_sgg_use_relationness == False"),
        ({"adaptive_calibration_enabled": False}, "adaptive_calibration_enabled == True"),
        ({"bayes_calibration_weight": 0.5}, "bayes_calibration_weight == 0.0"),
        ({"freq_bias_enabled": False}, "freq_bias_enabled == True"),
        ({"freq_bias_alpha": 1.0}, "freq_bias_alpha == 3.75"),
        ({"clip_input_res": 448}, "clip_input_res == 336"),
        ({"eval_sgg_grounding_dino_enabled": True}, "eval_sgg_grounding_dino_enabled == False"),
    ],
)
def test_each_protocol_drift_is_caught(tmp_path, overrides, expected_fragment):
    """Every frozen setting must be individually enforced."""
    code, out = run_verifier(write_metrics(tmp_path, make_row(overrides)))
    assert code == 1, f"drift in {list(overrides)} was not caught:\n{out}"
    assert expected_fragment in out


def test_ensemble_alpha_failure_explains_the_untrained_classifier(tmp_path):
    """The most consequential drift must say WHY it matters, not just that it differs."""
    _code, out = run_verifier(write_metrics(tmp_path, make_row({"eval_sgg_predicate_ensemble_alpha": 0.45})))
    assert "UNTRAINED classifier" in out


def test_missing_frequency_prior_file_is_flagged(tmp_path):
    """The silent-uncalibrated scenario: configured path does not exist."""
    row = make_row({"freq_bias_path": "checkpoints/does_not_exist/frequency_prior.json"})
    code, out = run_verifier(write_metrics(tmp_path, row))
    assert code == 1
    assert "frequency prior file exists" in out
    assert "SILENTLY" in out


def test_wrong_predicate_vocabulary_size_is_flagged(tmp_path):
    row = make_row()
    row["experiment"]["num_predicates"] = 51
    code, out = run_verifier(write_metrics(tmp_path, row))
    assert code == 1
    assert "predicate vocabulary size == 50" in out


def test_wrong_checkpoint_is_flagged(tmp_path):
    row = make_row({"resume_from": "checkpoints/some_other_run.pt"})
    code, out = run_verifier(write_metrics(tmp_path, row))
    assert code == 1
    assert "resumed from the historical checkpoint" in out


def test_zero_ground_truth_is_flagged(tmp_path):
    code, out = run_verifier(write_metrics(tmp_path, make_row(row_overrides={"num_gt": 0.0})))
    assert code == 1
    assert "ground-truth triplets were found" in out


def test_zero_images_is_flagged(tmp_path):
    code, out = run_verifier(write_metrics(tmp_path, make_row(row_overrides={"n_images_evaluated": 0.0})))
    assert code == 1
    assert "evaluated at least one image" in out


def test_nan_metric_is_flagged(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(make_row()).replace('"R@50": 0.61', '"R@50": NaN') + "\n", encoding="utf-8"
    )
    code, out = run_verifier(path)
    assert code == 1
    assert "R@50 is not NaN" in out


def test_predicate_diag_contradicting_config_is_caught(tmp_path):
    """An independent second source of truth inside the same metrics row."""
    row = make_row()
    row["predicate_diag"] = {"score_mode": "classifier", "ensemble_alpha": 0.45}
    code, out = run_verifier(write_metrics(tmp_path, row))
    assert code == 1
    assert "predicate_diag confirms score_mode=ensemble" in out


def test_missing_metrics_file_exits_two(tmp_path):
    code, out = run_verifier(tmp_path / "never_written.jsonl")
    assert code == 2
    assert "Nothing to verify" in out


def test_row_without_config_block_exits_two(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"R@50": 0.5}) + "\n", encoding="utf-8")
    code, out = run_verifier(path)
    assert code == 2
    assert "no 'config' block" in out


def test_last_row_wins_for_multi_epoch_files(tmp_path):
    """metrics.jsonl is append-per-epoch; the final row is the run's outcome."""
    path = tmp_path / "metrics.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(make_row({"freq_bias_enabled": False})) + "\n")
        f.write(json.dumps(make_row()) + "\n")
    code, _out = run_verifier(path)
    assert code == 0


def test_verdict_file_starts_with_the_status_word(tmp_path):
    """eval_historical_checkpoint.sh greps for a leading PASS to gate the full run."""
    metrics = write_metrics(tmp_path, make_row())
    verdict = tmp_path / "canary_verdict.txt"
    code, _out = run_verifier(metrics, "--out", str(verdict))
    assert code == 0
    assert verdict.exists()
    assert verdict.read_text(encoding="utf-8").splitlines()[0] == "PASS"


def test_failing_verdict_file_starts_with_fail(tmp_path):
    metrics = write_metrics(tmp_path, make_row({"freq_bias_enabled": False}))
    verdict = tmp_path / "canary_verdict.txt"
    code, _out = run_verifier(metrics, "--out", str(verdict))
    assert code == 1
    assert verdict.read_text(encoding="utf-8").splitlines()[0] == "FAIL"
