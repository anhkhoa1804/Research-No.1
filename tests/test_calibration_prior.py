"""Regression tests for the frequency-prior eval-split-fallback fix.

Prior behavior (the bug): _predicate_log_prior_for_eval tried
<vg150_root>/train.jsonl first, but on ANY failure (file missing, one
malformed line anywhere in the file, or a legitimately empty count) it
silently fell back to counting predicates from the `loader` argument --
i.e. potentially the validation/test split currently being scored -- with
no warning. If a run had calibration enabled (adaptive_calibration_enabled,
or logit_adj_tau/eval_logit_adj_tau > 0) and pointed --vg150_root at a
directory lacking train.jsonl, the resulting prior would be built from the
same labels being evaluated and then used to shift that evaluation's own
logits before ranking.

Fixed behavior: the eval-loader fallback is removed entirely. Missing/
unreadable/empty train-split statistics now raise MissingTrainStatisticsError
instead of silently substituting anything. The prior is only computed at
all when something will actually consume it (_calibration_prior_is_needed),
so evaluation runs that genuinely don't want calibration -- a config
choice -- never require train.jsonl to exist in the first place.

These tests use only tiny synthetic JSONL fixtures; no real VG150 data or
network access required. `openvocab_rel.evals` imports `transformers` at
module level (via clip_utils), so these are skipped if it isn't installed,
matching the pattern used elsewhere in this test suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from openvocab_rel.evals import (  # noqa: E402
    MissingTrainStatisticsError,
    _calibration_prior_is_needed,
    _predicate_log_prior_for_eval,
)
from openvocab_rel.config import TrainConfig  # noqa: E402


def _write_jsonl(path: Path, predicate_counts: dict) -> None:
    """Write one synthetic row per predicate occurrence requested."""
    rows = []
    obj_id = 0
    for pred, count in predicate_counts.items():
        for _ in range(count):
            rows.append(
                {
                    "image_id": f"img_{obj_id}",
                    "obj_boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
                    "objects": ["thing_a", "thing_b"],
                    "relationships": [{"subject_id": 0, "object_id": 1, "predicate": pred}],
                }
            )
            obj_id += 1
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _fake_loader_with_rows(rows: list) -> SimpleNamespace:
    """Mimics the shape of a DataLoader whose .dataset.rows the OLD buggy
    fallback would have read from -- used to prove those rows are now
    never consulted."""
    return SimpleNamespace(dataset=SimpleNamespace(rows=rows))


# ---- Test A: training statistics available -> training statistics are used ----


def test_A_uses_train_jsonl_statistics_when_available(tmp_path):
    train_path = tmp_path / "train.jsonl"
    # "on" is common, "riding" is rare -- the prior should reflect that.
    _write_jsonl(train_path, {"on": 20, "riding": 1})
    cfg = SimpleNamespace(vg150_root=str(tmp_path))

    prior = _predicate_log_prior_for_eval(cfg, loader=None, pred_vocab=["on", "riding"], device="cpu")

    assert prior.shape == (2,)
    # log-prior of the frequent predicate must be strictly greater (less negative)
    assert float(prior[0]) > float(prior[1])


# ---- Test B: training statistics missing -> evaluation-split statistics NOT used ----


def test_B_missing_train_jsonl_raises_instead_of_falling_back(tmp_path):
    # No train.jsonl written at all under tmp_path.
    eval_only_rows = [
        {
            "relationships": [
                {"subject_id": 0, "object_id": 1, "predicate": "riding"},
            ]
        }
        for _ in range(50)
    ]
    loader = _fake_loader_with_rows(eval_only_rows)
    cfg = SimpleNamespace(vg150_root=str(tmp_path))

    with pytest.raises(MissingTrainStatisticsError):
        _predicate_log_prior_for_eval(cfg, loader=loader, pred_vocab=["on", "riding"], device="cpu")


def test_B_malformed_train_jsonl_raises_instead_of_falling_back(tmp_path):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{not valid json\n", encoding="utf-8")
    loader = _fake_loader_with_rows(
        [{"relationships": [{"subject_id": 0, "object_id": 1, "predicate": "riding"}]}]
    )
    cfg = SimpleNamespace(vg150_root=str(tmp_path))

    with pytest.raises(MissingTrainStatisticsError):
        _predicate_log_prior_for_eval(cfg, loader=loader, pred_vocab=["on", "riding"], device="cpu")


# ---- Test C: evaluation labels cannot influence the constructed prior ----


def test_C_eval_split_predicate_distribution_never_leaks_into_the_prior(tmp_path):
    train_path = tmp_path / "train.jsonl"
    # Train says "on" is common, "riding" is rare.
    _write_jsonl(train_path, {"on": 20, "riding": 1})

    # Eval loader says the exact opposite -- if the old fallback (or any
    # blending) fired, the resulting prior would be pulled toward "riding".
    eval_rows_favoring_riding = [
        {"relationships": [{"subject_id": 0, "object_id": 1, "predicate": "riding"}]} for _ in range(200)
    ] + [{"relationships": [{"subject_id": 0, "object_id": 1, "predicate": "on"}]}]
    loader = _fake_loader_with_rows(eval_rows_favoring_riding)
    cfg = SimpleNamespace(vg150_root=str(tmp_path))

    prior = _predicate_log_prior_for_eval(cfg, loader=loader, pred_vocab=["on", "riding"], device="cpu")

    # The prior must reflect TRAIN (on >> riding), not eval (riding >> on).
    assert float(prior[0]) > float(prior[1])


# ---- Test D: normal documented behavior is unchanged when valid train stats exist ----


def test_D_calibration_off_never_requires_train_jsonl(tmp_path):
    # No train.jsonl anywhere under tmp_path -- this must NOT raise, because
    # nothing is configured to consume the prior. This is the "config choice"
    # the fix is required to respect.
    cfg = TrainConfig()
    cfg.adaptive_calibration_enabled = False
    cfg.logit_adj_tau = 0.0
    cfg.eval_logit_adj_tau = -1.0
    assert _calibration_prior_is_needed(cfg) is False


def test_D_calibration_on_still_requires_and_uses_train_jsonl(tmp_path):
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, {"on": 10, "riding": 5})
    cfg = TrainConfig()
    cfg.adaptive_calibration_enabled = True
    cfg.vg150_root = str(tmp_path)

    assert _calibration_prior_is_needed(cfg) is True
    prior = _predicate_log_prior_for_eval(cfg, loader=None, pred_vocab=["on", "riding"], device="cpu")
    assert prior.shape == (2,)
    assert float(prior[0]) > float(prior[1])


def test_D_logit_adj_tau_alone_also_requires_the_prior():
    cfg = TrainConfig()
    cfg.adaptive_calibration_enabled = False
    cfg.logit_adj_tau = 0.5
    cfg.eval_logit_adj_tau = -1.0  # reuse logit_adj_tau, per the documented default
    assert _calibration_prior_is_needed(cfg) is True
