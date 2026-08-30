"""Tests for tools/make_recalibrated_prior.py.

The recalibrated prior is consumed by the unmodified evaluator, so it must
remain a structurally valid frequency prior: same vocabulary, same keys, rows
that are still log-probabilities. tau=0 must be decision-identical to the
source, or the arm is not a controlled comparison.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.make_recalibrated_prior import recalibrate  # noqa: E402


def make_prior() -> dict:
    vocab = [f"p{i}" for i in range(5)]
    # A deliberately skewed marginal, like the real one.
    marginal = [math.log(x) for x in (0.60, 0.20, 0.10, 0.07, 0.03)]
    return {
        "predicate_vocab": vocab,
        "global_log_probs": marginal,
        "pair_log_probs": {"a||b": [math.log(x) for x in (0.5, 0.3, 0.1, 0.07, 0.03)]},
        "subject_log_probs": {"a": [math.log(x) for x in (0.4, 0.4, 0.1, 0.05, 0.05)]},
        "object_log_probs": {"b": [math.log(x) for x in (0.7, 0.1, 0.1, 0.05, 0.05)]},
        "smoothing": 1.0,
        "default_log_prob": -3.912023005428146,
    }


def test_rows_remain_normalised_log_probs():
    out = recalibrate(make_prior(), tau=0.5)
    for key in ("pair_log_probs", "subject_log_probs", "object_log_probs"):
        for row in out[key].values():
            total = sum(math.exp(v) for v in row)
            assert total == pytest.approx(1.0, abs=1e-9), f"{key} row sums to {total}"
    assert sum(math.exp(v) for v in out["global_log_probs"]) == pytest.approx(1.0, abs=1e-9)


def test_tau_zero_preserves_every_argmax():
    src = make_prior()
    out = recalibrate(src, tau=0.0)
    for key in ("pair_log_probs", "subject_log_probs", "object_log_probs"):
        for k, row in out[key].items():
            assert row.index(max(row)) == src[key][k].index(max(src[key][k]))


def test_tau_zero_is_a_pure_renormalisation():
    """tau=0 may shift a row by a constant but must not change its shape."""
    src = make_prior()
    out = recalibrate(src, tau=0.0)
    row_in = src["pair_log_probs"]["a||b"]
    row_out = out["pair_log_probs"]["a||b"]
    diffs = [b - a for a, b in zip(row_in, row_out)]
    assert max(diffs) - min(diffs) == pytest.approx(0.0, abs=1e-9)


def test_positive_tau_shifts_mass_toward_rare_predicates():
    src = make_prior()
    out = recalibrate(src, tau=0.5)
    row_in, row_out = src["pair_log_probs"]["a||b"], out["pair_log_probs"]["a||b"]
    # p0 is the most common predicate in the marginal, p4 the rarest.
    assert row_out[0] - row_in[0] < row_out[4] - row_in[4]


def test_structure_and_vocabulary_are_preserved():
    src = make_prior()
    out = recalibrate(src, tau=0.25)
    assert out["predicate_vocab"] == src["predicate_vocab"]
    assert out["smoothing"] == src["smoothing"]
    assert out["default_log_prob"] == src["default_log_prob"]
    assert set(out["pair_log_probs"]) == set(src["pair_log_probs"])
    for key in ("pair_log_probs", "subject_log_probs", "object_log_probs"):
        for row in out[key].values():
            assert len(row) == len(src["predicate_vocab"])


def test_provenance_records_the_tau():
    out = recalibrate(make_prior(), tau=0.1)
    assert out["provenance"]["recalibrated"] is True
    assert out["provenance"]["recalibration_tau"] == 0.1


def test_source_is_not_mutated():
    """The historical prior is immutable evidence; recalibrate must not touch it."""
    src = make_prior()
    before = src["pair_log_probs"]["a||b"][:]
    recalibrate(src, tau=0.5)
    assert src["pair_log_probs"]["a||b"] == before
    assert src["global_log_probs"] == make_prior()["global_log_probs"]
