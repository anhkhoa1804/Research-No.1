"""Regression tests for the eval batch cap.

A non-positive `max_batches` means "evaluate the whole loader". This is not a
cosmetic detail: `scripts/eval/eval_historical_checkpoint.sh --full` passes
EVAL_BATCHES=0 for the entire split, and before `_batch_limit_reached` existed
that broke on the first batch. The run evaluated ZERO images, exited 0, and
reported all-zero R@K/mR@K -- a silent failure that looks like a real result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openvocab_rel.evals import _batch_limit_reached  # noqa: E402


@pytest.mark.parametrize("cap", [0, -1, -100])
def test_non_positive_cap_never_limits(cap):
    """0 must mean 'the whole split', not 'nothing'."""
    for bi in (0, 1, 5, 1000, 100000):
        assert _batch_limit_reached(bi, cap) is False, (
            f"cap={cap} stopped the loop at batch {bi}; a non-positive cap means no limit"
        )


def test_positive_cap_behaves_exactly_as_before():
    """Any positive cap must be unchanged by the fix."""
    for cap in (1, 2, 50, 867):
        for bi in range(cap):
            assert _batch_limit_reached(bi, cap) is False
        assert _batch_limit_reached(cap, cap) is True
        assert _batch_limit_reached(cap + 1, cap) is True


def test_cap_of_two_evaluates_exactly_two_batches():
    """The canary's 2-batch cap must yield 2 batches, not 1 or 3."""
    evaluated = [bi for bi in range(10) if not _batch_limit_reached(bi, 2)]
    assert evaluated == [0, 1]


def test_cap_of_zero_evaluates_every_batch():
    evaluated = [bi for bi in range(10) if not _batch_limit_reached(bi, 0)]
    assert evaluated == list(range(10))


def test_unparseable_cap_does_not_limit():
    """A malformed cap must not silently truncate the evaluation."""
    for cap in (None, "", "abc", object()):
        assert _batch_limit_reached(0, cap) is False


def test_every_eval_loop_uses_the_helper():
    """No loop may compare against max_batches directly and reintroduce the bug."""
    source = (REPO_ROOT / "openvocab_rel" / "evals.py").read_text(encoding="utf-8")
    assert "if bi >= int(max_batches):" not in source, (
        "an eval loop compares bi against max_batches directly; use "
        "_batch_limit_reached so a cap of 0 means the whole split"
    )
    assert source.count("_batch_limit_reached(bi, max_batches)") >= 12
