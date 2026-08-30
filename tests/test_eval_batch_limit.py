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


def test_eval_sgg_standard_keeps_its_no_grad_decorator():
    """The helper must not be inserted between @torch.no_grad() and the function.

    Inserting a top-level def directly after the decorator silently transfers
    the decorator to the new function and strips eval_sgg_standard of
    no_grad -- evaluation would then build a graph.
    """
    source = (REPO_ROOT / "openvocab_rel" / "evals.py").read_text(encoding="utf-8")
    idx = source.index("def eval_sgg_standard(")
    preceding = source[:idx].rstrip().splitlines()[-1].strip()
    assert preceding == "@torch.no_grad()", (
        f"eval_sgg_standard is preceded by {preceding!r}, not @torch.no_grad()"
    )


def test_helper_is_not_decorated():
    source = (REPO_ROOT / "openvocab_rel" / "evals.py").read_text(encoding="utf-8")
    idx = source.index("def _batch_limit_reached(")
    preceding = source[:idx].rstrip().splitlines()[-1].strip()
    assert not preceding.startswith("@"), (
        f"_batch_limit_reached picked up the decorator {preceding!r}"
    )


def test_evals_py_keeps_its_crlf_line_endings():
    """evals.py is CRLF in this repository; a whole-file rewrite must not flip it.

    A tool that reads the file as text and writes it back converts all 3,691
    line endings, producing a diff that buries a real change in noise and
    changes the file's hash for everyone.
    """
    raw = (REPO_ROOT / "openvocab_rel" / "evals.py").read_bytes()
    crlf = raw.count(b"\r\n")
    lone_lf = raw.count(b"\n") - crlf
    assert crlf > 3000, f"expected CRLF line endings, found only {crlf}"
    assert lone_lf == 0, f"{lone_lf} lines use bare LF; the file is mixed"
