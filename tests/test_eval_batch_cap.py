"""`cap_batches` -- tightening a cap without destroying the unlimited sentinel.

`_batch_limit_reached` reads a non-positive cap as NO LIMIT, which is what makes
`--eval_batches 0` mean "the whole split". Any arithmetic on that value has to
respect the sentinel, and `min()` does not: min(25, 0) is 0, which flips a
25-batch diagnostic into an unbounded one. On the full 10,401-image validation
split that is a second full pass, silently, and it shows up only as the run
taking hours longer than budgeted.
"""
from __future__ import annotations

import pytest

from openvocab_rel.evals import _batch_limit_reached, cap_batches


def test_unlimited_sentinel_yields_the_cap_not_unlimited():
    """The regression. min(25, 0) == 0 == unlimited; cap_batches gives 25."""
    for unlimited in (0, -1, -100):
        assert cap_batches(unlimited, 25) == 25
        assert min(25, unlimited) != 25          # documents what was wrong


def test_a_tighter_explicit_limit_wins():
    assert cap_batches(10, 25) == 10
    assert cap_batches(1, 25) == 1


def test_a_looser_explicit_limit_is_clamped_to_the_cap():
    assert cap_batches(250, 25) == 25
    assert cap_batches(10401, 25) == 25


def test_the_result_is_always_a_real_limit_the_loop_will_honour():
    """Whatever comes out must make the loop stop -- never unlimited."""
    for limit in (-5, 0, 1, 25, 250):
        out = cap_batches(limit, 25)
        assert out > 0
        assert not _batch_limit_reached(out - 1, out)
        assert _batch_limit_reached(out, out)


def test_p10_configuration_is_unchanged():
    """eval_batches=250 must still give the historical 25, so the fix cannot
    silently alter any run that has already been reported."""
    assert cap_batches(250, 25) == 25


@pytest.mark.parametrize("bad", ["", None, "abc"])
def test_non_integer_limits_raise_rather_than_silently_becoming_unlimited(bad):
    with pytest.raises((TypeError, ValueError)):
        cap_batches(bad, 25)
