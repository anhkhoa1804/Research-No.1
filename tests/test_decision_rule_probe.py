"""Tests for tools/decision_rule_probe.py.

The probe's tau=0 arm must be numerically identical to the frequency-prior
baseline, or "gains are attributable to the adjustment alone" is false. The
verdict function must implement the criteria in
docs/DECISION_RULE_HYPOTHESIS.md exactly as written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.decision_rule_probe import (  # noqa: E402
    MAX_R_LOSS_SUPPORT,
    MIN_MR_GAIN_SUPPORT,
    verdict,
)


def row(tau: float, r50: float, mr50: float) -> dict:
    return {"tau": tau, "R@50": r50 / 100.0, "mR@50": mr50 / 100.0}


def test_tau_zero_with_no_alternatives_is_rejected():
    v, _why = verdict([row(0.0, 66.59, 22.30)])
    assert v == "H2 REJECTED"


def test_affordable_tau_is_found_even_when_a_larger_tau_dominates_mR():
    """The registered criterion is 'at some tau within the R@50 budget'.

    Taking the mR-argmax first picks the most aggressive tau on the sweep,
    which necessarily blows the R@50 budget, and wrongly reports PARETO-ONLY.
    These are the real measured numbers from the full validation split.
    """
    rows = [
        row(0.0, 66.59, 22.30),
        row(0.10, 66.04, 26.42),   # +4.11 mR for -0.56 R  <- satisfies both halves
        row(0.25, 59.27, 30.59),
        row(0.50, 44.48, 37.52),
        row(0.75, 26.81, 38.38),   # highest mR, but -39.79 R
        row(1.00, 12.03, 35.59),
    ]
    v, why = verdict(rows)
    assert v == "H2 SUPPORTED", why
    assert "tau=0.1" in why


def test_pareto_only_when_no_tau_fits_the_budget():
    rows = [
        row(0.0, 66.59, 22.30),
        row(0.5, 44.00, 37.00),    # big mR gain, far outside the R budget
    ]
    v, why = verdict(rows)
    assert v == "PARETO-ONLY", why


def test_rejected_when_no_tau_moves_mR_materially():
    rows = [row(0.0, 66.59, 22.30), row(0.1, 66.50, 22.60)]
    v, _why = verdict(rows)
    assert v == "H2 REJECTED"


def test_r50_budget_boundary_is_respected():
    """A tau exactly at the budget edge counts as affordable."""
    rows = [
        row(0.0, 66.59, 22.30),
        row(0.1, 66.59 - MAX_R_LOSS_SUPPORT, 22.30 + MIN_MR_GAIN_SUPPORT),
    ]
    v, _why = verdict(rows)
    assert v == "H2 SUPPORTED"


def test_a_tau_just_outside_the_budget_does_not_qualify():
    rows = [
        row(0.0, 66.59, 22.30),
        row(0.1, 66.59 - MAX_R_LOSS_SUPPORT - 0.01, 22.30 + MIN_MR_GAIN_SUPPORT),
    ]
    v, _why = verdict(rows)
    assert v != "H2 SUPPORTED"


@pytest.mark.skipif(
    not (REPO_ROOT / "datasets_vg150_clean" / "frequency_prior_train.json").exists(),
    reason="train-derived prior not present (gitignored, built by tools/build_vg150_frequency_prior.py)",
)
def test_tau_zero_matches_the_frequency_prior_baseline_exactly():
    """tau=0 must BE the baseline, not merely resemble it."""
    from tools.frequency_prior_baseline import FrequencyPrior, load_split, evaluate
    from tools.decision_rule_probe import evaluate_tau

    prior = FrequencyPrior.load(REPO_ROOT / "datasets_vg150_clean" / "frequency_prior_train.json")
    data = load_split(REPO_ROOT / "datasets_vg150_clean" / "validation.jsonl", prior, limit=120)

    baseline = evaluate(data, prior)
    adjusted = evaluate_tau(data, prior, tau=0.0)
    for key in ("R@20", "R@50", "mR@20", "mR@50"):
        assert adjusted[key] == pytest.approx(baseline[key], abs=1e-12), (
            f"{key}: tau=0 gave {adjusted[key]} but the baseline gives {baseline[key]}"
        )
