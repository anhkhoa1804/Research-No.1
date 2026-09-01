"""Pins for the learned candidate-restricted scorer.

The failure directions that matter here are asymmetric and both expensive:

  an OPTIMISTIC probe manufactures headroom out of the fitting procedure and
  buys GPU time for a reranker that cannot exist;
  a PESSIMISTIC probe closes the last live branch of the project.

So the tests pin the structural guarantees the pre-registration relies on --
leak-free folds, the prior always being reachable, coverage agreeing with the
independently-written oracle tool -- rather than pinning any measured value.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "runs/p10_model_recalibration/pair_logits.pt"
PRIOR = ROOT / "datasets_vg150_clean/frequency_prior_train.json"
pytestmark = pytest.mark.skipif(not (DUMP.exists() and PRIOR.exists()),
                                reason="C' cache or train-derived prior absent")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    csp = _load("candidate_scorer_probe")
    oc = _load("cprime_oracle_ceiling")
    return csp, oc


@pytest.fixture(scope="module")
def B(mods):
    csp, _ = mods
    return csp.Mech(str(DUMP), str(PRIOR), "raw50")


@pytest.fixture(scope="module")
def P(mods, B):
    csp, _ = mods
    return csp.CandidateProbe(B, 0.0, 5)


# --------------------------------------------------------------- 1. leakage
def test_folds_split_by_image_never_by_row(P, B):
    """Every GT row of one image must land in exactly one fold."""
    per_img = {}
    for row in range(P.n):
        per_img.setdefault(int(B.gt_img[row]), set()).add(int(P.fold[row]))
    assert all(len(v) == 1 for v in per_img.values())


def test_fold_assignment_is_independent_of_seed_and_order(mods):
    csp, _ = mods
    a = [csp.fold_of_image(x) for x in ("12345", "6789", "1")]
    torch.manual_seed(999)
    b = [csp.fold_of_image(x) for x in ("12345", "6789", "1")]
    assert a == b
    assert len(set(csp.fold_of_image(str(i)) for i in range(500))) == csp.N_FOLDS


def test_every_fold_is_populated(P):
    assert int(torch.bincount(P.fold, minlength=5).min()) > 0


# ------------------------------------------- 2. the prior must be reachable
def test_prior_top1_is_always_candidate_zero(P):
    """This is what makes the probe a fair test: weights reading off the prior
    logit alone reproduce the baseline, so the arm can only lose by estimation
    error, never by being denied the prior's answer."""
    assert torch.equal(P.cand[:, 0], P.prior_top1_col)


def test_candidate_set_has_no_duplicates(P):
    for j in range(P.cand.shape[1]):
        for i in range(j + 1, P.cand.shape[1]):
            assert not bool((P.cand[:, j] == P.cand[:, i]).any())


def test_falling_back_to_candidate_zero_reproduces_the_prior_baseline(P, B):
    m = P._metrics(P.cand[:, 0], "fallback")
    ref = B.metrics(B.score(0.0, P.B and 3.75, None))
    assert m["R"] == pytest.approx(ref["R"], abs=1e-9)


# ------------------------------------ 3. agreement with the oracle tool
def test_coverage_agrees_with_the_independent_oracle_tool(mods, B):
    """Two tools written separately must agree on P(GT in prior top-k).

    They disagreed at first: overwriting candidate 0 with the argmax duplicated
    a column on the 522 tied rows and shrank top-3 coverage from 85.5% to 85.2%.
    """
    csp, oc = mods
    O = oc.Oracle(B, 0.0)
    for k in (2, 3, 5):
        p = csp.CandidateProbe(B, 0.0, k)
        assert float(p.gt_in.float().mean()) == pytest.approx(
            O.coverage(k, float("inf"))["coverage_all_rows"], abs=1e-9)


def test_gt_position_is_consistent_with_membership(P):
    assert bool((P.gt_pos[~P.gt_in] == -1).all())
    inn = P.gt_in.nonzero().squeeze(1)
    got = P.cand[inn].gather(1, P.gt_pos[inn].unsqueeze(1)).squeeze(1)
    assert torch.equal(got, P.gt_col[inn])


# --------------------------------------------------------------- 4. the null
def test_shuffled_null_keeps_the_feature_marginal_but_breaks_the_row_link(P):
    g0 = torch.Generator().manual_seed(0)
    real = P._blocks("full", torch.Generator().manual_seed(0))
    null = P._blocks("shuffled_model", g0)
    assert real.shape == null.shape
    # The prior block is untouched: the arms differ ONLY in the model features,
    # so any gap between them is attributable to the model term and to nothing
    # else about the fitting procedure.
    assert torch.allclose(real[..., :-3], null[..., :-3])
    assert not torch.allclose(real[..., -3:], null[..., -3:])
    # The null draws each row's model features from a DIFFERENT row, read at
    # this row's candidate columns. That is deliberate: the candidate structure
    # has to survive or the arms would not be comparable, which also means the
    # null's value multiset is not a permutation of the real one. What must hold
    # is that every null value is a real entry of the model term somewhere.
    assert set(null[..., -3].flatten().tolist()) <= set(P.md.flatten().tolist())
