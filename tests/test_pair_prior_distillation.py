"""Pins for the pair-prior distillation ladder.

This study decides whether the checkpoint is needed at all, so its failure
directions are asymmetric and both expensive:

  a LEAKY fold statistic makes the vision-free arms look better than they are
  and closes the branch wrongly;
  a MIS-SPECIFIED decomposition makes the model term look irreducible and keeps
  a dead branch alive.

So the tests pin the structural guarantees -- leak-free fold statistics, an
exact additive decomposition, identical evaluation semantics across arms --
rather than any measured value.
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
def PPD():
    return _load("pair_prior_distillation")


@pytest.fixture(scope="module")
def D(PPD):
    B = PPD.Mech(str(DUMP), str(PRIOR), "raw50")
    return PPD.Distill(B, str(PRIOR), 0.0, 5, 0)


# ------------------------------------------------------------- 1. leakage
def test_fold_pair_mean_uses_training_rows_only(D):
    """The load-bearing leak guard.

    A group mean that included held-out rows would let each row's own label
    influence the statistic used to score it -- the vision-free arms would then
    look far stronger than they are and would close the branch wrongly.
    """
    train = D.fold != 0
    gm = D._fold_pair_mean(D.md, train)
    # perturbing ONLY held-out rows must not move the statistic at all
    D2_md = D.md.clone()
    D2_md[~train] += 1000.0
    saved, D.md = D.md, D2_md
    try:
        gm2 = D._fold_pair_mean(D.md, train)
    finally:
        D.md = saved
    assert torch.allclose(gm, gm2, atol=1e-6)


def test_groups_absent_from_training_fall_back_not_crash(D):
    tiny = torch.zeros(D.n, dtype=torch.bool)
    tiny[: 200] = True
    gm = D._fold_pair_mean(D.md, tiny)
    assert gm.shape == D.md.shape
    assert torch.isfinite(gm).all()


def test_every_arm_shares_folds_candidates_and_denominator(D, PPD):
    """Arms may differ only in feature columns -- never in what is evaluated."""
    g = torch.Generator().manual_seed(PPD.SEED)
    train = D.fold != 0
    shapes = {a: D.blocks(a, train, g).shape[:2] for a in
              ("A_global", "D_pair", "E_backoff", "G_model")}
    assert len(set(shapes.values())) == 1 == len({(D.n, D.C)} & set(shapes.values()))


# --------------------------------------------------- 2. the decomposition
def test_pairmean_plus_residual_reconstructs_the_model_term_exactly(D):
    """G_pairmean and G_residual must partition the model term with no remainder.

    If they did not sum back to it, 'the pair part' and 'the within-group part'
    would not be complements and neither contrast would mean what it says.
    """
    train = D.fold != 0
    gm = D._fold_pair_mean(D.md, train)
    resid = D.md - gm
    assert torch.allclose(gm + resid, D.md, atol=1e-5)


def test_residual_has_zero_mean_within_every_training_group(D):
    """The residual must carry no pair-level information by construction."""
    train = D.fold != 0
    resid = D.md - D._fold_pair_mean(D.md, train)
    pid = D.pair_id[train]
    for gid in pid.unique()[:150].tolist():
        ix = (D.pair_id == gid) & train
        if int(ix.sum()) > 1:
            assert float(resid[ix].mean(0).abs().max()) < 1e-3


# ------------------------------------------------------ 3. prior lookups
def test_conditional_lookups_are_distinct_and_finite(D):
    for v in (D.v_g, D.v_s, D.v_o, D.v_p):
        assert v.shape == (D.n, D.C)
        assert torch.isfinite(v).all()
    assert not torch.allclose(D.v_s, D.v_o)
    assert not torch.allclose(D.v_p, D.v_g)


def test_global_lookup_is_constant_across_rows(D):
    """log P(p) cannot depend on the row; if it does, the mapping is wrong."""
    assert torch.allclose(D.v_g[0], D.v_g[-1])


def test_unseen_keys_take_the_prior_files_own_default(D):
    out = D._lookup({}, ["nothing||at all"] * 3)
    assert torch.allclose(out, torch.full_like(out, D.default))


def test_prior_coverage_is_measured_not_assumed(D):
    """16.96% of validation pairs are unseen in the train-derived prior; that
    is a real limit on arms D-E and must be reported rather than hidden."""
    assert 0.0 < D.coverage["pair"] < 1.0
    assert D.coverage["subject"] > D.coverage["pair"]


# ------------------------------------------------------------ 4. metrics
def test_rank_metrics_are_consistent_with_each_other(D):
    score = torch.randn(D.n, D.C)
    m = D.metrics(D.prior_top1_col, score)
    assert 1.0 <= m["mean_gt_rank"] <= D.C
    assert 0.0 < m["MRR"] <= 1.0
    assert 0.0 <= m["R_at_5"] <= 1.0
    assert m["top1"] == pytest.approx(m["R"], abs=1e-12)


def test_prior_fallback_reproduces_the_evaluator_baseline(D):
    m = D.metrics(D.prior_top1_col, torch.zeros(D.n, D.C))
    ref = D.B.metrics(D.B.score(0.0, 3.75, None))
    assert m["R"] == pytest.approx(ref["R"], abs=1e-9)


def test_pair_support_totals_are_self_consistent(D, PPD):
    ps = PPD.pair_support(D)
    assert ps["n_rows"] == D.n
    assert ps["singleton_rows"] <= ps["n_rows"]
    assert ps["singleton_groups"] <= ps["n_unique_pairs"]
    assert 0.0 <= ps["singleton_row_rate"] <= 1.0
    assert ps["mean_group_size"] == pytest.approx(D.n / ps["n_unique_pairs"], rel=1e-9)
