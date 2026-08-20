"""Invariants for the prior-residual objective (architecture candidate A1).

The phase brief says explicitly: do NOT assume A1 automatically produces a
pure residual predictor. These tests check the property numerically rather
than trusting the derivation.

The defining claim is:

    with z = alpha * log P(p|s,o) + f_theta(x) and ordinary cross-entropy,
    the gradient reaching f_theta is proportional to what the PRIOR gets
    wrong -- near zero where the prior is already confidently correct, and
    full magnitude where it is confidently wrong.

If that does not hold numerically, A1 is not a residual objective and the
whole direction is mis-specified. test_gradient_vanishes_when_prior_is_right
and test_gradient_is_large_when_prior_is_wrong are the two that matter.

No GPU, no data, no CLIP, no model.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from openvocab_rel.config import TrainConfig
from openvocab_rel.prior_residual import (
    PairPriorTable,
    apply_visual_ablation,
    compose_prior_residual,
    residual_diagnostics,
)

P = 6  # predicates in the toy vocabulary


def _log_prior(peak_index: int, sharpness: float = 6.0, n: int = P) -> torch.Tensor:
    """A log-probability row peaked on `peak_index`."""
    logits = torch.full((n,), -sharpness)
    logits[peak_index] = 0.0
    return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# THE TWO TESTS THAT DECIDE WHETHER A1 IS ACTUALLY RESIDUAL
# ---------------------------------------------------------------------------

def test_gradient_vanishes_when_prior_is_right():
    """Prior confidently correct => f_theta receives almost no gradient.

    This is the entire point of the objective: capacity is not spent
    re-deriving co-occurrence statistics the prior already supplies.
    """
    target = torch.tensor([2])
    prior = _log_prior(2, sharpness=12.0).unsqueeze(0)
    model = torch.zeros(1, P, requires_grad=True)

    z = compose_prior_residual(model, prior, alpha=1.0, stopgrad=True)
    torch.nn.functional.cross_entropy(z, target).backward()

    grad_norm = float(model.grad.abs().sum())
    assert grad_norm < 0.05, (
        f"prior is confidently correct yet f_theta still receives gradient "
        f"{grad_norm:.4f} -- the objective is not residual"
    )


def test_gradient_is_large_when_prior_is_wrong():
    """Prior confidently wrong => f_theta receives a full-magnitude gradient."""
    target = torch.tensor([2])
    prior = _log_prior(5, sharpness=12.0).unsqueeze(0)   # peaked on the WRONG class
    model = torch.zeros(1, P, requires_grad=True)

    z = compose_prior_residual(model, prior, alpha=1.0, stopgrad=True)
    torch.nn.functional.cross_entropy(z, target).backward()

    grad_norm = float(model.grad.abs().sum())
    assert grad_norm > 1.5, (
        f"prior is confidently wrong but f_theta only receives gradient "
        f"{grad_norm:.4f} -- the model is not being asked to correct it"
    )
    # and the gradient must push UP on the true class
    assert float(model.grad[0, 2]) < 0.0


def test_gradient_ratio_wrong_vs_right_is_large():
    """The ratio is the quantitative statement of 'residual'."""
    def grad_for(prior_peak: int) -> float:
        model = torch.zeros(1, P, requires_grad=True)
        z = compose_prior_residual(model, _log_prior(prior_peak, 12.0).unsqueeze(0), alpha=1.0)
        torch.nn.functional.cross_entropy(z, torch.tensor([2])).backward()
        return float(model.grad.abs().sum())

    right, wrong = grad_for(2), grad_for(5)
    assert wrong / max(right, 1e-8) > 50.0, (
        f"gradient on prior-wrong ({wrong:.4f}) is not meaningfully larger than on "
        f"prior-right ({right:.4f}); the objective does not concentrate capacity"
    )


def test_a0_receives_gradient_regardless_of_prior():
    """Contrast arm: without the prior in the graph, the model is trained to
    reproduce the full label distribution, prior-correct examples included."""
    target = torch.tensor([2])
    model = torch.zeros(1, P, requires_grad=True)
    z = compose_prior_residual(model, None, alpha=1.0)          # A0
    torch.nn.functional.cross_entropy(z, target).backward()
    assert float(model.grad.abs().sum()) > 1.5, (
        "A0 should receive a full gradient -- if it does not, the test fixture "
        "is degenerate and the A1 comparison above proves nothing"
    )


# ---------------------------------------------------------------------------
# composition mechanics
# ---------------------------------------------------------------------------

def test_alpha_zero_is_exactly_a0():
    model = torch.randn(4, P)
    out = compose_prior_residual(model, torch.randn(4, P), alpha=0.0)
    assert torch.equal(out, model)


def test_none_prior_is_exactly_a0():
    model = torch.randn(4, P)
    assert torch.equal(compose_prior_residual(model, None, alpha=1.0), model)


def test_centering_does_not_change_the_loss():
    """Centering shifts along the all-ones direction; softmax is invariant."""
    model = torch.randn(3, P)
    prior = torch.log_softmax(torch.randn(3, P), dim=-1)
    tgt = torch.tensor([0, 1, 2])
    a = torch.nn.functional.cross_entropy(compose_prior_residual(model, prior, 1.0, center_prior=True), tgt)
    b = torch.nn.functional.cross_entropy(compose_prior_residual(model, prior, 1.0, center_prior=False), tgt)
    assert abs(float(a) - float(b)) < 1e-5


def test_shape_mismatch_raises_rather_than_broadcasting():
    with pytest.raises(ValueError, match="shape mismatch"):
        compose_prior_residual(torch.randn(4, P), torch.randn(4, P + 1), alpha=1.0)


def test_stopgrad_blocks_gradient_into_a_parameterised_prior():
    """A no-op for a fixed table, but not once the prior is learned."""
    model = torch.zeros(1, P, requires_grad=True)
    learned_prior = torch.zeros(1, P, requires_grad=True)
    z = compose_prior_residual(model, learned_prior, alpha=1.0, stopgrad=True)
    torch.nn.functional.cross_entropy(z, torch.tensor([1])).backward()
    assert learned_prior.grad is None or float(learned_prior.grad.abs().sum()) == 0.0

    model2 = torch.zeros(1, P, requires_grad=True)
    prior2 = torch.zeros(1, P, requires_grad=True)
    z2 = compose_prior_residual(model2, prior2, alpha=1.0, stopgrad=False)
    torch.nn.functional.cross_entropy(z2, torch.tensor([1])).backward()
    assert float(prior2.grad.abs().sum()) > 0.0


def test_alpha_scales_the_prior_contribution():
    model = torch.zeros(2, P)
    prior = torch.log_softmax(torch.randn(2, P), dim=-1)
    z1 = compose_prior_residual(model, prior, alpha=1.0)
    z2 = compose_prior_residual(model, prior, alpha=2.0)
    assert torch.allclose(z2, 2.0 * z1, atol=1e-5)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_detect_a_dead_residual():
    """A zero model must report a ~zero residual-to-prior ratio.

    Acceptance criterion 3 of the brief: the residual must not be numerically
    negligible. This is the number that detects a pass-through.
    """
    d = residual_diagnostics(torch.zeros(8, P), torch.log_softmax(torch.randn(8, P), -1), alpha=1.0)
    assert d["residual_abs_mean"] == 0.0
    assert d["residual_to_prior_ratio"] == 0.0


def test_diagnostics_detect_a_live_residual():
    d = residual_diagnostics(torch.randn(8, P) * 3.0, torch.log_softmax(torch.randn(8, P), -1), alpha=1.0)
    assert d["residual_abs_mean"] > 0.0
    assert d["residual_to_prior_ratio"] > 0.0


def test_diagnostics_report_accuracies_when_targets_given():
    prior = _log_prior(1).repeat(4, 1)
    d = residual_diagnostics(torch.zeros(4, P), prior, alpha=1.0, targets=torch.tensor([1, 1, 1, 1]))
    assert d["prior_top1_acc"] == pytest.approx(1.0)
    assert "total_top1_acc" in d and "model_only_top1_acc" in d


# ---------------------------------------------------------------------------
# visual ablation -- the hard scientific gate
# ---------------------------------------------------------------------------

def test_visual_ablation_none_is_identity():
    t = torch.randn(4, 9, 16)
    for mode in ("none", "", "off", "NONE"):
        assert torch.equal(apply_visual_ablation(t, mode), t)


def test_visual_ablation_zero_removes_all_evidence():
    out = apply_visual_ablation(torch.randn(4, 9, 16), "zero")
    assert float(out.abs().sum()) == 0.0


def test_visual_ablation_shuffle_preserves_content_but_breaks_pairing():
    torch.manual_seed(0)
    t = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    out = apply_visual_ablation(t, "shuffle")
    assert out.shape == t.shape
    assert torch.allclose(out.sum(), t.sum()), "shuffle must permute, not alter, content"
    rows_in = {tuple(r.flatten().tolist()) for r in t}
    rows_out = {tuple(r.flatten().tolist()) for r in out}
    assert rows_in == rows_out


def test_visual_ablation_shuffle_of_two_is_never_identity():
    """With n=2 a random permutation is identity half the time; that would
    silently turn the ablation into a no-op."""
    t = torch.tensor([[[1.0]], [[2.0]]])
    for _ in range(10):
        assert not torch.equal(apply_visual_ablation(t, "shuffle"), t)


def test_visual_ablation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown visual_ablation_mode"):
        apply_visual_ablation(torch.randn(2, 3, 4), "scramble")


# ---------------------------------------------------------------------------
# prior table loading -- must fail loudly, never silently
# ---------------------------------------------------------------------------

VOCAB = ["on", "has", "in", "wearing", "near", "relation"]


def _write_prior(tmp_path, vocab=VOCAB, n=None, pairs=None):
    n = n if n is not None else len(vocab)
    body = {
        "predicate_vocab": vocab,
        "global_log_probs": [-1.0] * n,
        "pair_log_probs": pairs if pairs is not None else {"man||shirt": [-0.1] + [-3.0] * (n - 1)},
        "subject_log_probs": {"man": [-0.5] * n},
        "object_log_probs": {"shirt": [-0.5] * n},
        "smoothing": 1.0,
        "default_log_prob": -3.9,
    }
    p = tmp_path / "prior.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def test_load_returns_none_on_missing_file(tmp_path):
    assert PairPriorTable.load(str(tmp_path / "nope.json"), VOCAB, torch.device("cpu")) is None


def test_load_returns_none_on_unparseable_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert PairPriorTable.load(str(bad), VOCAB, torch.device("cpu")) is None


def test_load_succeeds_and_looks_up_pairs(tmp_path):
    table = PairPriorTable.load(str(_write_prior(tmp_path)), VOCAB, torch.device("cpu"))
    assert table is not None
    out = table.logits_for_pairs([("man", "shirt"), ("dog", "car")], torch.device("cpu"))
    assert out is not None and tuple(out.shape) == (2, len(VOCAB))
    assert torch.isfinite(out).all()
    # exact pair hit must be sharper than the backed-off row
    assert float(out[0].max() - out[0].min()) > float(out[1].max() - out[1].min())


def test_lookup_of_empty_pair_list_returns_none(tmp_path):
    table = PairPriorTable.load(str(_write_prior(tmp_path)), VOCAB, torch.device("cpu"))
    assert table.logits_for_pairs([], torch.device("cpu")) is None


def test_unknown_labels_fall_back_without_raising(tmp_path):
    table = PairPriorTable.load(str(_write_prior(tmp_path)), VOCAB, torch.device("cpu"))
    out = table.logits_for_pairs([("zzz", "qqq")], torch.device("cpu"))
    assert out is not None and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_config_defaults_keep_the_research_arm_off():
    cfg = TrainConfig()
    assert cfg.prior_residual_enabled is False, "a research arm must never be a silent default"
    assert cfg.visual_ablation_mode == "none"
    assert cfg.prior_residual_path == ""


def test_default_alpha_is_one_not_the_evaluation_time_value():
    """At alpha=1 the residual is exactly the log-likelihood ratio. At the
    evaluation-time 3.75 the prior is over-counted and f_theta would have to
    spend capacity cancelling 2.75*log P(p|s,o)."""
    assert TrainConfig().prior_residual_alpha == pytest.approx(1.0)


def test_every_prior_residual_flag_exists_in_the_argparser():
    from openvocab_rel.train import build_argparser

    known = set(vars(build_argparser().parse_known_args([])[0]).keys())
    for flag in (
        "prior_residual_enabled", "prior_residual_alpha", "prior_residual_stopgrad",
        "prior_residual_train_only", "prior_residual_path", "visual_ablation_mode",
    ):
        assert flag in known, (
            f"--{flag} is not defined in build_argparser(); parse_known_args would "
            "discard it silently"
        )
        assert hasattr(TrainConfig(), flag)
