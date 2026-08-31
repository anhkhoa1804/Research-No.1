"""Regression tests for the experiment C' calibration pathway.

Three things are pinned here:

1. `eval_freq_bias_tau = 0.0` (the default) is a STRICT no-op, so the
   historical protocol is byte-identical to what it produced before the
   pathway existed.
2. `eval_freq_bias_tau > 0` changes the intended tensor -- the frequency-prior
   rows -- by exactly `-tau * log P(p)`.
3. The OLD flag, `eval_logit_adj_tau`, remains inert under the historical
   protocol's `eval_sgg_predicate_ensemble_alpha = 0.0`. That is the defect
   documented in runs/p8_tau_path_bug/; this test exists so nobody
   rediscovers it by spending a GPU-hour on a silent null.

Nothing here loads a checkpoint. CPU only.
"""
from __future__ import annotations

import torch

from openvocab_rel.evals import (
    _apply_eval_logit_adjustment,
    _apply_freq_bias_tau,
    _apply_frequency_bias,
    _relation_predicate_logits,
)

P = 51
N = 6
D = 8


class _Cfg:
    def __init__(self, **kw):
        self.freq_bias_alpha = 3.75
        self.bayes_calibration_weight = 0.0
        self.eval_freq_bias_tau = 0.0
        self.eval_sgg_predicate_score_mode = "ensemble"
        self.eval_sgg_use_predicate_classifier = True
        self.eval_sgg_predicate_ensemble_alpha = 0.0
        self.adaptive_calibration_enabled = True
        self.eval_sgg_classifier_temperature = 1.0
        self.eval_sgg_text_temperature = 1.0
        self.logit_adj_tau = 0.0
        self.eval_logit_adj_tau = -1.0
        for k, v in kw.items():
            setattr(self, k, v)


class _Out:
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self._text = torch.randn(N, P, generator=g)
        self._cls = torch.randn(N, P, generator=g) * 3.0 + 1.0

    def text_predicate_logits(self, rel_feat, pred_emb):
        return self._text

    def predicate_logits(self, rel_feat):
        return self._cls

    def calibrated_predicate_logits(self, rel_feat, pred_log_prior):
        return self._cls


def _freq_bias(seed=3):
    g = torch.Generator().manual_seed(seed)
    marginal = torch.log(torch.rand(P, generator=g).clamp_min(1e-6))
    pairs = {f"a{i}||b{i}": torch.log(torch.rand(P, generator=g).clamp_min(1e-6)) for i in range(4)}
    return {"global": marginal, "pairs": pairs, "subjects": {}, "objects": {},
            "smoothing": 1.0, "num_source_predicates": P}


def _pair_inputs():
    pair_list = [(0, 1)] * N
    obj_labels = ["a0", "b0"]
    return pair_list, obj_labels


# --------------------------------------------------------------------------
# 1. tau = 0 is a strict no-op
# --------------------------------------------------------------------------

def test_tau_zero_returns_the_same_object():
    fb = _freq_bias()
    prior = torch.randn(N, P)
    out = _apply_freq_bias_tau(_Cfg(eval_freq_bias_tau=0.0), prior, fb)
    assert out is prior, "tau=0 must return the identical object, not a copy"


def test_tau_zero_leaves_composition_byte_identical():
    fb = _freq_bias()
    pair_list, obj_labels = _pair_inputs()
    logits = torch.randn(N, P)
    dev = torch.device("cpu")
    a = _apply_frequency_bias(_Cfg(), logits, fb, pair_list, obj_labels, dev)
    b = _apply_frequency_bias(_Cfg(eval_freq_bias_tau=0.0), logits, fb, pair_list, obj_labels, dev)
    assert torch.equal(a, b)


def test_missing_attribute_behaves_as_tau_zero():
    """A cfg predating this field must behave exactly as tau=0."""
    class Legacy:
        freq_bias_alpha = 3.75
        bayes_calibration_weight = 0.0
    fb = _freq_bias()
    pair_list, obj_labels = _pair_inputs()
    logits = torch.randn(N, P)
    dev = torch.device("cpu")
    a = _apply_frequency_bias(Legacy(), logits, fb, pair_list, obj_labels, dev)
    b = _apply_frequency_bias(_Cfg(eval_freq_bias_tau=0.0), logits, fb, pair_list, obj_labels, dev)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# 2. tau > 0 changes the intended tensor, by exactly the intended amount
# --------------------------------------------------------------------------

def test_tau_positive_subtracts_tau_times_log_marginal():
    fb = _freq_bias()
    prior = torch.randn(N, P)
    tau = 0.1
    got = _apply_freq_bias_tau(_Cfg(eval_freq_bias_tau=tau), prior, fb)
    want = prior - tau * fb["global"].view(1, -1)
    assert torch.allclose(got, want, atol=0, rtol=0)


def test_tau_positive_changes_the_composed_logits():
    fb = _freq_bias()
    pair_list, obj_labels = _pair_inputs()
    logits = torch.randn(N, P)
    dev = torch.device("cpu")
    a = _apply_frequency_bias(_Cfg(), logits, fb, pair_list, obj_labels, dev)
    b = _apply_frequency_bias(_Cfg(eval_freq_bias_tau=0.1), logits, fb, pair_list, obj_labels, dev)
    assert not torch.equal(a, b)
    # and the difference is exactly -alpha * tau * log P(p), broadcast over pairs
    delta = b - a
    want = -3.75 * 0.1 * fb["global"].view(1, -1).expand_as(delta)
    assert torch.allclose(delta, want, atol=1e-5)


def test_tau_boosts_rare_predicates_relative_to_common_ones():
    """The whole point: rarer classes (more negative log P) gain more."""
    fb = _freq_bias()
    prior = torch.zeros(N, P)
    adj = _apply_freq_bias_tau(_Cfg(eval_freq_bias_tau=0.1), prior, fb)
    rarest = int(torch.argmin(fb["global"]))
    commonest = int(torch.argmax(fb["global"]))
    assert adj[0, rarest] > adj[0, commonest]


def test_tau_is_inert_when_freq_bias_absent():
    prior = torch.randn(N, P)
    assert _apply_freq_bias_tau(_Cfg(eval_freq_bias_tau=0.5), prior, None) is prior
    assert _apply_freq_bias_tau(_Cfg(eval_freq_bias_tau=0.5), None, _freq_bias()) is None


def test_tau_is_inert_on_a_shape_mismatch():
    fb = _freq_bias()
    fb["global"] = torch.zeros(P + 3)
    prior = torch.randn(N, P)
    assert _apply_freq_bias_tau(_Cfg(eval_freq_bias_tau=0.5), prior, fb) is prior


# --------------------------------------------------------------------------
# 3. the OLD flag stays inert under the historical protocol
# --------------------------------------------------------------------------

def test_eval_logit_adj_tau_is_inert_at_ensemble_alpha_zero():
    """Pins the defect proven in runs/p8_tau_path_bug/.

    If this test ever FAILS, the old path became live and this file's premise
    (that C' needs a separate pathway) must be re-examined -- do not simply
    delete the test.
    """
    out = _Out()
    rel_feat, pred_emb = torch.zeros(N, D), torch.zeros(P, D)
    g = torch.Generator().manual_seed(11)
    prior = torch.log(torch.rand(P, generator=g).clamp_min(1e-6))
    base = _relation_predicate_logits(_Cfg(), out, rel_feat, pred_emb, prior)
    tau = _relation_predicate_logits(_Cfg(eval_logit_adj_tau=1.0), out, rel_feat, pred_emb, prior)
    assert torch.equal(base, tau), "eval_logit_adj_tau unexpectedly became live at alpha=0"


def test_eval_logit_adj_tau_is_live_when_ensemble_alpha_is_positive():
    """Two-sided control: the old code is unreachable, not dead."""
    out = _Out()
    rel_feat, pred_emb = torch.zeros(N, D), torch.zeros(P, D)
    g = torch.Generator().manual_seed(11)
    prior = torch.log(torch.rand(P, generator=g).clamp_min(1e-6))
    base = _relation_predicate_logits(
        _Cfg(eval_sgg_predicate_ensemble_alpha=0.5), out, rel_feat, pred_emb, prior)
    tau = _relation_predicate_logits(
        _Cfg(eval_sgg_predicate_ensemble_alpha=0.5, eval_logit_adj_tau=0.5),
        out, rel_feat, pred_emb, prior)
    assert not torch.equal(base, tau)


def test_the_two_taus_are_different_mechanisms():
    """eval_logit_adj_tau touches cls_logits; eval_freq_bias_tau touches the prior."""
    cls = torch.randn(N, P)
    g = torch.Generator().manual_seed(5)
    prior_vec = torch.log(torch.rand(P, generator=g).clamp_min(1e-6))
    adjusted = _apply_eval_logit_adjustment(_Cfg(eval_logit_adj_tau=0.1), cls, prior_vec)
    assert not torch.equal(adjusted, cls)          # it does work, on cls_logits
    fb = _freq_bias()
    rows = torch.randn(N, P)
    assert torch.equal(_apply_freq_bias_tau(_Cfg(), rows, fb), rows)   # ours defaults off


# --------------------------------------------------------------------------
# 4. the dump hook is default-off
# --------------------------------------------------------------------------

def test_pair_dump_is_disabled_by_default():
    from openvocab_rel.config import TrainConfig
    from openvocab_rel.evals import _pair_dump_enabled
    assert _pair_dump_enabled(TrainConfig()) == ""
    assert TrainConfig().eval_freq_bias_tau == 0.0
