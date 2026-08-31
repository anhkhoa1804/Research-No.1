"""Regression tests for the experiment C' pair-logit cache (schema v2).

What is pinned here, and why each one exists:

1. The components helper recomposes to `_relation_predicate_logits` BIT FOR
   BIT, in every score mode. The dump stores the two branches separately; if
   the helper ever drifts from the real composition, the cache would describe
   a model that was never evaluated.

2. Under the historical protocol (`ensemble_alpha = 0.0`) the composed model
   term equals the TEXT branch alone and the classifier branch is discarded.
   This is the suppression point experiment C' exists to probe, so it is
   asserted rather than left as a comment.

3. The dump is disabled by default and the historical protocol is unchanged
   by its presence.

4. RAW GT. An earlier revision handed `_collect_pair_dump` the alias-normalised
   GT, which would have collapsed `near`->`next to` and `wears`->`wearing`
   inside a 60-GPU-minute artifact and made the 50-class arm unrecoverable.
   The test asserts both survive, and that the exported alias map reproduces
   the 48-class scheme.

5. `subj_label`/`obj_label` are redundant with `obj_labels[pairs]` by
   construction; the redundancy is a checked invariant.

CPU only. No checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from openvocab_rel.evals import (
    _build_vg_aliases,
    _collect_pair_dump,
    _pair_dump_enabled,
    _predicate_logit_components,
    _relation_predicate_logits,
    _write_pair_dump,
)

P = 51
N = 6
VOCAB = [f"p{i}" for i in range(P - 1)] + ["background"]


class _Cfg:
    def __init__(self, **kw):
        self.eval_sgg_predicate_score_mode = "ensemble"
        self.eval_sgg_use_predicate_classifier = True
        self.eval_sgg_predicate_ensemble_alpha = 0.0
        self.adaptive_calibration_enabled = True
        self.eval_sgg_classifier_temperature = 1.0
        self.eval_sgg_text_temperature = 1.0
        self.logit_adj_tau = 0.0
        self.eval_logit_adj_tau = -1.0
        self.freq_bias_alpha = 3.75
        self.bayes_calibration_weight = 0.0
        self.eval_freq_bias_tau = 0.0
        self.eval_sgg_dump_pair_logits_path = ""
        self.eval_sgg_use_gt_pairs = True
        self.eval_sgg_iou_thresh = 0.5
        self.eval_sgg_use_vg_aliases = True
        self.freq_bias_path = ""
        self.resume_from = ""
        self.vg150_root = ""
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


def _fixture():
    return _Out(), torch.randn(N, 8), torch.randn(P, 16), torch.randn(P)


def _recompose(cfg, comp):
    """Rebuild the composed model term from the stored branches."""
    if comp["branch"] in {"text_only", "text_only_shape_mismatch"}:
        return comp["text_logits"]
    if comp["branch"] == "classifier_only":
        return comp["cls_logits"]
    a = comp["ensemble_alpha_used"]
    return (a * comp["cls_norm"]) + ((1.0 - a) * comp["text_norm"])


# ---------------------------------------------------------------- 1. no drift
@pytest.mark.parametrize("mode", ["ensemble", "classifier", "text", "auto"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_components_recompose_to_the_composed_term(mode, alpha):
    cfg = _Cfg(eval_sgg_predicate_score_mode=mode, eval_sgg_predicate_ensemble_alpha=alpha)
    out, rel_feat, pred_emb, plp = _fixture()
    composed = _relation_predicate_logits(cfg, out, rel_feat, pred_emb, plp)
    comp = _predicate_logit_components(cfg, out, rel_feat, pred_emb, plp)
    assert torch.equal(_recompose(cfg, comp), composed)


# ------------------------------------------- 2. the suppression point itself
def test_historical_protocol_discards_the_classifier_branch_entirely():
    cfg = _Cfg(eval_sgg_predicate_ensemble_alpha=0.0)
    out, rel_feat, pred_emb, plp = _fixture()
    comp = _predicate_logit_components(cfg, out, rel_feat, pred_emb, plp)
    composed = _relation_predicate_logits(cfg, out, rel_feat, pred_emb, plp)
    assert comp["ensemble_alpha_used"] == 0.0
    # the composed term is the text branch alone...
    assert torch.equal(composed, comp["text_norm"])
    # ...and the classifier branch was computed but contributes nothing.
    assert comp["cls_logits"] is not None
    assert not torch.allclose(comp["cls_norm"], comp["text_norm"])


def test_classifier_branch_is_still_captured_when_it_is_discarded():
    """C' depends on the discarded branch being present in the cache."""
    cfg = _Cfg(eval_sgg_predicate_ensemble_alpha=0.0)
    out, rel_feat, pred_emb, plp = _fixture()
    comp = _predicate_logit_components(cfg, out, rel_feat, pred_emb, plp)
    assert isinstance(comp["cls_logits"], torch.Tensor)
    assert torch.isfinite(comp["cls_logits"]).all()


# ------------------------------------------------------------- 3. default off
def test_dump_is_disabled_by_default():
    assert _pair_dump_enabled(_Cfg()) == ""
    assert _pair_dump_enabled(_Cfg(eval_sgg_dump_pair_logits_path="  ")) == ""
    assert _pair_dump_enabled(_Cfg(eval_sgg_dump_pair_logits_path="x.pt")) == "x.pt"


# ------------------------------------------------------------------ 4. RAW GT
ALIASED_PAIRS = [("near", "next to"), ("wears", "wearing")]


def _vg150_predicates():
    raw = json.loads(Path("datasets_vg150_clean/vocabulary/predicates.json").read_text())["idx_to_predicate"]
    return [raw[str(i)] for i in range(1, len(raw) + 1)]


def _dump_once(tmp_path: Path, gt_raw, vocab=None):
    vocab = VOCAB if vocab is None else vocab
    cfg = _Cfg()
    out, rel_feat, pred_emb, plp = _fixture()
    model_logits = _relation_predicate_logits(cfg, out, rel_feat, pred_emb, plp)
    comps = _predicate_logit_components(cfg, out, rel_feat, pred_emb, plp)
    pair_list = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    obj_labels = ["man", "shirt", "table"]
    dump = {k: [] for k in ("image_id", "pairs", "pair_index", "model_logits", "prior_rows",
                            "text_logits", "cls_logits", "obj_labels", "subj_label", "obj_label",
                            "obj_boxes", "gt_subj_idx", "gt_obj_idx", "gt_pred",
                            "gt_subj_label", "gt_obj_label")}
    ex = {"image_id": "42", "obj_boxes": torch.zeros((3, 4))}
    _collect_pair_dump(dump, ex, gt_raw, pair_list, obj_labels, model_logits,
                       comps, None, torch.device("cpu"))
    path = tmp_path / "dump.pt"
    _write_pair_dump(dump, str(path), cfg, vocab)
    return torch.load(path, weights_only=False), pair_list, obj_labels


def test_dump_stores_raw_gt_predicates_not_aliased_ones(tmp_path):
    gt_raw = {"subj_idx": [0, 0], "obj_idx": [1, 2],
              "pred_labels": ["near", "wears"],
              "subj_labels": ["man", "man"], "obj_labels": ["shirt", "table"]}
    payload, _, _ = _dump_once(tmp_path, gt_raw)
    stored = payload["gt_pred"][0]
    assert stored == ["near", "wears"], stored
    for src, dst in ALIASED_PAIRS:
        assert dst not in stored, f"{src} was collapsed to {dst} inside the cache"


def test_exported_alias_map_reproduces_the_48_class_scheme(tmp_path):
    gt_raw = {"subj_idx": [0], "obj_idx": [1], "pred_labels": ["near"],
              "subj_labels": ["man"], "obj_labels": ["shirt"]}
    pv = _vg150_predicates()
    assert len(pv) == 50
    # dump under the REAL runtime vocabulary: VG150's 50 predicates + background
    payload, _, _ = _dump_once(tmp_path, gt_raw, vocab=pv + ["background"])
    amap = payload["predicate_alias_map"]
    fires = {k: v for k, v in _build_vg_aliases([], pv).items() if k != v and k in pv}
    assert fires == {"near": "next to", "wears": "wearing"}
    collapsed = {amap.get(p, p) for p in pv}
    assert len(collapsed) == 48
    # and the map the cache exports agrees with the evaluator's own
    for k, v in fires.items():
        assert amap[k] == v


# ----------------------------------------------------- 5. checked redundancy
def test_per_pair_labels_agree_with_obj_labels_indexed_by_pairs(tmp_path):
    gt_raw = {"subj_idx": [0], "obj_idx": [1], "pred_labels": ["on"],
              "subj_labels": ["man"], "obj_labels": ["shirt"]}
    payload, pair_list, obj_labels = _dump_once(tmp_path, gt_raw)
    pairs = payload["pairs"][0]
    for i, (a, b) in enumerate(pair_list):
        assert payload["subj_label"][0][i] == obj_labels[a]
        assert payload["obj_label"][0][i] == obj_labels[b]
        assert int(pairs[i, 0]) == a and int(pairs[i, 1]) == b
    assert payload["pair_index"][0].tolist() == list(range(len(pair_list)))


def test_schema_is_versioned_and_self_describing(tmp_path):
    gt_raw = {"subj_idx": [], "obj_idx": [], "pred_labels": [],
              "subj_labels": [], "obj_labels": []}
    payload, _, _ = _dump_once(tmp_path, gt_raw)
    assert payload["schema"] == "pair_logit_dump_v2"
    for key in ("composition", "pred_vocab", "freq_bias_alpha", "ensemble_alpha",
                "score_mode", "use_vg_aliases", "predicate_alias_map",
                "background_predicate_indices", "n_pairs", "n_images", "schema_doc"):
        assert key in payload, key
    assert payload["pred_vocab"] == VOCAB          # ordering preserved verbatim
    assert payload["background_predicate_indices"] == [P - 1]


def test_branches_are_stored_and_finite(tmp_path):
    gt_raw = {"subj_idx": [], "obj_idx": [], "pred_labels": [],
              "subj_labels": [], "obj_labels": []}
    payload, pair_list, _ = _dump_once(tmp_path, gt_raw)
    for key in ("model_logits", "prior_rows", "text_logits", "cls_logits"):
        t = payload[key][0]
        assert tuple(t.shape) == (len(pair_list), P), key
    assert torch.isfinite(payload["text_logits"][0]).all()
    assert torch.isfinite(payload["cls_logits"][0]).all()
    assert payload.get("missing_text_logits", 0) == 0
    assert payload.get("missing_cls_logits", 0) == 0
    assert payload["ensemble_alpha_used"] == 0.0
    assert payload["branch"] == "ensemble"
