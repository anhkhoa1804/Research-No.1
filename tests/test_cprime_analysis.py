"""Regression tests for the experiment C' CPU analysis.

Blocker 2 of the C' plan requires proof that the comparison is like-for-like.
What is pinned:

1. Every arm is scored over the SAME predicate set and the SAME mR denominator.
   A cross-arm delta computed over different class counts is not a measurement,
   and that defect is already on record in this repository
   (docs/research_sessions/POST_B_CLAIM_AUDIT.md 1.4).
2. Predicate ORDER is preserved: column i of the cached logits keeps its
   identity through the foreground filter and the scheme mapping.
3. raw50 gives 50 classes, eval48 gives 48, and the two differ by exactly
   near->next to and wears->wearing.
4. tau = 0 is a no-op on the prior term, and the prior-only arm is invariant to
   alpha (a positive rescale cannot move an argmax).
5. The Pareto helper reports a two-axis win as a win, not as indeterminate --
   the bug that would have recorded a positive C' as inconclusive.
6. The nulls permute what they claim to permute and nothing else.

CPU only. Builds a synthetic cache; never reads the GPU artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.cprime_analysis import ALPHA_HIST, Bench, null_rows, pareto_gap

# A synthetic prior file. The real 280 MB one is not parsed in tests; what
# matters here is the CONTRACT (global_log_probs is read, in vocabulary order),
# not its values.
@pytest.fixture(scope="module")
def PRIOR(tmp_path_factory):
    pv = _vg150()
    p = tmp_path_factory.mktemp("p") / "prior.json"
    p.write_text(json.dumps({
        "predicate_vocab": pv,
        "global_log_probs": [-1.0 - 0.1 * i for i in range(len(pv))],
        "default_log_prob": -20.0,
    }))
    return p


def _vg150():
    raw = json.loads(Path("datasets_vg150_clean/vocabulary/predicates.json").read_text())["idx_to_predicate"]
    return [raw[str(i)] for i in range(1, len(raw) + 1)]


@pytest.fixture(scope="module")
def dump_path(tmp_path_factory):
    """A small synthetic cache in the real schema, with real predicate names."""
    pv = _vg150()
    vocab = [p.strip().lower() for p in pv] + ["background"]
    P = len(vocab)
    g = torch.Generator().manual_seed(0)
    n_img, n_obj, n_pair = 12, 4, 6
    d = {k: [] for k in ("image_id", "pairs", "pair_index", "model_logits", "prior_rows",
                         "text_logits", "cls_logits", "obj_labels", "subj_label", "obj_label",
                         "obj_boxes", "gt_subj_idx", "gt_obj_idx", "gt_pred",
                         "gt_subj_label", "gt_obj_label")}
    marginal = torch.randn(P, generator=g)
    for i in range(n_img):
        pairs = torch.tensor([[a, b] for a in range(3) for b in range(3) if a != b][:n_pair])
        d["image_id"].append(str(i))
        d["pairs"].append(pairs)
        d["pair_index"].append(torch.arange(n_pair))
        for k in ("model_logits", "text_logits", "cls_logits"):
            d[k].append(torch.randn(n_pair, P, generator=g))
        # every second pair falls back to the marginal, so it is the modal row
        pr = torch.randn(n_pair, P, generator=g)
        pr[::2] = marginal
        d["prior_rows"].append(pr)
        labels = [f"obj{j}" for j in range(n_obj)]
        d["obj_labels"].append(labels)
        d["subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
        d["obj_boxes"].append(torch.zeros((n_obj, 4)))
        # GT includes both alias-collapsed sources, raw
        preds = ["near", "wears", "on", "has", "in", "next to"][: n_pair]
        d["gt_subj_idx"].append([int(a) for a, _ in pairs.tolist()])
        d["gt_obj_idx"].append([int(b) for _, b in pairs.tolist()])
        d["gt_pred"].append(preds)
        d["gt_subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["gt_obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
    d.update({
        "schema": "pair_logit_dump_v2", "pred_vocab": vocab,
        "background_predicate_indices": [P - 1],
        "predicate_alias_map": {"near": "next to", "wears": "wearing"},
        "freq_bias_alpha": ALPHA_HIST, "eval_freq_bias_tau": 0.0,
        "ensemble_alpha": 0.0, "score_mode": "ensemble",
        "n_pairs": n_img * n_pair, "n_images": n_img,
    })
    p = tmp_path_factory.mktemp("c") / "dump.pt"
    torch.save(d, p)
    return p


# ------------------------------------------------- 1/3 denominators + schemes
def test_raw50_and_eval48_class_counts(dump_path, PRIOR):
    assert Bench(dump_path, "raw50", PRIOR).n_classes == 50
    assert Bench(dump_path, "eval48", PRIOR).n_classes == 48


def test_the_two_schemes_differ_by_exactly_the_two_known_merges(dump_path, PRIOR):
    a, b = Bench(dump_path, "raw50", PRIOR), Bench(dump_path, "eval48", PRIOR)
    assert set(a.classes) - set(b.classes) == {"near", "wears"}
    assert set(b.classes) - set(a.classes) == set()


def test_every_arm_shares_one_predicate_set_and_one_denominator(dump_path, PRIOR):
    """The defect this pins: comparing a 48-class mean against a 50-class one."""
    B = Bench(dump_path, "raw50", PRIOR)
    arms = {
        "prior_only": B.score(0.1, ALPHA_HIST, None),
        "model_plus_prior": B.score(0.1, ALPHA_HIST, B.model),
        "classifier_plus_prior": B.score(0.1, ALPHA_HIST, B.cls),
        "text_plus_prior": B.score(0.1, ALPHA_HIST, B.text),
        "model_only": B.score(0.0, 0.0, B.model),
    }
    ms = {k: B.metrics(v) for k, v in arms.items()}
    counts = {k: v["n_classes"] for k, v in ms.items()}
    assert len(set(counts.values())) == 1, counts          # identical denominator
    ns = {k: v["n"] for k, v in ms.items()}
    assert len(set(ns.values())) == 1, ns                  # identical GT item set


def test_gt_item_set_is_identical_across_schemes_up_to_merging(dump_path, PRIOR):
    a, b = Bench(dump_path, "raw50", PRIOR), Bench(dump_path, "eval48", PRIOR)
    assert int(a.gt_y.numel()) == int(b.gt_y.numel())
    assert torch.equal(a.gt_row, b.gt_row)


# ------------------------------------------------------------- 2 ordering
def test_predicate_column_order_is_preserved(dump_path, PRIOR):
    B = Bench(dump_path, "raw50", PRIOR)
    assert B.fg_names_raw == [p.strip().lower() for p in _vg150()]
    assert B.col_label == B.fg_names_raw           # raw50 is the identity map
    assert len(B.fg_cols) == 50


def test_eval48_maps_columns_without_reordering_them(dump_path, PRIOR):
    B = Bench(dump_path, "eval48", PRIOR)
    for col, raw in enumerate(B.fg_names_raw):
        expected = {"near": "next to", "wears": "wearing"}.get(raw, raw)
        assert B.col_label[col] == expected


# ----------------------------------------------------------------- 4 tau/alpha
def test_tau_zero_is_a_no_op_on_the_prior_term(dump_path, PRIOR):
    B = Bench(dump_path, "raw50", PRIOR)
    assert torch.equal(B.score(0.0, 1.0, None), B.prior)


def test_prior_only_arm_is_invariant_to_alpha(dump_path, PRIOR):
    """A positive rescale cannot move an argmax; if it does, the metric is wrong."""
    B = Bench(dump_path, "raw50", PRIOR)
    base = B.metrics(B.score(0.1, 1.0, None))
    for a in (0.5, 2.0, ALPHA_HIST, 7.5):
        m = B.metrics(B.score(0.1, a, None))
        assert m["R"] == pytest.approx(base["R"])
        assert m["mR"] == pytest.approx(base["mR"])


def test_marginal_is_read_from_the_prior_file_not_inferred(dump_path, PRIOR):
    """The defect this pins: the modal prior row is the UNIFORM default_log_prob
    fallback, not the class marginal. Inferring tau from it made tau a silent
    no-op (a constant subtraction cannot move an argmax)."""
    B = Bench(dump_path, "raw50", PRIOR)
    raw = json.loads(PRIOR.read_text())
    src = [str(x).strip().lower() for x in raw["predicate_vocab"]]
    idx = {p: i for i, p in enumerate(src)}
    for col, name in enumerate(B.fg_names_raw):
        assert B.log_marginal[col] == pytest.approx(raw["global_log_probs"][idx[name]])
    assert B.marginal_missing == []
    # and it must NOT be uniform -- that was exactly the failure mode
    assert float(B.log_marginal.max() - B.log_marginal.min()) > 1.0


def test_bench_refuses_to_run_without_a_prior_file(dump_path, PRIOR):
    with pytest.raises(ValueError):
        Bench(dump_path, "raw50", None)


# ------------------------------------------------------------------ 5 pareto
def test_pareto_reports_a_two_axis_win_as_a_win_not_as_indeterminate():
    curve = [{"R": 0.60, "mR": 0.30}, {"R": 0.50, "mR": 0.36}]
    g = pareto_gap(curve, R=0.65, mR=0.34)        # better on BOTH axes
    assert g is not None and g == pytest.approx(4.0)


def test_pareto_interpolates_inside_the_range():
    curve = [{"R": 0.60, "mR": 0.30}, {"R": 0.50, "mR": 0.40}]
    assert pareto_gap(curve, R=0.55, mR=0.35) == pytest.approx(0.0)
    assert pareto_gap(curve, R=0.55, mR=0.37) == pytest.approx(2.0)


def test_pareto_refuses_to_extrapolate_below_the_frontier():
    curve = [{"R": 0.60, "mR": 0.30}, {"R": 0.50, "mR": 0.40}]
    assert pareto_gap(curve, R=0.20, mR=0.99) is None


# ------------------------------------------------------------------- 6 nulls
def test_N1_permutes_only_within_each_image(dump_path, PRIOR):
    B = Bench(dump_path, "raw50", PRIOR)
    idx = null_rows(B, "N1", 1)
    assert sorted(idx.tolist()) == list(range(int(B.model.shape[0])))
    assert torch.equal(B.img_of_row[idx], B.img_of_row)      # image membership kept


def test_N2_permutes_across_the_whole_split(dump_path, PRIOR):
    B = Bench(dump_path, "raw50", PRIOR)
    idx = null_rows(B, "N2", 1)
    assert sorted(idx.tolist()) == list(range(int(B.model.shape[0])))
    assert not torch.equal(idx, torch.arange(int(B.model.shape[0])))


def test_nulls_preserve_the_model_row_multiset(dump_path, PRIOR):
    """The null must change WHICH pair a row scores, never the rows themselves."""
    B = Bench(dump_path, "raw50", PRIOR)
    for kind in ("N1", "N2"):
        idx = null_rows(B, kind, 3)
        assert torch.equal(B.model[idx].sort(0).values, B.model.sort(0).values)
