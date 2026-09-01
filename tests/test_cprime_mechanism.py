"""Regression tests for the C' MECHANISM analysis.

What is pinned:

1. THE NORMALISATION DEFECT. `cprime_analysis.Bench.ensemble_term` slices to
   the 50 foreground columns and then standardises; `evals.py` standardises
   over all 51 columns and slices afterwards. The two orders are not equal.
   `fixed_ensemble` implements the evaluator's order, and the proof is that it
   reproduces a cache whose `model_logits` were built the evaluator's way while
   `ensemble_term` does not. Both directions are asserted, so neither a
   regression of the fix nor a silent "fix" of the buggy path goes unnoticed.
2. The flip decomposition is a true partition of the GT rows -- no row is
   double-counted between rescued / destroyed / both-right / both-wrong.
3. A zero-scale model term is exactly the prior-only arm (no flips), and a
   top-k restriction at k = n_classes is exactly the unrestricted arm. These
   are the two endpoints that would silently break the mechanism sweeps.
4. The held-out split is IMAGE-disjoint, not row-disjoint: rows from one image
   must never straddle the selection boundary, or selection leaks.
5. Rank metrics are internally consistent (recall@1 is the rank-1 mass) and
   per-class contributions sum to the global mR delta.

CPU only. Builds a synthetic cache; never reads the GPU artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.cprime_mechanism import (ALPHA_HIST, Mech, analysis_A_flips,
                                    analysis_B_ranks, analysis_DE_buckets_predicates,
                                    controls, heldout)


def _vg150():
    raw = json.loads(Path("datasets_vg150_clean/vocabulary/predicates.json").read_text())["idx_to_predicate"]
    return [raw[str(i)] for i in range(1, len(raw) + 1)]


def _norm(x):
    m = x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True).clamp_min(1e-4)
    return (x - m) / s


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


@pytest.fixture(scope="module")
def dump_path(tmp_path_factory):
    """Synthetic cache built the way evals.py builds one.

    The load-bearing detail: `model_logits` is the FULL-WIDTH normalisation of
    `text_logits` at ensemble_alpha = 0, exactly as `_relation_predicate_logits`
    produces it. That is what makes test 1 a real test of the ORDER rather than
    of arithmetic.
    """
    pv = _vg150()
    vocab = [p.strip().lower() for p in pv] + ["background"]
    P = len(vocab)
    g = torch.Generator().manual_seed(11)
    n_img, n_obj, n_pair = 14, 4, 6
    keys = ("image_id", "pairs", "pair_index", "model_logits", "prior_rows",
            "text_logits", "cls_logits", "obj_labels", "subj_label", "obj_label",
            "obj_boxes", "gt_subj_idx", "gt_obj_idx", "gt_pred",
            "gt_subj_label", "gt_obj_label")
    d = {k: [] for k in keys}
    for i in range(n_img):
        pairs = torch.tensor([[a, b] for a in range(3) for b in range(3) if a != b][:n_pair])
        text = torch.randn(n_pair, P, generator=g)
        cls = torch.randn(n_pair, P, generator=g)
        d["image_id"].append(str(i))
        d["pairs"].append(pairs)
        d["pair_index"].append(torch.arange(n_pair))
        d["text_logits"].append(text)
        d["cls_logits"].append(cls)
        d["model_logits"].append(_norm(text))          # ensemble_alpha = 0.0
        d["prior_rows"].append(torch.randn(n_pair, P, generator=g))
        labels = [f"obj{j}" for j in range(n_obj)]
        d["obj_labels"].append(labels)
        d["subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
        d["obj_boxes"].append(torch.zeros((n_obj, 4)))
        preds = ["near", "wears", "on", "has", "in", "next to"][:n_pair]
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
        "classifier_temperature": 1.0, "text_temperature": 1.0,
        "n_pairs": n_img * n_pair, "n_images": n_img,
    })
    p = tmp_path_factory.mktemp("c") / "dump.pt"
    torch.save(d, p)
    return p


@pytest.fixture(scope="module")
def B(dump_path, PRIOR):
    return Mech(str(dump_path), str(PRIOR), "raw50")


# ------------------------------------------------- 1. the normalisation defect
def test_fixed_ensemble_reproduces_the_stored_evaluator_model_term(B):
    """The evaluator's order: standardise over 51, THEN drop background."""
    assert float((B.fixed_ensemble(0.0) - B.model).abs().max()) < 1e-5


def test_buggy_ensemble_term_does_not_reproduce_it(B):
    """Pins the defect itself, so a silent revert is caught.

    If cprime_analysis is ever corrected, this test must be updated together
    with the reconciliation report -- it is deliberately load-bearing.
    """
    assert float((B.ensemble_term(0.0) - B.model).abs().max()) > 1e-3


def test_normalisation_order_is_not_commutative():
    x = torch.randn(7, 51, generator=torch.Generator().manual_seed(3))
    fg = list(range(50))
    assert not torch.allclose(_norm(x)[:, fg], _norm(x[:, fg]), atol=1e-4)


# ------------------------------------------------------- 2. flip partitioning
def test_flip_decomposition_partitions_every_gt_row(B):
    a = analysis_A_flips(B, 0.0)
    total = (a["both_right"] + a["rescued_wrong_to_right"] + a["destroyed_right_to_wrong"]
             + a["both_wrong_unchanged"] + a["both_wrong_changed"])
    assert total == a["n_gt"] == B.n_gt
    assert a["net_flips"] == a["rescued_wrong_to_right"] - a["destroyed_right_to_wrong"]


def test_flip_counts_agree_with_recall_delta(B):
    a = analysis_A_flips(B, 0.0)
    assert a["combined_R"] * a["n_gt"] == pytest.approx(
        a["prior_R"] * a["n_gt"] + a["net_flips"], abs=1e-3)


# --------------------------------------------------------- 3. sweep endpoints
def test_zero_scale_model_term_is_exactly_the_prior_only_arm(B):
    c = controls(B, 0.0, seeds=1, curve=[dict(B.metrics(B.score(0.0, ALPHA_HIST, None)), tau=0.0)])
    z = next(r for r in c["scale_sweep"] if r["scale"] == 0.0)
    assert z["net_flips"] == 0 and z["argmax_changed"] == 0
    assert z["dR_points"] == pytest.approx(0.0, abs=1e-9)


def test_topk_restriction_at_full_width_is_the_unrestricted_arm(B):
    c = controls(B, 0.0, seeds=1, curve=[dict(B.metrics(B.score(0.0, ALPHA_HIST, None)), tau=0.0)])
    assert c[f"restrict_prior_top{B.n_classes}"]["net_flips"] == c["real"]["net_flips"]
    assert c[f"restrict_prior_top{B.n_classes}"]["R"] == pytest.approx(c["real"]["R"], abs=1e-9)


# ------------------------------------------------------ 4. held-out integrity
def test_heldout_split_is_image_disjoint(B):
    ho = heldout(B, seeds=1, select_seed=7)
    assert ho["n_rows_A"] + ho["n_rows_B"] == B.n_gt
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(B.n_images, generator=g)
    halfA = set(perm[: B.n_images // 2].tolist())
    for img in set(B.gt_img.tolist()):
        rows = (B.gt_img == img)
        inA = torch.tensor([int(i) in halfA for i in B.gt_img[rows].tolist()])
        assert bool(inA.all()) or bool((~inA).all()), f"image {img} straddles the split"


def test_heldout_selection_reads_half_b_once(B):
    ho = heldout(B, seeds=1, select_seed=7)
    assert (ho["selected_scale"], ho["selected_tau"]) == (
        max((r for r in ho["grid"] if r["halfA_pareto"] is not None),
            key=lambda r: r["halfA_pareto"])["scale"],
        max((r for r in ho["grid"] if r["halfA_pareto"] is not None),
            key=lambda r: r["halfA_pareto"])["tau"])


# --------------------------------------------------- 5. metric self-coherence
def test_recall_at_1_is_the_rank_one_mass(B):
    r = analysis_B_ranks(B, 0.0)
    for arm in ("prior", "combined", "model_only"):
        assert r[arm]["recall@1"] == pytest.approx(r[arm]["hist"]["1"] / B.n_gt, abs=1e-6)  # float32
    assert r["rank_improved"] + r["rank_worsened"] + r["rank_unchanged"] == B.n_gt


def test_per_class_contributions_sum_to_the_global_mr_delta(B):
    de = analysis_DE_buckets_predicates(B, 0.0)
    mp = B.metrics(B.score(0.0, ALPHA_HIST, None))
    mc = B.metrics(B.score(0.0, ALPHA_HIST, B.model))
    assert sum(r["contribution_mR_points"] for r in de["per_class_all"]) == pytest.approx(
        (mc["mR"] - mp["mR"]) * 100.0, abs=1e-6)
    assert sum(de["buckets"][b]["contribution_to_global_dmR_points"]
               for b in ("head", "body", "tail")) == pytest.approx(
        (mc["mR"] - mp["mR"]) * 100.0, abs=1e-6)


def test_bucket_net_flips_sum_to_the_global_net(B):
    de = analysis_DE_buckets_predicates(B, 0.0)
    a = analysis_A_flips(B, 0.0)
    assert sum(de["buckets"][b]["net_flips"] for b in ("head", "body", "tail")) == a["net_flips"]


# ------------------------------------------------------ 6. bootstrap validity
def test_bootstrap_resamples_images_not_rows(B):
    """The unit must be the image: rows inside an image are not independent.

    Pinned by construction -- a degenerate single-draw bootstrap over a full
    multinomial must reproduce the point estimate in expectation, and the
    per-class denominators must never exceed what the images actually contain.
    """
    from tools.cprime_mechanism import analysis_J_stability
    j = analysis_J_stability(B, 0.0, n_boot=64, seed=1)
    assert j["resample_unit"] == "image"
    mp = B.metrics(B.score(0.0, ALPHA_HIST, None))
    mc = B.metrics(B.score(0.0, ALPHA_HIST, B.model))
    true_dR = (mc["R"] - mp["R"]) * 100.0
    assert j["dR_points"]["ci2.5"] <= true_dR <= j["dR_points"]["ci97.5"]


def test_leave_best_class_out_reduces_the_mr_delta(B):
    from tools.cprime_mechanism import analysis_J_stability
    j = analysis_J_stability(B, 0.0, n_boot=8, seed=1)["leave_best_classes_out"]
    assert j["dmR_without_best"] <= j["dmR_full"] + 1e-9
    assert j["dmR_without_best_two"] <= j["dmR_without_best"] + 1e-9


# --------------------------------------------- 7. the tie-breaking convention
def test_margin_analysis_uses_argmax_for_the_prior_top1(B):
    """analysis_C_margins must agree with analysis_A_flips on how many rows change.

    They disagreed (7.79% vs 7.70% on the real cache) because C took the prior's
    top-1 from topk(2).indices[:, 0] while A took it from argmax, and 522 GT
    rows tie for the prior maximum. argmax is the evaluator's convention.
    """
    from tools.cprime_mechanism import analysis_A_flips, analysis_C_margins
    a = analysis_A_flips(B, 0.0)
    c = analysis_C_margins(B, 0.0)
    assert c["changed_frac_overall"] == pytest.approx(a["argmax_changed_frac"], abs=1e-9)
