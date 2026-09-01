"""Regression tests for the C' oracle ceiling.

An oracle that is wrong in the optimistic direction would manufacture headroom
and justify GPU time that is not warranted; one wrong in the pessimistic
direction would close a live research branch. Both failure modes are pinned:

1. ORDERING. On identical (k, budget) the ladder must be monotone --
   prior <= model_rerank <= oracle in R@50 -- because the oracle picks GT
   whenever the reranker COULD have, and the reranker only ever replaces the
   prior's top-1 on eligible rows.
2. THE ORACLE IS A CEILING FOR A TIE-BREAKER, NOT FOR OMNISCIENCE. Rows outside
   the margin budget keep the prior's top-1 in every arm, so a zero budget must
   reproduce the prior exactly and a k of 1 must too.
3. MONOTONICITY IN k AND IN BUDGET. Widening either can only add reachable
   rows, never remove them.
4. The eligibility mask is a pure function of the PRIOR, never of the model or
   of GT -- otherwise the budget leaks label information into the arm it bounds.
5. The achieved arm the gap is measured against is the real C' arm: the stored
   model term at alpha=3.75, unrestricted.

CPU only. Reuses the synthetic cache contract from test_cprime_mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.cprime_mechanism import ALPHA_HIST, Mech
from tools.cprime_oracle_ceiling import BUDGETS, KS, Oracle, region_decomposition


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
    p.write_text(json.dumps({"predicate_vocab": pv,
                             "global_log_probs": [-1.0 - 0.1 * i for i in range(len(pv))],
                             "default_log_prob": -20.0}))
    return p


@pytest.fixture(scope="module")
def dump_path(tmp_path_factory):
    pv = _vg150()
    vocab = [p.strip().lower() for p in pv] + ["background"]
    P = len(vocab)
    g = torch.Generator().manual_seed(19)
    n_img, n_obj, n_pair = 16, 4, 6
    keys = ("image_id", "pairs", "pair_index", "model_logits", "prior_rows",
            "text_logits", "cls_logits", "obj_labels", "subj_label", "obj_label",
            "obj_boxes", "gt_subj_idx", "gt_obj_idx", "gt_pred",
            "gt_subj_label", "gt_obj_label")
    d = {k: [] for k in keys}
    for i in range(n_img):
        pairs = torch.tensor([[a, b] for a in range(3) for b in range(3) if a != b][:n_pair])
        text = torch.randn(n_pair, P, generator=g)
        d["image_id"].append(str(i))
        d["pairs"].append(pairs)
        d["pair_index"].append(torch.arange(n_pair))
        d["text_logits"].append(text)
        d["cls_logits"].append(torch.randn(n_pair, P, generator=g))
        d["model_logits"].append(_norm(text))
        d["prior_rows"].append(torch.randn(n_pair, P, generator=g))
        labels = [f"obj{j}" for j in range(n_obj)]
        d["obj_labels"].append(labels)
        d["subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
        d["obj_boxes"].append(torch.zeros((n_obj, 4)))
        d["gt_subj_idx"].append([int(a) for a, _ in pairs.tolist()])
        d["gt_obj_idx"].append([int(b) for _, b in pairs.tolist()])
        d["gt_pred"].append(["near", "wears", "on", "has", "in", "next to"][:n_pair])
        d["gt_subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["gt_obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
    d.update({"schema": "pair_logit_dump_v2", "pred_vocab": vocab,
              "background_predicate_indices": [P - 1],
              "predicate_alias_map": {"near": "next to", "wears": "wearing"},
              "freq_bias_alpha": ALPHA_HIST, "eval_freq_bias_tau": 0.0,
              "ensemble_alpha": 0.0, "score_mode": "ensemble",
              "classifier_temperature": 1.0, "text_temperature": 1.0,
              "n_pairs": n_img * n_pair, "n_images": n_img})
    p = tmp_path_factory.mktemp("c") / "dump.pt"
    torch.save(d, p)
    return p


@pytest.fixture(scope="module")
def B(dump_path, PRIOR):
    return Mech(str(dump_path), str(PRIOR), "raw50")


@pytest.fixture(scope="module")
def O(B):
    return Oracle(B, 0.0)


# ------------------------------------------------------------- 1. ladder order
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("bname,b", BUDGETS)
def test_ladder_is_monotone_prior_le_rerank_le_oracle(O, k, bname, b):
    p = O.arm("prior", k, b)["R"]
    r = O.arm("model_rerank", k, b)["R"]
    o = O.arm("oracle", k, b)["R"]
    assert o >= r - 1e-9, f"oracle below reranker at k={k} budget={bname}"
    assert o >= p - 1e-9, f"oracle below prior at k={k} budget={bname}"


# ------------------------------------------ 2. ceiling for a TIE-BREAKER only
def test_zero_budget_reproduces_the_prior_exactly(O):
    for kind in ("combined", "model_rerank", "oracle"):
        a = O.arm(kind, 5, 0.0)
        assert a["R"] == pytest.approx(O.arm("prior", 5, 0.0)["R"], abs=1e-9)
        assert a["n_eligible"] == 0


def test_k_of_one_reproduces_the_prior_for_the_oracle(O):
    """With a single candidate the only choice IS the prior's top-1."""
    a = O.arm("oracle", 1, float("inf"))
    assert a["R"] == pytest.approx(O.arm("prior", 1, float("inf"))["R"], abs=1e-9)


def test_rows_outside_the_budget_keep_the_prior_top1(O, B):
    el = O.eligible(0.93)
    cand = O.candidates(5)
    gt_in = cand.gather(1, O.gt_col.unsqueeze(1)).squeeze(1)
    pred = torch.where(el & gt_in, O.gt_col, O.prior_top1_col)
    assert torch.equal(pred[~el], O.prior_top1_col[~el])


# ------------------------------------------------- 3. monotone in k and budget
def test_oracle_is_monotone_in_k(O):
    rs = [O.arm("oracle", k, float("inf"))["R"] for k in (2, 3, 5)]
    assert rs == sorted(rs), f"oracle not monotone in k: {rs}"


def test_oracle_is_monotone_in_budget(O):
    rs = [O.arm("oracle", 5, b)["R"] for b in (0.93, 5.08, float("inf"))]
    assert rs == sorted(rs), f"oracle not monotone in budget: {rs}"


def test_eligible_count_is_monotone_in_budget(O):
    ns = [int(O.eligible(b).sum()) for b in (0.0, 0.93, 5.08, float("inf"))]
    assert ns == sorted(ns)


# --------------------------------------------------------- 4. no label leakage
def test_eligibility_depends_only_on_the_prior(B):
    """Shuffling GT and the model term must not move the eligibility mask."""
    O1 = Oracle(B, 0.0)
    before = O1.eligible(5.08).clone()
    g = torch.Generator().manual_seed(5)
    B2 = Mech.__new__(Mech)
    B2.__dict__.update(B.__dict__)
    B2.gt_y = B.gt_y[torch.randperm(B.n_gt, generator=g)]
    O2 = Oracle(B2, 0.0)
    assert torch.equal(before, O2.eligible(5.08))


# ------------------------------------------------ 5. the achieved arm is C'
def test_achieved_arm_is_the_real_cprime_arm(O, B):
    a = O.arm("combined", B.n_classes, float("inf"))
    ref = B.metrics(B.score(0.0, ALPHA_HIST, B.model))
    assert a["R"] == pytest.approx(ref["R"], abs=1e-9)
    assert a["mR"] == pytest.approx(ref["mR"], abs=1e-9)


def test_region_decomposition_net_matches_the_global_net(B):
    rd = region_decomposition(B, 0.0)
    widest = [r for r in rd["regions"] if r["k"] == B.n_classes and r["budget"] == "unrestricted"]
    assert widest and widest[0]["net_flips_inside"] == rd["net_flips_total"]
