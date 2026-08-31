# Source-level forensics: where the visual signal enters, and where it dies

Written while the pre-registered fp32 control arm of experiment B was still
running. **Read-only analysis** — no research semantics were modified, no GPU
was used, and nothing in the running experiment was touched.

Companions: `docs/research_sessions/POST_B_SCIENTIFIC_REASSESSMENT.md`
(synthesis), `docs/research_sessions/POST_B_CLAIM_AUDIT.md` (claim audit).

Everything below is read from the source at commit
`a46fbada796ab066b5533eb736cf74ca5f5f10d8` and, where a number appears, from a
run recorded under `runs/`.

---

## 1. The data path, end to end

This is the **PredCls / GT-pairs** path — the only protocol every result in
this repository has used. Line numbers are `openvocab_rel/evals.py` unless
marked otherwise.

```
                       image (PIL)  +  GT boxes  +  GT object labels
                                    │
   ┌────────────────────────────────┴────────────────────────────────┐
   │                                                                 │
   │  A. VISUAL BRANCH                          B. SYMBOLIC BRANCH   │
   │                                                                 │
   │  CLIP ViT-L/14-336 @336px                  obj_labels (strings) │
   │  frozen backbone                                    │           │
   │        │                                            │           │
   │  visual_tokens, sub_feat, obj_feat                  │           │
   │        │                                            │           │
   │  relational_model.forward_pairs()  L429             │           │
   │    F.normalize(sub_feat)           L450   ◄── unit sphere       │
   │    F.normalize(obj_feat)           L451   ◄── unit sphere       │
   │    geom → Fourier → geom_mlp       L444-447                     │
   │    F.normalize(geom_proj)          L453   ◄── unit sphere       │
   │    fused = gate*sem + (1-gate)*geom L457-460                    │
   │    rel_feat = rel_seed(fused)      L488                         │
   │    rel_feat = out_norm(rel_feat)   L549   ◄── LayerNorm         │
   │        │                                            │           │
   │  score() = normalize(x) @ normalize(t).T  relational_model:971  │
   │        │                                            │           │
   │   text_logits ∈ [-1, +1]  ◄── RAW COSINE, NO LOGIT SCALE        │
   │        │                                            │           │
   │  _relation_predicate_logits()      L1235            │           │
   │    cls_logits (adaptive calib)     L1250            │           │
   │    _apply_eval_logit_adjustment    L1256  ── tau=0 ⇒ NO-OP      │
   │    text_norm = z-score(text_logits) L1260 ◄── ①  std := 1       │
   │    alpha = 0.0                     L1271                        │
   │    return 0.0*cls_norm + 1.0*text_norm                          │
   │        │                                            │           │
   │   visual term: mean 0, std 1 across the 51 predicates           │
   │        │                                            │           │
   └────────┼────────────────────────────────────────────┼───────────┘
            │                                            │
            │        _load_frequency_bias()  L1086       │
            │        pair_log_probs[s||o]    L1157  ◄────┘
            │        backoff: pair → subj⊕obj → subj|obj → global
            │                       │
            └──────────┬────────────┘
                       │
     _apply_frequency_bias()  L1178
        final = text_norm + 3.75 * log P(p | s, o)   ◄── ②  THE JOIN
                       │
     _mask_background_logits() L1773  ── background class → -10000
                       │
     softmax over 51                L1773  ◄── ③ global competition
                       │
     _make_triplet_predictions      ── ONE predicate per pair (argmax)
                       │                  ⇒ K is inert, R@50 == R@100
     _normalize_triplet_labels      L1847  ◄── ④ ALIAS MERGE, both sides
                       │
     _compute_pred_matches (label equality + IoU ≥ thresh)
                       │
     _recall_from_matches           ── R@K
     _compute_global_mr             L1980 ── mR@K = unweighted mean over
                       │                     classes with GT count > 0
     _predicate_buckets_from_counts L1310 ◄── ⑤ head/body/tail = top 20 %
                                              / middle / bottom 20 % BY GT
                                              COUNT ON THE EVAL SPLIT
```

### Where the visual signal can be suppressed — five named points

| # | Operation | File:line | Effect on visual signal |
|---|---|---|---|
| ① | `_normalize_eval_logits` per-row z-score | `evals.py:975` | Destroys the *magnitude* of visual confidence. Every pair's visual term is rescaled to std 1 regardless of how certain the encoder was. A confidently-seen `eating` and a coin-flip `on` contribute the same dynamic range. |
| ② | `final = visual + 3.75 · log P` | `evals.py:1178` | **The dominant suppression.** A unit-variance term is added to one scaled by 3.75 across the full range of a log-probability. Quantified in §2. |
| ③ | `softmax` over all 51 predicates | `evals.py:1773` | Flat global competition. A tail predicate must out-score all 50 alternatives simultaneously; there is no restricted or hierarchical decision. |
| ④ | `_normalize_triplet_labels` alias merge | `evals.py:1847`, map at `evals.py:430` | Merges `near`→`next to` and `wears`→`wearing`, both canonical VG150 classes. Changes the mR denominator from 50 to 48. Quantified in §3. |
| ⑤ | `_predicate_buckets_from_counts` | `evals.py:1310` | head/body/tail are recomputed **from the evaluated split's own GT counts**, so bucket membership is not fixed across runs of different N. |

Two further points, not suppressions but scale hazards:

- `score()` (`relational_model.py:971`) returns a **raw cosine with no learned
  logit scale**. CLIP's own `logit_scale` (≈100) is not applied here. Before
  ①, the visual term therefore lives in `[-1, +1]`; ① rescales it to std 1,
  which is what makes ② a fixed 3.75-to-1 fight rather than a tunable one.
- `_apply_eval_logit_adjustment` (`evals.py:1074`) is a **no-op in every run
  recorded here**: `eval_logit_adj_tau = -1.0` falls back to
  `logit_adj_tau = 0.0`, so the `tau <= 0.0` early return fires. The
  tail-adjustment machinery exists and was never active. VERIFIED from
  `runs/p5_model_vs_leakfree_prior/latest_metrics.json`.

---

## 2. Quantifying ② — how much swing would the visual term need?

MEASURED, `runs/p6_prior_dominance_margin/` (CPU, 7.8 s, no GPU, no model):

For each of the 38,053 GT pairs in the first 3,000 validation images, in the
same units the argmax sees:

- `margin_12 = 3.75 · (logP[top1] − logP[top2])` — swing needed to change the
  answer at all
- `margin_gt = 3.75 · (logP[top1] − logP[gt])` — swing needed to make it
  correct, over the 12,633 pairs the prior gets wrong

| percentile | `margin_12` | `margin_gt` (errors only) |
|---|---:|---:|
| p10 | 0.93 | 0.90 |
| p25 | 2.60 | 2.60 |
| **p50** | **5.08** | **4.44** |
| p75 | 7.99 | 7.88 |
| p90 | 11.34 | 11.43 |

The visual term is a difference of two unit-variance z-scores, so its swing
between any two predicates has std ≈ 1.41.

| swing budget | ≈ σ | pairs flippable | **prior errors addressable** |
|---:|---:|---:|---:|
| 1.0 | 0.7 | 10.5 % | 11.0 % |
| 2.0 | 1.4 | 19.5 % | 20.2 % |
| 3.0 | 2.1 | 31.5 % | **35.6 %** |
| 4.0 | 2.8 | 39.5 % | 43.4 % |
| 6.0 | 4.2 | 58.3 % | 61.6 % |
| 10.0 | 7.1 | 86.0 % | 85.5 % |

**MEASURED conclusion:** at `freq_bias_alpha = 3.75`, roughly **two thirds of
the prior's errors are mathematically unreachable** by the visual term, no
matter what the encoder sees. A 3-unit swing is already a 2.1 σ excursion in
the visual term's own distribution, and it reaches only 35.6 % of the errors.

This is a property of **the composition**, not of any model. It was computed
without loading a checkpoint.

**What it does and does not license.** It explains why the *model arm* (A′)
behaves as it does. It does **not** explain experiment B's negative, because
B used a different composition — prior at weight 1.0, appearance at λ up to
2.0, i.e. up to **7.5× more favourable to appearance** than α = 3.75 — and
still captured nothing. The two negatives have different mechanisms; see the
reassessment's §Phase IV.

---

## 3. Quantifying ④ — the alias merge makes A′ not like-for-like

**This is a new finding from this session.** VERIFIED at source, MEASURED on
data.

`_build_vg_aliases` (`evals.py:430`) contains, among object-noun merges:

```python
"near":  "next to",
"wears": "wearing",
```

Both `near` and `wears` are canonical members of the 50-predicate VG150
vocabulary (`datasets_vg150_clean/vocabulary/predicates.json`, SHA256
`76b75952…`). `eval_sgg_use_vg_aliases` defaults to **True**
(`config.py:473`), is **not** among the 14 flags the canary verifies
(`tools/verify_canary.py`), and is **not** listed in `docs/known_issues.md`.

`_normalize_triplet_labels` is applied to the GT at `evals.py:1757` **and to
the predictions** at `evals.py:1847`, so a `next to` prediction scores a hit on
a `near` ground truth, and the mR denominator drops to 48 classes.

`tools/frequency_prior_baseline.py` and `tools/decision_rule_probe.py` — which
produced arms 2, 2b and 3 of A′ — contain **no alias handling at all**
(grep-verified) and average over 50 classes.

### Confirmation that this is exactly what happened

MEASURED on the first 3,000 validation images (CPU count):

| predicate | GT count |
|---|---:|
| `near` | 640 |
| `next to` | 718 |
| **merged** | **1,358** |
| `wears` | 330 |
| `wearing` | 2,399 |
| **merged** | **2,729** |

`runs/p5_model_vs_leakfree_prior/latest_metrics.json` reports exactly
`next to = 1,358` and `wearing = 2,729` in its GT counts, and its
`per_predicate_R@50` has **48** entries. The mechanism is confirmed by exact
arithmetic, not inferred.

### The size and sign of the resulting bias

MEASURED, `runs/p6_alias_control_3000/` (CPU, 17 s). The same leak-free prior,
the same 3,000 images, scored under both schemes.
**Validation gate: the `raw50` arm reproduces the recorded arm-2 numbers to 10
decimal places** (`R@50 0.6680156624`, `mR@50 0.2197582807`), which is what
makes the `eval48` column trustworthy.

| scheme | τ | classes | R@50 | mR@50 |
|---|---:|---:|---:|---:|
| `raw50` (prior tool, arms 2/3) | 0.0 | 50 | 66.80 % | 21.98 % |
| **`eval48`** (evals.py, arm 1) | 0.0 | 48 | **67.93 %** | **22.68 %** |
| `raw50` | 0.1 | 50 | 66.16 % | 26.00 % |
| **`eval48`** | 0.1 | 48 | **67.31 %** | **26.86 %** |

Aliasing is worth **+1.13 R@50 and +0.71 mR@50 to the prior**, purely from the
scheme. Per-predicate under `raw50`: `near` R@50 11.88 % (n=640), `wears`
1.21 % (n=330) — two low-recall classes that the merge removes from the
denominator and folds into `next to` (24.65 %) and `wearing` (94.04 %).

### Consequence for the A′ headline

`docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` reports
`Δ = arm 1 − arm 2 = +0.10 R@50, −0.82 mR@50`, comparing a 48-class aliased
average against a 50-class unaliased one.

Putting both on the evaluator's own scheme:

| | R@50 | mR@50 |
|---|---:|---:|
| arm 1 — model + historical prior (48-class, aliased) | 66.90 | 21.16 |
| arm 2 — leak-free prior **on the same scheme** | **67.93** | **22.68** |
| **Δ, like-for-like** | **−1.03** | **−1.52** |
| arm 3 — τ=0.1 prior on the same scheme | 67.31 | 26.86 |

**The correction strengthens the recorded conclusion and does not change the
pre-registered verdict** (NEGLIGIBLE requires `Δ_mR < +1.0`; −1.52 qualifies,
as −0.82 did). Like-for-like, the model is behind the leak-free prior on
**both** metrics, not just mR@50, and the one-parameter recalibration beats it
by **+5.70 mR@50** rather than +4.84.

**Epistemic status.** The `eval48` prior number is a MEASUREMENT of the prior
tool's scoring under the evaluator's predicate merge. The like-for-like Δ is
therefore an **INFERENCE** with one residual gap: the prior was not re-scored
through `evals.py` itself, so any difference between the two tools' *matching*
implementations (beyond aliasing) is unbounded by this control. The prior
tool's exact reproduction of arm 2 at `raw50` makes a large residual unlikely,
but closing it costs ~1 minute of CPU and is listed as experiment **N1** in
the reassessment's matrix.

`docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` has deliberately **not** been edited.
Recording the correction here rather than rewriting the original is the
convention this repository already uses (see the `SUPERSEDED IN PART` header
on `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md`).

---

## 4. Where the visual signal enters, summarised

| Stage | Visual? | Notes |
|---|:---:|---|
| CLIP backbone | **yes** | frozen; ViT-L/14-336 |
| `sub_feat` / `obj_feat` | yes | L2-normalised crops |
| geometry Fourier features | no | box coordinates only |
| `fused_feat` gate | mixed | learned scalar gate between semantic and geometric |
| `rel_feat` → `score()` | yes | raw cosine, no logit scale |
| z-score ① | **magnitude destroyed** | direction retained |
| `+ 3.75 · log P` ② | **overwhelmed** | §2 |
| softmax ③ | flat 51-way | no candidate restriction |
| argmax → one triplet per pair | — | K inert under GT pairs |
| alias merge ④ | — | changes the metric, not the signal |
| mR bucketing ⑤ | — | recomputed per split |

**Places where tail classes are mathematically dominated:**

1. **② at α = 3.75.** A tail predicate whose prior log-prob is ~1.2 nats below
   the head predicate needs a 4.5-unit visual swing — a 3.2 σ event.
2. **③ flat softmax.** No mechanism restricts competition to plausible
   predicates; `walking on` competes against `on` with `on` holding 45.1 % of
   the prior's argmax mass (MEASURED, `runs/p5_arm2_trainprior_3000/`).
3. **The metric.** mR@50 averages over classes, so a +18-point gain on a
   22-instance class contributes 18/50 = 0.36 aggregate points, while a
   small slip on `on` (n≈13,900 GT in this subset) cancels it.

---

## 5. Reproducibility and code-quality audit

Read-only. No fix has been applied. "Safe to auto-fix" means *changes no
scientific semantics and is covered, or coverable, by a test*.

| # | Severity | File / mechanism | Proposed fix | Safe to auto-fix? |
|---|---|---|---|---|
| **1** | **P1 — protocol drift, silent** | `evals.py:430` merges `near`→`next to`, `wears`→`wearing`; on by default (`config.py:473`); the prior tools do not. Makes model-vs-prior comparisons unequal in the model's favour by +0.71 mR / +1.13 R (§3). | Do **not** change the alias map — it may be intentional for object nouns. Add `eval_sgg_use_vg_aliases` to the canary's checked flags; record the effective class count next to every mR@K; add a `--aliases` flag to the prior tools so arms can be matched. | **No** — changes reported numbers. Needs pre-registration. |
| **2** | P2 — reporting inconsistency | mR denominator is 48 in `evals.py` but 50 in `tools/frequency_prior_baseline.py`, and neither emits the count beside the metric. `n_predicate_classes_with_gt` exists only in the prior tool. | Emit `n_predicate_classes_in_mR` in `evals.py`'s task blocks. | Yes — additive field only. |
| **3** | P2 — inconsistent bucketing | `evals.py:1310` uses top-20 %/bottom-20 % **by GT count on the evaluated split**; `tools/appearance_probe.py:score` uses a fixed 15 / 20 / 15 split of the train counts. "tail mR" is not the same quantity in the two documents. | Name them differently in output (`tail_mR@50_evalsplit` vs `tail_mR_probe`), or share one definition. | Yes — renaming an output key, but it breaks downstream readers; do it deliberately. |
| **4** | P2 — bucket instability | Because ⑤ recomputes buckets from the evaluated split, head/body/tail membership can differ between an N=240, N=3,000 and full-split run of the *same* checkpoint. Any head/tail delta across different N is partly a bucketing artifact. | Freeze buckets from the train split and record them in the manifest. | No — changes reported numbers. |
| **5** | P3 — metadata gap | `tools/appearance_probe.py` records `preprocess_resolution: null`. transformers 5.x returns `SizeDict`, so the `isinstance(..., dict)` test fails. Verified separately that the true value is 336. | `getattr(proc.crop_size, "height", None)` fallback. | Yes — but **not** while B's fp32 arm is running; the file must stay byte-identical to its registered version until B is closed. |
| **6** | P3 — inert machinery | `_apply_eval_logit_adjustment` never fires in any recorded run (`eval_logit_adj_tau=-1.0` → `logit_adj_tau=0.0`). The τ result that *did* work was applied in `tools/decision_rule_probe.py`, a different code path, and has never been run inside `evals.py`. | None yet — but note that "τ works" is currently established only outside the evaluator. Wiring it in is experiment **N2**. | No — that is an experiment, not a fix. |
| **7** | P3 — silent fallbacks | 41 `except Exception` handlers across `evals.py` + `tools/`; five in `evals.py` return `None`/`pass`. `_load_frequency_bias` returns `None` on a malformed prior, and `_apply_frequency_bias` then silently returns unmodified logits — a run with a broken prior looks like a run with no prior, at exit 0. This is the P1 fallback already registered in `docs/known_issues.md`; the canary is the mitigation. | Log at WARNING and record `freq_bias_loaded: bool` in the metrics blob. | Yes — additive diagnostic field. |
| **8** | P3 — missing regression test | No test asserts which predicates the alias map merges, so #1 could change silently. | Add `tests/test_predicate_alias_map.py` pinning the exact merge set and asserting the resulting class count. | Yes — pure addition, no semantics changed. |
| **9** | P3 — stale documentation | `docs/APPEARANCE_PROBE_FINDINGS.md` §5 caveat 1 ("ViT-B/32… confirming on GPU is the single experiment that could overturn this verdict") is now discharged by B. `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §3 still says **PENDING**. `tools/appearance_probe.py`'s docstring reports B/32 numbers (λ=0.25, 0.8 %) that disagree with the findings doc (λ=0.50, 0.6 %). | Add pointers to the B result; reconcile or label the two B/32 records as two different runs. | Yes for pointers. **No** for the probe docstring while B is open. |
| **9b** | P3 — silent data-handling defect | `tools/appearance_probe.py:crop` can emit crops 3 px on a side; a `(3, W, 3)` array is channel-ambiguous and `transformers` assumes channels-first, silently transposing it. MEASURED: **21** object crops of 70,709 (0.03 %) have a side of exactly 3 px; 95 (0.15 %) have a side ≤ 4 px. Identical in both precision runs, so it cannot explain any result. | Drop or pad crops below a minimum side, or pass `input_data_format="channels_last"` explicitly. | **No** while B is open — `appearance_probe.py` must stay byte-identical to its registered version. Safe afterwards, and it changes ≤ 0.03 % of crops. |
| **10** | P3 — naming | `runs/` mixes phase prefixes (`p3c_`, `p3e_`, `p4_`, `p5_`, `p6_`) with no index. `p3c_historical_full_val_ZEROBUG` is preserved evidence but reads like a failed run. | Add `runs/README.md` mapping prefix → question → verdict. | Yes — documentation only. |

**Determinism:** the appearance probe is deterministic given a cache.
`torch.pca_lowrank` is stochastic but is seeded by `torch.manual_seed(0)` at
the top of `score()`, and the whole RNG stream is consumed in a fixed order.
VERIFIED empirically: `tools/appearance_probe_decidable.py`, a separate
process, reproduced all 24 primary fields bit-for-bit.

**Provenance:** every run in this session carries `command.txt`,
`provenance.json` (git SHA, dirty state, env, GPU, artifact hashes),
`stdout/stderr.log` and `result.json` via `tools/run_experiment.py`.
Three runs in this session are recorded with a **dirty** working tree
(`p6_appearance_probe_l14_336_decidable`, `p6_bench_clip_l14_336_fp32`,
`p6_alias_control_3000`, `p6_prior_dominance_margin`) because new untracked
diagnostic tools were present. The headline run
`p6_appearance_probe_l14_336` and the pre-registered fp32 control arm
`p6_appearance_probe_l14_336_fp32` were both launched from a **clean** tree.

**Precision control:** fp16 vs fp32 changes `captured_total` by 0.23 pp and
nothing else that matters — same verdict, same selected λ, bit-identical
baselines and ZERO arm. Details in `docs/APPEARANCE_PROBE_L14_RESULT.md` §7.

**Test suite:** `288 passed, 1 skipped` at HEAD (77.5 s, CPU), run during this
session. No test covers issues #1 or #8.
