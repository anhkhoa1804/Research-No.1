# Experiment matrix — objective alignment before architecture

The plan for the first serious GCP training run, and the criteria that make
its result interpretable.

**This phase deliberately changes no model code.** The question it answers is
prior to any architecture question: *does PURE learn anything beyond label
co-occurrence once its objective is aligned with what the evaluator scores?*
Until that is answered, no architecture comparison is meaningful, because the
current headline numbers are ~97 % reproducible by a lookup table.

---

## 1. The control that reframes everything

`VERIFIED FROM EXPERIMENT` — `python tools/frequency_prior_baseline.py`, full
10,401-image validation split, GT-pairs protocol, **model contributing exactly
zero**:

| System | R@50 | mR@50 |
|---|---:|---:|
| **Frequency prior alone** | **64.37 %** | **20.30 %** |
| Historical claim (model + same prior) | 67.09 % | 22.64 % |
| Implied model contribution | +2.72 | +2.34 |

Supporting controls: the global predicate marginal alone (≈ always predict
`on`) gives 36.29 % / 2.00 %, so `(subject, object)` conditioning is worth
+28.09 / +18.30 by itself. The one-predicate-per-pair protocol ceiling is
96.74 %, so the protocol is not the limiter. α is inert in this regime
(identical metrics at 0.5 / 1.0 / 3.75 / 10.0).

**Every number this project reports must be a delta over 64.37 / 20.30.**

---

## 2. Three conceptual states — never conflate them

| | What it is | Where it lives | May it change? |
|---|---|---|---|
| **A — HISTORICAL** | recovered checkpoint + its compatibility config | `checkpoints/demo_best/` | **Never.** Immutable control. |
| **B — FROZEN BASELINE** | current source implementation | `main` @ `924eac58` | Only by explicit, committed change |
| **C — OBJECTIVE-ALIGNED** | B with the objective repaired | `research/pre-training-architecture` | This phase's work |

C changes **no model code**. `tests/test_objective_aligned_protocol.py`
asserts the architecture flags are byte-identical to B, so C→B comparisons
stay objective ablations rather than silently becoming architecture changes.

---

## 3. What is wrong with the current objective

`VERIFIED FROM CODE`.

**3.1 The evaluated head is not the trained head.** The evaluator scores the
predicate classifier (`score_mode=ensemble`), but the historical run used
`lr=2e-6` and left that head at/near random initialisation (row-norms 0.5752
vs 0.5796 for a fresh `nn.Linear`).

**3.2 The model never learned "no relation".** Shipped scripts train with
`use_all_pairs=false`, so CE only ever saw GT-positive pairs. The background
class exists (`predicate_classifier_classes=51`; negatives are labelled
`relation`) but nothing was ever assigned to it.

**3.3 Four imbalance corrections, one of them pointing backwards.** Training
could stack three corrections for the same imbalance —

| Mechanism | Config | Direction |
|---|---|---|
| Inverse-frequency class weights | `predicate_ce_weight_power=0.5`, `max_weight=5.0` | toward tail |
| Focal loss | `predicate_ce_loss="focal"`, `gamma=1.5` | toward hard examples |
| Logit adjustment | `tail_logit_adjustment_*` (off by default) | toward tail |

— while **evaluation applies a fourth in the opposite direction**: the
frequency prior at α=3.75 pushes predictions back *toward* the head classes,
and outweighs the model roughly 17:1 (model logits are layer-normed to ±2;
`3.75 × log-prior` spans ≈ 34).

The training objective de-biases; the evaluation re-biases; and the reported
metric is dominated by the re-biasing. That is the incoherence this run
removes.

---

## 4. The matrix

All arms: PredCls / GT pairs, `datasets_vg150_clean`, seed 0, identical
architecture, in-training eval raw and uncalibrated.

```bash
ARM=baseline  bash scripts/train/train_objective_aligned.sh
ARM=logit_adj bash scripts/train/train_objective_aligned.sh
ARM=none      bash scripts/train/train_objective_aligned.sh
```

| # | Arm | Corrections active | Question it answers |
|---|---|---|---|
| **C0** | `none` | — | How much does the representation alone do? Lower bound. |
| **C1** | `baseline` | focal + inv-freq weights | Does simply *training the head properly* close the gap? Isolates the fix from the correction. |
| **C2** | `logit_adj` | logit adjustment only | Does one principled correction beat the ad-hoc stack? |

Plus two zero-cost controls that require no training:

| # | Control | Command |
|---|---|---|
| **P0** | Frequency prior alone | `python tools/frequency_prior_baseline.py` |
| **P1** | Global marginal alone | `python tools/frequency_prior_baseline.py --mode global` |

And the historical reproduction (A), which must be run **with P0 alongside**.

### Ablation axes deferred until C0–C2 report

`use_all_pairs` on/off · `negative_pair_ratio` · `tau ∈ {0.5, 1.0, 1.5}` ·
`lambda_predicate_ce` · asymmetric pair fusion · text-vs-classifier scoring.
Running these before the main arms would confound the answer.

---

## 5. Metrics to report — always in this order

For every arm, report **raw first, calibrated second, never mixed**:

1. `R@20/50/100`, `mR@20/50/100` — headline fields. Note these are *top-1
   predicate accuracy* under GT pairs, and K is inert.
2. `multi_R@K`, `multi_mR@K` (`predcls_multi`) — **the literature-comparable
   numbers.** Compare these, not the headline fields, with published VG150
   results.
3. `image_mean_R@K` — the per-image-averaged variant most papers report.
4. `head/body/tail mR@50` — where any long-tail gain actually lands.
5. `prop@K`, `gt_pair_recall@K` — expected ≈ 1.0 under GT pairs; a drop means
   something broke.
6. Parameter count, VRAM, step time, throughput.
7. **Δ vs P0** for every one of the above.

---

## 6. Success and failure criteria — commit to these before seeing results

Stated in advance so the outcome cannot be rationalised afterwards.

**The run SUCCEEDS if** any arm's *raw, uncalibrated* `mR@50` exceeds the
frequency prior's **20.30 %** by a margin larger than seed variance, while
`R@50` is not catastrophically below 64.37 %. That would be the first
evidence PURE learns something a lookup table does not.

**The run is INFORMATIVE BUT NEGATIVE if** raw `mR@50` lands at 20–21 % —
i.e. the model reproduces co-occurrence statistics and nothing more. This is
a publishable finding about calibrated SGG evaluation and should be reported
as such, not buried.

**The run FAILS if** raw metrics collapse (`R@50` < 30 %) or training
diverges — an implementation problem, not a research result.

**Explicitly NOT a success criterion:** a *calibrated* number near 67 / 22.6.
That is reachable by the prior alone and is evidence of nothing.

---

## 7. Sequencing

1. `python tools/frequency_prior_baseline.py --ceiling --out runs/control_p0/metrics.json` — free, do it first
2. `python tools/gcp_preflight.py --strict`
3. `bash scripts/eval/eval_historical_checkpoint.sh --canary` — validates the whole path on 2 batches
4. **A** — historical reproduction, reported against P0
5. **C1** (`baseline`) — the objective fix in isolation
6. **C2** (`logit_adj`) and **C0** (`none`)
7. Only if some arm beats P0: begin the architecture phase, with C as its baseline

Steps 1–3 cost minutes. Do not start step 5 before step 3 passes.

---

## 8. What must not change

- `checkpoints/demo_best/*` — immutable historical evidence
- `datasets_vg150_clean/vocabulary/predicates.json` ordering
- the existing `R@K`/`mR@K` field definitions — `predcls_multi` is additive
  precisely so historical comparability survives
- the architecture flags asserted in `tests/test_objective_aligned_protocol.py`
- `main` — the frozen reproducibility baseline

## 9. Open risks

- **The historical protocol has five `UNKNOWN`s** (split, sample size,
  pooled-vs-image-mean, resolution, whether `FREQ_BIAS_ENABLED` was set). A
  gap between A and 67.09/22.64 has innocent explanations before "the model
  is worse than claimed".
- **`_load_frequency_bias` fails silently** on six conditions
  (`docs/known_issues.md`, P1). Preflight gates it; the evaluator does not.
- **No GPU measurement exists yet** for any arm — VRAM, step time and
  throughput are unmeasured, so wall-clock estimates are `UNKNOWN`.
- **`use_all_pairs=true` is untested at scale in this repo.** It changes the
  training data distribution and pair count per image; watch memory on the
  first run and reduce `max_pairs` if needed.
