# Pre-registration — full-validation confirmation of the candidate-restricted scorer

Status: PRE-REGISTERED. Committed before the GPU pass is launched.
Branch: research/architecture-breakthrough
Screening evidence: runs/p18, p21, p22 (3,000-image analysis set, CPU, cross-fit)

## What is being confirmed, and what is NOT

The screening result, with beta selected inside the training folds only
(runs/p22_scorer_nested, tau=0, k=5, 38,053 GT rows over 3,000 images):

| arm | R@50 | mR@50 | Pareto gap vs tau frontier | floor 66.5 |
|---|---|---|---|---|
| prior only (tau=0) | 66.802 | 21.976 | 0 (defines it) | ok |
| achieved additive C' | 67.474 | 22.837 | +0.861 | ok |
| prior_only learned | 65.651 | 26.354 | +0.059 | FAIL |
| shuffled-model null | 65.495 | 26.608 | +0.224 | FAIL |
| **full (prior + model)** | **66.673** | **25.161** | **+2.894** | **ok** |

Read: the two arms with no model information land ON the tau frontier, which is
what "everything is calibration" looks like. The arm carrying the model term is
the only one that clears the R@50 floor and the only one meaningfully off the
frontier.

This is a SCREENING result on 3,000 images and is labelled exploratory
everywhere it appears. It is not a headline and must not become one before this
confirmation runs.

## Why a GPU pass is justified

The C' cache covers 3,000 of the 10,401 validation images (`--eval_batches 250`
at `--batch_size 12`). The screening set therefore has a smaller denominator
than the protocol's, and cannot certify "all predicates present, exact
vocabulary, exact denominator". Nothing about the analysis needs the GPU; only
the cache extension does. One frozen forward pass, no training, no gradient.

## Protocol

- Identical to runs/p10_model_recalibration in every flag EXCEPT `--eval_batches`
  (250 -> 0/unbounded) and the output paths. The p10 command is reproduced
  verbatim from `runs/p10_model_recalibration/command.txt`.
- Checkpoint: checkpoints/demo_best/pure_best_adapt_light_mR50.pt, SHA256 verified.
- Prior: datasets_vg150_clean/frequency_prior_train.json (train-derived,
  leak-free). NOT the historical prior.
- Split: validation. No change to membership.
- The dump is validated by tools/validate_pair_dump.py before ANY analysis runs.
  If the cache fails validation, no number from it is reported.

## Analysis (fixed here, before the numbers exist)

Re-run, unchanged, on the full-split cache:
1. `cprime_oracle_ceiling` — coverage and the realizable ceiling.
2. `candidate_scorer_probe --nested` — the honest, operating-point-free estimate.
3. `candidate_scorer_probe --frontier` — the R/mR frontier vs tau's.

Same seed (0), same folds-by-image rule, same betas, same k=5, same tau=0.

## Primary criterion

Pareto gap of the `full` arm against the prior-only tau frontier, in mR points,
out-of-fold, subject to R@50 >= 66.5.

- **CONFIRMED**: full clears the floor, its Pareto gap > +1.5, and it exceeds
  both `prior_only` and `shuffled_model` by > +1.0 Pareto points.
- **WEAKENED**: full clears the floor and its gap is in [+0.861, +1.5] -- i.e.
  it still beats the additive arm but the screening magnitude did not survive.
- **REFUTED**: full fails the floor, or its gap <= +0.861 (the additive arm's),
  or it fails to separate from `shuffled_model` by > +1.0.

REFUTED closes the candidate-restricted direction. It will be reported as
prominently as CONFIRMED would be.

## Secondary (reported, not criteria)

R@50, mR@50, head/body/tail, per-predicate contribution, candidate coverage,
fraction of rows whose argmax changes, betas chosen per outer fold, and the full
operating-point separation table required by the directive's section 13.

## Compute budget and stop rule

Measured from p10: ~1.21 s/image after a ~36 s model load, and independent of
batch size (240-image pilots at bs=12/workers=4 and bs=48/workers=8 took 327 s
and 348 s). Full split therefore ~3.5 h.

Budget: 5 GPU-hours. Stop rule: if the dump fails `validate_pair_dump.py`, or if
throughput implies more than 5 h, the run is killed and the screening result
stands as exploratory rather than being confirmed on a partial cache.

A partial cache will NOT be silently analysed as though it were the full split.

---

# Addendum — the screening magnitude changed AFTER this was registered

Added 2026-09-01, while `runs/p24` was already executing. **No threshold above is
altered.** This records a fact that materially affects how the criterion should
be read, and hiding it would be worse than the awkwardness of recording it.

## What changed

The screening table at the top of this file is `runs/p22`: one partition of the
images into 5 folds. `runs/p25` repeats the identical nested procedure over 5
independent deterministic re-partitions:

| quantity | as registered (`p22`, salt 0) | resampled (`p25`, 5 partitions) |
|---|---|---|
| `full` Pareto gap | +2.894 | **+1.911 ± 1.056** (range +0.244…+2.941) |
| `full` clears R@50 floor | yes | **4 of 5 partitions** |
| `prior_only` Pareto gap | +0.059 | **−1.205 ± 0.793**, floor 0/5 |
| `shuffled_model` Pareto gap | +0.224 | **−1.137 ± 0.838**, floor 0/5 |
| `full − prior_only` | +2.835 | **min +1.904** across partitions |
| `full − shuffled_model` | +2.670 | **min +1.842** across partitions |

**+2.894 was a favourable draw.**

## Consequence for reading the criterion

The criterion has two halves and they are now known to be of very different
robustness:

- The **absolute** half — `gap > +1.5` — was calibrated against a
  single-partition +2.894. Against the resampled mean of +1.911 ± 1.056 that
  threshold sits well inside one standard deviation, so **CONFIRMED vs WEAKENED
  is close to a coin flip on screening evidence alone**. It is left in place
  because it was registered, but it should not be read as a strong test.
- The **separation** half — `full` exceeds both `prior_only` and
  `shuffled_model` by > +1.0 — is the robust one: the minimum observed
  separation across all five partitions is +1.842, comfortably clear of the
  threshold on every partition.

So if `runs/p24` returns WEAKENED on the absolute half while the separation half
holds, that is the outcome the resampled screening already predicts, and it
should be reported as *the magnitude was never as large as the first draw
suggested*, not as *the effect disappeared*.

If the separation half fails on the full split, that is a genuine refutation and
is reported as such.
