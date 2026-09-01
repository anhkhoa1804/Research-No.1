# Pre-registration — is the weak grounding a READOUT failure or a REPRESENTATION failure?

Status: **PRE-REGISTERED.** Committed before the GPU pass is launched and before
any `rel_feat` exists on disk.
Branch: `research/architecture-breakthrough`
Run: `p36_relfeat_cache` (GPU) → `p37_readout_vs_representation` (CPU)

---

## 1. What is already established

`runs/p33` / `runs/p35` measure Within-Pair Relational Discrimination (WPRD), a
metric that is prior-free by construction: within one (subject, object) group
the train-derived prior is constant to 9.4e-05, so it cancels exactly in the
double difference, along with any per-class calibration, any tau and any global
temperature. The prior control reads **exactly 0.5000 in every stratum**.

| arm | WPRD macro | 95% CI |
|---|---|---|
| text head (evaluated at `ensemble_alpha=0`) | 0.5542 | [0.5495, 0.5592] |
| classifier head (stored, **discarded**) | 0.5728 | [0.5681, 0.5779] |
| prior (must be 0.5) | 0.5000 | [0.5000, 0.5000] |
| random null | 0.5046 | — |

So image-conditioned relational signal **exists** and is **weak**, and the
checkpoint runs the weaker of its two readouts. Both heads are functions of the
same 768-d image-derived `rel_feat`:

```
text_logits = normalize(rel_feat) @ normalize(pred_emb).T     # a cosine
cls_logits  = predicate_classifier(rel_feat)                  # a linear head
```

## 2. The question

> Is WPRD ≈ 0.57 the ceiling of the **representation**, or only of the two
> **readouts** that happen to be attached to it?

This is the question that decides whether a successor to PURE is justified and
what it would have to change. A readout failure is cheap to fix and needs no
retraining. A representation failure means the encoder never learned relational
structure, and no readout will recover it.

## 3. The GPU pass (`p36`)

One frozen forward pass over the full 10,401-image validation split, identical
to `runs/p24` in every flag except `--eval_sgg_dump_rel_feat true` and the
output paths. No training, no gradient. Adds to the dump:

- `rel_feat`: 132,556 x 768 fp16 (~204 MB measured in pilot)
- `pred_emb`: 51 x 768 fp32, stored once

Pilot (24 images) confirmed: key present, dim 768, no NaN, `missing_rel_feat=0`,
and `text_logits` in [0.045, 0.278] as cosines must be.

**Budget 5 GPU-hours** (p24 took 3 h 27 m). Stop rule: if throughput implies
> 5 h, kill and report; no partial cache is analysed as though it were full.
GPU policy: checked idle (3 MiB, 0%, no compute apps) immediately before launch.

## 4. The CPU analysis (`p37`), fixed here

All probes are **cross-fitted over the same 5 image-level folds, seed 0** used
by every other analysis in this project, and are evaluated by WPRD on held-out
rows only. Nothing is selected on the evaluation rows.

| arm | what it is |
|---|---|
| `R1_text` | text head (reference, 0.5542) |
| `R2_cls` | classifier head (reference, 0.5728) |
| `R3_linear` | cross-fitted linear probe `rel_feat` → 50 predicates |
| `R4_mlp` | cross-fitted 1-hidden-layer MLP probe on `rel_feat` |
| `R5_residual` | probe on `rel_feat` after removing its (subject,object) group mean — grounding that is not pair identity |
| `R6_shuffled` | `R3` with labels shuffled — **must** read 0.5 |
| `R7_prior` | the prior — **must** read 0.5 |

Reported alongside, not criteria: the collapse measure — R² of a linear map from
(subject one-hot, object one-hot) to `rel_feat`, i.e. how much of the
representation is pair identity; and the Gram matrix of `pred_emb` (readout
geometry).

## 5. Primary criterion

Let `P* = max(R3_linear, R4_mlp)` (held-out WPRD macro) and `C = R2_cls`, the
better existing readout, measured on the same rows.

- **READOUT-LIMITED**: `P* >= C + 0.03`. The representation holds materially
  more than either head extracts. A readout-only successor is justified and
  needs no retraining.
- **REPRESENTATION-LIMITED**: `P* < C + 0.01`. The heads already extract
  essentially everything. A readout-only successor is **not** justified; the
  bottleneck is what the encoder learned.
- **INTERMEDIATE**: `C + 0.01 <= P* < C + 0.03`. Reported as such, with no
  successor recommended on this evidence alone.

Thresholds are anchored to quantities already measured, not chosen for this run:
the text→classifier gap is **0.0186** and that gap was large enough to change
the operating point in `runs/p34`, so `0.03` is "clearly more than the gap that
already mattered"; the CI half-width on these estimates is ~0.005, so `0.01` is
two half-widths, the smallest difference this instrument can resolve.

## 6. Validity gates — any failure voids the run

| gate | requirement |
|---|---|
| V1 | `missing_rel_feat == 0` and no NaN in `rel_feat` |
| V2 | recomputing `normalize(rel_feat) @ normalize(pred_emb).T` reproduces the stored `text_logits` to < 1e-2 (fp16 storage tolerance) |
| V3 | `R6_shuffled` and `R7_prior` both within [0.49, 0.51] |
| V4 | the new cache reproduces `p24`'s R@50/mR@50 exactly — same checkpoint, same split, same flags |
| V5 | `tools/validate_pair_dump.py` passes 12/12 as it did for `p24` |

V2 is the load-bearing one: it proves the stored feature is the tensor the
evaluated head actually read, not some other activation.

## 7. Interpretation rules, fixed in advance

- **READOUT-LIMITED** ⇒ design the smallest readout-only intervention that
  captures the gap, pre-register it, and test it on the cache. Still no
  retraining, still no new architecture.
- **REPRESENTATION-LIMITED** ⇒ the diagnosis is that PURE's relational encoder
  did not learn image-conditioned relational structure beyond object identity.
  That is a *training-objective* finding, and it is the strongest form of the
  diagnosis paper. It does **not** by itself justify building a bigger model.
- Either way, **no result here licenses claiming PURE "sees relations"**. WPRD
  0.55–0.60 against a 0.5 floor is weak in absolute terms and that framing
  survives every outcome below.

---

# Addendum — a geometry reference arm, added before `p37`'s numbers exist

Added 2026-09-01 while `runs/p36` was still executing, after `runs/p38`/`p39`
produced a result that changes what `p37` should be compared against.
**No threshold in the pre-registration above is altered.** This adds a
*reference arm and a second, clearly-labelled criterion*; the primary criterion
in §5 stands exactly as registered and will be reported either way.

## What changed

`runs/p38`/`p39` fit a probe on **19 scale-invariant box-geometry features** —
no pixels, no `rel_feat`, no predicate embeddings — and scored it with WPRD:

| arm | WPRD macro | 95% CI |
|---|---|---|
| **geometry, TRAIN-FITTED** (no validation fit at all) | **0.5961** | [0.5921, 0.6014] |
| geometry, cross-fitted on validation folds | 0.5883 | [0.5837, 0.5938] |
| classifier head (discarded) | 0.5728 | [0.5681, 0.5779] |
| text head (**evaluated**) | 0.5542 | [0.5495, 0.5592] |
| geometry with SHUFFLED labels | 0.4872 | [0.4820, 0.4926] |
| prior | 0.5000 | [0.5000, 0.5000] |

Paired over cells, the trained checkpoint is **behind** train-fitted geometry:
text **−0.0419** [−0.0489, −0.0354], classifier **−0.0232** [−0.0292, −0.0163].
Both intervals exclude zero.

The train-fitted probe is the fair comparison: it sees the same training split
the model saw and the same validation split the model is scored on, and no
validation statistic enters its fit. It also *beats* the cross-fitted version,
so the result is not an artifact of fitting on validation.

## Why this changes the right question for `p37`

The §5 criterion asks whether `rel_feat` holds more than its own readouts
extract. That is still worth answering. But the sharper question is now:

> Does `rel_feat` hold anything **beyond box geometry** at all?

Because if the answer is no, then PURE's visual encoder has learned spatial
layout and nothing else, and "readout vs representation" is a question about
how to better extract a signal that is *already inferior to two rectangles*.

## Secondary criterion (additional, does not replace §5)

Let `P*` be the best held-out probe on `rel_feat` (as defined in §4) and
`G = 0.5961`, the train-fitted geometry probe.

- **BEYOND GEOMETRY**: `P* >= G + 0.02`. The representation carries relational
  evidence that boxes do not. A successor has something real to build on.
- **GEOMETRY-EQUIVALENT**: `|P* − G| < 0.02`. The encoder's relational content
  is spatial layout. This is the strongest form of the diagnosis and it argues
  **against** a successor built on this representation.
- **BELOW GEOMETRY**: `P* <= G − 0.02`. The encoder is actively worse than
  boxes, and the finding is about the training objective, not the architecture.

`0.02` is chosen as slightly larger than the classifier-vs-text gap
(0.0186) that was already large enough to move the operating point in `p34`,
and roughly four CI half-widths on these estimates.

A further arm is added to §4 for this comparison, and is reported whatever the
outcome:

| arm | what it is |
|---|---|
| `R8_geom` | the train-fitted geometry probe (reference, 0.5961) |
| `R9_relfeat_plus_geom` | probe on `[rel_feat, geometry]` — does the pair beat geometry alone? |

`R9` is the direct test of incremental value: if `rel_feat` adds nothing on top
of geometry, `R9 ≈ R8`.
