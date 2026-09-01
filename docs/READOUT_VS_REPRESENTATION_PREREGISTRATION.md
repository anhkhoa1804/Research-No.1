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
