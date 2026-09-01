# Result — is the model's contribution image-conditioned, or a pair prior?

Pre-registration: `docs/CANDIDATE_SCORER_PREREGISTRATION.md` addendum (commit
`6efb79d`), written and committed **before** the run.
Runs: `runs/p26_pair_matched_null` (exit 0, 740.0 s), verification
`runs/p26_pair_matched_null/verification.json`.
Compute: CPU only. Cache read-only. No GPU.

**Pre-registered verdict: E2 SUPPORTED.** Destroying image content while
preserving (subject, object) identity costs **nothing**.

---

## 1. Artifact and construction integrity

`tools/verify_pair_matched_null.py` re-derives the null from the cache rather
than trusting the probe's bookkeeping. All seven checks pass —
**NULL CONSTRUCTION VALID**:

| check | result |
|---|---|
| C1 sample size | 38,053 GT rows, 50 classes, 3,000 images — matches the cache exactly |
| C2 pair matching exact | **0** rows drew their model term from a different (subject, object) group |
| C3 permutation is not a no-op | **20,737 / 38,053 rows (54.5%)** received another row's model term |
| C4 conservatism | 17,172 distinct groups; **12,056 rows (31.7%)** are singletons and keep their real term |
| C5 permutable subsample not biased | permutable fraction head 0.548 / body 0.497 / tail 0.541 (spread 0.051) |
| C6 differs from `full` in model columns only | shapes identical, prior block bit-identical → a gap cannot be a capacity difference |
| C7 values are real | the null re-uses model-term entries verbatim; it synthesises nothing |

**Asymmetry, measured (C5).** The permutable subsample is close to unbiased
across head/body/tail (spread 0.051). It is *not* unbiased in prior margin:
permuted rows have mean margin **6.589** against **4.433** for kept rows. So the
null perturbs the *high*-margin rows slightly more, and the mechanism report
established the model only acts on *low*-margin rows. This biases the null
**towards the real arm** — the same direction as the singleton conservatism —
and therefore cannot manufacture the result below.

## 2. Result, resampled over 5 independent fold partitions

Identical procedure to `runs/p25`: nested beta selection, folds by image,
out-of-fold, tau=0, k=5.

| salt | `full` Pareto | `pair_matched_null` Pareto | `full − null` | `full` R@50 | null R@50 | floors |
|---|---|---|---|---|---|---|
| 0 | +1.879 | +1.932 | −0.053 | 66.657 | 66.612 | ok / ok |
| 1 | +1.802 | +2.135 | −0.334 | 66.857 | 66.652 | ok / ok |
| 2 | +2.941 | +2.625 | **+0.316** | 66.809 | 66.817 | ok / ok |
| 3 | +0.244 | +0.443 | −0.199 | 66.216 | 66.197 | FAIL / FAIL |
| 4 | +2.690 | +2.993 | −0.303 | 66.741 | 66.699 | ok / ok |
| **mean ± sd** | **+1.911 ± 1.056** | **+2.026 ± 0.977** | **−0.114 ± 0.265** | 66.656 ± 0.257 | 66.596 ± 0.236 | 4/5 both |

`full` exceeds the pair-matched null on **1 of 5** partitions. The null tracks
`full` partition-for-partition, *including* failing the R@50 floor on the same
partition (salt 3).

Contrast with the identity-destroying null: `pair_matched_null − shuffled_model`
= **+3.163** Pareto points.

> **Destroying pair identity costs +3.163 points. Destroying image content costs
> −0.114 — the wrong sign.**

## 3. Why the "AMBIGUOUS" clause does not apply

The pre-registration reserved AMBIGUOUS for `full ≈ null`, on the grounds that
45.5% of rows keep their real model term and could carry the signal. That
argument makes a directional prediction: under **E1**, `full` holds the true term
on *every* row while the null holds it on only 45.5%, so `full` must be **≥** the
null. Observed: the null is ahead on 4 of 5 partitions.

An effect-size bound follows. If image conditioning contributed *X* Pareto
points, permuting 54.5% of rows should cost ≈ 0.545·X. The measured cost is
**−0.114**. So *X* ≲ 0.2 points and its point estimate has the wrong sign.

**MEASURED.** This is E2, not ambiguity.

## 4. The mechanism, from two structural facts

**VERIFIED FACT — the "model term" is 100% the text branch.** The cache records
`ensemble_alpha = 0.0` and
`model_term = ea·norm(cls_logits)/cls_temp + (1−ea)·norm(text_logits)/text_temp`.
At ea = 0 the **visual classifier head contributes exactly zero**. Every number
the C′ line of work has attributed to "the model" is the CLIP-**text** scoring
path. (This is deliberate: `README` §2 records that the classifier head is
untrained in this checkpoint, reaching 1.41% / 0.94%.)

**MEASURED — the term is overwhelmingly pair-determined.** Variance
decomposition of the model term over 38,053 GT rows × 50 predicates:

| component | sum of squares | share |
|---|---|---|
| **between** (subject, object) groups | 1,336,450 | **86.87%** |
| **within** (subject, object) groups | 201,980 | **13.13%** |
| within-group, restricted to non-singleton groups | — | 18.43% |

Within-group variance is the *only* place image conditioning can live. It is
13.13% of the term, and `p26` measures directly that destroying it costs nothing.

These two facts explain the result rather than merely accompanying it: a text
branch scored against predicate embeddings, with the visual head switched off,
is close to a function of the pair's category labels.

## 5. What this does and does not establish

**ESTABLISHED.** At this operating point, with this frozen representation and
this scorer, the model term's entire usable contribution to predicate decisions
is explained by (subject, object) identity. The +3.16 points it adds over an
identity-destroying null is real and large; the image-conditioned residue is
≲ 0.2 points and not distinguishable from zero.

**NOT ESTABLISHED — and the distinction matters.** This does **not** show the
image is uninformative in principle. It shows that *this* term — the text branch
of *this* checkpoint, with the visual head at weight zero — carries no usable
image-conditioned predicate signal. A trained visual classifier head, or a
different representation, is untested by this experiment. `p26` closes the
current branch, not the question.

**Also not established:** that 13.13% within-group variance is noise. It may
carry signal that a linear candidate-restricted scorer cannot extract.

## 6. Decision

**Close the visual-architecture branch as currently posed.** The quantity the
project has been trying to convert into long-tail decisions is a pair prior
expressed in text-embedding space, not visual evidence. Building any architecture
to better exploit it would be building a second frequency prior.
