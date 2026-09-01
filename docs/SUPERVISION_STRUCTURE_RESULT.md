# `runs/p50` — VG150's within-pair supervision, and the dissociation it exposes

The mechanism question: *a model can only learn to distinguish predicate `a`
from `b` within a fixed (s,o) group if training data contains rows of **both**
for the **same** group.* How many such comparisons exist?

---

## 1. Supply is extraordinarily skewed, and stable across splits

Within-pair contrastive comparisons available (Σ over `(group, {a,b})` cells of
`n_a · n_b`), TRAIN split:

| buckets | comparisons | share | val | test |
|---|---|---|---|---|
| **head–head** | **60,311,041** | **80.64%** | 81.57% | 80.26% |
| body–head | 9,155,982 | 12.24% | 12.25% | 13.64% |
| head–tail | 4,712,947 | 6.30% | 5.35% | 5.25% |
| body–body | 394,787 | 0.53% | 0.53% | 0.57% |
| body–tail | 188,268 | 0.25% | 0.26% | 0.25% |
| **tail–tail** | **31,699** | **0.04%** | 0.04% | 0.02% |

**A 1,900× imbalance between head–head and tail–tail**, reproduced to within
0.02 percentage points on all three splits. This is structural, not sampling.

Supporting structure (TRAIN / val / test):

| | train | val | test |
|---|---|---|---|
| groups | 212,981 | 45,607 | 46,476 |
| singleton **groups** | 65.3% | 67.8% | 68.4% |
| rows in singleton groups | 13.3% | 23.3% | 24.0% |
| P(group ≥2 distinct predicates) | **19.0%** | 16.5% | 15.9% |
| P(group ≥3 distinct predicates) | 7.2% | 5.0% | 4.9% |
| rows in decidable groups | 74.8% | 56.9% | 55.8% |
| mean within-group entropy | 0.4209 | 0.3788 | 0.3738 |

Train has **more** within-pair structure per row than validation (74.8% vs
56.9%, mean group size 4.91 vs 2.91), so the scarcity is not an artifact of
evaluating on a thinner split.

## 2. The dissociation — this is the actual finding

Does supply predict PURE's within-pair discrimination, bucket by bucket?

| buckets | train comparisons | text-head WPRD | classifier-head WPRD |
|---|---|---|---|
| head–head | 60,311,041 | 0.5612 | 0.5701 |
| head–body | 9,155,982 | 0.5485 | 0.5823 |
| head–tail | 4,712,947 | 0.5273 | 0.5471 |
| body–body | 394,787 | **0.5746** | **0.6167** |
| body–tail | 188,268 | 0.5230 | 0.5918 |
| tail–tail | 31,699 | 0.5126 | 0.5677 |

```
Spearman(supervision, TEXT-head WPRD)       = +0.657    (n=6)
Spearman(supervision, CLASSIFIER-head WPRD) = -0.086    (n=6)
Pearson(log10 supervision, TEXT WPRD)       = +0.518
```

**The evaluated head tracks supply. The discarded head does not.**

The classifier head reaches **0.6167** on body–body and **0.5918** on body–tail
— buckets holding **0.53%** and **0.25%** of all within-pair supervision. If
scarcity were the binding constraint, *no* readout could do that.

**So supervision scarcity is NOT a sufficient explanation.** It is a real and
severe property of VG150, and it constrains the head the checkpoint runs, but a
different readout over the *same* features and the *same* data partly escapes it.

## 3. The hypothesis this points to — testable, and `p36` already stores what it needs

The two heads differ in exactly one structural way:

```
text_logits = normalize(rel_feat) · normalize(pred_emb)ᵀ   # FIXED embedding geometry
cls_logits  = predicate_classifier(rel_feat)               # LEARNED map
```

The text head's ability to separate predicates `a` and `b` is bounded by the
angle between `pred_emb[a]` and `pred_emb[b]`, which is **fixed CLIP text
geometry and never trained**. If two rare predicates have near-collinear text
embeddings, that readout cannot separate them *no matter what `rel_feat`
contains or how much supervision exists*. A learned classifier has no such
constraint.

**HYP (not yet tested):** the text head's supply-tracking is partly an artifact
of CLIP predicate-embedding geometry, and its tail failure is a *readout
geometry* failure rather than purely a supervision failure.

`runs/p36` stores `pred_emb` (51×768) precisely so this can be checked: compute
the Gram matrix and test whether low-WPRD predicate pairs are the near-collinear
ones. That is now a **first-class arm of `p37`**, not a footnote.

## 4. Statistical honesty

- **n = 6 buckets.** Nothing here is significant at that size.
- The `+1.000` Spearman for the text head **excluding body–body** is a
  **post-hoc exclusion** and is reported only to locate the single outlier. It is
  **not** a confirmatory test and must not be quoted as one.
- Supply is measured on TRAIN; WPRD on VALIDATION. That is the right direction
  (supervision precedes behaviour) but is still an association across six points.
- **No causal claim.** This measures what supervision exists. `p48`'s failed
  contrastive objective is consistent with scarcity *and* with the geometry
  hypothesis, and does not separate them.

## 5. Consequence for the research thesis

The directive's candidate thesis — *metrics reward frequency reallocation while
data and architecture provide limited within-pair supervision* — is now
**half-supported and half-falsified**:

- **Supported (data side):** 80.6% of within-pair supervision is head–head;
  tail–tail is 0.04%. Learning tail-vs-tail discrimination from VG150 is close to
  impossible, and `p49` showed mR@K is anti-correlated with grounding — the only
  cheap route to mR@K is reallocation, which needs no image.
- **Falsified as stated (architecture side):** it is not that *models* cannot
  use within-pair evidence. One of this very checkpoint's two readouts does so in
  the scarcest buckets. The constraint binds on the readout that was deployed,
  not on the family.

---

# CORRECTION — `runs/p52`: the 19% figure was a granularity artifact

`p50` measured groups at **raw object-name** granularity. That dataset retains
16,929 distinct object labels, of which only 150 are the VG150 vocabulary
(`docs/DATASET_IDENTITY_OBJECT_VOCAB.md`). Re-run restricted to relationships
whose **both** endpoints are in the standard 150 categories:

| | raw-name (`p50`) | **VG150-only (`p52`)** |
|---|---|---|
| train rows | 1,046,427 | **293,376** |
| train groups | 212,981 | **8,392** |
| mean group size | 4.91 | **34.96** |
| **P(group ≥2 distinct predicates)** | **19.0%** | **58.1%** |
| P(group ≥3 distinct predicates) | 7.2% | 37.4% |
| rows in decidable groups | 74.8% | **96.8%** |
| within-group entropy | 0.4209 | 0.7171 |

**The scarcity headline is WITHDRAWN.** At standard VG150 granularity, 58.1% of
training groups carry ≥2 distinct predicates and 96.8% of rows live in such
groups. Within-pair supervision is **not scarce** in the sense `p50` claimed,
and the "supervision-scarcity" reading of `p48` loses its main support.

**The bucket skew SURVIVES, and is slightly worse:**

| buckets | raw-name train | **VG150-only train** | VG150 val | VG150 test |
|---|---|---|---|---|
| head–head | 80.64% | **84.70%** | 87.29% | 85.05% |
| body–head | 12.24% | 8.53% | 7.93% | 9.44% |
| head–tail | 6.30% | 6.14% | 4.20% | 4.81% |
| body–body | 0.53% | 0.40% | 0.38% | 0.48% |
| body–tail | 0.25% | 0.19% | 0.14% | 0.21% |
| **tail–tail** | **0.04%** | **0.05%** | 0.05% | 0.02% |

**~1,700× head–head vs tail–tail, robust to granularity and reproduced on all
three splits.** That is the finding that stands: not that within-pair
supervision is scarce overall, but that it is **overwhelmingly concentrated on
head-vs-head contrasts**, with essentially none for distinguishing one rare
predicate from another.

The `p50` dissociation (text head tracks supply, classifier head does not) was
computed against the raw-name supply figures and should be recomputed against
`p52`'s before being relied on.
