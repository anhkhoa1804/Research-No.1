# Literature gap analysis — is a prior-controlled relational benchmark novel?

Required by the directive before any novelty claim. Status: **a specific gap is
identified and is narrower than the one originally imagined.** The novel object
is a **metric/protocol**, not a new dataset.

Searched 2026-09-01. This is a working map, not an exhaustive survey; a formal
related-work pass is still required before submission.

---

## 1. What already exists

### SGG debiasing — the mechanism is well known
The frequency prior's dominance in VG150 is **not** a new observation and must
not be presented as one. Neural Motifs established the `FREQ` baseline; the
debiasing literature is large and mature:

- **Tang et al., *Unbiased SGG from Biased Training* (CVPR 2020)** — causal TDE,
  and the SGG **diagnosis toolkit**: mean Recall plus Sentence-to-Graph
  Retrieval. Closest in *spirit* to a diagnostic contribution.
- **Resistance Training using Prior Bias**, **NICEST** (noisy labels),
  **Semantic Diversity-aware Prototype Learning**, **type-aware message
  passing**, and many others — all *methods* that reduce prior reliance.
- **VG-OOD** — a **re-split** of VG to reduce frequency-bias influence.

Our diagnosis overlaps this literature and must be positioned inside it.

### SGG metrics — the closest prior art
- **Lorenz et al., *A Review and Efficient Implementation of SGG Metrics*
  (CVPRW 2024)** — the most relevant. Defines **Pair Recall@k** (strip the
  predicate; measures pair detection) and **Predicate Rank** (rank of the
  correct predicate *given a correct subject-object pair*).
- **Haystack (ICCVW 2023)** — a PSG dataset for **rare** predicates, notable for
  explicit **negative** annotations. Addresses tail *measurability*.

**The decisive check.** Predicate Rank stratifies by **predicate class**. Pair
Recall pools **across all subject-object category pairs**. mean Recall
stratifies by **predicate class**. *None of them conditions on the
subject-object category pair*, so in all of them a model can score well using
object identity alone — which is exactly the failure we are trying to measure.
Predicate Rank is the nearest miss: it conditions on the pair being *correct*,
not on the pair *identity*, so the prior still drives it.

### Compositional VLM benchmarks
- **Winoground** (400 sets, minimally-different captions), **ARO** (VG/COCO
  Attribution-Relation-Order), **SugarCrepe** (LLM-generated fluent hard
  negatives), **SugarCrepe++**.
- **SugarCrepe's central critique is the direct analogue of our finding**: *a
  blind model that never looks at the image* could solve ARO. Their fix was to
  construct negatives that drive **blind text models to chance**.

These are **image–text matching** benchmarks over constructed caption negatives.
They do not measure a scene-graph model's per-pair predicate decision, and they
do not control the (subject, object) prior — they control *text* plausibility.

## 1b. Deeper pass — the decisive enumeration (added 2026-09-01)

The directive's question is exact, so the check must be:

> **Has any existing SGG evaluation explicitly conditioned predicate
> discrimination on the subject-object category pair, so that P(p | s,o) becomes
> non-informative?**

### Every metric in the field's reference implementation

`Scene-Graph-Benchmark.pytorch/METRICS.md` (Tang et al.) is the de-facto
standard. Enumerated in full:

| metric | what is pooled | conditions on (s,o) category pair? |
|---|---|---|
| `R@K` | all pairs, top predicate | **no** |
| `ngR@K` | all pairs, all 50 predicates per pair | **no** |
| `mR@K` | per **predicate** category, then averaged | **no** — stratifies by predicate |
| `ng-mR@K` | per predicate, no graph constraint | **no** — stratifies by predicate |
| `zR@K` | triplets unseen in training | **no** |
| `ng-zR@K` | as above, no graph constraint | **no** |
| `A@K` | GT subject-object pairs only | **no** |
| `S2G` | graph↔sentence retrieval | **no** |

`zR@K` is the nearest miss and is worth being precise about: it conditions on the
**triplet** being unseen in training, which is a *dataset-level* novelty
condition. It does not make P(p | s,o) non-informative — a zero-shot triplet
still sits in an (s,o) group whose prior is perfectly informative about the
*other* rows of that group.

Adding Lorenz et al. (CVPRW 2024): `Pair Recall` pools across all (s,o) pairs;
`Predicate Rank` conditions on the pair being **correct**, not on its
**identity**. Neither cancels the prior.

### The closest prior work, and what it actually does

- **SpatialSense (ICCV 2019)** — the closest in spirit, and it must be cited
  prominently. It reduces prior informativeness by **adversarial crowdsourcing**:
  annotators are asked to find relations hard to predict from 2D configuration
  or language. It reports **language-only** and **2D-only** baselines — which are
  precisely the prior baseline and the geometry baseline this project
  reconstructed independently. **Those baselines are not novel and must be
  credited to SpatialSense.** The differences: it builds a *new dataset* and
  reduces bias *empirically by curation*; it is *binary relation verification*,
  not predicate ranking; and it does not condition on the (s,o) pair.
- **Plesse et al. (WACV 2020)** — states the confound explicitly: "the majority
  relation for each object category pair often represents from 50% to 75% of the
  examples … the evaluation does not reflect that." This project measured the
  same thing independently (69.23% ceiling for a pair-constant predictor on
  decidable rows; 48.7% of multi-row groups have a constant GT). **The
  observation is theirs, not ours.** Their contribution is a relevance model and
  a detector that beats MotifNet, not a pair-conditioned metric.
- **VG-OOD** — a *re-split*. Changes the data distribution; does not neutralise
  the prior within an evaluation.
- **Haystack (ICCVW 2023)** — rare-predicate PSG data with explicit *negative*
  annotations. Addresses tail measurability, not prior conditioning.

### Verdict of the deeper pass

**No existing SGG/VRD evaluation conditions predicate discrimination on the
(s,o) category pair.** The confound is named in the literature (Zellers, Plesse)
and attacked by *curation* (SpatialSense), *re-splitting* (VG-OOD), *predicate
stratification* (mR@K), and *causal adjustment* (TDE) — but never by
**conditioning**, which is the one route that makes the prior exactly
non-informative rather than approximately less informative.

## 2. The gap, stated precisely

> SugarCrepe made **blind text** models chance-level on image–text
> compositionality. **No one has made a pair-prior model chance-level on SGG
> predicate prediction.**

That is what WPRD does, and it does it *by construction* rather than by dataset
curation: conditioning on the (s,o) group makes the prior cancel exactly, and
the prior control measures **0.5000** with CI [0.5000, 0.5000] in every stratum.

Three properties that, in combination, we did not find in the literature — and
the emphasis is on *in combination*, since each has relatives:

1. **Exact analytic cancellation** of the subject-object prior, verified by a
   control that reads exactly chance — not a re-split, not a reweighting, not a
   causal adjustment estimated from data.
2. **Operating-point-free.** No tau, no alpha, no K, no floor. It cannot be
   moved by calibration, which is the failure mode that contaminates mR@K.
3. **Computable on existing VG150 annotations.** No new images, no new labels.
   56.9% of GT rows already qualify.

## 3. What is NOT novel, and must be said so

- That the frequency prior dominates VG150. **Known since Neural Motifs.**
- That the per-pair majority relation is 50–75% of examples and that evaluation
  hides it. **Plesse et al., WACV 2020, say this explicitly.**
- The **prior-only baseline** and the **geometry/2D-only baseline**.
  **SpatialSense (ICCV 2019) introduced both** as language-only and 2D-only
  probes. This project rediscovered them; it did not invent them.
- That mR@K is gameable by calibration. **Known**, and part of why the
  Tang et al. toolkit exists.
- That tail predicates are hard/under-measured. **Known**; Haystack is
  explicitly about it.
- Wanting a "prior-adversarial relational benchmark". The *aspiration* is not
  novel; VG-OOD and the debiasing literature share it.

The defensible claim is narrower and stronger: **a prior-free relational
discrimination metric with an exactly-chance prior control, and the measurement
it enables** — that a trained SGG checkpoint's evaluated head has *no*
measurable tail grounding while a head it discards does.

## 4. Consequence for the benchmark direction

**A new dataset is NOT yet justified.** The metric runs on existing data and has
already produced the finding. Building images before the metric is accepted
would invert the order of evidence.

What the metric does not cover, and what a dataset would eventually have to add:

- **Controlled counterfactuals** — same (s,o), *same instances*, relation
  physically changed. VG gives us same-(s,o)-different-image, not a controlled
  intervention.
- **Role binding / directional reversal** — (A, on, B) vs (B, on, A). Testable
  on existing data as a WPRD variant; worth trying **before** collecting.
- **Negative annotations** for the tail — Haystack already supplies these for
  PSG and should be reused rather than duplicated.

## 5. Immediate next steps on this track

1. Run WPRD on **published SGG checkpoints** (Motifs, VCTree, and a TDE-debiased
   variant). If the "no tail grounding" result generalises beyond PURE, the
   contribution is about the field, not about one checkpoint. **This is the
   single highest-value cheap test for the benchmark claim.**
2. Implement the **role-swap** WPRD variant on existing VG150 data.
3. Only then consider collecting anything.

## Sources (deeper pass appended)
- [Scene-Graph-Benchmark METRICS.md](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/METRICS.md)
- [SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition](https://arxiv.org/abs/1908.02660)
- [Focusing Visual Relation Detection on Relevant Relations with Prior Potentials (Plesse et al., WACV 2020)](https://openaccess.thecvf.com/content_WACV_2020/papers/PLESSE_Focusing_Visual_Relation_Detection_on_Relevant_Relations_with_Prior_Potentials_WACV_2020_paper.pdf)
- [Visual Relationship Detection with Language Priors](https://arxiv.org/abs/1608.00187)

## Sources
- [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949)
- [A Review and Efficient Implementation of Scene Graph Generation Metrics](https://arxiv.org/html/2404.09616)
- [Haystack: A Panoptic Scene Graph Dataset to Evaluate Rare Predicate Classes](https://arxiv.org/abs/2309.02286)
- [Rethinking the Evaluation of Unbiased Scene Graph Generation](https://arxiv.org/pdf/2208.01909)
- [SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality](https://arxiv.org/html/2306.14610)
- [SugarCrepe++](https://arxiv.org/html/2406.11171v1)
- [Winoground](https://www.researchgate.net/publication/363908770_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality)
- [Scene Graph Generation: A Comprehensive Survey](https://arxiv.org/pdf/2201.00443)
