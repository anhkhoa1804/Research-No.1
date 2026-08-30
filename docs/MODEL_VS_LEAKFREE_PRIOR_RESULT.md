# Result: how much remains beyond the leak-free prior?

Pre-registration: `docs/MODEL_VS_LEAKFREE_PRIOR_PREREGISTRATION.md` (written
before this run, criteria unchanged).

**Verdict: NEGLIGIBLE. The model's mR@50 is BELOW the leak-free prior's.**

---

## 1. The measurement

All four arms score the **identical** first 3,000 validation images. Verified
paired, not merely similar: every arm reports `n = 3000` and
`num_gt = 38,053`. All 50 predicates are present (rarest n=22).

| arm | vision | params | R@50 | mR@50 | head | body | tail |
|---|:---:|---:|---:|---:|---:|---:|---:|
| 2b. historical prior alone | no | 0 | 64.46 | 20.08 | — | — | — |
| **2. train-derived prior alone (leak-free)** | no | 0 | **66.80** | **21.98** | 42.5 | 13.5 | 12.8 |
| 1. model + historical prior (exact historical protocol) | yes | 79.9M | 66.90 | 21.16 | 54.1 | 14.9 | 5.8 |
| 3. train-derived prior, τ=0.1 | no | **1** | 66.16 | **26.00** | 42.7 | 18.3 | 19.6 |

Arm 1: `runs/p5_model_vs_leakfree_prior/`, 59.7 min on the L4, canary PASS on
all 14 protocol flags plus vocabulary size and ordering.

## 2. Against the pre-registered criteria

**Δ = arm 1 − arm 2 = +0.10 R@50, −0.82 mR@50.**

| criterion | threshold | actual | met? |
|---|---|---|---|
| SUBSTANTIAL | Δ_mR ≥ +3.0 and Δ_R ≥ −1.0 | −0.82 | no |
| MODEST | +1.0 ≤ Δ_mR < +3.0 | −0.82 | no |
| **NEGLIGIBLE** | **Δ_mR < +1.0** | **−0.82** | **yes** |

A 79.9M-parameter CLIP-based model, composed with a pair-conditioned
frequency prior, **does not beat a leak-free lookup table** on this protocol.
It matches it on R@50 (+0.10) and is *worse* on mR@50 (−0.82).

**Binding secondary criterion.** The no-vision recalibration (arm 3) gains
**+4.03 mR@50** over the same baseline for −0.64 R@50. The model gains −0.82.
A single scalar, with no image data, outperforms the entire visual system on
the class-averaged metric by **+4.84 mR@50**. Per the pre-registration, the
model's advantage is therefore **not evidence of visual information** — there
is no advantage to explain.

Note the upper-bound asymmetry works *against* the model here: arm 1 uses the
historical prior, which the manifest says cannot be verified leak-free, so any
leakage would have *inflated* arm 1. Even so inflated, it loses.

## 3. Correcting the 240-image pilot — a result of mine that did not survive

An earlier pilot on 240 images reported the model **+3.82 mR@50** ahead of the
leak-free prior and I highlighted it as a headline finding. **That is now
overturned.** At N=3,000 the same comparison is **−0.82**.

The cause is measurable, not mysterious: at N=240 only **47 of 50 predicates
occur at all**, so that mR@50 averaged over 47 classes, and the rarest classes
present had 1–2 instances each. mR@50 is an unweighted mean over classes, so a
handful of tiny classes dominated it. The comparison was internally consistent
(both arms shared the denominator) but the *magnitude* was an artifact of
sampling, and I should not have led with it.

Recorded here rather than quietly dropped, because the failure mode —
class-averaged metrics on small samples — will recur.

## 4. Where the model's behaviour actually differs

The aggregate hides a real and interpretable difference:

| | head mR@50 | tail mR@50 |
|---|---:|---:|
| leak-free prior | 42.5 | 12.8 |
| model + historical prior | **54.1** | **5.8** |
| prior + τ=0.1 | 42.7 | **19.6** |

The model is **substantially better on head predicates (+11.6)** and
**substantially worse on the tail (−7.0)**. It is a more confident version of
the prior's dominant mode, not a source of new discriminative information. The
τ adjustment moves in exactly the opposite direction, which is why it wins the
class-averaged metric.

INFERENCE: this is what a CLIP-text-cosine ranker composed with a skewed prior
should look like. It sharpens the majority decision. It does not resolve the
minority cases, which is where mR@50's leverage lives.

## 5. Also measured, and relevant to any literature comparison

Under GT pairs the headline `predcls R@K` emits one predicate per pair, making
K inert (R@50 = R@100 in every arm above). The literature-comparable
`predcls_multi` fields for arm 1: **multi_R@50 84.21 / multi_mR@50 47.37**.
Any comparison to published VG150 numbers must use those, not the headline.

## 6. What this settles, and what it does not

**Settled:** on VG150 PredCls under GT pairs, on 3,000 validation images with
all 50 predicates represented, the historical checkpoint contributes nothing
over a leak-free co-occurrence prior — and is beaten on mR@50 by a
one-parameter recalibration of that prior.

**Not settled:**

- Whether a *stronger frozen encoder* carries extractable appearance signal.
  That is the pre-registered ViT-L/14-336 probe
  (`docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md`), measured at ~20 min of GPU
  (60.4 crops/s, `runs/p4_bench_clip_l14_336/`). It is now the only open
  question that is cheap and decisive.
- Whether the historical 67.09 / 22.64 self-report is reproducible. Deferred:
  187 min, and bounded by four protocol fields the manifest records as never
  having been written down.
- Anything about SGCls/SGDet, where object identity is not given and
  co-occurrence is correspondingly weaker.

**Not licensed by this result:** any claim that vision is useless for SGG. This
measures one checkpoint, one protocol, one prior composition. It says the
*current* system adds nothing — not that nothing could.
