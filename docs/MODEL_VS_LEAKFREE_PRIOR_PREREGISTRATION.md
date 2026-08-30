# Pre-registration: how much remains beyond the leak-free prior?

**Written before the run. Registered at commit `d1e8f0a`-parent, branch
`research/architecture-breakthrough`.**

## The question

> How much visual/model information remains beyond the leak-free train-derived
> prior, when evaluated on the exact historical protocol?

## Design

**The historical protocol is not modified.** All 14 canary-verified flags are
unchanged, including `--freq_bias_path checkpoints/demo_best/frequency_prior.json`.
Swapping the prior file would have been a cleaner ablation but *is* a protocol
change; keeping it exact preserves comparability with every earlier arm.

Three arms, on the **identical first 3,000 validation images** (the loader runs
`shuffle=False`, so "first N rows" is well defined and matched — verified
earlier by exact `num_gt` agreement at N=240):

| arm | what it measures | cost |
|---|---|---|
| 1. model + historical prior (exact historical protocol) | the full system | GPU, ~55 min |
| 2. train-derived prior alone (τ=0) | the **leak-free** baseline | CPU, seconds |
| 3. train-derived prior, τ=0.1 | the no-vision recalibration control | CPU, seconds |

The quantity of interest is **arm 1 − arm 2**.

### Why N = 3,000, measured not guessed

Predicate coverage of the validation split as a function of N:

| N images | GT triplets | predicates present | rarest predicate n |
|---:|---:|---:|---:|
| 240 | 3,093 | **47 / 50** | 1 |
| 1,000 | 12,751 | 50 | 3 |
| 2,000 | 25,383 | 50 | 8 |
| **3,000** | **38,053** | **50** | **22** |
| 10,401 | 132,556 | 50 | 103 |

At N=240 only **47 of 50 predicates occur at all**, so the earlier pilot's
mR@50 averaged over 47 classes. That is why the 240-image numbers must not be
over-read. N=3,000 is the smallest N where all 50 predicates appear with a
usable count, at 29 % of the full-split cost (55 min vs 187 min, measured).

The comparison is **paired** — both arms score the same images and the same GT
triplets — so it is far tighter than independent sampling would be.

## The leak asymmetry, and why it is acceptable

Arm 1 uses the historical prior, of which
`data/manifests/historical_checkpoint_v1.yaml` says: *"Cannot be independently
verified as leak-free."* Arm 2 uses the train-split-only prior, which is
leak-free by construction.

If the historical prior leaks, it **inflates arm 1**. So `arm 1 − arm 2` is an
**upper bound** on the model's contribution over a leak-free prior. A small
upper bound is therefore a strong negative result; a large one is only
suggestive and would need the leak ruled out before it could be claimed.
Bounding in the conservative direction is the point.

## Success / failure criteria — fixed now

Let `Δ_mR = mR@50(arm 1) − mR@50(arm 2)` and `Δ_R` likewise, same images.

| Outcome | Criterion | Meaning |
|---|---|---|
| **SUBSTANTIAL** | Δ_mR ≥ +3.0 and Δ_R ≥ −1.0 | The model carries information beyond the leak-free prior (upper bound). Architecture work becomes justifiable. |
| **NEGLIGIBLE** | Δ_mR < +1.0 | The model adds essentially nothing over a lookup table. |
| **MODEST** | +1.0 ≤ Δ_mR < +3.0 | Real but small; not on its own a basis for architecture work. |

**Binding secondary criterion.** Whatever Δ_mR is, it is reported beside
`arm 3 − arm 2`, the gain from a one-parameter recalibration with **no visual
input**. If `Δ_mR ≤ (arm 3 − arm 2)`, then the model's advantage is **not
evidence of visual information**, because a scalar achieves as much or more
for free. This holds even under the SUBSTANTIAL verdict, and must be stated
whenever the model's Δ is quoted.

## Verified before launch

- checkpoint SHA256 `8845c3af…` — byte-identical to the manifest
- historical prior SHA256 `144d9f92…` — byte-identical
- dataset train/val — bit-for-bit reproducible from the immutable extract
- protocol — canary PASS on all 14 flags plus vocabulary size **and ordering**
- batch size — MEASURED: bs=48 gives *bit-identical* metrics to bs=12 on the
  same 240 images but is slower (348 s vs 327 s). Larger is not better here;
  bs=12 is both faster and protocol-identical to the canary.
- ETA — 1.073 s/image measured over 240 images ⇒ **~55 min** for 3,000.

## Known limitation

The evaluator writes `metrics.jsonl` only after the split completes, so the run
is **not resumable**. N=3,000 caps the loss from any failure at ~55 min. Making
eval checkpointing incremental is a worthwhile infrastructure fix but is not
attempted mid-experiment.
