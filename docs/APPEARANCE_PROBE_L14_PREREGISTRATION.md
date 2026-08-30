# Pre-registration: appearance probe at ViT-L/14-336 on the L4

**Written before the experiment was run. Not edited afterwards.**
Results go in a separate document so this one cannot be retrofitted.

Registered at commit `e7bfcfb`, branch `research/architecture-breakthrough`.

---

## 1. Why this experiment exists

`docs/APPEARANCE_PROBE_FINDINGS.md` concluded that frozen-CLIP appearance
carries real but negligible predicate signal (0.6 % of headroom captured), and
recommended re-scoping the project to a negative result. That conclusion
carries an explicit, self-identified caveat:

> **ViT-B/32, not ViT-L/14-336.** Forced: L/14-336 measured at 0.3 crops/s on
> this CPU-only machine, infeasible for ~70k crops. B/32 is weaker, so this is
> a **lower bound**. Confirming on GPU with L/14-336 is the single experiment
> that could overturn this verdict.

That machine had no GPU. This one has an NVIDIA L4. The blocking constraint is
gone, so the experiment the previous analysis named as decisive is now cheap.

**This is a pre-registered replication at a stronger encoder, not a new
hypothesis.** The prior work already named the threshold; §4 adopts it
unchanged rather than inventing a friendlier one.

## 2. Hypothesis

**H1.** The weak appearance signal measured with ViT-B/32 is an artifact of
encoder capacity. A stronger frozen encoder (ViT-L/14-336, the model this
repository's own checkpoint uses) captures materially more of the prior's
oracle headroom.

**H0 (null).** Appearance capacity is not the binding constraint. L/14-336
captures approximately as little as B/32, and the protocol is prior-saturated
regardless of encoder strength.

## 3. Design

Unchanged from the B/32 run except for the encoder and device. Same prior,
same splits, same K=5 candidate set, same PCA-48, same 30 epochs, same
image-level held-out split, same weight decay.

| | |
|---|---|
| Encoder | `openai/clip-vit-large-patch14-336`, frozen, never fine-tuned |
| Baseline for comparison | the same probe at `openai/clip-vit-base-patch32` |
| Prior | `datasets_vg150_clean/frequency_prior_train.json` (train-split only) |
| Eval split | `datasets_vg150_clean/validation.jsonl` |
| Candidate set | prior's top-K, K=5 |
| Composition | `score = log P(p|s,o) + λ · f(appearance)` |
| λ grid | **fixed in advance**: 0, 0.1, 0.25, 0.5, 1.0, 2.0 |
| Precision | fp16 autocast on CUDA; a fp32 control arm is run to confirm it is inert |

### Arms (all pre-specified)

1. **REAL** appearance
2. **SHUFFLED** appearance — features permuted across instances
3. **ZERO** appearance — features zeroed
4. **PRIOR-ONLY** (λ = 0), which is exactly P0
5. **ADDITIVE** real appearance + exact prior, over the full λ grid
6. **fp32 control** at the best λ, to confirm mixed precision changed nothing

Arms 1–3 are the validity gate. Arm 4 is the control every other arm is
measured against. Arm 6 guards the one methodological change made for speed.

## 4. Success and failure criteria — fixed now

**Primary metric.** Fraction of the oracle@5 mR@50 headroom captured on the
validation split:

```
captured = (mR_additive - mR_prior) / (mR_oracle@5 - mR_prior)
```

The prior work set the decision threshold; it is adopted verbatim:

> If L/14-336 also captures <5 %, the appearance thesis is settled and the
> project should stop pursuing it.

| Outcome | Criterion | Interpretation |
|---|---|---|
| **H1 SUPPORTED** | captured ≥ 5 % | Encoder capacity mattered. The B/32 negative was an artifact. Appearance is worth pursuing. |
| **H0 SUPPORTED** | captured < 5 % | Settled. Encoder strength is not the constraint. Stop pursuing appearance reranking. |
| **INVALID** | ablation gate fails (`real > shuffled > zero` does not hold) | The measurement is broken; the thesis is untested either way. Do not report a verdict. |

**Anti-cherry-picking rules, binding:**

- λ is selected on the **held-out-from-train** split, never on validation. The
  reported headline is validation mR@50 *at that λ*. Picking λ by validation
  score would make the headline the maximum of six correlated draws.
- The **entire λ curve is reported**, including λ values that hurt.
- The 5 % threshold is fixed above and will not be renegotiated after seeing
  the number.
- A result between 4 % and 6 % will be reported as **inconclusive at this
  sample size**, not rounded toward either verdict.

## 5. A second denominator, reported alongside

`tools/headroom_analysis.py`, run on this machine on the full validation split
(`runs/p3e_headroom_train_derived/`), measured that of 33.41 R@50 points of
total headroom over the prior, only **8.84 points are recoverable** — 64.2 % of
the prior's errors are generic→generic confusions (`on`↔`in`↔`of`), which are
annotation-style choices no encoder can resolve. Only 12.6 % of GT triplets
carry a decidable predicate.

So "0.6 % of headroom captured" uses a denominator that is mostly unreachable
by construction. The result will therefore be reported against **both**:

- **captured_total** — against full oracle@5 headroom (the pre-registered
  primary, comparable to the B/32 number)
- **captured_decidable** — against headroom on the decidable-predicate subset
  only (secondary, diagnostic)

The 5 % decision threshold applies to **captured_total** only. The secondary
number is reported because it is more informative, not to move the goalposts;
it cannot change the verdict.

## 6. What this experiment cannot settle

- It tests **frozen** features with a linear scorer. A negative does not prove
  no fine-tuned encoder could do better; it bounds what is linearly decodable
  from frozen CLIP.
- PCA-48 may discard signal. Retained from the B/32 run for comparability.
- It says nothing about SGCls/SGDet, where object identity is not given.
- A negative result does **not** validate the current architecture. It removes
  one candidate explanation for the architecture's failure to beat the prior.

## 7. Compute budget

One extraction pass (~70k crops, fp16, batched, L4) plus scoring on CPU.
Extraction is cached to disk keyed by encoder, so scoring can be re-run and
the cache reused without re-encoding. If extraction exceeds 90 minutes it will
be stopped and re-scoped rather than allowed to run unbounded.
