# Experiment D result — appearance adds nothing beyond calibration, at any capacity

Pre-registered in `docs/APPEARANCE_TAU_INTERACTION_PREREGISTRATION.md`, committed
at `aaa24e9` **before** any arm was run.
Run: `runs/p9_appearance_tau_interaction/` (exit 0, 764 s, CPU only, no GPU).
Co-registered defect check D0: `runs/p9_probe_tiebreak_forensics/` (exit 0, 2.3 s).

Nothing was overwritten. `tools/appearance_probe.py`, experiment B's
pre-registration and its two result JSONs stand verbatim.

---

## 1. Verdict

**H0 SUPPORTED at every representation capacity tested.**

| PCA dim | feature dim | selected (τ, λ) | real Pareto gap | shuffled null | margin | 2 SD gate | verdict |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 192 | (0.5, 0.1) | **+0.01** | +0.43 ± 0.17 | −0.42 | 0.34 | NULL |
| 128 | 512 | (0.5, 0.1) | **+0.08** | +0.26 ± 0.49 | −0.17 | 0.97 | NULL |
| 256 | 1024 | (0.3, 0.1) | **−0.51** | −0.65 ± 0.78 | +0.14 | 1.56 | NULL |

Pareto gap = the arm's mR@50 minus the λ=0 τ-curve's mR@50 **at the same R@50**,
in mR points. The registered criterion required gap > 0 **and** ≥ 2 SD above
the shuffled-appearance null. Neither PCA-48 nor PCA-128 clears the null at all;
PCA-256 clears a null that is itself negative, on a gap that is negative.

Across the **whole surface** — 104 real (τ, λ>0) points — the best Pareto gap
anywhere is **+0.085**, and only 2 of 104 points are positive at all. Zero
exceed +0.5. Shuffled noise at the selected point scores **higher** than real
appearance at PCA-48 and PCA-128.

At every τ and every capacity, held-out selection chose **λ = 0.1**, the
smallest non-zero weight on the grid. The optimiser's preference is to turn the
appearance term down as far as the grid allows.

## 2. What this closes

**Branch A (representation bottleneck) is falsified for this probe.**
PCA-48 retains only 55–57 % of each block's variance, which made "truncation,
not absence" a live alternative to B's null. Quadrupling the retained
dimension — 192 → 1024 features, ~56 % → ~83 % of variance — moved the Pareto
gap by **0.07 points**, inside the noise band. The bottleneck is not how much
of the CLIP representation the probe was allowed to see.

**B's headline is now safe to read the way it was being read**, but for a
different reason than B gave. B's own λ-only family could not express the
calibrated operating point, so its −1.2 % was not by itself evidence about
appearance. D removes that confound by giving appearance a calibrated prior to
work with at every τ from 0 to 0.5, and appearance still buys nothing.

## 3. CORRECTION to experiment B — D0

`tools/appearance_probe.py` computes its reported baseline as `Pva.argmax(-1)`,
but predicts every scored arm through `topk(K=5)` then `gather`-argmax. On
prior rows with tied top values these select different predicates: **444 val
rows carry a top-1 tie and 213 of 15,280 (1.39 %) resolve differently.** The
additive arms therefore never reduce to the reported P0 at λ = 0, and
`headroom_pct` compared two decision rules.

| baseline | R@50 | mR@50 |
|---|---:|---:|
| as published (`argmax`) | 67.38 | 20.98 |
| like-for-like (`topk`→gather) | 67.43 | **21.57** |

Restating B's additive sweep against the baseline its arms actually reduce to:

| λ | arm mR@50 | published headroom % | like-for-like ΔmR | like-for-like % |
|---:|---:|---:|---:|---:|
| 0.10 | 21.03 | +0.10 | **−0.54** | −1.26 |
| 0.25 | 21.06 | +0.19 | **−0.51** | −1.18 |
| 0.50 | 20.57 | −0.94 | −1.00 | −2.32 |
| 1.00 | 20.49 | −1.13 | −1.08 | −2.51 |
| 2.00 | 19.90 | −2.47 | −1.67 | −3.87 |

> **CORRECTION.** B's reported "cherry-picked upper bound of **+0.2 %** of
> oracle headroom" is an artifact of the tie-breaking mismatch. Like-for-like,
> **every λ on the grid is negative.** There is no positive upper bound to
> report. B's pre-registered verdict (H0 supported) is **unchanged and
> strengthened**; only the most favourable number it quoted is withdrawn.

The magnitude is small and the direction was not predicted in advance
(registered as such in the pre-registration, §7). It happens to move against
the appearance hypothesis.

## 4. Secondary observations

**The τ curve is bit-identical across all three PCA runs** (verified
programmatically). It is a prior-only arm and must not depend on the visual
features; that it does not is an internal consistency check on the harness.

**The shuffled-appearance null is not centred on zero.** It is **+0.43 ± 0.17**
at PCA-48. Adding *any* small scalar-weighted random term to a prior as peaked
as this one (0.557 nats) perturbs ties and marginally improves mR at matched
R@50. This is why the null had to be refit at the selected (τ, λ) rather than
assumed to be zero — a naive test would have read a +0.4 point "gain" from pure
noise as evidence for appearance.

**The fp16/fp32 pair gives an accidental stability check on B's ablation gate.**
The gate arms move between the two runs by far more than precision should
allow: shuffled mR 12.17 → 10.79, shuffled R@50 26.77 → 46.42, shuffled tail_mR
8.71 → 4.17. `fit()` early-stops on the **maximum** held-out mR over 30 epochs,
which for a noise-feature arm is a max-statistic over a noisy sequence — high
variance, biased upward. The gate's ordering (real > shuffled > zero) survives
in both runs and the bias is conservative, so the gate's conclusion stands, but
**its margin should not be quoted as a precise quantity from a single draw.**

## 5. Bottleneck branches after D

| Branch | Status |
|---|---|
| **A — representation bottleneck** | **FALSIFIED for frozen CLIP + linear probe.** 4× capacity, no movement. |
| **E — prior dominance** | **Confirmed as the dominant effect.** The whole (R@50, mR@50) frontier reachable here is traced by one zero-information scalar. |
| **J — no useful visual information** | Not established. B's ablation gate (real > shuffled > zero) says frozen CLIP *contains* predicate signal. D says that signal is not *convertible* through a linear probe composed with this prior. These are compatible. |
| **B — decision/composition bottleneck** | **Live and now the leading candidate.** D tested a *linear* probe in an *additive* composition. It did not test a non-linear probe, a multiplicative/gated composition, or the trained model's own logits. |
| **C, D, F, G, H, I** | Untested here. |

## 6. What D does **not** license

D used a linear head on PCA'd frozen CLIP features in an additive composition
with a fixed top-5 candidate set. It does **not** show that no visual system can
beat the τ curve. It shows that *this family* cannot, and that the family's
failure is not caused by representation truncation.

The next falsifiable step is **C′** (`docs/…C_PREREGISTRATION` — still to be
written; the pathway is committed at `b8ce06b`): ask the same Pareto question of
the trained model's own per-pair logits, which are neither frozen, nor linear in
CLIP features, nor restricted to top-5.
