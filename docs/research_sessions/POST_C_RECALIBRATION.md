# Post-C′ session record — what changed, what broke, and what the project now knows

Session of 2026-08-31, after the VM restart. Companion to
`docs/MODEL_RECALIBRATION_C_RESULT.md`, which carries the measurement. This
document carries the **process**: what was found, what was corrected, and what
the evidence now supports across the whole project.

---

## FACTS

**Recovered state.** The fp32 appearance-probe control had **completed**
(exit 0, 4,021 s) before the VM restart; only the watching shell died. Five p7
runs, one p8 run, four analysis documents and a fully-implemented but
never-committed C′ pathway were also found uncommitted.

**Work completed this session.**

| Run | What | Cost |
|---|---|---|
| `p9_probe_tiebreak_forensics` | experiment D0 — tie-breaking defect in experiment B | 2.3 s CPU |
| `p9_appearance_tau_interaction` | experiment D — (τ, λ, PCA) sweep | 764 s CPU |
| `p10_model_recalibration` | experiment C′ — **the only GPU pass** | 3,673.7 s GPU |
| `p11_cprime_analysis` | C′ CPU analysis family | 80.5 s CPU |

**One GPU pass total.** Everything else was CPU, most of it over caches that
already existed.

## METHOD CHANGES, AND WHY EACH WAS JUSTIFIED

Every change below preserves the historical protocol. `eval_sgg_use_vg_aliases`
keeps its default; the historical evaluator is untouched; 320 tests pass.

1. **`eval_freq_bias_tau`** — τ on the *prior* term. Needed because
   `eval_logit_adj_tau` is provably inert under the historical protocol
   (`runs/p8_tau_path_bug/`). τ = 0 returns the tensor by **identity**.
2. **`pair_logit_dump_v2`** — one GPU pass, all downstream questions on CPU.
   Default off.
3. **Denominator unified CPU-side, not in the evaluator.** See below.
4. **Pareto + null criteria replace ΔmR thresholds** throughout, because a
   zero-information scalar moves mR@50 by +4.03.

## SOURCE TRACE — the denominator, settled

`_build_vg_aliases` is predominantly an **object**-vocabulary normaliser
(`man/woman/boy/girl→person`, `bike→bicycle`, `sofa→couch`). Its predicate
entries are morphological free-text variants (`wear`/`wears`/`wearing`,
`ride`/`riding`, `upon`/`on top of`/`on`) — the signature of an open-vocabulary
text-matching helper. On VG150's closed 50-predicate vocabulary exactly **two**
fire:

```
near  -> next to
wears -> wearing            50 -> 48 classes
```

**Classification: accidental implementation artifact** in the closed-vocabulary
path. Not a historical protocol requirement — `eval_sgg_use_vg_aliases` is not
among the 14 flags the historical canary checks. Not a mere convenience — it
moves reported mR@50 by ~0.7–0.9 points off the literature-standard denominator.

**Resolution:** the evaluator is left alone; C′ unifies the denominator on the
CPU side, applying **one** scheme to **all** arms (primary raw50, robustness
column eval48). This is possible only because the cache stores raw GT plus the
alias map — the collapse is lossy and one-directional.

## CACHE

`runs/p10_model_recalibration/pair_logits.pt`, 42 MB, **CACHE VALID (12/12)**,
validated by a tool that re-implements the schema's invariants independently of
the analysis. Schema: `runs/p10_model_recalibration/cache_schema.md`.

## RESULTS

**Experiment D (frozen CLIP appearance) — H0 supported at every capacity.**
Across 104 (τ, λ > 0) points the best Pareto gap is **+0.085**; only 2 are
positive; shuffled noise beats real appearance at two of three capacities.
Quadrupling retained variance (192 → 1,024 features, ~56 % → ~83 %) moved the
answer 0.07 points. **Branch A — representation bottleneck — is falsified for
frozen CLIP through a linear probe.**

**Experiment C′ (the trained model) — complementary information confirmed.**
Model + prior reaches **R@50 67.07 / mR@50 24.27** against the prior's best-R@50
point of **66.80 / 21.98** — better on both axes, which no recalibration of the
prior can achieve. All five registered criteria met, under both schemes, with
held-out selection.

## NULL

C′'s null (pair-shuffled model rows, 5 seeds, rescored at every τ) is **strongly
negative**: −5.04 at τ = 0, −3.80 at τ = 0.05. Real margins of +2.1 to +6.1
points against SDs of 0.1–0.3.

D's null was **positive** (+0.43 ± 0.17). The signs differ because D's probe was
*fit* on shuffled features and could learn to ignore them, while C′ injects a
shuffled term without refitting. Each is the matched null for its own arm; the
lesson that carries is that **a null is not zero until it is measured.**

## CORRECTIONS ISSUED THIS SESSION

> **CORRECTION 1 — experiment B's most favourable number is withdrawn.**
> The probe's P0 baseline used `Pva.argmax(-1)` while every scored arm predicted
> through `topk`→`gather`. 213 of 15,280 val rows resolve ties differently, so
> the arms never reduce to the reported P0 at λ = 0. The like-for-like baseline
> is **21.57, not 20.98**, and restated, **every λ in B's sweep is negative**.
> B's quoted "+0.2 % cherry-picked upper bound" was an artifact. B's verdict is
> unchanged and strengthened.

> **CORRECTION 2 — S-1 is not the binding constraint.**
> I identified `ensemble_alpha = 0.0` discarding the trained classifier as the
> most severe suppression point. At usable operating points that is wrong: at
> τ = 0.05 the historical `0.0` is the *best* setting (+2.29) and restoring the
> classifier is *worse* (+1.41). Decision B is specifically not supported.

> **CORRECTION 3 — my registered prediction for C′ was wrong.**
> The pre-registration predicted "C or B". The measurement is **A**.

> **CORRECTION 4 (self-caught, pre-publication) — the inferred marginal.**
> `log P(p)` was initially inferred from the cache as the modal prior row. That
> row is the **uniform** `default_log_prob` fallback, not the class marginal, so
> τ was a silent no-op and the frontier came out nearly flat. Caught by the p7
> reproduction gate, which a τ = 0-only check would have passed. `log P(p)` is
> now read from the prior file and the gate is permanent.

**Four defects, three of them caught before they could contaminate a result;
two of them before any GPU time was spent.**

## LIMITATIONS

- C′'s effect is **small** (+0.27 R@50 / +2.29 mR@50) and its largest gaps sit
  at degenerate operating points (τ = 1.0, R@50 ≈ 18 %). The registered criterion
  lacks an R@50 floor; future registrations should carry one.
- GT pairs, PredCls, 3,000 validation images, one checkpoint.
- D and C′ differ in representation, head, candidate set and composition; neither
  predicts the other.

## DECISION

**C′: A — COMPLEMENTARY INFORMATION CONFIRMED**, small and reproducible.
**D: H0 supported** — frozen CLIP appearance adds nothing beyond calibration.

### Bottleneck branches after this session

| Branch | Status |
|---|---|
| **A** representation | **FALSIFIED** for frozen CLIP + linear probe (D) |
| **J** no useful visual information | **FALSIFIED** — C′ shows a real, null-exceeding contribution |
| **E** prior dominance | **Confirmed as dominant** — one zero-information scalar traces most of the reachable frontier |
| **B** decision rule destroys signal | **Weakened** — restoring the discarded branch does not help at usable operating points |
| **G** insufficient visual conditioning | **Live and now leading** — the model contributes, but only ~2 mR points |
| C, D, F, H, I | untested |

## NEXT STEP

**No architecture is to be implemented or trained on this result.**

The open question C′ raises and does not answer: the model ranks GT **worse** on
average (mean GT rank 1.83 → 2.78, rank worsened on 4,675 pairs vs improved on
2,013) yet still converts **+256 net** prior errors into hits. A term that
degrades the global ordering while improving the top-1 decision is a
**confidence-calibration** phenomenon, not a representation one.

That is answerable **on the existing cache, on CPU, with zero new GPU time** —
and it is the first question in this project whose answer would actually
constrain an architecture. It should be asked before any architecture memo is
written, not after.
