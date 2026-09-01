# WPRD — a prior-free relational-grounding metric, and what it finds

Runs: `p33_within_pair_discrimination`, `p34_ensemble_alpha_sweep`,
`p35_wprd_stratified`. All CPU, all on the `p24` full-validation cache
(10,401 images, 132,556 GT rows, cache validated 12/12).

---

## 1. The construction, and why it is prior-free

Within one (subject, object) **category** group, the train-derived prior row is
constant. Measured on the p24 cache: `max |prior_row − group mean| = 9.441e-05`
over every multi-row row. So for rows i, j in the same group and predicates a, b:

```
( score[i,a] − score[i,b] ) − ( score[j,a] − score[j,b] )
```

cancels the prior **exactly**. It also cancels any per-class additive
calibration, any tau, any logit adjustment and any global temperature — all of
them are functions of the class alone. What survives varies only with the image.

```
s_i(a,b)     = score[i,a] − score[i,b]
WPRD(g,a,b)  = AUC( {s_i : y_i = a}  vs  {s_j : y_j = b} )
```

**0.5 means no image-conditioned relational information.** No free parameters,
no operating point, no baseline to argue about.

**The metric validates itself.** The prior arm, scored through the identical
code path, reads **exactly 0.5000** — in every support band, in every predicate
bucket, with bootstrap CI [0.5000, 0.5000]. A random-tensor null reads 0.5046
macro / 0.4998 weighted.

## 2. The population, and what earlier nulls could not see

| population | rows | share | can a within-group permutation change anything? |
|---|---|---|---|
| singleton (s,o) groups | 30,918 | 23.3% | **no** — nothing to permute with |
| multi-row groups, CONSTANT GT | 26,272 | 19.8% | **no** — every row wants the same predicate |
| **decidable** (≥2 distinct GT) | **75,366** | **56.9%** | yes |

`runs/p26` and `runs/p29` measured `full − pair_matched_null = +0.031 ± 0.188`
and read it as "no image conditioning". **43.1% of the rows they averaged over
were structurally incapable of showing the effect.** Those runs are not wrong;
they bound a diluted quantity at one operating point, through an R@50/mR@50
metric a dominant prior has already saturated.

A pair-constant predictor's ceiling on the decidable rows is **69.23%** — the
remaining 30.77% require image evidence and nothing else can supply it.

## 3. Result — the signal exists, is weak, and the checkpoint runs the worse head

| arm | WPRD macro | 95% CI | weighted |
|---|---|---|---|
| **text head** (evaluated, `ensemble_alpha=0`) | **0.5542** | [0.5495, 0.5592] | 0.5266 |
| **classifier head** (stored, **discarded**) | **0.5728** | [0.5681, 0.5779] | 0.5349 |
| prior (control) | **0.5000** | [0.5000, 0.5000] | 0.5000 |
| random null | 0.5046 | — | 0.4998 |

Excluding same-instance rows — VG annotates one instance pair with several
predicates, and those tie at 0.5 by construction — raises both: text **0.5592**,
classifier **0.5797**. So the headline numbers are *conservative*.

Two conclusions, in the order of their strength:

1. **Image-conditioned relational signal exists.** The evaluated term's CI
   excludes 0.5 by more than nine half-widths.
2. **It is weak.** 0.55–0.58 against a 0.5 floor and a 1.0 ceiling.
3. **The checkpoint runs the weaker readout.** Both heads read the same
   768-d `rel_feat`; `ensemble_alpha` picks which readout, not whether vision is
   used.

## 4. Where the grounding lives — and does not

Stratified (`p35`). The classifier head wins in **all 6 predicate-bucket cells
and all 7 support bands, 13/13**.

| predicates compared | text head | classifier head |
|---|---|---|
| head–head | 0.5612 [0.5549, 0.5665] | 0.5701 [0.5651, 0.5759] |
| head–body | 0.5485 [0.5396, 0.5570] | 0.5823 [0.5725, 0.5913] |
| body–body | 0.5746 [0.5443, 0.6045] | 0.6167 [0.5871, 0.6464] |
| head–tail | 0.5273 [0.5113, 0.5458] | 0.5471 [0.5294, 0.5637] |
| **body–tail** | **0.5230 [0.4822, 0.5570]** — includes 0.5 | **0.5918 [0.5559, 0.6339]** — excludes 0.5 |
| tail–tail | 0.5126 [0.4174, 0.6246] — includes 0.5 | 0.5677 [0.4513, 0.6608] — includes 0.5 |

**The evaluated head has no measurable tail grounding.** Its `body–tail` and
`tail–tail` intervals both contain chance. The discarded head does have it:
`body–tail` 0.5918 with an interval clear of 0.5.

WPRD is **flat across pair support** (macro 0.548–0.562, all overlapping), so
this is not a data-density effect. The *weighted* score does decline with
support (0.5540 → 0.5211), i.e. within very frequent pairs the per-comparison
discrimination is closer to chance.

## 5. This answers the programme's founding question

> *Why does long-tail SGG performance appear better than the underlying
> relational reasoning?*

Because `mR@K` rewards moving probability mass toward rare predicates, and tau
does that for free without looking at the image — while the head the checkpoint
actually runs **cannot tell tail relations apart at all** (`body–tail` and
`tail–tail` CIs contain chance). The metric and the mechanism are decoupled: mR
goes up, grounding does not.

## 6. Operating-point consequence (`p34`), with its caveat

Sweeping `ensemble_alpha` on the same cache. WPRD rises monotonically with it
(0.5542 → 0.5728), and so does recall over the prior:

| tau | ΔR at α=0 | ΔR at α=0.5 | Pareto at α=0 | best Pareto |
|---|---|---|---|---|
| 0.00 | +0.575 | **+0.744** | +0.901 | +0.981 (α=0.10) |
| 0.05 | +0.475 | **+0.769** | +2.736 | +2.736 (α=0) |
| 0.10 | +0.137 | **+0.644** | +0.864 | **+4.126** (α=0.50) |

**Caveat, stated so it cannot be misread as a result:** these α values were read
off the validation split with no held-out selection. WPRD needs no selection, so
*"the classifier head is better grounded"* is robust; *"α=0.5 is the right
operating point"* is **not established** and must not be quoted as one.

## 7. What this does NOT license

It does not license "PURE sees relations". 0.55–0.58 against a 0.5 floor is
weak. The honest statement:

> The checkpoint has genuine but weak image-conditioned relational
> discrimination, concentrated on head predicates and absent from the tail in
> the head it actually runs. It is too weak to survive composition with a
> dominant frequency prior, and neither R@50/mR@50 nor the standard nulls can
> see it at all.

Whether the weakness is a **readout** failure or a **representation** failure is
not settled here. `p36`/`p37` are pre-registered to decide it
(`docs/READOUT_VS_REPRESENTATION_PREREGISTRATION.md`).
