# `runs/p63` — prior-override diagnostics, named and stratified

Extends `p43` (`docs/PRIOR_ADVERSARIAL_RESULT.md`) with the specific named
quantities a benchmark reader needs: not just "fixed/kept" counts, but rates
that separate *trying* to override the prior from *succeeding*. Same
population as `p43` (44,283 prior-adversarial rows, 33.4%, tau=0.0), same
train-derived prior, same cache. CPU only.

## Headline

| quantity | value |
|---|---|
| Prior Override Rate (moved off the prior's choice) | **16.45%** |
| Successful Override Rate (landed on GT) | **7.91%** — reproduces `p43`'s "fixed" fraction exactly |
| Wrong Override Rate (moved, missed GT) | **8.54%** |
| Tail Override Rate (successful, GT in tail bucket) | **0.60%** (n=2,176) |

`16.45% = 7.91% + 8.54%` exactly — every row where the model moves off the
prior either lands on GT or doesn't; there is no third outcome.

**The model overrides the prior on one row in six, and is right about half the
time it does.** On the other 83.55% of adversarial rows it simply repeats the
prior's wrong answer.

## By predicate bucket

| bucket | n adversarial | override rate | success rate | wrong rate |
|---|---|---|---|---|
| head | 35,325 | 16.79% | **8.73%** | 8.06% |
| body | 6,782 | 17.16% | 5.97% | 11.19% |
| tail | 2,176 | 8.69% | **0.60%** | 8.09% |

The tail's override rate is already the lowest (8.69% vs ~17% elsewhere) — the
model rarely even *tries* to move toward a tail predicate — and when it does
try, it succeeds only 0.60% of the time (vs failing 8.09% of the time). This is
consistent with, and sharpens, `p35`'s tail-tail WPRD finding: it is not merely
that tail discrimination is statistically indistinguishable from chance in
aggregate — at the decision level, on the specific rows where a tail predicate
was the correct override, the model almost never makes it.

## By prior confidence (quartile bins of the prior's top-1 probability)

| bin | n adversarial | successful override rate |
|---|---|---|
| Q0 (lowest confidence) | 15,665 | **12.63%** |
| Q1 | 13,050 | 8.02% |
| Q2 | 10,295 | 4.63% |
| Q3 (highest confidence) | 5,273 | **0.00%** |

A clean monotonic trend, and the endpoint is exact: **when the prior is most
confident and wrong, the model never once corrects it in this population.**
This is the sharpest single number in the diagnostic — it says the model's
occasional corrections are concentrated exactly where the prior was already
uncertain, i.e. where correcting it is cheapest, and contribute nothing where
the prior is confidently wrong.

## Margins

| | prior margin (top1−top2) | model margin |
|---|---|---|
| all rows | 5.56 | 6.26 |
| adversarial | 3.53 | 3.89 |
| prior-correct | 6.58 | 7.44 |

Adversarial rows have a smaller prior margin than prior-correct rows in both
the prior's own score and the model's — consistent with the confidence-bin
result: adversarial rows are disproportionately drawn from cases where the
prior was already less decisive, not cases where it was wrong with confidence.

## Standing

This is a decision-level diagnostic on the field's own accuracy scale, computed
without touching validation labels to define its population (the adversarial
set is a pure function of the train-derived prior). It requires no invented
metric and is immune to the "you built WPRD to make models look bad" objection
— making it, alongside `p43`, one of the most directly defensible results in
the programme. Recommended as a **benchmark-essential** component alongside
C7's existing fixed/kept table.

One checkpoint, PredCls with GT pairs, validation split.
