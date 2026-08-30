# Hypothesis: the bottleneck is the decision rule, not the representation

**Status: HYPOTHESIS. Written before the test was run.**
Registered at commit `91b24c7`.

---

## 1. The contradiction this is trying to resolve

Two tools in this repository, both re-measured on this machine today, disagree
about whether the information needed to improve mR@50 exists.

**`tools/predicate_discriminability.py` — the information IS there.**
Balanced binary discrimination between predicate pairs the prior actually
confuses, using 8 box-geometry features and 9 parameters:

| pair | AUC |
|---|---:|
| `walking on` vs `on` | 0.878 |
| `covered in` vs `on` | 0.867 |
| `above` vs `on` | 0.865 |
| `riding` vs `on` | 0.850 |
| `under` vs `on` | 0.832 |
| `part of` vs `of` | 0.559 ← annotation style, correctly ~chance |
| `near` vs `next to` | 0.580 ← annotation style, correctly ~chance |

10 of 22 confusions are strongly separable. The two that are not are exactly
the pairs no observer could resolve.

**`tools/candidate_reranking_analysis.py` — but it cannot be converted.**
Same features, same data:

> 200 dedicated pairwise probes → **0.0 %** of oracle headroom captured
> shared linear reranker → mR@50 11.90 (−10.40)

Both cannot be simply true. Either the AUC is an artifact, or something
between "separable" and "ranked" is destroying the signal.

## 2. The proposed resolution

The AUC measurement is **balanced**: equal numbers of each class. The ranker
faces the **natural** distribution, where the prior is extremely confident and
extremely skewed — `on` is 45.1 % of the prior's argmax predictions on
validation, and `H(pred|subj,obj)` = 3.69 bits of a possible 3.912.

For geometry to flip `on` → `above` on a pair where the prior says
`P(on) ≈ 0.7, P(above) ≈ 0.05`, the discriminator must supply
`log(0.7/0.05) ≈ 2.6` nats of evidence. An AUC-0.865 linear probe does not
produce log-odds of that magnitude on most instances. It is *right* more often
than chance, and *never confident enough to matter*.

Worse, every scorer tried so far was trained with cross-entropy on the natural
distribution. The loss-minimising behaviour under 45 % head dominance is to
**agree with the prior**. The reranker was trained, in effect, to reproduce the
thing it was meant to correct.

**H2.** The failure is a *decision-rule* failure, not an information failure.
mR@50 is an unweighted mean of per-class recalls, so it rewards deliberately
trading head-class recall for tail-class recall. No argmax-of-score rule under
the natural prior will do that, however good its features. Applying an explicit
class-balancing adjustment to the decision — subtracting `τ · log P(predicate)`
from the combined score — should convert measurable headroom into mR@50.

## 3. Why this is worth testing before any architecture work

- It is **cheap**: CPU, no training, no GPU, one pass over the prior's own
  scores. Minutes, not hours.
- It is **decisive in both directions** (§5).
- It tests a mechanism that the existing evidence *predicts* should work, and
  that nothing measured so far has ruled out. `logit_adj_tau` and
  `tail_logit_adjustment` exist in `config.py` and were **disabled** (0.0 /
  false) in the historical checkpoint's own configuration.
- It is a **necessary control** for a result measured earlier today: the
  historical checkpoint scored +3.82 mR@50 over the train-derived prior on a
  matched 240-image subset. If a pure recalibration of the prior — with no
  visual input whatsoever — reproduces a gain of that size, then that +3.82 is
  not evidence of visual understanding and must not be reported as such.

That last point is the reason to run this **now** rather than later.

## 4. The experiment

Applied to the prior's scores alone (no model, no vision):

```
score(p | s,o) = log P(p | s,o)  −  τ · log P(p)
```

- `τ = 0` is exactly the existing prior baseline (66.59 / 22.30), so any
  movement is attributable to the adjustment alone.
- `τ` swept over a grid fixed in advance: 0, 0.1, 0.25, 0.5, 0.75, 1.0.
- `log P(p)` is the **train-split** marginal. Using the evaluation split's
  marginal would leak.
- Full validation split, same protocol, same matching code path.
- **Both R@50 and mR@50 reported at every τ**, plus head/body/tail.

## 5. Success and failure criteria — fixed before running

| Outcome | Criterion | Interpretation |
|---|---|---|
| **H2 SUPPORTED** | mR@50 gain ≥ +3.0 points at some τ, with R@50 loss ≤ 3.0 points | The information was there; the decision rule was discarding it. Formulation is the bottleneck. |
| **H2 REJECTED** | no τ yields ≥ +1.0 mR@50, or every mR gain costs more R than it gains mR | Recalibration is not the missing piece. The reranking negative stands. |
| **PARETO-ONLY** | mR@50 rises but R@50 falls at least as much | The adjustment only moves along a known trade-off. Report it as such; it is **not** a research contribution. |

The PARETO-ONLY row exists deliberately. Raising a class-averaged metric by
sacrificing head recall is a well-known and often vacuous manoeuvre. If that
is all this does, it must be reported as a trade-off and not dressed up as an
improvement.

## 6. What a positive result would and would not mean

**Would mean:** the current flat-softmax-plus-argmax formulation provably
discards recoverable signal, and the project's forward direction is
*formulation*, not encoder capacity or model scale.

**Would NOT mean:** the model is good, the architecture is validated, or
appearance is useless. It would specifically mean that a large part of what
has been attributed to "the model beating the prior" is recalibration, and
must be re-attributed.
