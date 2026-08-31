# Forensics: why τ = 0.1 buys +4.03 mR@50 with no vision

CPU-only, read-only analysis. No GPU was used, no model was loaded, no
training was run, no historical artifact, manifest, dataset or pre-registered
criterion was modified, and no earlier result report was rewritten.

Companion: `docs/research_sessions/POST_B_PRIOR_CALIBRATION_ANALYSIS.md`
(interpretation, comparison to appearance, and the decision on experiment C).

**The one-sentence answer.** τ adds no information — it makes *strictly more
top-1 mistakes* (−245 net, −0.64 accuracy points) — but it moves ~1,000 of
38,053 decisions out of enormous classes into tiny ones, and because mR@50 is
an **unweighted mean over classes**, each such move is worth ~30× more to the
metric than it costs.

---

## 1. Source trace

### 1.1 The τ path, end to end

The τ result was produced by `tools/decision_rule_probe.py`. It never enters
`openvocab_rel/evals.py` (see §1.6).

```
image (not read — τ uses no pixels)
  │
  ├─ GT object labels  ex["objects"][i]["names"][0]
  │     tools/frequency_prior_baseline.py:load_split  L100-127
  │
  ├─ GT relationships  → the candidate set IS the GT pair set
  │     (PredCls / GT pairs: one candidate per GT relationship)
  │
  ▼
prior row:  log P(p | s, o)                    ┐
  FrequencyPrior.row  frequency_prior_baseline.py:L80-98
  backoff: pair  →  ½(subject+object)  →  single  →  global
  MEASURED backoff mix on this subset: pair 83.0 %, subject_object 13.2 %,
  single 3.7 %, global 0.1 %   (runs/p5_arm2_trainprior_3000/)
  │
  ▼
τ adjustment                                    ◄── THE ONLY CHANGE
  decision_rule_probe.py:evaluate_tau  L83-160
  L110   marginal = prior.global_lp            # TRAIN marginal log P(p)
  L116   adjusted[i] = row[i] - tau * marginal[i]
  │
  ▼
softmax
  L117   probs = softmax([alpha * v for v in adjusted])     alpha = 3.75
  │
  ▼
per-pair decision
  L118   best = argmax_i probs[i]              # ONE predicate per pair
  │
  ▼
cross-pair ranking key
  L122   order = sorted(pairs, key = -probs[best])
  │
  ▼
top-K truncation  (K = 20, 50, 100)            ◄── PROVABLY INERT AT K=50, §2
  L128-142  greedy match against GT (subject idx, object idx, predicate name)
  │
  ▼
metrics  L145-155
  R@K   = pooled hits / pooled GT
  mR@K  = unweighted mean of per-class recall over classes with GT > 0  (L147)
  head/body/tail = 15 / 20 / 15 by GT frequency   head_body_tail L72-79
```

### 1.2 The exact mathematical transformation

For predicate index *i*, subject label *s*, object label *o*:

```
s_i(τ)  =  log P(p_i | s, o)  −  τ · log P(p_i)
        =  log [ P(p_i | s, o) / P(p_i)^τ ]

probs   =  softmax( α · s(τ) )                  α = 3.75
pred    =  argmax_i probs_i  =  argmax_i s_i(τ)
```

Two limits identify it exactly:

- **τ = 0** → `argmax P(p|s,o)`. The MAP decision, Bayes-optimal for **pooled**
  0-1 error, i.e. for R@50.
- **τ = 1** → `argmax P(p|s,o)/P(p)` = `argmax PMI(p ; (s,o))`. This is the
  Bayes-optimal decision for **balanced** 0-1 error, i.e. for the
  class-averaged metric — mR@50.

So τ is a continuous interpolation between *the decision rule that optimises
R@50* and *the decision rule that optimises mR@50*. This is standard logit
adjustment / balanced softmax with τ as the adjustment strength. It is a
**decision-rule change, not an estimator change**: `P(p|s,o)` is untouched.

### 1.3 Does τ treat every predicate equally? **No.**

The shift applied to class *i* is `−τ · log P(p_i)`, which is a per-class
constant proportional to that class's log-rarity. MEASURED from the
train-derived prior's `global_log_probs`:

| | value | predicate |
|---|---:|---|
| max log P(p) | −1.0122 | `on` |
| min log P(p) | −7.1541 | `flying in` |
| **spread** | **6.1419 nats** | |

At τ = 0.1 the shift ranges `+0.1012` (`on`) to `+0.7154` (`flying in`) — a
**0.614-nat relative boost** to the rarest class over the most common one.
Every class is shifted; none by the same amount.

### 1.4 Does τ change entropy? Yes — but entropy is **not** the mechanism

MEASURED (`runs/p7_prior_temperature_forensics/`), mean Shannon entropy of the
per-pair softmax over 38,053 pairs:

| τ | mean H (nats), α=3.75 | classes ever emitted | mR@50 |
|---:|---:|---:|---:|
| 0.0 | 0.5567 | **49** | 21.98 % |
| 0.1 | 0.7137 | **50** | 26.00 % |
| 0.2 | 0.8750 | 50 | 28.42 % |

τ flattens the distribution, and it expands the *support*: at τ=0 one predicate
is **never** emitted as an argmax anywhere in the split, so its recall is
identically 0 and it contributes 0 to mR@50. At τ=0.1 all 50 are reachable.

**But entropy per se cannot be the cause**, and there is a clean control that
proves it. α is a pure temperature on the same scores, and softmax is monotone
in α, so α cannot change any per-pair argmax:

| α | mean H, τ=0 | R@50 | mR@50 |
|---:|---:|---:|---:|
| 3.75 | 0.5567 | **66.80 %** | **21.98 %** |
| 1.00 | 2.5441 | **66.80 %** | **21.98 %** |

A **4.6× change in entropy** moves R@50 and mR@50 by **exactly zero**. Only the
argmax matters. Entropy is a symptom of τ, not its mechanism.

### 1.5 Ranking, candidate inclusion, and top-K

- **Candidate inclusion: unchanged.** Under GT pairs the candidate set is the
  GT relationship set itself (`_extract_gt_pairs`). τ cannot add or remove a
  candidate.
- **Per-pair label choice: this is the entire effect.**
- **Cross-pair ranking key** (`probs[best]`) is changed by τ — and is inert at
  K=50, see §2.

### 1.6 Interaction with frequency bias and adaptive calibration

**With `freq_bias_alpha` (α = 3.75):** in `decision_rule_probe.py`, α is the
softmax temperature (L117) and is **provably and empirically inert** for R@50
and mR@50 (§1.4). Note this is *not* α's role in `evals.py`, where α scales the
prior term that is **added** to the model's visual term
(`_apply_frequency_bias`, `evals.py:1178-1193`) and therefore governs how much
the prior outweighs vision. Same constant, two entirely different roles.

**With adaptive calibration: none.** `decision_rule_probe.py` loads no
checkpoint and runs no calibration head.

**Inside `evals.py`, τ has never been applied.** `_apply_eval_logit_adjustment`
(`evals.py:1074-1080`) implements `logits − τ · pred_log_prior`, but:

1. it is called at `evals.py:1256` on **`cls_logits` only**, never on the text
   logits or on the frequency-bias term;
2. in the historical protocol `eval_sgg_predicate_ensemble_alpha = 0.0`, so
   `evals.py:1273` returns `(0.0 · cls_norm) + (1.0 · text_norm)` — the
   adjusted tensor is multiplied by **exactly zero**;
3. and in every recorded run `eval_logit_adj_tau = −1.0` falls back to
   `logit_adj_tau = 0.0`, so the `τ ≤ 0` early return fires anyway.

**Consequence, and it is a trap:** setting `--eval_logit_adj_tau 0.1` on the
historical protocol would produce results **byte-identical** to not setting it.
Anyone running "the model under a recalibrated prior" through that flag would
measure nothing and could easily misread the null. See
`POST_B_PRIOR_CALIBRATION_ANALYSIS.md` §6.

---

## 2. Top-K is provably inert — mR@50 *is* class-averaged top-1 accuracy

MEASURED (`runs/p7_prior_temperature_forensics/`), candidates per image on the
A′ 3,000-image subset:

| statistic | value |
|---|---:|
| mean candidates/image | 12.68 |
| median | 12 |
| **maximum** | **48** |
| images with > 20 candidates | 374 (12.47 %) |
| **images with > 50 candidates** | **0 (0.00 %)** |
| images with > 100 candidates | 0 |

No image has more than 48 candidates, so **top-50 retains every candidate in
every image**. Therefore, at K = 50:

- the cross-pair ranking key is irrelevant;
- α is irrelevant (it only enters that key);
- `R@50` is exactly **pooled top-1 predicate accuracy**;
- `mR@50` is exactly **class-averaged top-1 predicate accuracy**;
- `R@50 = R@100`, as observed in every recorded arm.

This is the structural fact that makes the rest of the analysis simple: **τ can
only change which single predicate each pair is labelled with.**

Independent confirmation: `runs/p6_prior_dominance_margin/` measured the
prior's top-1 accuracy on the same pairs as **66.80 %**, identical to the
recorded `R@50`.

---

## 3. Sweep results

`runs/p7_prior_temperature_sweep/`, produced by **unmodified**
`tools/decision_rule_probe.py` on the same 3,000-image subset as A′.
The pre-registered grid is `0, 0.1, 0.25, 0.5, 0.75, 1.0`
(`docs/DECISION_RULE_HYPOTHESIS.md`); `0.05, 0.15, 0.2, 0.3, 0.4` were added
**for forensic resolution only**. The headline τ remains 0.1 and no threshold
was changed.

| τ | R@50 | mR@50 | head | body | tail | #classes emitted | ΔR@50 | ΔmR@50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | **66.80 %** | **21.98 %** | 42.5 % | 13.5 % | 12.8 % | **49** | — | — |
| 0.05 | 66.47 % | 24.22 % | 42.6 % | 16.6 % | 16.0 % | 50 | −0.33 | **+2.24** |
| **0.10** | **66.16 %** | **26.00 %** | 42.7 % | 18.3 % | 19.6 % | 50 | **−0.64** | **+4.03** |
| 0.15 | 63.64 % | 27.31 % | 42.3 % | 20.4 % | 21.4 % | 50 | −3.16 | +5.33 |
| 0.20 | 61.95 % | 28.42 % | 42.1 % | 22.5 % | 22.7 % | 50 | −4.85 | +6.44 |
| 0.25 | 59.29 % | 30.58 % | 40.6 % | 25.6 % | 27.2 % | 50 | −7.51 | +8.60 |
| 0.30 | 56.71 % | 33.41 % | 39.5 % | 30.1 % | 31.8 % | 50 | −10.10 | +11.44 |
| 0.40 | 51.12 % | 35.98 % | 37.0 % | 33.8 % | 37.9 % | 50 | −15.68 | +14.00 |
| 0.50 | 44.64 % | 38.15 % | 33.8 % | 39.1 % | 41.3 % | 50 | −22.16 | +16.18 |
| 0.75 | 26.96 % | **38.89 %** | 25.1 % | 42.6 % | 47.8 % | 50 | −39.84 | +16.92 |
| 1.00 | 11.81 % | 34.82 % | 14.8 % | 37.8 % | 50.9 % | 50 | −54.99 | +12.84 |

The +4.03 at τ = 0.1 reproduces the recorded value exactly.

Three features worth naming:

1. **head mR is flat through τ = 0.2** (42.5 → 42.6 → 42.7 → 42.3 → 42.1) while
   body nearly doubles and tail nearly doubles. The early gain is *not* paid
   for out of head recall.
2. **R@50 falls monotonically and without exception.** τ never improves
   accuracy at any value.
3. **mR@50 peaks at τ ≈ 0.75, not τ = 1.** If `P(p|s,o)` were the true
   posterior, τ = 1 would be Bayes-optimal for the balanced metric. It is not,
   because the prior is a smoothed, backed-off empirical estimate; over-
   correcting amplifies its estimation noise on exactly the rare classes it is
   least certain about. The interior optimum is evidence that the prior is
   *imperfectly calibrated*, which is itself informative.

### Control: τ without pair-conditioning does nothing

`runs/p7_tau_global_control/` — same sweep with `--mode global`, so the score
uses only the marginal: `s_i = (1 − τ) · log P(p_i)`.

| τ | R@50 | mR@50 | #classes emitted |
|---:|---:|---:|---:|
| 0.0 | 36.53 % | 2.00 % | 1 |
| 0.1 | 36.53 % | 2.00 % | 1 |
| 0.5 | 36.53 % | 2.00 % | 1 |
| 1.0 | 0.85 % | 2.00 % | 1 |

Exactly as the algebra predicts: for τ < 1 the argmax is `on` for every pair
(R@50 = 36.53 % = `on`'s GT share, mR@50 = 1/50 = 2.00 %); at τ = 1 all scores
tie. **τ is not "predict rare classes".** It re-ranks strictly *within* the
pair-conditional distribution, and it is worthless without it.

---

## 4. Predicate-wise analysis

`runs/p7_prior_temperature_forensics/`, τ = 0 → τ = 0.1, on 38,053 GT pairs.
Buckets are `decision_rule_probe.py`'s 15/20/15 split by GT frequency (**not**
`evals.py`'s 20 %/60 %/20 % split — the two differ; see
`POST_B_SOURCE_FORENSICS.md` §5 issue #3).

Classes with n < 25 are flagged; their deltas turn on one or two instances and
must not be over-read.

| predicate | bucket | n | raw recall | calibrated recall | Δ |
|---|---|---:|---:|---:|---:|
| `covered in` | tail | 46 | 10.9 % | 41.3 % | **+30.4** |
| `riding` | body | 140 | 18.6 % | 45.7 % | **+27.1** |
| `walking on` | tail | 56 | 0.0 % | 26.8 % | **+26.8** |
| `eating` | tail | 57 | 50.9 % | 73.7 % | **+22.8** |
| `watching` | body | 61 | 26.2 % | 44.3 % | **+18.0** |
| `playing` | tail | 59 | 66.1 % | 76.3 % | +10.2 |
| `between` | body | 63 | 14.3 % | 22.2 % | +7.9 |
| `made of` | tail | 41 | 48.8 % | 56.1 % | +7.3 |
| `painted on` | body | 65 | 0.0 % | 6.2 % | +6.2 |
| `standing on` | body | 238 | 2.5 % | 7.6 % | +5.0 |
| `wrapped around` | body | 145 | 17.9 % | 22.8 % | +4.8 |
| `from` | body | 66 | 15.2 % | 19.7 % | +4.5 |
| `using` | tail | 24 | 4.2 % | 8.3 % | +4.2 ⚠ n<25 |
| `hanging from` | body | 144 | 13.2 % | 16.7 % | +3.5 |
| `carrying` | body | 87 | 37.9 % | 41.4 % | +3.4 |
| `for` | body | 195 | 17.4 % | 20.5 % | +3.1 |
| `across` | tail | 37 | 0.0 % | 2.7 % | +2.7 |
| `with` | head | 1,127 | 7.2 % | 9.8 % | +2.7 |
| `sitting on` | head | 343 | 13.1 % | 15.7 % | +2.6 |
| … 14 further classes with +0.9 to +2.5 … | | | | | |
| `wearing` | head | 2,399 | 94.0 % | 94.5 % | +0.5 |
| `holding` | head | 712 | 63.6 % | 63.9 % | +0.3 |
| *(11 classes unchanged at exactly 0.0)* | | | | | 0.0 |
| `above` | head | 323 | 36.2 % | 35.6 % | −0.6 |
| `in` | head | 4,473 | 66.3 % | 65.5 % | −0.8 |
| `has` | head | 5,661 | 79.5 % | 78.0 % | −1.5 |
| `looking at` | tail | 42 | 2.4 % | 0.0 % | −2.4 |
| `on` | head | **13,902** | 85.8 % | 83.1 % | **−2.7** |
| `behind` | head | 885 | 48.8 % | 45.0 % | **−3.8** |

**33 classes improve, 6 degrade, 11 are unchanged.**

Four classes go from *never correct* to non-zero: `walking on` 0.0 → 26.8,
`painted on` 0.0 → 6.2, `across` 0.0 → 2.7, `parked on` 0.0 → 1.6. Seven
classes stay at exactly 0.0 % in both arms (`laying on`, `lying on`,
`mounted on`, `on back of`, `walking in`, `belonging to`, `part of`) — τ does
not reach them at all.

`looking at` is the one tail class that *degrades* (2.4 → 0.0, n=42): it lost
its single correct prediction. That is a one-instance move and should be read
as noise, not as a systematic effect.

---

## 5. The mechanism, quantified

### 5.1 Flip accounting

Every pair whose argmax changes between τ=0 and τ=0.1
(`runs/p7_prior_temperature_forensics/`):

| | count | share of 38,053 |
|---|---:|---:|
| argmax changed | **1,584** | 4.16 % |
| ├ wrong → **right** | **376** | 0.99 % |
| ├ **right** → wrong | **621** | 1.63 % |
| └ wrong → wrong | 587 | 1.54 % |
| **net top-1 change** | **−245** | **−0.644 pp** |

Largest flows: `has`→`with` 116, `on`→`riding` 106, `on`→`of` 104,
`on`→`walking on` 81, `on`→`in` 72, `on`→`sitting on` 49, `on`→`has` 43,
`on`→`standing on` 39, `on`→`next to` 32, `has`→`wearing` 31, `on`→`eating` 30,
`in`→`covered in` 29.

**τ makes strictly more mistakes than it fixes.** It cannot be adding
information: an information-adding transformation would raise accuracy, and
this one lowers it by exactly the −0.64 R@50 points the sweep reports.

Only **997 pairs (2.62 %)** — the 376 + 621 that changed correctness — decide
the entire +4.03 mR@50 movement. The other 587 flips change no class's recall.

### 5.2 Leverage — why 997 pairs are worth 4 points

mR@50 is an **unweighted mean over 50 classes**, so a pair moved between
classes changes each class's recall by 1/n of that class.

| | classes | GT instances | mean class size | summed Δrecall |
|---|---:|---:|---:|---:|
| improved | 33 | 11,971 | **362.8** | **+2.1318** |
| degraded | 6 | 25,286 | **4,214.3** | **−0.1178** |
| net | | | | +2.0140 |

`+2.0140 / 50 = +4.03 mR points`, reproducing the sweep exactly.

Per instance:

```
gain:  +2.1318 recall  /  376 instances  =  +0.005670 per instance
cost:  −0.1178 recall  /  621 instances  =  −0.000190 per instance

per-instance leverage ratio  =  29.9×
```

**A correct answer moved into a rare class is worth ~30 times more to mR@50
than a correct answer lost from a common class costs.** The degrading classes
are 11.6× larger on average, and they were also more numerous per flip, giving
the ~30× per-instance figure.

That is the whole mechanism. τ trades 621 head-class hits for 376 rare-class
hits — a strictly worse trade in accuracy — and the metric pays 30:1 for it.

### 5.3 What it is, and what it is not

| candidate explanation | verdict | evidence |
|---|---|---|
| **Class-frequency compensation acting on the decision boundary** | **THIS IS IT** | τ is exactly logit adjustment; the shift is `−τ·log P(p)`, proportional to class log-rarity, spread 6.14 nats (§1.3); the global-mode control shows it does nothing without pair conditioning (§3) |
| Probability sharpening / flattening | **symptom, not cause** | entropy rises 0.5567 → 0.7137, but α changes entropy 4.6× with **zero** metric effect (§1.4) |
| Entropy correction | **rejected** | same control |
| Ranking correction (adding information) | **rejected** | net top-1 **falls** by 245 pairs; τ is strictly less accurate (§5.1) |
| Prior miscalibration | **partially — a real secondary finding** | if `P(p|s,o)` were the true posterior, τ=1 would be optimal for the balanced metric; the measured optimum is τ≈0.75, so the prior is imperfectly calibrated on rare classes (§3) |
| Top-K interaction | **rejected, provably** | no image exceeds 48 candidates; top-50 keeps everything (§2) |
| Support expansion | **real but minor** | 49 → 50 classes ever emitted; four classes move off exactly-zero recall (§4) |
| Something else | not required | the accounting closes to the reported value exactly (§5.2) |

### 5.4 Why one scalar can move a class-averaged metric so strongly

Because the decision rule and the metric were optimising different losses.

`argmax P(p|s,o)` minimises pooled 0-1 error. `mR@50` is (up to the top-K
inertness established in §2) one minus the **balanced** 0-1 error. The
Bayes-optimal rule for the balanced loss is `argmax P(p|s,o)/P(p)` — τ = 1.
Running at τ = 0 means the system was using the wrong decision rule *for the
metric being reported*, and the gap between the two rules is spanned by a
single scalar because the two losses differ by exactly one per-class term.

The size of the gap is set by how skewed the class distribution is. Here `on`
holds 45.1 % of the prior's argmax mass and 36.5 % of the GT, against a rarest
class at 0.06 %. With a 6.14-nat marginal spread, a very small τ already
reorders the near-ties, and near-ties are exactly where rare classes lose.

---

## 6. Limitations

1. **One split, one subset.** All numbers are the first 3,000 validation
   images (38,053 GT triplets, all 50 predicates present, rarest n=22). The
   full-split τ result (+4.11 mR@50, `runs/p4_decision_rule_sweep/`) is
   consistent but was measured separately.
2. **Rare-class deltas are noisy.** `using` (n=24) and `part of` (n=22) are
   flagged in §4; `looking at` (n=42) moves on a single instance. The
   aggregate accounting in §5.2 does not depend on any individual rare class.
3. **This analyses the prior-only path.** `tools/decision_rule_probe.py` does
   not load a model. Nothing here measures how τ would interact with the
   model's visual term — and §1.6 shows that interaction has never been run.
4. **Two bucketings coexist.** `decision_rule_probe.py` uses 15/20/15 by GT
   frequency; `evals.py:1310` uses top-20 %/bottom-20 % of the evaluated
   split. head/body/tail numbers are not comparable across the two tools.
5. **mR@50 = class-averaged top-1 accuracy is a property of *this* protocol**
   (GT pairs, ≤ 48 candidates/image, one predicate emitted per pair). It would
   not hold under SGDet, nor under `predcls_multi`, where K is live.
6. **τ = 0.1 is not claimed optimal.** It is the pre-registered operating
   point from `docs/DECISION_RULE_HYPOTHESIS.md`, chosen for its R@50 budget.
   The sweep shows mR@50 keeps rising to τ ≈ 0.75; nothing here re-selects τ.
