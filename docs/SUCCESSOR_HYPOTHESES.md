# Successor to PURE — four hypotheses, none yet committed to

**Design activity only.** No architecture is implemented, and none will be until
one hypothesis is strongly supported. Two of the four are already partly tested
*within* this cycle, and one of those came back negative — recorded here so the
list is an honest ledger rather than a menu.

The measured facts every hypothesis must respect:

| fact | source |
|---|---|
| prior = 85.8% of the model term's variance | `p42` |
| WPRD: text 0.5542, classifier 0.5728, geometry-linear 0.5961, **geometry-MLP 0.6153** | `p33`, `p39`, `p46` |
| the evaluated head reaches **47%** of the box-channel ceiling; the discarded head 63% | `p46` |
| 33.4% of rows are prior-adversarial; model fixes 7.9%, geometry 6.8%, both 10.5% | `p43` |
| within-pair variation has ~the right *magnitude*, weakly correct *direction* | `p44` |
| amplifying it recovers only +1.07 pts at ~1:1 cost | `p45` |

---

## H1 — READOUT bottleneck
*`rel_feat` contains substantially more relational evidence than either head extracts.*

- **Predicted signature:** a cross-fitted probe on `rel_feat` reaches WPRD
  materially above the classifier head's 0.5728 — the pre-registered threshold is
  **≥ C + 0.03**, and the geometry ceiling makes **≥ 0.6153** the number that
  matters practically.
- **Cheapest experiment:** `p37`, already pre-registered, **running now** on
  `p36`'s cache. **Zero additional GPU.**
- **Failure criterion:** probe < C + 0.01 ⇒ H1 dead.
- **GPU cost:** 0 (p36 already paid it).
- **Status:** **PENDING — the decisive test.**

## H2 — OBJECTIVE bottleneck
*The training loss makes reproducing P(p|s,o) the dominant gradient, so the encoder is never rewarded for within-pair discrimination.*

- **Predicted signature:** a model trained with an explicitly within-pair
  objective (contrastive across rows sharing an (s,o) group) should raise WPRD
  even at unchanged capacity, and should raise it *most* in the tail, where the
  prior gradient is weakest.
- **Cheapest experiment:** **do not retrain PURE.** Train the *smallest possible*
  head — a 2-layer MLP on frozen `rel_feat` — twice: once with plain CE (the
  current objective) and once with a within-group contrastive loss. Same
  features, same data, same capacity; the only difference is the objective. If
  the contrastive head beats the CE head on WPRD, the objective is implicated
  *without any claim about architecture*.
- **Failure criterion:** contrastive head ≤ CE head + 0.01 WPRD ⇒ H2 dead.
- **GPU cost:** ~0.5 h (or CPU-feasible on 768-d frozen features).
- **Status:** **REFUTED ON THE BOX CHANNEL** (`runs/p48`). Run exactly as
  specified above, on geometry features: contrastive **0.6020** vs CE
  **0.6163**, paired **−0.0143** [−0.0198, −0.0094]. Directly optimising WPRD's
  own surrogate scored *worse* on it than plain CE. Must be re-run on `rel_feat`
  before H2 is refuted for PURE, but plain CE now looks near-optimal at
  *extracting* within-pair signal from a given feature set — which points at H4.
  Likely mechanism, from the run's own output: only **19%** of train groups have
  ≥2 distinct predicates, so the contrastive arm trains on a small biased slice
  while CE uses every row. This may be a **supervision-scarcity** result about
  VG150 rather than an objective result.
- **Note (now outdated by `p48`):** this had been the hypothesis most consistent
  with the evidence, because `p44` showed the encoder's within-pair variation has roughly the right
  *magnitude* and the wrong *direction* — a signature of "never trained toward
  it" rather than "cannot represent it".

## H3 — PRIOR/EVIDENCE ENTANGLEMENT
*Prior and visual evidence are summed into one score, so evidence cannot be inspected, gated, or overridden.*

- **Predicted signature:** separating the streams and letting evidence override
  the prior only where the prior is uncertain should recover prior-adversarial
  rows at better than the ~1:1 rate the naive amplification achieved.
- **Cheapest experiment:** `p45` ran the naive version and returned **WEAK**
  (+1.07 pts against a registered +2.0). A margin-gated variant is the remaining
  form and would need its own registration.
- **Failure criterion:** already partially met. Any margin-gated variant must
  clear +2.0 pts with the floor held, or H3 is dead too.
- **GPU cost:** 0 — CPU on the existing cache.
- **Status:** **WEAKENED by `p45`.** Not dead, but the naive form failed and the
  burden of proof has risen.

## H4 — REPRESENTATION bottleneck
*`rel_feat` genuinely does not encode within-pair relational structure; no readout or objective can recover it.*

- **Predicted signature:** every probe on `rel_feat` — linear, MLP, and
  `rel_feat`+geometry jointly — stalls at or below the geometry ceiling of
  0.6153, and `R9_relfeat_plus_geom ≈ R8_geom`.
- **Cheapest experiment:** the same `p37`, which reports exactly those arms.
- **Failure criterion:** any probe clearly exceeding 0.6153 ⇒ H4 dead.
- **GPU cost:** 0.
- **Status:** **PENDING**, and it is the direct complement of H1 — `p37` decides
  between them in one run.

---

## The decision rule, fixed now

| `p37` outcome | conclusion | next |
|---|---|---|
| probe ≥ C + 0.03 **and** > 0.6153 | **H1 supported** | build a readout-only successor; no retraining |
| probe ≥ C + 0.03 but ≤ 0.6153 | H1 partly supported, still below boxes | readout fix is real but insufficient; test H2 |
| probe < C + 0.01 | **H4 supported, H1 dead** | test **H2** — the objective, on frozen features |
| any probe ≫ 0.6153 | H4 dead | the encoder is fine; the whole failure is downstream |

**If H2 then also fails**, the honest conclusion is that VG150's supervision
cannot teach within-pair relational discrimination beyond what boxes provide,
and the contribution is the diagnosis and the benchmark — **not a model**. That
is an acceptable outcome and is written down here so it cannot be quietly
avoided later.

## Architecture family, if and only if one of these lands

Not commitments. Listed so the eventual choice is visibly derived from a
measurement rather than from taste:

- **prior stream + evidence stream, scored separately** (targets H3)
- **prior-controlled readout** — evidence trained on the group-centred target so
  it can never re-learn the prior (targets H2, and is the natural implementation
  of what WPRD measures)
- **margin-aware override** — evidence permitted to move the argmax only where
  the prior's margin is small (targets H3; `p28` showed 61.5–86.9% of the
  model's beneficial flips already live in the lowest margin decile)
- **counterfactual / within-group contrastive training** (targets H2)

**Explicitly excluded:** more parameters, deeper transformers, a larger CLIP,
more fusion layers. Nothing measured implicates capacity, and `p46` shows a
20-parameter-per-class linear model on rectangles already beats the 79.9M-
parameter checkpoint's readout.
