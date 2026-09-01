# Audit — "the text branch" is image-conditioned, and the discarded head is trained

Status: **TWO CLAIMS IN `RESEARCH_STATE.md` ARE WITHDRAWN.** One is an
interpretation that does not follow from the code; the other is a factual claim
that is now measured to be false.

**No previously reported number changes.** `p24`, `p26`, `p28`, `p29`, `p31` are
all exactly as reported. What changes is what they mean, and one branch that was
closed on a false premise is reopened.

Evidence: source reading (`openvocab_rel/models/relational_model.py`,
`openvocab_rel/evals.py`), `runs/p33_within_pair_discrimination`,
`runs/p34_ensemble_alpha_sweep`.

---

## 1. The withdrawn claims

`RESEARCH_STATE.md` §2b:

> **VF** `ensemble_alpha = 0.0` — the "model term" is **100% the CLIP text
> branch**; the visual classifier head contributes **exactly zero**. Everything
> the C′ work attributed to "the model" is the text path.

§3:

> There is no visual evidence in this term to convert.

§7, the sentence that closed the architecture branch:

> The quantity being converted is a pair prior in text-embedding space; the
> visual head is at weight zero and **is untrained in this checkpoint**.
> Building an architecture to exploit it would be building a second frequency
> prior.

## 2. What the code actually computes

Both predicate heads consume the **same** image-derived relational feature.
`rel_feat` is produced by `_forward_eval_batch(cfg, model, clip_model,
processor, eval_batch, ...)` — the trained relational encoder running on the
actual image.

```
text_predicate_logits(rel_feats, text_feats) = score(text_relation_features(rel_feats), text_feats)
text_relation_features(x)                    = x        # projection disabled in p24
score(rel, text)                             = normalize(rel) @ normalize(text).T
predicate_logits(rel_feats)                  = predicate_classifier(rel_feats)
```

So the "text branch" is **not a lookup table**. It is the cosine between the
trained relational encoder's output *on this image* and the predicate text
embeddings. `ensemble_alpha` chooses a **readout**, not whether vision is used:

| ensemble_alpha | which readout | image-conditioned? |
|---|---|---|
| 0.0 (the checkpoint) | cosine against predicate text embeddings | **yes** |
| 1.0 | learned linear predicate classifier | **yes** |

"The visual classifier head contributes exactly zero" is true of the
*classifier*. It is not true that the term is non-visual. Calling it "the CLIP
text branch" invited exactly the wrong reading, and the project took it.

## 3. Both claims fail their measurements

`runs/p33` measures within-pair relational discrimination (WPRD), which is
prior-free by construction — see §4 and
`docs/WITHIN_PAIR_DISCRIMINATION_RESULT.md`. 0.5 means no image-conditioned
relational information.

| arm | WPRD macro | 95% CI |
|---|---|---|
| **text head** (the evaluated model term, `alpha=0`) | **0.5542** | [0.5495, 0.5592] |
| **classifier head** (stored, discarded at `alpha=0`) | **0.5728** | [0.5681, 0.5779] |
| prior (constant within group — must read 0.5) | **0.5000** | — |
| random null | 0.5046 | — |

Two conclusions:

1. **"No visual evidence in this term" is false.** The evaluated term scores
   0.5542 with a CI that excludes 0.5 by more than 9 CI half-widths. The signal
   is real. It is also **weak** — 0.55 against a 0.5 floor and a 1.0 ceiling.
2. **"The discarded head is untrained" is false.** An untrained head scores what
   the random null scores, 0.5046. The classifier head scores **0.5728**, the
   best of every arm tested. It is trained, and it is the **better** of the two
   readouts.

`runs/p34` confirms the second point at the operating point. Sweeping
`ensemble_alpha` on the same cache:

| tau | `alpha`=0 (checkpoint) Pareto | best `alpha` | best Pareto |
|---|---|---|---|
| 0.00 | +0.901 | 0.10 | +0.981 |
| 0.05 | **+2.736** | 0.00 | +2.736 |
| 0.10 | +0.864 | 0.50 | **+4.126** |

and ΔR over the prior is higher at `alpha`=0.5 than at `alpha`=0 for **every**
tau tested (+0.744 vs +0.575 at tau=0; +0.769 vs +0.475 at 0.05; +0.644 vs
+0.137 at 0.1). WPRD rises monotonically with `alpha` (0.5542 → 0.5728), and the
operating-point gain tracks it.

**Caveat, stated before it can be mistaken for a result:** the `alpha` values in
`p34` were read off the validation split with no held-out selection. WPRD is
parameter-free and needs no selection, so *"the classifier head is better
grounded"* is the robust claim. *"alpha=0.5 is the right operating point"* is
**not** established and must not be quoted as one.

## 4. Why every earlier null said "no image conditioning"

`p26` and `p29` permute the model term within (subject, object) groups and
measured `full − pair_matched_null = +0.031 ± 0.188`. That is a correct
measurement of a diluted quantity. Of the 132,556 GT rows:

| population | rows | share | can a within-group permutation change anything? |
|---|---|---|---|
| singleton (s,o) groups | 30,918 | 23.3% | **no** — nothing to permute with |
| multi-row groups, CONSTANT GT | 26,272 | 19.8% | **no** — every row wants the same predicate |
| decidable (≥2 distinct GT) | 75,366 | 56.9% | yes |

**43.1% of the rows are structurally inert to the null.** A real effect on the
decidable rows is diluted by a factor of ~0.57 before it reaches the reported
average, and it is then further compressed by an R@50/mR@50 metric that a
dominant frequency prior has already saturated.

So `p26`/`p29` are not wrong. They bound the effect *at the additive operating
point, averaged over a population that is 43% incapable of showing it*. WPRD
measures the same model on the 56.9% where the question is answerable and with
the prior cancelled exactly, and finds the signal.

## 5. Blast radius — checked, not assumed

- **Numbers: nothing changes.** `p24` (CONFIRMED, 3/3), `p26`, `p28`, `p29`,
  `p31` stand exactly as reported. No threshold moves.
- **`RESEARCH_STATE.md` §2b "VF" bullet** — withdrawn as written; the
  `ensemble_alpha = 0.0` fact is retained, the "not visual" reading is removed.
- **`RESEARCH_STATE.md` §3 "no visual evidence in this term"** — **falsified**
  by `p33`.
- **`RESEARCH_STATE.md` §7, architecture branch** — closed on the premise that
  the term was a non-visual pair prior and the visual head untrained. Both are
  false. The branch is **reopened as an open question**, not as a plan: what is
  now established is a *weak but real* grounding signal, and weakness is a
  different problem from absence.
- **H4** moves from "not established" to **ESTABLISHED BUT WEAK**.
- **The `MR` bullet (86.87% between-pair variance) is unaffected** and is in
  fact the mechanism: most of the term is pair identity, and the 13–17% that is
  image-conditioned yields only ~0.55 AUC.
- `p27` remains withdrawn for its own separate reason; `p32` is unaffected — it
  asks whether pair statistics reproduce the term, which is orthogonal to
  whether the term is image-conditioned.

## 6. What this does NOT license

It does not license "PURE sees relations". WPRD 0.554–0.573 against a 0.5 floor
is a **weak** signal, and the operating-point evidence still shows the additive
composition converting almost none of it. The honest statement is:

> The checkpoint has genuine but weak image-conditioned relational
> discrimination. It is too weak to survive composition with a dominant
> frequency prior, and the standard metrics and nulls cannot see it at all.

Whether that weakness is a training failure, a readout failure, or a
representation-capacity failure is **not** settled by these runs. It is the next
question.
