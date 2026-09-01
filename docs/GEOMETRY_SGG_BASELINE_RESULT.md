# Geometry on the field's own metric (`runs/p41`) — a corrective

Run: exit 0, 205 s. Full `p24` cache. **This result narrows the claim made in
`runs/p38`–`p40` and the first version of `docs/DIAGNOSIS.md`.**

## Why this was run

Every finding in this cycle up to here used **WPRD**, a metric this project
invented. That is the right instrument for isolating relational grounding, and
it is also a fair objection: a new metric can always be built to make a model
look bad. So this run drops WPRD entirely and uses **R@50 / mR@50 under the
evaluator's exact composition**, swapping the checkpoint's model term for the
train-fitted geometry probe.

## Result — the model WINS at its own operating point, geometry wins at high tau

Pareto gap against the prior-only tau frontier:

| tau | prior + **MODEL** | best prior + **GEOMETRY** | winner | prior + MODEL + GEOMETRY |
|---|---|---|---|---|
| 0.00 | **+0.901** | −1.048 (w=0.25) | **MODEL** +1.948 | +0.901 (w=0) — geometry adds nothing |
| 0.05 | **+2.736** | +1.020 (w=0.25) | **MODEL** +1.716 | +2.736 (w=0) — geometry adds nothing |
| 0.10 | +0.864 | **+2.690** (w=0.25) | **GEOMETRY** +1.826 | **+3.423** (w=0.25) |
| 0.20 | +1.071 | **+3.353** (w=1.00) | **GEOMETRY** +2.281 | **+4.051** (w=0.75) |

**At tau = 0 and 0.05 — the checkpoint's actual operating region — the trained
model beats geometry, and adding geometry to it helps not at all.**

## The reconciliation, and it is the interesting part

`p39`/`p40` are not contradicted. Both are true:

| metric | what it rewards | winner |
|---|---|---|
| **WPRD** | relational *discrimination*, prior cancelled exactly | **geometry**, by +0.023 to +0.042, and by +0.12 in tail–tail |
| **R@50/mR@50** | *calibrated argmax* under a dominant prior | **the model**, at tau ≤ 0.05 |

**Discrimination is not calibration.** Geometry separates tail predicates well
(WPRD 0.62–0.64 in the tail vs the model's 0.51–0.53) and yet *lowers* tail mR
when composed (tau=0, w=0.25: tail 7.78 → 6.72). It can tell the classes apart
and still never predict them, because its scores are not scaled to compete with
the prior for the argmax. The checkpoint's output, whatever its discriminative
quality, is already aligned to complement the prior — it was trained to be.

So the corrected statement is narrower than the one `p39` invited:

> The checkpoint's *advantage on the field's metric does not come from
> relational discrimination* — a linear model on two rectangles discriminates
> better, especially in the tail. What the checkpoint supplies that geometry
> does not is a term already calibrated against the prior.

That is **not** the same as "the model is worse than boxes", and the earlier
phrasing in `docs/DIAGNOSIS.md` §5 has been corrected accordingly.

## The complementarity is real and worth noting

At tau ≥ 0.1 the best arm measured anywhere in this cycle is
**model + geometry** (+3.423 at tau=0.1, **+4.051** at tau=0.2), beating either
alone. The two carry partially non-overlapping information — consistent with
`p40`, where geometry wins the spatially decidable contrasts (above vs next to,
behind vs under) and the model wins the functional ones (holding vs wearing,
of vs part of).

## Limitations, including a bug in this run

1. **The nested arm is uninformative and it is my error.** It selects the
   mixing weight `w` inside training folds using `key = (m["mR"],)` — mR alone.
   Geometry always lowers mR, so it picked `w = 0.0` on every fold at every tau.
   The rest of this project selects on `(R >= floor, mR)`. The nested numbers
   are therefore a control that did not run, not a result. It is reported as
   broken rather than re-run with a criterion chosen after seeing the grid.
2. `w` in the grid IS selected on validation, which **favours geometry**. The
   model still wins at tau ≤ 0.05 despite that handicap, which strengthens the
   model's side of the comparison.
3. Geometry is a *linear* probe. A stronger geometric model would presumably do
   better; this is a lower bound on what rectangles give you, not a proposal.
4. PredCls with GT pairs, validation split only.
