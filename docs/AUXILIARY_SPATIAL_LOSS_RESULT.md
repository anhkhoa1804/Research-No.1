# `p67` — the auxiliary loss does not help either: AUX-LOSS-NEUTRAL

Run: exit 0, 390 s, CPU only, no GPU. Pre-registered in
`docs/AUXILIARY_SPATIAL_LOSS_PREREGISTRATION.md` (commit `9a428f6`). All
gates PASS, including exact reproduction of `p60`'s `A_relfeat` anchor at
`lambda=0.0`.

## Result

| arm | WPRD | head–head | body–body | tail–tail |
|---|---|---|---|---|
| `lambda_0.0` (= `A_relfeat`, gate) | 0.5732 | 0.5823 | 0.6064 | 0.4859 |
| `lambda_0.5` | 0.5730 | 0.5840 | 0.5924 | 0.4209 |
| `lambda_1.0` | 0.5731 | 0.5817 | 0.5948 | 0.4985 |
| `lambda_2.0` | 0.5721 | 0.5788 | 0.5852 | 0.5105 |
| `P_prior` | 0.5000 | — | — | — |
| `N_shuffled` | 0.4950 | — | — | — |

```
delta_aux = WPRD(lambda=1.0) [0.5731] - WPRD(lambda=0.0) [0.5732] = -0.0001
-> AUX-LOSS-NEUTRAL
```

Every `lambda` reads within 0.0011 of the baseline — flat, not a gradient
in either direction. This is not a case of "the best lambda wasn't tried":
the sweep spans a 4x range in relative loss weight and the response is
uniformly flat.

## Reading

Forcing the shared trunk to reconstruct `dx_rel`/`dy_rel` alongside the
predicate label does not move classification-relevant discrimination at
all, even though `p66` showed the trunk's input (`rel_feat`) nonlinearly
carries real information about `dy_rel` (R²=0.292) that a plain classifier
never learns to prioritise on its own. Put the two together: **the
information can be extracted from `rel_feat` by a function trained
specifically to extract it, but forcing the classification pathway to share
a trunk with that extraction does not transfer any of it into
better predicate discrimination.** The two objectives do not conflict
destructively (WPRD does not drop, `lambda=2.0`'s -0.0011 is inside noise)
— but they also do not synergise. This is a cleaner, more specific version
of `p65`'s finding: it is not merely that concatenating the raw numbers
doesn't help (a readout-*input* fix); explicitly supervising the readout
*to retain* the relevant geometric quantity doesn't help either (a
readout-*training-signal* fix). Both forms of the cheapest, most direct
"give the readout what it's missing" intervention have now failed.

## Consequence for the successor ladder — the readout-fix branch is closed

Per the pre-registration's own stated consequence for this outcome: **every
additive, frozen-readout, and readout-training-signal intervention this
programme can cheaply test has now failed** —

| intervention | run | result |
|---|---|---|
| concatenate raw geometry, linear probe | `p37` R9 | destroys information (0.5735 < 0.5961) |
| concatenate raw geometry, matched MLP | `p60` `C_fusion` | destroys information (0.5922 < 0.5976) |
| concatenate `dx_rel`/`dy_rel` only, matched MLP | `p65` `D2` | NEUTRAL (+0.0050) |
| concatenate geometry with CLEANED (group-centred) `rel_feat` | `p65` `E` | HARMFUL (-0.0320) |
| auxiliary regression loss on `dx_rel`/`dy_rel`, shared trunk | `p67` | **NEUTRAL** (-0.0001, flat across a 4x lambda sweep) |

Six independent cheap tests, two estimator families, three distinct
intervention *kinds* (input concatenation, representation cleaning,
training-signal supervision) — none moves WPRD by a meaningful amount in the
helpful direction. This is now a strongly over-determined result for
anything that leaves the encoder frozen.

## What remains, and why it is not automatically justified

The one class left untested is a **jointly retrained encoder** — the
visual/CLIP pathway itself, trained (or fine-tuned) with gradient pressure
from a spatial signal, rather than a frozen `rel_feat` probed or
auxiliary-supervised after the fact. This is qualitatively different: a
frozen probe can only extract what is already linearly-or-nonlinearly present
in a fixed representation, while a jointly trained encoder could in
principle reorganise its features entirely. That possibility is real and
this run does not close it.

**But it is weaker evidence for that pilot's likely payoff than it would
have been before `p65`/`p67`, not stronger.** If the readout could not
convert an information-theoretically-real, moderate-sized signal (`dy_rel`
R²=0.292) into any WPRD movement even when *directly and explicitly*
supervised toward it, the bottleneck this specific pair of experiments
locates is at least partly downstream of "does the encoder retain the
information" — it is about whether the classification task, and the
representation geometry a 51-way softmax induces, can *use* spatial
information at all in the region these frozen features occupy. A GPU pilot
that only changes where the spatial signal enters (encoder vs. readout)
without addressing that downstream question has a real but reduced prior of
success, and per this session's directive ("do not run an expensive
experiment unless its outcome can change Paper B or Paper C"), it is not
recommended as the immediate next step. See `docs/TRACK_B_C_ACTION_QUEUE.md`
for the updated ranking.

Single checkpoint, validation split, PredCls with GT pairs, frozen
`rel_feat` — no GPU work in this run.
