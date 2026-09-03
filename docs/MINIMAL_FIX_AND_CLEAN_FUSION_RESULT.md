# `p65` — the minimal fix is neutral; cleaned fusion is actively harmful

Run: exit 0, 342 s, CPU only, no GPU. Pre-registered in
`docs/MINIMAL_FIX_AND_CLEAN_FUSION_PREREGISTRATION.md` (commit `22d21a3`),
before this tool's first run. All six gates PASS, including the two
reproduction gates (`G4`, `G5`) confirming this run's `A_relfeat`/`B_geometry`
match `p60`'s to within 0.0000/0.0000 — the estimator and folds are identical
to `p60`'s by construction, so every delta below is directly comparable.

## Result

| arm | WPRD | head–head | body–body | tail–tail |
|---|---|---|---|---|
| `A_relfeat` (reproduced from `p60`) | 0.5732 | 0.5823 | 0.6064 | 0.4859 |
| `B_geometry` (reproduced from `p60`) | **0.5976** | 0.5988 | 0.6200 | 0.5820 |
| **`D2_relfeat_plus_dxdy`** (minimal fix) | 0.5782 | 0.5856 | 0.6238 | 0.4899 |
| **`E_groupcentered_relfeat_plus_geometry`** (cleaned fusion) | 0.5656 | 0.5687 | 0.6022 | 0.3978 |
| `N_shuffled` | 0.4992 | — | — | — |
| `P_prior` | 0.5000 | — | — | — |

```
PRIMARY   delta_min        = D2 - A_relfeat = +0.0050  ->  MINIMAL-FIX-NEUTRAL
SECONDARY delta_clean_fuse = E - B_geometry = -0.0320  ->  CLEAN-FUSION-HARMFUL
```

## Reading

**The minimal fix does not work.** Handing the readout the exact two numbers
`p57` showed the encoder discards (`dx_rel` R²=0.052, `dy_rel` R²=0.223)
moves WPRD by +0.005 — inside the pre-registered neutral band, and a tenth
of the +0.02 threshold. `D2` does edge `A_relfeat` in body–body (0.6238 vs
0.6064) and marginally in head–head, but not by enough to register, and
tail–tail is essentially unchanged (0.4899 vs 0.4859). **Candidate B**
("`rel_feat` + `[dx_rel, dy_rel]`") from the successor-ladder list **is
closed**: the two missing numbers, added back explicitly, are not the
bottleneck by themselves.

**Cleaned fusion is worse than geometry alone, not merely redundant with
it.** `p37`'s R5 (group-centred `rel_feat` alone) is this project's best
`rel_feat` arm at 0.5807 — removing the (s,o) group mean helps a raw
readout. The hypothesis this run tested was whether that same cleaning,
applied before fusing with geometry, would let the fused arm finally beat
geometry alone, since every prior fusion attempt used the *contaminated*
representation. It does not: `E` reads 0.5656, **0.032 below** geometry
alone, past the harmful threshold in the wrong direction. Concatenating even
the cleaned representation with geometry costs the fused arm 3.2 WPRD
points relative to just using geometry and ignoring `rel_feat` entirely.
**Candidate C** ("`rel_feat` + normalised box geometry") **is closed for the
second time**, now including its most favorable variant.

`E`'s tail–tail cell reads 0.3978 — *below* the shuffled null's tail–tail
(0.5965) and below chance. Per this project's standing convention
(`docs/DIAGNOSIS.md`, `docs/PAPER_A_FREEZE_AUDIT.md`), tail–tail is
underpowered on every arm measured so far and this single point must not be
read as "cleaned fusion actively anti-discriminates in the tail" — it is
recorded, not interpreted, pending a cell-count check the way `p35`/`p59`
report theirs.

## What this closes, cumulatively

| candidate (task's successor ladder) | status before `p65` | status after `p65` |
|---|---|---|
| A. baseline PURE | — | unchanged, the thing being explained |
| **B. `rel_feat` + `[dx_rel, dy_rel]`** | untested (p58 void) | **CLOSED — MINIMAL-FIX-NEUTRAL** |
| **C. `rel_feat` + normalised box geometry** | closed for raw `rel_feat` (`p37` R9, `p60` `C_fusion`) | **CLOSED for the cleaned representation too — HARMFUL, not merely redundant** |
| D. `rel_feat` + small learned spatial MLP | partially addressed — `p60`'s `B_geometry` IS a learned MLP, on geometry alone, and already exceeds every representation-inclusive arm | a *frozen-probe* MLP on `rel_feat` alone tops out at 0.5732 (`p55`'s `A_ce_all` = this run's `A_relfeat`); no additive combination with geometry beats geometry alone under any estimator or cleaning tried |
| E. gated fusion (learned gate between `rel_feat` and geometry) | untested | **weakened, not closed**: a good estimator (MLP+CE, the same family used throughout) had every opportunity to learn to ignore `rel_feat` inside `C_fusion`/`E` and did not recover geometry's own level — evidence against a gate helping, since gating is a special case of the same function class these MLPs can already express, but a *hard*, explicitly-architected gate is not literally what was fit here |
| F. explicit spatial auxiliary loss | not attempted; would require retraining, out of scope for a frozen-probe analysis |

**Every additive, frozen-probe combination of `rel_feat` and geometry tried
across `p37`, `p58` (void), `p60`, and `p65` — four attempts, two estimator
families, both the raw and the cleaned representation — either fails to beat
geometry alone or actively underperforms it.** This is now a heavily
over-determined result for the *additive, frozen-readout* route specifically.

## What remains open

This is a readout-probe result on a **frozen** encoder. It cannot distinguish
"the information genuinely is not there in any usable form" (C1) from "a
*jointly trained* encoder, with these features injected during training
rather than concatenated post-hoc, could learn a different, better
combination" (a variant of C2/C3 this run does not test). Per
`docs/SUCCESSOR_HYPOTHESES.md`'s own decision rule, stated before this run:
if the additive/frozen route fails, "the honest conclusion is that VG150's
supervision cannot teach within-pair relational discrimination beyond what
boxes provide, and the contribution is the diagnosis and the benchmark — not
a model." `p65` is one more data point toward that conclusion, not a
final one — the joint-training question is untested and would require an
actual (small) GPU training pilot, not a frozen-probe CPU analysis, to
address honestly.

Single checkpoint, validation split, PredCls with GT pairs, frozen `rel_feat`
from `p36`'s GPU pass — no new GPU work in this run.
