# Pre-registration — `p65`, minimal `dx_rel`/`dy_rel` fix and cleaned-fusion, under the matched estimator

Status: **PRE-REGISTERED**, committed before the tool exists and before any
number exists. CPU only. Reads `runs/p36_relfeat_cache/pair_logits_relfeat.pt`
(already on disk). No GPU, no training beyond the small cross-fitted probes
this project already runs routinely (`p55`/`p58`/`p60`'s estimator).

## Why this run is owed

`docs/GEOMETRY_FUSION_PREREGISTRATION.md` (`p58`) registered exactly two
questions this run answers, plus one this run adds. `p58`'s own gate Y5
**failed** (the regularisation grid was saturated at its floor on every
fold), so **by its own registration, none of `p58`'s numbers are
reportable** — not "reportable but weak," not citable at all. `p60` then
showed the *cause* was the estimator (ridge-on-one-hot is a poor classifier
surrogate), not the fitting regime, and re-ran two of `p58`'s arms
(`B_geometry`, `C_fusion`) under a real cross-fitted MLP+softmax-CE
estimator — but `p60` did **not** re-run `p58`'s `D_relfeat_plus_dxdy` arm
(the minimal `p57`-motivated fix) under the corrected estimator. **That
specific, previously-identified question has therefore never been validly
tested.** This run closes that gap, and adds one more arm motivated by `p37`'s
own finding.

`p37` found that removing the (s,o) group mean from `rel_feat` before readout
(`R5_residual`) is the single best `rel_feat` arm (0.5807), beating even the
classifier head — but `p37`/`p58`/`p60` have only ever fused **raw**
`rel_feat` with geometry (`C_fusion`, both regimes), never the **cleaned**
(group-centred) `rel_feat`. Since raw-fusion has now failed twice under two
different estimators (`p37`: 0.5735 < 0.5961; `p60`: 0.5922 < 0.5976), and
`R5` shows the group mean is an *active distractor* for readout, this asks
whether fusion fails because geometry and rel_feat are genuinely redundant,
or because raw rel_feat's prior-driven variance is drowning out whatever
complementary residual signal it has — a distinction `p58`/`p60` could not
make because neither ever fused the cleaned representation.

## Design — reuses `p60`'s exact machinery, unmodified

Same file (`tools/estimator_matched_geometry.py`) is **not modified** — its
`p60` artifact stands as reported. This is a new tool
(`tools/estimator_matched_geometry_v2.py`) that imports the same estimator
(`objective_ablation_relfeat.mlp`, AdamW, identical hyperparameters:
hidden=256, epochs=20, lr=2e-3, l2=1e-4), the identical 5 image-level folds
(`objective_ablation_relfeat.folds_of(B, 0)`, expected sizes
`[26483, 26856, 27190, 26586, 25441]`), and the identical WPRD/CI/stratified
reporting path, so every number is comparable to `p60`'s by construction.

### Arms

| arm | features | dims | role |
|---|---|---|---|
| `A_relfeat` | `rel_feat`, standardised | 769 | reproduced from `p60`, gate anchor |
| `B_geometry` | 19 box features, standardised | 20 | reproduced from `p60`, delta baseline |
| **`D2_relfeat_plus_dxdy`** | `rel_feat` (standardised) + `dx_rel`, `dy_rel` (standardised) | 771 | **the minimal `p57`-motivated fix, re-tested under a real estimator for the first time** |
| **`E_groupcentered_relfeat_plus_geometry`** | group-centred `rel_feat` (standardised rel_feat minus its per-(s,o)-group mean) + full standardised geometry | 788 | **the cleaned-fusion question `p37`'s R5 motivates but never tested** |
| `N_shuffled` | `rel_feat`, shuffled labels | 769 | chance control, reproduced from `p60` |
| `P_prior` | the prior | — | must read exactly 0.5000 |

`dx_rel`/`dy_rel` are columns 4 and 5 of `wprd_geometry_control._geom`'s
19-feature output (verified by reading the function's construction order:
`(ocx-scx)/(sw+ow), (ocy-scy)/(sh+oh)` immediately follow the four absolute
positions), taken from `geometry_features_raw(B)` and standardised with the
same z-score convention as every other feature in this pipeline. Group
identity for `E` is `within_pair_discrimination.Groups(B).pair_id` — the
same (subject, object) category grouping WPRD itself conditions on, and the
same one `p37`'s R5 and `p42`'s variance split use.

## Validity gates — if any fails, no number is reportable

| gate | requirement |
|---|---|
| **G1** | `P_prior` WPRD is exactly 0.5000 (deviation < 1e-6) |
| **G2** | `N_shuffled` within [0.49, 0.51] |
| **G3** | folds identical to `p37`/`p55`/`p58`/`p60`: `[26483, 26856, 27190, 26586, 25441]` |
| **G4** | `A_relfeat` reproduces `p60`'s `A_relfeat` = 0.5732 ± 0.005 (same estimator, same folds, same features — must match, not merely resemble) |
| **G5** | `B_geometry` reproduces `p60`'s `B_geometry` = 0.5976 ± 0.005 |
| **G6** | every arm scores every one of the 132,556 rows, finite |

## Criteria — fixed here, before any number exists

### PRIMARY — does the minimal `dx_rel`/`dy_rel` fix work, under a real estimator?

`delta_min = D2_relfeat_plus_dxdy − A_relfeat`

| verdict | condition |
|---|---|
| **MINIMAL-FIX-WORKS** | `delta_min >= +0.02` |
| **MINIMAL-FIX-NEUTRAL** | `\|delta_min\| < 0.02` |
| **MINIMAL-FIX-HARMFUL** | `delta_min <= -0.02` |

(Threshold inherited unchanged from `p58`'s registration — this is that same
question, re-run under the estimator `p60` showed was necessary.)

### SECONDARY — does fusing the CLEANED representation beat geometry alone?

`delta_clean_fuse = E_groupcentered_relfeat_plus_geometry − B_geometry`

| verdict | condition |
|---|---|
| **CLEAN-FUSION-GAIN** | `delta_clean_fuse >= +0.02` |
| **CLEAN-FUSION-NEUTRAL** | `\|delta_clean_fuse\| < 0.02` |
| **CLEAN-FUSION-HARMFUL** | `delta_clean_fuse <= -0.02` |

(Threshold set here, before any number exists, at the same ±0.02 magnitude
`p58`/`p60` already use for the analogous raw-fusion question — chosen for
consistency with the existing decision scale, not tuned to this data.)

## What a result of each kind would mean for the successor ladder

- **Both NEUTRAL/HARMFUL**: closes candidates B ("`rel_feat` + `[dx_rel,
  dy_rel]`"), C ("`rel_feat` + normalised box geometry") and, by extension,
  the additive-injection form of E from the task's candidate list. Combined
  with `p60`'s closure of raw fusion, this would mean the representation's
  deficit is not fixable by adding these specific numbers back in any linear
  or shallow-MLP combination tested so far — strengthening H6/C1
  (representation genuinely lacks usable spatial information at the readout
  level tested) over C2 (present but badly read out) for this specific
  probe family, though **not** for every conceivable nonlinear encoder.
- **`MINIMAL-FIX-WORKS`**: the two literal numbers the encoder drops are
  sufficient once handed back explicitly — a genuinely cheap architectural
  fix (concatenate `dx_rel`, `dy_rel` before the head) would be justified,
  and a small GPU confirmation (retrain the actual head, not just a frozen
  probe) becomes the next experiment.
- **`CLEAN-FUSION-GAIN`**: the additive-fusion route is not closed after
  all — it was closed only for the *contaminated* representation. This
  would motivate a successor that removes the group mean before fusing
  (the "prior stream + evidence stream, scored separately" architecture
  family `docs/SUCCESSOR_HYPOTHESES.md` already lists as a target for H3),
  and would justify a small GPU training pilot to see whether a jointly
  trained encoder can learn that decomposition itself.

## What this cannot settle

Readout probes on a frozen encoder, cross-fitted on validation, one
checkpoint, PredCls with GT pairs. A negative result here closes the
*additive, frozen-probe* route; it does not by itself rule out a jointly
retrained encoder recovering the same information through a different
mechanism. Single checkpoint, validation split only — no held-out test claim
is made for this exploratory-but-preregistered analysis until/unless it
motivates something worth replicating.
