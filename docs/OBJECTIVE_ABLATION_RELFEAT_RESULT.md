# `runs/p55` — the objective is REFUTED on `rel_feat` too, at a matched budget

Run: exit 0, cross-fitted over 5 validation folds. Pre-registered in
`docs/OBJECTIVE_ABLATION_RELFEAT_PREREGISTRATION.md` (commit `75b1e8e`),
committed before the tool existed. No threshold moved.

## Validity gates — all PASS

| gate | requirement | observed |
|---|---|---|
| V1 | prior control exactly 0.5000 | **0.500000** |
| V2 | shuffled control at chance | **0.4981** |
| V3 | `rel_feat` complete & finite | (132556, 768), missing 0 |
| V4 | every arm scores every row | 10 arms x 132,556 |

## Result

| arm | WPRD | weighted | 95% CI | head | body | tail | train rows | contrasts |
|---|---|---|---|---|---|---|---|---|
| **`A_ce_all`** | **0.5732** | 0.5388 | [0.5674, 0.5782] | 0.5823 | 0.6064 | 0.4859 | 530,224 | — |
| `B_contr_decidable` | 0.5359 | 0.5352 | [0.5309, 0.5405] | 0.5406 | 0.5625 | 0.5265 | 288,735 | 21,489,866 |
| `C_ce_matched` | 0.5640 | 0.5339 | [0.5595, 0.5698] | 0.5689 | 0.5944 | 0.4152 | 288,735 | 21,489,866 |
| `D_ce_pairbal` | 0.5650 | 0.5355 | [0.5598, 0.5701] | 0.5719 | 0.5741 | 0.4624 | 530,224 | — |
| `E_contr_pairbal` | 0.5359 | 0.5297 | [0.5308, 0.5405] | 0.5373 | 0.5562 | 0.4082 | 288,735 | 21,489,866 |
| `F_ce_groupcentred` | **0.5173** | 0.5236 | [0.5126, 0.5227] | 0.5116 | 0.5596 | 0.4580 | 530,224 | — |
| `N_shuffled` (control) | 0.4981 | 0.5049 | [0.4926, 0.5035] | — | — | — | — | — |
| `P_prior` (control) | **0.5000** | 0.5000 | [0.5000, 0.5000] | — | — | — | — | — |
| ref: text head (evaluated) | 0.5542 | 0.5266 | [0.5495, 0.5592] | 0.5612 | 0.5746 | 0.5126 | — | — |
| ref: classifier head | 0.5728 | 0.5349 | [0.5681, 0.5779] | 0.5701 | 0.6167 | 0.5677 | — | — |

## Verdict

```
PRIMARY   d_obj = B_contr_decidable - C_ce_matched = -0.0281   -> REFUTED
SECONDARY d_sup = -0.0091   d_bal = -0.0081                    -> SUPERVISION-INSENSITIVE
TERTIARY  best fitted arm A_ce_all 0.5732 vs G = 0.5961        -> BELOW GEOMETRY
```

**`p48` was not a channel artifact and was not a confound artifact.** Moved to
`rel_feat` and given exactly the population the contrastive arm can learn from,
the within-group contrastive objective is **worse** than plain cross-entropy by
**0.0281** — twice the gap `p48` measured on box geometry (−0.0143). Directly
optimising WPRD's own surrogate scores worse on WPRD than plain CE, on both
channels, with the information budget matched and the optimisation budget
matched (contrastive acceptance ~32%, so 200k draws/fold/epoch yields ~64k
accepted pairs against `C_ce_matched`'s ~57.7k rows).

Pair-balanced sampling does not help either (`D − A` = −0.0081,
`E − B` = 0.0000). **Reweighting the supervision that exists is not a lever on
these features.**

## Two corrections this run forces on `p37`

### 1. A tuned probe on `rel_feat` DOES match the classifier head

`p37` reported `P* = 0.5601` and wrote that a probe "cannot even match the
classifier head that is already attached to it". `A_ce_all` here reaches
**0.5732** against the classifier's **0.5728** — a dead heat. The difference is
hyperparameters (hidden 256 / 20 epochs here, hidden 512 / 25 epochs there),
not the features.

**`p37`'s registered verdict is unchanged**: the rule is REPRESENTATION-LIMITED
when `P* < C + 0.01`, and 0.5732 < 0.5828, so it still fires. But the prose was
too strong and is corrected here: a probe on `rel_feat` **can match** the
classifier head; it cannot **exceed** it, and it stays **0.023 below geometry**.
The load-bearing half of `p37` — BELOW GEOMETRY — is untouched and is confirmed
by a second, better-tuned probe family.

### 2. `p37`'s R5 group-centring result does NOT survive a nonlinear readout

`p37` called R5 — a **linear** probe on group-centred `rel_feat`, 0.5807 — "the
one genuinely positive finding" and "a concrete, mechanism-derived design
principle... the one thing here that points at a successor".

`F_ce_groupcentred` runs the same transformation under an **MLP** readout and
gets **0.5173** — barely above the shuffled control, and **0.056 worse** than the
same MLP on raw `rel_feat`.

So the group-centring gain is **not robust to the readout family**. It is a
property of the linear probe, not of the representation. **The single design
principle this programme had extracted for a successor architecture is
withdrawn as a design principle.** It remains a real observation about linear
readouts and nothing more.

## What this run does not settle

The encoder is **frozen**. Every arm changes only the readout's objective on a
fixed representation, so REFUTED means *the objective is not the limit given
these features*. It does **not** mean an encoder trained from scratch under a
within-pair objective would fail; that experiment is not affordable here and is
not claimed. This limitation was registered in advance.
