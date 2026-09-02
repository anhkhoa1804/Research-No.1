# `p54`/`p59`/`p61`/`p62` — the held-out TEST split. The core findings REPLICATE.

Pre-registered in `docs/TEST_SPLIT_REPLICATION_PREREGISTRATION.md` (commit
`65e1105`). This resolves the caveat *"validation split only"* that appeared in
`DIAGNOSIS.md`, the Paper-1 table, the literature audit and the hypothesis
matrix.

## The GPU pass

`runs/p54_test_relfeat_cache`. Exit 0, **12,685 s (3 h 31 m)**, inside the
registered 5-hour budget. Commit `0dae133`, torch 2.9.1+cu129, NVIDIA L4 sm_89,
`test.jsonl` sha256 `d0722e56…a4509e5`, checkpoint
`pure_best_adapt_light_mR50.pt`. **10,403 images, 132,334 pairs.** Identical to
`p36` in every flag except `--eval_split_name test` and output paths.

PredCls on test: R@50 **0.6813**, mR@50 **0.2329**, ngR@50 0.8382
(validation: 0.6832 / 0.2411).

## Item-by-item against the registered criteria

| # | quantity | validation | **test** | holds? |
|---|---|---|---|---|
| 1 | prior control WPRD | 0.5000 | **0.5000** | **YES** — protocol intact on this split |
| 2 | shuffled / random null | 0.4983 | **0.5001** | **YES** (in [0.49, 0.51]) |
| 3 | ordering geometry > classifier > text > prior | 0.5961>0.5728>0.5542>0.5 | **0.5916 > 0.5712 > 0.5446 > 0.5000** | **YES**, strict |
| 6 | within-pair supervision skew | head–head 81.6% / tail–tail 0.02% | **80.3% / 0.01%** | **YES**, same order |
| 7 | `p37` verdict | REPRESENTATION-LIMITED, BELOW GEOMETRY | **both, all gates pass** | **YES** |

Items 4 and 5 (the Spearman correlations) are in `runs/p61`; see
`docs/METRIC_GROUNDING_TEST_RESULT.md`.

## The `p37` arms side by side

| arm | validation | **test** | Δ |
|---|---|---|---|
| R1 text head (evaluated) | 0.5542 | 0.5446 | −0.0096 |
| R2 classifier head (discarded) | 0.5728 | 0.5712 | −0.0016 |
| R3 linear on `rel_feat` | 0.5601 | 0.5551 | −0.0050 |
| R4 MLP on `rel_feat` | 0.5600 | 0.5596 | −0.0004 |
| R5 group-centred | 0.5807 | 0.5697 | −0.0110 |
| R6 shuffled | 0.4983 | 0.5001 | +0.0018 |
| R7 prior | 0.5000 | 0.5000 | 0.0000 |
| R8 geometry (train-fitted) | 0.5961 | 0.5916 | −0.0045 |
| R9 `rel_feat`+geometry | 0.5735 | 0.5685 | −0.0050 |

`P* − C = −0.0116` (val −0.0127) · `P* − G = −0.0320` (val −0.0360).
Collapse R² 0.1811 (val 0.1825).

Every arm moves by less than 0.011 and no ordering changes. **This is a
replication, not a similarity argument**: the registered criteria were fixed
before the cache existed and are met item by item.

## What the test split did NOT fix

**Tail power.** The registered hope that more data would make the tail–tail cell
readable is **not met**:

| | validation | test | pooled |
|---|---|---|---|
| tail–tail cells | 34 | **37** | 71 |
| tail–tail comparisons | 226 | **104** | 330 |

Tail–tail remains 0.01–0.02% of all within-pair contrastive supply. On test the
shuffled null reads **0.4379** in the tail–tail column and the evaluated head
reads **0.4443** — both far from 0.5, which is what an underpowered cell looks
like. **The tail–tail column must not be read as a point estimate on either
split, and pooling the two splits does not rescue it.** This is now a measured
property of VG150, not a limitation of one split, and it is the strongest
existing argument that a *rare-predicate* claim needs data VG150 does not have.

## Standing

The "validation only" caveat is **discharged** for items 1, 2, 3, 6, 7. Every
result remains a measurement of **one checkpoint**; the cross-model gate is
untouched and still binding.
