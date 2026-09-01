# Pre-registration — held-out replication on the VG150 TEST split

Status: **PRE-REGISTERED**, committed before the GPU pass launches.
Run: `p54_test_relfeat_cache` (GPU) → CPU replications.

## Why

Every result in this programme carries the caveat *"validation split only."* It
appears in `DIAGNOSIS.md`, the Paper-1 table, the literature audit and the
hypothesis matrix. The test split removes it, and the cache is reusable for
every CPU analysis already written.

`p36` established the pass is deterministic: it reproduced `p24`'s PredCls
metrics exactly (R@50 0.6832, mR@50 0.2411) and its model term **bit-for-bit**
(`max|diff| = 0.000e+00`). So a test-split difference is a property of the data,
not of the run.

## Protocol

Identical to `p36` in every flag except `--eval_split_name test` and output
paths. Frozen forward pass, no training, no gradient. Emits `pair_logits`,
`rel_feat` (fp16) and `pred_emb`.

`--eval_split_name` is new; it defaults to `"validation"`, so the historical
behaviour is unchanged, and `eval_on_train_split` still takes precedence.
Verified on a 2-batch smoke run (`split=test` loaded, exit 0) and the 20
dump-contract tests still pass.

GPU policy: verified idle (0 MiB, 0%, no compute apps) immediately before launch.
Budget 5 GPU-hours (`p36` took 3 h 53 m). Stop rule: if throughput implies > 5 h,
kill and report; no partial cache is analysed as if it were the full split.

## What is replicated, and the criteria — fixed here

All are **replications of already-registered analyses**, so the thresholds are
inherited, not new:

| # | quantity | replication criterion |
|---|---|---|
| 1 | prior control WPRD | **must be 0.5000**. If it is not, the protocol is broken on this split and nothing else is reported. |
| 2 | shuffled / random nulls | within [0.49, 0.51] |
| 3 | WPRD ordering: geometry > classifier head > text head > prior | same **strict ordering** |
| 4 | Spearman(R@50, WPRD) | same **sign**, and significant at p < 0.05 |
| 5 | Spearman(mR@50, WPRD) | same **sign** (negative), and significant at p < 0.05 |
| 6 | within-pair supervision skew | head–head ≫ tail–tail, same order of magnitude |
| 7 | `p37` verdict | REPRESENTATION-LIMITED and BELOW GEOMETRY again |

**REPLICATED** = 1, 2 hold and ≥5 of items 3–7 hold.
**PARTIAL** = 1, 2 hold and 3–4 of 3–7 hold.
**FAILED** = 1 or 2 fails, or ≤2 of 3–7 hold.

A FAILED outcome would mean the validation findings are split-specific and would
be reported as prominently as a replication.

## What this does NOT address

The **cross-model** gate. Test-split replication makes the PURE result
held-out; it does not make it a statement about SGG. That gate is unchanged and
remains binding on every general claim.
