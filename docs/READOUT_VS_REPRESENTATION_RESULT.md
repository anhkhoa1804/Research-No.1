# `runs/p37` — REPRESENTATION-LIMITED, and BELOW GEOMETRY

The decisive experiment. Thresholds pre-registered in
`docs/READOUT_VS_REPRESENTATION_PREREGISTRATION.md` (+ its geometry addendum),
committed before `p36` launched and before any number here existed.

---

## Validity gates — all PASS

| gate | requirement | observed |
|---|---|---|
| **V1** | `rel_feat` present, finite, complete | (132556, 768), `missing_rel_feat = 0` |
| **V2** | recomputed `normalize(rel_feat)·normalize(pred_emb)ᵀ` reproduces the **stored** `text_logits` | **1.536e-04** (fp16 tol 1e-2) |
| **V3** | null controls at chance | `R6_shuffled` 0.4983, `R7_prior` 0.5000 |
| **V4** | reproduces the `p24` cache | `max|model_term − p24| = **0.000e+00**`, 132556 rows both |

**V2 is the load-bearing one**: it proves the cached feature *is* the tensor the
evaluated head actually read, not some other activation. V4 shows the second GPU
pass reproduced the first **bit-for-bit** on the model term.

## Result

| arm | WPRD | weighted | 95% CI | head–head | body–body | tail–tail |
|---|---|---|---|---|---|---|
| R1_text (evaluated head) | 0.5542 | 0.5266 | [0.5495, 0.5592] | 0.5612 | 0.5746 | 0.5126 |
| **R2_cls (discarded head)** | **0.5728** | 0.5349 | [0.5681, 0.5779] | 0.5701 | 0.6167 | 0.5677 |
| R3_linear on `rel_feat` | 0.5601 | 0.5288 | [0.5554, 0.5645] | 0.5633 | 0.6210 | 0.5017 |
| R4_mlp on `rel_feat` | 0.5600 | 0.5345 | [0.5553, 0.5646] | 0.5629 | 0.6299 | 0.4731 |
| **R5_residual (group-centred `rel_feat`)** | **0.5807** | 0.5435 | [0.5762, 0.5850] | 0.5762 | **0.6806** | 0.5371 |
| R6_shuffled (control) | 0.4983 | — | [0.4928, 0.5028] | ✓ | | |
| R7_prior (control) | 0.5000 | 0.5000 | [0.5000, 0.5000] | ✓ | | |
| **R8_geom (train-fitted)** | **0.5961** | 0.5419 | [0.5921, 0.6014] | 0.5938 | 0.6410 | **0.6353** |
| R9_`rel_feat`+geometry | 0.5735 | 0.5316 | [0.5691, 0.5782] | 0.5797 | 0.6438 | 0.5193 |

## Verdict

```
P* = max(R3, R4) = 0.5601     C = R2_cls = 0.5728     G = R8_geom = 0.5961
P* − C = −0.0127   →  PRIMARY  : REPRESENTATION-LIMITED   (threshold: < C + 0.01)
P* − G = −0.0360   →  SECONDARY: BELOW GEOMETRY           (threshold: ≤ G − 0.02)
```

**H1 (readout bottleneck) is DEAD. H6 (representation) is SUPPORTED.**

A cross-fitted probe on `rel_feat` — linear *or* MLP — **cannot even match the
classifier head that is already attached to it**, despite the probe enjoying a
validation-fitting advantage the head never had. And the whole 768-d
representation lands **0.036 below a linear model on 19 numbers from two
rectangles**.

Note also **R9**: `rel_feat` + geometry (0.5735) is *worse* than geometry alone
(0.5961). Adding the learned representation to boxes **destroys** information.

## The one genuinely positive finding

**R5 — group-centred `rel_feat` — is the best `rel_feat` arm at 0.5807**,
beating the raw representation (0.5601, **+0.021**) and the classifier head
(0.5728, +0.008), with body–body **0.6806**, the highest of any arm anywhere in
this programme.

Removing the (s,o) group mean from the representation *before* the readout
**improves** within-pair discrimination. The between-group component is not
merely uninformative for this task — it is an **active distractor**. That is a
concrete, mechanism-derived design principle, and it is the one thing here that
points at a successor.

It does **not** change the verdict: R5 is a separately registered arm, the
primary criterion is defined on `max(R3, R4)`, and R5 still sits below geometry.

## Execution note — the run was OOM-killed after producing every number above

`p37` exited **-9 (SIGKILL)** at 601 s. Confirmed OOM from the kernel log
(`anon-rss 21.9 GB`, alongside `p36`'s 6 GB on a 31 GB host). The cause: the
*collapse measure* attempts a one-hot over object labels, and this dataset has
**16,929** of them rather than ~300, so the matrix is 132,556 × 14,503 ≈ 7.7 GB
(`docs/DATASET_IDENTITY_OBJECT_VOCAB.md`).

**Every arm and every gate had already been computed and printed** before the
kill; the verdict above is arithmetic on those printed values against thresholds
fixed in advance. What was lost is the JSON artifact and the collapse measure —
neither a criterion. This is recorded rather than smoothed over, and the run
should be repeated with a memory-safe collapse measure to produce the artifact.

## Consequence

| hypothesis | status after `p37` |
|---|---|
| H1 readout bottleneck | **DEAD** — probes cannot match the existing head |
| H2 objective bottleneck | REFUTED on the box channel (`p48`); untested on `rel_feat` |
| H3 prior/evidence entanglement | WEAKENED (`p45`), but **R5 revives a narrow form**: removing the pair mean helps |
| **H6 representation bottleneck** | **SUPPORTED** |

A readout-only successor is **not** justified. The representation is the limit,
and it is below what two rectangles supply.
