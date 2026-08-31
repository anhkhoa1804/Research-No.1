# Result: appearance probe at ViT-L/14-336 on the L4

Companion to `docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md`, which was written
before this run and is **not edited by this document**. The pre-registration
fixed the encoder, sampling, PCA, λ grid, λ-selection rule and the 5 %
decision threshold in advance; this document reports what came out.

**Verdict, applied mechanically from the pre-registered rule: H0 SUPPORTED.**
The visual ablation gate passes, so the measurement is valid — and appearance
converts **−1.2 %** of the oracle@5 headroom, against a threshold of ≥ 5 %.

Epistemic labels are used strictly: **MEASURED** (produced by a run recorded
under `runs/` on this machine), **VERIFIED FACT** (checked in code or by
cryptographic identity), **INFERENCE**, **CLAIM** (asserted elsewhere, not
checkable here).

---

## 1. Provenance

| | |
|---|---|
| Run directory | `runs/p6_appearance_probe_l14_336/` |
| Git commit | `a46fbada796ab066b5533eb736cf74ca5f5f10d8` |
| Branch | `research/architecture-breakthrough` |
| Working tree at launch | **clean** (`working_tree_clean: true` in `provenance.json`) |
| Started / ended (UTC) | 2026-08-31T11:39:07 → 2026-08-31T11:59:42 |
| Runtime | **1,235.4 s (20.6 min)**, extract + score, one process |
| Exit code | 0 |
| GPU | NVIDIA L4, 23,659,151,360 B (22.0 GiB), `sm_89`, driver 580.173.02 |
| Peak VRAM | 2,594 MiB allocated at batch 16 (benchmark); ~2,932 MiB resident observed in flight via `nvidia-smi`. 12 % of the L4's 22 GiB — VRAM was never the constraint. |
| Python / torch / transformers / numpy | 3.10.12 / 2.9.1+cu129 / 5.15.1 / 2.2.6 |
| Platform | Linux-6.8.0-1066-gcp-x86_64-with-glibc2.35, host `research-no-1` |

### Exact command

```
.venv/bin/python3 tools/appearance_probe.py --extract --score \
  --clip_name openai/clip-vit-large-patch14-336 \
  --device cuda --encode_batch 16 --n_train 1200 --n_val 1200 \
  --cache runs/p6_appearance_probe_l14_336/clip_l14_336_cache.pt \
  --out runs/p6_appearance_probe_l14_336/appearance_probe_l14.json
```

Wrapped in `tools/run_experiment.py --name p6_appearance_probe_l14_336`, which
captured `command.txt`, `provenance.json`, `stdout.log`, `stderr.log` and
`result.json` automatically.

`--encode_batch 16` is **not** a guess: it is `best_batch` from
`runs/p4_bench_clip_l14_336/benchmark.json`, produced by
`tools/benchmark_clip_encoder.py` on real crops. MEASURED there at 60.4
crops/s; MEASURED here in flight at 58.5 crops/s, so the schedule estimate
held. Note the benchmark curve is essentially flat (60.4 / 59.8 / 60.0 / 59.6
crops/s at batch 16 / 32 / 48 / 64) — batch size was not a meaningful lever on
this GPU, and 16 was taken because it is what the measurement selected.

### Pre-launch verification (rule 7)

Working tree clean, and every historical artifact hashed and matched **before**
the run:

| Artifact | SHA256 | Matches |
|---|---|---|
| `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` | `8845c3af…d442` | ✓ `historical_checkpoint_v1.yaml` |
| `checkpoints/demo_best/frequency_prior.json` | `144d9f92…6e6a` | ✓ `historical_checkpoint_v1.yaml` |
| `checkpoints/demo_best/demo_config.env` | `c7318069…26a6` | ✓ `historical_checkpoint_v1.yaml` |
| `data/manifests/historical_checkpoint_v1.yaml` | `75b83727…4326` | ✓ `machine_provenance_v1.yaml` |
| `datasets_vg150_clean/train.jsonl` | `306fc0db…c605a4` | ✓ `machine_provenance_v1.yaml` |
| `datasets_vg150_clean/validation.jsonl` | `74d99779…5343` | ✓ `machine_provenance_v1.yaml` |
| `datasets_vg150_clean/vocabulary/predicates.json` | `76b75952…0577` | ✓ `machine_provenance_v1.yaml` |
| `datasets_vg150_clean/frequency_prior_train.json` | `54c8c910…e4be` | first record (no prior reference existed) |

None of these were modified, and none were used as write targets. The probe
reads the **train-derived** prior (`frequency_prior_train.json`), never the
historical one — the two are different scientific artifacts and were not
substituted.

VERIFIED FACT: the pre-registration and the probe source are unchanged since
the commit that registered them — `git diff 91b24c7 HEAD -- tools/appearance_probe.py docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md`
is empty. The verdict-bearing script is byte-identical to its registered
version, so no threshold or selection rule could have been tuned after seeing
data.

### Pilot before the full run

`runs/p6_appearance_probe_l14_pilot/` — 40 + 40 images, same encoder and
device, exit 0 in 62.6 s. It validated the extract→cache→score path on GPU
end-to-end. Its metrics are **not a result** (470 train instances; the ablation
gate is not meaningful at that size and indeed failed there).

---

## 2. What was measured

**Setup, exactly as pre-registered.** Frozen `openai/clip-vit-large-patch14-336`
(never fine-tuned), fp16 autocast on CUDA, 1,200 train + 1,200 validation
images, K = 5 candidates from the train-derived prior, PCA-48 per feature
block fitted on non-held-out train only, 30 epochs, image-level held-out split
for epoch and λ selection, weight decay 3e-3.

MEASURED sample identity:

| | ViT-L/14-336 (this run) | ViT-B/32 (documented) |
|---|---:|---:|
| crops encoded | **70,709** | 70,709 |
| instances (train + val) | **30,026** (14,746 + 15,280) | 30,026 |
| projection dim | 768 | 512 |
| input resolution | 336 × 336 | 224 × 224 |

The crop and instance counts are **identical**, which is the check that the
two runs differ in the encoder and nothing else on the data side.

*Metadata note, benign:* `preprocess_resolution` is recorded as `null` in the
cache metadata because transformers 5.x returns a `SizeDict` rather than a
plain `dict` and the probe's `isinstance(..., dict)` test therefore fails.
Verified independently that the processor's `crop_size` is
`height=336, width=336` — the encoder really did run at 336. This is a
reporting gap in a metadata field, not a protocol deviation, and the probe was
deliberately **not** patched mid-experiment to fix it.

---

## 3. Visual ablation gate — **PASSES**

The validity gate. Arms 1–3, replacement scorer (no prior term), so the numbers
are low by construction; only their ordering is meaningful.

| arm | val R | val mR@50 | tail mR |
|---|---:|---:|---:|
| **REAL appearance** | 43.12 % | **16.90 %** | 3.8 % |
| SHUFFLED appearance | 26.77 % | **12.17 %** | 8.7 % |
| ZERO appearance | 53.58 % | **8.58 %** | 1.9 % |

`real > shuffled > zero` holds: 16.90 > 12.17 > 8.58.

MEASURED: **frozen ViT-L/14-336 appearance carries genuine predicate
information.** The run is therefore a valid test of the hypothesis, not a
broken measurement — the pre-registered `INVALID` branch does not apply.

---

## 4. Additive composition — the whole λ curve

`score = log P(p|s,o) + λ · f(appearance)`. λ = 0 is exactly P0, so any gain is
attributable to appearance alone. The full grid is reported, including the λ
values that hurt, as the pre-registration requires.

Baselines on this subsample: **P0 prior** R 67.38 % / mR@50 20.98 %;
**oracle@5** R 89.90 % / mR@50 64.60 %; coverage@5 89.90 %.
Headroom = **43.61 mR points**.

| λ | val R | val mR@50 | tail mR | headroom captured | held-out mR (selection) |
|---:|---:|---:|---:|---:|---:|
| 0.00 (P0) | 67.38 % | 20.98 % | 7.1 % | 0.0 % | — |
| 0.10 | 67.55 % | 21.02 % | 7.4 % | +0.1 % | 47.39 % |
| 0.25 | 67.54 % | **21.06 %** | 7.2 % | **+0.2 %** | 46.62 % |
| **0.50** ← selected | 66.87 % | **20.47 %** | 5.4 % | **−1.2 %** | **47.76 %** ← max |
| 1.00 | 65.11 % | 20.45 % | 5.3 % | −1.2 % | 46.73 % |
| 2.00 | 63.06 % | 19.89 % | 4.0 % | −2.5 % | 43.29 % |

**λ was selected on the held-out-from-train split, never on validation**, as
pre-registered. That rule picked λ = 0.5 (held-out mR 47.76 %, the maximum of
the column), whose validation mR@50 is 20.47 % — **below** P0.

MEASURED headline: **−0.51 mR@50 points vs the prior, i.e. −1.2 % of headroom
captured.**

For transparency about the size of the selection bias, the pre-registration
also requires reporting the cherry-picked upper bound: **validation-argmax**
λ = 0.25 gives mR@50 21.06 %, i.e. **+0.08 points, +0.2 % of headroom**. That
number is optimistic by construction — it is the maximum of five correlated
draws on the split it is reported on — and is **not** the headline.

**The verdict does not depend on which selection rule is used.** Across the
entire pre-registered λ grid the best achievable capture is +0.2 %, and the
honest held-out-selected value is −1.2 %. Both sit far outside the 4–6 %
inconclusive band and far below the 5 % threshold.

### One wording precision, found by re-reading the implementation

The **gate** arms (§3) are fitted on `subj+obj+union` — appearance only. The
**additive** arms in the table above are fitted on `subj+obj+union+geom` —
appearance **plus the 8 box-geometry features**. So the headline is properly
"appearance *and geometry*, composed additively with the prior, captured
−1.2 %".

This is faithful to the pre-registration, which specifies the design as
"unchanged from the B/32 run except for the encoder and device", and the B/32
run did exactly the same. It is recorded because the two arms are not measuring
the same feature set, not because the protocol drifted. Prior work measured
geometry alone as converting **0.0 %** of headroom across three scorer families
(`tools/candidate_reranking_analysis.py`), so the geometry block is unlikely to
be carrying this negative — but that is an inference from a different
experiment, not something this run controlled internally.

---

## 5. Per-predicate: the mechanism is real and is the same one B/32 found

At the selected λ, predicates with n ≥ 15, largest gains:

| predicate | n | P0 rank-1 | additive | Δ |
|---|---:|---:|---:|---:|
| `eating` | 22 | 40.9 % | 59.1 % | **+18.2** |
| `riding` | 49 | 10.2 % | 18.4 % | **+8.2** |
| `standing on` | 81 | 1.2 % | 7.4 % | **+6.2** |
| `holding` | 308 | 64.9 % | 69.8 % | **+4.9** |
| `walking on` | 23 | 0.0 % | 4.3 % | +4.3 |
| `from` | 23 | 8.7 % | 13.0 % | +4.3 |
| `with` | 414 | 9.2 % | 13.3 % | +4.1 |
| `under` | 185 | 20.5 % | 22.7 % | +2.2 |
| `in front of` | 136 | 14.7 % | 16.2 % | +1.5 |
| `behind` | 356 | 50.8 % | 51.1 % | +0.3 |

MEASURED: appearance improves exactly the action and pose predicates theory
predicts — the same set the B/32 run identified (`eating`, `riding`,
`standing on`, `holding`, `walking on`). **This is the important structural
result: the negative is not a null measurement.** The signal is present, it is
in the right place, and it still fails to move the class-averaged aggregate.

INFERENCE (the arithmetic is in `docs/APPEARANCE_PROBE_FINDINGS.md` §3, and it
reproduces here): under mR@50 averaged over 50 classes, +18.2 points on a
22-instance class contributes 18.2/50 = 0.36 points to the aggregate, while a
few-point slip on `on` (n ≈ 5,600), `has` and `in` cancels it. The predicates
where appearance helps are the ones where the metric has the least leverage.

---

## 6. Second denominator — `captured_decidable`

The pre-registration (§5) requires reporting against a second, more favourable
denominator, because 64.2 % of the prior's residual errors are
generic→generic (`on`↔`in`↔`of`) — annotation-style choices no encoder can
resolve — and only 12.6 % of GT triplets carry a decidable predicate
(`runs/p3e_headroom_train_derived/`).

Restricting the class average to the 27 decidable predicates (the complement
of `headroom_analysis.GENERIC_PREDICATES`, the same split that analysis used):

| | mR@50 (decidable classes only) |
|---|---:|
| P0 prior | 15.29 % |
| additive, λ = 0.5 (selected) | **16.11 %** |
| oracle@5 | 67.78 % |
| **`captured_decidable`** | **+1.5 %** |

All 27 decidable classes are present (1,873 validation instances).

So on the denominator that is *most* favourable to the hypothesis, appearance
still converts **1.5 %** — under a third of the 5 % threshold. As
pre-registered, this number is diagnostic and **cannot** change the verdict,
which rests on `captured_total` only. It is reported because it is more
informative, and it happens to point the same way.

**How this number was produced without touching the registered script.**
`tools/appearance_probe.py` does not emit it. Rather than edit a
verdict-bearing, pre-registered script mid-experiment, a separate
`tools/appearance_probe_decidable.py` was created: a verbatim copy plus
report-only additions (two lines differ — one docstring line, and binding two
counters the original computed and discarded). It draws no RNG and re-fits
nothing, so it must reproduce the primary numbers exactly. **VERIFIED: it
reproduces all 24 primary fields of `appearance_probe_l14.json` bit-for-bit**
(`runs/p6_appearance_probe_l14_336_decidable/`), which is what makes the
secondary number trustworthy rather than merely asserted.

---

## 7. fp32 control arm — mixed precision is inert

Arm 6 of the pre-registration: *"a fp32 control at the best λ, to confirm mixed
precision changed nothing."* Run in full rather than at the best λ only, so the
entire λ curve is available for comparison.

| | |
|---|---|
| Run directory | `runs/p6_appearance_probe_l14_336_fp32/` |
| Command | identical to §1 but `--no_amp --encode_batch 32` |
| Batch size | 32, from `runs/p6_bench_clip_l14_336_fp32/` (MEASURED 17.6 crops/s, 2,838 MiB peak) — the fp32 measurement, not the fp16 one |
| Runtime | **4,021.4 s (67.0 min)**, exit 0 — the benchmark predicted 67 min |
| Crops / instances | **70,709 / 30,026** — identical to the fp16 run |

### The comparison

| quantity | fp16 | fp32 | Δ (pp) |
|---|---:|---:|---:|
| coverage@5 | 89.90 % | 89.90 % | **0.0000** |
| P0 prior R | 67.38 % | 67.38 % | **0.0000** |
| P0 prior mR@50 | 20.98 % | 20.98 % | **0.0000** |
| oracle@5 mR@50 | 64.60 % | 64.60 % | **0.0000** |
| gate REAL mR | 16.90 % | 16.86 % | −0.037 |
| gate SHUFFLED mR | 12.17 % | 10.79 % | −1.377 |
| gate ZERO mR | 8.58 % | 8.58 % | **0.0000** |
| additive λ=0.10 | 21.02 % | 21.03 % | +0.002 |
| additive λ=0.25 | 21.06 % | 21.06 % | +0.003 |
| additive λ=0.50 | 20.47 % | 20.57 % | +0.099 |
| additive λ=1.00 | 20.45 % | 20.49 % | +0.041 |
| additive λ=2.00 | 19.89 % | 19.90 % | +0.014 |
| **`captured_total`** | **−1.16 %** | **−0.94 %** | **+0.227** |
| val-argmax captured | +0.18 % | +0.19 % | +0.007 |
| selected λ | 0.5 | **0.5** | same |
| val-argmax λ | 0.25 | **0.25** | same |
| gate passes | yes | **yes** | same |
| **verdict** | **H0 SUPPORTED** | **H0 SUPPORTED** | same |

### Reading it

**Precision is not a confound.** The verdict, the gate outcome, the selected λ
and the cherry-picked λ are all identical, and the entire additive curve moves
by at most 0.10 mR points. `captured_total` shifts by **+0.23 pp**, from −1.16 %
to −0.94 % — still roughly 5 pp below the 5 % threshold and well outside the
4–6 % inconclusive band. Nothing about the conclusion depends on fp16.

Three details worth stating rather than glossing:

1. **The baselines are bit-identical** — coverage, P0 and oracle all match to
   the printed precision with Δ exactly 0.0000. They are functions of the prior
   and the labels, not the encoder, so this is the check that both runs scored
   the same instances. Combined with the identical crop/instance counts, the
   two runs differ in precision and nothing else.
2. **The ZERO arm is bit-identical (8.58 %)**, as it must be: its features are
   zeroed before the scorer sees them, so no encoder arithmetic reaches it.
   That it matches exactly confirms the seeded RNG stream, the PCA and the
   held-out split were identical across the two processes.
3. **The SHUFFLED arm moved most (−1.38 pp)** — by far the largest change
   anywhere in the table. This is expected rather than alarming: it is the arm
   with the instance-level correspondence deliberately destroyed, so its
   scorer is fitting noise and is correspondingly the most sensitive to any
   perturbation. The gate ordering is unaffected, and the `real − shuffled`
   margin actually **widens** in fp32 (6.07 vs 4.73 points).

### A useful by-product: an empirical noise floor

The two runs are the same experiment under a small, structured perturbation of
the features. The resulting spread on the headline — **0.23 pp of captured
headroom** — is an independent estimate of this protocol's sensitivity, and it
agrees with the ~0.2 pp spread between the two documented ViT-B/32 runs. Both
are an order of magnitude smaller than the ~5 pp distance from the measured
value to the decision threshold.

### One data-handling defect found in both runs

Both logs contain `transformers` warnings: *"The channel dimension is
ambiguous. Got image shape torch.Size([3, 9, 3])"* (also `[3, 2, 3]`,
`[3, 6, 3]`, `[3, 8, 3]`). These are crops exactly 3 pixels high: a
`(H, W, C)` array of shape `(3, W, 3)` is ambiguous, and the processor assumes
channels-first, effectively transposing the crop.

MEASURED by replaying the probe's own crop geometry over the sampled images:
**21 object crops have a side of exactly 3 px**, and 95 (0.15 %) have a side
≤ 4 px. The four log lines are deduplicated by shape.

21 of 70,709 encoded crops is **0.03 %**, the warnings are byte-identical
between the two runs, and a 3-pixel crop carries no usable appearance signal
whichever way it is oriented. This cannot account for the result. It is
recorded because it is a real defect and because "I saw a warning and did not
check it" is how the four earlier bugs in this repository survived.


---

## 8. Comparison against the previous ViT-B/32 appearance probe

**A necessary caveat first.** No B/32 run artifact exists on this machine —
`runs/` did not exist when that work was done, and nothing was transferred
(the same gap `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §5 flags). The B/32
column below is a **CLAIM** taken from `docs/APPEARANCE_PROBE_FINDINGS.md`,
not a MEASURED result checkable here. This is a comparison of a measurement to
documentation.

Worse, the repository holds **two mutually inconsistent B/32 records**:

| source | gate (real / shuf / zero) | best λ | best mR@50 | captured |
|---|---|---:|---:|---:|
| `docs/APPEARANCE_PROBE_FINDINGS.md` (commit `1ac1c6f`) | 17.00 / 13.79 / 11.09 | 0.50 | 21.24 % | 0.6 % |
| `tools/appearance_probe.py` docstring (commit `9dd62f1`) | 16.75 / 14.54 / 8.83 | 0.25 | 21.34 % | 0.8 % |

These are two different B/32 runs. Neither is verifiable here. Their spread
(±0.1 mR points, ±0.2 pp of captured headroom, ~2 pp on the gate arms) is a
useful empirical floor on this protocol's run-to-run noise, and I use it as
such below rather than pretending either number is exact.

### Head to head

| | ViT-B/32 (CLAIM) | ViT-L/14-336 (MEASURED) |
|---|---:|---:|
| P0 prior mR@50 | 20.98 % | **20.98 %** |
| P0 prior R | 67.38 % | **67.38 %** |
| oracle@5 mR@50 | 64.93 % (implied) | **64.60 %** |
| headroom | 43.95 pts | **43.61 pts** |
| gate REAL mR | 17.00 % | **16.90 %** |
| gate ZERO mR | 11.09 % | **8.58 %** |
| **gate margin (real − zero)** | **5.91 pts** (findings doc) / **7.92 pts** (docstring run) | **8.32 pts** |
| best additive mR@50 | 21.24 % | **20.47 %** |
| **`captured_total`** | **+0.6 %** | **−1.2 %** |
| `captured_decidable` | not computed | **+1.5 %** |

Two observations, kept apart from each other:

1. **The baselines line up exactly.** P0 is 67.38 / 20.98 in both, to two
   decimal places, on identical crop and instance counts. INFERENCE: the
   prior, the sampling, the candidate set and the scoring path are the same;
   the encoder really is the only variable that changed.

   The oracle differs slightly (43.61 vs an implied 43.95 headroom). The
   oracle is a deterministic function of the prior and the labels, so with
   truly identical validation instances it should be identical. It is not,
   by 0.34 points. INFERENCE: the B/32 train/val split of those 30,026
   instances was not bit-identical to this one (only the total is documented,
   not the split). The gap is small and does not affect any conclusion, but
   it is stated rather than smoothed over.

2. **The stronger encoder does not convert more, and if anything carries a
   little more raw signal.** `captured_total` moves from +0.6 % to −1.2 %.
   The gate margin is 8.32 points here against **5.91** in the findings doc
   but **7.92** in the docstring run — so "the margin grew" is only true
   against one of the two B/32 records, and I do not lean on it. What is
   solid is the negative half: **the margin certainly did not shrink**, so
   the failure to convert cannot be attributed to the stronger encoder
   detecting less.

That combination is the substantive finding, and it is the one that answers
the pre-registered question. **H1 predicted that the B/32 negative was an
encoder-capacity artifact. It is not.** Capacity improved, the measurable
appearance signal improved with it, and conversion did not follow.

The −1.2 % vs +0.6 % difference is itself within the noise floor established
above; I do **not** claim L/14-336 is *worse* than B/32 at this task. The
defensible statement is that both are indistinguishable from zero conversion,
on both denominators.

---

## 9. Verdict, applied exactly as pre-registered

The pre-registered rule, verbatim:

| Outcome | Criterion |
|---|---|
| H1 SUPPORTED | captured ≥ 5 % |
| H0 SUPPORTED | captured < 5 % |
| INVALID | ablation gate fails |
| inconclusive | 4 % – 6 % |

- ablation gate: **PASS** → not INVALID.
- `captured_total` (primary, λ selected on held-out train): **−1.2 %**.
- −1.2 % < 4 %, so it is not in the inconclusive band.

> ### **H0 SUPPORTED.**
> *Encoder strength is not the constraint. Stop pursuing appearance
> reranking.*

The threshold was not renegotiated, the λ grid was not extended, and the
verdict string is derived in code from the measured number — the hardcoded
verdict that used to print "signal is REAL but far too WEAK" regardless of
input was removed at `91b24c7`, before registration.

---

## 10. Measured result vs interpretation

Kept deliberately apart, because the two have very different strength.

### MEASURED (this run, reproducible from `runs/p6_appearance_probe_l14_336/`)

1. Frozen ViT-L/14-336 appearance passes the ablation gate: 16.90 > 12.17 >
   8.58 mR@50.
2. Additive composition over the prior's top-5 captures −1.2 % of oracle@5
   headroom at the held-out-selected λ; +0.2 % at the cherry-picked λ; +1.5 %
   on the decidable-predicate denominator.
3. Gains concentrate on action/pose predicates: `eating` +18.2, `riding`
   +8.2, `standing on` +6.2, `holding` +4.9, `walking on` +4.3 rank-1 points.
4. The gate margin is 8.32 points. The two documented B/32 margins are 5.91
   (findings doc) and 7.92 (probe docstring), so the L/14-336 instrument
   registers at least as much appearance signal as B/32 did, and by one
   record materially more. Neither B/32 number is verifiable here.

### INFERENCE (supported, but not the same thing as measured)

5. Encoder capacity is not the binding constraint on this protocol. The one
   confound the previous analysis named as decisive has been removed and did
   not change the outcome.
6. The failure is a **leverage** failure, not a **perception** failure.
   Appearance sees what it should; class-averaged mR@50 over a 45 %-head
   distribution gives those classes almost no weight.

### NOT ESTABLISHED — do not let these be read into the above

7. That no fine-tuned encoder could do better. This bounds what is **linearly
   decodable from frozen features**, nothing more.
8. That PCA-48 is innocent. It compresses 768-dim blocks here against 512-dim
   at B/32 — a harsher ratio, retained deliberately for comparability. Partial
   counter-evidence: the gate is computed on those same PCA-48 features and
   got *stronger*, so PCA-48 is demonstrably not suppressing L/14-336's extra
   signal to zero. It may still be discarding some.
9. Anything about SGCls/SGDet, where object identity is not given.
10. That the current architecture is validated. A negative here removes one
    candidate explanation for the architecture's failure to beat the prior.
    It supplies no evidence in its favour.
11. That vision is useless for scene graphs. This measures one protocol
    (PredCls under GT pairs), one metric, one decoder family.

---

## 11. Is the evidence sufficient to close the appearance hypothesis?

The result is negative, so the pre-registration's own instruction applies:
assess closure explicitly rather than leaving it implied.

### Sufficient to close — YES, for the hypothesis actually registered

> *Frozen CLIP appearance, linearly decoded, reranking the prior's top-5,
> converts a material (≥ 5 %) share of oracle@5 mR@50 headroom on VG150
> PredCls under GT pairs.*

That is **closed, negatively**, and the closure is unusually clean for four
reasons:

1. **The measurement is valid, not null.** The gate passes and the mechanism
   reproduces on the predicted predicates. A negative from a working
   instrument is much stronger than a negative from a silent one.
2. **The single named confound was removed.** The prior analysis stated in
   advance that encoder capacity was the one thing that could overturn it.
   It was tested at the repository's own encoder and did not overturn it.
3. **The result is robust to the decision that could have rescued it.** Every
   λ in the grid, under either selection rule, and under both denominators,
   lands between −1.2 % and +1.5 %. Nothing near 5 %.
4. **The margin dwarfs the noise.** The documented B/32 run-to-run spread is
   ~0.2 pp of captured headroom. The distance to the threshold is ~4–6 pp,
   an order of magnitude larger.

### Not sufficient to close — the broader claim

The evidence does **not** close "vision cannot contribute here." Three things
remain untested, and none of them is a hedge invented after the fact — all
three are named in the pre-registration §6:

- **fine-tuned** encoders (this bounds frozen, linearly-decoded features);
- the **protocol and metric**, which are the more likely culprit: a
  no-vision τ = 0.1 recalibration already converts +4.11 mR@50 on the full
  split for zero parameters and zero pixels
  (`runs/p4_decision_rule_sweep/`). A metric that a scalar moves by 4 points
  and a 304M-parameter encoder moves by 0.1 is measuring the decision rule,
  not perception;
- **SGCls / SGDet**, where object identity is not given and the
  co-occurrence prior is correspondingly weaker.

### Practical reading

Appearance reranking of a saturated prior under PredCls/GT-pairs is a dead
end, and this is now the second independent encoder to say so. The honest
scope of the negative is "on this protocol", and the protocol — not the
encoder — is where the remaining doubt lives.

---

## 12. Recommendation

**Do not redesign the architecture on this result.** That is the explicit
constraint on this experiment and it is also the right call on the evidence:
a negative appearance probe says what does *not* explain the architecture's
failure. It supplies no design target, and it is not a mandate to build
anything.

Concretely:

1. **Close the appearance line.** Mark H1 falsified at both tested encoders.
   `docs/APPEARANCE_PROBE_FINDINGS.md` §5 caveat 1 — the ViT-B/32 lower-bound
   caveat that has qualified every appearance conclusion in this repository —
   is now **discharged**, and the B/32 negative should be re-read as a real
   negative rather than a lower bound.

2. **The next decisive experiment is already named and is *not* an
   architecture change.** `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §9 states
   two falsifiable conditions for the architecture being worth pursuing. This
   run settles the first, negatively. The second has **never been measured**:

   > does the model's contribution survive *after* the prior is recalibrated —
   > i.e. does model + τ-adjusted prior beat τ-adjusted prior alone?

   Every model-vs-prior comparison so far used the un-recalibrated prior, and
   `runs/p5_model_vs_leakfree_prior/` already showed the model does not beat
   the leak-free prior (−0.82 mR@50 at N = 3,000). Recalibrating the prior
   makes the bar *higher*, not lower. That experiment is cheap, decisive, and
   should be pre-registered the same way this one was.

   **It has deliberately not been started here.**

3. **If both conditions fail, the contribution is methodological.** Per
   `PHASE4_SCIENTIFIC_REASSESSMENT.md` §8, the strongest defensible claim this
   repository supports is about the protocol being recalibration-saturated —
   a real, reproducible finding — not about a better relational model. This
   run strengthens that claim by removing the last "but a stronger encoder
   might…" objection to it.

4. **Do not spend GPU on a larger encoder, a relation transformer, or a
   mixture of specialists** on the basis of this result. Nothing measured
   suggests capacity is the constraint; this run is direct evidence against it.

---

## 13. Artifacts

| Path | What |
|---|---|
| `runs/p6_appearance_probe_l14_336/` | **the headline run** — command, provenance, logs, `appearance_probe_l14.json`, feature cache (359 MB) |
| `runs/p6_appearance_probe_l14_336_decidable/` | secondary `captured_decidable` diagnostic, same cache |
| `runs/p6_appearance_probe_l14_336_fp32/` | pre-registered fp32 control arm — 67.0 min, exit 0, verdict identical |
| `runs/p6_bench_clip_l14_336_fp32/` | fp32 throughput measurement that sized the control arm |
| `runs/p6_appearance_probe_l14_pilot/` | 40+40 smoke pilot (plumbing only, not a result) |
| `runs/p4_bench_clip_l14_336/benchmark.json` | fp16 throughput measurement that set `--encode_batch 16` |
| `tools/appearance_probe.py` | the pre-registered probe, **unmodified** by this experiment |
| `tools/appearance_probe_decidable.py` | report-only copy for the secondary denominator |

The feature caches are kept: extraction is the expensive half (20.6 min fp16),
scoring is 16 s, so any re-scoring can reuse them without re-encoding.
