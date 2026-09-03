# Track B + Track C — ranked action queue

Mode: **PAPER B + PAPER C DEVELOPMENT**, Paper A frozen
(`docs/PAPER_A_FREEZE_AUDIT.md`) and not reopened here. GPU checked idle
before this session's only executed experiment (`p65`, CPU-only); a second
project's process (`extraction-gpu312`, PID 8131, ~2.4 GB) was observed
transiently occupying GPU memory at 0% utilization mid-session and had
exited by the end of it — logged, not competed with.

---

## Track B — ranked candidate models (info gain / cost), UPDATED after this session's concrete B0-equivalent checks

Builds on `docs/CROSS_MODEL_FEASIBILITY.md` (bknyaz/sgg IMP+ already secured
— n=1 additional model, `docs/CROSS_MODEL_IMP_PLUS_RESULT.md`). **This
session cloned and directly inspected three candidate codebases
(`yrcong/RelTR`, `naver-ai/egtr`, `mods333/energy-based-scene-graph`) and
attempted a real environment build for the third — this is no longer a
web-search-only ranking; two candidates were ruled out by reading their
actual code, and the third's exact failure mode was reproduced, not
assumed.**

| rank | candidate | family | verified status | PredCls support | env risk | info gain |
|---|---|---|---|---|---|---|
| **1** | **VCTree** (+ **TDE**, free once this stack runs) via `mods333/energy-based-scene-graph` or `KaihuaTang/Scene-Graph-Benchmark.pytorch` | Tree-LSTM context (Motifs-adjacent) + causal post-hoc adjustment | **checkpoints confirmed available** — the `mods333` fork links direct `VCTree-Predcls` weights (both CE and energy-based variants) — but the **environment build was attempted this session and confirmed to fail**: `maskrcnn_benchmark/csrc/cpu/ROIAlign_cpu.cpp` uses the pre-1.5 `input.type()` API inside `AT_DISPATCH_FLOATING_TYPES`, which does not compile against this host's torch 2.9.1/cu129 (`error: cannot convert 'const at::DeprecatedTypeProperties' to 'c10::ScalarType'`, `ATen/Dispatch.h:198`). This is the well-documented, mechanical maskrcnn-benchmark-on-modern-PyTorch problem (also affects ROIPool, NMS, SigmoidFocalLoss, deform_conv — same pattern, ~8–15 call sites across ~10 files, by the shape of this codebase) | **yes, natively** — this is a two-stage detect-then-classify architecture built around `MODEL.ROI_RELATION_HEAD.USE_GT_BOX`/`USE_GT_OBJECT_LABEL` flags, exactly PredCls's own definition | **confirmed, not assumed**: real but bounded — every failure is the same class of fix (`.type()` → `.scalar_type()`, and likely a few `THC.h`/`AT_CHECK` renames elsewhere), tedious but mechanical, historically well-trodden by other users of this exact codebase family | **highest of any tractable candidate**: TDE rides free once this compiles, VCTree is a materially different context-encoder from IMP+'s LSTM, and PredCls is native — no architecture mismatch |
| **2** | **RelTR** (`yrcong/RelTR`) | DETR/transformer, one-stage, sparse fixed-query decoding | **checkpoint public and downloadable; codebase cloned and read this session** | **RULED OUT — no PredCls mode exists, and none can be added by porting.** Grepped the full codebase for `predcls`/`gt_box`-as-input/`use_gt`: zero hits. RelTR's queries jointly predict subject/object boxes *and* the predicate — there is no mechanism to condition the model on an externally-specified (subject, object) box pair the way two-stage detectors do. Forcing GT boxes in would require replacing the box-prediction sub-network with a box-matching/re-routing scheme — a research-level architecture change, not a port | n/a — architectural mismatch, not an engineering cost | **downgraded from last session's rank 1.** The earlier ranking assumed "modern codebase = tractable" without checking whether the *task* (PredCls) is even representable in a one-stage sparse-query architecture. It is not, for any codebase in this family (RelTR, and see EGTR below) |
| **3** | **EGTR** (`naver-ai/egtr`, CVPR 2024) | DETR/deformable-attention, one-stage | **checkpoint public; codebase cloned and read this session** | **RULED OUT, same reason as RelTR** — no `predcls`/GT-box-input hooks found; `evaluate_egtr.py` computes standard joint-detection graph metrics only | n/a, same architectural mismatch. Additionally uses **deformable attention** (`model/deformable_detr.py`), historically requiring a compiled custom CUDA kernel — a second, independent risk this codebase carries on top of the PredCls mismatch, not investigated further since the mismatch alone is disqualifying | **downgraded**, same reason as RelTR |
| **4** | **VCTree**, official (`KaihuaTang/Scene-Graph-Benchmark.pytorch`) | Tree-LSTM context + TDE | not re-cloned this session (the `mods333` fork, rank 1, is built directly on top of it and shares its exact `maskrcnn_benchmark/` source tree and blocker) | yes, native | same confirmed blocker as rank 1 | **redundant with rank 1** — attempt rank 1's fork first; fall back here only if that fork's fix turns out to diverge from upstream in some way that matters |
| **5** | **VCTree**, oldest official repo (`KaihuaTang/VCTree-Scene-Graph-Generation`, pre-2019, pre-`Scene-Graph-Benchmark.pytorch`) | Tree-LSTM context | not cloned | yes, native | strictly worse than rank 1/4 (older, less maintained fork of the same problem) | not worth pursuing while rank 1 exists |

**The single highest-value finding this session produced for Track B**: the
architectural gate (does the model even have a place to put a GT box pair
in?) matters *before* the environment gate, and it was not checked last
session. RelTR/EGTR looked cheap because their *environments* looked
tractable; they are not viable *at all* for this specific comparison,
regardless of environment, because one-stage sparse/dense query SGG models
do not expose a PredCls-shaped input/output interface. This is a permanent,
architecture-level exclusion for this family, not a cost estimate.

**This session reverses last session's ranking, on evidence, not on a change
of mind.** Last session ranked RelTR above VCTree/TDE on the assumption that
a modern, self-contained codebase implies low cost; this session's direct
code inspection found that assumption failed on the more important axis —
whether the task (PredCls) is even representable — before the environment
axis was ever reached. VCTree/TDE's environment blocker, meanwhile, is
confirmed real but is now a *known, bounded, mechanical* problem (a specific
API-migration pattern with a well-understood fix shape) rather than an
open unknown. **Net effect: VCTree/TDE move back to rank 1, matching the
task's originally stated priority order** — not because the environment
work became easier, but because the alternative was found to be a dead end
rather than merely harder-to-rank.

**Recommended next step for Track B, scoped for a dedicated session (not
squeezed into a mixed-analysis one):** patch
`~/external_models/energy-based-scene-graph/maskrcnn_benchmark/csrc/`
mechanically — replace `.type()` with `.scalar_type()` in every
`AT_DISPATCH_*` call, then rebuild and iterate on whatever the next error is
(likely `THC/THC.h` removal or `AT_CHECK`→`TORCH_CHECK` renames, both
well-known, mechanical fixes for this exact codebase vintage). Budget: a
few hours of iterative compile-fix-recompile cycles, not the multi-day
estimate this doc carried before empirical confirmation — the failure mode
is now precisely known, which is most of what made the original estimate
uncertain. The cloned repo and the isolated build venv
(`/tmp/vctree_probe_venv`, not preserved across VM restarts — recreate with
`torch==2.9.1+cu129` to match this host exactly) are left in place at
`~/external_models/` for continuity.

**Target for this track: n=3 first** (current: PURE, IMP+; next: whichever
of ranks 1–4 clears its B0 pilot first), **then reassess** before committing
to a 4th/5th model, per the task's own instruction.

### Phase plan per candidate (generic, applies to whichever is attempted first)

- **PHASE B0 (100–300 images, plumbing).** Confirm: environment installs;
  checkpoint loads with the expected key structure (0 missing/unexpected, as
  was verified for IMP+); PredCls mode (GT boxes+labels) is actually
  reachable in the codebase, not just SGDet; per-pair predicate logits can be
  extracted in WPRD's required shape (`{model_term: Tensor(N,50)}` or
  `{per_image_logits: [...]}`, per `docs/PAPER1_EVALUATION_TABLE.md`'s
  interface). **Stop here if PredCls is not reachable without disproportionate
  engineering — do not force a SGDet-only model into a PredCls comparison.**
- **PHASE B1 (1,000–3,000 images, WPRD pilot).** Reproduce (approximately —
  see Paper A's evaluator-semantics caveat, `docs/PAPER_A_FREEZE_AUDIT.md`
  item 12, which applies to any new model's R@K too) a plausible R@50/mR@50
  range against the model's own published number, as a sanity gate on the
  port (this is exactly the check `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §1
  called a precondition for trusting WPRD). Compute WPRD, prior control,
  pair-matched null, geometry baseline (train-fit if the model's own train
  split converts cheaply, else cross-fit) on this subset.
- **PHASE B2 (full test split, only if B1 is scientifically informative).**
  Do not run this merely to get a third point with tighter CIs if B1 already
  gives a clean, decisive read (either clearly replicates or clearly breaks
  the geometry-≥-model pattern). A 1,000–3,000-image WPRD pilot has enough
  rows (comparable to or larger than `p26`'s original 3,000-image screening
  set that first found the effect) to be decisive by this programme's own
  precedent.

### Per-model report template (fill in for every candidate attempted)

- Standard evaluation reproduction status (plausible-range / exact / failed,
  with the published-target caveat stated)
- WPRD (macro, weighted, 95% CI)
- Prior control (must read exactly 0.5000)
- Pair-matched null (must collapse toward chance if pair identity, not
  image content, drives the raw score)
- Geometry baseline (train-fit if possible, else cross-fit, with the regime
  stated)
- Head/body/tail, only where cell counts are reported and support a point
  estimate (this programme's own tail-tail cells are consistently
  underpowered; a new model's will likely be too, at pilot scale especially)
- WPRD > 0.5? Geometry ≥ learned head? Does the PURE/IMP+ pattern replicate?
- For or against the current Paper B thesis, stated in one sentence with its
  evidence class (`MEASURED` vs. exploratory)

### Cross-model table (current, to be extended)

| Model | Family | Codebase | Split | N | WPRD | Geometry | Prior | Null | Reproduction status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| PURE | frozen-CLIP + frequency-prior ensemble | this repo | VG150 val+test | 132,556 / 132,334 | 0.5542 / 0.5446 | 0.5961 / 0.5916 (train-fit) | 0.5000 / 0.5000 | 0.4983 / 0.5001 | exact (own checkpoint) | `p33`, `p59` |
| IMP+ | Neural-Motifs (LSTM object-context) | `bknyaz/sgg` | VG150 test | 183,640 | 0.6205 | 0.6229 (cross-fit) | 0.5000 | 0.4997 (pair-matched) | plausible range, PARTIALLY EXPLAINED gap vs. correct published row | `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §10 |
| VCTree | Tree-LSTM context | `mods333/energy-based-scene-graph` | TBD | — | — | — | — | — | **BLOCKED**, confirmed this session: `ROIAlign_cpu.cpp` fails to compile against torch 2.9.1 (`AT_DISPATCH_FLOATING_TYPES`/`input.type()` API removed) | checkpoints confirmed available (direct PredCls links); mechanical fix identified, not yet applied |
| TDE | causal adjustment on Motifs/VCTree | `Scene-Graph-Benchmark.pytorch` | TBD | — | — | — | — | — | **BLOCKED**, same root cause (shares `maskrcnn_benchmark/` source tree) | rides free once VCTree's env is fixed |
| RelTR | DETR/transformer, one-stage sparse-query | `yrcong/RelTR` | n/a | — | — | — | — | — | **RULED OUT** | no PredCls mode exists in the architecture; confirmed by direct code read this session, not a porting-cost judgment |
| EGTR | DETR/deformable-attention, one-stage | `naver-ai/egtr` | n/a | — | — | — | — | — | **RULED OUT**, same reason as RelTR | also carries an unexamined deformable-attention CUDA-kernel risk, moot given the architectural exclusion |

---

## Track C — ranked experiments (info gain / GPU minute)

Post-`p67` state: readout (H5) dead, objective (H2) refuted on both
channels, supervision (H8) refuted, the minimal spatial-feature fix
(NEUTRAL), cleaned-fusion (HARMFUL), nonlinear decodability of `dy_rel`
(present, R²=0.292) and `dx_rel` (weakly present, R²=0.073), and now the
auxiliary-loss readout fix (NEUTRAL) are all closed. Every additive,
frozen-readout, and readout-training-signal intervention this programme has
designed has been tried.

| rank | experiment | cost | status | outcome |
|---|---|---|---|---|
| 1 | Minimal `dx_rel`/`dy_rel` fix + cleaned-fusion, matched estimator | CPU, 6 min | **DONE (`p65`)** | MINIMAL-FIX-NEUTRAL, CLEAN-FUSION-HARMFUL |
| 2 | Nonlinear (MLP) decodability of `dx_rel`/`dy_rel` from `rel_feat` | CPU, 4 min | **DONE (`p66`)** | `dy_rel` R²=0.292 (>chance, moderate); `dx_rel` R²=0.073 (>chance, marginal) |
| 3 | Auxiliary `dx_rel`/`dy_rel` regression loss on the frozen-`rel_feat` readout | CPU, 6.5 min | **DONE (`p67`)** | AUX-LOSS-NEUTRAL, flat across a 4x lambda sweep — the readout-fix branch is closed too |
| 4 | Held-out TEST replication of `p60`'s estimator-matched-geometry finding | CPU, ~5–8 min | **not run** | `p60` (the anchor `B_geometry`=0.5976 that `p65`/`p67` both build on) is validation-only, unlike most other load-bearing WPRD numbers. Cheap, closes a real freeze-quality gap |
| 5 | Role-swap / directional WPRD variant | CPU, ~30–60 min (new tool) | **not run**, motivated by `docs/BENCHMARK_LITERATURE_GAP.md` | more a Track B/benchmark-development item than a successor-architecture one |

**The successor-architecture ladder is now, by this session's own accounting,
exhausted for anything that leaves the encoder frozen** (6 independent
tests, 3 distinct intervention kinds, 2 estimator families — see
`docs/AUXILIARY_SPATIAL_LOSS_RESULT.md`'s summary table). The one remaining
untested class — a jointly retrained encoder — is not recommended as an
immediate next step: `p67` specifically weakens its expected payoff, since
the readout could not use even a directly, explicitly supervised,
information-theoretically-real signal. Rank 4 (test-split replication of
`p60`) is the highest-value remaining Track C item precisely because it is
a freeze-quality chore, not a new hypothesis — everything else on this
ladder that could cheaply be tried, has been.

---

## C. CPU/GPU parallelism

No GPU job is currently running (this session executed `p65` CPU-only).
When a Track B GPU pass is eventually launched (a B1/B2 pilot on a new
model), the following can run concurrently on CPU without contending for
the GPU or for `p65`'s already-freed memory:

- Track C ranks 2–4 above (all CPU-only, all reuse the existing `p36`
  cache — no dependency on any new GPU work)
- The new model's own annotation/format converter code, written and unit
  tested against small synthetic fixtures before the GPU pass needs it
- `docs/PAPER_A_FREEZE_AUDIT.md`'s two outstanding **SMALL ANALYSIS
  REQUIRED** items (evaluator-semantics reconciliation; systematic
  literature pass) — pure documentation/reading work
- Manuscript-scaffolding files for Paper A (`paper_a/00_outline.md` etc.,
  per the freeze audit's "next files" list) — no GPU or shared-cache
  dependency at all

## D. Runtime estimates

This project's own measured throughput (PURE, this exact L4, full
forward-pass + `rel_feat` dump): `p24` 12,423 s / 10,401 images = **1.194
s/image**; `p54` 12,685 s / 10,403 images = **1.219 s/image**. Used below as
a *planning anchor only* — a new model's real throughput depends on its own
architecture (transformer vs. CNN+LSTM, whether PredCls skips the detector
forward pass) and **must be measured via its own B0 pilot**, not assumed.

| pilot size | PURE-anchor estimate (this repo's own throughput) | new-model estimate |
|---|---|---|
| 300 images | ~360 s (~6 min) | **TBD — measure in B0** |
| 1,000 images | ~1,200 s (~20 min) | **TBD — extrapolate from B0's per-image time** |
| 3,000 images | ~3,660 s (~61 min) | **TBD** |
| full test split (10,403 images) | 12,685 s (3 h 31 m) — already measured (`p54`) | **TBD**, and per policy not to be run until a pilot justifies it |

## E. Executed this session

**`p65`** — the top-ranked Track C item, CPU-only, pre-registered, 342 s
wall clock, all 6 gates PASS, both pre-registered verdicts settled
(MINIMAL-FIX-NEUTRAL, CLEAN-FUSION-HARMFUL). Chosen because it was clearly
justified (closed a previously-void test and a never-attempted combination
named directly by this programme's own prior results), reproducible (reused
`p60`'s exact, already-validated estimator and folds), bounded (<6 minutes),
and did not touch the GPU at a moment when another project's process was
transiently resident on it. Full writeup:
`docs/MINIMAL_FIX_AND_CLEAN_FUSION_RESULT.md`.
