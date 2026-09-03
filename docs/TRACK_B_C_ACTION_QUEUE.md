# Track B + Track C — ranked action queue

Mode: **PAPER B + PAPER C DEVELOPMENT**, Paper A frozen
(`docs/PAPER_A_FREEZE_AUDIT.md`) and not reopened here. GPU checked idle
before this session's only executed experiment (`p65`, CPU-only); a second
project's process (`extraction-gpu312`, PID 8131, ~2.4 GB) was observed
transiently occupying GPU memory at 0% utilization mid-session and had
exited by the end of it — logged, not competed with.

---

## Track B — ranked candidate models (info gain / cost)

Builds on `docs/CROSS_MODEL_FEASIBILITY.md` (bknyaz/sgg IMP+ already secured
— n=1 additional model, `docs/CROSS_MODEL_IMP_PLUS_RESULT.md`). This
session's web research updates the landscape for the task's stated
priorities (VCTree, TDE, one substantially different family) with current
checkpoint/codebase availability.

| rank | candidate | family | codebase | checkpoint | env risk | PredCls support | est. porting effort | est. GPU cost | info gain |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **RelTR** (`yrcong/RelTR`) | DETR/transformer, one-stage, no explicit frequency-bias term | self-contained, modern-ish PyTorch, no maskrcnn-benchmark | **public, VG-pretrained, downloadable** | **low** — standard transformer ops, no exotic custom CUDA kernels expected (needs direct verification) | **unconfirmed** — architecture is end-to-end detection+relation; whether GT boxes can be substituted for strict PredCls needs a code read | ~0.5–1 day (mostly: confirm PredCls mode, write the WPRD-cache extraction hook) | ~0.5–1.5 GPU-h for a full-split PredCls pass, if supported | **highest** — the only candidate architecturally unrelated to the frequency-prior-conditioned, object-context-LSTM paradigm IMP+ and PURE both share; a genuine test of whether "geometry ≥ learned head" generalizes outside that paradigm |
| **2** | **VCTree** via `mods333/energy-based-scene-graph` | Tree-LSTM context, Motifs-adjacent family | reports pretrained VCTree weights; framework requirements **not yet checked** this session | reported available | **unknown**, cheap to check (read one README) | GT-boxes PredCls, per family convention | 0.5 day if framework is modern; else falls back to rank 3's cost | TBD pending B0 | moderate — sibling to IMP+'s family, but a materially different context-encoding mechanism (tree-structured vs. sequential LSTM), useful second point in the Motifs-adjacent family |
| **3** | **VCTree + TDE**, official (`KaihuaTang/Scene-Graph-Benchmark.pytorch`) | Tree-LSTM context (VCTree) + causal-effect post-hoc adjustment (TDE) | `maskrcnn-benchmark`, torch 1.4–1.9, CUDA 10/11, many custom CUDA ops | public, OneDrive | **high** — same sm_89/old-torch/custom-CUDA-kernel blocker documented in `docs/CROSS_MODEL_FEASIBILITY.md`, unresolved | yes, GT-boxes PredCls is this repo's native mode | ~1–2 days (env bring-up is the risk, not the science) | ~1–2 GPU-h once running | **VCTree: moderate** (redundant with rank 2 if that lands); **TDE: high and free** once any model in this repo runs — TDE is an *inference-time* causal adjustment on an already-trained checkpoint's logits, not a separately trained model, so cracking this codebase once yields both VCTree and TDE (and Motifs, already covered elsewhere) simultaneously |
| **4** | **EGTR** (`naver-ai/egtr`, CVPR 2024) | DETR/transformer, backbone-shared relation extraction | modern, HuggingFace-adjacent, actively maintained (2024) | needs confirmation this session did not check | **low**, likely lower than RelTR given recency | needs confirmation | ~0.5–1 day | ~0.5–1.5 GPU-h | high, same rationale as RelTR (different family) — backup/parallel candidate to rank 1, worth a same-cost B0 check in parallel with RelTR rather than strictly sequenced after it |
| **5** | **VCTree**, official (`KaihuaTang/VCTree-Scene-Graph-Generation`) | Tree-LSTM context | pre-`Scene-Graph-Benchmark.pytorch`, likely an even older Faster-RCNN-based stack | public | **highest** — oldest codebase in the survey | yes | 2+ days, high failure risk | unknown | low priority — strictly dominated by rank 3's VCTree path (same architecture, newer/better-maintained host codebase) unless rank 3's env work is somehow easier to redirect at this repo specifically, which is not expected |

**Divergence from the task's stated priority order (VCTree, TDE, different
family), noted explicitly rather than silently reordered:** ranking by
information-gain/cost puts the architecturally-different family first,
because (a) IMP+ already secured one Motifs-adjacent point, so a second
Motifs-adjacent point (VCTree) has a lower marginal information gain than a
structurally different paradigm, and (b) the cheapest realistic path to
VCTree/TDE both run through a documented, previously-encountered hard
environment blocker (old `maskrcnn-benchmark` + CUDA extensions vs. this
host's sm_89/CUDA-13 stack), while RelTR/EGTR are modern, self-contained
codebases with no comparable blocker on record. **This is a ranking by cost
and marginal information, not a claim that VCTree/TDE are less scientifically
valuable** — TDE in particular is the more conceptually on-point candidate
for Paper B's discrimination/calibration thesis, since it is itself a
calibration-family intervention. If the env blocker for rank 3 turns out to
be cheaper than estimated (untested this session), it should be promoted.

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
| VCTree | Tree-LSTM context | TBD (rank 2 or 3 above) | TBD | — | — | — | — | — | **NOT YET RUN** | next candidate |
| TDE | causal adjustment on Motifs/VCTree | `Scene-Graph-Benchmark.pytorch` | TBD | — | — | — | — | — | **NOT YET RUN**, rides on rank-3 env work | — |
| RelTR / EGTR | DETR/transformer | own | TBD | — | — | — | — | — | **NOT YET RUN** | rank-1/4 candidate |

---

## Track C — ranked experiments (info gain / GPU minute)

Post-`p65` state: readout (H5) dead, objective (H2) refuted on both
channels, supervision (H8) refuted, and now — this session — the minimal
spatial-feature fix (NEUTRAL) and cleaned-fusion (HARMFUL) are also closed.
Every additive, frozen-readout intervention this programme has designed has
been tried. Ranked by expected information gain per unit cost, **not**
executed in this session except rank 1:

| rank | experiment | cost | status | why it's next |
|---|---|---|---|---|
| **1** | **Minimal `dx_rel`/`dy_rel` fix + cleaned-fusion, matched estimator** | CPU, ~6 min | **DONE this session (`p65`)** — MINIMAL-FIX-NEUTRAL, CLEAN-FUSION-HARMFUL | closed the one previously-void (`p58`) test and the one never-attempted cleaned-fusion combination |
| 2 | Nonlinear (kernel or shallow-MLP, not linear-R²) decodability of `dx_rel`/`dy_rel` specifically from `rel_feat` | CPU, ~10–15 min | not run | `p57`'s R²=0.052 for `dx_rel` is a **linear** decodability bound; a nonlinear probe distinguishes "genuinely absent" (C1) from "present but nonlinearly entangled" (a sharper version of C2) more rigorously than `p65`'s linear-input MLP-readout test did, since `p65` fed `dx_rel` as an extra linear input rather than asking whether it's nonlinearly recoverable from the other 768 dimensions |
| 3 | Held-out TEST replication of `p60`'s estimator-matched-geometry finding | CPU, ~5–8 min | not run | `p60` (the anchor for `B_geometry` = 0.5976 that `p65` built on) was run on validation only, unlike most of the rest of the programme's load-bearing WPRD numbers, which now have test replications (`p59`). Cheap, closes a real freeze-quality gap before this number is leaned on further |
| 4 | Role-swap / directional WPRD variant | CPU, ~30–60 min (new tool) | not run, motivated by `docs/BENCHMARK_LITERATURE_GAP.md` | tests whether swapping subject/object roles specifically probes the directional-predicate gap `p40`/`p57` identified (geometry wins spatially-decidable, directional contrasts; the model wins functional ones) — more a Track B/benchmark-development item than a successor-architecture one, ranked here because it is cheap and CPU-only |
| 5 | Small **joint-training** pilot: retrain only a small head (not the full encoder) on frozen CLIP features **with** `dx_rel`/`dy_rel`/geometry injected as an explicit auxiliary input from the start of training, not concatenated post-hoc onto a frozen `rel_feat` | GPU, small (~0.5–1 h estimated, needs its own pilot-scale check first) | not run — **the one remaining untested class** per `p65`'s own limitation | `p65` only tested frozen post-hoc probes; SUCCESSOR_HYPOTHESES.md's decision rule anticipated this exact fork. **Do not run this yet** — items 2–3 are cheaper and could still change the picture (e.g., if item 2 finds `dx_rel` IS nonlinearly present in `rel_feat`, a joint-training pilot targeting a different mechanism than "feed it the raw box numbers" might be indicated instead) |

**Recommendation, per the task's own decision logic:** run ranks 2–3 (both
CPU, both cheap, both close real gaps) before considering rank 5's GPU
pilot. Do not invent a sixth architecture; three consecutive additive-fusion
failures (`p37`, `p60`, `p65`) plus a neutral minimal fix is strong enough
evidence to make "no model, only the diagnosis and the benchmark" a live,
acceptable outcome — exactly as `docs/SUCCESSOR_HYPOTHESES.md` pre-committed
to accepting if this happened.

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
