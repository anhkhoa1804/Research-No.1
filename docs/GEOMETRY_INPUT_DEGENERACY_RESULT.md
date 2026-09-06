# `p68` — the geometry vector PURE actually receives is 6/8 constant

Run: `runs/p68_geom_gate_pilot` (GPU, 3 batches x 8 images, ~1 min).
Tool: `tools/geom_gate_diagnostics.py` (purely observational wrapper; changes
no forward-pass numerics). Checkpoint:
`checkpoints/demo_best/pure_best_adapt_light_mR50.pt`
(sha256 `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442`).

This was **not pre-registered**: it is an audit/observation of an existing
forward pass, not a hypothesis test. It states measurements and a source-level
cause. Every causal claim about what *fixing* it would do is explicitly marked
untested below.

---

## 1. The measurement (MEASURED, bit-exact)

Captured in-situ from the live forward pass, immediately after the
normalisation at `openvocab_rel/models/relational_model.py:725`, over
**17,196 real validation pairs**:

| col | min | max | std | frac exactly 0 | distinct values |
|---|---|---|---|---|---|
| `dx` | −1.0000 | 1.0000 | 0.32582 | 0.033 | 2101 |
| `dy` | −0.9982 | 0.9982 | 0.27539 | 0.006 | 1969 |
| `rw` | −0.0000 | −0.0000 | 0.00000 | 0.000 | **1** |
| `rh` | −0.0000 | −0.0000 | 0.00000 | 0.000 | **1** |
| `ar1` | −0.0000 | −0.0000 | 0.00000 | 0.000 | **1** |
| `ar2` | −0.0000 | −0.0000 | 0.00000 | 0.000 | **1** |
| `a1` | 0.0000 | 0.0000 | 0.00000 | **1.000** | **1** |
| `a2` | 0.0000 | 0.0000 | 0.00000 | **1.000** | **1** |

**Six of the eight geometry channels are bit-exactly constant.** Only `dx` and
`dy` vary, and they vary in units of the *whole frame*, not of the subject box.

## 2. The cause (VERIFIED FACT, source-level)

`relational_model.py:725` normalises boxes to `[0, 1]`:

```python
norm_boxes = obj_boxes[bi] / float(self.decoder.img_res)   # img_res = 336
geom_feat  = geom_feats_torch(norm_boxes[s_idx], norm_boxes[o_idx])
```

`geometry.py::geom_feats_torch` then applies, to those already-normalised
boxes:

```python
w1 = (x2 - x1).clamp_min(1.0)   # written for PIXEL-space boxes
h1 = (y2 - y1).clamp_min(1.0)
```

A `[0, 1]`-normalised box has width and height **at most 1.0**, so every
`clamp_min(1.0)` binds and returns exactly `1.0`. Therefore, identically and
for every pair:

- `rw  = log(w2/w1)  = log(1/1) = 0`
- `rh  = log(h2/h1)  = 0`
- `ar1 = log(w1/h1)  = 0`
- `ar2 = log(w2/h2)  = 0`
- `a1  = log(w1*h1)  = log(1) = 0`
- `a2  = log(w2*h2)  = 0`
- `dx  = (c2x − c1x)/(1.0 + 1e-6)` — survives, but divided by the frame, not by
  the subject's width
- `dy  = (c2y − c1y)/(1.0 + 1e-6)` — likewise

This is a **units-contract violation at the call site**, not a bug inside
`geom_feats_torch`. The function is correct for the pixel-space domain it was
written for.

**Why the test suite never caught it.** Every box in `tests/test_geometry.py`
is pixel-scale (`10.0`, `20.0`, `100.0`, `200.0`), so `clamp_min(1.0)` never
binds in any test. The production call site is the only place the function is
used in the normalised regime, and nothing tests that regime. The tests are
passing and correct; they simply do not cover the domain production uses.

## 3. What the model consequently never receives

Subject size, object size, their ratio, both aspect ratios, and both areas.
Equivalently: **the model has no access to box scale at all** through the
geometry path — only to frame-relative centre offsets.

This is the crux for `p57`. `p57` reported that `dx_rel` is barely decodable
from `rel_feat` (R² = 0.052) and named it "the encoder discards relative
position". `dx_rel` as the *probe* defines it is `(c2x − c1x)/w_subject`. The
model's `dx` is `(c2x − c1x)/W_frame`. The two differ by exactly the factor
`w_subject / W_frame` — subject width — which is one of the six channels
zeroed above. So `dx_rel` is not merely under-used by the encoder; it is
**not a function of the encoder's inputs** (except through vision).

`p66` fits this too: `dy_rel` is *partly* nonlinearly recoverable (R² = 0.292)
and `dx_rel` only weakly (R² = 0.073) — consistent with the residual being
supplied by the visual branch rather than the geometry branch.

## 4. Reclassification this forces (INTERPRETATION, not new numbers)

No measured number anywhere in this project changes. What changes is the
*reading* of H6.

- **Before:** "the 768-d learned relational representation sits below 19 raw
  box numbers — the encoder destroys spatial information."
- **After (accurate):** "the 768-d representation sits below 19 raw box
  numbers, and 6 of the 8 geometry channels it was supposed to be built from
  were bit-exactly constant. The comparison was never between a learned
  encoder and raw geometry; it was between an encoder given 2 frame-relative
  offsets and a probe given 19 real box numbers."

The geometry probe's advantage (0.5961 / 0.5976 vs `rel_feat`'s 0.5732) is
therefore **partially mechanically explained** rather than unexplained. It does
*not* become an artifact — the probe's numbers are correct and the checkpoint's
deficit is real; what changes is that the deficit now has an identified,
verifiable, non-mysterious cause.

## 5. What this does NOT establish (stated explicitly)

- **It does not show that fixing the units will raise WPRD.** Untested. The
  visual branch may already encode object scale; `dx`/`dy` in frame units may
  carry most of the usable signal; and `p67` showed the readout failed to
  convert even a directly supervised, information-theoretically-present
  signal. The expected payoff is *higher* than before this finding, but it is
  a hypothesis, not a result.
- **It does not invalidate any frozen Paper A claim.** Paper A's claims
  describe what the checkpoint *does* (prior dominance, the model term being
  pair identity in text-embedding space, WPRD's exact 0.5000 prior control,
  geometry out-discriminating the checkpoint). All remain true as measured.
  This finding explains one of them; it contradicts none. **Flagged, not
  acted on:** any Paper A prose asserting that PURE's architecture *uses* box
  geometry should be re-read against §1 before the manuscript is submitted.
- **It does not mean the checkpoint is invalid or should be retrained.** The
  checkpoint is the historical scientific artifact and stays untouched.

## 6. Second finding: the vector fusion gate is empirically constant

Same run, same 17,196 pairs.

- `decoder.geom_alpha` = **0.10000000149011612**, bit-identical to its
  `nn.Parameter(torch.tensor([0.1]))` initialisation. It received **zero**
  gradient. Reason (VERIFIED FACT): the checkpoint's own saved `cfg` dict has
  `use_geom_bias=True`, `vector_fusion_gate=True`, so `forward_pairs` takes
  the vector-gate branch (`relational_model.py:454-463`) and the `geom_alpha`
  branch (`:464-467`) is **never executed**. `geom_alpha` is dead code in this
  checkpoint, not a collapsed gate.
- The gate that *is* live, `decoder.fusion_gate`, is architecturally
  input-dependent (an MLP over `[sub_norm, obj_norm, geom_norm]`, sigmoid at
  temperature 0.7, per-dimension). **Empirically it is constant**: mean
  **0.50155**, std across pairs **0.00013**, full range
  [0.50112, 0.50191], and **no** dimension anywhere below 0.1 or above 0.9.

Interpretation, carefully: gate ≈ 0.5 means
`fused_feat = 0.5*sem_feat + 0.5*geom_norm` — geometry is **not** suppressed;
it enters at equal weight. What has collapsed is the gate's *adaptivity*: the
model does not modulate geometry use per pair. Note `gate_regularizer_weight`
= 0.002 with penalty `((mean_gate − 0.5)**2)`, which explicitly pulls toward
0.5. **Do not read "gate = 0.5" as "geometry ignored".** The geometry channel
is open at half weight; it is simply carrying a 6/8-constant vector (§1).

## 7. Sanity check on the run

PredCls R@50 = 0.6683 on this 24-image subset, against the historical
full-split 0.6709 — plausible range, so the pilot's forward pass is behaving
like the historical evaluator and the diagnostics are not measuring a broken
configuration. `[ActiveBranches]` confirmed `vector_fusion_gate: true`,
`geom_bias: true`.

## 8. The experiment this makes decisive (NOT YET RUN)

The minimal intervention is now unusually well-specified and small: compute
the geometry features in a scale-correct way (or equivalently pass pixel-space
boxes), change nothing else, and jointly retrain from the same
initialisation on the same subset with the same optimizer, budget, seed and
evaluation.

- **C0** = historical baseline configuration.
- **C1** = identical, with the geometry units fixed.
- Primary metric: **WPRD**. Secondary: R@50 / mR@50.
- Success requires WPRD improving materially beyond noise, reproducing on a
  second seed or held-out sample, and not being explained by a prior /
  calibration shift.

This is one hypothesis, one change, and it directly targets a *measured*
defect rather than a suspected one — which is what every previous Track C
intervention (`p58`, `p60`, `p65`, `p67`) lacked.
