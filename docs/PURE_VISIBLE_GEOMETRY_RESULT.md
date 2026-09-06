# `p69` — the geometry PURE cannot see is worth +0.034 WPRD. Verdict: MATERIAL

Pre-registered in `docs/PURE_VISIBLE_GEOMETRY_PREREGISTRATION.md`, committed
before the tool existed (`19c8aac`). CPU only, no GPU. Run:
`runs/p69_pure_visible_geometry`. Tool: `tools/pure_visible_geometry.py`,
reusing `p60`'s estimator, folds and reporting path verbatim.

**This experiment was designed to kill the `p68` hypothesis. It failed to.**

## Result

| arm | WPRD | 95% CI | head–head | tail–tail |
|---|---|---|---|---|
| `A_relfeat` (768-d) | 0.5732 | [0.5674, 0.5782] | 0.5823 | 0.4859 |
| `B_geometry` (19 numbers) | **0.5976** | [0.5930, 0.6027] | 0.5988 | 0.5820 |
| `P_pure_visible` (**the 2 channels PURE receives**) | **0.5635** | [0.5581, 0.5688] | 0.5687 | 0.5398 |
| `Q_visible_plus_sizes` (those 2 + the 4 size channels) | 0.5940 | [0.5892, 0.5992] | 0.5967 | 0.4622 |
| `N_shuffled` (null) | 0.4992 | [0.4940, 0.5043] | 0.5022 | 0.5965 |
| `P_prior` (control) | 0.5000 | [0.5000, 0.5000] | 0.5000 | 0.5000 |

```
PRIMARY   delta_missing = B_geometry - P_pure_visible = +0.0341  -> MATERIAL
SECONDARY delta_size    = Q_visible_plus_sizes - P    = +0.0305
```

**Gates: 6/6 PASS.** `A_relfeat` reproduces `p60`'s 0.5732 to **±0.0000** and
`B_geometry` reproduces 0.5976 to **±0.0000**; folds match the registered
sizes exactly; prior control reads exactly 0.5000; shuffled null 0.4992.

## What it establishes (MEASURED)

1. **The geometry PURE is structurally denied is worth +0.0341 WPRD** —
   above the registered MATERIAL threshold of +0.03. This is the same order
   as the entire `rel_feat`-vs-geometry deficit the programme has been
   chasing (`B − A` = +0.0244).
2. **~90% of it is box size specifically.** `delta_size` = +0.0305 of the
   +0.0341 total comes from restoring just the four size channels
   (`sw/W, sh/H, ow/W, oh/H`) — all of which `p68` showed are bit-exactly
   zero in what the model receives. IoU, containment, aspect and log-area
   ratio add only the remaining ~+0.004 on top.
3. **The 2 channels PURE actually gets are genuinely weak on their own**:
   0.5635, well clear of chance but the weakest non-null geometry arm here.

## The finding that reframes H6

`A_relfeat` (0.5732) is **above** `P_pure_visible` (0.5635), by +0.0097 with
only marginal CI overlap (0.5674 vs 0.5688).

So, on this measurement, **the encoder is not destroying spatial information
relative to what it is given** — it ends up slightly *ahead* of its own
geometry inputs, presumably by recovering some layout from the visual branch.

This materially revises H6. The "representation bottleneck" was stated as
*the encoder discards spatial information*. The evidence now favours a
different account:

> **Input starvation, not representational destruction.** The encoder is
> denied 6 of 8 geometry channels — worth +0.034 WPRD, ~90% of it box size —
> and, given that impoverished input, still produces a representation
> slightly better than the input itself. The gap to the 19-number probe is
> mostly a gap in *what was delivered*, not in *what was retained*.

That is a sharper, more falsifiable, and considerably more publishable claim
than the elimination-based H6, and it predicts a specific intervention.

## Second defect found while `p69` ran: the geometry encoder is a random hash

Independent check against the checkpoint's own weights (`geom_B`, frozen,
`requires_grad=False`, std **9.91**, |max| 35.9). The Fourier map is
`proj = 2π·x @ geom_B`, so the phase rate is `2π·|B|`:

- **median 40.6 radians per unit `dx`**; mean 47.1, max 169.8.
- A full 2π phase cycle occurs every **0.155** of `dx` — and `dx` only spans
  [−1, 1].
- Empirical kernel on 20,000 random pairs (6 channels held at zero exactly as
  the model receives them): mean cosine similarity of the Fourier features is
  +0.178 at input distance 0.02–0.05 and **statistically indistinguishable
  from zero (|·| < 0.032) at every distance beyond 0.05**.

So even the two surviving channels are delivered through an encoding whose
effective length-scale is ~2–5% of the frame: two pairs whose centres differ
by more than 5% of the image get **uncorrelated** geometry embeddings. The
geometry pathway is, in effect, a **random hash of position with no
smoothness** — it cannot support interpolation or generalisation across
nearby layouts.

This is a *second*, independent defect from `p68`'s units bug, on the same
pathway, and any intervention that fixes the units must also keep the
encoding's bandwidth sane or the restored channels will be hashed away too.

## Explicit limits

- **Tail–tail is not readable here.** `N_shuffled`'s tail–tail reads 0.5965 —
  a pure null scoring far from chance is direct evidence that these cells are
  underpowered, exactly as `p35`/`p40` warned. Ignore every tail column above.
- **`P_pure_visible` is a proxy, and a generous one.** It uses per-image
  normalisation `(ocx−scx)/W` where PURE divides by a fixed 336. Per-image
  normalisation is mildly *more* informative, so this arm **overstates** what
  PURE receives — biasing against the MATERIAL verdict that was nonetheless
  reached. Registered in advance as such.
- **This is still not a causal demonstration.** It bounds the information
  content of the missing channels under one estimator. It does **not** show
  that a retrained encoder would convert them into discrimination. `p67` is
  the standing warning: a directly supervised, information-theoretically real
  signal still failed to move WPRD. The C1 pilot remains the test.

## Consequence

The registered decision rule returns **MATERIAL**, so the C0/C1 joint
retraining pilot is justified. For the first time in this programme, a
proposed intervention targets a **measured, mechanically identified,
quantified** defect rather than a suspected one — which is precisely what
`p58`, `p60`, `p65` and `p67` all lacked.
