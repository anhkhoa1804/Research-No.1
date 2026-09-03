# VCTree/TDE — environment FIXED and verified; checkpoint acquisition is the new, different blocker

GPU checked idle before and throughout this work (`nvidia-smi`: 0% util,
0 MiB, no processes at every check). No GPU training or evaluation was run
this session for Track B — everything below is environment engineering and
a checkpoint-download attempt, both bounded and logged.

## What changed: the environment blocker is closed

`docs/TRACK_B_C_ACTION_QUEUE.md` confirmed, empirically, that
`maskrcnn_benchmark`'s C++/CUDA extensions fail to compile against this
host's torch 2.9.1+cu129 (`AT_DISPATCH_FLOATING_TYPES`'s `.type()` vs modern
ATen's `c10::ScalarType`). This session fixed it:

- **Patch**: `tools/cross_model/patches/vctree_maskrcnn_benchmark_torch2_compat.patch`
  (mechanical `.type()`→`.scalar_type()`/`.is_cuda()`, a minimal THC shim
  for the two symbols actually used, `nms.cu`'s removed caching-allocator
  calls replaced with plain `cudaMalloc`/`cudaFree`, deformable conv/pool
  excluded as unused by the default PredCls config). Full rationale:
  `tools/cross_model/patches/README.md`.
- **Verified on two independent codebases**, both maskrcnn-benchmark
  descendants: `mods333/energy-based-scene-graph` (the VCTree fork) and
  `KaihuaTang/Scene-Graph-Benchmark.pytorch` (the original, TDE's home) —
  the identical patch applies cleanly and builds (`exit 0`) on both. This
  is stronger evidence than fixing one fork: **the fix is a property of the
  torch2 migration, not of one repository's fork drift.**
- **Smoke-tested, not just compiled**: `tools/cross_model/smoke_test_vctree_extensions.py`
  runs the two extensions PredCls actually needs (NMS, ROIAlign) on real
  GPU tensors and checks correctness (NMS suppresses the correct
  overlapping box; ROIAlign returns the correct shape, all finite) — passes
  on both codebases.
- **Data pipeline already 90% built**: `tools/cross_model/convert_vg150_coco_to_sgg_h5.py`
  (written for the earlier IMP+ cross-model work) already emits the exact
  `VG-SGG.h5` schema (`boxes_1024` in `cx,cy,w,h`, `img_to_first/last_box`,
  `img_to_first/last_rel`, `relationships`, `predicates`, `split`) that
  `maskrcnn_benchmark/data/datasets/visual_genome.py::load_graphs` reads —
  confirmed by reading `load_graphs`'s source directly, not assumed. One
  gap found: it does not emit an `attributes` dataset, which
  `load_graphs` also reads (`roi_h5['attributes'][:, :]`) — a small,
  known, not-yet-applied fix (add a dummy all-zero array of the right
  shape; attributes are irrelevant to PredCls's relation-only task and
  `USE_GT_OBJECT_LABEL=True` bypasses attribute prediction entirely).

**This environment fix is reusable for any maskrcnn-benchmark-family
checkpoint** (Motifs, VCTree, VTransE, and TDE's causal adjustment on any
of them) — it is not specific to the one checkpoint this session tried to
acquire.

## What did NOT change: no checkpoint has been evaluated

### VCTree: checkpoints confirmed unavailable, not merely hard to reach

Every VCTree-PredCls checkpoint link this session could find is dead:

- `mods333/energy-based-scene-graph`'s README links (`VCTree-Predcls` CE
  and EBM variants) all resolve (via `tinyurl.com`) to
  `csubcca-my.sharepoint.com/personal/suhail33_cs_ubc_ca/...` — a personal
  academic OneDrive account. All three checked return **404 from
  SharePoint itself** (not a redirect or format issue — traced the full
  redirect chain, confirmed the final destination 404s).
- `KaihuaTang/Scene-Graph-Benchmark.pytorch`'s own README states directly:
  *"I won't upload all the pretrained SGG models here... you can follow
  the rest instructions to train your own"* — **no VCTree checkpoint was
  ever published by the paper's own authors.** Training one from scratch
  (2 GPUs, unspecified but substantial wall-clock) is out of scope for a
  cross-model *measurement* study and was not attempted.

**VCTree is closed for this session, not indefinitely** — if a VCTree
PredCls checkpoint surfaces from another source, the environment above is
ready for it immediately.

### TDE (via Motifs): environment ready, checkpoint download blocked by anti-automation, not by anything this session controls

Tang's own README **does** publish a usable checkpoint: **Causal
MOTIFS-SUM, PredCls**, at
`https://1drv.ms/u/s!AmRLLNf6bzcir9xx725wYjN7lytynA?e=0B65Ws` — this is
exactly what Track B's TDE requirement needs (Motifs-family, PredCls,
native `MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE` support for `none`,
`TDE`, `NIE`, `TE` on the *same* forward pass, so TDE is genuinely "a
second CPU-side scoring arm" on already-computed logits, not a second GPU
pass, exactly as the directive anticipated).

**Blocked at download, confirmed by direct testing, not assumed:**

| attempt | result |
|---|---|
| Direct `curl` on the `1drv.ms` short link, following redirects | 301 → `onedrive.live.com/...&migratedtospo=true&redeem=...` → **403** |
| Appending `&download=1` to the redeemed destination URL | **403** (871-byte error body, not a file) |
| Legacy `api.onedrive.com/v1.0/shares/{token}/root/content` direct-download API (base64 share-token encoding) | **401** — this endpoint no longer serves shares migrated to SharePoint Online (`migratedtospo=true` in the redirect is the tell) |
| Web search for a mirror (Hugging Face, another fork) | none found for this specific checkpoint |

This is a **personal Microsoft OneDrive/SharePoint anonymous-link
redemption flow**, which requires a real browser session (JavaScript
execution, session cookies, CSRF token from the redemption page) that no
scripted HTTP client can complete — not a URL-format problem, and not
something more patching or retrying resolves. This is the same class of
blocker as `docs/CROSS_MODEL_FEASIBILITY.md`'s earlier, now-resolved
`VG-SGG.h5` acquisition problem, except this time the resolution (a
COCO-format mirror) does not apply because the object being fetched is a
*trained checkpoint*, not a *dataset*, and no equivalent alternative-format
mirror exists for it.

**Stopped here rather than continuing to search for workarounds**, per
this session's own decision rule (do not grind indefinitely on one
blocker) — after four independent automated-download attempts failed for
a structural, not incidental, reason.

## What would unblock this, cheaply

1. **A human downloads the file via browser** (the OneDrive link works
   fine in an actual browser — this is exactly the friction the
   anonymous-link redemption flow is designed to require) and places it at
   `~/external_models/checkpoints/causal_motifs_predcls.pth`. This is a
   ~5-minute manual step that fully unblocks the rest of the pipeline,
   which is otherwise ready.
2. Alternatively, a different published Motifs/VCTree/VTransE PredCls
   checkpoint hosted somewhere other than personal OneDrive (Google Drive
   direct-download links, and most academic institutional hosting, do not
   have this specific blocker).

## Consequence for Track B's status

**No B0 pilot ran.** The gate that failed was checkpoint acquisition, not
environment, not architecture, not data. This is a materially different,
and better, position than either prior session: the specific, confirmed
blocker from last session (compile failure) is now fixed and reusable;
this session's new finding (checkpoint hosting) is narrower and has an
obvious, cheap human-in-the-loop resolution rather than requiring more
engineering.

Per `docs/TRACK_B_C_ACTION_QUEUE.md`'s stop condition ("VCTree cannot be
made reproducible without changing the architecture") — **not triggered**;
the architecture and environment are both fine. The correct read is
"blocked on checkpoint acquisition," not "blocked on reproducibility."
