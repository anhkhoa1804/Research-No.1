# VCTree environment patch — torch2/CUDA-12.9 compatibility

`docs/CROSS_MODEL_FEASIBILITY.md` and `docs/TRACK_B_C_ACTION_QUEUE.md`
documented, then reproduced, the exact blocker preventing any
`maskrcnn-benchmark`-family PredCls codebase (VCTree, TDE, Motifs) from
building against this host's torch 2.9.1+cu129: the C++/CUDA extensions in
`maskrcnn_benchmark/csrc/` use ATen/THC APIs removed from modern PyTorch.
This directory holds the fix, applied to
[`mods333/energy-based-scene-graph`](https://github.com/mods333/energy-based-scene-graph)
(a VCTree fork of `Scene-Graph-Benchmark.pytorch` with direct PredCls
checkpoint links).

**Status: BUILDS AND RUNS.** `vctree_maskrcnn_benchmark_torch2_compat.patch`
applies cleanly to a fresh clone; the resulting `_C` extension imports and
both extensions PredCls actually needs (NMS, ROIAlign) pass
`tools/cross_model/smoke_test_vctree_extensions.py` on this host's GPU.

## Reproducing the build

```bash
git clone https://github.com/mods333/energy-based-scene-graph.git
cd energy-based-scene-graph
git apply /path/to/Research-No.1/tools/cross_model/patches/vctree_maskrcnn_benchmark_torch2_compat.patch

python3 -m venv /path/to/venv
/path/to/venv/bin/pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu129
/path/to/venv/bin/pip install ninja yacs cython matplotlib tqdm opencv-python-headless overrides

python3 setup.py build develop   # (using /path/to/venv's python)

python3 /path/to/Research-No.1/tools/cross_model/smoke_test_vctree_extensions.py
```

Pin torch/torchvision to whatever this host's own `.venv` uses
(`torch.__version__`, `torch.version.cuda`) — the patch's kernels were
compiled and tested against exactly `torch==2.9.1+cu129`,
`torchvision==0.24.1+cu129`. A different torch minor version may need a
re-check of the same failure classes below, though the fixes are unlikely
to change in kind.

## What the patch does, and why each piece is there

1. **Excludes deformable conv/pool from the build**
   (`maskrcnn_benchmark/csrc/_disabled_deform/`, see its `WHY_DISABLED.md`).
   Not used by the default R-50-FPN VCTree PredCls config
   (`STAGE_WITH_DC` is off); excluding it cut the compatibility surface
   from 15 files / 98 legacy-API occurrences to 9 files / ~30 occurrences.
   `vision.cpp`'s pybind11 bindings for `deform_*` symbols are removed to
   match — restorable together if a future config needs deformable convs.

2. **`.type()` → `.scalar_type()` / `.is_cuda()`.** `AT_DISPATCH_FLOATING_TYPES(x.type(), ...)`
   and `x.type().is_cuda()` are the pre-1.5 ATen API; modern ATen's
   `AT_DISPATCH_*` macros take a `c10::ScalarType`
   (`x.scalar_type()`), and `.is_cuda()` is a direct `Tensor` method.
   Purely mechanical, applied identically everywhere the pattern occurred
   (`ROIAlign.h`, `ROIPool.h`, `nms.h`, `SigmoidFocalLoss.h`, both `.cpp`
   and `.cu` implementations).

3. **`THC/THC.h` and friends → `maskrcnn_benchmark/csrc/compat/thc_shim.h`.**
   THC.h (and `THCAtomics.cuh`, `THCDeviceUtils.cuh`) were removed
   entirely from libtorch. The only two symbols this codebase's kernels
   actually call from them are `THCCeilDiv` (ceiling division) and
   `THCudaCheck` (post-launch CUDA error checking) — the shim defines
   both against modern `c10`/`ATen` (`C10_CUDA_CHECK`,
   `__host__ __device__` ceil-div), and nothing else. Not a general THC
   replacement, scoped to exactly what's called here.

4. **`nms.cu`'s `THCState`/`THCudaMalloc`/`THCudaFree` → plain
   `cudaMalloc`/`cudaFree`.** The legacy THC caching-allocator API is gone;
   NMS allocates one scratch buffer per call, so bypassing PyTorch's
   caching allocator for it costs a small, one-off allocation overhead per
   NMS call and nothing else — acceptable for inference, not something to
   optimise before it's shown to matter.

## What was NOT touched

- No change to any `.py` file, any predictor (`VCTreePredictor` etc.), any
  training loop, or the checkpoint format.
- ROIPool and SigmoidFocalLoss got the identical mechanical patch (same
  three fix classes) because `vision.cpp`'s single extension module must
  compile as a unit, but PredCls with GT boxes does not call either at
  inference time (they are detection/SGDet-path only) — the smoke test
  does not exercise them for that reason, not because they were assumed
  correct without the mechanical fix being necessary.
- `apex` (mixed-precision training helper) is in `INSTALL.md` but is a
  training-only dependency; not installed or needed for a PredCls
  inference pass.
