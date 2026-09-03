#!/usr/bin/env python
"""Smoke test for the patched maskrcnn_benchmark._C CUDA extensions.

Run inside the venv that built the patched
`mods333/energy-based-scene-graph` clone (see
`tools/cross_model/patches/README.md` for the exact build steps). Verifies
the two extensions the VCTree PredCls forward pass actually needs -- NMS and
ROIAlign -- both compile-time (import succeeds) and run-time (correct,
finite output on synthetic tensors) correct, on GPU.

This is not a general maskrcnn_benchmark test suite -- it targets exactly
the symbols this compatibility patch touched, per the "add a tiny
regression/smoke test for every compatibility change" instruction. ROIPool
and SigmoidFocalLoss got the identical mechanical patch (`.type()` ->
`.scalar_type()`, THC shim) but are not exercised by PredCls (they are
detection-path-only) and are not tested here for that reason, not because
they're assumed fine.

Exit 0 = both extensions produce correct, finite output. Exit 1 otherwise.
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    try:
        from maskrcnn_benchmark import _C
    except ImportError as e:
        print(f"FAIL: could not import maskrcnn_benchmark._C: {e}")
        return 1

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available in this venv -- the patched "
              "extensions were built for GPU inference")
        return 1

    ok = True

    # NMS: three boxes, two overlapping (should suppress the lower-score
    # one), one disjoint (should always survive).
    boxes = torch.tensor(
        [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.95]],
        dtype=torch.float32, device="cuda")
    keep = _C.nms(boxes[:, :4], boxes[:, 4], 0.5)
    if keep.numel() != 2 or set(keep.tolist()) != {0, 2}:
        print(f"FAIL: nms expected keep={{0,2}}, got {keep.tolist()}")
        ok = False
    else:
        print(f"OK: nms keep={keep.tolist()}")

    # ROIAlign: pool a known region out of a random feature map; check
    # shape and finiteness (this compiled kernel previously failed to
    # build at all -- correctness of the pooled *values* against a
    # reference implementation is not re-derived here, only that the
    # kernel runs and returns a sane tensor).
    feat = torch.randn(1, 4, 20, 20, device="cuda")
    rois = torch.tensor([[0, 2.0, 2.0, 10.0, 10.0]], dtype=torch.float32,
                        device="cuda")
    out = _C.roi_align_forward(feat, rois, 1.0, 7, 7, 2)
    if tuple(out.shape) != (1, 4, 7, 7) or not torch.isfinite(out).all():
        print(f"FAIL: roi_align_forward shape={tuple(out.shape)}, "
              f"finite={bool(torch.isfinite(out).all())}")
        ok = False
    else:
        print(f"OK: roi_align_forward shape={tuple(out.shape)}, all finite")

    print("ALL SMOKE TESTS PASSED" if ok else "SMOKE TEST FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
