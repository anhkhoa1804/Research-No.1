#!/usr/bin/env python
"""Measure CLIP encoder throughput and VRAM, so batch size is chosen from data.

Experiment scheduling should rest on measured throughput, not on defaults or
intuition. This encodes REAL crops taken from the dataset (not random noise,
whose preprocessing and memory behaviour differ) and reports crops/s and peak
VRAM per batch size.

The encoder is frozen and run under inference_mode. fp16 autocast is used on
CUDA by default and can be disabled to check it is numerically inert.

Usage:
    python tools/benchmark_clip_encoder.py --clip_name openai/clip-vit-large-patch14-336 \
        --device cuda --batches 16,32,48,64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def load_crops(split: Path, images_root: Path, n_crops: int) -> List[Any]:
    """Take real subject/object/union crops, the same shapes the probe encodes."""
    from PIL import Image

    index = {}
    for sub in ("VG_100K", "VG_100K_2"):
        d = images_root / sub
        if d.is_dir():
            for f in d.iterdir():
                index[f.stem] = f

    def crop(img, box, pad=0.0):
        w, h = img.size
        x1, y1, x2, y2 = box
        if pad:
            bw, bh = x2 - x1, y2 - y1
            x1 -= bw * pad; x2 += bw * pad; y1 -= bh * pad; y2 += bh * pad
        x1 = max(0, min(w - 1, int(x1))); y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2))); y2 = max(y1 + 1, min(h, int(y2)))
        return img.crop((x1, y1, x2, y2))

    crops: List[Any] = []
    with split.open(encoding="utf-8") as fh:
        for line in fh:
            if len(crops) >= n_crops:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            boxes = ex.get("obj_boxes") or []
            fp = index.get(str(ex.get("image_id", "")))
            if fp is None or not boxes:
                continue
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            crops.append(img)
            for b in boxes[:8]:
                if len(crops) >= n_crops:
                    break
                crops.append(crop(img, b, 0.1))
    return crops[:n_crops]


def _image_features(model, px):
    out = model.vision_model(pixel_values=px)
    pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
    return model.visual_projection(pooled)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip_name", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="16,32,48,64")
    ap.add_argument("--n_crops", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from transformers import CLIPImageProcessor, CLIPModel

    dev = torch.device(args.device)
    use_amp = (not args.no_amp) and dev.type == "cuda"

    print(f"[bench] loading {args.clip_name} on {dev}{' fp16-autocast' if use_amp else ' fp32'}", flush=True)
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(args.clip_name).eval().to(dev)
    proc = CLIPImageProcessor.from_pretrained(args.clip_name)
    load_s = time.perf_counter() - t0
    res = proc.crop_size if isinstance(getattr(proc, "crop_size", None), dict) else {}
    print(f"[bench] loaded in {load_s:.1f}s   input res {res.get('height')}   "
          f"proj dim {model.config.projection_dim}", flush=True)

    crops = load_crops(Path("datasets_vg150_clean/validation.jsonl"),
                       Path("datasets_vg150_clean/images"), args.n_crops)
    print(f"[bench] {len(crops)} real crops loaded", flush=True)

    rows: List[Dict[str, Any]] = []
    print(f"\n{'batch':>7}{'crops/s':>11}{'ms/crop':>10}{'peak VRAM MiB':>16}{'preproc %':>11}")
    for bs in [int(b) for b in args.batches.split(",") if b.strip()]:
        if dev.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                for _ in range(args.warmup):
                    px = proc(images=crops[:bs], return_tensors="pt")["pixel_values"].to(dev)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            _image_features(model, px)
                    else:
                        _image_features(model, px)
                if dev.type == "cuda":
                    torch.cuda.synchronize()

                n_done = 0; pre_s = 0.0
                t_start = time.perf_counter()
                for i in range(0, len(crops), bs):
                    chunk = crops[i:i + bs]
                    tp = time.perf_counter()
                    px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(dev)
                    pre_s += time.perf_counter() - tp
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            f = _image_features(model, px)
                    else:
                        f = _image_features(model, px)
                    F.normalize(f.float(), dim=-1).cpu()
                    n_done += len(chunk)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                total_s = time.perf_counter() - t_start
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>7}{'OOM':>11}")
            rows.append({"batch": bs, "oom": True})
            continue

        peak = torch.cuda.max_memory_allocated() / (1 << 20) if dev.type == "cuda" else 0.0
        cps = n_done / total_s
        rows.append({"batch": bs, "crops_per_s": cps, "ms_per_crop": 1000.0 / cps,
                     "peak_vram_mib": peak, "preprocess_frac": pre_s / total_s,
                     "n_crops": n_done, "seconds": total_s})
        print(f"{bs:>7}{cps:>11.1f}{1000.0/cps:>10.2f}{peak:>16.0f}{pre_s/total_s*100:>10.1f}%")

    ok = [r for r in rows if not r.get("oom")]
    if ok:
        best = max(ok, key=lambda r: r["crops_per_s"])
        print(f"\n[bench] best batch = {best['batch']}  ({best['crops_per_s']:.1f} crops/s, "
              f"{best['peak_vram_mib']:.0f} MiB peak)")
        out = {"clip_name": args.clip_name, "device": str(dev), "fp16_autocast": use_amp,
               "input_res": res.get("height"), "load_seconds": load_s,
               "rows": rows, "best_batch": best["batch"],
               "best_crops_per_s": best["crops_per_s"]}
        if args.out:
            p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"[bench] written to {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
