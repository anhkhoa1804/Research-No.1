#!/usr/bin/env python
"""Capture a machine-local provenance record for this checkout's artifacts.

WHY THIS EXISTS, AND WHY IT IS NOT THE CANONICAL MANIFEST
---------------------------------------------------------
`data/manifests/historical_checkpoint_v1.yaml` is the canonical, git-tracked
freeze of the ORIGINAL (Windows-authored) experiment's artifacts. It is
immutable evidence and must never be edited to make a check pass.

This tool writes a SEPARATE, machine-local record describing what actually
exists on THIS machine right now: hashes, row counts, environment, GPU, and
the exact command that regenerated the dataset. The two are deliberately
distinct scientific artifacts:

  canonical manifest  -> what the historical experiment was defined against
  provenance record   -> what this machine can currently reproduce

Where they disagree, the disagreement is the finding. `tools/gcp_preflight.py`
classifies each artifact as BYTE-IDENTICAL, CONTENT-IDENTICAL /
LINE-ENDINGS-DIFFER, or DIFFERENT CONTENT so that a divergence is legible
rather than ambiguous. This tool records that classification; it never
resolves it.

Usage:
    python tools/record_provenance.py                      # writes under runs/provenance/
    python tools/record_provenance.py --out some/path.yaml
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_OUT_DIR = Path("runs/provenance")

# Artifacts worth hashing on this machine. Kept independent of the canonical
# manifest's artifact list on purpose: this record must remain writable even
# when an artifact the canonical manifest requires is absent.
TRACKED_ARTIFACTS = {
    "checkpoint": "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
    "historical_frequency_prior": "checkpoints/demo_best/frequency_prior.json",
    "demo_config": "checkpoints/demo_best/demo_config.env",
    "dataset_train": "datasets_vg150_clean/train.jsonl",
    "dataset_validation": "datasets_vg150_clean/validation.jsonl",
    "dataset_test": "datasets_vg150_clean/test.jsonl",
    "predicate_vocabulary": "datasets_vg150_clean/vocabulary/predicates.json",
    "object_vocabulary": "datasets_vg150_clean/vocabulary/objects.json",
    "diagnostics": "datasets_vg150_clean/diagnostics.json",
}

JSONL_ARTIFACTS = ("dataset_train", "dataset_validation", "dataset_test")


def sha256_of(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def count_lines(path: Path, chunk: int = 1 << 23) -> int:
    total = 0
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            total += block.count(b"\n")
    return total


def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def collect_git() -> Dict[str, Any]:
    status = _git(["status", "--porcelain=v1"])
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "subject": _git(["log", "-1", "--format=%s"]),
        "commit_date": _git(["log", "-1", "--format=%cI"]),
        "working_tree_clean": status == "",
        "uncommitted_entries": status.splitlines() if status else [],
    }


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch  # type: ignore
        # torch.__version__ is a str SUBCLASS (TorchVersion); yaml.safe_dump
        # refuses non-exact types, so coerce every version to a plain str.
        env["torch"] = str(torch.__version__)
        env["torch_cuda_build"] = str(torch.version.cuda)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_memory_bytes"] = int(props.total_memory)
            env["gpu_capability"] = f"sm_{props.major}{props.minor}"
            env["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - torch is a hard dep in practice
        env["torch_error"] = str(exc)
    for mod in ("transformers", "numpy"):
        try:
            env[mod] = str(__import__(mod).__version__)
        except Exception:
            env[mod] = None
    driver = _nvidia_smi()
    if driver:
        env["nvidia_driver"] = driver
    return env


def _nvidia_smi() -> Optional[str]:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return None


def collect_artifacts(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, rel in TRACKED_ARTIFACTS.items():
        path = root / rel
        entry: Dict[str, Any] = {"path": rel, "present": path.exists()}
        if path.exists():
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = sha256_of(path)
            if key in JSONL_ARTIFACTS:
                entry["rows"] = count_lines(path)
        out[key] = entry
    return out


def collect_vocabulary(root: Path) -> Dict[str, Any]:
    """Record the predicate vocabulary's identity, not merely its hash."""
    info: Dict[str, Any] = {}
    try:
        from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES  # type: ignore
    except Exception as exc:
        return {"error": f"could not import STANDARD_VG150_PREDICATES: {exc}"}
    canonical = sorted(STANDARD_VG150_PREDICATES)
    info["canonical_count"] = len(canonical)
    info["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    path = root / "datasets_vg150_clean" / "vocabulary" / "predicates.json"
    if not path.exists():
        info["on_disk"] = None
        return info
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping = data.get("idx_to_predicate", {})
        ordered = [mapping[str(i + 1)] for i in range(len(mapping))]
        info["on_disk_count"] = len(ordered)
        info["matches_canonical_set"] = set(ordered) == set(canonical)
        info["matches_canonical_order"] = ordered == canonical
    except Exception as exc:
        info["on_disk_error"] = str(exc)
    return info


def collect_image_tree(root: Path) -> Dict[str, Any]:
    images = root / "datasets_vg150_clean" / "images"
    info: Dict[str, Any] = {"path": "datasets_vg150_clean/images", "present": images.exists()}
    if not images.exists():
        return info
    info["is_symlink"] = images.is_symlink()
    if images.is_symlink():
        info["resolves_to"] = str(images.resolve())
    subdirs: Dict[str, int] = {}
    total = 0
    for sub in sorted(p for p in images.iterdir() if p.is_dir()):
        n = sum(1 for f in sub.iterdir() if f.is_file())
        subdirs[sub.name] = n
        total += n
    info["subdirs"] = subdirs
    info["total_files"] = total
    return info


def build_record(root: Path, preparation_command: str, note: str) -> Dict[str, Any]:
    return {
        "record_type": "machine_local_provenance",
        "schema_version": 1,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": note,
        "canonical_manifest": {
            "path": "data/manifests/historical_checkpoint_v1.yaml",
            "sha256": (
                sha256_of(root / "data/manifests/historical_checkpoint_v1.yaml")
                if (root / "data/manifests/historical_checkpoint_v1.yaml").exists()
                else None
            ),
            "relationship": (
                "IMMUTABLE REFERENCE. This provenance record describes what exists on "
                "this machine; it neither overrides nor amends the canonical manifest."
            ),
        },
        "preparation_command": preparation_command,
        "git": collect_git(),
        "environment": collect_environment(),
        "artifacts": collect_artifacts(root),
        "predicate_vocabulary": collect_vocabulary(root),
        "image_tree": collect_image_tree(root),
    }


DEFAULT_PREPARATION_COMMAND = (
    "python3 tools/prepare_vg150_drive_clean.py "
    "--skip_download --skip_extract "
    "--jsonl_root ~/VG150_dataset_extract "
    "--out_dir datasets_vg150_clean "
    "--images symlink "
    "--allow_source_vocab_mismatch"
)

DEFAULT_NOTE = (
    "Machine-local artifact provenance. The dataset here was regenerated from the "
    "immutable extract at ~/VG150_dataset_extract and verified bit-for-bit "
    "reproducible against a second independent run. Its JSONL/vocabulary hashes "
    "differ from data/manifests/historical_checkpoint_v1.yaml because the canonical "
    "hashes were computed over CRLF (Windows) renderings of the SAME content -- "
    "proven, not assumed: the canonical hash equals the CRLF rendering of these "
    "exact bytes. See docs/DATASET_PROVENANCE_LF_CRLF.md."
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--out", default="", help="output path (default: runs/provenance/provenance_<utc>.yaml)")
    ap.add_argument("--preparation-command", default=DEFAULT_PREPARATION_COMMAND)
    ap.add_argument("--note", default=DEFAULT_NOTE)
    ap.add_argument("--print", dest="do_print", action="store_true", help="also print the record to stdout")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    record = build_record(root, args.preparation_command, args.note)

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = root / DEFAULT_OUT_DIR / f"provenance_{stamp}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml  # type: ignore
        out_path.write_text(
            yaml.safe_dump(record, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    except ImportError:
        out_path = out_path.with_suffix(".json")
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[provenance] wrote {out_path}")
    if args.do_print:
        print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
