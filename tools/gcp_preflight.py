#!/usr/bin/env python
"""Fail-safe preflight check for the historical-checkpoint GCP reproduction run.

Verifies every precondition established during the local audit BEFORE any GPU
time is spent. Fails loudly (nonzero exit, no partial success) on the first
violation rather than letting a missing/wrong artifact silently degrade into
a valid-looking evaluation.

Does not import torch/CLIP and does not touch openvocab_rel's model or
evaluator code -- pure filesystem/git/hash checks plus a light JSON parse.

Usage:
    python tools/gcp_preflight.py [--out runs/<run_name>/manifest.yaml]

Exit code 0 = all checks passed, manifest written.
Exit code 1 = at least one check failed; see printed [FAIL] lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT_MARKERS = ("openvocab_rel", "tools", ".git")

CHECKPOINT_PATH = Path("checkpoints/demo_best/pure_best_adapt_light_mR50.pt")
CHECKPOINT_SHA256 = "8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442"

FREQ_PRIOR_PATH = Path("checkpoints/demo_best/frequency_prior.json")
FREQ_PRIOR_SHA256 = "144d9f928ed9cc213acbe081b1a8791488a92ddab0ed885da7fd6bb6058c6e6a"

VG150_ROOT = Path("datasets_vg150_clean")
VALIDATION_JSONL = VG150_ROOT / "validation.jsonl"
EXPECTED_VALIDATION_ROWS = 10401

VOCAB_PATH = VG150_ROOT / "vocabulary" / "predicates.json"

failures: list[str] = []
manifest: dict = {}


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 23)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_repo_root() -> None:
    cwd = Path.cwd()
    missing = [m for m in REPO_ROOT_MARKERS if not (cwd / m).exists()]
    if missing:
        fail(f"CWD does not look like the repository root: {cwd} (missing {missing})")
    else:
        ok(f"CWD is repository root: {cwd}")
    manifest["cwd"] = str(cwd)


def check_git_state() -> None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=True
        ).stdout
    except Exception as exc:
        fail(f"git status failed: {exc}")
        return
    if status.strip():
        fail(f"working tree is dirty:\n{status}")
    else:
        ok("working tree is clean")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:
        fail(f"git rev-parse HEAD failed: {exc}")
        return
    ok(f"HEAD = {commit}")
    manifest["git_commit"] = commit
    manifest["git_dirty"] = bool(status.strip())

    # The three required fixes must be ancestors of HEAD.
    required_fixes = {
        "220c5c2e": "GT-pair extraction fix",
        "9dc8f45d": "canonical predicate vocabulary fix",
        "7d91af49": "opt-in source-vocab-mismatch override",
    }
    for short_sha, desc in required_fixes.items():
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", short_sha, "HEAD"],
                check=True, capture_output=True,
            )
            ok(f"required fix is in HEAD's ancestry: {short_sha} ({desc})")
        except subprocess.CalledProcessError:
            fail(f"required fix NOT in HEAD's ancestry: {short_sha} ({desc})")
        except Exception as exc:
            fail(f"could not check ancestry of {short_sha}: {exc}")


def check_file_sha256(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        fail(f"{label} missing: {path}")
        manifest[f"{label}_sha256"] = None
        return
    actual = sha256_of(path)
    manifest[f"{label}_sha256"] = actual
    manifest[f"{label}_path"] = str(path)
    manifest[f"{label}_size_bytes"] = path.stat().st_size
    if actual != expected:
        fail(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    else:
        ok(f"{label} SHA256 matches: {actual}")


def check_validation_split() -> None:
    if not VALIDATION_JSONL.exists():
        fail(f"validation split missing: {VALIDATION_JSONL}")
        manifest["validation_rows"] = None
        return
    with VALIDATION_JSONL.open("r", encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    manifest["validation_rows"] = n
    manifest["validation_jsonl_sha256"] = sha256_of(VALIDATION_JSONL)
    if n != EXPECTED_VALIDATION_ROWS:
        fail(f"validation.jsonl row count {n} != expected {EXPECTED_VALIDATION_ROWS}")
    else:
        ok(f"validation.jsonl row count == {EXPECTED_VALIDATION_ROWS}")

    train_jsonl = VG150_ROOT / "train.jsonl"
    if train_jsonl.exists():
        manifest["train_jsonl_sha256"] = sha256_of(train_jsonl)


def check_canonical_vocab() -> None:
    if not VOCAB_PATH.exists():
        fail(f"predicate vocabulary missing: {VOCAB_PATH}")
        return
    try:
        sys.path.insert(0, str(Path.cwd()))
        from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES  # type: ignore
    except Exception as exc:
        fail(f"could not import STANDARD_VG150_PREDICATES: {exc}")
        return
    try:
        data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        idx_to_pred = data["idx_to_predicate"]
        order = [idx_to_pred[str(i)] for i in range(1, 51)]
    except Exception as exc:
        fail(f"could not parse {VOCAB_PATH}: {exc}")
        return
    expected_order = sorted(STANDARD_VG150_PREDICATES)
    manifest["vocabulary_sha256"] = sha256_of(VOCAB_PATH)
    if order != expected_order:
        first_diff = next((i for i, (a, b) in enumerate(zip(order, expected_order)) if a != b), None)
        fail(f"predicate vocabulary is not sorted(STANDARD_VG150_PREDICATES); first diff at index {first_diff}")
    else:
        ok("predicate vocabulary == sorted(STANDARD_VG150_PREDICATES)")


def print_environment() -> None:
    manifest["python_version"] = platform.python_version()
    manifest["platform"] = platform.platform()
    print(f"[INFO] Python version: {manifest['python_version']}")
    print(f"[INFO] Platform: {manifest['platform']}")

    try:
        import torch  # type: ignore
        manifest["torch_version"] = str(torch.__version__)
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        print(f"[INFO] PyTorch version: {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            manifest["gpu_name"] = str(torch.cuda.get_device_name(0))
            manifest["cuda_version"] = str(torch.version.cuda)
            manifest["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            print(f"[INFO] GPU name: {manifest['gpu_name']}")
            print(f"[INFO] CUDA version (torch.version.cuda): {manifest['cuda_version']}")
            print(f"[INFO] GPU VRAM: {manifest['gpu_vram_gb']} GB")
        else:
            print("[WARN] CUDA not available -- this preflight is running on a non-GPU machine. "
                  "That is expected for a LOCAL dry run; the actual GCP instance must report CUDA=True.")
    except ImportError:
        print("[WARN] torch not importable in this environment -- version/CUDA info skipped.")
        manifest["torch_version"] = None
        manifest["cuda_available"] = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="path to write the manifest YAML/JSON (default: print only)")
    ap.add_argument("--allow_no_cuda", action="store_true",
                     help="do not fail if CUDA is unavailable (use for a local dry run only; "
                          "the real GCP preflight should NOT pass this flag)")
    args = ap.parse_args()

    print("=" * 70)
    print("GCP historical-reproduction preflight")
    print("=" * 70)

    check_repo_root()
    check_git_state()
    check_file_sha256(CHECKPOINT_PATH, CHECKPOINT_SHA256, "checkpoint")
    check_file_sha256(FREQ_PRIOR_PATH, FREQ_PRIOR_SHA256, "frequency_prior")
    check_validation_split()
    check_canonical_vocab()
    print_environment()

    if manifest.get("cuda_available") is False and not args.allow_no_cuda:
        fail("CUDA is not available on this machine -- refusing to treat this as a GCP-ready preflight "
             "(pass --allow_no_cuda for a local dry run only)")

    print("=" * 70)
    if failures:
        print(f"PREFLIGHT FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        manifest["preflight_status"] = "FAILED"
        manifest["preflight_failures"] = failures
    else:
        print("PREFLIGHT PASSED: all checks green")
        manifest["preflight_status"] = "PASSED"
        manifest["preflight_failures"] = []
    print("=" * 70)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml  # type: ignore
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
        except ImportError:
            out_path = out_path.with_suffix(".json")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        print(f"[INFO] manifest written to {out_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
