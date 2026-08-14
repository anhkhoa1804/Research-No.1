#!/usr/bin/env python
"""Fail-loud preflight for the historical-checkpoint reproduction run.

Verifies every precondition established during the local audit BEFORE any
GPU time is spent, and captures a complete environment/artifact manifest so
the run is identifiable after the fact.

Design rules, in priority order:

1. **Never invent an expected value.** Every hash, row count and vocabulary
   expectation is read from ``data/manifests/historical_checkpoint_v1.yaml``,
   which was populated from artifacts actually measured on disk. This script
   contains no hardcoded hashes.
2. **Check what the evaluator will not tell you.** ``_load_frequency_bias``
   (``openvocab_rel/evals.py``) returns ``None`` -- meaning *silently no
   calibration* -- on six separate conditions and warns on none of them. A
   run against a missing or truncated frequency prior completes normally and
   emits a full, plausible, silently-uncalibrated ``metrics.jsonl``. This
   script checks all six conditions independently. See
   ``docs/known_issues.md``.
3. **Collect all failures, then exit nonzero.** A first-failure abort would
   hide a second broken artifact behind the first, costing another round
   trip to a cloud machine.
4. **Stay cheap.** No model is constructed, no checkpoint tensor is
   deserialized, no CLIP weights are fetched. Filesystem, git, hashing and
   JSON/YAML parsing only. ``torch`` is imported solely for environment
   reporting and is optional.

Usage
-----
    # local dry run (CUDA absent is tolerated, dirty tree is reported not fatal)
    python tools/gcp_preflight.py

    # the real thing, on the GCP instance
    python tools/gcp_preflight.py --strict --out runs/<run_name>/manifest.yaml \\
        --command "$(cat runs/<run_name>/command.txt)"

Exit codes
----------
    0  every check passed; manifest written if --out was given
    1  at least one check failed; see the [FAIL] lines and the manifest's
       preflight_failures list
    2  the script could not run at all (manifest missing/unparseable)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Minimum Python this project is known to run on. The local dev environment
# and every recorded run to date are 3.12.x; 3.10 is the floor because the
# codebase uses PEP 604 unions and match-free but 3.10+ typing idioms.
MIN_PYTHON = (3, 10)

REPO_ROOT_MARKERS = ("openvocab_rel", "tools", "tests", ".git")
MANIFEST_PATH = Path("data/manifests/historical_checkpoint_v1.yaml")

# Commits that MUST be in HEAD's ancestry. Each is a validity fix whose
# absence would silently corrupt this specific experiment.
REQUIRED_FIXES = {
    "220c5c2e": "GT-pair extraction index alignment fix",
    "9dc8f45d": "canonical predicate vocabulary enforcement",
    "7d91af49": "opt-in source-vocab-mismatch override",
    "fa8c0c3b": "predicate metadata 'wrapped around' entry",
    "65686b5f": "frequency-prior eval-split fallback fails loudly",
}

# How many probability rows to spot-check in the frequency prior. The file is
# ~97 MiB with 74,884 pair entries; checking every row costs seconds for no
# extra signal, since a truncated/corrupt file fails within the first few.
FREQ_PRIOR_SAMPLE_ROWS = 2000


class Preflight:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.manifest: Dict[str, Any] = {}

    # -- reporting -----------------------------------------------------
    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"[FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[WARN] {msg}")

    def ok(self, msg: str) -> None:
        print(f"[ OK ] {msg}")

    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def section(self, title: str) -> None:
        print(f"\n--- {title} " + "-" * max(0, 66 - len(title)))


def sha256_of(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


# ======================================================================
# checks
# ======================================================================

def check_repo_root(pf: Preflight) -> None:
    pf.section("repository root")
    cwd = Path.cwd()
    pf.manifest["cwd"] = str(cwd)
    missing = [m for m in REPO_ROOT_MARKERS if not (cwd / m).exists()]
    if missing:
        pf.fail(
            f"CWD does not look like the repository root: {cwd} (missing {missing}). "
            "Run this from the repository root -- every path in the manifest is "
            "relative to it."
        )
    else:
        pf.ok(f"CWD is repository root: {cwd}")


def load_manifest(pf: Preflight) -> Dict[str, Any]:
    pf.section("artifact manifest")
    if not MANIFEST_PATH.exists():
        pf.fail(f"artifact manifest missing: {MANIFEST_PATH}")
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        pf.fail(
            "PyYAML is not installed, so the artifact manifest cannot be read. "
            "Install it (`pip install pyyaml`) -- it is in requirements.txt."
        )
        return {}
    try:
        data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        pf.fail(f"could not parse {MANIFEST_PATH}: {exc}")
        return {}
    if not isinstance(data, dict) or "artifacts" not in data:
        pf.fail(f"{MANIFEST_PATH} has no 'artifacts' section")
        return {}
    pf.manifest["manifest_id"] = data.get("manifest_id", "")
    pf.manifest["manifest_sha256"] = sha256_of(MANIFEST_PATH)
    pf.ok(f"loaded manifest {data.get('manifest_id')} from {MANIFEST_PATH}")
    return data


def check_git_state(pf: Preflight, strict: bool) -> None:
    pf.section("git state")
    status = _git(["status", "--porcelain=v1"])
    if status is None:
        pf.fail("git status failed -- is this a git repository?")
        return
    dirty = bool(status.strip())
    pf.manifest["git_dirty"] = dirty
    if dirty:
        detail = "\n".join(f"        {line}" for line in status.strip().splitlines())
        msg = f"working tree is DIRTY -- the run would not be identifiable:\n{detail}"
        if strict:
            pf.fail(msg)
        else:
            pf.warn(msg + "\n        (not fatal without --strict; it IS fatal with it)")
    else:
        pf.ok("working tree is clean")

    commit = _git(["rev-parse", "HEAD"])
    if commit is None:
        pf.fail("git rev-parse HEAD failed")
        return
    pf.manifest["git_commit"] = commit
    pf.manifest["git_commit_short"] = commit[:8]
    pf.manifest["git_branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or ""
    pf.manifest["git_describe"] = _git(["log", "-1", "--format=%s"]) or ""
    pf.ok(f"HEAD = {commit}")
    pf.info(f"branch = {pf.manifest['git_branch']} | subject = {pf.manifest['git_describe']}")

    for short_sha, desc in sorted(REQUIRED_FIXES.items()):
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", short_sha, "HEAD"],
                check=True, capture_output=True,
            )
            pf.ok(f"required fix in ancestry: {short_sha} ({desc})")
        except subprocess.CalledProcessError:
            pf.fail(
                f"required fix NOT in HEAD's ancestry: {short_sha} ({desc}) -- "
                "this checkout predates a validity fix and would produce invalid results"
            )
        except Exception as exc:
            pf.fail(f"could not check ancestry of {short_sha}: {exc}")


def check_artifact(pf: Preflight, key: str, spec: Dict[str, Any]) -> Optional[Path]:
    """Verify one artifact's existence, size and SHA256 against the manifest."""
    path = Path(str(spec.get("path", "")))
    required = bool(spec.get("required", False))
    entry: Dict[str, Any] = {"path": str(path), "required": required}
    pf.manifest.setdefault("artifacts", {})[key] = entry

    if not path.exists():
        entry["present"] = False
        msg = f"{key} missing: {path}"
        if required:
            pf.fail(
                msg + " -- this artifact is gitignored and must be transferred "
                "out of band; `git clone` does not provide it"
            )
        else:
            pf.warn(msg + " (optional)")
        return None

    entry["present"] = True
    actual_size = path.stat().st_size
    entry["size_bytes"] = actual_size

    expected_size = spec.get("size_bytes")
    if expected_size is not None and int(expected_size) != actual_size:
        pf.fail(
            f"{key} SIZE mismatch: expected {int(expected_size):,} bytes, "
            f"got {actual_size:,} -- likely a truncated or partial transfer"
        )

    expected_sha = str(spec.get("sha256", "")).strip()
    actual_sha = sha256_of(path)
    entry["sha256"] = actual_sha
    if not expected_sha:
        pf.warn(f"{key}: manifest records no sha256; recorded actual {actual_sha}")
    elif actual_sha != expected_sha:
        pf.fail(
            f"{key} SHA256 MISMATCH\n"
            f"        expected: {expected_sha}\n"
            f"        actual:   {actual_sha}\n"
            f"        path:     {path}\n"
            "        This is not the artifact the experiment was defined against."
        )
    else:
        pf.ok(f"{key} sha256 verified ({actual_size:,} bytes)")
    return path


def _canonical_predicates(pf: Preflight) -> Optional[List[str]]:
    try:
        sys.path.insert(0, str(Path.cwd()))
        from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES  # type: ignore
    except Exception as exc:
        pf.fail(f"could not import STANDARD_VG150_PREDICATES: {exc}")
        return None
    return sorted(STANDARD_VG150_PREDICATES)


def check_predicate_vocabulary(pf: Preflight, path: Optional[Path], spec: Dict[str, Any]) -> Optional[List[str]]:
    pf.section("predicate vocabulary")
    if path is None:
        pf.fail("predicate vocabulary unavailable -- cannot verify canonical ordering")
        return None
    expected = _canonical_predicates(pf)
    if expected is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        idx_to_pred = data["idx_to_predicate"]
    except Exception as exc:
        pf.fail(f"could not parse {path}: {exc}")
        return None

    expected_n = int(spec.get("num_predicates", 50))
    actual_n = len(idx_to_pred)
    pf.manifest["predicate_vocab_len"] = actual_n
    if actual_n != expected_n:
        pf.fail(
            f"predicate vocabulary has {actual_n} entries, expected {expected_n} -- "
            "index mapping would not match the checkpoint or the frequency prior"
        )
        return None
    pf.ok(f"predicate vocabulary length == {expected_n}")

    try:
        order = [idx_to_pred[str(i)] for i in range(1, expected_n + 1)]
    except KeyError as exc:
        pf.fail(f"predicate vocabulary is not 1..{expected_n}-indexed: missing key {exc}")
        return None

    if order != expected:
        diffs = [
            f"idx {i + 1}: on-disk {a!r} != canonical {b!r}"
            for i, (a, b) in enumerate(zip(order, expected)) if a != b
        ]
        pf.fail(
            f"predicate vocabulary is NOT sorted(STANDARD_VG150_PREDICATES) -- "
            f"{len(diffs)} of {expected_n} indices differ. First three:\n"
            + "\n".join(f"        {d}" for d in diffs[:3])
        )
        return None
    pf.ok("predicate vocabulary == sorted(STANDARD_VG150_PREDICATES)")
    return order


def check_frequency_prior_structure(
    pf: Preflight, path: Optional[Path], spec: Dict[str, Any], canonical: Optional[List[str]]
) -> None:
    """Independently check every condition _load_frequency_bias tolerates silently.

    ``openvocab_rel/evals.py:_load_frequency_bias`` returns ``None`` -- i.e.
    applies no calibration at all -- on each condition below, without logging
    anything. A run configured with ``--freq_bias_enabled true --freq_bias_alpha
    3.75`` against a bad prior therefore produces a complete, plausible,
    silently-uncalibrated result. Since the uncalibrated text path is already
    known to score mR@50 ~5.84% against a historical claim of 22.64%, that
    failure would masquerade as a clean "failed to reproduce".
    """
    pf.section("frequency prior structure")
    if path is None:
        pf.fail("frequency prior unavailable -- calibration would silently not apply")
        return

    structure = spec.get("structure", {}) or {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        pf.fail(
            f"frequency prior does not parse as JSON: {exc} -- "
            "_load_frequency_bias would swallow this and run UNCALIBRATED"
        )
        return
    pf.ok("frequency prior parses as JSON")

    source_vocab = raw.get("predicate_vocab", [])
    n_src = len(source_vocab)
    pf.manifest["freq_prior_vocab_len"] = n_src
    expected_len = int(structure.get("predicate_vocab_len", 50))
    if n_src == 0:
        pf.fail("frequency prior 'predicate_vocab' is empty -- loader returns None silently")
        return
    if n_src != expected_len:
        pf.fail(f"frequency prior 'predicate_vocab' has {n_src} entries, expected {expected_len}")
    else:
        pf.ok(f"frequency prior 'predicate_vocab' length == {expected_len}")

    if canonical is not None:
        norm_src = [str(x).strip().lower() for x in source_vocab]
        norm_can = [str(x).strip().lower() for x in canonical]
        if norm_src != norm_can:
            missing = [p for p in norm_can if p not in set(norm_src)]
            pf.fail(
                "frequency prior 'predicate_vocab' != canonical vocabulary. "
                f"{len(missing)} canonical predicate(s) absent from the prior "
                f"(first three: {missing[:3]}). Those would be silently filled with "
                "default_log_prob instead of a real learned prior."
            )
        else:
            pf.ok("frequency prior 'predicate_vocab' == canonical vocabulary")

    glp = raw.get("global_log_probs", [])
    if not isinstance(glp, list) or len(glp) != n_src:
        pf.fail(
            f"'global_log_probs' has length {len(glp) if isinstance(glp, list) else 'n/a'}, "
            f"expected {n_src} -- loader's remap() returns None, so the loader returns "
            "None and the run is silently UNCALIBRATED"
        )
    else:
        pf.ok(f"'global_log_probs' length == {n_src}")

    for field in ("pair_log_probs", "subject_log_probs", "object_log_probs"):
        table = raw.get(field, {})
        if not isinstance(table, dict):
            pf.fail(f"'{field}' is not an object/dict")
            continue
        expected_count = structure.get(f"{field}_count")
        actual_count = len(table)
        pf.manifest[f"freq_prior_{field}_count"] = actual_count
        if expected_count is not None and int(expected_count) != actual_count:
            pf.fail(
                f"'{field}' has {actual_count:,} entries, manifest expects "
                f"{int(expected_count):,} -- this is not the recovered prior"
            )
        bad = 0
        for i, (_k, v) in enumerate(table.items()):
            if i >= FREQ_PRIOR_SAMPLE_ROWS:
                break
            if not isinstance(v, list) or len(v) != n_src:
                bad += 1
        checked = min(actual_count, FREQ_PRIOR_SAMPLE_ROWS)
        if bad:
            pf.fail(
                f"'{field}': {bad}/{checked} sampled rows are not length-{n_src} lists -- "
                "such rows are silently DROPPED by the loader"
            )
        else:
            pf.ok(f"'{field}': {actual_count:,} entries, {checked:,} sampled rows all length-{n_src}")

    for scalar in ("smoothing", "default_log_prob"):
        if scalar in structure:
            expected_v = structure[scalar]
            actual_v = raw.get(scalar)
            if actual_v is None:
                pf.fail(f"frequency prior missing '{scalar}'")
            elif abs(float(actual_v) - float(expected_v)) > 1e-12:
                pf.fail(f"frequency prior '{scalar}' = {actual_v}, expected {expected_v}")
            else:
                pf.ok(f"frequency prior '{scalar}' == {expected_v}")


def check_dataset_split(pf: Preflight, key: str, path: Optional[Path], spec: Dict[str, Any]) -> None:
    expected_rows = spec.get("rows")
    if path is None or expected_rows is None:
        return
    n = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except Exception as exc:
        pf.fail(f"{key} could not be read: {exc}")
        return
    pf.manifest.setdefault("dataset_rows", {})[key] = n
    if n != int(expected_rows):
        pf.fail(
            f"{key} row count {n:,} != expected {int(expected_rows):,} -- "
            "this is a different dataset build than the experiment was defined against"
        )
    else:
        pf.ok(f"{key} row count == {n:,}")


def _count_files(directory: Path) -> int:
    n = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    n += 1
    except Exception:
        return -1
    return n


def check_images(pf: Preflight, vg150_root: Path, spec: Dict[str, Any]) -> None:
    """Verify the image tree actually materialized.

    Two ways this silently goes wrong, both yielding a run that completes and
    reports numbers rather than crashing -- because
    ``VG150JSONLDataset._resolve_image`` falls back to a solid gray placeholder
    for any path it cannot open:

    1. On the source machine ``images/`` is an NTFS **junction**, not a copy.
       An archive that did not follow it lands an empty directory here.
    2. Visual Genome's standard layout nests images under ``VG_100K/`` and
       ``VG_100K_2/``. A flat scan sees 2 entries -- which is why this check
       descends one level rather than counting top-level entries.
    """
    pf.section("image tree")
    images = vg150_root / "images"
    if not images.exists():
        pf.fail(
            f"image directory missing: {images} -- every image would silently fall back "
            "to a gray placeholder in VG150JSONLDataset._resolve_image, producing "
            "meaningless but non-crashing results"
        )
        pf.manifest["image_files_found"] = 0
        return

    expected_subdirs = (spec.get("subdirs") or {}) if isinstance(spec, dict) else {}
    total = max(0, _count_files(images))
    per_subdir: Dict[str, int] = {}
    try:
        with os.scandir(images) as it:
            subdirs = sorted(e.name for e in it if e.is_dir())
    except Exception as exc:
        pf.fail(f"could not scan {images}: {exc}")
        return
    for name in subdirs:
        c = _count_files(images / name)
        per_subdir[name] = c
        if c > 0:
            total += c

    pf.manifest["image_files_found"] = total
    pf.manifest["image_subdirs"] = per_subdir

    if total == 0:
        pf.fail(
            f"image tree {images} contains NO image files (subdirectories seen: "
            f"{subdirs or 'none'}). On the source machine this path is an NTFS junction; "
            "an archive that did not follow it produces an empty tree here, every image "
            "silently degrades to a gray placeholder, and the run completes reporting "
            "meaningless numbers"
        )
        return

    for name, expected_n in expected_subdirs.items():
        actual_n = per_subdir.get(name)
        if actual_n is None:
            pf.fail(f"expected image subdirectory missing: {images / name}")
        elif int(expected_n) != actual_n:
            pf.fail(
                f"image subdirectory {name} has {actual_n:,} files, manifest expects "
                f"{int(expected_n):,} -- incomplete transfer"
            )
        else:
            pf.ok(f"image subdirectory {name}: {actual_n:,} files")

    expected_total = spec.get("total_files") if isinstance(spec, dict) else None
    if expected_total is not None and int(expected_total) != total:
        pf.fail(f"image tree has {total:,} files, manifest expects {int(expected_total):,}")
    else:
        pf.ok(f"image tree: {total:,} files total")


def check_environment(pf: Preflight, strict: bool) -> None:
    pf.section("environment")
    pf.manifest["preflight_timestamp_utc"] = (
        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    )
    pf.manifest["python_version"] = platform.python_version()
    pf.manifest["python_executable"] = sys.executable
    pf.manifest["platform"] = platform.platform()
    pf.manifest["hostname"] = platform.node()
    pf.info(f"timestamp (UTC): {pf.manifest['preflight_timestamp_utc']}")
    pf.info(f"hostname: {pf.manifest['hostname']}")
    pf.info(f"platform: {pf.manifest['platform']}")

    vi = sys.version_info
    if (vi.major, vi.minor) < MIN_PYTHON:
        pf.fail(
            f"Python {platform.python_version()} is below this project's floor "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        )
    else:
        pf.ok(f"Python {platform.python_version()} >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")

    try:
        import torch  # type: ignore
        pf.manifest["torch_version"] = str(torch.__version__)
        cuda_ok = bool(torch.cuda.is_available())
        pf.manifest["cuda_available"] = cuda_ok
        pf.manifest["torch_cuda_build"] = str(torch.version.cuda)
        pf.ok(f"torch {torch.__version__} (built against CUDA {torch.version.cuda})")
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            pf.manifest["gpu_name"] = str(torch.cuda.get_device_name(0))
            pf.manifest["gpu_count"] = int(torch.cuda.device_count())
            pf.manifest["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
            pf.manifest["gpu_capability"] = f"{props.major}.{props.minor}"
            pf.ok(
                f"CUDA available: {pf.manifest['gpu_name']} "
                f"({pf.manifest['gpu_vram_gb']} GB, sm_{props.major}{props.minor}, "
                f"{pf.manifest['gpu_count']} device(s))"
            )
        else:
            msg = (
                "CUDA is NOT available. The full validation split costs ~104 h on CPU "
                "(~36 s/image, measured) versus hours on an L4."
            )
            if strict:
                pf.fail(msg + " Refusing to certify a GPU run without a GPU.")
            else:
                pf.warn(msg + " Expected for a LOCAL dry run; --strict would fail here.")
    except ImportError:
        pf.manifest["torch_version"] = None
        pf.manifest["cuda_available"] = None
        msg = "torch is not importable -- the evaluation cannot run in this environment"
        if strict:
            pf.fail(msg)
        else:
            pf.warn(msg + " (tolerated without --strict)")

    try:
        import transformers  # type: ignore
        pf.manifest["transformers_version"] = str(transformers.__version__)
        pf.ok(f"transformers {transformers.__version__}")
    except ImportError:
        pf.manifest["transformers_version"] = None
        msg = "transformers is not importable -- CLIP cannot be loaded"
        if strict:
            pf.fail(msg)
        else:
            pf.warn(msg + " (tolerated without --strict)")

    # Deliberately NOT pinning exact torch/transformers versions: the recorded
    # local snapshot is torch 2.8.0+cpu, but a GCP L4 image will legitimately
    # ship a different CUDA build. Over-constraining here would reject valid
    # environments. Versions are captured in the manifest for after-the-fact
    # identification instead.


def write_manifest(pf: Preflight, out: str) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(pf.manifest, f, sort_keys=False, default_flow_style=False)
    except ImportError:
        out_path = out_path.with_suffix(".json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(pf.manifest, f, indent=2)
    print(f"[INFO] manifest written to {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail-loud preflight for the historical-checkpoint reproduction run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default="", help="write the captured manifest here (YAML, JSON fallback)")
    ap.add_argument(
        "--strict", action="store_true",
        help="GCP mode: a dirty tree, absent CUDA, or missing torch/transformers become FAILURES. "
             "Always pass this on the actual GCP instance.",
    )
    ap.add_argument(
        "--command", default="",
        help="the exact evaluation command this preflight is gating; recorded verbatim in the manifest",
    )
    ap.add_argument(
        "--run-name", default="",
        help="run identifier to record in the manifest (for cross-referencing runs/<name>/)",
    )
    args = ap.parse_args(argv)

    pf = Preflight()
    print("=" * 74)
    print("  GCP historical-checkpoint reproduction preflight")
    print(f"  mode: {'STRICT (GCP)' if args.strict else 'permissive (local dry run)'}")
    print("=" * 74)

    pf.manifest["preflight_mode"] = "strict" if args.strict else "permissive"
    if args.run_name:
        pf.manifest["run_name"] = args.run_name
    if args.command:
        pf.manifest["eval_command"] = args.command

    check_repo_root(pf)
    spec = load_manifest(pf)
    if not spec:
        print("\nPREFLIGHT ABORTED: artifact manifest unusable.")
        return 2

    check_git_state(pf, args.strict)

    artifacts = spec.get("artifacts", {})
    pf.section("artifact hashes")
    paths: Dict[str, Optional[Path]] = {}
    for key in (
        "checkpoint", "frequency_prior", "demo_config",
        "dataset_train", "dataset_validation",
        "predicate_vocabulary", "object_vocabulary",
    ):
        if key in artifacts:
            paths[key] = check_artifact(pf, key, artifacts[key])

    pf.section("dataset row counts")
    check_dataset_split(pf, "dataset_train", paths.get("dataset_train"), artifacts.get("dataset_train", {}))
    check_dataset_split(pf, "dataset_validation", paths.get("dataset_validation"), artifacts.get("dataset_validation", {}))

    canonical = check_predicate_vocabulary(
        pf, paths.get("predicate_vocabulary"), artifacts.get("predicate_vocabulary", {})
    )
    check_frequency_prior_structure(
        pf, paths.get("frequency_prior"), artifacts.get("frequency_prior", {}), canonical
    )

    vocab_path = paths.get("predicate_vocabulary")
    vg150_root = vocab_path.parent.parent if vocab_path is not None else Path("datasets_vg150_clean")
    check_images(pf, vg150_root, artifacts.get("images", {}))

    check_environment(pf, args.strict)

    print("\n" + "=" * 74)
    if pf.failures:
        print(f"  PREFLIGHT FAILED -- {len(pf.failures)} check(s) failed")
        print("=" * 74)
        for f in pf.failures:
            print(f"  [FAIL] {f.splitlines()[0]}")
        print("\n  DO NOT START THE EVALUATION. Fix every item above first.")
        pf.manifest["preflight_status"] = "FAILED"
    else:
        print("  PREFLIGHT PASSED -- all checks green")
        print("=" * 74)
        if pf.warnings:
            print(f"  ({len(pf.warnings)} warning(s); re-run with --strict on the GCP instance)")
        else:
            print("  Next step: run the canary, then verify it before the full run.")
        pf.manifest["preflight_status"] = "PASSED"
    print("=" * 74)

    pf.manifest["preflight_failures"] = pf.failures
    pf.manifest["preflight_warnings"] = pf.warnings

    if args.out:
        write_manifest(pf, args.out)

    return 1 if pf.failures else 0


if __name__ == "__main__":
    sys.exit(main())
