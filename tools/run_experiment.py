#!/usr/bin/env python
"""Run one experiment under a canonical, self-describing run directory.

Every GPU-hour spent on a result that cannot later be attributed to an exact
commit, configuration and dataset is a wasted GPU-hour. This wrapper makes the
provenance capture automatic rather than a thing to remember:

    runs/<name>/
      command.txt        the exact argv, re-runnable verbatim
      provenance.json    git sha, dirty state, env, GPU, dataset+prior hashes
      stdout.log         full stdout
      stderr.log         full stderr
      result.json        exit code, wall-clock runtime, start/end timestamps

It deliberately does NOT interpret the wrapped command's output. Whatever
metrics the command writes, it writes itself; this tool only guarantees that
the surrounding facts are recorded.

Usage:
    python tools/run_experiment.py --name my_run -- python tools/foo.py --flag
    python tools/run_experiment.py --name my_run --note "why" -- <cmd...>
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNS_ROOT = Path("runs")

# Artifacts whose identity determines whether two runs are comparable.
IDENTITY_ARTIFACTS = {
    "dataset_train": "datasets_vg150_clean/train.jsonl",
    "dataset_validation": "datasets_vg150_clean/validation.jsonl",
    "dataset_test": "datasets_vg150_clean/test.jsonl",
    "predicate_vocabulary": "datasets_vg150_clean/vocabulary/predicates.json",
    "historical_frequency_prior": "checkpoints/demo_best/frequency_prior.json",
    "train_derived_frequency_prior": "datasets_vg150_clean/frequency_prior_train.json",
    "checkpoint": "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
}

# Hashing a ~931 MiB checkpoint on every run costs ~2 s of pure I/O and tells
# us nothing new once it is known-good, so large artifacts are identified by
# (size, mtime) unless --hash-large is passed.
LARGE_ARTIFACT_BYTES = 64 << 20


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


def collect_provenance(note: str, hash_large: bool) -> Dict[str, Any]:
    status = _git(["status", "--porcelain=v1"])
    prov: Dict[str, Any] = {
        "note": note,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "subject": _git(["log", "-1", "--format=%s"]),
            "working_tree_clean": status == "",
            "uncommitted_entries": status.splitlines() if status else [],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cwd": str(Path.cwd()),
        },
    }
    try:
        import torch  # type: ignore
        prov["environment"]["torch"] = str(torch.__version__)
        prov["environment"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            prov["environment"]["gpu_name"] = torch.cuda.get_device_name(0)
            prov["environment"]["gpu_total_memory_bytes"] = int(props.total_memory)
            prov["environment"]["gpu_capability"] = f"sm_{props.major}{props.minor}"
    except Exception as exc:
        prov["environment"]["torch_error"] = str(exc)
    for mod in ("transformers", "numpy"):
        try:
            prov["environment"][mod] = str(__import__(mod).__version__)
        except Exception:
            prov["environment"][mod] = None

    artifacts: Dict[str, Any] = {}
    for key, rel in IDENTITY_ARTIFACTS.items():
        path = Path(rel)
        if not path.exists():
            artifacts[key] = {"path": rel, "present": False}
            continue
        size = path.stat().st_size
        entry: Dict[str, Any] = {"path": rel, "present": True, "size_bytes": size}
        if size <= LARGE_ARTIFACT_BYTES or hash_large:
            entry["sha256"] = sha256_of(path)
        else:
            entry["sha256"] = None
            entry["mtime"] = path.stat().st_mtime
            entry["note"] = "large artifact: identified by size+mtime; --hash-large to hash it"
        artifacts[key] = entry
    prov["artifacts"] = artifacts
    return prov


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run directory name under runs/")
    ap.add_argument("--note", default="", help="one line on what this run is testing")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--hash-large", action="store_true", help="sha256 even multi-hundred-MiB artifacts")
    ap.add_argument("--allow-existing", action="store_true",
                    help="reuse the run directory instead of refusing to overwrite it")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="the command to run, after a literal --")
    args = ap.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("no command given; pass it after a literal --")

    run_dir = Path(args.runs_root) / args.name
    if run_dir.exists() and not args.allow_existing:
        # Silently overwriting a previous result is how provenance is lost.
        print(f"[run_experiment] REFUSING: {run_dir} already exists. "
              f"Choose another --name or pass --allow-existing.", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    provenance = collect_provenance(args.note, args.hash_large)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    if not provenance["git"]["working_tree_clean"]:
        print(f"[run_experiment] WARNING: working tree is dirty; this run is not "
              f"exactly identifiable by commit alone. Uncommitted: "
              f"{provenance['git']['uncommitted_entries']}", file=sys.stderr)

    print(f"[run_experiment] {args.name}: {' '.join(command)}")
    start = time.time()
    start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with (run_dir / "stdout.log").open("w", encoding="utf-8") as out_f, \
         (run_dir / "stderr.log").open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(command, stdout=out_f, stderr=err_f, env=dict(os.environ))

    runtime = time.time() - start
    result = {
        "name": args.name,
        "command": command,
        "exit_code": proc.returncode,
        "runtime_seconds": round(runtime, 3),
        "started_at_utc": start_iso,
        "ended_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[run_experiment] {args.name}: {status} in {runtime:.1f}s -> {run_dir}")
    if proc.returncode != 0:
        tail = (run_dir / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("[run_experiment] stderr tail:", file=sys.stderr)
        for line in tail:
            print("   " + line, file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
