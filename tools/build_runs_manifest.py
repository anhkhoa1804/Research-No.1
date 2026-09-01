#!/usr/bin/env python
"""Manifest of runs/ -- the bridge between ignored artifacts and git.

runs/ is gitignored (.gitignore:24) because it holds multi-hundred-megabyte
caches. That is the right call for the binaries and the wrong one for the
evidence: without a manifest, a result quoted in a report cannot be tied to the
artifact it came from once the VM is gone.

This records, per run: path, size, sha256 of the small artifacts, the experiment
name, the commit it ran at, whether the tree was clean, the exact command, and
the run's status. Large binaries are identified by (size, mtime) unless
--hash-large is passed, matching tools/run_experiment.py's convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

RUNS = Path("runs")
LARGE = 64 << 20
SMALL_SUFFIXES = {".json", ".txt", ".log", ".md", ".jsonl"}


def sha256_of(p: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def describe(f: Path, hash_large: bool) -> Dict[str, Any]:
    st = f.stat()
    big = st.st_size > LARGE
    return {"path": str(f), "size_bytes": st.st_size,
            "sha256": (sha256_of(f) if (not big or hash_large) else None),
            "note": None if (not big or hash_large) else "large: size+mtime identity",
            "mtime": st.st_mtime if big and not hash_large else None}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/RUNS_MANIFEST.json")
    ap.add_argument("--hash-large", action="store_true")
    args = ap.parse_args(argv)

    runs: List[Dict[str, Any]] = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        entry: Dict[str, Any] = {"name": d.name, "path": str(d)}
        cmd = d / "command.txt"
        entry["command"] = cmd.read_text(encoding="utf-8").strip() if cmd.exists() else None
        for meta, key in ((d / "provenance.json", "provenance"),
                          (d / "result.json", "result")):
            if meta.exists():
                try:
                    j = json.loads(meta.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                if key == "provenance":
                    g = j.get("git", {})
                    entry.update({"commit": g.get("commit"),
                                  "branch": g.get("branch"),
                                  "working_tree_clean": g.get("working_tree_clean"),
                                  "note": j.get("note")})
                else:
                    entry.update({"exit_code": j.get("exit_code"),
                                  "runtime_seconds": j.get("runtime_seconds"),
                                  "started_at_utc": j.get("started_at_utc")})
        files = [f for f in sorted(d.rglob("*")) if f.is_file()]
        entry["files"] = [describe(f, args.hash_large) for f in files]
        entry["total_bytes"] = sum(x["size_bytes"] for x in entry["files"])
        entry["status"] = ("OK" if entry.get("exit_code") == 0 else
                           "FAILED" if entry.get("exit_code") is not None else
                           "NO_RESULT_JSON")
        runs.append(entry)

    out = {"tool": "build_runs_manifest",
           "why": "runs/ is gitignored; this ties every reported number to an artifact",
           "n_runs": len(runs),
           "total_bytes": sum(r["total_bytes"] for r in runs),
           "runs": runs}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[written] {args.out}: {len(runs)} runs, "
          f"{out['total_bytes']/(1<<30):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
