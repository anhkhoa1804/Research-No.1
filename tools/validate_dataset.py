#!/usr/bin/env python3
"""Pre-flight dataset readiness check, keyed off data/manifests/<dataset>.yaml.

This is intentionally NOT the same tool as tools/check_vg150_diagnostics.py:
that one validates the *output* of tools/prepare_vg150_drive_clean.py (row
counts, predicate coverage, alias-mapping stats, written after conversion).
This one validates *input readiness* before you launch a training/eval run
-- it only checks that the expected files exist, parse, and roughly match
the manifest's expected vocabulary, so a misconfigured --vg150_root fails
loudly here instead of silently degrading a run an hour in (see the
conditional eval-time leak risk documented in docs/known_issues.md, which is
exactly the kind of silent-fallback failure mode this tool exists to catch
earlier and louder).

Usage:
    python3 tools/validate_dataset.py --dataset vg150 [--vg150_root PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover
    print(
        "[validate_dataset] pyyaml is required (pip install -r requirements.txt).",
        file=sys.stderr,
    )
    raise


def _load_manifest(dataset: str, manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise SystemExit(f"[validate_dataset] no manifest for dataset={dataset!r} at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise SystemExit(f"[validate_dataset] manifest at {manifest_path} did not parse to a mapping")
    return manifest


def _resolve_root(args: argparse.Namespace, manifest: Dict[str, Any]) -> Path:
    if args.vg150_root:
        return Path(args.vg150_root)
    env_var = str(manifest.get("root_env_var", "DATA_ROOT"))
    env_val = os.environ.get(env_var, "")
    if env_val:
        return Path(env_val)
    return Path(str(manifest.get("root_default", "datasets_vg150_clean")))


def _layout_satisfied(root: Path, layout: Dict[str, Any]) -> List[str]:
    """Return a list of missing required files/dirs for this layout (empty = satisfied)."""
    missing: List[str] = []
    for rel in layout.get("required_files", []):
        candidate = root / str(rel).rstrip("/")
        if not candidate.exists():
            missing.append(str(rel))
    return missing


def _sample_jsonl_rows(path: Path, max_rows: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[validate_dataset] malformed JSONL at {path}:{line_num + 1}: {exc}"
                )
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _check_row_fields(rows: List[Dict[str, Any]], required_fields: List[str], source: str) -> List[str]:
    problems: List[str] = []
    field_aliases = {
        "objects": ("objects", "obj_labels"),
        "relationships": ("relationships", "relations", "rels"),
    }
    for field in required_fields:
        aliases = field_aliases.get(field, (field,))
        n_present = sum(1 for row in rows if any(a in row for a in aliases))
        if rows and n_present == 0:
            problems.append(f"{source}: none of the first {len(rows)} rows contain any of {aliases}")
    return problems


def _collect_predicate_vocab(rows: List[Dict[str, Any]]) -> set:
    vocab = set()
    for row in rows:
        rels = row.get("relationships", row.get("relations", row.get("rels", [])))
        if not isinstance(rels, list):
            continue
        for rel in rels:
            if isinstance(rel, dict):
                pred = str(rel.get("predicate", rel.get("pred", ""))).strip().lower()
                if pred:
                    vocab.add(pred)
    return vocab


def validate_vg150(args: argparse.Namespace) -> None:
    manifest = _load_manifest("vg150", Path(args.manifest))
    root = _resolve_root(args, manifest)
    print(f"[validate_dataset] dataset=vg150 root={root}")

    if not root.exists():
        raise SystemExit(
            f"[validate_dataset] FAIL: root does not exist: {root}\n"
            f"  Set --vg150_root, or the {manifest.get('root_env_var', 'DATA_ROOT')} env var, "
            f"to a directory prepared per data/README.md."
        )

    layouts = manifest.get("layouts", {})
    layout_results = {name: _layout_satisfied(root, layout) for name, layout in layouts.items()}
    satisfied = [name for name, missing in layout_results.items() if not missing]

    if not satisfied:
        lines = [f"[validate_dataset] FAIL: no known VG150 layout is satisfied under {root}:"]
        for name, missing in layout_results.items():
            lines.append(f"  layout={name} missing={missing}")
        lines.append("  See data/README.md for the two accepted layouts and how to produce one.")
        raise SystemExit("\n".join(lines))

    print(f"[validate_dataset] layout OK: {satisfied}")

    failures: List[str] = []

    if "local-jsonl" in satisfied:
        train_path = root / "train.jsonl"
        val_path = root / "validation.jsonl"
        if not val_path.exists():
            val_path = root / "val.jsonl"
        train_rows = _sample_jsonl_rows(train_path)
        val_rows = _sample_jsonl_rows(val_path)
        if not train_rows:
            failures.append(f"train.jsonl at {train_path} produced zero parsed rows")
        if not val_rows:
            failures.append(f"validation split at {val_path} produced zero parsed rows")

        required_row_fields = layouts.get("local-jsonl", {}).get("required_row_fields", [])
        failures.extend(_check_row_fields(train_rows, required_row_fields, "train.jsonl"))
        failures.extend(_check_row_fields(val_rows, required_row_fields, "validation split"))

        observed_predicates = _collect_predicate_vocab(train_rows)
        expected_num_predicates = int(manifest.get("expected_vocab", {}).get("num_predicates", 50))
        # Sampling max_rows rows will not see the full 50-class vocab reliably on a small
        # sample; this is a coarse sanity floor (at least a few distinct predicates seen),
        # not a strict count match -- a strict check belongs to
        # tools/check_vg150_diagnostics.py, which runs over the full converted dataset.
        if train_rows and len(observed_predicates) < min(5, expected_num_predicates):
            failures.append(
                f"only {len(observed_predicates)} distinct predicate strings observed in the "
                f"first {len(train_rows)} train rows (manifest expects {expected_num_predicates} "
                f"total) -- this is a coarse floor, not proof of full coverage; run "
                f"tools/check_vg150_diagnostics.py for a real coverage check"
            )

    if failures:
        lines = ["[validate_dataset] FAIL:"]
        lines.extend(f"  - {msg}" for msg in failures)
        raise SystemExit("\n".join(lines))

    print("[validate_dataset] PASS: VG150 layout and JSONL structure look ready.")
    print(
        "[validate_dataset] note: this is a pre-flight structural check, not a full "
        "coverage/diagnostics pass -- also run tools/check_vg150_diagnostics.py "
        "against your prepared dataset's diagnostics.json before a long training run."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["vg150"], help="dataset name (only vg150 today)")
    parser.add_argument("--vg150_root", default="", help="override DATA_ROOT / manifest default")
    parser.add_argument(
        "--manifest",
        default="data/manifests/vg150.yaml",
        help="path to the dataset manifest (default: data/manifests/vg150.yaml)",
    )
    args = parser.parse_args()

    if args.dataset == "vg150":
        validate_vg150(args)
    else:  # pragma: no cover -- unreachable given argparse choices=["vg150"]
        raise SystemExit(f"[validate_dataset] unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
