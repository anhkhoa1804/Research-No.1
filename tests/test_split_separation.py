"""The split invariant every leak-free claim in this project rests on.

`datasets_vg150_clean/frequency_prior_train.json` is called "leak-free" because
it is built from train.jsonl and evaluated on validation.jsonl. That word is
load-bearing -- it is the difference between the prior being a legitimate
baseline and being a lookup of the answers -- and until now nothing in the test
suite checked it. It was verified by hand on 2026-09-01 (83,249 / 10,401 /
10,403 images, zero pairwise overlap) and is pinned here so a future dataset
regeneration cannot quietly break it.

These read the real split files and skip if absent, so a checkout without the
dataset still passes.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {n: ROOT / f"datasets_vg150_clean/{n}.jsonl"
          for n in ("train", "validation", "test")}
pytestmark = pytest.mark.skipif(not all(p.exists() for p in SPLITS.values()),
                                reason="VG150 clean splits absent")


def _image_ids(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            key = d.get("image_id") or d.get("id") or d.get("img_id")
            assert key is not None, f"no image id field in {path.name}"
            out.add(str(key))
    return out


@pytest.fixture(scope="module")
def ids():
    return {n: _image_ids(p) for n, p in SPLITS.items()}


def test_no_image_appears_in_two_splits(ids):
    for a, b in itertools.combinations(sorted(ids), 2):
        overlap = ids[a] & ids[b]
        assert not overlap, (
            f"{len(overlap)} image(s) in both {a} and {b}, e.g. "
            f"{sorted(overlap)[:5]}. Every 'leak-free prior' claim is void.")


def test_split_sizes_match_the_recorded_membership(ids):
    """Sizes are pinned, so a silent regeneration that changes membership fails
    here rather than surfacing as an unexplained metric shift months later."""
    assert len(ids["train"]) == 83249
    assert len(ids["validation"]) == 10401
    assert len(ids["test"]) == 10403


def test_ids_are_unique_within_each_split(ids):
    for name, path in SPLITS.items():
        with path.open(encoding="utf-8") as f:
            rows = sum(1 for line in f if line.strip())
        assert rows == len(ids[name]), f"{name} has duplicate image ids"
