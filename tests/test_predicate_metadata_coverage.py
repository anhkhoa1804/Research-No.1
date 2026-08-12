"""configs/predicate_metadata_vg150.json must cover the canonical VG150 vocabulary.

The metadata file supplies each predicate's coarse `group` (used by the
gt_by_group / pred_by_group diagnostics) and `symmetric` flag (used by
eval_sgg_standard's role-swap diagnostic to skip order-invariant predicates).

A canonical predicate missing from this file does not crash -- load_predicate_metadata
pre-populates defaults for every predicate in the passed vocabulary before overlaying
the file -- it silently falls back to group="other", which is a less-curated
classification than the file exists to provide. "wrapped around" was missing for
exactly this reason and was only noticed during a manual audit, hence this test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES

METADATA_PATH = Path(__file__).resolve().parents[1] / "configs" / "predicate_metadata_vg150.json"

VALID_KEYS = {"group", "symmetric"}


def _load_predicates() -> dict:
    with METADATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["predicates"]


def test_every_canonical_predicate_has_metadata():
    """No canonical VG150 predicate may silently fall back to the default group."""
    preds = _load_predicates()
    missing = STANDARD_VG150_PREDICATES - set(preds)
    assert not missing, (
        f"canonical predicates absent from predicate_metadata_vg150.json: {sorted(missing)} "
        f"-- these silently degrade to group='other'"
    )


def test_metadata_entries_are_well_formed():
    preds = _load_predicates()
    for name, entry in preds.items():
        assert isinstance(entry, dict), f"{name!r} metadata is not an object"
        assert set(entry) <= VALID_KEYS, f"{name!r} has unexpected keys: {set(entry) - VALID_KEYS}"
        assert isinstance(entry.get("group", ""), str), f"{name!r} group is not a string"
        assert isinstance(entry.get("symmetric", False), bool), f"{name!r} symmetric is not a bool"


def test_extra_non_canonical_entries_are_inert_and_documented():
    """The file carries a few non-canonical keys ("around", "growing on", "says").

    They are harmless: load_predicate_metadata only ever queries predicates that are
    in the active vocabulary, so entries outside it are never read. They are retained
    rather than deleted because "around" documents an alias relationship
    (VG150_PREDICATE_ALIASES maps "around" -> "wrapped around"). This test pins the
    known set so a *new* unexplained entry gets noticed instead of accumulating.
    """
    preds = _load_predicates()
    extra = set(preds) - STANDARD_VG150_PREDICATES
    assert extra == {"around", "growing on", "says"}, (
        f"unexpected non-canonical predicate metadata entries: "
        f"{sorted(extra - {'around', 'growing on', 'says'})}"
    )
