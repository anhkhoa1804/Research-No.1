"""Predicate vocabulary invariants for VG150.

Scientific invariant: the 50-predicate vocabulary used during dataset
preparation, model training, and evaluation must be exactly
STANDARD_VG150_PREDICATES in sorted order (1-indexed in the JSON file).

Historical evidence: checkpoints/demo_best/frequency_prior.json contains a
predicate_vocab field that is sorted(STANDARD_VG150_PREDICATES), establishing
the canonical ordering used by the original training run.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES
from tools.prepare_vg150_drive_clean import _canonical_predicate_vocab

CANONICAL_ORDERED = sorted(STANDARD_VG150_PREDICATES)

# Predicates confirmed present in the wrong vocabulary that was previously
# shipped in datasets_vg150_clean/vocabulary/predicates.json.
BAD_PREDICATES = {"growing on", "says"}
# Predicates that the bad vocabulary was missing.
MISSING_FROM_BAD = {"next to", "wrapped around"}


# ---------------------------------------------------------------------------
# A. Exactly 50 predicates
# ---------------------------------------------------------------------------

def test_A_standard_set_has_exactly_50_entries():
    assert len(STANDARD_VG150_PREDICATES) == 50, (
        f"STANDARD_VG150_PREDICATES has {len(STANDARD_VG150_PREDICATES)} entries, expected 50"
    )


def test_A_canonical_vocab_dict_has_exactly_50_entries():
    vocab = _canonical_predicate_vocab()
    assert len(vocab["idx_to_predicate"]) == 50


# ---------------------------------------------------------------------------
# B. Set equality with STANDARD_VG150_PREDICATES
# ---------------------------------------------------------------------------

def test_B_canonical_vocab_set_equals_standard():
    vocab = _canonical_predicate_vocab()
    generated = set(vocab["idx_to_predicate"].values())
    assert generated == STANDARD_VG150_PREDICATES, (
        f"Extra: {generated - STANDARD_VG150_PREDICATES}  "
        f"Missing: {STANDARD_VG150_PREDICATES - generated}"
    )


# ---------------------------------------------------------------------------
# C. Deterministic ordering
# ---------------------------------------------------------------------------

def test_C_canonical_vocab_is_sorted():
    vocab = _canonical_predicate_vocab()
    values = [vocab["idx_to_predicate"][str(i + 1)] for i in range(50)]
    assert values == CANONICAL_ORDERED, (
        "canonical_predicate_vocab() ordering differs from sorted(STANDARD_VG150_PREDICATES)"
    )


def test_C_ordering_is_stable_across_calls():
    v1 = _canonical_predicate_vocab()
    v2 = _canonical_predicate_vocab()
    assert v1 == v2


# ---------------------------------------------------------------------------
# D. No bad predicate can survive — bad vocab is rejected by _copy_vocab
# ---------------------------------------------------------------------------

def test_D_bad_source_predicates_json_raises():
    """A source predicates.json containing the previously-observed wrong predicates
    must be rejected with a loud error, not silently accepted."""
    from tools.prepare_vg150_drive_clean import _copy_vocab

    bad_vocab = {
        "idx_to_predicate": {
            str(i + 1): p
            for i, p in enumerate(
                sorted((STANDARD_VG150_PREDICATES - MISSING_FROM_BAD) | BAD_PREDICATES)
            )
        }
    }
    assert set(bad_vocab["idx_to_predicate"].values()) & BAD_PREDICATES  # confirm bad entries present
    assert not (MISSING_FROM_BAD <= set(bad_vocab["idx_to_predicate"].values()))  # confirm missing entries

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        dst = Path(tmp) / "dst"
        vocab_src = src / "vocabulary"
        vocab_src.mkdir(parents=True)
        with (vocab_src / "predicates.json").open("w") as fh:
            json.dump(bad_vocab, fh)

        with pytest.raises(SystemExit, match="incompatible with STANDARD_VG150_PREDICATES"):
            _copy_vocab(src, dst)


# ---------------------------------------------------------------------------
# E. Historical frequency_prior.json matches the canonical ordering
# ---------------------------------------------------------------------------

FREQ_PRIOR_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "demo_best" / "frequency_prior.json"

@pytest.mark.skipif(not FREQ_PRIOR_PATH.exists(), reason="historical checkpoint not present locally")
def test_E_frequency_prior_predicate_vocab_matches_canonical_order():
    with FREQ_PRIOR_PATH.open(encoding="utf-8") as fh:
        prior = json.load(fh)
    hist_vocab = prior.get("predicate_vocab", [])
    assert hist_vocab == CANONICAL_ORDERED, (
        f"frequency_prior.json predicate_vocab does not match sorted(STANDARD_VG150_PREDICATES).\n"
        f"First difference at index "
        f"{next(i for i,(a,b) in enumerate(zip(hist_vocab, CANONICAL_ORDERED)) if a!=b)}"
        if hist_vocab != CANONICAL_ORDERED else ""
    )


# ---------------------------------------------------------------------------
# F. Correct source vocab is accepted without error
# ---------------------------------------------------------------------------

def test_F_correct_source_predicates_json_is_accepted():
    """A source predicates.json with exactly STANDARD_VG150_PREDICATES (any order) is accepted."""
    from tools.prepare_vg150_drive_clean import _copy_vocab

    good_vocab = {"idx_to_predicate": {str(i + 1): p for i, p in enumerate(CANONICAL_ORDERED)}}

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        dst = Path(tmp) / "dst"
        vocab_src = src / "vocabulary"
        vocab_src.mkdir(parents=True)
        with (vocab_src / "predicates.json").open("w") as fh:
            json.dump(good_vocab, fh)

        _copy_vocab(src, dst)  # must not raise

        # _copy_vocab writes to out_dir/vocabulary/predicates.json
        written = json.loads((dst / "vocabulary" / "predicates.json").read_text(encoding="utf-8"))
        assert set(written["idx_to_predicate"].values()) == STANDARD_VG150_PREDICATES
        assert [written["idx_to_predicate"][str(i + 1)] for i in range(50)] == CANONICAL_ORDERED


# ---------------------------------------------------------------------------
# G. Missing source vocab — canonical is written anyway
# ---------------------------------------------------------------------------

def test_G_no_source_predicates_json_writes_canonical():
    """When no source predicates.json is present, the canonical one is written."""
    from tools.prepare_vg150_drive_clean import _copy_vocab

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        dst = Path(tmp) / "dst"
        src.mkdir()  # no vocabulary subdir

        _copy_vocab(src, dst)

        # _copy_vocab writes to out_dir/vocabulary/predicates.json
        out = dst / "vocabulary" / "predicates.json"
        assert out.exists(), "predicates.json should be written even when source vocabulary dir is absent"
        written = json.loads(out.read_text(encoding="utf-8"))
        values = [written["idx_to_predicate"][str(i + 1)] for i in range(50)]
        assert values == CANONICAL_ORDERED
