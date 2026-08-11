"""Regression test for the R@K metric-semantics presentation fix.

Prior behavior: tools/model_report_card.py only ever read the headline,
dataset-global-pooled "R@50" field and printed it as a bare "R@50=..." --
image_mean_R@50 (the per-image-averaged variant most VG150 literature
reports under the same name) was silently never surfaced anywhere in this
tool's output, so a reader had no way to tell which statistic they were
looking at, or that a second one even existed.

Fixed behavior: image_mean_R@50 is read into its own, separately-labeled
field and displayed alongside R@50(pooled) rather than being dropped.
Ranking/selection logic (_best/_combined_score) is untouched -- this is a
presentation-only change; no reported number's meaning changed, only
whether both are now visible.

Pure unit test on _summaries(); no CLI subprocess, no real metrics.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = str(Path(__file__).parent.parent / "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from model_report_card import _summaries  # noqa: E402


def test_pooled_and_image_mean_r_at_50_are_both_surfaced_and_distinct():
    row = {
        "run_name": "smoke",
        "epoch": 1,
        "val_sgg": {
            "predcls": {
                "R@50": 0.55,  # dataset-global-pooled (the headline field)
                "image_mean_R@50": 0.42,  # per-image-averaged (the literature-matching variant)
                "mR@50": 0.20,
            }
        },
        "config": {},
    }
    summaries = _summaries([(Path("fake/metrics.jsonl"), row)], buckets={}, lambda_mr=1.0, mu_tail=1.0)

    assert len(summaries) == 1
    entry = summaries[0]
    # Both fields present, neither silently dropped.
    assert "R@50" in entry
    assert "image_mean_R@50" in entry
    # And -- the whole point -- they must not be conflated into one value.
    assert entry["R@50"] == 0.55
    assert entry["image_mean_R@50"] == 0.42
    assert entry["R@50"] != entry["image_mean_R@50"]


def test_missing_image_mean_r_at_50_defaults_to_zero_not_a_crash():
    # Older metrics rows (pre-existing this variant) must not break the tool.
    row = {
        "run_name": "old_run",
        "epoch": 1,
        "val_sgg": {"predcls": {"R@50": 0.60, "mR@50": 0.15}},
        "config": {},
    }
    summaries = _summaries([(Path("fake/metrics.jsonl"), row)], buckets={}, lambda_mr=1.0, mu_tail=1.0)
    assert summaries[0]["image_mean_R@50"] == 0.0
