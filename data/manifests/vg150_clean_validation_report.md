# VG150-clean dataset validation report

Generated `2026-08-11` against a local run of
`tools/prepare_vg150_drive_clean.py --skip_download --skip_extract
--jsonl_root datasets/vg_raw --out_dir datasets_vg150_clean` on this
machine's pre-existing raw archive (`datasets/vg_raw/`, gitignored, not
committed). This file documents what was produced and how it was
independently checked — it is tracked in git; the dataset itself
(`datasets_vg150_clean/`) is not (see `.gitignore`).

## Source

- `datasets/vg_raw/train.jsonl` (83,881 raw rows), `validation.jsonl`
  (10,485 raw rows), `test.jsonl` (10,486 raw rows) — pre-existing local
  archive, not produced by this validation pass.
- Images: `datasets/vg_raw/images/{VG_100K,VG_100K_2}`, 108,249 files,
  14.6 GB, reused via an NTFS directory junction at
  `datasets_vg150_clean/images` (`mklink /J`, no admin rights required,
  confirmed via `Get-Item` → `LinkType: Junction`) — **not copied**, to
  avoid duplicating 14.6 GB per prepared dataset copy.

## Tool self-reported diagnostics (`datasets_vg150_clean/diagnostics.json`)

| Split | Rows | Relationships | Predicate coverage | Validation issues |
|---|---|---|---|---|
| train | 83,249 | 1,046,427 | 50/50 | none |
| validation | 10,401 | 132,556 | 50/50 | none |
| test | 10,403 | 132,334 | 50/50 | none |

Row loss from raw → clean (632/84/83 dropped for train/val/test) is rows
that had zero boxes or fell below `--min_relationships` after filtering,
not data corruption — expected behavior of `_convert_split`.

`tools/check_vg150_diagnostics.py --require_no_validation_issues` against
this `diagnostics.json`: **PASS** for train and validation (by design this
tool only gates train/validation, not test — `VERIFIED FROM CODE`,
`tools/check_vg150_diagnostics.py` hardcodes the two-split loop).

`tools/validate_dataset.py --dataset vg150 --vg150_root
datasets_vg150_clean`: **PASS** — `local-jsonl` layout satisfied.

## Independent validation (re-derived directly from the JSONL files, not reusing the tool's own counters)

Checked per split: duplicate `image_id` within split, malformed boxes
(non-4-tuple, non-positive width/height, negative coordinate), predicate
strings outside the canonical 50, relationship subject/object indices
outside `[0, n_objects)`, remaining self-relationships, rows with zero
relationships, and (sampled 500 rows/split) whether the referenced image
file exists on disk under the junctioned `images/` path.

| Check | train | validation | test |
|---|---|---|---|
| Rows | 83,249 | 10,401 | 10,403 |
| Duplicate `image_id` within split | 0 | 0 | 0 |
| Malformed boxes | 0 | 0 | 0 |
| Predicate strings outside canonical 50 | 0 | 0 | 0 |
| Relationship indices outside `[0, n_objects)` | 0 | 0 | 0 |
| Remaining self-relationships | 0 | 0 | 0 |
| Rows with zero relationships | 0 | 0 | 0 |
| Missing image files (sampled 500/split) | 0 | 0 | 0 |

Cross-split contamination (`image_id` overlap):

| Pair | Overlap |
|---|---|
| train ∩ validation | 0 |
| train ∩ test | 0 |
| validation ∩ test | 0 |

**Result: no split contamination, no invalid predicates/boxes/indices, no
duplicate or missing images found.** All findings `VERIFIED FROM DATA`.

## Disk footprint

| Artifact | Size |
|---|---|
| `train.jsonl` | 221 MB |
| `validation.jsonl` | 28 MB |
| `test.jsonl` | 28 MB |
| `diagnostics.json` + `vocabulary/` | ~16 KB |
| `images/` | 0 additional bytes (NTFS junction to `datasets/vg_raw/images`) |
| **Total new disk usage** | **~277 MB** (not 14.6 GB — images are not duplicated) |

## Git policy compliance

`datasets_vg150_clean/` was untracked and **not** covered by `.gitignore`
before this pass — `git status` showed the full image tree (tens of
thousands of files) as untracked-and-addable. Fixed by adding
`/datasets_vg150_clean/` to `.gitignore` (one-line, root-anchored, same
pattern as the existing `/datasets/` entry). After the fix, `git status`
shows only the `.gitignore` edit itself — the dataset directory is fully
ignored. This document (and the manifest/validator tooling) are the only
tracked artifacts related to this dataset.
