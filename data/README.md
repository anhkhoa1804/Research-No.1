# External data

Nothing under `data/` is a dataset. This directory holds *documentation and
manifests only* — small, human-readable files that describe what external
data this repo's code expects, so its absence is explicit and checkable
rather than a mystery `FileNotFoundError` three layers deep in a training
run. Real datasets stay outside Git entirely (see the root `.gitignore`:
`datasets/`, `runs/`, `checkpoints/` are all excluded).

Only **VG150** is documented here, because it is the only dataset any
loader in `openvocab_rel/datasets/` actually reads (traced end-to-end:
`openvocab_rel/datasets/vg150_loader.py`). If your local `datasets/`
directory also has `gqa/`, `open_images_v6/`, or `core/` subfolders, those
are not wired into any code path in this repo today — they are out of scope
for this manifest until/unless a loader is written for them.

## VG150 — what's required

The loader supports two on-disk layouts, selected automatically or via
`--vg150_source` (see `openvocab_rel/datasets/vg150_loader.py:_load_vg150_vocab`
and `VG150DataLoader.__init__`):

**1. `local-jsonl` (the maintained path — what every current `scripts/train/*`
and `scripts/eval/*` entrypoint uses by default):**
```text
<DATA_ROOT>/
├── train.jsonl
├── validation.jsonl        (or val.jsonl)
└── images/                 (referenced by relative path from each JSONL row)
```
Each JSONL row needs (at minimum) `image_id`, `obj_boxes`, `objects` (or
`obj_labels`), and `relationships` (list of `{subject_id, object_id,
predicate}` — see `_build_relation_entries` in `vg150_loader.py` for the
exact field-name fallbacks accepted).

**2. `local-files` (raw Visual Genome / VG-SGG format):**
```text
<DATA_ROOT>/
├── VG-SGG-dicts-with-attri.json    (idx_to_label: 150 entries, idx_to_predicate: 50 entries)
├── image_data.json
├── scene_graphs.json
└── images/
```

**Expected vocabulary sizes:** 150 object classes, 50 predicate classes
(+1 synthetic `"relation"` background class appended by the loader at
runtime for the 51-way classifier — see `VG150DataLoader.__init__`). The
authoritative 50-predicate list with group/symmetry metadata is
`configs/predicate_metadata_vg150.json`.

## Where to obtain it

This preserves exactly what the root `README.md`'s "Dataset Preparation"
section already documents — nothing new is invented here:

- **Preferred:** download the Google Drive VG JSONL archive (default file id
  baked into `--drive_id` in `tools/prepare_vg150_drive_clean.py` — see that
  script's `--help` rather than duplicating the id here, to avoid a second
  copy that can drift) and convert it to the `local-jsonl` layout:
  ```bash
  python3 tools/prepare_vg150_drive_clean.py --out_dir datasets_vg150_clean
  ```
- **Small smoke subset alternative**, from the Hugging Face dataset
  `anhkhoa1804/VG150-SGG-Standard`:
  ```bash
  python3 tools/prepare_vg150_subset.py \
    --dataset_id anhkhoa1804/VG150-SGG-Standard \
    --out_dir datasets --train_images 5000 --val_images 500
  ```
- Raw image source (used by the `local-files` layout / any manual setup):
  `https://homes.cs.washington.edu/~ranjay/visualgenome/api.html` — this URL
  was found in a local, third-party-authored `datasets/vg150/README.md`
  bundled with a raw VG150 metadata download on this machine, not in this
  repo's own tracked docs; treat it as a pointer to verify independently,
  not as something this repo vouches for.

## How to validate your local setup

```bash
python3 tools/check_vg150_diagnostics.py \
  --diagnostics datasets_vg150_clean/diagnostics.json \
  --min_train_rows 50000 --min_val_rows 5000 \
  --min_predicate_coverage 50 --require_no_validation_issues
```
validates the *output* of `prepare_vg150_drive_clean.py` (row counts,
predicate coverage, alias-mapping/filter stats).

```bash
python3 tools/validate_dataset.py --dataset vg150 --vg150_root datasets_vg150_clean
```
is a **pre-flight** check (new — added in this cleanup pass) against
`data/manifests/vg150.yaml`: confirms the required files/dirs exist, the
JSONL splits parse, and the observed predicate vocabulary is close to the
expected 50-class coverage, *before* you launch a training run rather than
after it's already been running for an hour.

## What is intentionally not here

No `.jsonl`, `.h5`, `.json` annotation file, or image is committed. No
checksum/hash of the real dataset is stored (VG150 releases have historically
been redistributed through several different mirrors with the same logical
content but no single canonical checksum this repo can pin to — if you need
byte-for-byte dataset identity across machines, compute and compare your own
hash of `train.jsonl`/`validation.jsonl` and record it in your own run notes;
that mechanism doesn't exist here yet, see `docs/known_issues.md`).
