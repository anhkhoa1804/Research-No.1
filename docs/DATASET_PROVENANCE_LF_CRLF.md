# Dataset provenance: why the strict preflight fails on this machine

**Status: RESOLVED AS A PROVENANCE FACT, NOT A DATA DEFECT.**
**Do not "fix" this by editing `data/manifests/historical_checkpoint_v1.yaml`.**

`python tools/gcp_preflight.py --strict` fails on this Linux machine with
three SHA256 mismatches. This document records what was measured, what was
proven, and what remains genuinely unresolved.

---

## 1. The observation

| Artifact | Manifest SHA256 | This machine | Rows |
|---|---|---|---|
| `train.jsonl` | `36bc2923…` | `306fc0db…` | 83,249 (match) |
| `validation.jsonl` | `4348ddbb…` | `74d99779…` | 10,401 (match) |
| `vocabulary/predicates.json` | `e4e88e87…` | `76b75952…` | 50 (match) |
| `vocabulary/objects.json` | `9f536376…` | `9f536376…` | **byte-identical** |
| checkpoint, historical prior, `demo_config.env` | — | — | **all byte-identical** |

Row counts, relationship counts, predicate coverage and every filter counter
match `data/manifests/vg150_clean_validation_report.md` exactly.

## 2. The proof — VERIFIED FACT, not inference

The size deltas are exactly one byte per line:

```
230,804,337 + 83,249 = 230,887,586   (train.jsonl,      manifest size)
 28,940,211 + 10,401 =  28,950,612   (validation.jsonl, manifest size)
      1,071 +     53 =       1,124   (predicates.json,  manifest size)
```

That is the signature of CRLF vs LF, but the size arithmetic alone is only
suggestive. The decisive check is cryptographic: `tools/gcp_preflight.py`
now re-hashes this machine's bytes rendered with CRLF line endings and
compares that to the manifest's expected digest.

**It matches, for all three artifacts.** The manifest's expected SHA256 *is*
the SHA256 of this machine's exact content written with CRLF. A collision is
a 1-in-2^256 event, so this is proof, not correlation:

> The canonical hashes were computed on the original Windows machine, where
> Python's default text-mode write emitted CRLF. This checkout writes LF.
> The relationship content is identical.

`objects.json` is byte-identical precisely because `_copy_vocab` copies it
with `shutil.copy2` (binary) rather than re-serializing it — which is exactly
what the CRLF explanation predicts, and is an independent confirmation.

## 3. Independent corroboration: the dataset is bit-for-bit reproducible

Re-running the preparation tool into a separate output directory reproduces
`datasets_vg150_clean/` **byte-for-byte** (all five files, plus
`diagnostics.json` modulo its embedded output paths), in 58 seconds:

```bash
python3 tools/prepare_vg150_drive_clean.py \
  --skip_download --skip_extract \
  --jsonl_root ~/VG150_dataset_extract \
  --out_dir <scratch> \
  --images symlink \
  --allow_source_vocab_mismatch
```

This establishes two things at once: the on-disk dataset really was produced
by this tool from the immutable extract, and the pipeline is deterministic.

### The `--allow_source_vocab_mismatch` flag is required, and was previously unrecorded

Without it the tool exits with:

```
Source predicates.json is incompatible with STANDARD_VG150_PREDICATES.
  Extra (in source, not canonical):   ['growing on', 'says']
  Missing (canonical, not in source): ['next to', 'wrapped around']
```

The raw archive's *vocabulary file* disagrees with the canonical 50-class
set. The raw *relationship data* does not: both `next to` (19,410 train
instances) and `wrapped around` (617) are present and survive filtering, and
`growing on`/`says` are filtered out as non-canonical. The written
`predicates.json` is always derived from `STANDARD_VG150_PREDICATES`, never
copied, so the index↔label mapping is safe. This is the documented purpose of
the flag (commit `7d91af49`).

## 4. What this does and does not license

**Proven:** the content of this machine's dataset is identical to the content
the canonical manifest describes, up to line endings.

**NOT proven, and deliberately not claimed:** that this dataset is the same
*corpus* the historical checkpoint was trained on. It is not. The manifest
itself records that the historical `frequency_prior.json` was built from a
`datasets/train.jsonl` with 251,126 relationships, whereas this dataset has
1,046,427. The original training corpus no longer exists. Byte-equivalence of
the *evaluation* dataset is a separate and weaker claim than provenance of
the *training* corpus, and the two must not be conflated.

## 5. Why the manifest was not edited

`data/manifests/historical_checkpoint_v1.yaml` is the frozen description of
the original experiment. Rewriting its hashes to the LF values would:

- destroy the only record of what the original artifacts' bytes were;
- make a future genuine corruption indistinguishable from this benign case;
- convert a documented provenance gap into a silent assumption.

Instead the preflight was made **more** informative, not more permissive.

## 6. The preflight change (additive)

`tools/gcp_preflight.py` now classifies every hash mismatch as one of:

| Class | Meaning |
|---|---|
| `BYTE-IDENTICAL` | digest matches the manifest exactly |
| `CONTENT-IDENTICAL / LINE-ENDINGS-DIFFER` | same content, CRLF↔LF only |
| `DIFFERENT CONTENT` | differs beyond line endings |

**The gate is unchanged.** Anything short of `BYTE-IDENTICAL` is still a
`[FAIL]` and still exits non-zero, because the historical protocol was
defined against exact bytes. The classification only makes the failure
legible. `tests/test_gcp_preflight.py` pins this explicitly, including a test
asserting that a line-ending-only difference **still fails**.

## 7. Remaining preflight failures on this machine

1. Three `CONTENT-IDENTICAL / LINE-ENDINGS-DIFFER` artifacts (this document).
2. `working tree is DIRTY` — expected during active development; a real run
   must be gated on a clean tree so the run is identifiable.

## 8. A separate provenance gap worth stating

`runs/` does not exist on this machine. Every measured result cited in
`docs/APPEARANCE_PROBE_FINDINGS.md`, `docs/PRIOR_RESIDUAL_HYPOTHESIS.md` and
the experiment matrix refers to run artifacts that were **never transferred
here**. Those numbers are currently claims from documentation, not
independently checkable measurements on this machine. Any of them that
matters to a conclusion should be re-measured rather than cited.

## 9. Machine-local provenance record

`tools/record_provenance.py` writes a machine-local record (hashes, row
counts, environment, GPU identity, preparation command, timestamp) to
`runs/provenance/`. It is explicitly **not** a second canonical manifest and
never amends the first; where the two disagree, the disagreement is the
finding.
