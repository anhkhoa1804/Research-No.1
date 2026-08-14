# GCP experiment protocol — historical checkpoint reproduction

The exact, ordered workflow for the first serious GCP run. Follow it in
order; every step gates the next.

**What this experiment is**: a *measurement* of whether the recovered
historical checkpoint, evaluated under its recorded protocol on current
code and current data, reproduces its self-reported `R@50 = 67.09 %` /
`mR@50 = 22.64 %`.

**What this experiment is not**: a test the model can pass or fail.
Reproduces, partially reproduces, and fails to reproduce are all valid,
informative outcomes. There is **no acceptance threshold on the metric**,
and none should be invented after seeing the number.

Companion documents: `docs/HISTORICAL_CHECKPOINT_MANIFEST.md` (what the
artifacts are and why each override exists), `docs/PROJECT_STATUS.md`
(current state of record), `docs/known_issues.md` (registered hazards).

---

## 0. Before you provision anything

Read `docs/HISTORICAL_CHECKPOINT_MANIFEST.md` §0. If you cannot state the
difference between the *historical claim*, the *current verified
behavior*, and the *reproduction target* without looking, read it again.
Conflating them is the single most likely way this experiment produces a
wrong conclusion.

**Known hazards that this workflow is built around** (details in
`docs/known_issues.md`):

| Hazard | Consequence if unguarded | Guard |
|---|---|---|
| `_load_frequency_bias` returns `None` on six conditions, silently | run completes **uncalibrated**, looks like a failed reproduction | preflight validates the prior structurally |
| `parse_known_args` discards unknown flags, silently | a typo leaves a stage-3 default active, no runtime signal | `tests/test_historical_eval_protocol.py` |
| `--stage 3` forces 4 architecture flags on | checkpoint loads into untrained heads | dedicated entrypoint sets all 13 explicitly |
| `images/` is an NTFS junction | every image degrades to a gray placeholder | preflight counts files in `VG_100K*` |
| Fixed `RUN_NAME` overwrites prior runs | previous results lost | timestamped run dirs + overwrite refusal |

---

## 1. Provision

An **L4 24 GB** instance matches the `l4_24gb` preset the entrypoint uses
(`batch_size=12`, `num_workers=4`). An A100 works; adjust nothing — the
preset is passed explicitly and `_reapply_explicit_cli_args` keeps the
batch size the script chose.

Disk: **≥ 40 GB free** — 1.2 GiB of artifacts, ~14.6 GB of images, plus
the CLIP download and run outputs.

---

## 2. Clone the code

```bash
git clone <this repository> && cd <repo>
git checkout <the commit you intend to run>
git log -1 --format='%H %s'
```

Record that commit hash. Every result must be attributable to it.

---

## 3. Transfer the external artifacts

**`git clone` does not give you a runnable experiment.** Everything below
is gitignored and must be copied separately — **1.20 GiB** plus images:

```
checkpoints/demo_best/pure_best_adapt_light_mR50.pt    931,057,422 B
checkpoints/demo_best/frequency_prior.json             101,944,045 B
checkpoints/demo_best/demo_config.env                          355 B
datasets_vg150_clean/train.jsonl                       230,887,586 B
datasets_vg150_clean/validation.jsonl                   28,950,612 B
datasets_vg150_clean/vocabulary/predicates.json              1,124 B
datasets_vg150_clean/vocabulary/objects.json                 3,056 B
datasets_vg150_clean/images/VG_100K/                    64,346 files
datasets_vg150_clean/images/VG_100K_2/                  43,903 files
```

> **⚠ The image tree is an NTFS junction on the Windows source machine**,
> pointing at `datasets\vg_raw\images`. It is not a copy. Whether an
> archive follows it depends entirely on the tool and its flags. **Verify
> the file counts on the target** — the preflight does this for you, and
> it is not a formality: an unfollowed junction produces an empty
> directory, every image silently falls back to a gray placeholder
> (`VG150JSONLDataset._resolve_image`), and the run completes reporting
> meaningless numbers.

`train.jsonl` is required **even though this is an eval-only run** —
`adaptive_calibration_enabled=true` reads train-split statistics at eval
time.

Set up the environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 4. Preflight — the first gate

```bash
python tools/gcp_preflight.py --strict
```

`--strict` is **mandatory on GCP**. It turns a dirty working tree, absent
CUDA, and missing torch/transformers into failures rather than warnings.

It verifies: repository root; clean tree; five required fix commits in
ancestry (`220c5c2e`, `9dc8f45d`, `7d91af49`, `fa8c0c3b`, `65686b5f`);
seven artifact SHA256s and sizes; train and validation row counts (83,249
/ 10,401); the predicate vocabulary's length *and* canonical ordering; all
six frequency-prior structural conditions; the image tree file counts;
Python ≥ 3.10; CUDA present with GPU name and VRAM.

**Exit 0 or stop.** Exit 1 = a check failed (all failures are listed
together, so fix them in one pass). Exit 2 = the preflight itself could
not run.

Expected artifact hashes live in
`data/manifests/historical_checkpoint_v1.yaml`. **Never edit that file to
make a failing check pass** — a mismatch means you do not have the
artifact the experiment was defined against.

---

## 5. Canary — the second gate

```bash
bash scripts/eval/eval_historical_checkpoint.sh --canary
```

Two batches. It prints the fully resolved protocol before starting, runs
the preflight itself, evaluates, then verifies the **resolved** runtime
settings read back out of `metrics.jsonl` — not the ones the script
intended.

The verifier checks: all 13 compatibility settings, the 50-predicate
vocabulary, checkpoint identity, that the frequency prior file exists at
the configured path, and that the run is not pathological (non-zero
images, non-zero GT, no NaN). It cross-checks `predicate_diag` against the
config as an independent second source.

It prints `PASS` or `FAIL`.

> **A `PASS` means the protocol is correct. It says nothing about whether
> the numbers are good** — the verifier deliberately does not assert on
> `R@50`/`mR@50`, and prints them marked "recorded, NOT asserted". Judging
> the number is the research phase's job, on the full run, not this one.

To see what would run without running it:

```bash
bash scripts/eval/eval_historical_checkpoint.sh --canary --dry-run
```

---

## 6. Full run

Blocked until a canary verdict starting with `PASS` exists:

```bash
bash scripts/eval/eval_historical_checkpoint.sh --full
```

Entire 10,401-image validation split. Output goes to a timestamped
`runs/historical_full_<UTC>/`; the script refuses to overwrite an existing
directory.

Expect hours, not minutes. The measured CPU cost is ~36 s/image (~104 h
for the split); a GPU is dramatically faster but this is not a
five-minute job. Run it under `tmux`/`screen`.

---

## 7. Artifacts to keep

Every run directory should end up with:

| File | Written | Contents |
|---|---|---|
| `command.txt` | before | exact command, shell-quoted |
| `git_commit.txt` | before | HEAD at launch |
| `git_status.txt` | before | working-tree state at launch |
| `environment.txt` | before + after | mode, run name, host, Python, platform, torch, CUDA, GPU, start/finish, exit status |
| `manifest.yaml` | preflight | every artifact hash, dataset row counts, vocab hash, GPU, timestamp, the gated command |
| `run.log` | during | full stdout/stderr |
| `metrics.jsonl` | during | metrics + embedded resolved config + `experiment` snapshot |
| `canary_verdict.txt` | after | `PASS`/`FAIL` and every individual check |

`command.txt`, `git_commit.txt` and `environment.txt` are deliberately
written **before** the run, so a crashed run is still identifiable.

`metrics.jsonl` additionally carries `predicate_diag` (score mode,
ensemble alpha, GT vs predicted distributions), `pair_proposal`
(`gt_pair_recall@K`, candidate counts), `object_diag` (CLIP top-1/top-k
object accuracy, triplet endpoint coverage), `routing_diag` and
`role_swap_diag`.

Copy the whole run directory off the instance before tearing it down.
`runs/` is gitignored.

---

## 8. Interpreting the result

Compare against `HISTORICAL EVIDENCE: R@50 = 67.09 %, mR@50 = 22.64 %` —
and read the caveats before drawing any conclusion.

**Five things about the historical protocol are `UNKNOWN`** and any of
them can move the number:

1. which **split** it was measured on
2. **how many images** it covered
3. **pooled vs per-image-averaged** R@K aggregation — this repo computes
   both (`R@50` and `image_mean_R@50`) and they differ materially; report
   both
4. `clip_input_res` at eval time
5. whether `FREQ_BIAS_ENABLED` was actually set (inferred, never recorded)

**Therefore: a gap is not evidence of a model defect.** All five innocent
explanations must be considered before "the checkpoint is worse than
claimed" becomes the leading hypothesis. Record the verdict in
`docs/PROJECT_STATUS.md` as one of:

- **REPRODUCED** — within a stated tolerance you commit to *before*
  seeing the number
- **PARTIALLY REPRODUCED** — one metric lands, the other does not; say
  which and by how much
- **NOT REPRODUCED** — with the ranked candidate explanations, `UNKNOWN`s
  first

Whatever the outcome, the number becomes the project's **first
trustworthy baseline** only if the canary passed and the preflight was
clean. Otherwise it is another diagnostic.

---

## 9. What must not be changed to make this work

If the experiment is inconvenient, the answer is never to relax a check:

- **Do not** edit `data/manifests/historical_checkpoint_v1.yaml` hashes.
- **Do not** pass `ALLOW_UNGATED_FULL=1` to skip the canary.
- **Do not** pass `ALLOW_OVERWRITE=1` onto an existing run directory.
- **Do not** drop `--strict` from the GCP preflight.
- **Do not** modify `checkpoints/demo_best/*` — ever, for any reason.
- **Do not** regenerate `vocabulary/predicates.json` — the checkpoint's
  index mapping and the frequency prior both depend on its current order.
- **Do not** change any of the 13 compatibility overrides.
- **Do not** use `scripts/eval/eval_l4_phase34.sh` for this checkpoint.

---

## 10. Quick reference

```bash
# gate 1 -- artifacts, dataset identity, environment
python tools/gcp_preflight.py --strict

# gate 2 -- 2 batches, verifies the resolved protocol, prints PASS/FAIL
bash scripts/eval/eval_historical_checkpoint.sh --canary

# the experiment -- refuses to start without a passing canary
bash scripts/eval/eval_historical_checkpoint.sh --full

# inspect without executing
bash scripts/eval/eval_historical_checkpoint.sh --canary --dry-run

# re-verify any completed run
python tools/verify_canary.py runs/<run_name>/metrics.jsonl

# summarise
python tools/model_report_card.py runs/<run_name>/metrics.jsonl \
  --train_jsonl datasets_vg150_clean/train.jsonl
```
