"""Regression tests for tools/gcp_preflight.py.

The preflight is the gate that stops a misconfigured or incompletely
transferred experiment before any GPU time is spent. Each check it performs is
exercised here against a synthetic fixture repository -- tiny stand-in
artifacts with a manifest generated to match, then mutated one failure at a
time.

The real checkpoint (~888 MiB), frequency prior (~97 MiB) and dataset are
gitignored and absent from a fresh clone, so none of these tests may depend on
them.

The fixture is a fresh ``git init`` repository and therefore genuinely lacks
the five required fix commits. The preflight is *right* to fail on those, so
assertions filter ancestry failures out and check them separately.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT = REPO_ROOT / "tools" / "gcp_preflight.py"

pytest.importorskip("yaml", reason="preflight reads a YAML manifest")

sys.path.insert(0, str(REPO_ROOT))
from tools.prepare_vg150_subset import STANDARD_VG150_PREDICATES  # noqa: E402

CANONICAL = sorted(STANDARD_VG150_PREDICATES)


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _git_available(), reason="git is required by the preflight")


def sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path, **overrides) -> None:
    ckpt_dir = root / "checkpoints" / "demo_best"
    data_dir = root / "datasets_vg150_clean"
    values = {
        "checkpoint_sha": sha256_of(ckpt_dir / "pure_best_adapt_light_mR50.pt"),
        "checkpoint_size": (ckpt_dir / "pure_best_adapt_light_mR50.pt").stat().st_size,
        "prior_sha": sha256_of(ckpt_dir / "frequency_prior.json"),
        "train_sha": sha256_of(data_dir / "train.jsonl"),
        "train_rows": 2,
        "val_sha": sha256_of(data_dir / "validation.jsonl"),
        "val_rows": 1,
        "vocab_sha": sha256_of(data_dir / "vocabulary" / "predicates.json"),
        "img_count": 3,
        "pair_count": 2,
    }
    values.update(overrides)
    (root / "data" / "manifests" / "historical_checkpoint_v1.yaml").write_text(
        f"""schema_version: 1
manifest_id: test_fixture_v1
artifacts:
  checkpoint:
    path: checkpoints/demo_best/pure_best_adapt_light_mR50.pt
    sha256: {values['checkpoint_sha']}
    size_bytes: {values['checkpoint_size']}
    required: true
  frequency_prior:
    path: checkpoints/demo_best/frequency_prior.json
    sha256: {values['prior_sha']}
    required: true
    structure:
      predicate_vocab_len: 50
      global_log_probs_len: 50
      pair_log_probs_count: {values['pair_count']}
      subject_log_probs_count: 1
      object_log_probs_count: 1
      smoothing: 1.0
      default_log_prob: -3.912023005428146
  dataset_train:
    path: datasets_vg150_clean/train.jsonl
    sha256: {values['train_sha']}
    rows: {values['train_rows']}
    required: true
  dataset_validation:
    path: datasets_vg150_clean/validation.jsonl
    sha256: {values['val_sha']}
    rows: {values['val_rows']}
    required: true
  predicate_vocabulary:
    path: datasets_vg150_clean/vocabulary/predicates.json
    sha256: {values['vocab_sha']}
    num_predicates: 50
    required: true
  images:
    path: datasets_vg150_clean/images
    required: true
    subdirs:
      VG_100K: {values['img_count']}
    total_files: {values['img_count']}
""",
        encoding="utf-8",
    )


@pytest.fixture
def fixture_repo(tmp_path):
    """A minimal directory that satisfies every preflight check except ancestry."""
    root = tmp_path / "repo"
    for sub in ("openvocab_rel", "tools", "tests", "data/manifests"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy(PREFLIGHT, root / "tools" / "gcp_preflight.py")
    (root / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tools" / "prepare_vg150_subset.py").write_text(
        "STANDARD_VG150_PREDICATES = " + repr(list(STANDARD_VG150_PREDICATES)) + "\n",
        encoding="utf-8",
    )

    ckpt_dir = root / "checkpoints" / "demo_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "pure_best_adapt_light_mR50.pt").write_bytes(b"stand-in-checkpoint")
    (ckpt_dir / "frequency_prior.json").write_text(
        json.dumps({
            "predicate_vocab": CANONICAL,
            "global_log_probs": [-1.0] * 50,
            "pair_log_probs": {"a|b": [-1.0] * 50, "c|d": [-1.0] * 50},
            "subject_log_probs": {"a": [-1.0] * 50},
            "object_log_probs": {"b": [-1.0] * 50},
            "smoothing": 1.0,
            "default_log_prob": -3.912023005428146,
        }),
        encoding="utf-8",
    )

    data_dir = root / "datasets_vg150_clean"
    (data_dir / "vocabulary").mkdir(parents=True, exist_ok=True)
    (data_dir / "images" / "VG_100K").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (data_dir / "images" / "VG_100K" / f"{i}.jpg").write_bytes(b"x")
    (data_dir / "train.jsonl").write_text('{"image_id":1}\n{"image_id":2}\n', encoding="utf-8")
    (data_dir / "validation.jsonl").write_text('{"image_id":3}\n', encoding="utf-8")
    (data_dir / "vocabulary" / "predicates.json").write_text(
        json.dumps({"idx_to_predicate": {str(i + 1): p for i, p in enumerate(CANONICAL)}}),
        encoding="utf-8",
    )

    write_manifest(root)

    for cmd in (["init", "-q"], ["config", "user.email", "t@example.com"],
                ["config", "user.name", "test"], ["add", "-A"],
                ["commit", "-q", "-m", "fixture"]):
        subprocess.run(["git"] + cmd, cwd=root, check=True, capture_output=True)
    return root


def run_preflight(root: Path, *args: str, cwd: Path | None = None) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "gcp_preflight.py"), *args],
        cwd=cwd or root, capture_output=True, text=True, env=env,
    )
    return result.returncode, result.stdout + result.stderr


def non_ancestry_failures(output: str) -> list[str]:
    return [
        line for line in output.splitlines()
        if line.startswith("[FAIL]") and "ancestry" not in line
    ]


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_preflight_exists():
    assert PREFLIGHT.exists()


def test_all_artifact_checks_pass_on_a_well_formed_fixture(fixture_repo):
    _code, out = run_preflight(fixture_repo)
    assert not non_ancestry_failures(out), out
    assert "checkpoint sha256 verified" in out
    assert "predicate vocabulary == sorted(STANDARD_VG150_PREDICATES)" in out
    assert "'global_log_probs' length == 50" in out
    assert "dataset_validation row count == 1" in out
    assert "image tree: 3 files total" in out


def test_no_hardcoded_hashes_in_the_preflight_source():
    """Expected values must come from the manifest, never from the script."""
    import re
    source = PREFLIGHT.read_text(encoding="utf-8")
    # A 64-hex-character literal would be a hardcoded sha256.
    assert not re.search(r"['\"][0-9a-f]{64}['\"]", source), (
        "gcp_preflight.py contains a hardcoded sha256; expected values must be "
        "read from data/manifests/historical_checkpoint_v1.yaml"
    )


# ---------------------------------------------------------------------------
# artifact integrity
# ---------------------------------------------------------------------------

def test_missing_required_artifact_fails(fixture_repo):
    (fixture_repo / "checkpoints/demo_best/pure_best_adapt_light_mR50.pt").unlink()
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "checkpoint missing" in out
    assert "out of band" in out, "the message must say the artifact is not in git"


def test_wrong_checkpoint_hash_fails(fixture_repo):
    (fixture_repo / "checkpoints/demo_best/pure_best_adapt_light_mR50.pt").write_bytes(b"tampered!")
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "SHA256 MISMATCH" in out


def test_truncated_artifact_reports_a_size_mismatch(fixture_repo):
    (fixture_repo / "checkpoints/demo_best/pure_best_adapt_light_mR50.pt").write_bytes(b"short")
    _code, out = run_preflight(fixture_repo)
    assert "SIZE mismatch" in out
    assert "truncated" in out


# ---------------------------------------------------------------------------
# frequency prior -- every condition _load_frequency_bias swallows silently
# ---------------------------------------------------------------------------

def test_unparseable_frequency_prior_fails(fixture_repo):
    prior = fixture_repo / "checkpoints/demo_best/frequency_prior.json"
    prior.write_text("{ not json", encoding="utf-8")
    write_manifest(fixture_repo, prior_sha=sha256_of(prior))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "does not parse as JSON" in out
    assert "UNCALIBRATED" in out


def test_wrong_length_global_log_probs_fails(fixture_repo):
    prior = fixture_repo / "checkpoints/demo_best/frequency_prior.json"
    data = json.loads(prior.read_text(encoding="utf-8"))
    data["global_log_probs"] = [-1.0] * 7
    prior.write_text(json.dumps(data), encoding="utf-8")
    write_manifest(fixture_repo, prior_sha=sha256_of(prior))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "global_log_probs" in out and "expected 50" in out


def test_non_canonical_prior_vocabulary_fails(fixture_repo):
    prior = fixture_repo / "checkpoints/demo_best/frequency_prior.json"
    data = json.loads(prior.read_text(encoding="utf-8"))
    data["predicate_vocab"] = ["not_a_real_predicate"] + CANONICAL[1:]
    prior.write_text(json.dumps(data), encoding="utf-8")
    write_manifest(fixture_repo, prior_sha=sha256_of(prior))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "!= canonical vocabulary" in out


def test_wrong_length_probability_rows_fail(fixture_repo):
    prior = fixture_repo / "checkpoints/demo_best/frequency_prior.json"
    data = json.loads(prior.read_text(encoding="utf-8"))
    data["pair_log_probs"]["a|b"] = [-1.0] * 3
    prior.write_text(json.dumps(data), encoding="utf-8")
    write_manifest(fixture_repo, prior_sha=sha256_of(prior))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "silently DROPPED" in out


# ---------------------------------------------------------------------------
# dataset and vocabulary identity
# ---------------------------------------------------------------------------

def test_wrong_validation_row_count_fails(fixture_repo):
    val = fixture_repo / "datasets_vg150_clean/validation.jsonl"
    val.write_text('{"image_id":3}\n{"image_id":4}\n{"image_id":5}\n', encoding="utf-8")
    write_manifest(fixture_repo, val_sha=sha256_of(val))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "row count 3 != expected 1" in out


def test_swapped_vocabulary_indices_fail(fixture_repo):
    vocab = fixture_repo / "datasets_vg150_clean/vocabulary/predicates.json"
    order = list(CANONICAL)
    order[3], order[4] = order[4], order[3]
    vocab.write_text(
        json.dumps({"idx_to_predicate": {str(i + 1): p for i, p in enumerate(order)}}),
        encoding="utf-8",
    )
    write_manifest(fixture_repo, vocab_sha=sha256_of(vocab))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "NOT sorted(STANDARD_VG150_PREDICATES)" in out
    assert "2 of 50 indices differ" in out


def test_wrong_vocabulary_length_fails(fixture_repo):
    vocab = fixture_repo / "datasets_vg150_clean/vocabulary/predicates.json"
    vocab.write_text(
        json.dumps({"idx_to_predicate": {str(i + 1): p for i, p in enumerate(CANONICAL[:40])}}),
        encoding="utf-8",
    )
    write_manifest(fixture_repo, vocab_sha=sha256_of(vocab))
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "has 40 entries, expected 50" in out


def test_empty_image_tree_fails(fixture_repo):
    images = fixture_repo / "datasets_vg150_clean/images/VG_100K"
    for child in list(images.iterdir()):
        child.unlink()
    images.rmdir()
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "NO image files" in out
    assert "gray placeholder" in out, "the message must explain the silent-degradation risk"


# ---------------------------------------------------------------------------
# strict mode
# ---------------------------------------------------------------------------

def test_dirty_tree_warns_permissively_but_fails_strictly(fixture_repo):
    (fixture_repo / "uncommitted.txt").write_text("x", encoding="utf-8")
    _code, permissive = run_preflight(fixture_repo)
    _code, strict = run_preflight(fixture_repo, "--strict")
    assert "[WARN] working tree is DIRTY" in permissive
    assert "[FAIL] working tree is DIRTY" not in permissive
    assert "[FAIL] working tree is DIRTY" in strict


def test_absent_cuda_warns_permissively_but_fails_strictly(fixture_repo):
    """These tests run on CPU, so this exercises the real code path."""
    pytest.importorskip("torch")
    import torch
    if torch.cuda.is_available():
        pytest.skip("this assertion is only meaningful on a CPU-only machine")
    _code, permissive = run_preflight(fixture_repo)
    _code, strict = run_preflight(fixture_repo, "--strict")
    assert "[WARN] CUDA is NOT available" in permissive
    assert "[FAIL] CUDA is NOT available" in strict
    assert "Refusing to certify a GPU run" in strict


# ---------------------------------------------------------------------------
# repository and commit preconditions
# ---------------------------------------------------------------------------

def test_missing_required_fix_commits_are_reported(fixture_repo):
    """The fixture is a fresh repo, so all five required fixes are absent."""
    _code, out = run_preflight(fixture_repo)
    assert out.count("required fix NOT in HEAD's ancestry") >= 5
    assert "would produce invalid results" in out


def test_running_outside_the_repository_root_fails(fixture_repo, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    code, out = run_preflight(fixture_repo, cwd=outside)
    assert code != 0
    assert "does not look like the repository root" in out


def test_missing_manifest_exits_two(fixture_repo):
    (fixture_repo / "data/manifests/historical_checkpoint_v1.yaml").unlink()
    code, out = run_preflight(fixture_repo)
    assert code == 2, "an unusable manifest is distinct from a failed check"
    assert "PREFLIGHT ABORTED" in out


# ---------------------------------------------------------------------------
# manifest capture
# ---------------------------------------------------------------------------

def test_manifest_captures_environment_and_command(fixture_repo, tmp_path):
    import yaml
    out_path = tmp_path / "manifest.yaml"
    run_preflight(
        fixture_repo, "--out", str(out_path),
        "--run-name", "unit_test_run",
        "--command", "python -m openvocab_rel.train --stage 3",
    )
    assert out_path.exists()
    captured = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    for key in ("git_commit", "git_dirty", "python_version", "platform",
                "hostname", "preflight_timestamp_utc", "preflight_status",
                "run_name", "eval_command", "artifacts", "dataset_rows"):
        assert key in captured, f"manifest is missing {key}"
    assert captured["run_name"] == "unit_test_run"
    assert "--stage 3" in captured["eval_command"]
    assert captured["artifacts"]["checkpoint"]["sha256"]


def test_repo_manifest_is_parseable_and_complete():
    """The real manifest in this repository must stay valid."""
    import yaml
    path = REPO_ROOT / "data" / "manifests" / "historical_checkpoint_v1.yaml"
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["manifest_id"] == "historical_checkpoint_v1"
    for key in ("checkpoint", "frequency_prior", "dataset_train",
                "dataset_validation", "predicate_vocabulary", "images"):
        assert key in data["artifacts"], f"manifest lacks an entry for {key}"
        assert data["artifacts"][key]["path"]
    for key in ("checkpoint", "frequency_prior", "dataset_train",
                "dataset_validation", "predicate_vocabulary"):
        sha = str(data["artifacts"][key]["sha256"])
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
            f"{key} sha256 is not a well-formed hex digest"
        )
    assert len(data["compatibility"]["required_overrides"]) >= 13
    assert data["historical_claim"]["status"].startswith("HISTORICAL_EVIDENCE")


def test_historical_claim_is_never_labelled_a_baseline():
    """Guard the epistemic labelling the manifest exists to enforce."""
    path = REPO_ROOT / "data" / "manifests" / "historical_checkpoint_v1.yaml"
    import yaml
    claim = yaml.safe_load(path.read_text(encoding="utf-8"))["historical_claim"]
    status = claim["status"].lower()
    # Compare whole tokens: "unreproduced" legitimately contains "reproduced".
    tokens = {t.strip() for t in status.replace("/", " ").split()}
    for forbidden in ("baseline", "verified", "reproduced"):
        assert forbidden not in tokens, (
            f"historical_claim.status must not describe itself as {forbidden!r}: {claim['status']!r}"
        )
    assert "unreproduced" in tokens
    assert "single_source" in tokens


# ---------------------------------------------------------------------------
# artifact identity classification (byte / content / different)
#
# These cover the additive diagnostic that explains WHY a hash mismatched.
# The diagnostic must never soften the gate: the historical experiment was
# defined against exact bytes, so anything short of BYTE-IDENTICAL still fails.
# ---------------------------------------------------------------------------

def _load_preflight_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_pf_under_test", PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_line_ending_variant_hashes_match_hand_computed_values(tmp_path):
    import hashlib
    pf = _load_preflight_module()
    path = tmp_path / "lf.txt"
    path.write_bytes(b"alpha\nbeta\ngamma\n")
    sha_lf, sha_crlf = pf.sha256_line_ending_variants(path)
    assert sha_lf == hashlib.sha256(b"alpha\nbeta\ngamma\n").hexdigest()
    assert sha_crlf == hashlib.sha256(b"alpha\r\nbeta\r\ngamma\r\n").hexdigest()


def test_line_ending_variants_are_symmetric_for_a_crlf_file(tmp_path):
    """A CRLF file and its LF twin must produce the same variant pair."""
    pf = _load_preflight_module()
    lf_path = tmp_path / "a.txt"
    crlf_path = tmp_path / "b.txt"
    lf_path.write_bytes(b"one\ntwo\n")
    crlf_path.write_bytes(b"one\r\ntwo\r\n")
    assert pf.sha256_line_ending_variants(lf_path) == pf.sha256_line_ending_variants(crlf_path)


def test_line_ending_variants_handle_a_crlf_split_across_chunks(tmp_path):
    """A CRLF straddling the read boundary must not be misread as CR + LF."""
    pf = _load_preflight_module()
    path = tmp_path / "split.txt"
    path.write_bytes(b"xxxx\r\nyyyy\r\n")
    # chunk=5 puts the boundary exactly between the CR and the LF.
    assert pf.sha256_line_ending_variants(path, chunk=5) == pf.sha256_line_ending_variants(path)


def test_lone_cr_is_treated_as_content_not_a_line_ending(tmp_path):
    import hashlib
    pf = _load_preflight_module()
    path = tmp_path / "mac.txt"
    path.write_bytes(b"a\rb")
    sha_lf, sha_crlf = pf.sha256_line_ending_variants(path)
    assert sha_lf == hashlib.sha256(b"a\rb").hexdigest()
    assert sha_crlf == hashlib.sha256(b"a\rb").hexdigest()


def test_classify_identity_reports_byte_identical_on_a_match(tmp_path):
    pf = _load_preflight_module()
    path = tmp_path / "f.txt"
    path.write_bytes(b"content\n")
    actual = pf.sha256_of(path)
    identity, _detail = pf.classify_identity(path, actual, actual)
    assert identity == pf.IDENTITY_BYTE


def test_classify_identity_detects_a_line_ending_only_difference(tmp_path):
    import hashlib
    pf = _load_preflight_module()
    path = tmp_path / "f.txt"
    path.write_bytes(b"row1\nrow2\n")
    crlf_sha = hashlib.sha256(b"row1\r\nrow2\r\n").hexdigest()
    identity, detail = pf.classify_identity(path, crlf_sha, pf.sha256_of(path))
    assert identity == pf.IDENTITY_LINE_ENDINGS
    assert "LF" in detail


def test_classify_identity_reports_different_content_when_bytes_really_differ(tmp_path):
    import hashlib
    pf = _load_preflight_module()
    path = tmp_path / "f.txt"
    path.write_bytes(b"row1\nrow2\n")
    other = hashlib.sha256(b"totally different\n").hexdigest()
    identity, detail = pf.classify_identity(path, other, pf.sha256_of(path))
    assert identity == pf.IDENTITY_DIFFERENT
    assert "beyond line endings" in detail


def test_line_ending_mismatch_is_reported_but_STILL_FAILS(fixture_repo):
    """The gate must not be weakened by the diagnostic.

    Rewriting train.jsonl with CRLF leaves the content identical but the bytes
    different. The preflight must say so explicitly AND still fail, because the
    historical run was defined against exact bytes.
    """
    train = fixture_repo / "datasets_vg150_clean" / "train.jsonl"
    train.write_bytes(train.read_bytes().replace(b"\n", b"\r\n"))
    code, out = run_preflight(fixture_repo)
    assert code == 1, "a line-ending-only difference must still fail the gate"
    assert "dataset_train SHA256 MISMATCH" in out
    assert "CONTENT-IDENTICAL / LINE-ENDINGS-DIFFER" in out
    assert any("dataset_train SHA256 MISMATCH" in f for f in non_ancestry_failures(out)), \
        "the mismatch must be recorded as a real failure, not merely printed"


def test_genuinely_different_content_is_classified_as_different(fixture_repo):
    train = fixture_repo / "datasets_vg150_clean" / "train.jsonl"
    train.write_text('{"image_id":99}\n{"image_id":98}\n', encoding="utf-8")
    code, out = run_preflight(fixture_repo)
    assert code == 1
    assert "DIFFERENT CONTENT" in out
    assert "CONTENT-IDENTICAL" not in out


def test_identity_class_is_recorded_in_the_captured_manifest(fixture_repo, tmp_path):
    import yaml
    train = fixture_repo / "datasets_vg150_clean" / "train.jsonl"
    train.write_bytes(train.read_bytes().replace(b"\n", b"\r\n"))
    out_file = tmp_path / "captured.yaml"
    run_preflight(fixture_repo, "--out", str(out_file))
    captured = yaml.safe_load(out_file.read_text(encoding="utf-8"))
    assert captured["artifacts"]["dataset_train"]["identity"] == \
        "CONTENT-IDENTICAL / LINE-ENDINGS-DIFFER"
    assert captured["artifacts"]["checkpoint"]["identity"] == "BYTE-IDENTICAL"
    assert captured["preflight_status"] == "FAILED"
