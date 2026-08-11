# Smoke-test suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```

What this deliberately does and doesn't cover:

- **No real VG150 data, no real CLIP/Grounding-DINO weights, no network
  access.** `test_dataset_loader.py` reads
  `tests/fixtures/tiny_vg150/{train,validation}.jsonl` (3 + 2 synthetic
  rows, no real images — `VG150JSONLDataset._resolve_image` already falls
  back to a solid gray placeholder image when the file path doesn't exist,
  which is exercised here rather than worked around) with `processor=None`,
  so no CLIP image preprocessing runs. `test_model_forward.py` constructs
  `RelationalModel` directly from `(clip_vision_dim, text_dim)` integers and
  feeds it a random synthetic feature map — the constructor never touches
  an actual CLIP model, so no weights are loaded.
- **`transformers` is imported by a couple of tests** (anything that
  touches `openvocab_rel.train` or `openvocab_rel.clip_utils`, both of
  which import it at module level) but only for its Python API, never for
  an actual model download — those specific tests use
  `pytest.importorskip("transformers")` so the rest of the suite stays
  runnable even in a minimal environment.
- **What's NOT covered**: the full `eval_sgg_standard` pipeline (needs real
  CLIP, optionally Grounding-DINO, both requiring network/HF-cache access),
  and anything that needs real VG150-scale data. This is a known,
  intentional coverage gap — see `docs/known_issues.md`.
