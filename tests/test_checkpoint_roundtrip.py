"""Checkpoint save/load roundtrip smoke test.

Mirrors the real checkpoint dict shape written by openvocab_rel/train.py
(epoch, model, cfg, experiment -- see docs/architecture/training.md
"Checkpointing"), but with a tiny synthetic state_dict instead of real model
weights, so this needs no CLIP/RelationalModel instantiation at all.
"""
import torch

from openvocab_rel.config import TrainConfig


def test_checkpoint_dict_roundtrip(tmp_path):
    cfg = TrainConfig()
    cfg.run_name = "tiny_roundtrip_test"

    fake_state_dict = {
        "decoder.visual_proj.weight": torch.randn(8, 4),
        "decoder.visual_proj.bias": torch.randn(8),
    }

    checkpoint = {
        "epoch": 3,
        "model": fake_state_dict,
        "cfg": cfg.__dict__,
        "experiment": {
            "git_commit": "deadbeef",
            "train_config_hash": "abc123",
            "predicate_vocab_hash": "def456",
            "num_predicates": 50,
        },
    }

    ckpt_path = tmp_path / "tiny_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 3
    assert loaded["cfg"]["run_name"] == "tiny_roundtrip_test"
    assert loaded["experiment"]["num_predicates"] == 50
    assert set(loaded["model"].keys()) == set(fake_state_dict.keys())
    for key, tensor in fake_state_dict.items():
        assert torch.allclose(loaded["model"][key], tensor)


def test_config_dict_is_json_shape_compatible():
    """TrainConfig.__dict__ is embedded directly into checkpoints and
    metrics.jsonl rows (see train.py's _experiment_snapshot); every value
    must be a JSON-serializable primitive, not e.g. a tensor or callable."""
    import json

    cfg = TrainConfig()
    # will raise TypeError if any field is not JSON-serializable
    serialized = json.dumps(cfg.__dict__)
    restored = json.loads(serialized)
    assert restored["emb_dim"] == cfg.emb_dim
    assert restored["run_name"] == cfg.run_name
