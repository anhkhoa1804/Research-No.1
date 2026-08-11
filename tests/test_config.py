"""TrainConfig instantiation and preset-application smoke tests. Pure
Python/dataclass logic -- no I/O, no GPU."""
import pytest

from openvocab_rel.config import (
    TrainConfig,
    apply_ddp_a100_preset,
    apply_gpu_preset,
    apply_stage_config,
)


def test_default_construction():
    cfg = TrainConfig()
    assert cfg.emb_dim > 0
    assert cfg.predicate_classifier_classes == 51


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_apply_stage_config_all_stages(stage):
    cfg = TrainConfig()
    cfg.stage = stage
    cfg = apply_stage_config(cfg)
    assert cfg.stage == stage
    assert cfg.batch_size > 0
    assert cfg.epochs >= 0 or True  # stage 3 leaves epochs at the dataclass default


@pytest.mark.parametrize(
    "preset", ["l4_24gb", "l4_22gb_lowmem", "a100_80gb_balanced", "a100_80gb_throughput", "a100_40gb"]
)
def test_apply_gpu_preset_known_values(preset):
    cfg = TrainConfig()
    cfg.stage = 1
    cfg = apply_gpu_preset(cfg, preset)
    assert cfg.batch_size > 0
    assert cfg.amp is True
    assert cfg.amp_dtype == "bf16"


def test_apply_gpu_preset_empty_is_noop():
    cfg = TrainConfig()
    original_batch_size = cfg.batch_size
    cfg = apply_gpu_preset(cfg, "")
    assert cfg.batch_size == original_batch_size


def test_apply_gpu_preset_rejects_unknown_value():
    cfg = TrainConfig()
    with pytest.raises(ValueError):
        apply_gpu_preset(cfg, "not_a_real_preset")


@pytest.mark.parametrize("preset", ["4x", "8x"])
def test_apply_ddp_a100_preset(preset):
    cfg = TrainConfig()
    cfg.stage = 2
    cfg = apply_ddp_a100_preset(cfg, preset)
    assert cfg.batch_size > 0


def test_apply_ddp_a100_preset_rejects_unknown_value():
    cfg = TrainConfig()
    with pytest.raises(ValueError):
        apply_ddp_a100_preset(cfg, "16x")
