"""Loss-function smoke tests on small synthetic tensors -- asserting finite
(non-NaN/Inf) output, not any particular numeric value."""
import torch

import pytest

from openvocab_rel.losses import RingQueue, info_nce_inbatch, info_nce_with_queue
from openvocab_rel.config import TrainConfig


def test_info_nce_inbatch_is_finite_scalar():
    x = torch.randn(6, 16)
    y = torch.randn(6, 16)
    loss = info_nce_inbatch(x, y, temp=0.07)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_info_nce_with_queue_falls_back_to_inbatch_when_queue_too_small():
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    queue = RingQueue(dim=8, size=16, device=torch.device("cpu"))
    # queue starts empty -- below min_queue_negatives, so this must take the
    # documented in-batch fallback path rather than erroring
    loss = info_nce_with_queue(x, y, queue, temp=0.07, min_queue_negatives=1024)
    assert torch.isfinite(loss)


def test_ring_queue_fills_exactly_at_capacity():
    queue = RingQueue(dim=4, size=10, device=torch.device("cpu"))
    queue.add(torch.randn(3, 4))
    assert queue.get().shape == (3, 4)
    assert queue.full is False
    queue.add(torch.randn(7, 4))  # 3 + 7 == size: write pointer wraps exactly to 0
    assert queue.full is True
    assert queue.get().shape == (10, 4)


def test_ring_queue_single_add_larger_than_capacity():
    queue = RingQueue(dim=4, size=10, device=torch.device("cpu"))
    queue.add(torch.randn(15, 4))  # one add larger than capacity
    assert queue.full is True
    assert queue.get().shape == (10, 4)


def test_predicate_ce_loss_is_finite():
    pytest.importorskip("transformers")  # openvocab_rel.train imports transformers at module level
    from openvocab_rel.train import _predicate_ce_loss

    cfg = TrainConfig()
    logits = torch.randn(5, 6)
    targets = torch.randint(0, 6, (5,))
    weights = torch.ones(6)
    loss = _predicate_ce_loss(logits, targets, weights, cfg)
    assert torch.isfinite(loss)


def test_predicate_ce_loss_empty_input_is_finite_zero():
    pytest.importorskip("transformers")
    from openvocab_rel.train import _predicate_ce_loss

    cfg = TrainConfig()
    logits = torch.zeros((0, 6))
    targets = torch.zeros((0,), dtype=torch.long)
    weights = torch.ones(6)
    loss = _predicate_ce_loss(logits, targets, weights, cfg)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
