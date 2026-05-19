import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def ddp_is_enabled() -> bool:
    return int(os.environ.get("RANK", "-1")) != -1


def ddp_init() -> Tuple[int, int]:
    if not ddp_is_enabled():
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return int(rank), int(world)


def is_main() -> bool:
    return (not ddp_is_enabled()) or dist.get_rank() == 0


def unwrap_ddp(m):
    return m.module if isinstance(m, DDP) else m


def ddp_all_gather_tensor(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    xs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


def ddp_rank_offset(n_local: int) -> int:
    if not ddp_is_enabled():
        return 0
    n = torch.tensor([int(n_local)], device="cuda")
    ns = [torch.zeros_like(n) for _ in range(dist.get_world_size())]
    dist.all_gather(ns, n)
    ns_int = [int(t.item()) for t in ns]
    return int(sum(ns_int[: dist.get_rank()]))
