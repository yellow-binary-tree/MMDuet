import os
import random
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def setup_seeds(seed):
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# functions below inherited from lavis

def get_rank():
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_deepspeed_distributed_mode():
    import deepspeed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        return

    torch.cuda.set_device(gpu)
    dist_backend = "nccl"
    dist_url = 'env://'
    print(
        "| distributed init (rank {}, world {}): {}".format(
            rank, world_size, dist_url
        ),
        flush=True,
    )
    deepspeed.init_distributed(
        dist_backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
