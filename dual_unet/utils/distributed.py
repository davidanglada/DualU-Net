import os
import subprocess
import pickle

import torch
import torch.distributed as dist


def setup_for_distributed(is_master: bool) -> None:
    """
    Disable printing when not in the master process in a distributed environment.

    This function monkey-patches the built-in print function so only the master
    process actually prints, unless 'force=True' is passed to print().

    Args:
        is_master (bool): Whether the current process is the master.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """
    Check if PyTorch distributed is available and has been initialized.

    Returns:
        bool: True if distributed is available and initialized, else False.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """
    Get the number of processes in the current distributed job.

    Returns:
        int: World size (1 if not distributed).
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank (ID) of the current process in the distributed job.

    Returns:
        int: Rank (0 if not distributed).
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size() -> int:
    """
    Get the number of processes on the current node (local size).

    Returns:
        int: Local world size (1 if not distributed).
    """
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ.get('LOCAL_SIZE', 1))


def get_local_rank() -> int:
    """
    Get the local rank of the current process on this node.

    Returns:
        int: Local rank (0 if not distributed).
    """
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """
    Check if the current process is the main (rank 0) process.

    Returns:
        bool: True if current process is rank 0, else False.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    """
    A helper function to save a checkpoint only on the main process.

    Usage: 
        save_on_master(model.state_dict(), "checkpoint.pth")

    The file is written only if `is_main_process()` is True.
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args: dict) -> None:
    """
    Initialize distributed training if environment variables indicate a distributed run.

    This function sets up the environment for PyTorch distributed, 
    sets CUDA devices, and modifies the print function for the main process.

    Supported environment variable configurations:
      - 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_SIZE' for a typical multi-node setup.
      - 'SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_NODELIST' if running under SLURM.

    Args:
        args (dict): A dictionary that will be updated with:
          - rank (int): Global rank
          - world_size (int): Number of processes
          - gpu (int): Local GPU index
          - dist_url (str): URL for initializing the process group (e.g., 'env://')
          - distributed (bool): Whether distributed is activated
          - dist_backend (str): The backend, e.g. 'nccl'
    """
    if dist.is_initialized():
        print("Distributed mode already initialized.")
        return

    # Check environment variables for RANK / WORLD_SIZE
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args['rank'] = int(os.environ["RANK"])
        args['world_size'] = int(os.environ['WORLD_SIZE'])
        args['gpu'] = int(os.environ['LOCAL_RANK'])
        args['dist_url'] = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()

        # Figure out the host IP (MASTER_ADDR) from SLURM_NODELIST
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1'
        )
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)

        args['dist_url'] = 'env://'
        args['world_size'] = ntasks
        args['rank'] = proc_id
        args['gpu'] = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args['distributed'] = False
        args['world_size'] = 1
        args['rank'] = 0
        args['gpu'] = 0
        return

    args['distributed'] = True
    torch.cuda.set_device(args['gpu'])
    args['dist_backend'] = 'nccl'

    print(f'| distributed init (rank {args["rank"]}): {args["dist_url"]}', flush=True)
    dist.init_process_group(
        backend=args['dist_backend'],
        init_method=args['dist_url'],
        world_size=args['world_size'],
        rank=args['rank']
    )
    dist.barrier()

    setup_for_distributed(args['rank'] == 0)


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce the values in the given dictionary across all processes so that 
    each key has the same final (averaged or summed) value on every process.

    Typically used to gather metrics (like losses, etc.) across multiple GPUs.

    Args:
        input_dict (dict): A dictionary of {metric_name: torch.Tensor or scalar} to be reduced.
        average (bool): If True, do an average. If False, do a sum.

    Returns:
        dict: A dictionary with the same keys and the reduced (averaged or summed) values.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        # sort the keys so they are consistent across processes
        names = sorted(input_dict.keys())
        values = [input_dict[k] for k in names]
        # ensure they're torch tensors
        values = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]
        stacked = torch.stack(values).to("cuda")

        # Reduce across all processes
        dist.all_reduce(stacked)
        if average:
            stacked /= world_size

        reduced = {k: v for k, v in zip(names, stacked)}
    return reduced


def all_gather(data):
    """
    Perform all_gather operation on arbitrary picklable (Python) data across all processes.

    Args:
        data (any picklable): The data to gather from all ranks. Not limited to tensors.

    Returns:
        list: A list of data gathered from each rank (of length `world_size`).
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialize the data to a ByteTensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # gather the sizes of each tensor from all ranks
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(s.item()) for s in size_list]
    max_size = max(size_list)

    # pad the tensor if necessary
    padded_list = []
    if local_size != max_size:
        padding = torch.empty(max_size - local_size, dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)

    # gather the padded tensors
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device="cuda") for _ in size_list]
    dist.all_gather(tensor_list, tensor)

    # unpickle the data from each rank
    data_list = []
    for size, t in zip(size_list, tensor_list):
        buffer = t[:size].cpu().numpy().tobytes()
        data_list.append(pickle.loads(buffer))

    return data_list
