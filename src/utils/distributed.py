"""
Distributed Training Utilities
"""
import os
import torch
import torch.distributed as dist


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize distributed training.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        backend: Distributed backend (nccl, gloo)
    """
    # Set device first
    torch.cuda.set_device(rank)
    
    # Initialize process group with device_id to suppress warnings
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f'cuda:{rank}')
    )
    
    # Wait for all processes to initialize
    torch.cuda.synchronize()


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = None) -> bool:
    """
    Check if current process is the main process.
    
    Args:
        rank: Process rank (if None, will check if distributed is initialized)
        
    Returns:
        True if main process
    """
    if rank is not None:
        return rank == 0
    
    if not dist.is_initialized():
        return True
    
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size (number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        world_size: Number of processes
        
    Returns:
        Reduced tensor
    """
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def synchronize():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
