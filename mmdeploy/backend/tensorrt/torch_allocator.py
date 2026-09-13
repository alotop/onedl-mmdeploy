# Copyright (c) OpenMMLab. All rights reserved.
import ctypes

import tensorrt as trt
import torch

from mmdeploy.utils import get_root_logger

_CUDART = None
_CUDART_RESOLVED = False


def _get_cudart():
    """Best-effort handle to the CUDA runtime library.

    Used only to consume a pending CUDA error, so failures are silent.
    """
    global _CUDART, _CUDART_RESOLVED
    if _CUDART_RESOLVED:
        return _CUDART
    _CUDART_RESOLVED = True
    candidates = []
    cuda_ver = getattr(torch.version, 'cuda', None)
    if cuda_ver:
        candidates.append('cudart64_{}.dll'.format(
            str(cuda_ver).split('.')[0]))
    candidates += ['cudart64_13.dll', 'cudart64_12.dll', 'libcudart.so']
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaGetLastError.restype = ctypes.c_int
            lib.cudaGetLastError.argtypes = []
            _CUDART = lib
            break
        except Exception:
            continue
    return _CUDART


def _clear_pending_cuda_error() -> None:
    """Consume a pending CUDA error of the calling thread.

    ``torch.cuda.caching_allocator_alloc`` raises if *any* CUDA error is
    pending, even one recorded by an unrelated kernel launch (for example a
    probe launch with no kernel image for the current GPU architecture).
    TensorRT reports that exception as a bogus ``OutOfMemory`` and aborts
    inference, so clear it before allocating.
    """
    lib = _get_cudart()
    if lib is None:
        return
    try:
        lib.cudaGetLastError()
    except Exception:
        pass


class TorchAllocator(trt.IGpuAllocator):
    """PyTorch Cuda Allocator Wrapper."""

    def __init__(self, device_id: int = 0) -> None:
        super().__init__()

        self.device_id = device_id
        self.mems = set()
        self.caching_delete = torch._C._cuda_cudaCachingAllocator_raw_delete

    def __del__(self):
        """destructor."""
        mems = self.mems.copy()
        (self.deallocate(mem) for mem in mems)

    def allocate(self: trt.IGpuAllocator, size: int, alignment: int,
                 flags: int) -> int:
        """Allocate gpu memory.

        Args:
            self (trt.IGpuAllocator): gpu allocator
            size (int): memory size.
            alignment (int): memory alignment.
            flags (int): flags.

        Returns:
            int: memory address.
        """
        torch_stream = torch.cuda.current_stream(self.device_id)
        logger = get_root_logger()
        logger.debug(f'allocate {size} memory with TorchAllocator.')
        assert alignment >= 0
        if alignment > 0:
            size = size | (alignment - 1) + 1
        # clear any stale CUDA error first, otherwise torch raises and
        # TensorRT turns it into a bogus OutOfMemory that kills inference
        _clear_pending_cuda_error()
        try:
            mem = torch.cuda.caching_allocator_alloc(
                size, device=self.device_id, stream=torch_stream)
        except RuntimeError:
            logger.warning(
                'torch caching allocator failed, clearing the pending CUDA '
                'error and retrying once.')
            _clear_pending_cuda_error()
            mem = torch.cuda.caching_allocator_alloc(
                size, device=self.device_id, stream=torch_stream)
        self.mems.add(mem)
        return mem

    def deallocate(self: trt.IGpuAllocator, memory: int) -> bool:
        """Deallocate memory.

        Args:
            self (trt.IGpuAllocator): gpu allocator
            memory (int): memory address.

        Returns:
            bool: deallocate success.
        """
        if memory not in self.mems:
            return False

        self.caching_delete(memory)
        self.mems.discard(memory)
        return True
