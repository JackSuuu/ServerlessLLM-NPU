"""CANN utilities for ServerlessLLM"""
import torch

def is_cann_available():
    """Check if CANN/NPU is available"""
    try:
        # Check if torch_npu is available
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False

def get_device_type():
    """Get the device type (cuda or npu)"""
    if is_cann_available():
        return "npu"
    else:
        return "cuda"

def get_memory_functions():
    """Get the appropriate memory allocation functions"""
    if is_cann_available():
        try:
            from sllm_store._C import allocate_cann_memory, get_cann_memory_handles
            return allocate_cann_memory, get_cann_memory_handles
        except ImportError:
            raise ImportError("CANN functions not available in compiled extension")
    else:
        from sllm_store._C import allocate_cuda_memory, get_cuda_memory_handles
        return allocate_cuda_memory, get_cuda_memory_handles