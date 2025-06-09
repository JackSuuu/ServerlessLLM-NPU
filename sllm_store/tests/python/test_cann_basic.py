import os
import sys
import pytest
import torch

# Add the parent directory to sys.path to import sllm_store
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch_npu
    CANN_AVAILABLE = True
    print("torch_npu successfully imported")
except ImportError:
    CANN_AVAILABLE = False
    print("torch_npu not available")

def test_cann_basic_import():
    """Test basic CANN functionality"""
    if not CANN_AVAILABLE:
        pytest.skip("CANN not available")
    
    # Test device detection
    try:
        device_count = torch_npu.npu.device_count()
        print(f"NPU devices detected: {device_count}")
        assert device_count > 0, "No NPU devices found"
    except Exception as e:
        pytest.fail(f"Failed to get NPU device count: {e}")

def test_sllm_store_import():
    """Test that sllm_store can be imported"""
    try:
        from sllm_store.transformers import load_model
        print("sllm_store.transformers imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import sllm_store.transformers: {e}")

def test_cann_utils_import():
    """Test CANN utils import"""
    if not CANN_AVAILABLE:
        pytest.skip("CANN not available")
    
    try:
        from sllm_store.cann_utils import get_memory_functions, get_device_type
        allocate_memory, get_memory_handles = get_memory_functions()
        device_type = get_device_type()
        print(f"Device type: {device_type}")
        assert device_type == "npu", f"Expected 'npu', got '{device_type}'"
    except ImportError as e:
        print(f"CANN utils not available (expected if not compiled with CANN): {e}")

def test_basic_npu_operations():
    """Test basic NPU tensor operations"""
    if not CANN_AVAILABLE:
        pytest.skip("CANN not available")
    
    try:
        # Test basic tensor creation and movement
        device = torch_npu.npu.device(0)
        x = torch.randn(2, 3)
        x_npu = x.to(device)
        
        print(f"Tensor device: {x_npu.device}")
        print(f"Tensor shape: {x_npu.shape}")
        
        # Test basic operation
        y_npu = x_npu * 2
        y_cpu = y_npu.cpu()
        
        assert torch.allclose(y_cpu, x * 2), "NPU computation failed"
        print("Basic NPU operations successful")
    except Exception as e:
        pytest.fail(f"NPU operations failed: {e}")

if __name__ == "__main__":
    print("Testing CANN basic functionality...")
    test_cann_basic_import()
    test_sllm_store_import()
    test_cann_utils_import()
    test_basic_npu_operations()
    print("All basic tests completed!") 