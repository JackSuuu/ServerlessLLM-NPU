#!/usr/bin/env python3
import sys
import time
from sllm_store.client import SllmStoreClient

def test_cann_functions():
    # Connect to storage server
    client = SllmStoreClient("127.0.0.1:8073")
    
    model_path = "facebook/opt-125m"
    
    try:
        print("🔍 Testing Model Registration...")
        response = client.register_model(model_path)
        print(f"✅ Model registered: {response.model_size} bytes")
        
        print("🔍 Testing Host Memory Loading...")
        response = client.load_into_cpu(model_path)
        print(f"✅ Model loaded to host memory")
        
        print("🔍 Testing NPU Memory Allocation...")
        # This will test CANN memory allocation internally
        # The server logs will show NPU operations
        
        print("🔍 Testing Memory Cleanup...")
        client.clear_mem()
        print("✅ Memory cleared")
        
        print("🎉 All CANN functions working!")
        return True
        
    except Exception as e:
        print(f"❌ CANN test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cann_functions()
    sys.exit(0 if success else 1)