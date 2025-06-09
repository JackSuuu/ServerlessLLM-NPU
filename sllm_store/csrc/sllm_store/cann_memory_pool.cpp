// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#include "cann_memory_pool.h"

#include <glog/logging.h>
#include <sstream>
#include <iomanip>

CannMemoryPool::CannMemoryPool(int device_count, size_t size_per_device)
    : device_count_(device_count), size_per_device_(size_per_device),
      pool_(device_count), handles_(device_count), offsets_(device_count, 0) {
  LOG(INFO) << "Creating CannMemoryPool with " << device_count
            << " devices and " << size_per_device << " bytes per device";
#ifdef USE_CANN
  for (int i = 0; i < device_count; ++i) {
    aclError ret = aclrtSetDevice(i);
    if (ret != ACL_ERROR_NONE) {
      LOG(FATAL) << "Failed to set device: " << ret;
    }
    void* ptr;
    ret = aclrtMalloc(&ptr, size_per_device, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
      LOG(FATAL) << "Failed to allocate memory on device " << i << ": " << ret;
    }
    
    // Generate IPC handle (CANN uses pointer address as handle string)
    std::ostringstream oss;
    oss << std::hex << reinterpret_cast<uintptr_t>(ptr);
    handles_[i] = oss.str();
    
    pool_[i] = ptr;
  }
#else
  LOG(FATAL) << "CANN support not compiled";
#endif
}

CannMemoryPool::~CannMemoryPool() {
#ifdef USE_CANN
  for (int i = 0; i < device_count_; ++i) {
    aclError ret = aclrtSetDevice(i);
    if (ret != ACL_ERROR_NONE) {
      LOG(FATAL) << "Failed to set device: " << ret;
    }
    ret = aclrtFree(pool_[i]);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "Failed to free memory on device " << i << ": " << ret;
    }
  }
#endif
}

int CannMemoryPool::Allocate(size_t size, int device_id, void*& ptr,
                             std::string& handle) {
  if (device_id >= device_count_ || device_id < 0) {
    LOG(ERROR) << "Invalid device_id: " << device_id;
    return -1;
  }

  if (offsets_[device_id] + size > size_per_device_) {
    LOG(ERROR) << "Not enough memory on device " << device_id;
    return -1;
  }

  ptr = static_cast<char*>(pool_[device_id]) + offsets_[device_id];
  handle = handles_[device_id];
  offsets_[device_id] += size;
  
  return 0;
}

int CannMemoryPool::Deallocate(int device_id, void* ptr) {
  if (device_id >= device_count_ || device_id < 0) {
    LOG(ERROR) << "Invalid device_id: " << device_id;
    return -1;
  }
  
  // Note: Simple pool implementation doesn't actually deallocate individual chunks
  // This would need more sophisticated implementation for production use
  LOG(WARNING) << "CannMemoryPool deallocate is a no-op in simple implementation";
  return 0;
} 