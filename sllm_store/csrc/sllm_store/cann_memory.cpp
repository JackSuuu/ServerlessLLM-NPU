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
#include "cann_memory.h"

#include <glog/logging.h>
#include <sstream>
#include <iomanip>

CannMemory::CannMemory() : data_(nullptr), size_(0), device_(-1) {}

CannMemory::~CannMemory() {
  if (data_) {  // Ensure we have data to free
#ifdef USE_CANN
    aclError ret = aclrtFree(data_);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "Failed to free CANN memory: " << ret;
    }
#endif
  }
}

int CannMemory::Allocate(size_t size, int device) {
  if (data_) {
    LOG(ERROR) << "Memory already allocated\n";
    return 1;  // Indicate error
  }

#ifdef USE_CANN
  // Check if device and size are valid
  uint32_t deviceCount = 0;
  aclError ret = aclrtGetDeviceCount(&deviceCount);
  if (ret != ACL_ERROR_NONE || device >= static_cast<int>(deviceCount) || size == 0) {
    LOG(ERROR) << "Invalid device or size\n";
    return 1;  // Indicate error
  }

  // Set device and allocate memory on it
  ret = aclrtSetDevice(device);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed to set device " << device << ": " << ret << "\n";
    return ret;
  }

  ret = aclrtMalloc(&data_, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed to allocate memory on device " << device << ": " << ret << "\n";
    return ret;
  }
  device_ = device;
  size_ = size;

  // Generate IPC handle (CANN uses pointer address as handle string)
  std::ostringstream oss;
  oss << std::hex << reinterpret_cast<uintptr_t>(data_);
  handle_ = oss.str();

  return 0;  // Indicate success
#else
  LOG(ERROR) << "CANN support not compiled\n";
  return 1;
#endif
}

void* CannMemory::get() const { return data_; }

std::string CannMemory::getHandle() const { return handle_; } 