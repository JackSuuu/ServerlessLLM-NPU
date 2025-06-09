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
#pragma once

#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.

#include <string>
#include <unordered_map>

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path);

std::unordered_map<std::string, torch::Tensor> RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets);

// Memory allocation and handle functions for both CUDA and CANN
#ifdef USE_CANN
#include "cann_ipc.h"

std::unordered_map<int, void*> AllocateCannMemory(
    const std::unordered_map<int, size_t>& tensor_sizes) {
  std::unordered_map<int, void*> memory_ptrs;
  for (const auto& p : tensor_sizes) {
    int device = p.first;
    size_t size = p.second;
    void* ptr = nullptr;
    aclrtSetDevice(device);
    aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
      LOG(ERROR) << "Failed to allocate CANN memory: " << ret;
      continue;
    }
    memory_ptrs[device] = ptr;
  }
  return memory_ptrs;
}

std::unordered_map<int, std::string> GetCannMemoryHandles(
    const std::unordered_map<int, void*>& memory_ptrs) {
  std::unordered_map<int, std::string> memory_handles;
  CannIpcManager& ipc_manager = CannIpcManager::getInstance();
  
  for (const auto& p : memory_ptrs) {
    int device = p.first;
    void* ptr = p.second;
    aclrtSetDevice(device);
    
    // Get memory size (you may need to track this separately)
    // For now, assuming you have a way to get the size
    size_t size = 0; // You'll need to track allocated sizes
    
    // Create IPC handle using the manager
    std::string handle = ipc_manager.createIpcHandle(ptr, size, device);
    if (!handle.empty()) {
      memory_handles[device] = handle;
    }
  }
  return memory_handles;
}

std::unordered_map<int, std::vector<std::string>> GetCannMemoryHandles(
    const std::unordered_map<int, std::vector<void*>>& memory_ptrs) {
  std::unordered_map<int, std::vector<std::string>> memory_handles;
  CannIpcManager& ipc_manager = CannIpcManager::getInstance();
  
  for (const auto& p : memory_ptrs) {
    auto device = p.first;
    const auto& ptrs = p.second;
    aclrtSetDevice(device);

    std::vector<std::string> handles;
    for (const auto& ptr : ptrs) {
      // You'll need to track sizes for each pointer
      size_t size = 0; // Track this separately
      std::string handle = ipc_manager.createIpcHandle(ptr, size, device);
      if (!handle.empty()) {
        handles.push_back(handle);
      }
    }
    memory_handles[device] = handles;
  }
  return memory_handles;
}
#else
// CUDA functions
std::unordered_map<int, void*> AllocateCudaMemory(
    const std::unordered_map<int, size_t>& tensor_sizes);
std::unordered_map<int, std::string> GetCudaMemoryHandles(
    const std::unordered_map<int, void*>& memory_ptrs);
std::unordered_map<int, std::vector<std::string>> GetCudaMemoryHandles(
    const std::unordered_map<int, std::vector<void*>>& memory_ptrs);
#endif

std::unordered_map<int, std::string> GetDeviceUuidMap();
std::unordered_map<std::string, int> GetGpuUUID();